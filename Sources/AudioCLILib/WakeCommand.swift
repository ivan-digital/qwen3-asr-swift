import Foundation
import ArgumentParser
import AudioCommon
import SpeechWakeWord

public struct WakeCommand: ParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "wake",
        abstract: "Detect wake words / keyword phrases (KWS Zipformer)"
    )

    @Argument(help: "Audio file to analyze (WAV, any sample rate)")
    public var audioFile: String

    @Option(
        name: .long,
        parsing: .upToNextOption,
        help: """
            One or more keywords. Formats:
              'hey soniqo' — greedy BPE over the phrase
              'hey soniqo:0.15:0.5' — phrase with threshold and boost
              'LIGHT UP|▁ L IGHT ▁UP:0.25:2.0' — phrase | explicit BPE pieces (space-separated, sherpa-onnx style) : threshold : boost
            """
    )
    public var keywords: [String] = []

    @Option(name: .long, help: "Path to a keywords file (one `phrase[|pieces][:threshold:boost]` per line).")
    public var keywordsFile: String?

    @Option(name: .shortAndLong, help: "Model ID on HuggingFace")
    public var model: String?

    @Flag(name: .long, help: "Output as JSON")
    public var json: Bool = false

    public init() {}

    public func run() throws {
        try runAsync {
            let specs = try resolveKeywords()
            guard !specs.isEmpty else {
                throw ValidationError("Provide at least one --keywords or --keywords-file entry.")
            }

            print("Loading audio: \(audioFile)")
            let audio = try AudioFileLoader.load(
                url: URL(fileURLWithPath: audioFile), targetSampleRate: 16000)
            print("  Loaded \(audio.count) samples (\(formatDuration(audio.count, sampleRate: 16000))s)")

            let modelId = model ?? WakeWordDetector.defaultModelId
            print("Loading KWS Zipformer: \(modelId)")
            let detector = try await WakeWordDetector.fromPretrained(
                modelId: modelId,
                keywords: specs,
                progressHandler: reportProgress
            )

            print("Detecting keywords for: \(specs.map { $0.phrase }.joined(separator: ", "))")
            let start = Date()
            let detections = try detector.detect(audio: audio, sampleRate: 16000)
            let elapsed = Date().timeIntervalSince(start)

            if json {
                printJSON(detections)
            } else if detections.isEmpty {
                print("No keywords detected.")
            } else {
                for d in detections {
                    let t = String(format: "%.2f", d.time(frameShiftSeconds: 0.04))
                    print("[\(t)s] \(d.phrase)")
                }
                print("\n\(detections.count) detection(s)")
            }
            print("Detection took \(String(format: "%.2f", elapsed))s")
        }
    }

    // MARK: - parsing helpers

    private func resolveKeywords() throws -> [KeywordSpec] {
        var specs: [KeywordSpec] = []
        for entry in keywords { specs.append(parseSpec(entry)) }
        if let file = keywordsFile {
            let text = try String(contentsOfFile: file, encoding: .utf8)
            for line in text.split(whereSeparator: { $0 == "\n" || $0 == "\r" }) {
                let trimmed = line.trimmingCharacters(in: .whitespaces)
                if trimmed.isEmpty || trimmed.hasPrefix("#") { continue }
                specs.append(parseSpec(trimmed))
            }
        }
        return specs
    }

    /// Format: "phrase[|piece1 piece2 ...][:threshold:boost]". Examples:
    ///   "hey soniqo"
    ///   "hey soniqo:0.15:0.5"
    ///   "LIGHT UP|▁ L IGHT ▁UP"
    ///   "LIGHT UP|▁ L IGHT ▁UP:0.25:2.0"
    private func parseSpec(_ raw: String) -> KeywordSpec {
        // Split the ``phrase[|pieces]`` prefix from the ``:threshold:boost`` tail.
        let tailStart = raw.range(of: ":")
        let (head, tail) = tailStart.map {
            (String(raw[..<$0.lowerBound]), String(raw[$0.upperBound...]))
        } ?? (raw, "")

        let phrase: String
        let tokens: [String]?
        if let pipe = head.range(of: "|") {
            phrase = String(head[..<pipe.lowerBound]).trimmingCharacters(in: .whitespaces)
            let pieces = String(head[pipe.upperBound...])
                .split(whereSeparator: { $0.isWhitespace })
                .map(String.init)
            tokens = pieces.isEmpty ? nil : pieces
        } else {
            phrase = head.trimmingCharacters(in: .whitespaces)
            tokens = nil
        }

        var threshold = 0.0
        var boost = 0.0
        if !tail.isEmpty {
            let parts = tail.split(separator: ":", omittingEmptySubsequences: false).map(String.init)
            if parts.count >= 1 { threshold = Double(parts[0]) ?? 0 }
            if parts.count >= 2 { boost = Double(parts[1]) ?? 0 }
        }
        return KeywordSpec(phrase: phrase, acThreshold: threshold, boost: boost, tokens: tokens)
    }

    private func printJSON(_ detections: [KeywordDetection]) {
        var items = [[String: Any]]()
        for d in detections {
            items.append([
                "phrase": d.phrase,
                "frame": d.frameIndex,
                "time": Double(String(format: "%.3f", d.time(frameShiftSeconds: 0.04)))!,
                "tokens": d.tokenIds
            ])
        }
        if let data = try? JSONSerialization.data(withJSONObject: items, options: .prettyPrinted),
           let str = String(data: data, encoding: .utf8) {
            print(str)
        }
    }
}
