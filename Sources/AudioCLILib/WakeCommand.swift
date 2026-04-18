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
        help: "One or more keywords. Either a bare phrase ('hey soniqo') or phrase:ac_threshold:boost ('hey soniqo:0.15:0.5')."
    )
    public var keywords: [String] = []

    @Option(name: .long, help: "Path to a keywords file (one `phrase:ac_threshold:boost` per line).")
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

    /// "phrase" | "phrase:threshold" | "phrase:threshold:boost"
    private func parseSpec(_ raw: String) -> KeywordSpec {
        let parts = raw.split(separator: ":", omittingEmptySubsequences: false).map(String.init)
        let phrase = parts[0].trimmingCharacters(in: .whitespaces)
        let threshold = parts.count > 1 ? Double(parts[1]) ?? 0 : 0
        let boost = parts.count > 2 ? Double(parts[2]) ?? 0 : 0
        return KeywordSpec(phrase: phrase, acThreshold: threshold, boost: boost)
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
