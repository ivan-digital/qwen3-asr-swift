import Foundation
import ArgumentParser
import SpeechVAD
import AudioCommon

public struct VadCommand: ParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "vad",
        abstract: "Detect speech segments using pyannote Voice Activity Detection"
    )

    @Argument(help: "Audio file to analyze (WAV, any sample rate)")
    public var audioFile: String

    @Option(name: .shortAndLong, help: "Model ID on HuggingFace")
    public var model: String = PyannoteVADModel.defaultModelId

    @Option(name: .long, help: "Onset threshold (speech start)")
    public var onset: Float = VADConfig.default.onset

    @Option(name: .long, help: "Offset threshold (speech end)")
    public var offset: Float = VADConfig.default.offset

    @Option(name: .long, help: "Minimum speech duration in seconds")
    public var minSpeech: Float = VADConfig.default.minSpeechDuration

    @Option(name: .long, help: "Minimum silence duration in seconds")
    public var minSilence: Float = VADConfig.default.minSilenceDuration

    @Flag(name: .long, help: "Output as JSON")
    public var json: Bool = false

    public init() {}

    public func run() throws {
        try runAsync {
            print("Loading audio: \(audioFile)")
            let audio = try AudioFileLoader.load(
                url: URL(fileURLWithPath: audioFile), targetSampleRate: 16000)
            let duration = formatDuration(audio.count, sampleRate: 16000)
            print("  Loaded \(audio.count) samples (\(duration)s)")

            let vadConfig = VADConfig(
                onset: onset,
                offset: offset,
                minSpeechDuration: minSpeech,
                minSilenceDuration: minSilence,
                windowDuration: VADConfig.default.windowDuration,
                stepRatio: VADConfig.default.stepRatio
            )

            print("Loading VAD model: \(model)")
            let vad = try await PyannoteVADModel.fromPretrained(
                modelId: model,
                vadConfig: vadConfig,
                progressHandler: reportProgress
            )

            print("Detecting speech segments...")
            let start = Date()
            let segments = vad.detectSpeech(audio: audio, sampleRate: 16000)
            let elapsed = Date().timeIntervalSince(start)

            if json {
                printJSON(segments)
            } else {
                if segments.isEmpty {
                    print("No speech detected.")
                } else {
                    for seg in segments {
                        let s = String(format: "%.2f", seg.startTime)
                        let e = String(format: "%.2f", seg.endTime)
                        let d = String(format: "%.2f", seg.duration)
                        print("[\(s)s - \(e)s] (\(d)s)")
                    }

                    let totalSpeech = segments.reduce(Float(0)) { $0 + $1.duration }
                    print("\n\(segments.count) segment(s), \(String(format: "%.2f", totalSpeech))s total speech")
                }
                print("Detection took \(String(format: "%.2f", elapsed))s")
            }
        }
    }

    private func printJSON(_ segments: [SpeechSegment]) {
        var items = [[String: Any]]()
        for seg in segments {
            items.append([
                "start": Double(String(format: "%.3f", seg.startTime))!,
                "end": Double(String(format: "%.3f", seg.endTime))!,
                "duration": Double(String(format: "%.3f", seg.duration))!,
            ])
        }
        if let data = try? JSONSerialization.data(withJSONObject: items, options: .prettyPrinted),
           let str = String(data: data, encoding: .utf8) {
            print(str)
        }
    }
}
