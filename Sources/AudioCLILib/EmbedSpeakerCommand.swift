import Foundation
import ArgumentParser
import SpeechVAD
import AudioCommon

public struct EmbedSpeakerCommand: ParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "embed-speaker",
        abstract: "Extract a 256-dim speaker embedding from audio"
    )

    @Argument(help: "Audio file containing speaker voice (WAV, any sample rate)")
    public var audioFile: String

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

            print("Loading speaker embedding model...")
            let model = try await WeSpeakerModel.fromPretrained(
                progressHandler: reportProgress
            )

            print("Extracting speaker embedding...")
            let start = Date()
            let embedding = model.embed(audio: audio, sampleRate: 16000)
            let elapsed = Date().timeIntervalSince(start)

            if json {
                let embStrings = embedding.map { String(format: "%.6f", $0) }
                print("{\"embedding\": [\(embStrings.joined(separator: ", "))], \"dimension\": \(embedding.count)}")
            } else {
                // Print summary statistics
                let minVal = embedding.min() ?? 0
                let maxVal = embedding.max() ?? 0
                let norm = sqrt(embedding.reduce(Float(0)) { $0 + $1 * $1 })

                print("Embedding dimension: \(embedding.count)")
                print("L2 norm: \(String(format: "%.4f", norm))")
                print("Range: [\(String(format: "%.4f", minVal)), \(String(format: "%.4f", maxVal))]")
                print("First 8 values: \(embedding.prefix(8).map { String(format: "%.4f", $0) }.joined(separator: ", "))")
                print("Extraction took \(String(format: "%.2f", elapsed))s")
            }
        }
    }
}
