import Foundation
import ArgumentParser
import Qwen3ASR
import AudioCommon

public struct AlignCommand: ParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "align",
        abstract: "Forced alignment: align text to audio with word-level timestamps"
    )

    @Argument(help: "Audio file (WAV, any sample rate)")
    public var audioFile: String

    @Option(name: .shortAndLong, help: "Text to align (if omitted, transcribes first)")
    public var text: String?

    @Option(name: .shortAndLong, help: "ASR model for transcription: 0.6B (default), 1.7B, or full ID")
    public var model: String = "0.6B"

    @Option(name: .long, help: "Forced aligner model ID")
    public var alignerModel: String = "aufklarer/Qwen3-ForcedAligner-0.6B-4bit"

    @Option(name: .long, help: "Language hint (optional)")
    public var language: String?

    public init() {}

    public func run() throws {
        try runAsync {
            print("Loading audio: \(audioFile)")
            let audio = try AudioFileLoader.load(
                url: URL(fileURLWithPath: audioFile), targetSampleRate: 24000)
            print("  Loaded \(audio.count) samples (\(formatDuration(audio.count))s)")

            var textToAlign = text

            // If no text provided, transcribe first
            if textToAlign == nil {
                let modelId = resolveASRModelId(model)
                let detectedSize = ASRModelSize.detect(from: modelId)
                let sizeLabel = detectedSize == .large ? "1.7B" : "0.6B"
                print("Loading ASR model (\(sizeLabel)): \(modelId)")

                let asrModel = try await Qwen3ASRModel.fromPretrained(
                    modelId: modelId, progressHandler: reportProgress)

                print("Transcribing...")
                textToAlign = asrModel.transcribe(audio: audio, sampleRate: 24000, language: language)
                print("Transcription: \(textToAlign!)")
            }

            guard let alignText = textToAlign, !alignText.isEmpty else {
                print("Error: no text to align")
                throw ExitCode(1)
            }

            print("Loading aligner model: \(alignerModel)")
            let aligner = try await Qwen3ForcedAligner.fromPretrained(
                modelId: alignerModel, progressHandler: reportProgress)

            print("Aligning...")
            let start = Date()
            let aligned = aligner.align(audio: audio, text: alignText, sampleRate: 24000)
            let elapsed = Date().timeIntervalSince(start)

            for word in aligned {
                let startStr = String(format: "%.2f", word.startTime)
                let endStr = String(format: "%.2f", word.endTime)
                print("[\(startStr)s - \(endStr)s] \(word.text)")
            }
            print("Alignment took \(String(format: "%.2f", elapsed))s")
        }
    }
}
