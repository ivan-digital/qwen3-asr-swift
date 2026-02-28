import Foundation
import ArgumentParser
import Qwen3ASR
import ParakeetASR
import SpeechVAD
import AudioCommon

public struct TranscribeCommand: ParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "transcribe",
        abstract: "Transcribe speech to text (Qwen3-ASR or Parakeet-TDT)"
    )

    @Argument(help: "Audio file to transcribe (WAV, any sample rate)")
    public var audioFile: String

    @Option(name: .long, help: "ASR engine: qwen3 (default) or parakeet")
    public var engine: String = "qwen3"

    @Option(name: .shortAndLong, help: "[qwen3] Model: 0.6B (default), 1.7B, or full HuggingFace model ID")
    public var model: String = "0.6B"

    @Option(name: .long, help: "Language hint (optional)")
    public var language: String?

    @Flag(name: .long, help: "Enable streaming transcription with VAD")
    public var stream: Bool = false

    @Option(name: .long, help: "Maximum segment duration in seconds (default 10)")
    public var maxSegment: Float = 10.0

    @Flag(name: .long, help: "Emit partial results during speech")
    public var partial: Bool = false

    public init() {}

    public func validate() throws {
        let eng = engine.lowercased()
        guard eng == "qwen3" || eng == "parakeet" else {
            throw ValidationError("--engine must be 'qwen3' or 'parakeet'")
        }
    }

    public func run() throws {
        if engine.lowercased() == "parakeet" {
            try runParakeetTranscription()
        } else if stream {
            try runStreamingTranscription()
        } else {
            try runBatchTranscription()
        }
    }

    private func runBatchTranscription() throws {
        try runAsync {
            let modelId = resolveASRModelId(model)
            let detectedSize = ASRModelSize.detect(from: modelId)
            let sizeLabel = detectedSize == .large ? "1.7B" : "0.6B"

            print("Loading audio: \(audioFile)")
            let audio = try AudioFileLoader.load(
                url: URL(fileURLWithPath: audioFile), targetSampleRate: 24000)
            print("  Loaded \(audio.count) samples (\(formatDuration(audio.count))s)")

            print("Loading model (\(sizeLabel)): \(modelId)")
            let asrModel = try await Qwen3ASRModel.fromPretrained(
                modelId: modelId, progressHandler: reportProgress)

            print("Transcribing...")
            let startTime = CFAbsoluteTimeGetCurrent()
            let result = asrModel.transcribe(audio: audio, sampleRate: 24000, language: language)
            let elapsed = CFAbsoluteTimeGetCurrent() - startTime
            let duration = Float(audio.count) / 24000.0
            let rtf = elapsed / Double(duration)

            print("Result: \(result)")
            print(String(format: "  Time: %.2fs, RTF: %.3f", elapsed, rtf))
        }
    }

    private func runStreamingTranscription() throws {
        try runAsync {
            let modelId = resolveASRModelId(model)

            print("Loading audio: \(audioFile)")
            let audio = try AudioFileLoader.load(
                url: URL(fileURLWithPath: audioFile), targetSampleRate: 16000)
            let duration = Float(audio.count) / 16000.0
            print("  Loaded \(audio.count) samples (\(String(format: "%.2f", duration))s)")

            print("Loading models...")
            let streaming = try await StreamingASR.fromPretrained(
                asrModelId: modelId, progressHandler: reportProgress)

            let config = StreamingASRConfig(
                maxSegmentDuration: maxSegment,
                language: language,
                emitPartialResults: partial
            )

            print("Streaming transcription (VAD + ASR)...")
            let stream = streaming.transcribeStream(
                audio: audio, sampleRate: 16000, config: config)

            for try await segment in stream {
                let tag = segment.isFinal ? "FINAL" : "partial"
                let start = String(format: "%.2f", segment.startTime)
                let end = String(format: "%.2f", segment.endTime)
                print("[\(start)s-\(end)s] [\(tag)] \(segment.text)")
            }
        }
    }

    private func runParakeetTranscription() throws {
        try runAsync {
            print("Loading audio: \(audioFile)")
            let audio = try AudioFileLoader.load(
                url: URL(fileURLWithPath: audioFile), targetSampleRate: 16000)
            let duration = Float(audio.count) / 16000.0
            print("  Loaded \(audio.count) samples (\(String(format: "%.2f", duration))s)")

            print("Loading Parakeet-TDT model...")
            let model = try await ParakeetASRModel.fromPretrained(
                progressHandler: reportProgress)

            print("Warming up CoreML...")
            let warmupStart = CFAbsoluteTimeGetCurrent()
            try model.warmUp()
            let warmupTime = CFAbsoluteTimeGetCurrent() - warmupStart

            print("Transcribing...")
            let startTime = CFAbsoluteTimeGetCurrent()
            let result = try model.transcribeAudio(audio, sampleRate: 16000, language: language)
            let elapsed = CFAbsoluteTimeGetCurrent() - startTime
            let rtf = elapsed / Double(duration)

            print("Result: \(result)")
            print(String(format: "  Time: %.2fs, RTF: %.3f (warmup: %.2fs)", elapsed, rtf, warmupTime))
        }
    }
}

/// Resolve shorthand model specifiers to HuggingFace model IDs.
public func resolveASRModelId(_ specifier: String) -> String {
    switch specifier.lowercased() {
    case "0.6b", "small":
        return ASRModelSize.small.defaultModelId
    case "1.7b", "large":
        return ASRModelSize.large.defaultModelId
    default:
        return specifier
    }
}
