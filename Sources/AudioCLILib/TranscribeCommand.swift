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

    @Option(name: .long, help: "ASR engine: qwen3 (default), parakeet, qwen3-coreml, or qwen3-coreml-full")
    public var engine: String = "qwen3"

    @Option(name: .shortAndLong, help: "[qwen3] Model: 0.6B (default), 0.6B-8bit, 1.7B, 1.7B-4bit, or full HuggingFace model ID")
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
        guard eng == "qwen3" || eng == "parakeet" || eng == "qwen3-coreml" || eng == "qwen3-coreml-full" else {
            throw ValidationError("--engine must be 'qwen3', 'parakeet', 'qwen3-coreml', or 'qwen3-coreml-full'")
        }
    }

    public func run() throws {
        switch engine.lowercased() {
        case "parakeet":
            try runParakeetTranscription()
        case "qwen3-coreml":
            try runCoreMLTranscription()
        case "qwen3-coreml-full":
            try runFullCoreMLTranscription()
        default:
            if stream {
                try runStreamingTranscription()
            } else {
                try runBatchTranscription()
            }
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

    private func runCoreMLTranscription() throws {
        #if canImport(CoreML)
        try runAsync {
            let modelId = resolveASRModelId(model)

            print("Loading audio: \(audioFile)")
            let audio = try AudioFileLoader.load(
                url: URL(fileURLWithPath: audioFile), targetSampleRate: 16000)
            let duration = Float(audio.count) / 16000.0
            print("  Loaded \(audio.count) samples (\(String(format: "%.2f", duration))s)")

            // Load CoreML encoder
            print("Loading CoreML encoder...")
            let coremlEncoder = try await CoreMLASREncoder.fromPretrained(
                progressHandler: reportProgress)

            print("Warming up CoreML...")
            try coremlEncoder.warmUp()

            // Load MLX text decoder
            print("Loading text decoder: \(modelId)")
            let asrModel = try await Qwen3ASRModel.fromPretrained(
                modelId: modelId, progressHandler: reportProgress)

            print("Transcribing (CoreML encoder + MLX decoder)...")
            let startTime = CFAbsoluteTimeGetCurrent()
            let result = try asrModel.transcribe(
                audio: audio, sampleRate: 16000, language: language,
                coremlEncoder: coremlEncoder)
            let elapsed = CFAbsoluteTimeGetCurrent() - startTime
            let rtf = elapsed / Double(duration)

            print("Result: \(result)")
            print(String(format: "  Time: %.2fs, RTF: %.3f", elapsed, rtf))
        }
        #else
        print("CoreML is not available on this platform.")
        #endif
    }

    private func runFullCoreMLTranscription() throws {
        #if canImport(CoreML)
        try runAsync {
            print("Loading audio: \(audioFile)")
            let audio = try AudioFileLoader.load(
                url: URL(fileURLWithPath: audioFile), targetSampleRate: 16000)
            let duration = Float(audio.count) / 16000.0
            print("  Loaded \(audio.count) samples (\(String(format: "%.2f", duration))s)")

            if #available(macOS 15, iOS 18, *) {
                print("Loading full CoreML ASR pipeline...")
                let asrModel = try await CoreMLASRModel.fromPretrained(
                    progressHandler: reportProgress)

                print("Warming up CoreML...")
                let warmupStart = CFAbsoluteTimeGetCurrent()
                try asrModel.warmUp()
                let warmupTime = CFAbsoluteTimeGetCurrent() - warmupStart

                print("Transcribing (full CoreML: encoder + decoder)...")
                let startTime = CFAbsoluteTimeGetCurrent()
                let result = try asrModel.transcribe(
                    audio: audio, sampleRate: 16000, language: language)
                let elapsed = CFAbsoluteTimeGetCurrent() - startTime
                let rtf = elapsed / Double(duration)

                print("Result: \(result)")
                print(String(format: "  Time: %.2fs, RTF: %.3f (warmup: %.2fs)", elapsed, rtf, warmupTime))
            } else {
                print("Full CoreML ASR requires macOS 15+ / iOS 18+ for MLState KV cache.")
                print("Use --engine qwen3-coreml for hybrid CoreML encoder + MLX decoder.")
            }
        }
        #else
        print("CoreML is not available on this platform.")
        #endif
    }

    private func runParakeetTranscription() throws {
        try runAsync {
            print("Loading audio: \(audioFile)")
            let audio = try AudioFileLoader.load(
                url: URL(fileURLWithPath: audioFile), targetSampleRate: 16000)
            let duration = Float(audio.count) / 16000.0
            print("  Loaded \(audio.count) samples (\(String(format: "%.2f", duration))s)")

            let parakeetModelId = model.lowercased().contains("8")
                ? ParakeetASRModel.int8ModelId
                : ParakeetASRModel.defaultModelId
            print("Loading Parakeet-TDT model: \(parakeetModelId)")
            let parakeetModel = try await ParakeetASRModel.fromPretrained(
                modelId: parakeetModelId, progressHandler: reportProgress)

            print("Warming up CoreML...")
            let warmupStart = CFAbsoluteTimeGetCurrent()
            try parakeetModel.warmUp()
            let warmupTime = CFAbsoluteTimeGetCurrent() - warmupStart

            print("Transcribing...")
            let startTime = CFAbsoluteTimeGetCurrent()
            let result = try parakeetModel.transcribeAudio(audio, sampleRate: 16000, language: language)
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
    case "0.6b-8bit", "small-8bit":
        return "aufklarer/Qwen3-ASR-0.6B-MLX-8bit"
    case "1.7b", "large":
        return ASRModelSize.large.defaultModelId
    case "1.7b-4bit", "large-4bit":
        return "aufklarer/Qwen3-ASR-1.7B-MLX-4bit"
    default:
        return specifier
    }
}
