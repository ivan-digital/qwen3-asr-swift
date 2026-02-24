import Foundation
import ArgumentParser
import Qwen3TTS
import CosyVoiceTTS
import AudioCommon

public struct SpeakCommand: ParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "speak",
        abstract: "Text-to-speech synthesis (Qwen3-TTS or CosyVoice)"
    )

    @Argument(help: "Text to synthesize (omit when using --list-speakers or --batch-file)")
    public var text: String?

    @Option(name: .long, help: "TTS engine: qwen3 (default) or cosyvoice")
    public var engine: String = "qwen3"

    @Option(name: .shortAndLong, help: "Output WAV file path")
    public var output: String = "output.wav"

    @Option(name: .long, help: "Language (english, chinese, german, japanese, spanish, french, korean, russian, italian, portuguese)")
    public var language: String = "english"

    @Flag(name: .long, help: "Enable streaming synthesis")
    public var stream: Bool = false

    // MARK: - Qwen3-specific options

    @Option(name: .long, help: "[qwen3] Speaker voice (requires CustomVoice model)")
    public var speaker: String?

    @Option(name: .long, help: "[qwen3] Style instruction (requires CustomVoice model)")
    public var instruct: String?

    @Option(name: .long, help: "[qwen3] Model variant: base (default), customVoice, or full HF model ID")
    public var model: String = "base"

    @Flag(name: .long, help: "[qwen3] List available speakers and exit")
    public var listSpeakers: Bool = false

    @Option(name: .long, help: "[qwen3] Sampling temperature")
    public var temperature: Float = 0.9

    @Option(name: .long, help: "[qwen3] Top-k sampling")
    public var topK: Int = 50

    @Option(name: .long, help: "[qwen3] Maximum tokens (500 = ~40s audio)")
    public var maxTokens: Int = 500

    @Option(name: .long, help: "[qwen3] File with one text per line for batch synthesis")
    public var batchFile: String?

    @Option(name: .long, help: "[qwen3] Maximum batch size for parallel generation")
    public var batchSize: Int = 4

    @Option(name: .long, help: "[qwen3] Codec frames in first streamed chunk (default 3)")
    public var firstChunkFrames: Int = 3

    @Option(name: .long, help: "Codec frames per streamed chunk (default 25)")
    public var chunkFrames: Int = 25

    // MARK: - CosyVoice-specific options

    @Option(name: .long, help: "[cosyvoice] HuggingFace model ID")
    public var modelId: String = "aufklarer/CosyVoice3-0.5B-MLX-4bit"

    @Flag(name: .long, help: "Show detailed timing info")
    public var verbose: Bool = false

    public init() {}

    public func validate() throws {
        let eng = engine.lowercased()
        guard eng == "qwen3" || eng == "cosyvoice" else {
            throw ValidationError("--engine must be 'qwen3' or 'cosyvoice'")
        }
        if text == nil && batchFile == nil && !listSpeakers {
            throw ValidationError("Either a text argument, --batch-file, or --list-speakers must be provided")
        }
    }

    public func run() throws {
        if engine.lowercased() == "cosyvoice" {
            try runCosyVoice()
        } else {
            try runQwen3()
        }
    }

    // MARK: - Qwen3 engine

    private func runQwen3() throws {
        try runAsync {
            // Resolve model ID
            let resolvedModelId: String
            switch model.lowercased() {
            case "base":
                resolvedModelId = TTSModelVariant.base.rawValue
            case "customvoice", "custom_voice", "custom-voice":
                resolvedModelId = TTSModelVariant.customVoice.rawValue
            default:
                resolvedModelId = model
            }

            print("Loading Qwen3-TTS model (\(resolvedModelId))...")
            let ttsModel = try await Qwen3TTSModel.fromPretrained(
                modelId: resolvedModelId, progressHandler: reportProgress)

            // --list-speakers
            if listSpeakers {
                let speakers = ttsModel.availableSpeakers
                if speakers.isEmpty {
                    print("No speakers available for this model.")
                    print("Use --model customVoice to load a model with speaker support.")
                } else {
                    print("Available speakers:")
                    for name in speakers {
                        let dialect = ttsModel.speakerConfig?.speakerDialects[name]
                        let suffix = dialect != nil ? " (\(dialect!))" : ""
                        print("  - \(name)\(suffix)")
                    }
                }
                return
            }

            let config = SamplingConfig(
                temperature: temperature,
                topK: topK,
                maxTokens: maxTokens)

            // Resolve effective instruct
            let effectiveInstruct: String?
            let instructIsDefault: Bool
            if let explicit = instruct {
                effectiveInstruct = explicit
                instructIsDefault = false
            } else if ttsModel.speakerConfig != nil {
                effectiveInstruct = Qwen3TTSModel.defaultInstruct
                instructIsDefault = true
            } else {
                effectiveInstruct = nil
                instructIsDefault = false
            }

            if stream, let inputText = text {
                try await runQwen3Streaming(
                    model: ttsModel, text: inputText,
                    instruct: effectiveInstruct, instructIsDefault: instructIsDefault,
                    config: config)
            } else if let batchFile = batchFile {
                try runQwen3Batch(model: ttsModel, batchFile: batchFile, config: config)
            } else if let inputText = text {
                try runQwen3Standard(
                    model: ttsModel, text: inputText,
                    instruct: effectiveInstruct, instructIsDefault: instructIsDefault,
                    config: config)
            }
        }
    }

    private func runQwen3Streaming(
        model: Qwen3TTSModel, text: String,
        instruct: String?, instructIsDefault: Bool,
        config: SamplingConfig
    ) async throws {
        let streamingConfig = StreamingConfig(
            firstChunkFrames: firstChunkFrames,
            chunkFrames: chunkFrames)

        var info = "Streaming synthesis: \"\(text)\""
        if let spk = speaker { info += " [speaker: \(spk)]" }
        if let inst = instruct { info += " [instruct: \(inst)\(instructIsDefault ? " (default)" : "")]" }
        print(info)
        print("  First chunk: \(firstChunkFrames) frames, subsequent: \(chunkFrames) frames")

        var allSamples: [Float] = []
        var chunkCount = 0
        var firstPacketLatency: Double?

        let audioStream = model.synthesizeStream(
            text: text,
            language: language,
            speaker: speaker,
            instruct: instruct,
            sampling: config,
            streaming: streamingConfig)

        for try await chunk in audioStream {
            chunkCount += 1
            allSamples.append(contentsOf: chunk.samples)

            if firstPacketLatency == nil {
                firstPacketLatency = chunk.elapsedTime
            }

            let chunkDuration = Double(chunk.samples.count) / 24000.0
            let marker = chunk.isFinal ? " [FINAL]" : ""
            print("  Chunk \(chunkCount): \(chunk.samples.count) samples " +
                  "(\(String(format: "%.3f", chunkDuration))s) | " +
                  "frame \(chunk.frameIndex) | " +
                  "elapsed \(String(format: "%.3f", chunk.elapsedTime ?? 0))s\(marker)")
        }

        guard !allSamples.isEmpty else {
            print("Error: No audio generated")
            throw ExitCode(1)
        }

        print("  First-packet latency: \(String(format: "%.0f", (firstPacketLatency ?? 0) * 1000))ms")
        print("  Total: \(chunkCount) chunks, \(allSamples.count) samples (\(formatDuration(allSamples.count))s)")

        let outputURL = URL(fileURLWithPath: output)
        try WAVWriter.write(samples: allSamples, sampleRate: 24000, to: outputURL)
        print("Saved to \(output)")
    }

    private func runQwen3Batch(
        model: Qwen3TTSModel, batchFile: String, config: SamplingConfig
    ) throws {
        let content = try String(contentsOfFile: batchFile, encoding: .utf8)
        let texts = content.components(separatedBy: .newlines)
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }

        guard !texts.isEmpty else {
            print("Error: No texts found in \(batchFile)")
            throw ExitCode(1)
        }

        print("Batch synthesizing \(texts.count) texts...")
        let audioList = model.synthesizeBatch(
            texts: texts,
            language: language,
            instruct: instruct,
            sampling: config,
            maxBatchSize: batchSize)

        let basePath = (output as NSString).deletingPathExtension
        let ext = (output as NSString).pathExtension.isEmpty ? "wav" : (output as NSString).pathExtension

        for (i, audio) in audioList.enumerated() {
            guard !audio.isEmpty else {
                print("Warning: Item \(i) produced no audio")
                continue
            }
            let path = "\(basePath)_\(i).\(ext)"
            let url = URL(fileURLWithPath: path)
            try WAVWriter.write(samples: audio, sampleRate: 24000, to: url)
            print("Saved item \(i): \(audio.count) samples (\(formatDuration(audio.count))s) to \(path)")
        }
    }

    private func runQwen3Standard(
        model: Qwen3TTSModel, text: String,
        instruct: String?, instructIsDefault: Bool,
        config: SamplingConfig
    ) throws {
        var info = "Synthesizing: \"\(text)\""
        if let spk = speaker { info += " [speaker: \(spk)]" }
        if let inst = instruct { info += " [instruct: \(inst)\(instructIsDefault ? " (default)" : "")]" }
        print(info)

        let audio = model.synthesize(
            text: text,
            language: language,
            speaker: speaker,
            instruct: instruct,
            sampling: config)

        guard !audio.isEmpty else {
            print("Error: No audio generated")
            throw ExitCode(1)
        }

        let outputURL = URL(fileURLWithPath: output)
        try WAVWriter.write(samples: audio, sampleRate: 24000, to: outputURL)
        print("Saved \(audio.count) samples (\(formatDuration(audio.count))s) to \(output)")
    }

    // MARK: - CosyVoice engine

    private func runCosyVoice() throws {
        try runAsync {
            print("Loading CosyVoice3 model...")
            let cosyModel = try await CosyVoiceTTSModel.fromPretrained(
                modelId: modelId, progressHandler: reportProgress)

            guard let inputText = text else {
                print("Error: text argument is required for CosyVoice")
                throw ExitCode(1)
            }

            print("Synthesizing: \"\(inputText)\"")
            print("  Language: \(language)")

            let startTime = CFAbsoluteTimeGetCurrent()

            if stream {
                var allSamples: [Float] = []
                var chunkCount = 0
                for try await chunk in cosyModel.synthesizeStream(text: inputText, language: language) {
                    allSamples.append(contentsOf: chunk.samples)
                    chunkCount += 1
                    let chunkDuration = Double(chunk.samples.count) / Double(chunk.sampleRate)
                    print("  Chunk \(chunkCount): \(String(format: "%.2f", chunkDuration))s (\(chunk.samples.count) samples)")
                }

                let elapsed = CFAbsoluteTimeGetCurrent() - startTime
                let duration = Double(allSamples.count) / 24000.0
                print(String(format: "  Duration: %.2fs, Time: %.2fs, RTF: %.2f",
                             duration, elapsed, elapsed / max(duration, 0.001)))

                let outputURL = URL(fileURLWithPath: output)
                try WAVWriter.write(samples: allSamples, sampleRate: 24000, to: outputURL)
                print("Saved to \(output)")
            } else {
                let samples = cosyModel.synthesize(
                    text: inputText, language: language, verbose: verbose)

                let elapsed = CFAbsoluteTimeGetCurrent() - startTime
                let duration = Double(samples.count) / 24000.0
                print(String(format: "  Duration: %.2fs, Time: %.2fs, RTF: %.2f",
                             duration, elapsed, elapsed / max(duration, 0.001)))

                let outputURL = URL(fileURLWithPath: output)
                try WAVWriter.write(samples: samples, sampleRate: 24000, to: outputURL)
                print("Saved to \(output)")
            }
        }
    }
}
