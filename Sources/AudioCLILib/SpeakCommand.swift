import Foundation
import AVFoundation
import ArgumentParser
import Qwen3TTS
import CosyVoiceTTS
import Qwen3TTSCoreML
import AudioCommon

public struct SpeakCommand: ParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "speak",
        abstract: "Text-to-speech synthesis (Qwen3-TTS or CosyVoice)"
    )

    @Argument(help: "Text to synthesize (omit when using --list-speakers or --batch-file)")
    public var text: String?

    @Option(name: .long, help: "TTS engine: qwen3 (default), cosyvoice, or coreml")
    public var engine: String = "qwen3"

    @Option(name: .shortAndLong, help: "Output WAV file path")
    public var output: String = "output.wav"

    @Option(name: .long, help: "Language (english, chinese, german, japanese, spanish, french, korean, russian, italian, portuguese). Default: english. Omit to use speaker's native dialect when --speaker is set.")
    public var language: String?

    @Flag(name: .long, help: "Enable streaming synthesis")
    public var stream: Bool = false

    @Flag(name: .long, help: "Play audio through default output device instead of (or in addition to) saving a file")
    public var play: Bool = false

    // MARK: - Qwen3-specific options

    @Option(name: .long, help: "[qwen3] Speaker voice (requires --model customVoice)")
    public var speaker: String?

    @Option(name: .long, help: "[qwen3] Style instruction (requires CustomVoice model)")
    public var instruct: String?

    @Option(name: .long, help: "Reference audio file for voice cloning (qwen3 Base or cosyvoice)")
    public var voiceSample: String?

    @Option(name: .long, help: "[qwen3] Model variant: base (default), base-8bit, 1.7b, 1.7b-8bit, customVoice, or full HF model ID. Note: --speaker requires customVoice.")
    public var model: String = "base"

    @Flag(name: .long, help: "[qwen3] List available speakers and exit")
    public var listSpeakers: Bool = false

    @Option(name: .long, help: "[qwen3] Sampling temperature (default: 0.3)")
    public var temperature: Float = 0.3

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

    @Option(name: .long, help: "[cosyvoice] Speaker mapping: s1=alice.wav,s2=bob.wav")
    public var speakers: String?

    @Option(name: .long, help: "[cosyvoice] Style instruction (overrides default)")
    public var cosyInstruct: String?

    @Option(name: .long, help: "[cosyvoice] Silence gap between turns in seconds (default 0.2)")
    public var turnGap: Float = 0.2

    @Option(name: .long, help: "[cosyvoice] Crossfade between turns in seconds (default 0)")
    public var crossfade: Float = 0.0

    @Flag(name: .long, help: "Show detailed timing info")
    public var verbose: Bool = false

    public init() {}

    /// Resolved language: explicit value or default "english"
    private var effectiveLanguage: String { language ?? "english" }

    /// Whether the user explicitly passed --language
    private var languageIsExplicit: Bool { language != nil }

    public func validate() throws {
        let eng = engine.lowercased()
        guard eng == "qwen3" || eng == "cosyvoice" || eng == "coreml" else {
            throw ValidationError("--engine must be 'qwen3', 'cosyvoice', or 'coreml'")
        }
        if text == nil && batchFile == nil && !listSpeakers {
            throw ValidationError("Either a text argument, --batch-file, or --list-speakers must be provided")
        }
    }

    public func run() throws {
        if engine.lowercased() == "cosyvoice" {
            try runCosyVoice()
        } else if engine.lowercased() == "coreml" {
            try runCoreML()
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
            case "base-8bit", "base8bit":
                resolvedModelId = TTSModelVariant.base8bit.rawValue
            case "1.7b", "large":
                resolvedModelId = TTSModelVariant.base17B.rawValue
            case "1.7b-8bit", "large-8bit":
                resolvedModelId = TTSModelVariant.base17B8bit.rawValue
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
            language: effectiveLanguage,
            speaker: speaker,
            instruct: instruct,
            sampling: config,
            streaming: streamingConfig,
            languageExplicit: languageIsExplicit)

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

        if !play {
            let outputURL = URL(fileURLWithPath: output)
            try WAVWriter.write(samples: allSamples, sampleRate: 24000, to: outputURL)
            print("Saved to \(output)")
        } else {
            playAudio(samples: allSamples, sampleRate: 24000)
        }
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
            language: effectiveLanguage,
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
        if let vs = voiceSample { info += " [voice clone: \(vs)]" }
        print(info)

        let audio: [Float]
        if let voiceSamplePath = voiceSample {
            // Voice cloning mode
            let refURL = URL(fileURLWithPath: voiceSamplePath)
            let refSamples = try AudioFileLoader.load(url: refURL, targetSampleRate: 24000)
            print("  Reference audio: \(refSamples.count) samples, \(String(format: "%.1f", Double(refSamples.count) / 24000.0))s")

            audio = model.synthesizeWithVoiceClone(
                text: text,
                referenceAudio: refSamples,
                referenceSampleRate: 24000,
                language: effectiveLanguage,
                sampling: config)
        } else {
            audio = model.synthesize(
                text: text,
                language: effectiveLanguage,
                speaker: speaker,
                instruct: instruct,
                sampling: config,
                languageExplicit: languageIsExplicit)
        }

        guard !audio.isEmpty else {
            print("Error: No audio generated")
            throw ExitCode(1)
        }

        if !play {
            let outputURL = URL(fileURLWithPath: output)
            try WAVWriter.write(samples: audio, sampleRate: 24000, to: outputURL)
            print("Saved \(audio.count) samples (\(formatDuration(audio.count))s) to \(output)")
        } else {
            playAudio(samples: audio, sampleRate: 24000)
        }
    }

    // MARK: - CoreML engine

    private func runCoreML() throws {
        guard let text else {
            throw ValidationError("Text is required for CoreML TTS")
        }
        try runAsync {
            // If modelId looks like a local path, use it directly
            let localDir: String? = self.modelId.hasPrefix("/") ? self.modelId : nil
            let model = try await Qwen3TTSCoreMLModel.fromPretrained(
                localPath: localDir
            ) { progress, status in
                print("\r  [\(Int(progress * 100))%] \(status)", terminator: "")
                fflush(stdout)
            }
            print()

            let lang = self.language ?? "english"
            // CoreML backend needs higher temperature (0.8) — low temp is degenerate
            let coremlTemp: Float = self.temperature == 0.3 ? 0.8 : self.temperature
            print("Synthesizing with CoreML engine (language: \(lang))...")
            let start = CFAbsoluteTimeGetCurrent()
            let audio = try model.synthesize(
                text: text, language: lang,
                temperature: coremlTemp, topK: Int(self.topK),
                maxTokens: self.maxTokens)
            let elapsed = CFAbsoluteTimeGetCurrent() - start
            let duration = Double(audio.count) / 24000.0

            print(String(format: "  Generated %.2fs audio in %.1fs (RTF: %.2f)",
                         duration, elapsed, elapsed / max(duration, 0.01)))

            let outputURL = URL(fileURLWithPath: self.output)
            try WAVWriter.write(samples: audio, sampleRate: 24000, to: outputURL)
            print("  Saved: \(outputURL.path)")

            if self.play {
                let player = try AVAudioPlayer(contentsOf: outputURL)
                player.play()
                while player.isPlaying { Thread.sleep(forTimeInterval: 0.1) }
            }

            model.unload()
        }
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

            // Parse speaker mapping: "s1=alice.wav,s2=bob.wav"
            var speakerFiles: [String: String] = [:]
            if let speakersArg = speakers {
                for pair in speakersArg.split(separator: ",") {
                    let parts = pair.split(separator: "=", maxSplits: 1)
                    guard parts.count == 2 else {
                        print("Error: Invalid speaker mapping '\(pair)'. Expected format: name=file.wav")
                        throw ExitCode(1)
                    }
                    speakerFiles[String(parts[0]).uppercased()] = String(parts[1])
                }
            }

            // Load speaker embeddings from voice samples
            var speakerEmbeddings: [String: [Float]] = [:]
            #if canImport(CoreML)
            // Single --voice-sample (no --speakers) → used as default embedding
            var defaultEmbedding: [Float]?
            if let voiceSamplePath = voiceSample, speakerFiles.isEmpty {
                let refURL = URL(fileURLWithPath: voiceSamplePath)
                let refSamples = try AudioFileLoader.load(url: refURL, targetSampleRate: 16000)
                print("  Reference audio: \(refSamples.count) samples (\(String(format: "%.1f", Double(refSamples.count) / 16000.0))s)")

                print("Loading CAM++ speaker encoder...")
                let campp = try await CamPlusPlusSpeaker.fromPretrained { progress, status in
                    reportProgress(progress, status)
                }

                let embedding = try campp.embed(audio: refSamples, sampleRate: 16000)
                defaultEmbedding = embedding
                print("  Speaker embedding: \(embedding.count)-dim")
            }

            // Multi-speaker: load CAM++ once, extract embedding per speaker file
            if !speakerFiles.isEmpty {
                print("Loading CAM++ speaker encoder...")
                let campp = try await CamPlusPlusSpeaker.fromPretrained { progress, status in
                    reportProgress(progress, status)
                }

                for (name, path) in speakerFiles {
                    let refURL = URL(fileURLWithPath: path)
                    let refSamples = try AudioFileLoader.load(url: refURL, targetSampleRate: 16000)
                    let embedding = try campp.embed(audio: refSamples, sampleRate: 16000)
                    speakerEmbeddings[name] = embedding
                    print("  Speaker \(name): \(embedding.count)-dim embedding from \(path)")
                }
            }
            #else
            let defaultEmbedding: [Float]? = nil
            #endif

            // Parse dialogue segments
            let segments = DialogueParser.parse(inputText)
            let isDialogue = segments.count > 1
                || segments.first?.speaker != nil
                || segments.first?.emotion != nil

            let defaultInstruction = cosyInstruct ?? "You are a helpful assistant."

            print("  Language: \(effectiveLanguage)")

            let startTime = CFAbsoluteTimeGetCurrent()

            if isDialogue {
                // Multi-segment dialogue synthesis
                if verbose {
                    print("  Dialogue: \(segments.count) segments")
                    for (i, seg) in segments.enumerated() {
                        var desc = "    [\(i + 1)] \"\(seg.text)\""
                        if let spk = seg.speaker { desc += " speaker=\(spk)" }
                        if let emo = seg.emotion { desc += " emotion=\(emo)" }
                        print(desc)
                    }
                }

                // Merge default embedding into per-speaker map for segments without speaker tags
                var allEmbeddings = speakerEmbeddings
                if let defEmb = defaultEmbedding {
                    // Assign default embedding to any speaker tag not in the map
                    for seg in segments {
                        if let spk = seg.speaker?.uppercased(), allEmbeddings[spk] == nil {
                            allEmbeddings[spk] = defEmb
                        }
                    }
                }

                let config = DialogueSynthesisConfig(
                    turnGapSeconds: turnGap,
                    crossfadeSeconds: self.crossfade,
                    defaultInstruction: defaultInstruction
                )

                let samples = DialogueSynthesizer.synthesize(
                    segments: segments,
                    speakerEmbeddings: allEmbeddings,
                    model: cosyModel,
                    language: effectiveLanguage,
                    config: config,
                    verbose: verbose
                )

                let elapsed = CFAbsoluteTimeGetCurrent() - startTime
                let duration = Double(samples.count) / 24000.0
                print(String(format: "  Duration: %.2fs, Time: %.2fs, RTF: %.2f",
                             duration, elapsed, elapsed / max(duration, 0.001)))

                if !self.play {
                    let outputURL = URL(fileURLWithPath: self.output)
                    try WAVWriter.write(samples: samples, sampleRate: 24000, to: outputURL)
                    print("Saved to \(self.output)")
                } else {
                    self.playAudio(samples: samples, sampleRate: 24000)
                }
            } else if stream {
                // Streaming (single segment, no dialogue)
                var allSamples: [Float] = []
                var chunkCount = 0
                for try await chunk in cosyModel.synthesizeStream(text: inputText, language: effectiveLanguage) {
                    allSamples.append(contentsOf: chunk.samples)
                    chunkCount += 1
                    let chunkDuration = Double(chunk.samples.count) / Double(chunk.sampleRate)
                    print("  Chunk \(chunkCount): \(String(format: "%.2f", chunkDuration))s (\(chunk.samples.count) samples)")
                }

                let elapsed = CFAbsoluteTimeGetCurrent() - startTime
                let duration = Double(allSamples.count) / 24000.0
                print(String(format: "  Duration: %.2fs, Time: %.2fs, RTF: %.2f",
                             duration, elapsed, elapsed / max(duration, 0.001)))

                if !self.play {
                    let outputURL = URL(fileURLWithPath: self.output)
                    try WAVWriter.write(samples: allSamples, sampleRate: 24000, to: outputURL)
                    print("Saved to \(self.output)")
                } else {
                    self.playAudio(samples: allSamples, sampleRate: 24000)
                }
            } else {
                // Single segment synthesis
                let instruction = segments.first?.emotion.map {
                    DialogueParser.emotionToInstruction($0)
                } ?? defaultInstruction

                var info = "Synthesizing: \"\(inputText)\""
                if defaultEmbedding != nil || !speakerEmbeddings.isEmpty { info += " [voice clone]" }
                if instruction != "You are a helpful assistant." { info += " [instruction: \(instruction)]" }
                print(info)

                let samples = cosyModel.synthesize(
                    text: inputText, language: effectiveLanguage,
                    instruction: instruction,
                    speakerEmbedding: defaultEmbedding,
                    verbose: verbose
                )

                let elapsed = CFAbsoluteTimeGetCurrent() - startTime
                let duration = Double(samples.count) / 24000.0
                print(String(format: "  Duration: %.2fs, Time: %.2fs, RTF: %.2f",
                             duration, elapsed, elapsed / max(duration, 0.001)))

                if !self.play {
                    let outputURL = URL(fileURLWithPath: self.output)
                    try WAVWriter.write(samples: samples, sampleRate: 24000, to: outputURL)
                    print("Saved to \(self.output)")
                } else {
                    self.playAudio(samples: samples, sampleRate: 24000)
                }
            }
        }
    }

    // MARK: - Audio Playback

    private func playAudio(samples: [Float], sampleRate: Int) {
        let engine = AVAudioEngine()
        let playerNode = AVAudioPlayerNode()
        let format = AVAudioFormat(standardFormatWithSampleRate: Double(sampleRate), channels: 1)!

        engine.attach(playerNode)
        engine.connect(playerNode, to: engine.mainMixerNode, format: format)

        let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(samples.count))!
        buffer.frameLength = AVAudioFrameCount(samples.count)
        samples.withUnsafeBufferPointer { src in
            buffer.floatChannelData![0].update(from: src.baseAddress!, count: samples.count)
        }

        do {
            try engine.start()
        } catch {
            print("Error: Failed to start audio engine: \(error)")
            return
        }

        let semaphore = DispatchSemaphore(value: 0)
        playerNode.play()
        playerNode.scheduleBuffer(buffer) {
            semaphore.signal()
        }

        print("Playing \(formatDuration(samples.count))s audio...")
        semaphore.wait()
        // Small delay for audio to finish draining
        usleep(100_000)
        engine.stop()
    }
}
