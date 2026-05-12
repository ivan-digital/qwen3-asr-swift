import Foundation
import MLXCommon
import MLX
import MLXNN
import AudioCommon

/// Error types for CosyVoice TTS
public enum CosyVoiceTTSError: Error, LocalizedError {
    case modelLoadFailed(String)
    case downloadFailed(String)
    case invalidInput(String)
    case generationFailed(String)

    public var errorDescription: String? {
        switch self {
        case .modelLoadFailed(let msg): return "Model load failed: \(msg)"
        case .downloadFailed(let msg): return "Download failed: \(msg)"
        case .invalidInput(let msg): return "Invalid input: \(msg)"
        case .generationFailed(let msg): return "Generation failed: \(msg)"
        }
    }
}

/// CosyVoice3 TTS model — generates speech from text.
///
/// Three-stage pipeline:
/// 1. LLM (Qwen2.5-0.5B) generates speech tokens from text
/// 2. Flow matching (DiT) converts tokens to mel spectrogram
/// 3. HiFi-GAN vocoder converts mel to 24kHz audio waveform
///
/// - Warning: This class is not thread-safe. Create separate instances for concurrent use.
public final class CosyVoiceTTSModel {
    public let config: CosyVoiceConfig

    let llm: CosyVoiceLLM
    let flow: CosyVoiceFlowModel
    let hifigan: HiFiGANGenerator
    let tokenizer: Qwen3Tokenizer

    /// Whether the model weights are loaded and ready for inference.
    var _isLoaded = true

    /// Initialize with config
    public init(config: CosyVoiceConfig = .default) {
        self.config = config
        self.llm = CosyVoiceLLM(config: config.llm)
        self.flow = CosyVoiceFlowModel(config: config.flow)
        self.hifigan = HiFiGANGenerator(config: config.hifigan)
        self.tokenizer = Qwen3Tokenizer()
    }

    /// Download and load model from HuggingFace
    ///
    /// Downloads three safetensors files: llm.safetensors, flow.safetensors, hifigan.safetensors
    /// Caches to ~/Library/Caches/qwen3-speech/
    public static func fromPretrained(
        modelId: String = "aufklarer/CosyVoice3-0.5B-MLX-4bit",
        cacheDir: URL? = nil,
        offlineMode: Bool = false,
        progressHandler: ((Double, String) -> Void)? = nil
    ) async throws -> CosyVoiceTTSModel {
        let config = CosyVoiceConfig.default
        let model = CosyVoiceTTSModel(config: config)

        // Get cache directory
        let cacheDir = try cacheDir ?? HuggingFaceDownloader.getCacheDirectory(for: modelId)

        // Download if needed (check both weights and tokenizer)
        let needsWeights = !HuggingFaceDownloader.weightsExist(in: cacheDir)
        let needsTokenizer = !FileManager.default.fileExists(
            atPath: cacheDir.appendingPathComponent("vocab.json").path)

        if needsWeights || needsTokenizer {
            progressHandler?(0.0, "Downloading model files...")
            try await HuggingFaceDownloader.downloadWeights(
                modelId: modelId,
                to: cacheDir,
                additionalFiles: [
                    "llm.safetensors", "flow.safetensors", "hifigan.safetensors",
                    "vocab.json", "merges.txt", "tokenizer_config.json",
                ],
                offlineMode: offlineMode
            ) { progress in
                progressHandler?(progress * 0.5, "Downloading...")
            }
        }

        // Load weights
        progressHandler?(0.5, "Loading LLM weights...")
        let llmURL = cacheDir.appendingPathComponent("llm.safetensors")
        try CosyVoiceWeightLoader.loadLLM(model.llm, from: llmURL)

        progressHandler?(0.7, "Loading flow weights...")
        let flowURL = cacheDir.appendingPathComponent("flow.safetensors")
        try CosyVoiceWeightLoader.loadFlow(model.flow, from: flowURL)

        progressHandler?(0.9, "Loading vocoder weights...")
        let hifiganURL = cacheDir.appendingPathComponent("hifigan.safetensors")
        try CosyVoiceWeightLoader.loadHiFiGAN(model.hifigan, from: hifiganURL)

        // Load tokenizer (Qwen2.5 BPE)
        progressHandler?(0.95, "Loading tokenizer...")
        let vocabURL = cacheDir.appendingPathComponent("vocab.json")
        try model.tokenizer.load(from: vocabURL)

        // Warmup: compile LLM and run dummy forward passes to pre-compile Metal shaders
        progressHandler?(0.98, "Warming up...")
        model.warmUp()

        MetalBudget.pinMemory()
        progressHandler?(1.0, "Model loaded")
        return model
    }

    /// Run minimal forward passes to compile Metal shaders and set up compiled generation.
    ///
    /// This eliminates first-inference latency from shader compilation (~200ms) and enables
    /// Metal kernel fusion for the LLM generation loop (~360 kernel dispatches fused)
    /// and DiT flow matching (~330 kernel dispatches × 10 ODE steps fused).
    public func warmUp() {
        // Set up compiled LLM generation step (shapeless=true, traced on first call)
        llm.setupCompilation()

        // Run a minimal prefill to compile all 24-layer attention + MLP shaders
        let textTokens: [Int32] = [2610]  // single token "You"
        let prefixEmbeds = llm.buildInputSequence(textTokens: textTokens)
        let (prefillLogits, warmupCache) = llm.forwardStep(
            prefixEmbeds, offset: MLXArray(Int32(0)), cache: nil)
        eval(prefillLogits)

        // Trace the compiled step with a single-token generation pass
        let warmupEmbed = llm.speechEmbedding(
            MLXArray([Int32(0)]).expandedDimensions(axis: 0))
        let (warmupLogits, _) = llm.executeStep(
            embeds: warmupEmbed, offset: prefixEmbeds.dim(1), cache: warmupCache)
        eval(warmupLogits)

        // Set up compiled DiT forward pass (shapeless=false, ~330 kernel dispatches fused)
        flow.decoder.setupCompilation()
        flow.decoder.warmUp()
    }

    /// Synthesize speech from text (non-streaming).
    ///
    /// Returns: Array of float audio samples at 24kHz
    public func synthesize(
        text: String,
        language: String = "english",
        verbose: Bool = false
    ) -> [Float] {
        synthesize(text: text, language: language, instruction: "You are a helpful assistant.",
                   speakerEmbedding: nil, verbose: verbose)
    }

    /// Synthesize speech from text with a cloned voice.
    ///
    /// Uses a 192-dim CAM++ speaker embedding to condition the flow model,
    /// producing speech that mimics the voice characteristics of the embedding.
    public func synthesize(
        text: String,
        language: String = "english",
        speakerEmbedding: [Float],
        verbose: Bool = false
    ) -> [Float] {
        synthesize(text: text, language: language, instruction: "You are a helpful assistant.",
                   speakerEmbedding: speakerEmbedding, verbose: verbose)
    }

    /// Unified synthesis with both instruction and optional speaker embedding.
    ///
    /// - Parameters:
    ///   - text: Text to synthesize
    ///   - language: Target language
    ///   - instruction: Style instruction prefix (before `<|endofprompt|>`)
    ///   - speakerEmbedding: Optional 192-dim CAM++ speaker embedding
    ///   - verbose: Print timing info
    /// - Returns: Array of float audio samples at 24kHz
    public func synthesize(
        text: String,
        language: String = "english",
        instruction: String = "You are a helpful assistant.",
        speakerEmbedding: [Float]? = nil,
        promptToken: MLXArray? = nil,
        promptFeat: MLXArray? = nil,
        promptText: String? = nil,
        verbose: Bool = false
    ) -> [Float] {
        // 1. Tokenize text via Qwen2.5 BPE tokenizer. Track the content text
        //    length separately (without the instruction frame) so the LLM's
        //    min/max-len constraints scale to the actual content, not the
        //    instruction. Upstream: `min_len = (text_len - prompt_text_len) * ratio`.
        let contentTokens = tokenizer.encode(text).map { Int32($0) }

        // For zero-shot voice cloning, upstream's text input is literally
        //   concat(transcript_tokens + [<|endofprompt|>], content_tokens)
        // with NO "You are a helpful assistant. " system frame. Adding the
        // system frame puts the LLM off the training distribution and it
        // reads back the tail of the transcript instead of synthesising the
        // user text. So when promptText is set we bypass tokenizeText entirely
        // and build the sequence directly.
        let textTokens: [Int32]
        if let pt = promptText, !pt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            let promptTextTokens = tokenizer.encode(pt).map { Int32($0) }
            textTokens = promptTextTokens + [Self.endOfPromptToken] + contentTokens
        } else {
            textTokens = tokenizeText(text, language: language, instruction: instruction)
        }

        // 2. Generate speech tokens via LLM.
        //    For zero-shot cloning, the reference's FSQ codes are passed as
        //    `promptSpeechTokens` so the LLM's autoregressive state already
        //    encodes the target speaker before generation begins. Without this
        //    the LLM emits "neutral default voice" tokens that conflict with
        //    the flow's prompt_token + prompt_feat anchors and the cloned
        //    output drifts to a different voice (see PR #247).
        let promptSpeechTokensArr: [Int32]? = promptToken.map { pt in
            pt.reshaped(-1).asArray(Int32.self)
        }
        var t0 = CFAbsoluteTimeGetCurrent()
        let speechTokens = llm.generate(
            textTokens: textTokens,
            promptSpeechTokens: promptSpeechTokensArr,
            contentTextLength: contentTokens.count,
            maxTokens: 500  // ~20 seconds of audio at 25 Hz
        )
        if verbose {
            let llmTime = CFAbsoluteTimeGetCurrent() - t0
            print(String(format: "  LLM: %.0fms (%d tokens, %.1fms/token)",
                         llmTime * 1000, speechTokens.count,
                         speechTokens.isEmpty ? 0 : llmTime * 1000 / Double(speechTokens.count)))
        }

        guard !speechTokens.isEmpty else {
            return []
        }

        // 3. Convert speech tokens to mel spectrogram via flow matching.
        //    When promptToken + promptFeat are supplied (the upstream zero-shot
        //    cloning path), the flow returns mel for the *full* prompt + generation
        //    span; we slice off the prompt region before HiFi-GAN.
        t0 = CFAbsoluteTimeGetCurrent()
        let tokenArray = MLXArray(speechTokens).expandedDimensions(axis: 0)  // [1, T]
        let spkEmb: MLXArray? = speakerEmbedding.map {
            MLXArray($0).expandedDimensions(axis: 0)
        }
        let fullMel = flow(
            tokens: tokenArray,
            spkEmbedding: spkEmb,
            promptToken: promptToken,
            promptFeat: promptFeat
        )
        eval(fullMel)

        // Pass the FULL mel (prompt + generation) to HiFi-GAN. Slicing the mel
        // here means HiFi-GAN's causal convolutions warm up against zero-padded
        // boundaries, producing a click/transient ~10 mel frames into the
        // generated audio (the convolutional receptive field). Instead we let
        // HiFi-GAN render both regions continuously and trim the prompt-region
        // audio AFTER, where the boundary is smooth.
        let mel = fullMel
        let promptAudioSamples: Int
        if let pf = promptFeat {
            let promptMelLen = pf.dim(2)
            // mel-rate is 50 Hz, audio sample rate is 24 kHz → 480 samples/mel
            promptAudioSamples = promptMelLen * (config.sampleRate / 50)
        } else {
            promptAudioSamples = 0
        }

        // Optional debug dump of the mel that reaches HiFi-GAN.
        if let dumpDir = ProcessInfo.processInfo.environment["COSY_DEBUG_DUMP_DIR"] {
            CosyVoiceDebugDump.tryWrite(mel, name: "swift_hifigan_input_mel", in: dumpDir)
        }

        if verbose {
            var path: [String] = []
            if speakerEmbedding != nil { path.append("spk") }
            if promptToken != nil { path.append("prompt_token") }
            if promptFeat != nil { path.append("prompt_feat") }
            let suffix = path.isEmpty ? "" : " (\(path.joined(separator: "+")))"
            print(String(format: "  Flow: %.0fms%@", (CFAbsoluteTimeGetCurrent() - t0) * 1000, suffix))
        }

        // 4. Convert mel to waveform via HiFi-GAN
        t0 = CFAbsoluteTimeGetCurrent()
        let audio = hifigan(mel)
        eval(audio)
        if verbose {
            print(String(format: "  HiFi-GAN: %.0fms", (CFAbsoluteTimeGetCurrent() - t0) * 1000))
        }

        // 5. Extract float samples, trimming the prompt-region audio so the
        //    caller only sees the synthesised content. The boundary inside
        //    HiFi-GAN's continuous render is smoother than a pre-sliced mel,
        //    so trimming here removes the slice-boundary click that appeared
        //    ~10 mel frames into the audio in the old path.
        var samples = audio.reshaped(-1).asArray(Float.self)
        if promptAudioSamples > 0 && promptAudioSamples < samples.count {
            samples = Array(samples[promptAudioSamples...])
        }
        return samples
    }

    /// Synthesize with streaming output.
    /// Synthesize speech asynchronously, yielding audio in chunks.
    ///
    /// Currently yields a single chunk with the full result (not yet
    /// incrementally streaming). The async stream interface is preserved
    /// for forward compatibility with chunked codec decoding.
    public func synthesizeStream(
        text: String,
        language: String = "english",
        chunkSize: Int = 25
    ) -> AsyncThrowingStream<AudioChunk, Error> {
        let rate = config.sampleRate
        return AsyncThrowingStream { continuation in
            let task = Task { [weak self] in
                guard let self else {
                    continuation.finish()
                    return
                }
                do {
                    let samples = self.synthesize(text: text, language: language)
                    let chunk = AudioChunk(
                        samples: samples,
                        sampleRate: rate,
                        frameIndex: 0,
                        isFinal: true
                    )
                    continuation.yield(chunk)
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { @Sendable _ in task.cancel() }
        }
    }

    /// Token ID for `<|endofprompt|>` — added by CosyVoice3 but not in base tokenizer config.
    /// The text embedding table (151936 entries) includes this trained embedding at index 151646.
    private static let endOfPromptToken: Int32 = 151646

    /// System-prompt frame that upstream uses in every official example.
    /// Custom style instructions are *appended* to this frame, not substituted
    /// for it — otherwise the model treats the instruction as content to speak.
    private static let assistantPrefix = "You are a helpful assistant."

    /// Format and tokenize text for CosyVoice3 LLM.
    ///
    /// Upstream training format: `"You are a helpful assistant. {style}<|endofprompt|>{text}"`.
    /// Stripping the assistant prefix pushes the model out of distribution and it
    /// reads the style instruction aloud instead of using it as conditioning.
    private func tokenizeText(
        _ text: String, language: String,
        instruction: String = "You are a helpful assistant."
    ) -> [Int32] {
        let framedInstruction: String
        let trimmed = instruction.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty || trimmed == Self.assistantPrefix {
            framedInstruction = Self.assistantPrefix
        } else if trimmed.hasPrefix(Self.assistantPrefix) {
            framedInstruction = trimmed
        } else {
            framedInstruction = "\(Self.assistantPrefix) \(trimmed)"
        }

        let instructionTokens = tokenizer.encode(framedInstruction).map { Int32($0) }
        let textTokens = tokenizer.encode(text).map { Int32($0) }

        return instructionTokens + [Self.endOfPromptToken] + textTokens
    }
}
