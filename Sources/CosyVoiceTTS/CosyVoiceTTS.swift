import Foundation
import MLX
import MLXNN
import Qwen3Common

/// Audio chunk for streaming synthesis
public struct CosyVoiceAudioChunk: Sendable {
    public let samples: [Float]
    public let sampleRate: Int
    public let frameIndex: Int
    public let isFinal: Bool
}

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
public final class CosyVoiceTTSModel {
    public let config: CosyVoiceConfig

    let llm: CosyVoiceLLM
    let flow: CosyVoiceFlowModel
    let hifigan: HiFiGANGenerator
    let tokenizer: Qwen3Tokenizer

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
        progressHandler: ((Double, String) -> Void)? = nil
    ) async throws -> CosyVoiceTTSModel {
        let config = CosyVoiceConfig.default
        let model = CosyVoiceTTSModel(config: config)

        // Get cache directory
        let cacheDir = try HuggingFaceDownloader.getCacheDirectory(for: modelId)

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
                ]
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

        progressHandler?(1.0, "Model loaded")
        return model
    }

    /// Run minimal forward passes to compile Metal shaders and set up compiled generation.
    ///
    /// This eliminates first-inference latency from shader compilation (~200ms) and enables
    /// Metal kernel fusion for the LLM generation loop (~360 kernel dispatches fused).
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
    }

    /// Synthesize speech from text (non-streaming).
    ///
    /// Returns: Array of float audio samples at 24kHz
    public func synthesize(
        text: String,
        language: String = "english",
        verbose: Bool = false
    ) -> [Float] {
        // 1. Tokenize text via Qwen2.5 BPE tokenizer
        let textTokens = tokenizeText(text, language: language)

        // 2. Generate speech tokens via LLM
        var t0 = CFAbsoluteTimeGetCurrent()
        let speechTokens = llm.generate(
            textTokens: textTokens,
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

        // 3. Convert speech tokens to mel spectrogram via flow matching
        t0 = CFAbsoluteTimeGetCurrent()
        let tokenArray = MLXArray(speechTokens).expandedDimensions(axis: 0)  // [1, T]
        let mel = flow(tokens: tokenArray)  // [1, 80, T_mel]
        eval(mel)
        if verbose {
            print(String(format: "  Flow: %.0fms", (CFAbsoluteTimeGetCurrent() - t0) * 1000))
        }

        // 4. Convert mel to waveform via HiFi-GAN
        t0 = CFAbsoluteTimeGetCurrent()
        let audio = hifigan(mel)  // [1, samples] or [samples]
        eval(audio)
        if verbose {
            print(String(format: "  HiFi-GAN: %.0fms", (CFAbsoluteTimeGetCurrent() - t0) * 1000))
        }

        // 5. Extract float samples
        return audio.reshaped(-1).asArray(Float.self)
    }

    /// Synthesize with streaming output.
    public func synthesizeStream(
        text: String,
        language: String = "english",
        chunkSize: Int = 25
    ) -> AsyncThrowingStream<CosyVoiceAudioChunk, Error> {
        let (stream, continuation) = AsyncThrowingStream<CosyVoiceAudioChunk, Error>.makeStream()

        Task { [weak self] in
            guard let self else {
                continuation.finish()
                return
            }
            // Non-streaming for now — stream the full result as a single chunk
            let samples = self.synthesize(text: text, language: language)
            let chunk = CosyVoiceAudioChunk(
                samples: samples,
                sampleRate: config.sampleRate,
                frameIndex: 0,
                isFinal: true
            )
            continuation.yield(chunk)
            continuation.finish()
        }

        return stream
    }

    /// Token ID for `<|endofprompt|>` — added by CosyVoice3 but not in base tokenizer config.
    /// The text embedding table (151936 entries) includes this trained embedding at index 151646.
    private static let endOfPromptToken: Int32 = 151646

    /// Format and tokenize text for CosyVoice3 LLM.
    ///
    /// CosyVoice3 requires the text format: `{instruction}<|endofprompt|>{text_to_synthesize}`
    /// The `<|endofprompt|>` token (ID 151646) marks the boundary between instruction and content.
    private func tokenizeText(_ text: String, language: String) -> [Int32] {
        // Encode instruction prefix
        let instruction = "You are a helpful assistant."
        let instructionTokens = tokenizer.encode(instruction).map { Int32($0) }

        // Encode text to synthesize
        let textTokens = tokenizer.encode(text).map { Int32($0) }

        // Concatenate: [instruction_tokens, <|endofprompt|>, text_tokens]
        return instructionTokens + [Self.endOfPromptToken] + textTokens
    }
}
