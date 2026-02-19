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

    /// Initialize with config
    public init(config: CosyVoiceConfig = .default) {
        self.config = config
        self.llm = CosyVoiceLLM(config: config.llm)
        self.flow = CosyVoiceFlowModel(config: config.flow)
        self.hifigan = HiFiGANGenerator(config: config.hifigan)
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

        // Download if needed
        if !HuggingFaceDownloader.weightsExist(in: cacheDir) {
            progressHandler?(0.0, "Downloading model weights...")
            try await HuggingFaceDownloader.downloadWeights(
                modelId: modelId,
                to: cacheDir,
                additionalFiles: ["llm.safetensors", "flow.safetensors", "hifigan.safetensors"]
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

        progressHandler?(1.0, "Model loaded")
        return model
    }

    /// Synthesize speech from text (non-streaming).
    ///
    /// Returns: Array of float audio samples at 24kHz
    public func synthesize(
        text: String,
        language: String = "english"
    ) -> [Float] {
        // 1. Tokenize text (simple: convert to UTF-8 bytes or use a tokenizer)
        // For now, use a simple approach — full tokenizer support added later
        let textTokens = tokenizeText(text, language: language)

        // 2. Generate speech tokens via LLM
        let speechTokens = llm.generate(
            textTokens: textTokens,
            maxTokens: 500  // ~20 seconds of audio at 25 Hz
        )

        guard !speechTokens.isEmpty else {
            return []
        }

        // 3. Convert speech tokens to mel spectrogram via flow matching
        let tokenArray = MLXArray(speechTokens).expandedDimensions(axis: 0)  // [1, T]
        let mel = flow(tokens: tokenArray)  // [1, 80, T_mel]

        // 4. Convert mel to waveform via HiFi-GAN
        let audio = hifigan(mel)  // [1, samples] or [samples]

        // 5. Extract float samples
        eval(audio)
        let flatAudio = audio.reshaped(-1)
        let count = flatAudio.dim(0)
        var samples = [Float](repeating: 0, count: count)
        for i in 0..<count {
            samples[i] = flatAudio[i].item(Float.self)
        }

        return samples
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

    /// Simple tokenizer — converts text to token IDs.
    /// In a full implementation, this would use the Qwen2.5 tokenizer.
    /// For now, uses UTF-8 byte encoding as a placeholder.
    private func tokenizeText(_ text: String, language: String) -> [Int32] {
        // Placeholder: convert characters to their Unicode codepoint values
        // This will be replaced with proper Qwen2.5 tokenizer
        return text.unicodeScalars.map { Int32($0.value) }
    }
}
