import Foundation
import MLX
import MLXNN
import MLXFast
import Qwen3Common

/// Special token IDs for Qwen3-ASR
public struct Qwen3ASRTokens {
    public static let audioTokenId = 151676        // <|audio_pad|>
    public static let audioStartTokenId = 151669   // <|audio_start|>
    public static let audioEndTokenId = 151670     // <|audio_end|>
    public static let eosTokenId = 151645          // <|im_end|>
    public static let padTokenId = 151643          // <|endoftext|>
    public static let imStartTokenId = 151644      // <|im_start|>
    public static let imEndTokenId = 151645        // <|im_end|>
    public static let timestampTokenId = 151705    // <|timestamp|>
}

/// Main Qwen3-ASR model for speech recognition
public class Qwen3ASRModel {
    public let audioEncoder: Qwen3AudioEncoder
    public let featureExtractor: WhisperFeatureExtractor
    public var textDecoder: QuantizedTextModel?

    /// Tokenizer for decoding output tokens
    private var tokenizer: Qwen3Tokenizer?

    /// Text decoder config
    public let textConfig: TextDecoderConfig

    public init(
        audioConfig: Qwen3AudioEncoderConfig = .default,
        textConfig: TextDecoderConfig = .small
    ) {
        self.audioEncoder = Qwen3AudioEncoder(config: audioConfig)
        self.featureExtractor = WhisperFeatureExtractor()
        self.textConfig = textConfig
        // Text decoder will be initialized when loading weights
        self.textDecoder = nil
    }

    /// Set tokenizer for text decoding
    public func setTokenizer(_ tokenizer: Qwen3Tokenizer) {
        self.tokenizer = tokenizer
    }

    /// Initialize text decoder (called after loading)
    public func initializeTextDecoder() {
        self.textDecoder = QuantizedTextModel(config: textConfig)
    }

    /// Transcribe audio to text
    public func transcribe(
        audio: [Float],
        sampleRate: Int = 16000,
        language: String? = nil,
        maxTokens: Int = 448
    ) -> String {
        // Extract mel features
        let melFeatures = featureExtractor.process(audio, sampleRate: sampleRate)

        // Add batch dimension: [mel, time] -> [1, mel, time]
        let batchedFeatures = melFeatures.expandedDimensions(axis: 0)

        // Encode audio - returns [time, features] without batch dim (matching Python)
        var audioEmbeds = audioEncoder(batchedFeatures)

        // Add batch dimension for consistency: [time, features] -> [1, time, features]
        audioEmbeds = audioEmbeds.expandedDimensions(axis: 0)

        // Check if text decoder is loaded
        guard let textDecoder = textDecoder else {
            let shape = audioEmbeds.shape
            return "[Audio encoded: \(shape)] - Text decoder not loaded"
        }

        // Generate text using the text decoder
        return generateText(
            audioEmbeds: audioEmbeds,
            textDecoder: textDecoder,
            language: language,
            maxTokens: maxTokens
        )
    }

    /// Generate text from audio embeddings
    private func generateText(
        audioEmbeds: MLXArray,
        textDecoder: QuantizedTextModel,
        language: String?,
        maxTokens: Int
    ) -> String {
        // Special token IDs
        let imStartId = 151644
        let imEndId = 151645
        let audioStartId = 151669
        let audioEndId = 151670
        let audioPadId = 151676
        let asrTextId = 151704
        let newlineId = 198

        // Token IDs for "system", "user", "assistant"
        let systemId = 8948
        let userId = 872
        let assistantId = 77091

        // Number of audio tokens (from audio encoder output)
        let numAudioTokens = audioEmbeds.dim(1)

        // Build input_ids array with audio_pad placeholder tokens
        var inputIds: [Int32] = []

        // <|im_start|>system\n<|im_end|>\n
        inputIds.append(contentsOf: [imStartId, systemId, newlineId, imEndId, newlineId].map { Int32($0) })

        // <|im_start|>user\n<|audio_start|>
        inputIds.append(contentsOf: [imStartId, userId, newlineId, audioStartId].map { Int32($0) })

        // <|audio_pad|> * numAudioTokens (placeholder tokens that will be replaced)
        let audioStartIndex = inputIds.count
        for _ in 0..<numAudioTokens {
            inputIds.append(Int32(audioPadId))
        }
        let audioEndIndex = inputIds.count

        // <|audio_end|><|im_end|>\n
        inputIds.append(contentsOf: [audioEndId, imEndId, newlineId].map { Int32($0) })

        // <|im_start|>assistant\n
        inputIds.append(contentsOf: [imStartId, assistantId, newlineId].map { Int32($0) })

        if let lang = language, let tokenizer = tokenizer {
            let langPrefix = "language \(lang)"
            let langTokens = tokenizer.encode(langPrefix)
            inputIds.append(contentsOf: langTokens.map { Int32($0) })
            inputIds.append(Int32(asrTextId))
        }

        // Get text embeddings for all tokens
        let inputIdsTensor = MLXArray(inputIds).expandedDimensions(axis: 0)
        var inputEmbeds = textDecoder.embedTokens(inputIdsTensor)

        // Replace audio_pad token positions with actual audio embeddings
        let audioEmbedsTyped = audioEmbeds.asType(inputEmbeds.dtype)
        let beforeAudio = inputEmbeds[0..., 0..<audioStartIndex, 0...]
        let afterAudio = inputEmbeds[0..., audioEndIndex..., 0...]

        inputEmbeds = concatenated([beforeAudio, audioEmbedsTyped, afterAudio], axis: 1)

        // Initialize KV cache
        var cache: [(MLXArray, MLXArray)]? = nil

        // Generate tokens
        var generatedTokens: [Int32] = []

        // First pass: process the full input embeddings
        var (hiddenStates, newCache) = textDecoder(inputsEmbeds: inputEmbeds, cache: cache)
        cache = newCache

        // Get logits from the last position using embedding as LM head (tied weights)
        let seqLen = hiddenStates.dim(1)
        let lastHidden = hiddenStates[0..., (seqLen-1)..<seqLen, 0...]
        var logits = textDecoder.embedTokens.asLinear(lastHidden)
        var nextToken = argMax(logits, axis: -1).squeezed().item(Int32.self)
        generatedTokens.append(nextToken)

        // Continue generating
        for _ in 1..<maxTokens {
            // Check for EOS
            if nextToken == Int32(Qwen3ASRTokens.eosTokenId) {
                break
            }

            // Get embedding for the new token
            let tokenEmbeds = textDecoder.embedTokens(MLXArray([nextToken]).expandedDimensions(axis: 0))

            // Forward pass with cache
            (hiddenStates, newCache) = textDecoder(inputsEmbeds: tokenEmbeds, cache: cache)
            cache = newCache

            // Get next token
            let lastHiddenNext = hiddenStates[0..., (-1)..., .ellipsis]
            logits = textDecoder.embedTokens.asLinear(lastHiddenNext)
            nextToken = argMax(logits, axis: -1).squeezed().item(Int32.self)
            generatedTokens.append(nextToken)
        }

        // Decode tokens to text
        if let tokenizer = tokenizer {
            let rawText = tokenizer.decode(tokens: generatedTokens.map { Int($0) })
            // Strip "language XX<asr_text>" prefix if present (auto-detection output)
            if let range = rawText.range(of: "<asr_text>") {
                return String(rawText[range.upperBound...]).trimmingCharacters(in: .whitespaces)
            }
            return rawText
        } else {
            // Fallback: return token IDs
            return generatedTokens.map { String($0) }.joined(separator: " ")
        }
    }
}

// MARK: - Backward Compatibility (delegates to HuggingFaceDownloader)

public extension Qwen3ASRModel {
    static func sanitizedCacheKey(for modelId: String) -> String {
        HuggingFaceDownloader.sanitizedCacheKey(for: modelId)
    }

    static func validatedRemoteFileName(_ file: String) throws -> String {
        try HuggingFaceDownloader.validatedRemoteFileName(file)
    }

    static func validatedLocalPath(directory: URL, fileName: String) throws -> URL {
        try HuggingFaceDownloader.validatedLocalPath(directory: directory, fileName: fileName)
    }
}

// MARK: - Model Size Detection

/// Supported ASR model sizes
public enum ASRModelSize {
    case small  // 0.6B
    case large  // 1.7B

    /// Default model IDs on HuggingFace
    public var defaultModelId: String {
        switch self {
        case .small: return "mlx-community/Qwen3-ASR-0.6B-4bit"
        case .large: return "mlx-community/Qwen3-ASR-1.7B-8bit"
        }
    }

    /// Audio encoder config for this model size
    public var audioConfig: Qwen3AudioEncoderConfig {
        switch self {
        case .small: return .small
        case .large: return .large
        }
    }

    /// Text decoder config for this model size
    public var textConfig: TextDecoderConfig {
        switch self {
        case .small: return .small
        case .large: return .large
        }
    }

    /// Detect model size from a HuggingFace model ID
    public static func detect(from modelId: String) -> ASRModelSize {
        if modelId.contains("1.7B") || modelId.contains("1.7b") {
            return .large
        }
        return .small
    }
}

// MARK: - Model Loading

public extension Qwen3ASRModel {
    /// Load model from HuggingFace hub with automatic weight downloading
    static func fromPretrained(
        modelId: String = "mlx-community/Qwen3-ASR-0.6B-4bit",
        progressHandler: ((Double, String) -> Void)? = nil
    ) async throws -> Qwen3ASRModel {
        progressHandler?(0.0, "Downloading model...")

        // Auto-detect model size from model ID
        let modelSize = ASRModelSize.detect(from: modelId)

        // Get cache directory
        let cacheDir = try HuggingFaceDownloader.getCacheDirectory(for: modelId)

        // Download weights and tokenizer files (skips files that already exist on disk)
        // Download is the slowest part — give it 0-80% of progress
        try await HuggingFaceDownloader.downloadWeights(
            modelId: modelId,
            to: cacheDir,
            additionalFiles: ["vocab.json", "merges.txt", "tokenizer_config.json"],
            progressHandler: { progress in
                progressHandler?(progress * 0.8, "Downloading weights...")
            }
        )

        progressHandler?(0.80, "Loading tokenizer...")

        // Create model with appropriate config for detected size
        let model = Qwen3ASRModel(
            audioConfig: modelSize.audioConfig,
            textConfig: modelSize.textConfig
        )

        // Load tokenizer from vocab.json
        let vocabPath = cacheDir.appendingPathComponent("vocab.json")
        if FileManager.default.fileExists(atPath: vocabPath.path) {
            let tokenizer = Qwen3Tokenizer()
            try tokenizer.load(from: vocabPath)
            model.setTokenizer(tokenizer)
        }

        progressHandler?(0.85, "Loading audio encoder weights...")

        // Load audio encoder weights
        try WeightLoader.loadWeights(into: model.audioEncoder, from: cacheDir)

        progressHandler?(0.92, "Loading text decoder weights...")

        // Initialize and load text decoder
        model.initializeTextDecoder()
        if let textDecoder = model.textDecoder {
            try WeightLoader.loadTextDecoderWeights(into: textDecoder, from: cacheDir)
        }

        progressHandler?(1.0, "Ready")

        return model
    }
}
