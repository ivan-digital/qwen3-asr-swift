import Foundation
import MLXCommon
import MLX
import MLXNN
import MLXFast
import AudioCommon

/// Optional decoder tunables for `Qwen3ASRModel.transcribe(audio:options:)`.
///
/// Defaults match the historical greedy behaviour of `transcribe(audio:)`
/// so existing callers see zero change. Tune these when greedy decoding
/// collapses onto a single token (typical on silence or ambiguous phonemes).
public struct Qwen3DecodingOptions: Sendable {
    /// Cap on decoder output per chunk.
    public var maxTokens: Int = 448

    /// Optional language hint ("en", "zh", …). `nil` = auto-detect.
    public var language: String?

    /// Context hint prepended to the decoder prompt.
    public var context: String?

    /// HuggingFace-style repetition penalty. Divides the logits of tokens
    /// already generated this chunk by this factor before `argMax`.
    /// `1.0` disables; `1.1`–`1.3` is the common tuning range.
    public var repetitionPenalty: Float = 1.0

    /// If > 0, masks any next-token whose emission would form a repeated
    /// n-gram of this size. `0` disables.
    public var noRepeatNgramSize: Int = 0

    /// `0` = greedy (argmax). `> 0` = sample with this temperature via
    /// Gumbel-max. Higher = more random.
    public var temperature: Float = 0.0

    public init(
        maxTokens: Int = 448,
        language: String? = nil,
        context: String? = nil,
        repetitionPenalty: Float = 1.0,
        noRepeatNgramSize: Int = 0,
        temperature: Float = 0.0
    ) {
        self.maxTokens = maxTokens
        self.language = language
        self.context = context
        self.repetitionPenalty = repetitionPenalty
        self.noRepeatNgramSize = noRepeatNgramSize
        self.temperature = temperature
    }
}

/// Special token IDs for Qwen3-ASR
public struct Qwen3ASRTokens: Sendable {
    public static let audioTokenId = 151676        // <|audio_pad|>
    public static let audioStartTokenId = 151669   // <|audio_start|>
    public static let audioEndTokenId = 151670     // <|audio_end|>
    public static let eosTokenId = 151645          // <|im_end|>
    public static let padTokenId = 151643          // <|endoftext|>
    public static let imStartTokenId = 151644      // <|im_start|>
    public static let imEndTokenId = 151645        // <|im_end|>
    public static let timestampTokenId = 151705    // <|timestamp|>
}

/// Main Qwen3-ASR model for speech recognition.
///
/// - Warning: This class is not thread-safe. Create separate instances for concurrent use.
public class Qwen3ASRModel {
    public let audioEncoder: Qwen3AudioEncoder
    public let featureExtractor: WhisperFeatureExtractor
    public var textDecoder: QuantizedTextModel?

    /// Tokenizer for decoding output tokens
    private var tokenizer: Qwen3Tokenizer?

    /// Text decoder config
    public let textConfig: TextDecoderConfig

    /// Whether the model weights are loaded and ready for inference.
    var _isLoaded = true

    init(
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
    func setTokenizer(_ tokenizer: Qwen3Tokenizer) {
        self.tokenizer = tokenizer
    }

    /// Initialize text decoder (called after loading)
    func initializeTextDecoder() {
        self.textDecoder = QuantizedTextModel(config: textConfig)
    }

    /// Transcribe audio to text with explicit decoder options.
    ///
    /// The legacy `transcribe(audio:sampleRate:language:maxTokens:context:)`
    /// overload below forwards into this path with default (greedy) options.
    public func transcribe(
        audio: [Float],
        sampleRate: Int = 16000,
        options: Qwen3DecodingOptions
    ) -> String {
        let melFeatures = featureExtractor.process(audio, sampleRate: sampleRate)
        let batchedFeatures = melFeatures.expandedDimensions(axis: 0)
        var audioEmbeds = audioEncoder(batchedFeatures)
        audioEmbeds = audioEmbeds.expandedDimensions(axis: 0)
        guard let textDecoder = textDecoder else {
            let shape = audioEmbeds.shape
            return "[Audio encoded: \(shape)] - Text decoder not loaded"
        }
        return generateText(
            audioEmbeds: audioEmbeds,
            textDecoder: textDecoder,
            language: options.language,
            maxTokens: options.maxTokens,
            context: options.context,
            decodingOptions: options
        )
    }

    /// Transcribe audio to text
    public func transcribe(
        audio: [Float],
        sampleRate: Int = 16000,
        language: String? = nil,
        maxTokens: Int = 448,
        context: String? = nil
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
            maxTokens: maxTokens,
            context: context
        )
    }

    /// Generate text from audio embeddings.
    ///
    /// When `decodingOptions` is supplied, the decoder loop applies an
    /// HF-style repetition penalty, an optional no-repeat n-gram mask, and
    /// optional temperature sampling before each token selection. With the
    /// default `Qwen3DecodingOptions()` (repetition=1.0, no-repeat=0,
    /// temperature=0) behaviour is bit-identical to plain greedy.
    func generateText(
        audioEmbeds: MLXArray,
        textDecoder: QuantizedTextModel,
        language: String?,
        maxTokens: Int,
        context: String? = nil,
        decodingOptions: Qwen3DecodingOptions = Qwen3DecodingOptions()
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

        // <|im_start|>system\n{context}<|im_end|>\n
        inputIds.append(contentsOf: [imStartId, systemId, newlineId].map { Int32($0) })
        if let context = context, !context.isEmpty, let tokenizer = tokenizer {
            let contextTokens = tokenizer.encode(context)
            inputIds.append(contentsOf: contextTokens.map { Int32($0) })
        }
        inputIds.append(contentsOf: [imEndId, newlineId].map { Int32($0) })

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

        // Add language hint if specified, then always add <|asr_text|> marker.
        // Without <|asr_text|>, the model doesn't know it should transcribe.
        // Without language hint, the model auto-detects and prepends "language XX" to output.
        if let lang = language, let tokenizer = tokenizer {
            let langPrefix = "language \(lang)"
            let langTokens = tokenizer.encode(langPrefix)
            inputIds.append(contentsOf: langTokens.map { Int32($0) })
        }
        inputIds.append(Int32(asrTextId))

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
        var nextToken = Self.pickNextToken(
            logits: logits,
            generatedSoFar: generatedTokens,
            options: decodingOptions
        )
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
            nextToken = Self.pickNextToken(
                logits: logits,
                generatedSoFar: generatedTokens,
                options: decodingOptions
            )
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

    // MARK: - Decoder knobs

    /// Pick the next token from a logits tensor, applying repetition
    /// penalty, no-repeat n-gram masking, and optional temperature sampling.
    ///
    /// With default options (repetition=1.0, noRepeat=0, temperature=0) the
    /// result is the same `argMax` the decoder used pre-refactor.
    /// Implementation pulls logits to CPU (a 1-D Float array of vocab size)
    /// so we can manipulate entries in-place without fighting MLX indexing.
    private static func pickNextToken(
        logits: MLXArray,
        generatedSoFar: [Int32],
        options: Qwen3DecodingOptions
    ) -> Int32 {
        // Fast path — pure greedy, no modifications.
        if options.repetitionPenalty == 1.0,
           options.noRepeatNgramSize == 0,
           options.temperature == 0 {
            return argMax(logits, axis: -1).squeezed().item(Int32.self)
        }

        // Pull logits to CPU. `logits` is [1, 1, vocabSize]; after squeeze
        // and conversion we have a plain `[Float]` of length vocabSize.
        let flat = logits.squeezed().asType(.float32)
        let vocabSize = flat.size
        var scores: [Float] = flat.asArray(Float.self)
        precondition(scores.count == vocabSize, "pickNextToken: vocab size mismatch")

        // Repetition penalty: divide logits for already-generated tokens.
        if options.repetitionPenalty > 1.0 && !generatedSoFar.isEmpty {
            let penalty = options.repetitionPenalty
            for token in Set(generatedSoFar) {
                let idx = Int(token)
                guard idx >= 0, idx < vocabSize else { continue }
                let v = scores[idx]
                // Positive logits divide; negative logits multiply — matches
                // HuggingFace's implementation so the penalty always reduces
                // the probability of the repeated token.
                scores[idx] = v > 0 ? v / penalty : v * penalty
            }
        }

        // No-repeat-ngram: any next token whose emission would form a
        // repeated n-gram of size N gets pushed to -infinity.
        let n = options.noRepeatNgramSize
        if n > 0 && generatedSoFar.count >= n - 1 {
            let lastPrefix = Array(generatedSoFar.suffix(n - 1))
            // Walk every position where `lastPrefix` already appeared —
            // the token that followed it becomes forbidden as the NEXT
            // token now.
            if generatedSoFar.count >= n {
                for i in 0...(generatedSoFar.count - n) {
                    let window = Array(generatedSoFar[i..<(i + n - 1)])
                    guard window == lastPrefix else { continue }
                    let forbidden = Int(generatedSoFar[i + n - 1])
                    if forbidden >= 0 && forbidden < vocabSize {
                        scores[forbidden] = -.infinity
                    }
                }
            }
        }

        // Temperature sampling via Gumbel-max trick:
        // argmax(logits/T + Gumbel(0,1)) ~ categorical(softmax(logits/T)).
        if options.temperature > 0 {
            let t = options.temperature
            for i in 0..<vocabSize {
                let u = Float.random(in: 1e-6...1.0)
                scores[i] = scores[i] / t - Float.log(-Float.log(u))
            }
        }

        // Argmax of the adjusted scores.
        var bestIdx = 0
        var bestScore = -Float.infinity
        for i in 0..<vocabSize where scores[i] > bestScore {
            bestScore = scores[i]
            bestIdx = i
        }
        return Int32(bestIdx)
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
        case .small: return "aufklarer/Qwen3-ASR-0.6B-MLX-4bit"
        case .large: return "aufklarer/Qwen3-ASR-1.7B-MLX-8bit"
        }
    }

    /// Audio encoder config for this model size
    public var audioConfig: Qwen3AudioEncoderConfig {
        switch self {
        case .small: return .small
        case .large: return .large
        }
    }

    /// Text decoder config for this model size and quantization bits
    public func textConfig(bits: Int) -> TextDecoderConfig {
        switch (self, bits) {
        case (.small, 8): return .small8bit
        case (.small, _): return .small
        case (.large, 8): return .large8bit
        case (.large, _): return .large
        }
    }

    /// Text decoder config for this model size (default bits)
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

    /// Detect quantization bits from a HuggingFace model ID.
    /// Returns 4 by default for 0.6B, 8 for 1.7B if not specified.
    public static func detectBits(from modelId: String) -> Int {
        let lower = modelId.lowercased()
        if lower.contains("8bit") || lower.contains("8-bit") {
            return 8
        }
        if lower.contains("4bit") || lower.contains("4-bit") {
            return 4
        }
        // Default: 4 for small, 8 for large (backwards-compatible)
        let size = detect(from: modelId)
        return size == .large ? 8 : 4
    }
}

// MARK: - Model Loading

public extension Qwen3ASRModel {
    /// Load model from HuggingFace hub with automatic weight downloading
    static func fromPretrained(
        modelId: String = "aufklarer/Qwen3-ASR-0.6B-MLX-4bit",
        cacheDir: URL? = nil,
        offlineMode: Bool = false,
        progressHandler: ((Double, String) -> Void)? = nil
    ) async throws -> Qwen3ASRModel {
        progressHandler?(0.0, "Downloading model...")

        // Auto-detect model size and quantization bits from model ID
        let modelSize = ASRModelSize.detect(from: modelId)
        let detectedBits = ASRModelSize.detectBits(from: modelId)

        // Get cache directory
        let cacheDir = try cacheDir ?? HuggingFaceDownloader.getCacheDirectory(for: modelId)

        // Download weights and tokenizer files (skips files that already exist on disk)
        // Download is the slowest part — give it 0-80% of progress
        try await HuggingFaceDownloader.downloadWeights(
            modelId: modelId,
            to: cacheDir,
            additionalFiles: ["vocab.json", "merges.txt", "tokenizer_config.json"],
            offlineMode: offlineMode,
            progressHandler: { progress in
                progressHandler?(progress * 0.8, "Downloading weights...")
            }
        )

        progressHandler?(0.80, "Loading tokenizer...")

        // Create model with appropriate config for detected size and bits
        let model = Qwen3ASRModel(
            audioConfig: modelSize.audioConfig,
            textConfig: modelSize.textConfig(bits: detectedBits)
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

        MetalBudget.pinMemory()
        progressHandler?(1.0, "Ready")

        return model
    }
}
