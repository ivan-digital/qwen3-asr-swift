import Foundation

/// Configuration for MADLAD-400 (T5 v1.1 encoder-decoder).
///
/// Defaults match `google/madlad400-3b-mt`:
/// - 32 encoder + 32 decoder layers
/// - d_model=1024, d_kv=128, num_heads=16, d_ff=8192
/// - vocab=256000 (SentencePiece, includes 400+ language tokens like `<2en>`, `<2es>`)
/// - relative position bias (32 buckets, max distance 128) — added to attention scores
///   only at the first encoder layer and first decoder layer (T5 convention)
/// - gated GeLU FFN (`wi_0`, `wi_1`, `wo`)
/// - RMSNorm with learnable scale, no bias
/// - Separate `lm_head` (NOT tied to embeddings — T5 v1.1 / MADLAD)
/// - decoder_start_token_id = 0, eos_token_id = 2, pad_token_id = 1
public struct MADLADTranslationConfig: Codable, Sendable {
    public let dModel: Int
    public let dKv: Int
    public let dFf: Int
    public let numLayers: Int
    public let numDecoderLayers: Int
    public let numHeads: Int
    public let vocabSize: Int
    public let relativeAttentionNumBuckets: Int
    public let relativeAttentionMaxDistance: Int
    public let layerNormEpsilon: Double
    public let decoderStartTokenId: Int
    public let eosTokenId: Int
    public let padTokenId: Int
    public let tieWordEmbeddings: Bool
    public let quantization: String?

    enum CodingKeys: String, CodingKey {
        case dModel = "d_model"
        case dKv = "d_kv"
        case dFf = "d_ff"
        case numLayers = "num_layers"
        case numDecoderLayers = "num_decoder_layers"
        case numHeads = "num_heads"
        case vocabSize = "vocab_size"
        case relativeAttentionNumBuckets = "relative_attention_num_buckets"
        case relativeAttentionMaxDistance = "relative_attention_max_distance"
        case layerNormEpsilon = "layer_norm_epsilon"
        case decoderStartTokenId = "decoder_start_token_id"
        case eosTokenId = "eos_token_id"
        case padTokenId = "pad_token_id"
        case tieWordEmbeddings = "tie_word_embeddings"
        case quantization
    }

    public init(
        dModel: Int = 1024,
        dKv: Int = 128,
        dFf: Int = 8192,
        numLayers: Int = 32,
        numDecoderLayers: Int = 32,
        numHeads: Int = 16,
        vocabSize: Int = 256_000,
        relativeAttentionNumBuckets: Int = 32,
        relativeAttentionMaxDistance: Int = 128,
        layerNormEpsilon: Double = 1e-6,
        decoderStartTokenId: Int = 0,
        eosTokenId: Int = 2,
        padTokenId: Int = 1,
        tieWordEmbeddings: Bool = false,
        quantization: String? = nil
    ) {
        self.dModel = dModel
        self.dKv = dKv
        self.dFf = dFf
        self.numLayers = numLayers
        self.numDecoderLayers = numDecoderLayers
        self.numHeads = numHeads
        self.vocabSize = vocabSize
        self.relativeAttentionNumBuckets = relativeAttentionNumBuckets
        self.relativeAttentionMaxDistance = relativeAttentionMaxDistance
        self.layerNormEpsilon = layerNormEpsilon
        self.decoderStartTokenId = decoderStartTokenId
        self.eosTokenId = eosTokenId
        self.padTokenId = padTokenId
        self.tieWordEmbeddings = tieWordEmbeddings
        self.quantization = quantization
    }

    /// MADLAD-400-3B-MT defaults.
    public static let madlad3B = MADLADTranslationConfig()

    public static func load(from url: URL) throws -> MADLADTranslationConfig {
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(MADLADTranslationConfig.self, from: data)
    }
}

/// Greedy / sampling parameters for translation decoding.
public struct TranslationSamplingConfig: Sendable {
    public var temperature: Float
    public var topK: Int
    public var topP: Float
    public var maxTokens: Int
    public var repetitionPenalty: Float

    public init(
        temperature: Float = 0.0,
        topK: Int = 0,
        topP: Float = 1.0,
        maxTokens: Int = 256,
        repetitionPenalty: Float = 1.0
    ) {
        self.temperature = temperature
        self.topK = topK
        self.topP = topP
        self.maxTokens = maxTokens
        self.repetitionPenalty = repetitionPenalty
    }

    /// Greedy decoding (deterministic). Recommended default for MT.
    public static let greedy = TranslationSamplingConfig()

    /// Light sampling for paraphrase-style variation.
    public static let sampling = TranslationSamplingConfig(
        temperature: 0.7, topK: 50, topP: 0.9
    )
}
