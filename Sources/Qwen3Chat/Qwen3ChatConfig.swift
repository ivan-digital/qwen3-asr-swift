import Foundation

/// Model architecture type.
public enum ChatModelArch: String, Codable, Sendable {
    /// Standard Qwen3 transformer (all layers use full attention)
    case qwen3 = "qwen3"
    /// Qwen3.5 hybrid (DeltaNet linear attention + GatedAttention)
    case qwen35 = "qwen3_5_text"
}

/// Configuration for Qwen3/3.5 chat model (CoreML).
public struct Qwen3ChatConfig: Codable, Sendable {
    public let hiddenSize: Int
    public let numHiddenLayers: Int
    public let numAttentionHeads: Int
    public let numKeyValueHeads: Int
    public let headDim: Int
    public let intermediateSize: Int
    public let vocabSize: Int
    public let maxSeqLen: Int
    public let ropeTheta: Double
    public let rmsNormEps: Double
    public let eosTokenId: Int
    public let padTokenId: Int
    public let quantization: String

    // Qwen3.5-specific fields
    public let modelType: ChatModelArch?
    /// Per-layer type: "linear_attention" (DeltaNet) or "full_attention" (GatedAttention)
    public let layerTypes: [String]?
    /// How often a full_attention layer appears (e.g., 4 = every 4th layer)
    public let fullAttentionInterval: Int?
    /// DeltaNet linear attention head config
    public let linearNumKeyHeads: Int?
    public let linearKeyHeadDim: Int?
    public let linearNumValueHeads: Int?
    public let linearValueHeadDim: Int?
    /// Causal conv1d kernel size for DeltaNet
    public let linearConvKernelDim: Int?
    /// Partial RoPE factor for GatedAttention (e.g., 0.25)
    public let partialRotaryFactor: Double?
    /// Whether embeddings are tied (lm_head = embed_tokens)
    public let tieWordEmbeddings: Bool?

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case intermediateSize = "intermediate_size"
        case vocabSize = "vocab_size"
        case maxSeqLen = "max_seq_len"
        case ropeTheta = "rope_theta"
        case rmsNormEps = "rms_norm_eps"
        case eosTokenId = "eos_token_id"
        case padTokenId = "pad_token_id"
        case quantization
        case modelType = "model_type"
        case layerTypes = "layer_types"
        case fullAttentionInterval = "full_attention_interval"
        case linearNumKeyHeads = "linear_num_key_heads"
        case linearKeyHeadDim = "linear_key_head_dim"
        case linearNumValueHeads = "linear_num_value_heads"
        case linearValueHeadDim = "linear_value_head_dim"
        case linearConvKernelDim = "linear_conv_kernel_dim"
        case partialRotaryFactor = "partial_rotary_factor"
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    /// Whether this is a Qwen3.5 hybrid model.
    public var isQwen35: Bool {
        modelType == .qwen35 || layerTypes != nil
    }

    /// Number of full-attention layers (that need KV cache).
    public var numFullAttentionLayers: Int {
        guard let types = layerTypes else { return numHiddenLayers }
        return types.filter { $0 == "full_attention" }.count
    }

    /// Default config for Qwen3-0.6B.
    public static let qwen3_06B = Qwen3ChatConfig(
        hiddenSize: 1024,
        numHiddenLayers: 28,
        numAttentionHeads: 16,
        numKeyValueHeads: 8,
        headDim: 128,
        intermediateSize: 3072,
        vocabSize: 151936,
        maxSeqLen: 2048,
        ropeTheta: 1_000_000.0,
        rmsNormEps: 1e-6,
        eosTokenId: 151645,
        padTokenId: 151643,
        quantization: "int4",
        modelType: .qwen3,
        layerTypes: nil,
        fullAttentionInterval: nil,
        linearNumKeyHeads: nil,
        linearKeyHeadDim: nil,
        linearNumValueHeads: nil,
        linearValueHeadDim: nil,
        linearConvKernelDim: nil,
        partialRotaryFactor: nil,
        tieWordEmbeddings: false
    )

    /// Default config for Qwen3.5-0.8B.
    public static let qwen35_08B = Qwen3ChatConfig(
        hiddenSize: 1024,
        numHiddenLayers: 24,
        numAttentionHeads: 8,
        numKeyValueHeads: 2,
        headDim: 256,
        intermediateSize: 3584,
        vocabSize: 248320,
        maxSeqLen: 2048,
        ropeTheta: 10_000_000.0,
        rmsNormEps: 1e-6,
        eosTokenId: 248044,
        padTokenId: 248043,
        quantization: "int4",
        modelType: .qwen35,
        layerTypes: [
            "linear_attention", "linear_attention", "linear_attention", "full_attention",
            "linear_attention", "linear_attention", "linear_attention", "full_attention",
            "linear_attention", "linear_attention", "linear_attention", "full_attention",
            "linear_attention", "linear_attention", "linear_attention", "full_attention",
            "linear_attention", "linear_attention", "linear_attention", "full_attention",
            "linear_attention", "linear_attention", "linear_attention", "full_attention",
        ],
        fullAttentionInterval: 4,
        linearNumKeyHeads: 16,
        linearKeyHeadDim: 128,
        linearNumValueHeads: 16,
        linearValueHeadDim: 128,
        linearConvKernelDim: 4,
        partialRotaryFactor: 0.25,
        tieWordEmbeddings: true
    )

    /// Load config from a JSON file.
    public static func load(from url: URL) throws -> Qwen3ChatConfig {
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(Qwen3ChatConfig.self, from: data)
    }
}

/// Sampling parameters for text generation.
public struct ChatSamplingConfig: Sendable {
    public var temperature: Float
    public var topK: Int
    public var topP: Float
    public var maxTokens: Int
    public var repetitionPenalty: Float

    public init(
        temperature: Float = 0.7,
        topK: Int = 50,
        topP: Float = 0.9,
        maxTokens: Int = 256,
        repetitionPenalty: Float = 1.1
    ) {
        self.temperature = temperature
        self.topK = topK
        self.topP = topP
        self.maxTokens = maxTokens
        self.repetitionPenalty = repetitionPenalty
    }

    public static let `default` = ChatSamplingConfig()
    public static let creative = ChatSamplingConfig(temperature: 0.9, topP: 0.95)
    public static let precise = ChatSamplingConfig(temperature: 0.3, topK: 20, topP: 0.8)
}
