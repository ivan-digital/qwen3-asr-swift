import Foundation

/// Configuration for Qwen3 chat model (CoreML).
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
        quantization: "int4"
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
    /// Token IDs to suppress (set logit to -inf before sampling).
    public var suppressedTokenIds: [Int]
    /// Prefill `<think>\n</think>\n` so the model skips thinking and responds directly.
    public var disableThinking: Bool
    /// Max tokens allowed inside `<think>...</think>`. After this many thinking tokens,
    /// `</think>` is force-injected and generation continues with the response.
    /// 0 = no cap (unlimited thinking).
    public var maxThinkingTokens: Int

    public init(
        temperature: Float = 0.7,
        topK: Int = 50,
        topP: Float = 0.9,
        maxTokens: Int = 256,
        repetitionPenalty: Float = 1.1,
        suppressedTokenIds: [Int] = [],
        disableThinking: Bool = false,
        maxThinkingTokens: Int = 0
    ) {
        self.temperature = temperature
        self.topK = topK
        self.topP = topP
        self.maxTokens = maxTokens
        self.repetitionPenalty = repetitionPenalty
        self.suppressedTokenIds = suppressedTokenIds
        self.disableThinking = disableThinking
        self.maxThinkingTokens = maxThinkingTokens
    }

    public static let `default` = ChatSamplingConfig()
    public static let creative = ChatSamplingConfig(temperature: 0.9, topP: 0.95)
    public static let precise = ChatSamplingConfig(temperature: 0.3, topK: 20, topP: 0.8)
}
