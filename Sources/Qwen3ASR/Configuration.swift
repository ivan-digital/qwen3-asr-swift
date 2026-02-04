import Foundation

/// Configuration for Qwen3-ASR audio encoder
public struct AudioEncoderConfig: Codable, Sendable {
    public var inputDim: Int = 128          // Mel filterbank bins
    public var hiddenDim: Int = 1024        // Transformer hidden dim
    public var numLayers: Int = 18          // Transformer layers
    public var numHeads: Int = 16           // Attention heads
    public var kernelSize: Int = 3          // Conv kernel size
    public var headDim: Int = 64            // Head dimension (hiddenDim / numHeads)
    public var ffnDim: Int = 4096           // FFN intermediate dim
    public var maxSourcePositions: Int = 1500
    public var layerNormEps: Float = 1e-5
    public var attentionDropout: Float = 0.0
    public var dropoutRate: Float = 0.0
    public var layerdrop: Float = 0.0
    public var numMelBins: Int = 128
    public var projectorHiddenAct: String = "silu"

    public init() {}

    /// Config for Qwen3-ASR-0.6B
    public static var small: AudioEncoderConfig {
        var config = AudioEncoderConfig()
        config.hiddenDim = 768
        config.numLayers = 12
        config.numHeads = 12
        config.headDim = 64
        config.ffnDim = 3072
        return config
    }

    /// Config for Qwen3-ASR-1.7B
    public static var large: AudioEncoderConfig {
        var config = AudioEncoderConfig()
        config.hiddenDim = 1024
        config.numLayers = 18
        config.numHeads = 16
        config.headDim = 64
        config.ffnDim = 4096
        return config
    }
}

/// Configuration for Qwen3 text decoder
public struct TextDecoderConfig: Codable, Sendable {
    public var vocabSize: Int = 151936
    public var hiddenSize: Int = 1024        // Model dimension
    public var numLayers: Int = 28           // Transformer layers
    public var numHeads: Int = 16            // Attention heads
    public var numKVHeads: Int = 8           // KV heads for GQA
    public var headDim: Int = 64             // Head dimension (hiddenSize / numHeads for 0.6B)
    public var intermediateSize: Int = 3072  // FFN intermediate size
    public var maxPositionEmbeddings: Int = 65536
    public var rmsNormEps: Float = 1e-6
    public var ropeTheta: Float = 1000000.0
    public var ropeScaling: RopeScaling? = nil
    public var tieWordEmbeddings: Bool = true

    // Quantization config
    public var groupSize: Int = 64
    public var bits: Int = 4

    public init() {}

    /// Config for Qwen3-ASR-0.6B decoder (from HuggingFace model config)
    public static var small: TextDecoderConfig {
        var config = TextDecoderConfig()
        config.hiddenSize = 1024
        config.numLayers = 28
        config.numHeads = 16
        config.numKVHeads = 8
        config.headDim = 128  // From config.json: head_dim = 128
        config.intermediateSize = 3072
        config.groupSize = 64
        config.bits = 4
        return config
    }

    /// Config for Qwen3-ASR-1.7B decoder
    public static var large: TextDecoderConfig {
        var config = TextDecoderConfig()
        config.hiddenSize = 1536
        config.numLayers = 28
        config.numHeads = 12
        config.numKVHeads = 2
        config.headDim = 128
        config.intermediateSize = 8960
        config.groupSize = 64
        config.bits = 4
        return config
    }
}

/// RoPE scaling configuration
public struct RopeScaling: Codable, Sendable {
    public var type: String
    public var factor: Float?
    public var originalMaxPositionEmbeddings: Int?

    enum CodingKeys: String, CodingKey {
        case type
        case factor
        case originalMaxPositionEmbeddings = "original_max_position_embeddings"
    }
}

/// Combined Qwen3-ASR model configuration
public struct Qwen3ASRConfig: Codable, Sendable {
    public var audioEncoder: AudioEncoderConfig
    public var textDecoder: TextDecoderConfig
    public var audioTokenIndex: Int = 151646
    public var eosTokenId: Int = 151645
    public var padTokenId: Int = 151643

    public init(
        audioEncoder: AudioEncoderConfig = AudioEncoderConfig(),
        textDecoder: TextDecoderConfig = TextDecoderConfig()
    ) {
        self.audioEncoder = audioEncoder
        self.textDecoder = textDecoder
    }

    /// Config for Qwen3-ASR-0.6B
    public static var small: Qwen3ASRConfig {
        Qwen3ASRConfig(
            audioEncoder: .small,
            textDecoder: .small
        )
    }

    /// Config for Qwen3-ASR-1.7B
    public static var large: Qwen3ASRConfig {
        Qwen3ASRConfig(
            audioEncoder: .large,
            textDecoder: .large
        )
    }
}
