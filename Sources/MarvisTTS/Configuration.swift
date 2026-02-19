import Foundation

// MARK: - StringOrNumber (for rope_scaling config values)

public enum StringOrNumber: Codable, Equatable, Hashable, Sendable {
    case string(String)
    case float(Float)

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let v = try? container.decode(Float.self) {
            self = .float(v)
        } else {
            self = .string(try container.decode(String.self))
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .string(let s): try container.encode(s)
        case .float(let n): try container.encode(n)
        }
    }
}

// MARK: - MarvisJSONValue (generic JSON for quantization config)

public enum MarvisJSONValue: Codable, Equatable, Sendable {
    case string(String)
    case number(Double)
    case bool(Bool)
    case object([String: MarvisJSONValue])
    case array([MarvisJSONValue])
    case null

    public init(from decoder: Swift.Decoder) throws {
        let c = try decoder.singleValueContainer()
        if c.decodeNil() { self = .null; return }
        if let b = try? c.decode(Bool.self) { self = .bool(b); return }
        if let i = try? c.decode(Int.self) { self = .number(Double(i)); return }
        if let d = try? c.decode(Double.self) { self = .number(d); return }
        if let s = try? c.decode(String.self) { self = .string(s); return }
        if let a = try? c.decode([MarvisJSONValue].self) { self = .array(a); return }
        if let o = try? c.decode([String: MarvisJSONValue].self) { self = .object(o); return }
        throw DecodingError.dataCorruptedError(in: c, debugDescription: "Unsupported JSON value")
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.singleValueContainer()
        switch self {
        case .null: try c.encodeNil()
        case .bool(let b): try c.encode(b)
        case .number(let d): try c.encode(d)
        case .string(let s): try c.encode(s)
        case .array(let a): try c.encode(a)
        case .object(let o): try c.encode(o)
        }
    }
}

// MARK: - CSM Model Configuration

/// Decoded from the model's config.json.
/// Flat format: backbone params at top level, decoder in `depth_decoder_config`.
public struct CSMModelArgs: Codable, Sendable {
    public let audioNumCodebooks: Int
    public let audioVocabSize: Int
    public let textVocabSize: Int
    public let quantization: [String: MarvisJSONValue]?

    // Backbone (top-level fields)
    public let hiddenSize: Int
    public let numHiddenLayers: Int
    public let intermediateSize: Int
    public let numAttentionHeads: Int
    public let numKeyValueHeads: Int
    public let rmsNormEps: Double
    public let maxPositionEmbeddings: Int?
    public let ropeTheta: Double?
    public let headDim: Int?
    public let ropeScaling: [String: MarvisJSONValue]?
    public let attentionBias: Bool?
    public let mlpBias: Bool?
    public let tieWordEmbeddings: Bool?
    public let backboneFlavor: String?

    // Decoder
    public let depthDecoderConfig: DepthDecoderConfig?
    public let decoderFlavor: String?

    private enum CodingKeys: String, CodingKey {
        case audioNumCodebooks = "audio_num_codebooks"
        case audioVocabSize = "audio_vocab_size"
        case textVocabSize = "text_vocab_size"
        case quantization
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case rmsNormEps = "rms_norm_eps"
        case maxPositionEmbeddings = "max_position_embeddings"
        case ropeTheta = "rope_theta"
        case headDim = "head_dim"
        case ropeScaling = "rope_scaling"
        case attentionBias = "attention_bias"
        case mlpBias = "mlp_bias"
        case tieWordEmbeddings = "tie_word_embeddings"
        case backboneFlavor = "backbone_flavor"
        case depthDecoderConfig = "depth_decoder_config"
        case decoderFlavor = "decoder_flavor"
    }

    /// Build backbone CSMLlamaConfiguration from top-level fields
    public func backboneConfiguration() -> CSMLlamaConfiguration {
        CSMLlamaConfiguration(
            hiddenSize: hiddenSize,
            hiddenLayers: numHiddenLayers,
            intermediateSize: intermediateSize,
            attentionHeads: numAttentionHeads,
            headDimensions: headDim,
            rmsNormEps: Float(rmsNormEps),
            vocabularySize: textVocabSize,
            kvHeads: numKeyValueHeads,
            maxPositionEmbeddings: maxPositionEmbeddings ?? 2048,
            ropeTheta: Float(ropeTheta ?? 500000),
            ropeScaling: convertRopeScaling(ropeScaling),
            tieWordEmbeddings: tieWordEmbeddings ?? false,
            attentionBias: attentionBias ?? false,
            mlpBias: mlpBias ?? false
        )
    }

    /// Build decoder CSMLlamaConfiguration from depth_decoder_config
    public func decoderConfiguration() -> CSMLlamaConfiguration {
        guard let dec = depthDecoderConfig else {
            // Fallback: use top-level params with smaller defaults
            return backboneConfiguration()
        }
        return CSMLlamaConfiguration(
            hiddenSize: dec.hiddenSize,
            hiddenLayers: dec.numHiddenLayers,
            intermediateSize: dec.intermediateSize,
            attentionHeads: dec.numAttentionHeads,
            headDimensions: dec.headDim,
            rmsNormEps: Float(dec.rmsNormEps ?? 1e-5),
            vocabularySize: dec.vocabSize ?? audioVocabSize,
            kvHeads: dec.numKeyValueHeads,
            maxPositionEmbeddings: dec.maxPositionEmbeddings ?? 33,
            ropeTheta: Float(dec.ropeTheta ?? 500000),
            ropeScaling: convertRopeScaling(dec.ropeScaling),
            tieWordEmbeddings: false,
            attentionBias: dec.attentionBias ?? false,
            mlpBias: dec.mlpBias ?? false
        )
    }
}

/// Depth decoder configuration embedded in config.json
public struct DepthDecoderConfig: Codable, Sendable {
    public let hiddenSize: Int
    public let numHiddenLayers: Int
    public let intermediateSize: Int
    public let numAttentionHeads: Int
    public let numKeyValueHeads: Int
    public let rmsNormEps: Double?
    public let maxPositionEmbeddings: Int?
    public let ropeTheta: Double?
    public let vocabSize: Int?
    public let headDim: Int?
    public let ropeScaling: [String: MarvisJSONValue]?
    public let attentionBias: Bool?
    public let mlpBias: Bool?
    public let backboneHiddenSize: Int?

    private enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case rmsNormEps = "rms_norm_eps"
        case maxPositionEmbeddings = "max_position_embeddings"
        case ropeTheta = "rope_theta"
        case vocabSize = "vocab_size"
        case headDim = "head_dim"
        case ropeScaling = "rope_scaling"
        case attentionBias = "attention_bias"
        case mlpBias = "mlp_bias"
        case backboneHiddenSize = "backbone_hidden_size"
    }
}

/// Legacy sub-config for nested format (used in tests)
public struct LlamaSubConfig: Codable, Sendable {
    public let hiddenSize: Int
    public let numHiddenLayers: Int
    public let intermediateSize: Int
    public let numAttentionHeads: Int
    public let numKeyValueHeads: Int
    public let rmsNormEps: Double
    public let maxPositionEmbeddings: Int?
    public let ropeTheta: Double?
    public let vocabSize: Int?
    public let attentionBias: Bool?
    public let mlpBias: Bool?
    public let tieWordEmbeddings: Bool?
    public let headDim: Int?
    public let ropeScaling: [String: MarvisJSONValue]?

    private enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case rmsNormEps = "rms_norm_eps"
        case maxPositionEmbeddings = "max_position_embeddings"
        case ropeTheta = "rope_theta"
        case vocabSize = "vocab_size"
        case attentionBias = "attention_bias"
        case mlpBias = "mlp_bias"
        case tieWordEmbeddings = "tie_word_embeddings"
        case headDim = "head_dim"
        case ropeScaling = "rope_scaling"
    }

    public func toLlamaConfiguration() -> CSMLlamaConfiguration {
        CSMLlamaConfiguration(
            hiddenSize: hiddenSize,
            hiddenLayers: numHiddenLayers,
            intermediateSize: intermediateSize,
            attentionHeads: numAttentionHeads,
            headDimensions: headDim,
            rmsNormEps: Float(rmsNormEps),
            vocabularySize: vocabSize ?? 128256,
            kvHeads: numKeyValueHeads,
            maxPositionEmbeddings: maxPositionEmbeddings ?? 2048,
            ropeTheta: Float(ropeTheta ?? 500000),
            ropeScaling: convertRopeScaling(ropeScaling),
            tieWordEmbeddings: tieWordEmbeddings ?? true,
            attentionBias: attentionBias ?? false,
            mlpBias: mlpBias ?? false
        )
    }
}

// MARK: - CSM Llama Configuration (internal, used to build model)

public struct CSMLlamaConfiguration: Sendable {
    public var hiddenSize: Int
    public var hiddenLayers: Int
    public var intermediateSize: Int
    public var attentionHeads: Int
    public var headDimensions: Int?
    public var rmsNormEps: Float
    public var vocabularySize: Int
    public var kvHeads: Int
    public var maxPositionEmbeddings: Int
    public var ropeTheta: Float = 500000
    public var ropeTraditional: Bool = false
    public var ropeScaling: [String: StringOrNumber]?
    public var tieWordEmbeddings: Bool = true
    public var attentionBias: Bool = false
    public var mlpBias: Bool = false

    public var resolvedHeadDimensions: Int {
        headDimensions ?? (hiddenSize / attentionHeads)
    }

    public init(
        hiddenSize: Int = 2048,
        hiddenLayers: Int = 16,
        intermediateSize: Int = 8192,
        attentionHeads: Int = 32,
        headDimensions: Int? = nil,
        rmsNormEps: Float = 1e-5,
        vocabularySize: Int = 128256,
        kvHeads: Int = 8,
        maxPositionEmbeddings: Int = 2048,
        ropeTheta: Float = 500000,
        ropeTraditional: Bool = false,
        ropeScaling: [String: StringOrNumber]? = nil,
        tieWordEmbeddings: Bool = true,
        attentionBias: Bool = false,
        mlpBias: Bool = false
    ) {
        self.hiddenSize = hiddenSize
        self.hiddenLayers = hiddenLayers
        self.intermediateSize = intermediateSize
        self.attentionHeads = attentionHeads
        self.headDimensions = headDimensions
        self.rmsNormEps = rmsNormEps
        self.vocabularySize = vocabularySize
        self.kvHeads = kvHeads
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.ropeTheta = ropeTheta
        self.ropeTraditional = ropeTraditional
        self.ropeScaling = ropeScaling
        self.tieWordEmbeddings = tieWordEmbeddings
        self.attentionBias = attentionBias
        self.mlpBias = mlpBias
    }
}

// MARK: - Rope Scaling Conversion

func convertRopeScaling(_ dict: [String: MarvisJSONValue]?) -> [String: StringOrNumber]? {
    guard let dict else { return nil }
    var out: [String: StringOrNumber] = [:]
    for (k, v) in dict {
        switch v {
        case .string(let s): out[k] = .string(s)
        case .number(let d): out[k] = .float(Float(d))
        case .bool(let b): out[k] = .float(b ? 1.0 : 0.0)
        case .null, .array, .object: continue
        }
    }
    return out.isEmpty ? nil : out
}

// MARK: - Mimi Codec Configuration (hardcoded for kyutai/moshiko-pytorch-bf16)

public struct MimiConfig {
    public let channels: Int
    public let sampleRate: Double
    public let frameRate: Double
    public let dimension: Int
    public let numCodebooks: Int
    public let codebookSize: Int
    public let codebookDim: Int
    public let seanet: SeanetConfig
    public let transformer: TransformerConfig

    // Aliases used by Mimi init
    public var quantizerDim: Int { codebookDim }
    public var quantizerNQ: Int { numCodebooks }
    public var quantizerBins: Int { codebookSize }

    public static func moshiko(numCodebooks: Int = 32) -> MimiConfig {
        let seanet = SeanetConfig(
            dimension: 512, channels: 1, causal: true,
            nFilters: 64, nResidualLayers: 1, ratios: [8, 6, 5, 4],
            kernelSize: 7, residualKernelSize: 3, lastKernelSize: 3,
            dilationBase: 2, trueSkip: true, compress: 2,
            padMode: .edge
        )
        let transformer = TransformerConfig(
            dModel: 512, numHeads: 8, numLayers: 8,
            causal: true, biasFF: false, biasAttn: false,
            layerScale: 0.01, context: 250, maxPeriod: 10000,
            dimFeedforward: 2048, gating: false
        )
        return MimiConfig(
            channels: 1, sampleRate: 24000, frameRate: 12.5,
            dimension: 512, numCodebooks: numCodebooks,
            codebookSize: 2048, codebookDim: 256,
            seanet: seanet, transformer: transformer
        )
    }
}

public struct SeanetConfig {
    public let dimension: Int
    public let channels: Int
    public let causal: Bool
    public let nFilters: Int
    public let nResidualLayers: Int
    public let ratios: [Int]
    public let kernelSize: Int
    public let residualKernelSize: Int
    public let lastKernelSize: Int
    public let dilationBase: Int
    public let trueSkip: Bool
    public let compress: Int
    public let padMode: PadMode

    public init(
        dimension: Int, channels: Int, causal: Bool,
        nFilters: Int, nResidualLayers: Int, ratios: [Int],
        kernelSize: Int, residualKernelSize: Int, lastKernelSize: Int,
        dilationBase: Int, trueSkip: Bool, compress: Int,
        padMode: PadMode = .constant
    ) {
        self.dimension = dimension
        self.channels = channels
        self.causal = causal
        self.nFilters = nFilters
        self.nResidualLayers = nResidualLayers
        self.ratios = ratios
        self.kernelSize = kernelSize
        self.residualKernelSize = residualKernelSize
        self.lastKernelSize = lastKernelSize
        self.dilationBase = dilationBase
        self.trueSkip = trueSkip
        self.compress = compress
        self.padMode = padMode
    }

    // Aliases used by SeanetCodec
    public var nfilters: Int { nFilters }
    public var nresidualLayers: Int { nResidualLayers }
    public var ksize: Int { kernelSize }
    public var residualKsize: Int { residualKernelSize }
    public var lastKsize: Int { lastKernelSize }
}

// MARK: - Mimi Transformer Configuration

/// Configuration for the Mimi internal transformer.
public struct MimiTransformerConfig {
    public let dModel: Int
    public let numHeads: Int
    public let numLayers: Int
    public let causal: Bool
    public let biasFF: Bool
    public let biasAttn: Bool
    public let layerScale: Float?
    public let context: Int
    public let maxPeriod: Int
    public let dimFeedforward: Int
    public let gating: Bool

    // Defaults for Mimi (not typically overridden)
    public var kvRepeat: Int = 1
    public var positionalEmbedding: String = "rope"
    public var useConvBlock: Bool = false
    public var crossAttention: Bool = false
    public var norm: String = "layer_norm"
    public var convLayout: Bool = true

    public var headDim: Int { dModel / numHeads }
}

public typealias TransformerConfig = MimiTransformerConfig
