import Foundation

public struct AcousticTokenizerConfiguration: Codable, Sendable {
    public var channels: Int
    public var vaeDim: Int
    public var encoderNFilters: Int
    public var encoderRatios: [Int]
    public var decoderNFilters: Int
    public var decoderRatios: [Int]
    public var encoderDepths: String
    public var decoderDepths: String?
    public var causal: Bool
    public var convBias: Bool
    public var convNorm: String
    public var padMode: String
    public var layernorm: String
    public var layernormEps: Float
    public var layernormElementwiseAffine: Bool
    public var mixerLayer: String
    public var layerScaleInitValue: Float
    public var disableLastNorm: Bool
    public var fixStd: Float

    enum CodingKeys: String, CodingKey {
        case channels
        case vaeDim = "vae_dim"
        case encoderNFilters = "encoder_n_filters"
        case encoderRatios = "encoder_ratios"
        case decoderNFilters = "decoder_n_filters"
        case decoderRatios = "decoder_ratios"
        case encoderDepths = "encoder_depths"
        case decoderDepths = "decoder_depths"
        case causal
        case convBias = "conv_bias"
        case convNorm = "conv_norm"
        case padMode = "pad_mode"
        case layernorm
        case layernormEps = "layernorm_eps"
        case layernormElementwiseAffine = "layernorm_elementwise_affine"
        case mixerLayer = "mixer_layer"
        case layerScaleInitValue = "layer_scale_init_value"
        case disableLastNorm = "disable_last_norm"
        case fixStd = "fix_std"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        self.channels = try container.decodeIfPresent(Int.self, forKey: .channels) ?? 1
        self.vaeDim = try container.decodeIfPresent(Int.self, forKey: .vaeDim) ?? 64
        self.encoderNFilters = try container.decodeIfPresent(Int.self, forKey: .encoderNFilters) ?? 32
        self.encoderRatios = try container.decodeIfPresent([Int].self, forKey: .encoderRatios) ?? [8, 5, 5, 4, 2, 2]
        self.decoderNFilters = try container.decodeIfPresent(Int.self, forKey: .decoderNFilters) ?? 32
        self.decoderRatios = try container.decodeIfPresent([Int].self, forKey: .decoderRatios) ?? [8, 5, 5, 4, 2, 2]
        self.encoderDepths = try container.decodeIfPresent(String.self, forKey: .encoderDepths) ?? "3-3-3-3-3-3-8"
        self.decoderDepths = try container.decodeIfPresent(String.self, forKey: .decoderDepths)
        self.causal = try container.decodeIfPresent(Bool.self, forKey: .causal) ?? true
        self.convBias = try container.decodeIfPresent(Bool.self, forKey: .convBias) ?? true
        self.convNorm = try container.decodeIfPresent(String.self, forKey: .convNorm) ?? "none"
        self.padMode = try container.decodeIfPresent(String.self, forKey: .padMode) ?? "constant"
        self.layernorm = try container.decodeIfPresent(String.self, forKey: .layernorm) ?? "RMSNorm"
        self.layernormEps = try container.decodeIfPresent(Float.self, forKey: .layernormEps) ?? 1e-5
        self.layernormElementwiseAffine = try container.decodeIfPresent(Bool.self, forKey: .layernormElementwiseAffine) ?? true
        self.mixerLayer = try container.decodeIfPresent(String.self, forKey: .mixerLayer) ?? "depthwise_conv"
        self.layerScaleInitValue = try container.decodeIfPresent(Float.self, forKey: .layerScaleInitValue) ?? 1e-6
        self.disableLastNorm = try container.decodeIfPresent(Bool.self, forKey: .disableLastNorm) ?? true
        self.fixStd = try container.decodeIfPresent(Float.self, forKey: .fixStd) ?? 0.5
    }

    public init(
        channels: Int = 1,
        vaeDim: Int = 64,
        encoderNFilters: Int = 32,
        encoderRatios: [Int] = [8, 5, 5, 4, 2, 2],
        decoderNFilters: Int = 32,
        decoderRatios: [Int] = [8, 5, 5, 4, 2, 2],
        encoderDepths: String = "3-3-3-3-3-3-8",
        decoderDepths: String? = nil,
        causal: Bool = true,
        convBias: Bool = true,
        convNorm: String = "none",
        padMode: String = "constant",
        layernorm: String = "RMSNorm",
        layernormEps: Float = 1e-5,
        layernormElementwiseAffine: Bool = true,
        mixerLayer: String = "depthwise_conv",
        layerScaleInitValue: Float = 1e-6,
        disableLastNorm: Bool = true,
        fixStd: Float = 0.5
    ) {
        self.channels = channels
        self.vaeDim = vaeDim
        self.encoderNFilters = encoderNFilters
        self.encoderRatios = encoderRatios
        self.decoderNFilters = decoderNFilters
        self.decoderRatios = decoderRatios
        self.encoderDepths = encoderDepths
        self.decoderDepths = decoderDepths
        self.causal = causal
        self.convBias = convBias
        self.convNorm = convNorm
        self.padMode = padMode
        self.layernorm = layernorm
        self.layernormEps = layernormEps
        self.layernormElementwiseAffine = layernormElementwiseAffine
        self.mixerLayer = mixerLayer
        self.layerScaleInitValue = layerScaleInitValue
        self.disableLastNorm = disableLastNorm
        self.fixStd = fixStd
    }

    public var encoderDepthsArray: [Int] {
        encoderDepths.split(separator: "-").compactMap { Int($0) }
    }

    public var decoderDepthsArray: [Int] {
        if let decoderDepths = decoderDepths {
            return decoderDepths.split(separator: "-").compactMap { Int($0) }
        }
        return encoderDepthsArray.reversed()
    }

    public var hopLength: Int {
        decoderRatios.reduce(1, *)
    }
}
