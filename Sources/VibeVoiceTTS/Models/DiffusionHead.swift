import Foundation
import MLX
import MLXNN

public struct DiffusionHeadConfiguration: Codable, Sendable {
    public var hiddenSize: Int
    public var latentSize: Int
    public var headLayers: Int
    public var headFfnRatio: Float
    public var rmsNormEps: Float
    public var ddpmNumSteps: Int
    public var ddpmNumInferenceSteps: Int
    public var ddpmBetaSchedule: BetaSchedule
    public var predictionType: PredictionType

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case latentSize = "latent_size"
        case headLayers = "head_layers"
        case headFfnRatio = "head_ffn_ratio"
        case rmsNormEps = "rms_norm_eps"
        case ddpmNumSteps = "ddpm_num_steps"
        case ddpmNumInferenceSteps = "ddpm_num_inference_steps"
        case ddpmBetaSchedule = "ddpm_beta_schedule"
        case predictionType = "prediction_type"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 896
        self.latentSize = try container.decodeIfPresent(Int.self, forKey: .latentSize) ?? 64
        self.headLayers = try container.decodeIfPresent(Int.self, forKey: .headLayers) ?? 4
        self.headFfnRatio = try container.decodeIfPresent(Float.self, forKey: .headFfnRatio) ?? 3.0
        self.rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-5
        self.ddpmNumSteps = try container.decodeIfPresent(Int.self, forKey: .ddpmNumSteps) ?? 1000
        self.ddpmNumInferenceSteps = try container.decodeIfPresent(Int.self, forKey: .ddpmNumInferenceSteps) ?? 20
        self.ddpmBetaSchedule = try container.decodeIfPresent(BetaSchedule.self, forKey: .ddpmBetaSchedule) ?? .cosine
        self.predictionType = try container.decodeIfPresent(PredictionType.self, forKey: .predictionType) ?? .vPrediction
    }

    public init(
        hiddenSize: Int = 896,
        latentSize: Int = 64,
        headLayers: Int = 4,
        headFfnRatio: Float = 3.0,
        rmsNormEps: Float = 1e-5,
        ddpmNumSteps: Int = 1000,
        ddpmNumInferenceSteps: Int = 20,
        ddpmBetaSchedule: BetaSchedule = .cosine,
        predictionType: PredictionType = .vPrediction
    ) {
        self.hiddenSize = hiddenSize
        self.latentSize = latentSize
        self.headLayers = headLayers
        self.headFfnRatio = headFfnRatio
        self.rmsNormEps = rmsNormEps
        self.ddpmNumSteps = ddpmNumSteps
        self.ddpmNumInferenceSteps = ddpmNumInferenceSteps
        self.ddpmBetaSchedule = ddpmBetaSchedule
        self.predictionType = predictionType
    }
}

public class DiffusionFFN: Module {
    @ModuleInfo(key: "gate_proj") public var gate: Linear
    @ModuleInfo(key: "up_proj") public var up: Linear
    @ModuleInfo(key: "down_proj") public var down: Linear

    public init(embedDim: Int, ffnDim: Int) {
        _gate.wrappedValue = Linear(embedDim, ffnDim, bias: false)
        _up.wrappedValue = Linear(embedDim, ffnDim, bias: false)
        _down.wrappedValue = Linear(ffnDim, embedDim, bias: false)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(silu(gate(x)) * up(x))
    }
}

public class HeadLayer: Module {
    public let embedDim: Int
    public let condDim: Int
    public let ffnDim: Int

    @ModuleInfo public var ffn: DiffusionFFN
    @ModuleInfo public var norm: RMSNorm

    @ModuleInfo(key: "adaLN_modulation") public var adaLNModulation: AdaLNModulation

    public class AdaLNModulation: Module {
        @ModuleInfo public var linear: Linear

        public init(condDim: Int, embedDim: Int) {
            _linear.wrappedValue = Linear(condDim, 3 * embedDim, bias: false)
            super.init()
        }

        public func callAsFunction(_ c: MLXArray) -> MLXArray {
            linear(silu(c))
        }
    }

    public init(embedDim: Int, ffnDim: Int, condDim: Int, normEps: Float = 1e-5) {
        self.embedDim = embedDim
        self.condDim = condDim
        self.ffnDim = ffnDim

        _ffn.wrappedValue = DiffusionFFN(embedDim: embedDim, ffnDim: ffnDim)
        _norm.wrappedValue = RMSNorm(dimensions: embedDim, eps: normEps)
        _adaLNModulation.wrappedValue = AdaLNModulation(condDim: condDim, embedDim: embedDim)

        super.init()
    }

    public func callAsFunction(_ x: MLXArray, c: MLXArray) -> MLXArray {
        let modulation = adaLNModulation(c)
        let chunks = split(modulation, parts: 3, axis: -1)
        let shiftFFN = chunks[0]
        let scaleFFN = chunks[1]
        let gateFFN = chunks[2]

        let normed = norm(x)
        let modulated = modulate(normed, shift: shiftFFN, scale: scaleFFN)
        let ffnOut = ffn(modulated)

        return x + gateFFN * ffnOut
    }
}

public class FinalLayer: Module {
    public let normFinal: RMSNorm
    @ModuleInfo public var linear: Linear

    @ModuleInfo(key: "adaLN_modulation") public var adaLNModulation: FinalAdaLNModulation

    public class FinalAdaLNModulation: Module {
        @ModuleInfo public var linear: Linear

        public init(condSize: Int, hiddenSize: Int) {
            _linear.wrappedValue = Linear(condSize, 2 * hiddenSize, bias: false)
            super.init()
        }

        public func callAsFunction(_ c: MLXArray) -> MLXArray {
            linear(silu(c))
        }
    }

    public init(hiddenSize: Int, outputSize: Int, condSize: Int, normEps: Float = 1e-5) {
        self.normFinal = RMSNorm(dimensions: hiddenSize, eps: normEps)
        _linear.wrappedValue = Linear(hiddenSize, outputSize, bias: false)
        _adaLNModulation.wrappedValue = FinalAdaLNModulation(condSize: condSize, hiddenSize: hiddenSize)

        super.init()
    }

    public func callAsFunction(_ x: MLXArray, c: MLXArray) -> MLXArray {
        let modulation = adaLNModulation(c)
        let chunks = split(modulation, parts: 2, axis: -1)
        let shift = chunks[0]
        let scale = chunks[1]

        var out = modulate(normFinal(x), shift: shift, scale: scale)
        out = linear(out)
        return out
    }
}

public class VibeVoiceDiffusionHead: Module {
    public let config: DiffusionHeadConfiguration
    public let condDim: Int

    @ModuleInfo(key: "noisy_images_proj") public var noisyImagesProj: Linear
    @ModuleInfo(key: "cond_proj") public var condProj: Linear
    @ModuleInfo(key: "t_embedder") public var tEmbedder: TimestepEmbedder
    @ModuleInfo public var layers: [HeadLayer]
    @ModuleInfo(key: "final_layer") public var finalLayer: FinalLayer

    public init(_ config: DiffusionHeadConfiguration) {
        self.config = config
        self.condDim = config.hiddenSize

        _noisyImagesProj.wrappedValue = Linear(config.latentSize, config.hiddenSize, bias: false)
        _condProj.wrappedValue = Linear(config.hiddenSize, condDim, bias: false)
        _tEmbedder.wrappedValue = TimestepEmbedder(hiddenSize: condDim)

        let ffnDim = Int(Float(config.hiddenSize) * config.headFfnRatio)

        var layersList: [HeadLayer] = []
        for _ in 0..<config.headLayers {
            layersList.append(HeadLayer(
                embedDim: config.hiddenSize,
                ffnDim: ffnDim,
                condDim: condDim,
                normEps: config.rmsNormEps
            ))
        }
        _layers.wrappedValue = layersList

        _finalLayer.wrappedValue = FinalLayer(
            hiddenSize: config.hiddenSize,
            outputSize: config.latentSize,
            condSize: condDim,
            normEps: config.rmsNormEps
        )

        super.init()
    }

    public func callAsFunction(
        noisyImages: MLXArray,
        timesteps: MLXArray,
        condition: MLXArray
    ) -> MLXArray {
        var x = noisyImagesProj(noisyImages)
        let t = tEmbedder(timesteps)
        let cond = condProj(condition)

        let c = cond + t

        for layer in layers {
            x = layer(x, c: c)
        }

        x = finalLayer(x, c: c)
        return x
    }
}
