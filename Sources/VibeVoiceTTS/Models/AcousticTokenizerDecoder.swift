import Foundation
import MLX
import MLXNN

public class TokenizerFFN: Module {
    @ModuleInfo(key: "linear1") public var linear1: Linear
    @ModuleInfo(key: "linear2") public var linear2: Linear

    public init(embedDim: Int, ffnDim: Int, bias: Bool = false) {
        _linear1.wrappedValue = Linear(embedDim, ffnDim, bias: bias)
        _linear2.wrappedValue = Linear(ffnDim, embedDim, bias: bias)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = linear1(x)
        out = gelu(out)
        out = linear2(out)
        return out
    }
}

public class Block1D: Module {
    public let dim: Int
    @ModuleInfo(key: "norm") public var norm: ConvRMSNorm
    @ModuleInfo(key: "ffnNorm") public var ffnNorm: ConvRMSNorm
    @ModuleInfo(key: "mixer") public var mixer: SConv1d
    @ModuleInfo(key: "ffn") public var ffn: TokenizerFFN
    public var gamma: MLXArray?
    public var ffnGamma: MLXArray?

    public init(
        dim: Int,
        kernelSize: Int = 7,
        dropPath: Float = 0.0,
        mixerLayer: String = "depthwise_conv",
        layerScaleInitValue: Float = 1e-6,
        causal: Bool = true,
        padMode: String = "constant",
        norm: String = "none",
        bias: Bool = true,
        ffnExpansion: Int = 4,
        eps: Float = 1e-6
    ) {
        self.dim = dim

        _norm.wrappedValue = ConvRMSNorm(dimensions: dim, eps: eps)
        _ffnNorm.wrappedValue = ConvRMSNorm(dimensions: dim, eps: eps)

        let groups = mixerLayer == "depthwise_conv" ? dim : 1
        _mixer.wrappedValue = SConv1d(
            inChannels: dim,
            outChannels: dim,
            kernelSize: kernelSize,
            stride: 1,
            groups: groups,
            bias: bias,
            causal: causal,
            padMode: padMode
        )

        _ffn.wrappedValue = TokenizerFFN(embedDim: dim, ffnDim: ffnExpansion * dim, bias: bias)

        if layerScaleInitValue > 0 {
            self.gamma = MLXArray.ones([dim]) * layerScaleInitValue
            self.ffnGamma = MLXArray.ones([dim]) * layerScaleInitValue
        }

        super.init()
    }

    public func callAsFunction(_ x: MLXArray, cache: StreamingConvCache? = nil, useCache: Bool = false) -> MLXArray {
        var residual = x
        var out = norm(x)
        out = mixer(out, cache: cache, useCache: useCache)
        if let g = gamma {
            let gExpanded = expandedDimensions(expandedDimensions(g, axis: 0), axis: -1)
            out = out * gExpanded
        }
        out = residual + out

        residual = out
        out = ffnNorm.forwardToNLC(out)
        out = ffn(out)
        out = out.transposed(0, 2, 1)
        if let g = ffnGamma {
            let gExpanded = expandedDimensions(expandedDimensions(g, axis: 0), axis: -1)
            out = out * gExpanded
        }
        out = residual + out

        return out
    }
}

public protocol UpsampleLayer: Module {
    func callAsFunction(_ x: MLXArray, cache: StreamingConvCache?, useCache: Bool) -> MLXArray
}

extension SConv1d: UpsampleLayer {}
extension SConvTranspose1d: UpsampleLayer {}

public class TokenizerDecoder: Module {
    public let dimension: Int
    public let channels: Int
    public let nFilters: Int
    public let ratios: [Int]
    public let depths: [Int]
    public let causal: Bool

    public let upsampleLayers: [any UpsampleLayer]

    @ModuleInfo(key: "stages") public var blocks: [Block1D]

    public let stageOffsets: [Int]

    public let normFinal: Module
    @ModuleInfo(key: "head") public var head: SConv1d

    public init(_ config: AcousticTokenizerConfiguration) {
        self.dimension = config.vaeDim
        self.channels = config.channels
        self.nFilters = config.decoderNFilters
        self.ratios = config.decoderRatios
        self.depths = config.decoderDepthsArray
        self.causal = config.causal

        let numStages = depths.count

        var upsampleLayersList: [any UpsampleLayer] = []
        let stem = SConv1d(
            inChannels: dimension,
            outChannels: nFilters * Int(pow(2.0, Double(numStages - 1))),
            kernelSize: 7,
            bias: config.convBias,
            causal: causal,
            padMode: config.padMode
        )
        upsampleLayersList.append(stem)

        for i in 0..<ratios.count {
            let inCh = nFilters * Int(pow(2.0, Double(numStages - 1 - i)))
            let outCh = nFilters * Int(pow(2.0, Double(numStages - 1 - i - 1)))
            let upsample = SConvTranspose1d(
                inChannels: inCh,
                outChannels: outCh,
                kernelSize: ratios[i] * 2,
                stride: ratios[i],
                bias: config.convBias,
                causal: causal,
                trimRightRatio: 1.0
            )
            upsampleLayersList.append(upsample)
        }

        var blocksList: [Block1D] = []
        var offsets: [Int] = []
        for i in 0..<numStages {
            offsets.append(blocksList.count)
            let inCh = nFilters * Int(pow(2.0, Double(numStages - 1 - i)))
            for _ in 0..<depths[i] {
                blocksList.append(Block1D(
                    dim: inCh,
                    kernelSize: 7,
                    mixerLayer: config.mixerLayer,
                    layerScaleInitValue: config.layerScaleInitValue,
                    causal: causal,
                    padMode: config.padMode,
                    bias: config.convBias,
                    eps: config.layernormEps
                ))
            }
        }
        _blocks.wrappedValue = blocksList
        self.stageOffsets = offsets

        let lastCh = nFilters * Int(pow(2.0, Double(numStages - 1 - (numStages - 1))))
        if config.disableLastNorm {
            self.normFinal = Identity()
        } else {
            self.normFinal = ConvRMSNorm(dimensions: lastCh, eps: config.layernormEps)
        }

        _head.wrappedValue = SConv1d(
            inChannels: lastCh,
            outChannels: channels,
            kernelSize: 7,
            bias: config.convBias,
            causal: causal,
            padMode: config.padMode
        )

        self.upsampleLayers = upsampleLayersList

        super.init()
    }

    public func callAsFunction(_ x: MLXArray, cache: StreamingConvCache? = nil, useCache: Bool = false) -> MLXArray {
        var out = x

        for i in 0..<depths.count {
            out = upsampleLayers[i].callAsFunction(out, cache: cache, useCache: useCache)

            let startIdx = stageOffsets[i]
            let endIdx = (i + 1 < stageOffsets.count) ? stageOffsets[i + 1] : blocks.count
            for blockIdx in startIdx..<endIdx {
                out = blocks[blockIdx](out, cache: cache, useCache: useCache)
            }
        }

        if let convNorm = normFinal as? ConvRMSNorm {
            out = convNorm(out)
        }

        out = head(out, cache: cache, useCache: useCache)

        return out
    }
}

public class Identity: Module {
    public override init() {
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        x
    }
}

public class VibeVoiceAcousticTokenizer: Module {
    public let config: AcousticTokenizerConfiguration
    @ModuleInfo(key: "encoder") public var encoder: TokenizerEncoder
    @ModuleInfo(key: "decoder") public var decoder: TokenizerDecoder
    public let fixStd: Float

    public init(_ config: AcousticTokenizerConfiguration) {
        self.config = config
        self.fixStd = config.fixStd
        _encoder.wrappedValue = TokenizerEncoder(config)
        _decoder.wrappedValue = TokenizerDecoder(config)
        super.init()
    }

    public func encode(
        _ audio: MLXArray,
        cache: StreamingConvCache? = nil,
        useCache: Bool = false
    ) -> MLXArray {
        var x = audio

        if x.ndim == 2 {
            x = expandedDimensions(x, axis: 1)
        }

        let latents = encoder(x, cache: cache, useCache: useCache)

        return latents.transposed(0, 2, 1)
    }

    public func sample(
        _ latents: MLXArray,
        distType: String = "gaussian"
    ) -> MLXArray {
        if distType == "none" {
            return latents
        }

        let noise = MLXRandom.normal(latents.shape, dtype: latents.dtype)
        return latents + fixStd * noise
    }

    public func decode(
        _ latents: MLXArray,
        cache: StreamingConvCache? = nil,
        useCache: Bool = false
    ) -> MLXArray {
        var x = latents

        if x.dim(1) != config.vaeDim {
            x = x.transposed(0, 2, 1)
        }

        return decoder(x, cache: cache, useCache: useCache)
    }
}
