import Foundation
import MLX
import MLXNN

/// Stack of pre-norm wav2vec2 transformer encoder layers followed by a final
/// layer norm. Tensor names match fairseq2 / the published Omnilingual MLX
/// repos: `encoder.layers.{i}.…` and `encoder.layer_norm.{weight,bias}`.
public class Wav2Vec2Encoder: Module {
    @ModuleInfo public var layers: [Wav2Vec2EncoderLayer]
    @ModuleInfo(key: "layer_norm") public var layerNorm: LayerNorm

    public init(config: OmnilingualMLXConfig) {
        self._layers.wrappedValue = (0..<config.numLayers).map { _ in
            Wav2Vec2EncoderLayer(config: config)
        }
        self._layerNorm.wrappedValue = LayerNorm(
            dimensions: config.modelDim, eps: config.layerNormEps)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for layer in layers {
            h = layer(h)
        }
        return layerNorm(h)
    }
}

/// Quantised linear CTC head: `modelDim → vocabSize` (10288 for v2 tokenizer).
public class CTCHead: Module {
    @ModuleInfo public var proj: QuantizedLinear

    public init(config: OmnilingualMLXConfig) {
        self._proj.wrappedValue = QuantizedLinear(
            config.modelDim, config.vocabSize,
            bias: true, groupSize: config.groupSize, bits: config.bits)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        return proj(x)
    }
}
