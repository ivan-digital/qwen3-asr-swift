import Foundation
import MLX
import MLXNN
import MLXFast
import MLXCommon

/// Pre-norm wav2vec2 transformer encoder layer.
///
/// Layout matches fairseq2 `StandardTransformerEncoderLayer` with pre-norm
/// order and bias-everywhere linears. Self-attention uses 4 quantised
/// projections (q, k, v, out) plus a separate layer norm; the FFN is a
/// vanilla 2-linear stack with GELU activation.
public class Wav2Vec2EncoderLayer: Module {
    @ModuleInfo(key: "self_attn") public var selfAttn: Wav2Vec2SelfAttention
    @ModuleInfo(key: "self_attn_layer_norm") public var selfAttnLayerNorm: LayerNorm
    @ModuleInfo public var ffn: Wav2Vec2FFN
    @ModuleInfo(key: "ffn_layer_norm") public var ffnLayerNorm: LayerNorm

    public init(config: OmnilingualMLXConfig) {
        self._selfAttn.wrappedValue = Wav2Vec2SelfAttention(config: config)
        self._selfAttnLayerNorm.wrappedValue = LayerNorm(
            dimensions: config.modelDim, eps: config.layerNormEps)
        self._ffn.wrappedValue = Wav2Vec2FFN(config: config)
        self._ffnLayerNorm.wrappedValue = LayerNorm(
            dimensions: config.modelDim, eps: config.layerNormEps)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x + selfAttn(selfAttnLayerNorm(x))
        h = h + ffn(ffnLayerNorm(h))
        return h
    }
}

/// Multi-head self-attention with quantised q/k/v/output projections.
public class Wav2Vec2SelfAttention: Module {
    @ModuleInfo(key: "q_proj") public var qProj: QuantizedLinear
    @ModuleInfo(key: "k_proj") public var kProj: QuantizedLinear
    @ModuleInfo(key: "v_proj") public var vProj: QuantizedLinear
    @ModuleInfo(key: "output_proj") public var outputProj: QuantizedLinear

    public let numHeads: Int
    public let headDim: Int
    public let scale: Float

    public init(config: OmnilingualMLXConfig) {
        self.numHeads = config.numHeads
        self.headDim = config.headDim
        self.scale = 1.0 / sqrt(Float(config.headDim))

        let dim = config.modelDim
        let g = config.groupSize
        let b = config.bits

        self._qProj.wrappedValue = QuantizedLinear(dim, dim, bias: true, groupSize: g, bits: b)
        self._kProj.wrappedValue = QuantizedLinear(dim, dim, bias: true, groupSize: g, bits: b)
        self._vProj.wrappedValue = QuantizedLinear(dim, dim, bias: true, groupSize: g, bits: b)
        self._outputProj.wrappedValue = QuantizedLinear(dim, dim, bias: true, groupSize: g, bits: b)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let q = qProj(x)
        let k = kProj(x)
        let v = vProj(x)
        let merged = SDPA.multiHead(
            q: q, k: k, v: v,
            numHeads: numHeads, headDim: headDim, scale: scale,
            mask: nil)
        return outputProj(merged)
    }
}

/// Standard 2-linear feed-forward network with GELU. Both projections are
/// quantised in the published weights (bits = 4 or 8, group size = 64).
public class Wav2Vec2FFN: Module {
    @ModuleInfo(key: "inner_proj") public var innerProj: QuantizedLinear
    @ModuleInfo(key: "output_proj") public var outputProj: QuantizedLinear

    public init(config: OmnilingualMLXConfig) {
        let g = config.groupSize
        let b = config.bits
        self._innerProj.wrappedValue = QuantizedLinear(
            config.modelDim, config.ffnDim, bias: true, groupSize: g, bits: b)
        self._outputProj.wrappedValue = QuantizedLinear(
            config.ffnDim, config.modelDim, bias: true, groupSize: g, bits: b)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        return outputProj(gelu(innerProj(x)))
    }
}
