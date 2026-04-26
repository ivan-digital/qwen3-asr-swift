import Foundation
import MLX
import MLXFast
import MLXNN

public class Qwen2TransformerBlock: Module {
    @ModuleInfo(key: "self_attn") public var attention: Qwen2Attention
    public let mlp: Qwen2MLP

    @ModuleInfo(key: "input_layernorm") public var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") public var postAttentionLayerNorm: RMSNorm

    public init(_ config: Qwen2Configuration) {
        _attention.wrappedValue = Qwen2Attention(config)
        self.mlp = Qwen2MLP(config)
        _inputLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        super.init()
    }

    public func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode = .causal,
        cache: KVCache? = nil
    ) -> MLXArray {
        let normedInput = inputLayerNorm(x)
        let r = attention(normedInput, mask: mask, cache: cache)
        let h = x + r

        let mlpOut = mlp(postAttentionLayerNorm(h))
        return h + mlpOut
    }
}
