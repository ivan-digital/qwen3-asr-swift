import Foundation
import MLX
import MLXFast
import MLXNN

public class Qwen2Attention: Module {
    public let config: Qwen2Configuration
    public let scale: Float

    @ModuleInfo(key: "q_proj") public var wq: Linear
    @ModuleInfo(key: "k_proj") public var wk: Linear
    @ModuleInfo(key: "v_proj") public var wv: Linear
    @ModuleInfo(key: "o_proj") public var wo: Linear

    public let rope: RoPE

    public init(_ config: Qwen2Configuration) {
        self.config = config

        let dim = config.hiddenSize
        let heads = config.attentionHeads
        let kvHeads = config.kvHeads
        let headDim = config.headDim

        self.scale = pow(Float(headDim), -0.5)

        _wq.wrappedValue = Linear(dim, heads * headDim, bias: true)
        _wk.wrappedValue = Linear(dim, kvHeads * headDim, bias: true)
        _wv.wrappedValue = Linear(dim, kvHeads * headDim, bias: true)
        _wo.wrappedValue = Linear(heads * headDim, dim, bias: false)

        self.rope = RoPE(
            dimensions: headDim,
            traditional: config.ropeTraditional,
            base: config.ropeTheta
        )

        super.init()
    }

    public func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode = .causal,
        cache: KVCache? = nil
    ) -> MLXArray {
        let (B, L, _) = (x.dim(0), x.dim(1), x.dim(2))

        var queries = wq(x)
        var keys = wk(x)
        var values = wv(x)

        queries = queries.reshaped(B, L, config.attentionHeads, -1).transposed(0, 2, 1, 3)
        keys = keys.reshaped(B, L, config.kvHeads, -1).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, config.kvHeads, -1).transposed(0, 2, 1, 3)

        if let cache = cache {
            queries = rope(queries, offset: cache.offset)
            keys = rope(keys, offset: cache.offset)
        } else {
            queries = rope(queries)
            keys = rope(keys)
        }

        let effectiveMask: MLXFast.ScaledDotProductAttentionMaskMode
        if let cache = cache {
            effectiveMask = createAttentionMask(h: x, cache: cache)
        } else {
            effectiveMask = mask
        }

        let output = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: scale,
            mask: effectiveMask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return wo(output)
    }
}
