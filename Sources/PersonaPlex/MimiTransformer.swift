import Foundation
import MLX
import MLXFast
import MLXNN

// MARK: - LayerScale

public final class LayerScale: Module {
    public var scale: MLXArray
    public init(dim: Int) {
        self.scale = MLXArray.ones([dim])
    }

    public func callAsFunction(_ xs: MLXArray) -> MLXArray {
        xs * scale
    }
}

// MARK: - Mimi Attention

public final class MimiAttention: Module {
    private let cfg: MimiTransformerConfig
    @ModuleInfo public var in_proj: Linear
    @ModuleInfo public var out_proj: Linear
    @ModuleInfo public var rope: RoPE?

    private let scale: Float

    public init(cfg: MimiTransformerConfig) {
        self.cfg = cfg
        precondition(cfg.kvRepeat == 1, "only kv_repeat == 1 is supported")

        let numKV = cfg.numHeads / cfg.kvRepeat
        let outDim = cfg.dModel + 2 * numKV * (cfg.dModel / cfg.numHeads)
        self._in_proj = ModuleInfo(wrappedValue: Linear(cfg.dModel, outDim, bias: cfg.biasAttn))
        self._out_proj = ModuleInfo(wrappedValue: Linear(cfg.dModel, cfg.dModel, bias: cfg.biasAttn))
        self.scale = 1.0 / Float(Double(cfg.headDim).squareRoot())

        if cfg.positionalEmbedding == "rope" {
            self._rope = ModuleInfo(wrappedValue: RoPE(
                dimensions: cfg.headDim, traditional: true, base: Float(cfg.maxPeriod)))
        } else {
            self._rope = ModuleInfo(wrappedValue: nil)
        }
    }

    public func callAsFunction(_ xs: MLXArray, cache: KVCache) -> MLXArray {
        let b = xs.shape[0]
        let t = xs.shape[1]
        let hd = xs.shape[2]

        let qkv = in_proj(xs).reshaped([b, t, 3, cfg.numHeads, cfg.headDim])

        var q = swappedAxes(qkv[0..<b, 0..<t, 0, 0..<cfg.numHeads, 0..<cfg.headDim], 1, 2)
        var k = swappedAxes(qkv[0..<b, 0..<t, 1, 0..<cfg.numHeads, 0..<cfg.headDim], 1, 2)
        var v = swappedAxes(qkv[0..<b, 0..<t, 2, 0..<cfg.numHeads, 0..<cfg.headDim], 1, 2)

        if let rope {
            q = rope(q, offset: cache.offset)
            k = rope(k, offset: cache.offset)
        }

        (k, v) = cache.update(keys: k, values: v)

        let kLen = k.shape[2]
        let kTargetLen = t + min(cfg.context, kLen - t)
        if kTargetLen < kLen {
            let start = kLen - kTargetLen
            k = split(k, indices: [start], axis: 2)[1]
            v = split(v, indices: [start], axis: 2)[1]
        }

        let actualKVLen = k.shape[2]
        let maskMode: MLXFast.ScaledDotProductAttentionMaskMode
        if !cfg.causal || t <= 1 {
            maskMode = .none
        } else {
            let causal = MLXArray.tri(t, m: actualKVLen, k: actualKVLen - t, type: Float.self) * 1e9 - 1e9
            maskMode = .array(causal.reshaped([1, 1, t, actualKVLen]).asType(q.dtype))
        }

        var out = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: maskMode)
        out = swappedAxes(out, 1, 2).reshaped([b, t, hd])
        return out_proj(out)
    }
}

// MARK: - MLP (Gated)

public final class MimiMlpGating: Module {
    @ModuleInfo public var linear_in: Linear
    @ModuleInfo public var linear_out: Linear

    public init(cfg: MimiTransformerConfig) {
        var hidden = 2 * cfg.dimFeedforward / 3
        if cfg.dimFeedforward == 4 * cfg.dModel {
            hidden = 11 * cfg.dModel / 4
        }
        self._linear_in = ModuleInfo(wrappedValue: Linear(cfg.dModel, 2 * hidden, bias: cfg.biasFF))
        self._linear_out = ModuleInfo(wrappedValue: Linear(hidden, cfg.dModel, bias: cfg.biasFF))
    }

    public func callAsFunction(_ xs: MLXArray) -> MLXArray {
        let b = xs.shape[0], t = xs.shape[1]
        let doubled = linear_in(xs)
        let hidden = doubled.shape[2] / 2
        let split2 = doubled.reshaped([b, t, 2, hidden])
        let parts = split(split2, indices: [1], axis: 2)
        let a = parts[0]
        let bpart = parts[1]
        let gated = silu(a) * bpart
        let flat = gated.reshaped([b, t, hidden])
        return linear_out(flat)
    }
}

public final class MimiMlpNoGating: Module {
    @ModuleInfo public var linear1: Linear
    @ModuleInfo public var linear2: Linear

    public init(cfg: MimiTransformerConfig) {
        self._linear1 = ModuleInfo(wrappedValue: Linear(cfg.dModel, cfg.dimFeedforward, bias: cfg.biasFF))
        self._linear2 = ModuleInfo(wrappedValue: Linear(cfg.dimFeedforward, cfg.dModel, bias: cfg.biasFF))
    }

    public func callAsFunction(_ xs: MLXArray) -> MLXArray {
        linear2(gelu(linear1(xs)))
    }
}

// MARK: - Transformer Layer

public final class MimiTransformerLayer: Module {
    @ModuleInfo public var gating: Module
    @ModuleInfo public var norm1: Module
    @ModuleInfo public var norm2: Module
    @ModuleInfo public var layer_scale_1: Module
    @ModuleInfo public var layer_scale_2: Module
    @ModuleInfo public var self_attn: MimiAttention

    private let useGating: Bool
    private let useLayerScale: Bool

    public init(cfg: MimiTransformerConfig) {
        self.useGating = cfg.gating
        self.useLayerScale = cfg.layerScale != nil

        if cfg.gating {
            self._gating = ModuleInfo(wrappedValue: MimiMlpGating(cfg: cfg))
        } else {
            self._gating = ModuleInfo(wrappedValue: MimiMlpNoGating(cfg: cfg))
        }

        switch cfg.norm {
        case "layer_norm":
            self._norm1 = ModuleInfo(wrappedValue: LayerNorm(dimensions: cfg.dModel, eps: 1e-5))
            self._norm2 = ModuleInfo(wrappedValue: LayerNorm(dimensions: cfg.dModel, eps: 1e-5))
        default:
            self._norm1 = ModuleInfo(wrappedValue: RMSNorm(dimensions: cfg.dModel, eps: 1e-8))
            self._norm2 = ModuleInfo(wrappedValue: RMSNorm(dimensions: cfg.dModel, eps: 1e-8))
        }

        if cfg.layerScale != nil {
            self._layer_scale_1 = ModuleInfo(wrappedValue: LayerScale(dim: cfg.dModel))
            self._layer_scale_2 = ModuleInfo(wrappedValue: LayerScale(dim: cfg.dModel))
        } else {
            self._layer_scale_1 = ModuleInfo(wrappedValue: Identity())
            self._layer_scale_2 = ModuleInfo(wrappedValue: Identity())
        }

        self._self_attn = ModuleInfo(wrappedValue: MimiAttention(cfg: cfg))
    }

    public func callAsFunction(_ xs: MLXArray, cache: KVCache) -> MLXArray {
        var x = xs

        var n1: MLXArray
        if let ln = norm1 as? LayerNorm { n1 = ln(x) }
        else if let rn = norm1 as? RMSNorm { n1 = rn(x) }
        else { n1 = x }

        n1 = self_attn(n1, cache: cache)

        if let ls = layer_scale_1 as? LayerScale { n1 = ls(n1) }
        x = x + n1

        var n2: MLXArray
        if let ln = norm2 as? LayerNorm { n2 = ln(x) }
        else if let rn = norm2 as? RMSNorm { n2 = rn(x) }
        else { n2 = x }

        if let g = gating as? MimiMlpGating { n2 = g(n2) }
        else if let g = gating as? MimiMlpNoGating { n2 = g(n2) }

        if let ls = layer_scale_2 as? LayerScale { n2 = ls(n2) }
        x = x + n2

        return x
    }
}

// MARK: - Transformer

public final class MimiTransformer: Module {
    private let cfg: MimiTransformerConfig
    @ModuleInfo public var layers: [MimiTransformerLayer]

    public init(cfg: MimiTransformerConfig) {
        self.cfg = cfg
        self._layers = ModuleInfo(wrappedValue: (0..<cfg.numLayers).map { _ in MimiTransformerLayer(cfg: cfg) })
    }

    public func callAsFunction(_ xs: MLXArray, cache: [KVCache]) -> MLXArray {
        var x = xs
        for (layer, c) in zip(layers, cache) {
            x = layer(x, cache: c)
        }
        return x
    }

    public func makeCache() -> [KVCacheSimple] {
        (0..<cfg.numLayers).map { _ in KVCacheSimple() }
    }
}

// MARK: - ProjectedTransformer

public final class ProjectedTransformer: Module {
    private let convLayout: Bool
    @ModuleInfo public var transformer: MimiTransformer
    @ModuleInfo public var input_proj: Linear?
    @ModuleInfo public var output_projs: [Linear?]

    public init(cfg: MimiTransformerConfig, inputDim: Int, outputDims: [Int]) {
        self.convLayout = cfg.convLayout
        self._transformer = ModuleInfo(wrappedValue: MimiTransformer(cfg: cfg))

        if inputDim == cfg.dModel {
            self._input_proj = ModuleInfo(wrappedValue: nil)
        } else {
            self._input_proj = ModuleInfo(wrappedValue: Linear(inputDim, cfg.dModel, bias: false))
        }

        var outs: [Linear?] = []
        for od in outputDims {
            outs.append(od == cfg.dModel ? nil : Linear(cfg.dModel, od, bias: false))
        }
        self._output_projs = ModuleInfo(wrappedValue: outs)
    }

    public func callAsFunction(_ xsIn: MLXArray, cache: [KVCache]) -> [MLXArray] {
        var xs = xsIn
        if convLayout { xs = swappedAxes(xs, 1, 2) }

        if let ip = input_proj { xs = ip(xs) }
        xs = transformer(xs, cache: cache)

        if output_projs.compactMap({ $0 }).isEmpty {
            return [convLayout ? swappedAxes(xs, 1, 2) : xs]
        } else {
            var outs: [MLXArray] = []
            for op in output_projs {
                guard let op else { continue }
                var out = op(xs)
                if convLayout { out = swappedAxes(out, 1, 2) }
                outs.append(out)
            }
            return outs
        }
    }

    public func makeCache() -> [KVCacheSimple] { transformer.makeCache() }
}
