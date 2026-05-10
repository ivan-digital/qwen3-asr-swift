import Foundation
import MLX
import MLXCommon
import MLXFast
import MLXNN
import PersonaPlex   // RMSNormF32, KVCache, makeLinear, applyLinear

// MARK: - Temporal Attention (GQA)

/// Multi-head attention with **Grouped-Query Attention** for Hibiki Zero-3B.
///
/// Differences vs `PersonaPlex.TemporalAttention`:
/// - `in_proj` is `[Q(dim) | K(kvDim) | V(kvDim)]` packed = `[dim + 2*kvDim, dim]`,
///   not `[3*dim, dim]`. We slice by feature offset, not via reshape-to-3.
/// - Q has `numHeads` heads (16); K/V have `numKVHeads` (8). `MLXFast.scaledDotProductAttention`
///   handles the broadcast natively.
/// - RoPE base/layout selected from `cfg.positionalEmbedding` (`rope_concat` for Zero-3B
///   maps to `RoPE(traditional: false)`).
public final class HibikiTemporalAttention: Module {
    private let cfg: HibikiTemporalConfig
    @ModuleInfo public var in_proj: Module
    @ModuleInfo public var out_proj: Module
    @ModuleInfo public var rope: RoPE

    private let scale: Float

    public init(cfg: HibikiTemporalConfig) {
        self.cfg = cfg
        let qkvOut = cfg.dim + 2 * cfg.kvDim   // = 4096 for Zero-3B
        self._in_proj = ModuleInfo(wrappedValue:
            makeLinear(cfg.dim, qkvOut, bias: false, groupSize: cfg.groupSize, bits: cfg.bits))
        self._out_proj = ModuleInfo(wrappedValue:
            makeLinear(cfg.dim, cfg.dim, bias: false, groupSize: cfg.groupSize, bits: cfg.bits))
        self._rope = ModuleInfo(wrappedValue: RoPE(
            dimensions: cfg.headDim,
            traditional: cfg.positionalEmbedding.traditional,
            base: cfg.maxPeriod))
        self.scale = 1.0 / Float(Double(cfg.headDim).squareRoot())
    }

    public func callAsFunction(_ xs: MLXArray, cache: any KVCache, offset: Int) -> MLXArray {
        let b = xs.shape[0]
        let t = xs.shape[1]

        let qkv = applyLinear(in_proj, xs)             // [B, T, dim + 2*kvDim]

        // Slice Q | K | V by feature offset.
        let dim = cfg.dim
        let kvDim = cfg.kvDim
        let qFlat = qkv[0..<b, 0..<t, 0..<dim]                          // [B, T, dim]
        let kFlat = qkv[0..<b, 0..<t, dim..<(dim + kvDim)]              // [B, T, kvDim]
        let vFlat = qkv[0..<b, 0..<t, (dim + kvDim)..<(dim + 2 * kvDim)] // [B, T, kvDim]

        var q = qFlat.reshaped([b, t, cfg.numHeads, cfg.headDim]).transposed(0, 2, 1, 3)
        var k = kFlat.reshaped([b, t, cfg.numKVHeads, cfg.headDim]).transposed(0, 2, 1, 3)
        let v = vFlat.reshaped([b, t, cfg.numKVHeads, cfg.headDim]).transposed(0, 2, 1, 3)

        q = rope(q, offset: offset)
        k = rope(k, offset: offset)

        var (kCached, vCached) = cache.update(keys: k, values: v)

        // Context window limiting (matches PersonaPlex pattern)
        let kLen = kCached.shape[2]
        let kTargetLen = t + min(cfg.context, kLen - t)
        if kTargetLen < kLen {
            let start = kLen - kTargetLen
            kCached = split(kCached, indices: [start], axis: 2)[1]
            vCached = split(vCached, indices: [start], axis: 2)[1]
        }

        let actualKVLen = kCached.shape[2]
        let maskMode: MLXFast.ScaledDotProductAttentionMaskMode
        if t <= 1 {
            maskMode = .none
        } else {
            let causal = MLXArray.tri(t, m: actualKVLen, k: actualKVLen - t, type: Float.self) * 1e9 - 1e9
            maskMode = .array(causal.reshaped([1, 1, t, actualKVLen]).asType(q.dtype))
        }

        // SDPA broadcasts when q has more heads than k/v (GQA).
        let merged = SDPA.attendAndMerge(
            qHeads: q, kHeads: kCached, vHeads: vCached,
            scale: scale, mask: maskMode)
        return applyLinear(out_proj, merged)
    }

    /// Compile-compatible forward step (T=1 autoregressive).
    /// Cache shape is `[B, numKVHeads, T_cache, headDim]`.
    public func forwardStep(
        _ xs: MLXArray, offset: MLXArray,
        cacheK: MLXArray, cacheV: MLXArray
    ) -> (MLXArray, MLXArray, MLXArray) {
        let qkv = applyLinear(in_proj, xs)              // [B, 1, dim + 2*kvDim]
        let dim = cfg.dim
        let kvDim = cfg.kvDim

        // split with constant indices is shapeless-compile-compatible.
        let parts = split(qkv, indices: [dim, dim + kvDim], axis: -1)
        let qFlat = parts[0]   // [B, 1, dim]
        let kFlat = parts[1]   // [B, 1, kvDim]
        let vFlat = parts[2]   // [B, 1, kvDim]

        var q = qFlat.reshaped([-1, 1, cfg.numHeads, cfg.headDim]).transposed(0, 2, 1, 3)
        var k = kFlat.reshaped([-1, 1, cfg.numKVHeads, cfg.headDim]).transposed(0, 2, 1, 3)
        let v = vFlat.reshaped([-1, 1, cfg.numKVHeads, cfg.headDim]).transposed(0, 2, 1, 3)

        q = rope(q, offset: offset)
        k = rope(k, offset: offset)

        let newK = concatenated([cacheK, k], axis: 2)
        let newV = concatenated([cacheV, v], axis: 2)

        let merged = SDPA.attendAndMerge(
            qHeads: q, kHeads: newK, vHeads: newV, scale: scale,
            mask: MLXFast.ScaledDotProductAttentionMaskMode.none)
        return (applyLinear(out_proj, merged), newK, newV)
    }
}

// MARK: - Temporal FFN (SwiGLU)

public final class HibikiTemporalFFN: Module {
    @ModuleInfo public var linear_in: Module
    @ModuleInfo public var linear_out: Module
    let ffnDim: Int

    public init(cfg: HibikiTemporalConfig) {
        self.ffnDim = cfg.intermediateSize
        let ffnDim = cfg.intermediateSize
        self._linear_in = ModuleInfo(wrappedValue:
            makeLinear(cfg.dim, 2 * ffnDim, bias: false, groupSize: cfg.groupSize, bits: cfg.bits))
        self._linear_out = ModuleInfo(wrappedValue:
            makeLinear(ffnDim, cfg.dim, bias: false, groupSize: cfg.groupSize, bits: cfg.bits))
    }

    public func callAsFunction(_ xs: MLXArray) -> MLXArray {
        let doubled = applyLinear(linear_in, xs)
        let split2 = doubled.reshaped([-1, xs.shape[1], 2, ffnDim])
        let gate = split2.take(MLXArray(Int32(0)), axis: 2)
        let value = split2.take(MLXArray(Int32(1)), axis: 2)
        let gated = silu(gate) * value
        return applyLinear(linear_out, gated)
    }
}

// MARK: - Layer

public final class HibikiTemporalTransformerLayer: Module {
    @ModuleInfo public var norm1: RMSNormF32
    @ModuleInfo public var norm2: RMSNormF32
    @ModuleInfo public var self_attn: HibikiTemporalAttention
    @ModuleInfo public var gating: HibikiTemporalFFN

    public init(cfg: HibikiTemporalConfig) {
        self._norm1 = ModuleInfo(wrappedValue: RMSNormF32(dimensions: cfg.dim, eps: cfg.rmsNormEps))
        self._norm2 = ModuleInfo(wrappedValue: RMSNormF32(dimensions: cfg.dim, eps: cfg.rmsNormEps))
        self._self_attn = ModuleInfo(wrappedValue: HibikiTemporalAttention(cfg: cfg))
        self._gating = ModuleInfo(wrappedValue: HibikiTemporalFFN(cfg: cfg))
    }

    public func callAsFunction(_ xs: MLXArray, cache: any KVCache, offset: Int) -> MLXArray {
        var x = xs
        x = x + self_attn(norm1(x), cache: cache, offset: offset)
        x = x + gating(norm2(x))
        return x
    }

    public func forwardStep(
        _ xs: MLXArray, offset: MLXArray,
        cacheK: MLXArray, cacheV: MLXArray
    ) -> (MLXArray, MLXArray, MLXArray) {
        let (attnOut, newK, newV) = self_attn.forwardStep(
            norm1(xs), offset: offset, cacheK: cacheK, cacheV: cacheV)
        var x = xs + attnOut
        x = x + gating(norm2(x))
        return (x, newK, newV)
    }
}

// MARK: - Temporal Transformer

public final class HibikiTemporalTransformer: Module {
    public let cfg: HibikiTemporalConfig

    @ModuleInfo public var layers: [HibikiTemporalTransformerLayer]
    @ModuleInfo public var out_norm: RMSNormF32

    /// Text embedding (vocab + 1 for padding/init).
    @ModuleInfo public var text_emb: Embedding
    /// 32 audio embedding tables: 16 source + 16 target. Each card+1 entries.
    @ModuleInfo public var emb: [Embedding]
    /// Text logit head (textCard outputs, no +1 — special token only in embedding).
    @ModuleInfo public var text_linear: Linear

    public private(set) var cache: [any KVCache]

    /// Compiled per-step function (T=1 autoregressive).
    public private(set) var compiledStep: (([MLXArray]) -> [MLXArray])?

    public init(cfg: HibikiTemporalConfig) {
        self.cfg = cfg

        self._layers = ModuleInfo(wrappedValue:
            (0..<cfg.numLayers).map { _ in HibikiTemporalTransformerLayer(cfg: cfg) })
        self._out_norm = ModuleInfo(wrappedValue: RMSNormF32(dimensions: cfg.dim, eps: cfg.rmsNormEps))

        self._text_emb = ModuleInfo(wrappedValue:
            Embedding(embeddingCount: cfg.textCard + 1, dimensions: cfg.dim))

        var audioEmbs: [Embedding] = []
        for _ in 0..<cfg.numAudioEmbeddings {
            audioEmbs.append(Embedding(embeddingCount: cfg.card + 1, dimensions: cfg.dim))
        }
        self._emb = ModuleInfo(wrappedValue: audioEmbs)

        self._text_linear = ModuleInfo(wrappedValue: Linear(cfg.dim, cfg.textCard, bias: false))

        self.cache = (0..<cfg.numLayers).map { _ in KVCacheSimple() }
    }

    public func resetCache() {
        for c in cache { c.trim(c.offset) }
    }

    /// Forward over one or more steps with computed text + audio token IDs.
    /// - Parameters:
    ///   - textTokens: `[B, T]` token IDs.
    ///   - audioTokens: `[B, numAudioEmbeddings, T]` token IDs (-1 = invalid/masked).
    ///   - offset: RoPE position offset for position 0 of this batch.
    /// - Returns: `(normedHidden [B, T, dim], textLogits [B, T, textCard])`.
    public func forward(
        textTokens: MLXArray,
        audioTokens: MLXArray,
        offset: Int
    ) -> (MLXArray, MLXArray) {
        let b = textTokens.shape[0]
        let t = textTokens.shape[1]

        var hidden = text_emb(textTokens)
        for i in 0..<cfg.numAudioEmbeddings {
            let rawTokens = audioTokens[0..<b, i, 0..<t]
            let isValid = rawTokens .>= MLXArray(Int32(0))
            let safeTokens = MLX.maximum(rawTokens, MLXArray(Int32(0)))
            let embResult = emb[i](safeTokens)
            let mask = isValid.expandedDimensions(axis: -1)
            hidden = hidden + MLX.where(mask, embResult, MLXArray(Float(0)))
        }

        for (layer, c) in zip(layers, cache) {
            hidden = layer(hidden, cache: c, offset: offset)
        }

        let normed = out_norm(hidden)
        let textLogits = text_linear(normed)
        return (normed, textLogits)
    }

    /// Forward through the layer stack with a pre-computed embedding (no token lookup).
    /// Useful for streaming prefill where the embedding sum is computed externally.
    public func forwardEmbedding(_ embedding: MLXArray, offset: Int) {
        var hidden = embedding
        for (layer, c) in zip(layers, cache) {
            hidden = layer(hidden, cache: c, offset: offset)
        }
        eval(hidden)
    }

    /// Batched prefill version of `forwardEmbedding`.
    public func forwardBatchEmbedding(_ embeddings: MLXArray, offset: Int) {
        var hidden = embeddings
        for (layer, c) in zip(layers, cache) {
            hidden = layer(hidden, cache: c, offset: offset)
        }
        eval(hidden)
    }

    // MARK: - Compiled step (T=1)

    public func setupCompilation() {
        let selfRef = self
        let numLayers = cfg.numLayers

        compiledStep = compile(
            inputs: [selfRef], outputs: [selfRef], shapeless: true
        ) { inputs in
            var hidden = inputs[0]
            let offset = inputs[1]

            var outCache: [MLXArray] = []
            for i in 0..<numLayers {
                let cK = inputs[2 + i * 2]
                let cV = inputs[3 + i * 2]
                let (h, newK, newV) = selfRef.layers[i].forwardStep(
                    hidden, offset: offset, cacheK: cK, cacheV: cV)
                hidden = h
                outCache.append(newK)
                outCache.append(newV)
            }

            let normed = selfRef.out_norm(hidden)
            let textLogits = selfRef.text_linear(normed)

            var result = [normed, textLogits]
            result.append(contentsOf: outCache)
            return result
        }
    }

    public func executeStep(hidden: MLXArray, offset: Int) -> (MLXArray, MLXArray) {
        guard let compiled = compiledStep else {
            var h = hidden
            for (layer, c) in zip(layers, cache) {
                h = layer(h, cache: c, offset: offset)
            }
            let normed = out_norm(h)
            let textLogits = text_linear(normed)
            return (normed, textLogits)
        }

        let offsetArr = MLXArray(Int32(offset))
        var flatInputs: [MLXArray] = [hidden, offsetArr]
        for c in cache {
            if let k = c.keysArray, let v = c.valuesArray {
                flatInputs.append(k)
                flatInputs.append(v)
            } else {
                var h = hidden
                for (layer, c2) in zip(layers, cache) {
                    h = layer(h, cache: c2, offset: offset)
                }
                let normed = out_norm(h)
                let textLogits = text_linear(normed)
                return (normed, textLogits)
            }
        }

        let out = compiled(flatInputs)
        for i in 0..<cfg.numLayers {
            cache[i].replaceArrays(keys: out[2 + i * 2], values: out[3 + i * 2])
        }
        return (out[0], out[1])
    }
}
