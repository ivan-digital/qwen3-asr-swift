import Foundation
import MLX
import MLXFast
import MLXNN

// MARK: - RMSNorm (float32 computation)

public final class RMSNormF32: Module {
    public var weight: MLXArray
    private let eps: Float

    public init(dimensions: Int, eps: Float = 1e-8) {
        self.weight = MLXArray.ones([dimensions])
        self.eps = eps
    }

    public func callAsFunction(_ xs: MLXArray) -> MLXArray {
        let x32 = xs.asType(.float32)
        let ms = (x32 * x32).mean(axis: -1, keepDims: true)
        let normed = x32 * rsqrt(ms + MLXArray(eps))
        return (normed * weight).asType(xs.dtype)
    }
}

// MARK: - Temporal Attention

public final class TemporalAttention: Module {
    private let cfg: TemporalTransformerConfig
    @ModuleInfo public var in_proj: Module    // QuantizedLinear: Q/K/V packed
    @ModuleInfo public var out_proj: Module   // QuantizedLinear for output
    @ModuleInfo public var rope: RoPE

    private let scale: Float

    public init(cfg: TemporalTransformerConfig) {
        self.cfg = cfg
        let totalDim = 3 * cfg.dim  // Q + K + V packed (no GQA, all 32 heads)
        self._in_proj = ModuleInfo(wrappedValue:
            QuantizedLinear(cfg.dim, totalDim, bias: false, groupSize: cfg.groupSize, bits: cfg.bits))
        self._out_proj = ModuleInfo(wrappedValue:
            QuantizedLinear(cfg.dim, cfg.dim, bias: false, groupSize: cfg.groupSize, bits: cfg.bits))
        self._rope = ModuleInfo(wrappedValue: RoPE(
            dimensions: cfg.headDim, traditional: false, base: Float(cfg.maxPeriod)))
        self.scale = 1.0 / Float(Double(cfg.headDim).squareRoot())
    }

    public func callAsFunction(_ xs: MLXArray, cache: KVCacheSimple, offset: Int) -> MLXArray {
        let b = xs.shape[0]
        let t = xs.shape[1]

        let qkv = applyLinear(in_proj, xs)
        let qkvR = qkv.reshaped([b, t, 3, cfg.numHeads, cfg.headDim])

        // [B, T, H, D] -> [B, H, T, D]
        var q = swappedAxes(qkvR[0..<b, 0..<t, 0, 0..<cfg.numHeads, 0..<cfg.headDim], 1, 2)
        var k = swappedAxes(qkvR[0..<b, 0..<t, 1, 0..<cfg.numHeads, 0..<cfg.headDim], 1, 2)
        var v = swappedAxes(qkvR[0..<b, 0..<t, 2, 0..<cfg.numHeads, 0..<cfg.headDim], 1, 2)

        q = rope(q, offset: offset)
        k = rope(k, offset: offset)

        (k, v) = cache.update(keys: k, values: v)

        // Context window limiting
        let kLen = k.shape[2]
        let kTargetLen = t + min(cfg.context, kLen - t)
        if kTargetLen < kLen {
            let start = kLen - kTargetLen
            k = split(k, indices: [start], axis: 2)[1]
            v = split(v, indices: [start], axis: 2)[1]
        }

        let actualKVLen = k.shape[2]
        let maskMode: MLXFast.ScaledDotProductAttentionMaskMode
        if t <= 1 {
            maskMode = .none
        } else {
            let causal = MLXArray.tri(t, m: actualKVLen, k: actualKVLen - t, type: Float.self) * 1e9 - 1e9
            maskMode = .array(causal.reshaped([1, 1, t, actualKVLen]).asType(q.dtype))
        }

        // SDPA returns [B, H, T, D]
        var out = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: maskMode)
        // [B, H, T, D] -> [B, T, H, D] -> [B, T, dim]
        out = swappedAxes(out, 1, 2).reshaped([b, t, cfg.dim])
        return applyLinear(out_proj, out)
    }
}

// MARK: - Temporal FFN (SiLU-gated / SwiGLU)

public final class TemporalFFN: Module {
    @ModuleInfo public var linear_in: Module   // QuantizedLinear: dim -> 2 * intermediateSize
    @ModuleInfo public var linear_out: Module   // QuantizedLinear: intermediateSize -> dim

    public init(cfg: TemporalTransformerConfig) {
        let ffnDim = cfg.intermediateSize
        self._linear_in = ModuleInfo(wrappedValue:
            QuantizedLinear(cfg.dim, 2 * ffnDim, bias: false, groupSize: cfg.groupSize, bits: cfg.bits))
        self._linear_out = ModuleInfo(wrappedValue:
            QuantizedLinear(ffnDim, cfg.dim, bias: false, groupSize: cfg.groupSize, bits: cfg.bits))
    }

    public func callAsFunction(_ xs: MLXArray) -> MLXArray {
        let b = xs.shape[0], t = xs.shape[1]
        let doubled = applyLinear(linear_in, xs)
        let ffnDim = doubled.shape[2] / 2
        let split2 = doubled.reshaped([b, t, 2, ffnDim])
        let parts = split(split2, indices: [1], axis: 2)
        let gate = parts[0]
        let value = parts[1]
        let gated = silu(gate) * value
        let flat = gated.reshaped([b, t, ffnDim])
        return applyLinear(linear_out, flat)
    }
}

// MARK: - Temporal Transformer Layer

public final class TemporalTransformerLayer: Module {
    @ModuleInfo public var norm1: RMSNormF32
    @ModuleInfo public var norm2: RMSNormF32
    @ModuleInfo public var self_attn: TemporalAttention
    @ModuleInfo public var gating: TemporalFFN

    public init(cfg: TemporalTransformerConfig) {
        self._norm1 = ModuleInfo(wrappedValue: RMSNormF32(dimensions: cfg.dim, eps: cfg.rmsNormEps))
        self._norm2 = ModuleInfo(wrappedValue: RMSNormF32(dimensions: cfg.dim, eps: cfg.rmsNormEps))
        self._self_attn = ModuleInfo(wrappedValue: TemporalAttention(cfg: cfg))
        self._gating = ModuleInfo(wrappedValue: TemporalFFN(cfg: cfg))
    }

    public func callAsFunction(_ xs: MLXArray, cache: KVCacheSimple, offset: Int) -> MLXArray {
        var x = xs
        x = x + self_attn(norm1(x), cache: cache, offset: offset)
        x = x + gating(norm2(x))
        return x
    }
}

// MARK: - Temporal Transformer

public final class TemporalTransformer: Module {
    public let cfg: TemporalTransformerConfig

    @ModuleInfo public var layers: [TemporalTransformerLayer]
    @ModuleInfo public var out_norm: RMSNormF32

    // Embeddings: text + 16 audio (8 user + 8 agent)
    @ModuleInfo public var text_emb: Embedding
    @ModuleInfo public var emb: [Embedding]       // 16 audio embeddings

    // Output heads
    @ModuleInfo public var text_linear: Linear     // text logit head

    public private(set) var cache: [KVCacheSimple]

    public init(cfg: TemporalTransformerConfig) {
        self.cfg = cfg

        self._layers = ModuleInfo(wrappedValue:
            (0..<cfg.numLayers).map { _ in TemporalTransformerLayer(cfg: cfg) })
        self._out_norm = ModuleInfo(wrappedValue: RMSNormF32(dimensions: cfg.dim, eps: cfg.rmsNormEps))

        // text_emb: vocab + 1 for padding
        self._text_emb = ModuleInfo(wrappedValue: Embedding(embeddingCount: cfg.textCard + 1, dimensions: cfg.dim))

        // 16 audio embeddings: card + 1 for initial token
        var audioEmbs: [Embedding] = []
        for _ in 0..<cfg.numAudioEmbeddings {
            audioEmbs.append(Embedding(embeddingCount: cfg.card + 1, dimensions: cfg.dim))
        }
        self._emb = ModuleInfo(wrappedValue: audioEmbs)

        // Text output head (textCard outputs, no +1 — special token only in embedding)
        self._text_linear = ModuleInfo(wrappedValue: Linear(cfg.dim, cfg.textCard, bias: false))

        self.cache = (0..<cfg.numLayers).map { _ in KVCacheSimple() }
    }

    public func resetCache() {
        for c in cache { c.trim(c.offset) }
    }

    /// Forward pass with pre-computed embedding (for voice prompt replay).
    /// Feeds the embedding through all layers to populate KV caches.
    public func forwardEmbedding(_ embedding: MLXArray, offset: Int) {
        var hidden = embedding  // [B, 1, dim]
        for (layer, c) in zip(layers, cache) {
            hidden = layer(hidden, cache: c, offset: offset)
        }
        eval(hidden)  // force evaluation to populate caches
    }

    /// Forward pass: takes per-stream token IDs, returns hidden states + text logits
    /// - Parameters:
    ///   - textTokens: [B, T] text token IDs
    ///   - audioTokens: [B, 16, T] audio token IDs (8 user + 8 agent)
    ///   - offset: RoPE offset
    /// - Returns: (hiddenStates [B, T, dim], textLogits [B, T, textCard+1])
    public func forward(
        textTokens: MLXArray,
        audioTokens: MLXArray,
        offset: Int
    ) -> (MLXArray, MLXArray) {
        let b = textTokens.shape[0]
        let t = textTokens.shape[1]

        // Sum all 17 embeddings
        var hidden = text_emb(textTokens)  // [B, T, dim]
        for i in 0..<cfg.numAudioEmbeddings {
            hidden = hidden + emb[i](audioTokens[0..<b, i, 0..<t])
        }

        // Pass through transformer layers
        for (layer, c) in zip(layers, cache) {
            hidden = layer(hidden, cache: c, offset: offset)
        }

        let normed = out_norm(hidden)
        let textLogits = text_linear(normed)

        return (normed, textLogits)
    }
}
