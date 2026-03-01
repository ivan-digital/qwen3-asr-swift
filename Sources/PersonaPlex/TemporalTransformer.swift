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
            makeLinear(cfg.dim, totalDim, bias: false, groupSize: cfg.groupSize, bits: cfg.bits))
        self._out_proj = ModuleInfo(wrappedValue:
            makeLinear(cfg.dim, cfg.dim, bias: false, groupSize: cfg.groupSize, bits: cfg.bits))
        self._rope = ModuleInfo(wrappedValue: RoPE(
            dimensions: cfg.headDim, traditional: true, base: Float(cfg.maxPeriod)))
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

    /// Compile-compatible forward: takes explicit cache arrays + MLXArray offset.
    /// For T=1 autoregressive steps only (no causal mask needed, no context limiting).
    /// Returns (output, newK, newV) where newK/newV include the new entry.
    public func forwardStep(
        _ xs: MLXArray, offset: MLXArray,
        cacheK: MLXArray, cacheV: MLXArray
    ) -> (MLXArray, MLXArray, MLXArray) {
        let qkv = applyLinear(in_proj, xs)
        // T=1: [B, 1, 3*dim] → [B, 1, 3, H, D]
        let qkvR = qkv.reshaped([-1, 1, 3, cfg.numHeads, cfg.headDim])
        // Use take (Gather) instead of split/slice — compile(shapeless:true) compatible.
        // take with scalar index removes the axis: [B,1,3,H,D] → [B,1,H,D]
        var q = qkvR.take(MLXArray(Int32(0)), axis: 2).transposed(0, 2, 1, 3) // [B,H,1,D]
        var k = qkvR.take(MLXArray(Int32(1)), axis: 2).transposed(0, 2, 1, 3)
        let v = qkvR.take(MLXArray(Int32(2)), axis: 2).transposed(0, 2, 1, 3)

        q = rope(q, offset: offset)
        k = rope(k, offset: offset)

        // Concatenate with cache
        let newK = concatenated([cacheK, k], axis: 2)
        let newV = concatenated([cacheV, v], axis: 2)

        // T=1 autoregressive: no causal mask needed
        var out = MLXFast.scaledDotProductAttention(
            queries: q, keys: newK, values: newV, scale: scale, mask: .none)
        out = out.transposed(0, 2, 1, 3).reshaped([-1, 1, cfg.dim])
        return (applyLinear(out_proj, out), newK, newV)
    }
}

// MARK: - Temporal FFN (SiLU-gated / SwiGLU)

public final class TemporalFFN: Module {
    @ModuleInfo public var linear_in: Module   // QuantizedLinear: dim -> 2 * intermediateSize
    @ModuleInfo public var linear_out: Module   // QuantizedLinear: intermediateSize -> dim
    let ffnDim: Int

    public init(cfg: TemporalTransformerConfig) {
        self.ffnDim = cfg.intermediateSize
        let ffnDim = cfg.intermediateSize
        self._linear_in = ModuleInfo(wrappedValue:
            makeLinear(cfg.dim, 2 * ffnDim, bias: false, groupSize: cfg.groupSize, bits: cfg.bits))
        self._linear_out = ModuleInfo(wrappedValue:
            makeLinear(ffnDim, cfg.dim, bias: false, groupSize: cfg.groupSize, bits: cfg.bits))
    }

    public func callAsFunction(_ xs: MLXArray) -> MLXArray {
        let doubled = applyLinear(linear_in, xs)
        // [B, T, 2*ffnDim] → [B, T, 2, ffnDim]
        let split2 = doubled.reshaped([-1, xs.shape[1], 2, ffnDim])
        // Use take (Gather) instead of split — compile(shapeless:true) compatible.
        // take with scalar index removes axis: [B,T,2,ffnDim] → [B,T,ffnDim]
        let gate = split2.take(MLXArray(Int32(0)), axis: 2)
        let value = split2.take(MLXArray(Int32(1)), axis: 2)
        let gated = silu(gate) * value
        return applyLinear(linear_out, gated)
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

    /// Compile-compatible forward with explicit cache arrays.
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

    /// Compiled temporal step function for Metal kernel fusion.
    /// Input: [hidden, offset, K0, V0, K1, V1, ..., K31, V31]
    /// Output: [normed, textLogits, K0, V0, ..., K31, V31]
    public private(set) var compiledStep: (([MLXArray]) -> [MLXArray])?

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

    /// Forward pass with pre-computed embedding (for voice prompt replay, single step).
    /// Feeds the embedding through all layers to populate KV caches.
    public func forwardEmbedding(_ embedding: MLXArray, offset: Int) {
        var hidden = embedding  // [B, 1, dim]
        for (layer, c) in zip(layers, cache) {
            hidden = layer(hidden, cache: c, offset: offset)
        }
        eval(hidden)  // force evaluation to populate caches
    }

    /// Batched forward pass with pre-computed embeddings (for voice prompt replay).
    /// Processes all T embeddings in a single pass with causal attention.
    /// - Parameters:
    ///   - embeddings: [B, T, dim] pre-computed embeddings for all voice frames
    ///   - offset: RoPE offset for position 0 of this batch
    public func forwardBatchEmbedding(_ embeddings: MLXArray, offset: Int) {
        var hidden = embeddings  // [B, T, dim]
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

        // Sum all 17 embeddings.
        // Original ScaledEmbedding returns zero vectors for -1 tokens.
        // We clamp -1 → 0 for safe lookup, then zero-mask the result.
        var hidden = text_emb(textTokens)  // [B, T, dim]
        for i in 0..<cfg.numAudioEmbeddings {
            let rawTokens = audioTokens[0..<b, i, 0..<t]        // [B, T]
            let isValid = rawTokens .>= MLXArray(Int32(0))       // [B, T] bool
            let safeTokens = MLX.maximum(rawTokens, MLXArray(Int32(0)))  // clamp -1 → 0
            let embResult = emb[i](safeTokens)                   // [B, T, dim]
            let mask = isValid.expandedDimensions(axis: -1)      // [B, T, 1]
            hidden = hidden + MLX.where(mask, embResult, MLXArray(Float(0)))
        }

        // Pass through transformer layers
        for (layer, c) in zip(layers, cache) {
            hidden = layer(hidden, cache: c, offset: offset)
        }

        let normed = out_norm(hidden)
        let textLogits = text_linear(normed)

        return (normed, textLogits)
    }

    // MARK: - Compiled Step (T=1 autoregressive)

    /// Set up compiled temporal step for Metal kernel fusion.
    /// Compiles the layer stack (no embedding) with explicit cache arrays.
    /// Call after model weights are loaded.
    public func setupCompilation() {
        let selfRef = self
        let numLayers = cfg.numLayers

        compiledStep = compile(
            inputs: [selfRef], outputs: [selfRef], shapeless: true
        ) { inputs in
            var hidden = inputs[0]           // [B, 1, dim] — pre-computed embedding sum
            let offset = inputs[1]           // MLXArray scalar

            // Pass through all layers with explicit cache
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

    /// Execute a single autoregressive step through the compiled temporal transformer.
    /// Manages KVCacheSimple objects, delegating to compiled function when available.
    /// - Parameters:
    ///   - hidden: [B, 1, dim] pre-computed embedding sum
    ///   - offset: RoPE position offset
    /// - Returns: (normedHidden [B, 1, dim], textLogits [B, 1, textCard])
    public func executeStep(hidden: MLXArray, offset: Int) -> (MLXArray, MLXArray) {
        guard let compiled = compiledStep else {
            // Fallback: uncompiled path through layers
            var h = hidden
            for (layer, c) in zip(layers, cache) {
                h = layer(h, cache: c, offset: offset)
            }
            let normed = out_norm(h)
            let textLogits = text_linear(normed)
            return (normed, textLogits)
        }

        // Build flat input array: [hidden, offset, K0, V0, ..., K31, V31]
        let offsetArr = MLXArray(Int32(offset))
        var flatInputs: [MLXArray] = [hidden, offsetArr]
        for c in cache {
            if let k = c.keysArray, let v = c.valuesArray {
                flatInputs.append(k)
                flatInputs.append(v)
            } else {
                // Cache empty — not yet initialized, fall back to uncompiled
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

        // Update KVCacheSimple objects from compiled output
        for i in 0..<cfg.numLayers {
            cache[i].replaceArrays(keys: out[2 + i * 2], values: out[3 + i * 2])
        }

        return (out[0], out[1])
    }
}
