import Foundation
import MLX
import MLXFast
import MLXNN

// MARK: - MultiLinear

/// Stores weights for N steps as a single tensor and performs step-specific matmul.
/// Weight shape: [numSteps * outDim, inDim] (concatenated along rows)
/// At step k: slice weight[k*outDim..<(k+1)*outDim, :] and matmul.
public final class MultiLinear: Module {
    public var weight: MLXArray
    public var bias: MLXArray?
    private let numSteps: Int
    private let outDim: Int

    public init(numSteps: Int, inDim: Int, outDim: Int, bias: Bool = false) {
        self.numSteps = numSteps
        self.outDim = outDim
        let scale: Float = 1.0 / Float(inDim)
        self.weight = MLXRandom.uniform(low: -scale, high: scale, [numSteps * outDim, inDim])
        self.bias = bias ? MLXArray.zeros([numSteps, outDim]) : nil
    }

    public func callAsFunction(_ xs: MLXArray, step: Int) -> MLXArray {
        let start = step * outDim
        let end = start + outDim
        let w = weight[start..<end, 0...]  // [outDim, inDim]
        var result = xs.matmul(w.T)
        if let b = bias {
            result = result + b[step]
        }
        return result
    }
}

// MARK: - Depformer Attention

public final class DepformerAttention: Module {
    private let cfg: DepformerConfig
    @ModuleInfo public var in_proj: MultiLinear   // Projects to Q+K+V packed
    @ModuleInfo public var out_proj: MultiLinear   // Projects output back

    private let scale: Float

    public init(cfg: DepformerConfig) {
        self.cfg = cfg
        let totalDim = 3 * cfg.dim  // Q + K + V packed
        self._in_proj = ModuleInfo(wrappedValue: MultiLinear(
            numSteps: cfg.numSteps, inDim: cfg.dim, outDim: totalDim, bias: false))
        self._out_proj = ModuleInfo(wrappedValue: MultiLinear(
            numSteps: cfg.numSteps, inDim: cfg.dim, outDim: cfg.dim, bias: false))
        self.scale = 1.0 / Float(Double(cfg.headDim).squareRoot())
    }

    public func callAsFunction(
        _ xs: MLXArray, step: Int, cache: KVCacheSimple
    ) -> MLXArray {
        let b = xs.shape[0]
        let t = xs.shape[1]

        let qkv = in_proj(xs, step: step)
        let qkvR = qkv.reshaped([b, t, 3, cfg.numHeads, cfg.headDim])

        let q = swappedAxes(qkvR[0..<b, 0..<t, 0, 0..<cfg.numHeads, 0..<cfg.headDim], 1, 2)
        var k = swappedAxes(qkvR[0..<b, 0..<t, 1, 0..<cfg.numHeads, 0..<cfg.headDim], 1, 2)
        var v = swappedAxes(qkvR[0..<b, 0..<t, 2, 0..<cfg.numHeads, 0..<cfg.headDim], 1, 2)

        // No positional embeddings for depformer (depformer_pos_emb = "none")
        (k, v) = cache.update(keys: k, values: v)

        // Context window limiting
        let kLen = k.shape[2]
        if kLen > cfg.context {
            let start = kLen - cfg.context
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

        var out = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: maskMode)
        out = swappedAxes(out, 1, 2).reshaped([b, t, cfg.dim])
        return out_proj(out, step: step)
    }
}

// MARK: - Depformer FFN (SiLU-gated with MultiLinear)

public final class DepformerFFN: Module {
    private let cfg: DepformerConfig
    @ModuleInfo public var linear_in: MultiLinear   // dim -> 2 * dimFeedforward
    @ModuleInfo public var linear_out: MultiLinear   // dimFeedforward -> dim

    public init(cfg: DepformerConfig) {
        self.cfg = cfg
        self._linear_in = ModuleInfo(wrappedValue: MultiLinear(
            numSteps: cfg.numSteps, inDim: cfg.dim, outDim: 2 * cfg.dimFeedforward, bias: false))
        self._linear_out = ModuleInfo(wrappedValue: MultiLinear(
            numSteps: cfg.numSteps, inDim: cfg.dimFeedforward, outDim: cfg.dim, bias: false))
    }

    public func callAsFunction(_ xs: MLXArray, step: Int) -> MLXArray {
        let b = xs.shape[0], t = xs.shape[1]
        let doubled = linear_in(xs, step: step)
        let ffnDim = cfg.dimFeedforward
        let split2 = doubled.reshaped([b, t, 2, ffnDim])
        let parts = split(split2, indices: [1], axis: 2)
        let gate = parts[0]
        let value = parts[1]
        let gated = silu(gate) * value
        let flat = gated.reshaped([b, t, ffnDim])
        return linear_out(flat, step: step)
    }
}

// MARK: - Depformer Layer

public final class DepformerLayer: Module {
    @ModuleInfo public var norm1: RMSNormF32
    @ModuleInfo public var norm2: RMSNormF32
    @ModuleInfo public var self_attn: DepformerAttention
    @ModuleInfo public var gating: DepformerFFN

    public init(cfg: DepformerConfig) {
        self._norm1 = ModuleInfo(wrappedValue: RMSNormF32(dimensions: cfg.dim, eps: cfg.rmsNormEps))
        self._norm2 = ModuleInfo(wrappedValue: RMSNormF32(dimensions: cfg.dim, eps: cfg.rmsNormEps))
        self._self_attn = ModuleInfo(wrappedValue: DepformerAttention(cfg: cfg))
        self._gating = ModuleInfo(wrappedValue: DepformerFFN(cfg: cfg))
    }

    public func callAsFunction(_ xs: MLXArray, step: Int, cache: KVCacheSimple) -> MLXArray {
        var x = xs
        x = x + self_attn(norm1(x), step: step, cache: cache)
        x = x + gating(norm2(x), step: step)
        return x
    }
}

// MARK: - Depformer

public final class Depformer: Module {
    public let cfg: DepformerConfig

    @ModuleInfo public var layers: [DepformerLayer]

    // Per-codebook input projections: temporal dim -> depformer dim
    @ModuleInfo public var depformer_in: [Linear]

    // Per-codebook embeddings for previous token
    // depformer_text_emb: text embedding for step 0
    // depformer_emb: audio embeddings for steps 1...(numSteps-1)
    @ModuleInfo public var depformer_text_emb: Embedding
    @ModuleInfo public var depformer_emb: [Embedding]

    // Per-codebook output linear heads (no per-codebook norms in original model)
    @ModuleInfo public var linears: [Linear]

    public init(cfg: DepformerConfig, temporalDim: Int) {
        self.cfg = cfg

        self._layers = ModuleInfo(wrappedValue:
            (0..<cfg.numLayers).map { _ in DepformerLayer(cfg: cfg) })

        // Input projections: one per step (multi_linear mode)
        var inProjs: [Linear] = []
        for _ in 0..<cfg.numSteps {
            inProjs.append(Linear(temporalDim, cfg.dim, bias: false))
        }
        self._depformer_in = ModuleInfo(wrappedValue: inProjs)

        // Text embedding for step 0
        self._depformer_text_emb = ModuleInfo(wrappedValue:
            Embedding(embeddingCount: cfg.textCard + 1, dimensions: cfg.dim))

        // Audio embeddings for steps 1..<numSteps
        var audioEmbs: [Embedding] = []
        for _ in 0..<(cfg.numSteps - 1) {
            audioEmbs.append(Embedding(embeddingCount: cfg.card + 1, dimensions: cfg.dim))
        }
        self._depformer_emb = ModuleInfo(wrappedValue: audioEmbs)

        // Per-codebook output heads (card outputs, no +1 — special token only in embedding)
        var heads: [Linear] = []
        for _ in 0..<cfg.numSteps {
            heads.append(Linear(cfg.dim, cfg.card, bias: false))
        }
        self._linears = ModuleInfo(wrappedValue: heads)
    }

    /// Generate all codebook tokens for one timestep.
    /// - Parameters:
    ///   - temporalHidden: [B, 1, temporalDim] hidden state from temporal transformer
    ///   - textToken: [B] sampled text token (input to step 0)
    ///   - sampleFn: sampling function (logits, codebookIndex) -> token
    /// - Returns: [B, numSteps] generated audio codebook tokens
    public func generate(
        temporalHidden: MLXArray,
        textToken: MLXArray,
        sampleFn: (MLXArray, Int) -> MLXArray
    ) -> MLXArray {
        var tokens: [MLXArray] = []
        var prevToken = textToken  // [B]

        // Create KV caches ONCE for this temporal step — all codebook steps
        // share the same caches so step k can attend to steps 0..k-1
        let caches = (0..<cfg.numLayers).map { _ in KVCacheSimple() }

        for k in 0..<cfg.numSteps {
            // Project temporal hidden to depformer dim
            var input = depformer_in[k](temporalHidden)  // [B, 1, depformerDim]

            // Add previous token embedding
            if k == 0 {
                input = input + depformer_text_emb(prevToken.expandedDimensions(axis: 1))
            } else {
                input = input + depformer_emb[k - 1](prevToken.expandedDimensions(axis: 1))
            }

            // Pass through depformer layers (shared caches accumulate across steps)
            var hidden = input
            for (layer, cache) in zip(layers, caches) {
                hidden = layer(hidden, step: k, cache: cache)
            }

            // Output head (no per-codebook norm)
            let logits = linears[k](hidden)  // [B, 1, card]

            // Sample
            let token = sampleFn(logits.squeezed(axis: 1), k)  // [B]
            tokens.append(token)
            prevToken = token
        }

        return stacked(tokens, axis: 1)  // [B, numSteps]
    }
}
