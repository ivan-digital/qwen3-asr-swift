import Foundation
import MLX
import MLXCommon
import MLXFast
import MLXNN
import MLXRandom
import PersonaPlex   // RMSNormF32, KVCache, KVCacheSimple, makeLinear, applyLinear

// MARK: - Scheduled MultiLinear

/// Per-step linear layer where each generation step indexes into a (possibly
/// smaller) pool of unique slice weights via a `schedule` array.
///
/// For PersonaPlex (no schedule), each of `numSteps` generation steps has its
/// own slice → storage shape `[numSteps * outDim, packedInDim]`.
///
/// For Hibiki Zero-3B / 2B (with `weights_per_step_schedule = [0..8, 8x8]`),
/// 16 generation steps map onto **9 unique slices** → storage shape
/// `[9 * outDim, packedInDim]`. At step k, slice index = `schedule[k]`.
///
/// Storage and quantization layout match MLX `QuantizedLinear`:
///   - bf16: `weight: [numUniqueSlices * outDim, inDim]`
///   - 4-bit: `weight: [numUniqueSlices * outDim, inDim/8]` packed uint32,
///            `scales/biases: [numUniqueSlices * outDim, inDim/groupSize]`
public final class ScheduledMultiLinear: Module {
    public var weight: MLXArray
    public var scales: MLXArray?
    public var biases: MLXArray?
    public var bias: MLXArray?

    private let schedule: [Int]
    private let numUniqueSlices: Int
    private let outDim: Int
    private let groupSize: Int
    private let bits: Int

    /// - Parameters:
    ///   - schedule: maps generation step → slice index. Length = numSteps.
    ///   - inDim: input feature dim (unpacked).
    ///   - outDim: per-step output feature dim.
    ///   - bias: whether to allocate a per-step bias `[numSteps, outDim]`.
    ///   - groupSize/bits: MLX quantization params (bits<16 enables quantization).
    public init(schedule: [Int], inDim: Int, outDim: Int, bias: Bool = false,
                groupSize: Int = 64, bits: Int = 16) {
        self.schedule = schedule
        self.numUniqueSlices = (schedule.max() ?? -1) + 1
        precondition(numUniqueSlices > 0, "schedule must be non-empty with non-negative indices")
        self.outDim = outDim
        self.groupSize = groupSize
        self.bits = bits

        let storedRows = numUniqueSlices * outDim
        if bits < 16 {
            let packedCols = inDim / (32 / bits)
            let numGroups = inDim / groupSize
            self.weight = MLXArray.zeros([storedRows, packedCols], dtype: .uint32)
            self.scales = MLXArray.zeros([storedRows, numGroups], dtype: .float16)
            self.biases = MLXArray.zeros([storedRows, numGroups], dtype: .float16)
        } else {
            let scale: Float = 1.0 / Float(inDim)
            self.weight = MLXRandom.uniform(low: -scale, high: scale, [storedRows, inDim])
            self.scales = nil
            self.biases = nil
        }
        // Bias is per generation step (matches PersonaPlex MultiLinear).
        self.bias = bias ? MLXArray.zeros([schedule.count, outDim]) : nil
    }

    public func callAsFunction(_ xs: MLXArray, step: Int) -> MLXArray {
        precondition(step >= 0 && step < schedule.count,
                     "ScheduledMultiLinear: step \(step) out of range \(schedule.count)")
        let slice = schedule[step]
        let start = slice * outDim
        let end = start + outDim
        let w = weight[start..<end, 0...]

        var result: MLXArray
        if let s = scales, let b = biases {
            let ws = s[start..<end, 0...]
            let wb = b[start..<end, 0...]
            result = quantizedMM(
                xs, w, scales: ws, biases: wb,
                transpose: true, groupSize: groupSize, bits: bits)
        } else {
            result = xs.matmul(w.T)
        }

        if let b = bias {
            result = result + b[step]
        }
        return result
    }
}

// MARK: - Depformer Attention

public final class HibikiDepformerAttention: Module {
    private let cfg: HibikiDepformerConfig
    @ModuleInfo public var in_proj: ScheduledMultiLinear
    @ModuleInfo public var out_proj: ScheduledMultiLinear

    private let scale: Float

    public init(cfg: HibikiDepformerConfig) {
        self.cfg = cfg
        let totalDim = 3 * cfg.dim   // Q + K + V packed (depformer is MHA, kvRepeat=1)
        let schedule = cfg.weightsPerStepSchedule ?? Array(0..<cfg.numSteps)
        self._in_proj = ModuleInfo(wrappedValue: ScheduledMultiLinear(
            schedule: schedule, inDim: cfg.dim, outDim: totalDim, bias: false,
            groupSize: cfg.groupSize, bits: cfg.bits))
        self._out_proj = ModuleInfo(wrappedValue: ScheduledMultiLinear(
            schedule: schedule, inDim: cfg.dim, outDim: cfg.dim, bias: false,
            groupSize: cfg.groupSize, bits: cfg.bits))
        self.scale = 1.0 / Float(Double(cfg.headDim).squareRoot())
    }

    public func callAsFunction(
        _ xs: MLXArray, step: Int, cache: any KVCache
    ) -> MLXArray {
        let b = xs.shape[0]
        let t = xs.shape[1]

        let qkv = in_proj(xs, step: step)
        let qkvR = qkv.reshaped([b, t, 3, cfg.numHeads, cfg.headDim])

        let q = swappedAxes(qkvR[0..<b, 0..<t, 0, 0..<cfg.numHeads, 0..<cfg.headDim], 1, 2)
        var k = swappedAxes(qkvR[0..<b, 0..<t, 1, 0..<cfg.numHeads, 0..<cfg.headDim], 1, 2)
        var v = swappedAxes(qkvR[0..<b, 0..<t, 2, 0..<cfg.numHeads, 0..<cfg.headDim], 1, 2)

        // depformer_pos_emb = "none" — no RoPE in depformer.
        (k, v) = cache.update(keys: k, values: v)

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

        let merged = SDPA.attendAndMerge(
            qHeads: q, kHeads: k, vHeads: v, scale: scale, mask: maskMode)
        return out_proj(merged, step: step)
    }
}

// MARK: - Depformer FFN

public final class HibikiDepformerFFN: Module {
    private let cfg: HibikiDepformerConfig
    @ModuleInfo public var linear_in: ScheduledMultiLinear
    @ModuleInfo public var linear_out: ScheduledMultiLinear

    public init(cfg: HibikiDepformerConfig) {
        self.cfg = cfg
        let schedule = cfg.weightsPerStepSchedule ?? Array(0..<cfg.numSteps)
        self._linear_in = ModuleInfo(wrappedValue: ScheduledMultiLinear(
            schedule: schedule, inDim: cfg.dim, outDim: 2 * cfg.dimFeedforward, bias: false,
            groupSize: cfg.groupSize, bits: cfg.bits))
        self._linear_out = ModuleInfo(wrappedValue: ScheduledMultiLinear(
            schedule: schedule, inDim: cfg.dimFeedforward, outDim: cfg.dim, bias: false,
            groupSize: cfg.groupSize, bits: cfg.bits))
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

public final class HibikiDepformerLayer: Module {
    @ModuleInfo public var norm1: RMSNormF32
    @ModuleInfo public var norm2: RMSNormF32
    @ModuleInfo public var self_attn: HibikiDepformerAttention
    @ModuleInfo public var gating: HibikiDepformerFFN

    public init(cfg: HibikiDepformerConfig) {
        self._norm1 = ModuleInfo(wrappedValue: RMSNormF32(dimensions: cfg.dim, eps: cfg.rmsNormEps))
        self._norm2 = ModuleInfo(wrappedValue: RMSNormF32(dimensions: cfg.dim, eps: cfg.rmsNormEps))
        self._self_attn = ModuleInfo(wrappedValue: HibikiDepformerAttention(cfg: cfg))
        self._gating = ModuleInfo(wrappedValue: HibikiDepformerFFN(cfg: cfg))
    }

    public func callAsFunction(_ xs: MLXArray, step: Int, cache: any KVCache) -> MLXArray {
        var x = xs
        x = x + self_attn(norm1(x), step: step, cache: cache)
        x = x + gating(norm2(x), step: step)
        return x
    }
}

// MARK: - Depformer

public final class HibikiDepformer: Module {
    public let cfg: HibikiDepformerConfig

    @ModuleInfo public var layers: [HibikiDepformerLayer]
    /// Per-slice input projection: temporal dim → depformer dim. **Scheduled**
    /// — only `numUniqueSlices` modules (9 for Zero-3B), indexed at each step
    /// via `cfg.sliceIndex(forStep:)`. Upstream `forward_depformer` (lm.py
    /// line 471-475): `in_index = schedule[depformer_cb_index]`.
    @ModuleInfo public var depformer_in: [Module]
    /// Text embedding for step 0 input token.
    @ModuleInfo public var depformer_text_emb: Embedding
    /// Audio embeddings for steps 1..(numSteps-1) (one per previous codebook).
    @ModuleInfo public var depformer_emb: [Embedding]
    /// Per-step LayerNorm applied between the depformer hidden state and the
    /// output `linears[k]` head. Zero-3B uses LayerNorm (with weight + bias);
    /// PersonaPlex omits this layer entirely.
    @ModuleInfo public var depformer_norms: [LayerNorm]
    /// Per-step output linear heads (16 unique heads — NOT scheduled).
    @ModuleInfo public var linears: [Linear]

    public init(cfg: HibikiDepformerConfig, temporalDim: Int) {
        self.cfg = cfg

        self._layers = ModuleInfo(wrappedValue:
            (0..<cfg.numLayers).map { _ in HibikiDepformerLayer(cfg: cfg) })

        // Allocate `numUniqueSlices` (= 9 for Zero-3B) input projections,
        // not `numSteps` (= 16). They will be indexed via the schedule at
        // generation time. Allocating numSteps and indexing by k directly
        // leaves slices 9..15 at random init (the real bug we fixed).
        var inProjs: [Module] = []
        for _ in 0..<cfg.numUniqueSlices {
            inProjs.append(makeLinear(temporalDim, cfg.dim, bias: false,
                                      groupSize: cfg.groupSize, bits: cfg.bits))
        }
        self._depformer_in = ModuleInfo(wrappedValue: inProjs)

        self._depformer_text_emb = ModuleInfo(wrappedValue:
            Embedding(embeddingCount: cfg.textCard + 1, dimensions: cfg.dim))

        var audioEmbs: [Embedding] = []
        for _ in 0..<(cfg.numSteps - 1) {
            audioEmbs.append(Embedding(embeddingCount: cfg.card + 1, dimensions: cfg.dim))
        }
        self._depformer_emb = ModuleInfo(wrappedValue: audioEmbs)

        // Per-step LayerNorms: depformer_norms[0..15], dim=1024 each.
        var norms: [LayerNorm] = []
        for _ in 0..<cfg.numSteps {
            norms.append(LayerNorm(dimensions: cfg.dim, eps: cfg.rmsNormEps))
        }
        self._depformer_norms = ModuleInfo(wrappedValue: norms)

        var heads: [Linear] = []
        for _ in 0..<cfg.numSteps {
            heads.append(Linear(cfg.dim, cfg.card, bias: false))
        }
        self._linears = ModuleInfo(wrappedValue: heads)
    }

    /// Generate target codebook tokens for one temporal frame.
    ///
    /// For Hibiki, the depformer generates the **target-side** codebooks
    /// (output language audio). Each step k receives the previous token
    /// (text token at k=0, target-codebook[k-1] otherwise), projects the
    /// shared temporal hidden state through `depformer_in[k]`, runs the
    /// depformer transformer with shared KV caches across the 16 steps,
    /// and emits one codebook token via `linears[k]`.
    ///
    /// - Parameters:
    ///   - temporalHidden: `[B, 1, temporalDim]` from the temporal transformer.
    ///   - textToken: `[B]` sampled text token (input to step 0).
    ///   - sampleFn: `(logits [B, card], codebookIndex) -> token [B]`.
    /// - Returns: `[B, numSteps]` target codebook tokens.
    public func generate(
        temporalHidden: MLXArray,
        textToken: MLXArray,
        sampleFn: (MLXArray, Int) -> MLXArray
    ) -> MLXArray {
        var tokens: [MLXArray] = []
        var prevToken = textToken

        // Shared KV caches across all 16 steps so step k attends to 0..<k.
        let caches: [any KVCache] = (0..<cfg.numLayers).map { _ in KVCacheSimple() }

        for k in 0..<cfg.numSteps {
            // depformer_in is scheduled — index via the slice schedule.
            let inSlice = cfg.sliceIndex(forStep: k)
            var input = applyLinear(depformer_in[inSlice], temporalHidden)

            if k == 0 {
                input = input + depformer_text_emb(prevToken.expandedDimensions(axis: 1))
            } else {
                input = input + depformer_emb[k - 1](prevToken.expandedDimensions(axis: 1))
            }

            var hidden = input
            for (layer, cache) in zip(layers, caches) {
                hidden = layer(hidden, step: k, cache: cache)
            }

            // Per-step LayerNorm before the output head.
            let normed = depformer_norms[k](hidden)
            let logits = linears[k](normed)
            let token = sampleFn(logits.squeezed(axis: 1), k)
            tokens.append(token)
            prevToken = token
        }

        return stacked(tokens, axis: 1)
    }
}
