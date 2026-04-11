import Foundation
import MLXCommon
import MLX
import MLXNN
import MLXFast

// MARK: - DeltaNet Linear Attention

/// DeltaNet linear attention layer for Qwen3.5 hybrid model.
///
/// Uses linear attention (no softmax) with a recurrent state matrix S of shape [B, H, D, D].
/// The state evolves per-token: S = alpha * S + beta * (v outer k), where alpha/beta
/// are learned per-head scalar gates derived from the input via softplus/sigmoid.
///
/// A causal conv1d (kernel=4) provides short-range local context before the attention.
/// Output is gated: `o_proj(attention_output * silu(z))` where both are 2*hiddenSize = numHeads*headDim.
///
/// Weight shapes (HuggingFace safetensors):
///   - in_proj_qkv.weight: [6144, 1024]   (3 * 16 * 128)
///   - in_proj_z.weight: [2048, 1024]      (2 * hiddenSize, for gate)
///   - in_proj_b.weight: [16, 1024]        (beta gate, per head)
///   - in_proj_a.weight: [16, 1024]        (alpha gate, per head)
///   - conv1d.weight: [6144, 4, 1]         (depthwise causal conv, MLX [C, K, 1] format)
///   - dt_bias: [16]                       (time-step bias)
///   - A_log: [16]                         (log of decay rate)
///   - norm.weight: [128]                  (per-head RMSNorm)
///   - out_proj.weight: [1024, 2048]       (gated output projection)
public final class DeltaNetLayer: Module {
    let numHeads: Int
    let headDim: Int
    let hiddenSize: Int
    let convKernel: Int
    let qkvDim: Int

    @ModuleInfo(key: "in_proj_qkv") var inProjQKV: QuantizedLinear
    @ModuleInfo(key: "in_proj_z") var inProjZ: QuantizedLinear
    @ModuleInfo(key: "in_proj_b") var inProjB: QuantizedLinear
    @ModuleInfo(key: "in_proj_a") var inProjA: QuantizedLinear

    /// Conv1d weight: [C, 1, K] depthwise convolution applied to QKV before attention.
    /// Stored under a flat key to avoid nested key path issues in MLXNN module traversal.
    /// Weight loading applies this directly via `layer.convWeight = ...`.
    @ParameterInfo(key: "conv1d_weight") var convWeight: MLXArray

    @ParameterInfo(key: "dt_bias") var dtBias: MLXArray
    @ParameterInfo(key: "A_log") var aLog: MLXArray

    @ModuleInfo var norm: RMSNorm

    @ModuleInfo(key: "out_proj") var outProj: QuantizedLinear

    public init(config: Qwen3ChatConfig) {
        self.numHeads = config.linearNumKeyHeads ?? 16
        self.headDim = config.linearKeyHeadDim ?? 128
        self.hiddenSize = config.hiddenSize
        self.convKernel = config.linearConvKernelDim ?? 4
        self.qkvDim = 3 * numHeads * headDim

        let groupSize = 64
        let bits = 4
        self._inProjQKV = ModuleInfo(wrappedValue: QuantizedLinear(hiddenSize, qkvDim, bias: false, groupSize: groupSize, bits: bits))
        self._inProjZ = ModuleInfo(wrappedValue: QuantizedLinear(hiddenSize, 2 * hiddenSize, bias: false, groupSize: groupSize, bits: bits))
        self._inProjB = ModuleInfo(wrappedValue: QuantizedLinear(hiddenSize, numHeads, bias: false, groupSize: groupSize, bits: bits))
        self._inProjA = ModuleInfo(wrappedValue: QuantizedLinear(hiddenSize, numHeads, bias: false, groupSize: groupSize, bits: bits))

        self._convWeight = ParameterInfo(
            wrappedValue: MLXArray.zeros([qkvDim, convKernel, 1]))

        self._dtBias = ParameterInfo(wrappedValue: MLXArray.zeros([numHeads]))
        self._aLog = ParameterInfo(wrappedValue: MLXArray.zeros([numHeads]))

        self._norm = ModuleInfo(
            wrappedValue: RMSNorm(dimensions: headDim, eps: Float(config.rmsNormEps)))

        self._outProj = ModuleInfo(wrappedValue: QuantizedLinear(2 * hiddenSize, hiddenSize, bias: false, groupSize: groupSize, bits: bits))

        super.init()
    }

    /// Recurrent state for a DeltaNet layer.
    public struct State {
        /// Recurrent state matrix [B, H, D, D]
        var s: MLXArray
        /// Conv1d ring buffer [B, C, K-1] storing last K-1 inputs
        var convState: MLXArray

        public static func initial(
            batchSize: Int, numHeads: Int, headDim: Int,
            qkvDim: Int, convKernel: Int, dtype: DType = .float32
        ) -> State {
            State(
                s: MLXArray.zeros([batchSize, numHeads, headDim, headDim], dtype: dtype),
                convState: MLXArray.zeros([batchSize, qkvDim, convKernel - 1], dtype: dtype)
            )
        }
    }

    /// Forward pass processing a sequence of tokens.
    ///
    /// Implements the gated delta rule recurrence (reference: mlx-lm/gated_delta.py):
    ///   1. Decay: S = g * S
    ///   2. Error: kv_mem = (S * k).sum(-1); delta = (v - kv_mem) * beta
    ///   3. Update: S = S + k * delta
    ///   4. Output: y = (S * q).sum(-1)
    ///
    /// - Parameters:
    ///   - x: Input hidden states [B, T, hiddenSize]
    ///   - state: Previous recurrent state (nil for first call)
    /// - Returns: (output [B, T, hiddenSize], updated state)
    public func callAsFunction(_ x: MLXArray, state: State? = nil) -> (MLXArray, State) {
        let b = x.dim(0)
        let t = x.dim(1)

        // Project inputs (separate projections matching HuggingFace weight format)
        let qkvRaw = inProjQKV(x)    // [B, T, 3*H*D=6144]
        let zRaw = inProjZ(x)        // [B, T, 2*hiddenSize=2048]
        let bRaw = inProjB(x)        // [B, T, H=16]
        let aRaw = inProjA(x)        // [B, T, H=16]


        // Causal conv1d on QKV only (not Z, B, A)
        let prevConvState: MLXArray
        if let s = state {
            prevConvState = s.convState
        } else {
            prevConvState = MLXArray.zeros([b, qkvDim, convKernel - 1], dtype: x.dtype)
        }

        let qkvTransposed = qkvRaw.transposed(0, 2, 1)  // [B, C, T]
        let padded = concatenated([prevConvState, qkvTransposed], axis: 2)  // [B, C, T+K-1]

        // Save new conv state (last K-1 columns)
        let totalLen = padded.dim(2)
        let newConvState = padded[0..., 0..., (totalLen - convKernel + 1)...]

        // Apply depthwise causal conv1d + SiLU
        let qkvConv = depthwiseConv1dCausal(padded, outputLen: t)
        let qkvActivated = silu(qkvConv.transposed(0, 2, 1))  // [B, T, C]

        // Split into Q, K, V — each [B, T, H, D]
        let hd = numHeads * headDim
        var q = qkvActivated[0..., 0..., ..<hd].reshaped(b, t, numHeads, headDim)
        var k = qkvActivated[0..., 0..., hd..<(2 * hd)].reshaped(b, t, numHeads, headDim)
        let v = qkvActivated[0..., 0..., (2 * hd)...].reshaped(b, t, numHeads, headDim)

        // Q/K normalization (reference: inv_scale * rms_norm, different scaling for Q and K)
        // rms_norm(x, None, eps) = x / sqrt(mean(x^2) + eps)
        // q = inv_scale^2 * rms_norm(q)  where inv_scale = head_dim^(-0.5)
        // k = inv_scale * rms_norm(k)
        let invScale = Float(1.0) / sqrt(Float(headDim))
        q = MLXArray(invScale * invScale) * rmsNormNoWeight(q)
        k = MLXArray(invScale) * rmsNormNoWeight(k)

        // Compute gating: g = exp(-exp(A_log) * softplus(a + dt_bias))
        let g = computeDecayGate(aRaw: aRaw)  // [B, T, H]
        // beta = sigmoid(b_raw)  (independent learned gate, NOT 1-alpha)
        let beta = sigmoid(bRaw)  // [B, T, H]

        // Sequential gated delta rule recurrence
        var currentS: MLXArray
        if let s = state {
            currentS = s.s
        } else {
            currentS = MLXArray.zeros([b, numHeads, headDim, headDim], dtype: x.dtype)
        }

        var outputSteps: [MLXArray] = []
        outputSteps.reserveCapacity(t)

        for step in 0..<t {
            // Extract step: [B, H, D] or [B, H]
            let qStep = q[0..., step..<(step + 1), 0..., 0...].squeezed(axis: 1)  // [B, H, D]
            let kStep = k[0..., step..<(step + 1), 0..., 0...].squeezed(axis: 1)  // [B, H, D]
            let vStep = v[0..., step..<(step + 1), 0..., 0...].squeezed(axis: 1)  // [B, H, D]
            let gStep = g[0..., step..<(step + 1), 0...].squeezed(axis: 1)        // [B, H]
            let betaStep = beta[0..., step..<(step + 1), 0...].squeezed(axis: 1)  // [B, H]

            // 1. Decay: S = g * S   (g is scalar per-head: [B, H, 1, 1])
            let decay = gStep.reshaped(b, numHeads, 1, 1)
            currentS = currentS * decay

            // 2. Error correction:
            //    kv_mem = (S * k[..., None, :]).sum(-1)  →  [B, H, Dv]
            //    delta = (v - kv_mem) * beta[..., None]  →  [B, H, Dv]
            let kExpanded = kStep.expandedDimensions(axis: -2)  // [B, H, 1, Dk]
            let kvMem = (currentS * kExpanded).sum(axis: -1)     // [B, H, Dv]
            let delta = (vStep - kvMem) * betaStep.expandedDimensions(axis: -1)  // [B, H, Dv]

            // 3. Update: S = S + k[..., None, :] * delta[..., None]
            //    k: [B, H, Dk] → [B, H, 1, Dk], delta: [B, H, Dv] → [B, H, Dv, 1]
            currentS = currentS + kExpanded * delta.expandedDimensions(axis: -1)

            // 4. Output: y = (S * q[..., None, :]).sum(-1)  →  [B, H, Dv]
            let qExpanded = qStep.expandedDimensions(axis: -2)  // [B, H, 1, Dk]
            let oStep = (currentS * qExpanded).sum(axis: -1)     // [B, H, Dv]
            outputSteps.append(oStep)
        }

        // Stack: [B, T, H, D]
        let output = stacked(outputSteps, axis: 1)

        // RMSNormGated: norm(output) * silu(z)
        // z has shape [B, T, 2*hiddenSize=2048], reshape to [B, T, H, D] for per-head norm
        let zReshaped = zRaw.reshaped(b, t, numHeads, headDim)
        let normedOutput = norm(output)  // per-head RMSNorm, [B, T, H, D]
        let gated = normedOutput * silu(zReshaped)  // [B, T, H, D]

        // Reshape to [B, T, H*D=2048] and project to hiddenSize
        let result = outProj(gated.reshaped(b, t, numHeads * headDim))  // [B, T, 1024]

        return (result, State(s: currentS, convState: newConvState))
    }

    /// Compute decay gate: g = exp(-exp(A_log) * softplus(a + dt_bias))
    private func computeDecayGate(aRaw: MLXArray) -> MLXArray {
        let a = aRaw + dtBias.reshaped(1, 1, numHeads)
        let dt = softplus(a)
        let negExpA = -exp(aLog.asType(.float32)).reshaped(1, 1, numHeads)
        return exp(negExpA * dt.asType(.float32)).asType(aRaw.dtype)
    }

    /// RMS normalization without learnable weight (used for Q/K normalization).
    private func rmsNormNoWeight(_ x: MLXArray) -> MLXArray {
        let meanSq = (x * x).mean(axis: -1, keepDims: true)
        return x * rsqrt(meanSq + MLXArray(Float(1e-6)))
    }

    // MARK: - Depthwise Conv1d

    /// Depthwise causal conv1d via unfolding + element-wise multiply + sum.
    /// - Parameter input: [B, C, T+K-1] (pre-padded with conv state)
    /// - Parameter outputLen: number of output time steps T
    /// - Returns: [B, C, T]
    private func depthwiseConv1dCausal(_ input: MLXArray, outputLen: Int) -> MLXArray {
        let c = input.dim(1)
        let k = convKernel

        // Unfold: gather windows of size K for each output position
        var windows: [MLXArray] = []
        windows.reserveCapacity(outputLen)
        for t in 0..<outputLen {
            windows.append(input[0..., 0..., t..<(t + k)])  // [B, C, K]
        }
        let unfolded = stacked(windows, axis: 2)  // [B, C, T, K]

        // Kernel: [C, K, 1] -> squeeze axis 2 -> [C, K] -> [1, C, 1, K]
        let kernelBcast = convWeight.squeezed(axis: 2).reshaped(1, c, 1, k)

        return (unfolded * kernelBcast).sum(axis: -1)  // [B, C, T]
    }
}

// MARK: - Softplus

private func softplus(_ x: MLXArray) -> MLXArray {
    // Numerically stable: for large x, softplus(x) ~ x
    MLX.where(x .> MLXArray(Float(20.0)), x, log(1 + exp(x)))
}

// MARK: - GatedAttention (Full Attention)

/// GatedAttention layer for Qwen3.5 hybrid model.
///
/// Standard multi-head attention with:
///   - GQA: 8 query heads, 2 KV heads, head_dim=256
///   - Partial RoPE: only first 25% of head_dim (64 dims) get rotary encoding
///   - QK norm: RMSNorm applied per-head to Q and K before RoPE
///   - Gated output: q_proj produces [Q; gate], both of dim numQHeads*headDim.
///     After attention, output is element-wise multiplied with silu(gate),
///     then projected through o_proj.
///
/// Weight shapes:
///   - q_proj: [4096, 1024]  = [2 * numQHeads * headDim, hiddenSize] (Q + gate)
///   - k_proj: [512, 1024]   = [numKVHeads * headDim, hiddenSize]
///   - v_proj: [512, 1024]   = [numKVHeads * headDim, hiddenSize]
///   - o_proj: [1024, 2048]  = [hiddenSize, numQHeads * headDim]
///   - q_norm: [256]         = [headDim]
///   - k_norm: [256]         = [headDim]
public final class GatedAttentionLayer: Module {
    let numQHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let hiddenSize: Int
    let scale: Float
    let ropeDims: Int  // partial RoPE dimensions

    @ModuleInfo(key: "q_proj") var qProj: QuantizedLinear
    @ModuleInfo(key: "k_proj") var kProj: QuantizedLinear
    @ModuleInfo(key: "v_proj") var vProj: QuantizedLinear
    @ModuleInfo(key: "o_proj") var oProj: QuantizedLinear

    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    let rope: MLXNN.RoPE

    public init(config: Qwen3ChatConfig) {
        self.numQHeads = config.numAttentionHeads   // 8
        self.numKVHeads = config.numKeyValueHeads    // 2
        self.headDim = config.headDim                // 256
        self.hiddenSize = config.hiddenSize           // 1024
        self.scale = 1.0 / sqrt(Float(headDim))

        let factor = config.partialRotaryFactor ?? 0.25
        self.ropeDims = Int(Double(headDim) * factor)  // 64

        let groupSize = 64
        let bits = 4
        let qDim = numQHeads * headDim  // 2048

        // q_proj outputs 2 * qDim (Q + gate)
        self._qProj = ModuleInfo(wrappedValue:  QuantizedLinear(
                hiddenSize, 2 * qDim, bias: false,
                groupSize: groupSize, bits: bits))
        self._kProj = ModuleInfo(wrappedValue:  QuantizedLinear(
                hiddenSize, numKVHeads * headDim, bias: false,
                groupSize: groupSize, bits: bits))
        self._vProj = ModuleInfo(wrappedValue:  QuantizedLinear(
                hiddenSize, numKVHeads * headDim, bias: false,
                groupSize: groupSize, bits: bits))
        // o_proj: qDim -> hiddenSize (after gating reduces 2*qDim to qDim)
        self._oProj = ModuleInfo(wrappedValue:  QuantizedLinear(
                qDim, hiddenSize, bias: false,
                groupSize: groupSize, bits: bits))

        self._qNorm = ModuleInfo(wrappedValue:  RMSNorm(dimensions: headDim, eps: Float(config.rmsNormEps)))
        self._kNorm = ModuleInfo(wrappedValue:  RMSNorm(dimensions: headDim, eps: Float(config.rmsNormEps)))

        // Partial RoPE: only rotates first `ropeDims` of each head
        self.rope = MLXNN.RoPE(
            dimensions: ropeDims,
            traditional: false,
            base: Float(config.ropeTheta))

        super.init()
    }

    /// Forward pass.
    ///
    /// - Parameters:
    ///   - hiddenStates: [B, T, hiddenSize]
    ///   - cache: Optional (keys, values) from previous steps, each [B, H_kv, S, D]
    ///   - offset: RoPE position offset (used when cache is nil, e.g. first call)
    /// - Returns: (output [B, T, hiddenSize], updated KV cache)
    public func callAsFunction(
        _ hiddenStates: MLXArray,
        cache: (MLXArray, MLXArray)? = nil,
        offset: Int = 0
    ) -> (MLXArray, (MLXArray, MLXArray)) {
        let b = hiddenStates.dim(0)
        let seqLen = hiddenStates.dim(1)
        let qDim = numQHeads * headDim

        // Q projection: [B, T, 2*qDim] → reshape to [B, T, H, 2*D] → split Q/gate INTERLEAVED per head
        // CRITICAL: Must reshape BEFORE split (per Python reference).
        // Interleaved format: for each head, first D dims are Q, next D are gate.
        let qProjOut = qProj(hiddenStates)  // [B, T, 4096 = 2*numQHeads*headDim]
        let qProjReshaped = qProjOut.reshaped(b, seqLen, numQHeads, 2 * headDim)
        let qgSplit = qProjReshaped.split(parts: 2, axis: -1)
        var queries = qgSplit[0]                                  // [B, T, H, D=256]
        let gateSignal = qgSplit[1].reshaped(b, seqLen, qDim)    // [B, T, 2048]

        var keys = kProj(hiddenStates)     // [B, T, numKVHeads * headDim]
        var values = vProj(hiddenStates)   // [B, T, numKVHeads * headDim]

        // Reshape K/V to multi-head: [B, T, H, D]
        keys = keys.reshaped(b, seqLen, numKVHeads, headDim)
        values = values.reshaped(b, seqLen, numKVHeads, headDim)

        // QK norm (per-head)
        queries = qNorm(queries)
        keys = kNorm(keys)

        // Transpose to [B, H, T, D]
        queries = queries.transposed(0, 2, 1, 3)
        keys = keys.transposed(0, 2, 1, 3)
        values = values.transposed(0, 2, 1, 3)

        // Partial RoPE: MLXNN.RoPE with dimensions=ropeDims only rotates first ropeDims
        let ropeOffset = cache?.0.dim(2) ?? offset
        queries = rope(queries, offset: ropeOffset)
        keys = rope(keys, offset: ropeOffset)

        // Update KV cache
        var cachedKeys = keys
        var cachedValues = values
        if let (prevK, prevV) = cache {
            cachedKeys = concatenated([prevK, keys], axis: 2)
            cachedValues = concatenated([prevV, values], axis: 2)
        }

        // Causal mask
        let mask: MLXFast.ScaledDotProductAttentionMaskMode
        if seqLen <= 1 && (cache != nil || offset > 0) {
            mask = .none
        } else {
            let kvLen = cachedKeys.dim(2)
            let pastLen = kvLen - seqLen
            let causal = MLXArray.tri(seqLen, m: kvLen, k: pastLen, type: Float.self) - 1
            let additiveMask = causal * Float.greatestFiniteMagnitude  // 0 for attended, -FLT_MAX for masked
            mask = .array(additiveMask.reshaped(1, 1, seqLen, kvLen).asType(queries.dtype))
        }

        // SDPA (handles GQA natively)
        let attnOut = SDPA.attendAndMerge(
            qHeads: queries, kHeads: cachedKeys, vHeads: cachedValues,
            scale: scale, mask: mask)

        // Gated output: attn_out * sigmoid(gate), then o_proj
        // Reference: self.o_proj(output * mx.sigmoid(gate))
        let gated = attnOut * sigmoid(gateSignal)  // [B, T, qDim=2048]
        let output = oProj(gated)                   // [B, T, hiddenSize=1024]

        return (output, (cachedKeys, cachedValues))
    }
}

// MARK: - Qwen3.5 Transformer Layer

/// A single transformer layer in the Qwen3.5 hybrid model.
///
/// Either a DeltaNet (linear_attention) or GatedAttention (full_attention) layer,
/// both sharing the same pre-norm structure and SwiGLU MLP.
///
/// The attention submodule is stored as the base `Module` type and registered
/// under the key `"self_attn"` via `@ModuleInfo`. This ensures the MLX Module
/// system discovers it for parameter traversal (`eval`, `clearParameters`, etc.)
/// and weight loading maps to the correct key path.
public final class Qwen35TransformerLayer: Module {
    public let layerType: String

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm
    @ModuleInfo var mlp: Qwen35MLP

    /// The attention submodule — either DeltaNetLayer or GatedAttentionLayer.
    /// Key is "linear_attn" for DeltaNet, "self_attn" for GatedAttention (HuggingFace convention).
    @ModuleInfo var attn: Module

    public init(config: Qwen3ChatConfig, layerType: String) {
        self.layerType = layerType

        self._inputLayerNorm = ModuleInfo(wrappedValue:  RMSNorm(dimensions: config.hiddenSize, eps: Float(config.rmsNormEps)))
        self._postAttentionLayerNorm = ModuleInfo(wrappedValue:  RMSNorm(dimensions: config.hiddenSize, eps: Float(config.rmsNormEps)))
        self._mlp = ModuleInfo(
            wrappedValue: Qwen35MLP(config: config))

        if layerType == "linear_attention" {
            self._attn = ModuleInfo(
                wrappedValue: DeltaNetLayer(config: config),
                key: "linear_attn")
        } else {
            self._attn = ModuleInfo(
                wrappedValue: GatedAttentionLayer(config: config),
                key: "self_attn")
        }

        super.init()
    }

    /// Access the DeltaNet submodule (only valid for linear_attention layers).
    public var deltaNet: DeltaNetLayer? { attn as? DeltaNetLayer }

    /// Access the GatedAttention submodule (only valid for full_attention layers).
    public var gatedAttn: GatedAttentionLayer? { attn as? GatedAttentionLayer }

    /// Forward for DeltaNet (linear attention) layer.
    public func forwardDeltaNet(
        _ x: MLXArray,
        state: DeltaNetLayer.State?
    ) -> (MLXArray, DeltaNetLayer.State) {
        guard let dn = deltaNet else {
            fatalError("forwardDeltaNet called on full_attention layer")
        }
        let normed = inputLayerNorm(x)
        let (attnOut, newState) = dn(normed, state: state)
        var h = x + attnOut
        h = h + mlp(postAttentionLayerNorm(h))
        return (h, newState)
    }

    /// Forward for GatedAttention (full attention) layer.
    public func forwardGatedAttention(
        _ x: MLXArray,
        cache: (MLXArray, MLXArray)?,
        offset: Int
    ) -> (MLXArray, (MLXArray, MLXArray)) {
        guard let ga = gatedAttn else {
            fatalError("forwardGatedAttention called on linear_attention layer")
        }
        let normed = inputLayerNorm(x)
        let (attnOut, newCache) = ga(normed, cache: cache, offset: offset)
        var h = x + attnOut

        h = h + mlp(postAttentionLayerNorm(h))
        return (h, newCache)
    }
}

// MARK: - SwiGLU MLP

/// SwiGLU MLP for Qwen3.5 (quantized INT4).
public final class Qwen35MLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: QuantizedLinear
    @ModuleInfo(key: "up_proj") var upProj: QuantizedLinear
    @ModuleInfo(key: "down_proj") var downProj: QuantizedLinear

    public init(config: Qwen3ChatConfig) {
        let hs = config.hiddenSize
        let is_ = config.intermediateSize
        let gs = 64, bits = 4

        self._gateProj = ModuleInfo(
            wrappedValue: QuantizedLinear(hs, is_, bias: false, groupSize: gs, bits: bits),
            key: "gate_proj")
        self._upProj = ModuleInfo(
            wrappedValue: QuantizedLinear(hs, is_, bias: false, groupSize: gs, bits: bits),
            key: "up_proj")
        self._downProj = ModuleInfo(
            wrappedValue: QuantizedLinear(is_, hs, bias: false, groupSize: gs, bits: bits),
            key: "down_proj")

        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

// MARK: - Qwen3.5 Full Model

/// Qwen3.5-0.8B hybrid transformer with DeltaNet linear attention and GatedAttention.
///
/// Architecture: 24 layers in pattern [3x DeltaNet, 1x GatedAttention] x 6.
/// - DeltaNet layers (18 of 24): O(1) memory per step via recurrent state, no KV cache.
/// - GatedAttention layers (6 of 24): standard SDPA with KV cache, partial RoPE (25%).
/// - Tied embeddings: lm_head reuses embed_tokens weights (PreQuantizedEmbedding.asLinear).
///
/// This gives a favorable memory/compute tradeoff: recurrent DeltaNet layers handle
/// most computation with fixed memory, while sparse full attention layers provide
/// global context at every 4th layer.
public final class Qwen35MLXModel: Module {
    public let config: Qwen3ChatConfig
    public let layerTypes: [String]
    public let fullAttentionIndices: [Int]

    @ModuleInfo(key: "embed_tokens") var embedTokens: PreQuantizedEmbedding
    @ModuleInfo var layers: [Qwen35TransformerLayer]
    @ModuleInfo var norm: RMSNorm

    public init(config: Qwen3ChatConfig) {
        self.config = config

        let types = config.layerTypes ?? Array(
            repeating: "full_attention", count: config.numHiddenLayers)
        self.layerTypes = types
        self.fullAttentionIndices = types.enumerated().compactMap {
            $0.element == "full_attention" ? $0.offset : nil
        }

        self._embedTokens = ModuleInfo(wrappedValue:  PreQuantizedEmbedding(
                embeddingCount: config.vocabSize,
                dimensions: config.hiddenSize,
                groupSize: 64, bits: 4))

        self._layers = ModuleInfo(
            wrappedValue: types.map { Qwen35TransformerLayer(config: config, layerType: $0) })

        self._norm = ModuleInfo(
            wrappedValue: RMSNorm(dimensions: config.hiddenSize, eps: Float(config.rmsNormEps)))

        super.init()
    }

    // MARK: - Inference State

    /// Combined inference state: DeltaNet recurrent states + GatedAttention KV caches.
    public struct InferenceState {
        /// Per-layer DeltaNet state (nil for full_attention layers).
        public var deltaNetStates: [DeltaNetLayer.State?]
        /// Per-layer KV cache (nil for linear_attention layers, and for full_attention
        /// layers before any tokens have been processed).
        public var kvCaches: [(MLXArray, MLXArray)?]
        /// Current sequence position (for RoPE offset in GatedAttention layers).
        public var position: Int

        public static func initial(config: Qwen3ChatConfig, batchSize: Int = 1) -> InferenceState {
            let types = config.layerTypes ?? Array(
                repeating: "full_attention", count: config.numHiddenLayers)
            let numHeads = config.linearNumKeyHeads ?? 16
            let headDim = config.linearKeyHeadDim ?? 128
            let qkvDim = 3 * numHeads * headDim
            let convKernel = config.linearConvKernelDim ?? 4

            return InferenceState(
                deltaNetStates: types.map { type in
                    type == "linear_attention"
                        ? DeltaNetLayer.State.initial(
                            batchSize: batchSize, numHeads: numHeads, headDim: headDim,
                            qkvDim: qkvDim, convKernel: convKernel)
                        : nil
                },
                kvCaches: types.map { _ in nil },
                position: 0
            )
        }
    }

    // MARK: - Forward Pass

    /// Forward pass through the full model.
    ///
    /// - Parameters:
    ///   - inputIds: Token IDs [B, T]
    ///   - state: Inference state
    /// - Returns: (logits [B, T, vocabSize], updated state)
    public func forward(
        inputIds: MLXArray,
        state: InferenceState
    ) -> (MLXArray, InferenceState) {
        let seqLen = inputIds.dim(1)
        var hidden = embedTokens(inputIds)  // [B, T, hiddenSize]

        var newDeltaStates = state.deltaNetStates
        var newKVCaches = state.kvCaches

        for (i, layer) in layers.enumerated() {
            if layerTypes[i] == "linear_attention" {
                let (h, newState) = layer.forwardDeltaNet(hidden, state: state.deltaNetStates[i])
                hidden = h
                newDeltaStates[i] = newState
            } else {
                let (h, newCache) = layer.forwardGatedAttention(
                    hidden, cache: state.kvCaches[i], offset: state.position)
                hidden = h
                newKVCaches[i] = newCache
            }
        }

        hidden = norm(hidden)

        // Tied LM head
        let logits = embedTokens.asLinear(hidden)

        let newState = InferenceState(
            deltaNetStates: newDeltaStates,
            kvCaches: newKVCaches,
            position: state.position + seqLen)

        return (logits, newState)
    }

    // MARK: - Text Generation

    /// Generate text tokens autoregressively.
    ///
    /// - Parameters:
    ///   - promptIds: Prompt token IDs
    ///   - sampling: Sampling configuration
    /// - Returns: Generated token IDs (excluding prompt)
    public func generate(
        promptIds: [Int],
        sampling: ChatSamplingConfig = .default
    ) -> [Int] {
        var state = InferenceState.initial(config: config)

        // Prefill
        let prompt = MLXArray(promptIds.map { Int32($0) }).expandedDimensions(axis: 0)
        let (prefillLogits, prefillState) = forward(inputIds: prompt, state: state)
        state = prefillState
        eval(prefillLogits)

        // Sample first token
        var token = sampleFromLogits(prefillLogits, at: promptIds.count - 1,
                                     config: sampling, history: promptIds)
        if token == config.eosTokenId { return [] }

        var generated = [token]

        // Decode loop
        for _ in 1..<sampling.maxTokens {
            let input = MLXArray([Int32(token)]).expandedDimensions(axis: 0)
            let (logits, newState) = forward(inputIds: input, state: state)
            state = newState
            eval(logits)

            token = sampleFromLogits(logits, at: 0,
                                     config: sampling, history: promptIds + generated)
            if token == config.eosTokenId { break }
            generated.append(token)
        }

        return generated
    }

    // MARK: - Helpers

    private func sampleFromLogits(
        _ logits: MLXArray, at position: Int,
        config: ChatSamplingConfig, history: [Int]
    ) -> Int {
        let posLogits = logits[0, position]  // [vocabSize]
        let f32 = posLogits.asType(.float32)
        eval(f32)
        let count = self.config.vocabSize
        let floats: [Float] = f32.asArray(Float.self)
        return ChatSampler.sample(logits: Array(floats.prefix(count)), config: config, previousTokens: history)
    }
}
