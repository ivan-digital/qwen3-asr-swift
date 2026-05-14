import Foundation
import MLX
import MLXFast
import MLXNN
import MLXRandom
import MLXCommon

// MARK: - RMSNorm

public final class RMSNorm: Module {
    @ParameterInfo public var weight: MLXArray
    public let eps: Float

    public init(dimensions: Int, eps: Float = 1e-6) {
        voxCPM2InitLog("RMSNorm start (\(dimensions))")
        self._weight = ParameterInfo(
            wrappedValue: MLXArray(Array(repeating: Float(1.0), count: dimensions), [dimensions])
        )
        self.eps = eps
        super.init()
        voxCPM2InitLog("RMSNorm done (\(dimensions))")
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let originalDType = x.dtype
        let xFloat = x.asType(.float32)
        let variance = mean(xFloat * xFloat, axis: xFloat.ndim - 1, keepDims: true)
        let normalized = (xFloat * rsqrt(variance + MLXArray(eps))).asType(originalDType)
        return normalized * weight.asType(originalDType)
    }
}

// MARK: - Rotary Embedding

public final class MiniCPMLongRoPE: Module {
    @ParameterInfo var invFreq: MLXArray
    private let scalingFactor: Float
    @ParameterInfo var shortFactor: MLXArray
    @ParameterInfo var longFactor: MLXArray
    private let originalMaxPositionEmbeddings: Int
    private let useLongFactor: Bool

    public init(config: LMConfig) {
        let headDim = config.kvChannels ?? (config.hiddenSize / config.numAttentionHeads)
        let halfDim = headDim / 2

        let exponents = MLXArray(0..<Int32(halfDim)).asType(.float32)
            / MLXArray(Float(halfDim))
        self._invFreq = ParameterInfo(
            wrappedValue: exp(exponents * (-log(MLXArray(Float(config.ropeTheta))))),
            key: "inv_freq"
        )

        let ropeScaling = config.ropeScaling ?? RopeScalingConfig()
        let short = ropeScaling.shortFactor.isEmpty
            ? Array(repeating: 1.0, count: halfDim)
            : ropeScaling.shortFactor
        let long = ropeScaling.longFactor.isEmpty
            ? Array(repeating: 1.0, count: halfDim)
            : ropeScaling.longFactor
        self._shortFactor = ParameterInfo(wrappedValue: MLXArray(short).asType(.float32), key: "short_factor")
        self._longFactor = ParameterInfo(wrappedValue: MLXArray(long).asType(.float32), key: "long_factor")
        self.originalMaxPositionEmbeddings = max(1, ropeScaling.originalMaxPositionEmbeddings)
        self.useLongFactor = max(1, config.maxPositionEmbeddings) > self.originalMaxPositionEmbeddings

        let scale = Double(max(config.maxPositionEmbeddings, 1))
            / Double(max(config.originalMaxPositionEmbeddings, 1))
        self.scalingFactor = Float(sqrt(1.0 + log(max(scale, 1.0))
            / log(Double(max(config.originalMaxPositionEmbeddings, 2)))))
        super.init()
    }

    public func callAsFunction(_ positionIds: MLXArray) -> (MLXArray, MLXArray) {
        let seqLen = Int(positionIds.max().item(Int32.self)) + 1
        // Upstream VoxCPM2 precomputes the RoPE cache once using the
        // max_position_embeddings-backed factor choice, so the effective
        // factor is fixed for a given checkpoint rather than varying per call.
        let factors = useLongFactor ? longFactor : shortFactor

        let t = MLXArray(0..<Int32(seqLen)).asType(.float32)
        let freqs = (t.expandedDimensions(axis: 1)
            * (1.0 / factors.expandedDimensions(axis: 0)))
            * invFreq.expandedDimensions(axis: 0)
        let emb = concatenated([freqs, freqs], axis: -1)

        let cos = cos(emb) * MLXArray(scalingFactor)
        let sin = sin(emb) * MLXArray(scalingFactor)
        return (cos[positionIds], sin[positionIds])
    }
}

@inline(__always)
private func rotateHalf(_ x: MLXArray) -> MLXArray {
    let half = x.shape.last! / 2
    let parts = split(x, indices: [half], axis: x.ndim - 1)
    return concatenated([parts[1] * -1, parts[0]], axis: x.ndim - 1)
}

@inline(__always)
private func applyRotaryPosEmb(
    q: MLXArray, k: MLXArray, cos: MLXArray, sin: MLXArray
) -> (MLXArray, MLXArray) {
    // q / k arrive as (B, L, H, D); cos / sin as (L, D). Expand to
    // (1, L, 1, D) so they broadcast across batch and heads.
    let cosExpanded = cos.expandedDimensions(axis: 0).expandedDimensions(axis: 2)
    let sinExpanded = sin.expandedDimensions(axis: 0).expandedDimensions(axis: 2)
    let originalDType = q.dtype
    let q = q.asType(.float32)
    let k = k.asType(.float32)
    let qEmbed = (q * cosExpanded) + (rotateHalf(q) * sinExpanded)
    let kEmbed = (k * cosExpanded) + (rotateHalf(k) * sinExpanded)
    return (qEmbed.asType(originalDType), kEmbed.asType(originalDType))
}

@inline(__always)
private func debugRange(_ label: String, _ x: MLXArray) {
    let values = x.asArray(Float.self)
    let minValue = values.min() ?? 0
    let maxValue = values.max() ?? 0
    let message = "  \(label) range: min=\(minValue), max=\(maxValue), count=\(values.count)\n"
    if let data = message.data(using: .utf8) {
        FileHandle.standardOutput.write(data)
    }
}

@inline(__always)
private func voxCPM2InitDebugEnabled() -> Bool {
    ProcessInfo.processInfo.environment["VOXCPM2_DEBUG_INIT"] == "1"
}

@inline(__always)
private func voxCPM2InitLog(_ message: String) {
    guard voxCPM2InitDebugEnabled() else { return }
    let line = "  VoxCPM2 init: \(message)\n"
    if let data = line.data(using: .utf8) {
        FileHandle.standardOutput.write(data)
    }
}

@inline(__always)
func zeroLinear(_ inputDimensions: Int, _ outputDimensions: Int, bias: Bool = true) -> Linear {
    // Use Linear's standard init (random weights) — the actual weights are
    // overwritten by the safetensors load. Earlier we tried zero-initialising
    // via the designated `init(weight:bias:)` to save memory, but update
    // (parameters:) silently failed to overwrite those arrays, leaving every
    // Linear at exactly zero post-load. The random-init path works because
    // MLX correctly walks the items() reflection in that case.
    return Linear(inputDimensions, outputDimensions, bias: bias)
}

@inline(__always)
func makeUnifiedCFMTimeSpan(timesteps: Int, scheduler _: String, sigmaMin _: Float) -> [Float] {
    // Match upstream VoxCPM2 inference:
    //   t_span = linspace(1, 0, n_timesteps + 1)
    //   t_span = t_span + sway_sampling_coef * (cos(pi/2 * t_span) - 1 + t_span)
    // The upstream default sway_sampling_coef is 1.0.
    let steps = max(timesteps, 1)
    let swaySamplingCoef = Double(1.0)

    return (0...steps).map { step in
        let progress = Double(step) / Double(steps)
        let t = 1.0 - progress
        let shaped = t + swaySamplingCoef * (cos(Double.pi / 2.0 * t) - 1.0 + t)
        return Float(shaped)
    }
}

@inline(__always)
private func clampUnitInterval(_ value: Double, epsilon: Double = 1e-7) -> Double {
    min(max(value, epsilon), 1.0 - epsilon)
}

@inline(__always)
private func inverseStandardNormalCDF(_ p: Double) -> Double {
    precondition(p > 0.0 && p < 1.0)

    // Peter J. Acklam's rational approximation.
    let a1 = -3.969683028665376e+01
    let a2 = 2.209460984245205e+02
    let a3 = -2.759285104469687e+02
    let a4 = 1.383577518672690e+02
    let a5 = -3.066479806614716e+01
    let a6 = 2.506628277459239e+00

    let b1 = -5.447609879822406e+01
    let b2 = 1.615858368580409e+02
    let b3 = -1.556989798598866e+02
    let b4 = 6.680131188771972e+01
    let b5 = -1.328068155288572e+01

    let c1 = -7.784894002430293e-03
    let c2 = -3.223964580411365e-01
    let c3 = -2.400758277161838e+00
    let c4 = -2.549732539343734e+00
    let c5 = 4.374664141464968e+00
    let c6 = 2.938163982698783e+00

    let d1 = 7.784695709041462e-03
    let d2 = 3.224671290700398e-01
    let d3 = 2.445134137142996e+00
    let d4 = 3.754408661907416e+00

    let plow = 0.02425
    let phigh = 1.0 - plow

    if p < plow {
        let q = sqrt(-2.0 * log(p))
        return (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
            ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
    }

    if p > phigh {
        let q = sqrt(-2.0 * log(1.0 - p))
        return -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
            ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
    }

    let q = p - 0.5
    let r = q * q
    return (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q /
        (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
}

// MARK: - Attention / MLP

public final class MiniCPMAttention: Module {
    public let numHeads: Int
    public let numKVHeads: Int
    public let headDim: Int
    public let scale: Float

    @ModuleInfo(key: "q_proj") public var qProj: Linear
    @ModuleInfo(key: "k_proj") public var kProj: Linear
    @ModuleInfo(key: "v_proj") public var vProj: Linear
    @ModuleInfo(key: "o_proj") public var oProj: Linear
    public init(config: LMConfig) {
        voxCPM2InitLog("Attention start")
        self.numHeads = config.numAttentionHeads
        self.numKVHeads = config.numKeyValueHeads
        self.headDim = config.kvChannels ?? (config.hiddenSize / config.numAttentionHeads)
        self.scale = 1.0 / sqrt(Float(headDim))

        let qDim = numHeads * headDim
        let kvDim = numKVHeads * headDim

        self._qProj = ModuleInfo(wrappedValue: zeroLinear(config.hiddenSize, qDim, bias: false), key: "q_proj")
        voxCPM2InitLog("Attention q_proj")
        self._kProj = ModuleInfo(wrappedValue: zeroLinear(config.hiddenSize, kvDim, bias: false), key: "k_proj")
        voxCPM2InitLog("Attention k_proj")
        self._vProj = ModuleInfo(wrappedValue: zeroLinear(config.hiddenSize, kvDim, bias: false), key: "v_proj")
        voxCPM2InitLog("Attention v_proj")
        self._oProj = ModuleInfo(wrappedValue: zeroLinear(qDim, config.hiddenSize, bias: false), key: "o_proj")
        voxCPM2InitLog("Attention o_proj")

        super.init()
        voxCPM2InitLog("Attention done")
    }

    public func callAsFunction(
        _ hiddenStates: MLXArray,
        rope: MiniCPMLongRoPE?,
        cache: (MLXArray, MLXArray)? = nil,
        isCausal: Bool = true
    ) -> (MLXArray, (MLXArray, MLXArray)) {
        let batch = hiddenStates.dim(0)
        let seqLen = hiddenStates.dim(1)

        var queries = qProj(hiddenStates)
        var keys = kProj(hiddenStates)
        var values = vProj(hiddenStates)

        queries = queries.reshaped(batch, seqLen, numHeads, headDim)
        keys = keys.reshaped(batch, seqLen, numKVHeads, headDim)
        values = values.reshaped(batch, seqLen, numKVHeads, headDim)

        // RoPE is applied while q/k still have layout (B, L, H, D) — the cos/sin
        // tables broadcast as (1, L, 1, D) and need the seq dim at axis 1, not
        // axis 2. Apply the rotary embedding first, then transpose to head-first.
        if let rope {
            let offset = cache?.0.dim(2) ?? 0
            let positionIds = MLXArray(0..<Int32(seqLen)) + Int32(offset)
            let (cos, sin) = rope(positionIds)
            (queries, keys) = applyRotaryPosEmb(q: queries, k: keys, cos: cos, sin: sin)
        }

        queries = queries.transposed(0, 2, 1, 3)
        keys = keys.transposed(0, 2, 1, 3)
        values = values.transposed(0, 2, 1, 3)

        // Mirror the upstream MPS safeguard: SDPA behaves more reliably when
        // q/k/v are materialized into contiguous buffers after the transpose.
        queries = queries.contiguous()
        keys = keys.contiguous()
        values = values.contiguous()

        var cachedKeys = keys
        var cachedValues = values
        if let (prevKeys, prevValues) = cache {
            cachedKeys = concatenated([prevKeys, keys], axis: 2)
            cachedValues = concatenated([prevValues, values], axis: 2)
        }

        let mask: MLXFast.ScaledDotProductAttentionMaskMode
        if !isCausal {
            mask = .none
        } else if seqLen <= 1 && cache == nil {
            mask = .none
        } else {
            let kvLen = cachedKeys.dim(2)
            let pastLen = kvLen - seqLen
            let causal = MLXArray.tri(seqLen, m: kvLen, k: pastLen, type: Float.self)
            let additiveMask = (1 - causal) * -Float.greatestFiniteMagnitude
            mask = .array(additiveMask.reshaped(1, 1, seqLen, kvLen).asType(hiddenStates.dtype))
        }

        let attn = SDPA.attendAndMerge(
            qHeads: queries, kHeads: cachedKeys, vHeads: cachedValues,
            scale: scale, mask: mask)
        let output = oProj(attn)
        return (output, (cachedKeys, cachedValues))
    }
}

public final class MiniCPMMLP: Module {
    @ModuleInfo(key: "gate_proj") public var gateProj: Linear
    @ModuleInfo(key: "up_proj") public var upProj: Linear
    @ModuleInfo(key: "down_proj") public var downProj: Linear

    public init(config: LMConfig) {
        self._gateProj = ModuleInfo(wrappedValue: zeroLinear(config.hiddenSize, config.intermediateSize, bias: false), key: "gate_proj")
        self._upProj = ModuleInfo(wrappedValue: zeroLinear(config.hiddenSize, config.intermediateSize, bias: false), key: "up_proj")
        self._downProj = ModuleInfo(wrappedValue: zeroLinear(config.intermediateSize, config.hiddenSize, bias: false), key: "down_proj")
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

public final class MiniCPMDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") public var selfAttn: MiniCPMAttention
    @ModuleInfo public var mlp: MiniCPMMLP
    @ModuleInfo(key: "input_layernorm") public var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") public var postAttentionLayerNorm: RMSNorm
    public let scaleDepth: Float
    public let useMup: Bool
    public let numHiddenLayers: Int

    public init(config: LMConfig) {
        voxCPM2InitLog("DecoderLayer start")
        self._selfAttn = ModuleInfo(wrappedValue: MiniCPMAttention(config: config), key: "self_attn")
        voxCPM2InitLog("DecoderLayer self_attn")
        self._mlp = ModuleInfo(wrappedValue: MiniCPMMLP(config: config))
        voxCPM2InitLog("DecoderLayer mlp")
        self._inputLayerNorm = ModuleInfo(wrappedValue: RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps), key: "input_layernorm")
        voxCPM2InitLog("DecoderLayer input_layernorm")
        self._postAttentionLayerNorm = ModuleInfo(wrappedValue: RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps), key: "post_attention_layernorm")
        voxCPM2InitLog("DecoderLayer post_attention_layernorm")
        self.scaleDepth = config.scaleDepth
        self.useMup = config.useMup
        self.numHiddenLayers = config.numHiddenLayers
        super.init()
        voxCPM2InitLog("DecoderLayer done")
    }

    public func callAsFunction(
        _ x: MLXArray,
        rope: MiniCPMLongRoPE?,
        cache: (MLXArray, MLXArray)? = nil,
        isCausal: Bool = true,
        debugLabel: String? = nil
    ) -> (MLXArray, (MLXArray, MLXArray)) {
        let debugStatsEnabled = ProcessInfo.processInfo.environment["VOXCPM2_DEBUG_VERBOSE"] == "1"
        if debugStatsEnabled, let debugLabel {
            debugRange("\(debugLabel) input", x)
        }

        let residual = x
        let normed = inputLayerNorm(x)
        if debugStatsEnabled, let debugLabel {
            debugRange("\(debugLabel) normed", normed)
        }
        let (attnOut, newCache) = selfAttn(normed, rope: rope, cache: cache, isCausal: isCausal)
        if debugStatsEnabled, let debugLabel {
            debugRange("\(debugLabel) attnOut", attnOut)
        }
        let residualScale = useMup ? (scaleDepth / sqrt(Float(numHiddenLayers))) : 1.0

        var h = residual + (attnOut * MLXArray(residualScale))
        let mlpOut = mlp(postAttentionLayerNorm(h)) * MLXArray(residualScale)
        if debugStatsEnabled, let debugLabel {
            debugRange("\(debugLabel) mlpOut", mlpOut)
        }
        h = h + mlpOut
        if debugStatsEnabled, let debugLabel {
            debugRange("\(debugLabel) output", h)
        }
        return (h, newCache)
    }
}

public final class MiniCPMModel: Module {
    public let config: LMConfig

    @ModuleInfo(key: "embed_tokens") public var embedTokens: Embedding?
    @ModuleInfo public var layers: [MiniCPMDecoderLayer]
    @ModuleInfo public var norm: RMSNorm
    @ModuleInfo public var rope: MiniCPMLongRoPE?

    public init(_ config: LMConfig) {
        voxCPM2InitLog("MiniCPMModel start")
        self.config = config
        let embeddingCount = max(config.vocabSize, 1)
        self._embedTokens = ModuleInfo(
            wrappedValue: Embedding(embeddingCount: embeddingCount, dimensions: config.hiddenSize),
            key: "embed_tokens"
        )
        voxCPM2InitLog("MiniCPMModel embed_tokens")
        self._layers = ModuleInfo(wrappedValue: (0..<config.numHiddenLayers).map { _ in MiniCPMDecoderLayer(config: config) })
        voxCPM2InitLog("MiniCPMModel layers")
        self._norm = ModuleInfo(wrappedValue: RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps))
        voxCPM2InitLog("MiniCPMModel norm")
        self._rope = ModuleInfo(wrappedValue: config.noRope ? nil : MiniCPMLongRoPE(config: config), key: "rope")
        voxCPM2InitLog("MiniCPMModel rope")
        super.init()
        voxCPM2InitLog("MiniCPMModel done")
    }

    public func callAsFunction(
        inputsEmbeds: MLXArray? = nil,
        inputIds: MLXArray? = nil,
        mask: MLXArray? = nil,
        cache: [(MLXArray, MLXArray)]? = nil,
        isCausal: Bool = true
    ) -> (MLXArray, [(MLXArray, MLXArray)]) {
        let hiddenStates: MLXArray
        if let inputsEmbeds {
            hiddenStates = inputsEmbeds
        } else if let inputIds {
            guard let embedTokens else {
                fatalError("MiniCPMModel called with inputIds but no embed_tokens layer")
            }
            hiddenStates = embedTokens(inputIds)
        } else {
            fatalError("MiniCPMModel requires inputsEmbeds or inputIds")
        }

        let rope = self.rope

        var h = hiddenStates
        var newCaches: [(MLXArray, MLXArray)] = []
        newCaches.reserveCapacity(layers.count)
        let debugStatsEnabled = ProcessInfo.processInfo.environment["VOXCPM2_DEBUG_VERBOSE"] == "1"

        for (idx, layer) in layers.enumerated() {
            let layerCache = cache?[idx]
            let debugLabel = debugStatsEnabled ? "MiniCPM layer \(idx)" : nil
            let (nextH, nextCache) = layer(
                h,
                rope: rope,
                cache: layerCache,
                isCausal: isCausal,
                debugLabel: debugLabel
            )
            h = nextH
            newCaches.append(nextCache)
        }

        h = norm(h)
        if debugStatsEnabled {
            debugRange("MiniCPM final norm", h)
        }
        return (h, newCaches)
    }
}

// MARK: - VoxCPM Local Encoder

public final class VoxCPMLocEnc: Module {
    public let config: LMConfig

    @ParameterInfo(key: "special_token") public var specialToken: MLXArray
    @ModuleInfo(key: "in_proj") var inProj: Linear
    @ModuleInfo public var encoder: MiniCPMModel

    public init(config: LMConfig, inputDim: Int = 64) {
        self.config = config
        self._specialToken = ParameterInfo(
            wrappedValue: MLXArray(
                Array(repeating: Float(0.0), count: config.hiddenSize),
                [1, 1, 1, config.hiddenSize]
            ),
            key: "special_token"
        )
        self._inProj = ModuleInfo(wrappedValue: zeroLinear(inputDim, config.hiddenSize, bias: true), key: "in_proj")
        self._encoder = ModuleInfo(wrappedValue: MiniCPMModel(config))
        super.init()
    }

    public func loadSpecialToken(_ token: MLXArray) {
        self.update(parameters: ModuleParameters(values: ["special_token": .value(token)]))
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let batch = x.dim(0)
        let steps = x.dim(1)
        let patches = x.dim(2)
        let debugEnabled = voxCPM2InitDebugEnabled()
        if debugEnabled {
            voxCPM2InitLog("LocEnc input shape=\(x.shape) dtype=\(x.dtype)")
        }

        var h = inProj(x)
        if debugEnabled {
            voxCPM2InitLog("LocEnc inProj output shape=\(h.shape) dtype=\(h.dtype)")
            voxCPM2InitLog("LocEnc specialToken shape=\(specialToken.shape) dtype=\(specialToken.dtype)")
        }
        let special = repeated(repeated(specialToken.asType(h.dtype), count: batch, axis: 0), count: steps, axis: 1)
        if debugEnabled {
            voxCPM2InitLog("LocEnc special repeated shape=\(special.shape) dtype=\(special.dtype)")
        }
        h = concatenated([special, h], axis: 2)
        if debugEnabled {
            voxCPM2InitLog("LocEnc concat shape=\(h.shape) dtype=\(h.dtype)")
        }
        h = h.reshaped(batch * steps, patches + 1, -1)
        if debugEnabled {
            voxCPM2InitLog("LocEnc reshaped shape=\(h.shape) dtype=\(h.dtype)")
        }

        let (outputs, _) = encoder(inputsEmbeds: h, isCausal: false)
        if debugEnabled {
            voxCPM2InitLog("LocEnc encoder output shape=\(outputs.shape) dtype=\(outputs.dtype)")
        }
        let cls = outputs[0..., 0..<1, 0...].squeezed(axis: 1)
        return cls.reshaped(batch, steps, -1)
    }
}

// MARK: - VoxCPM DiT

public final class SinusoidalPosEmb: Module {
    public let dim: Int

    public init(dim: Int) {
        precondition(dim % 2 == 0)
        self.dim = dim
        super.init()
    }

    public func callAsFunction(_ x: MLXArray, scale: Float = 1000) -> MLXArray {
        let values = x.shape.isEmpty ? x.reshaped(1) : x.asType(.float32)
        let half = dim / 2
        let embScale = log(MLXArray(10000.0)) / MLXArray(Float(half - 1))
        let freq = exp(MLXArray(0..<Int32(half)).asType(.float32) * (-embScale))
        let emb = MLXArray(scale) * values.reshaped(-1, 1) * freq.reshaped(1, -1)
        return concatenated([sin(emb), cos(emb)], axis: -1)
    }
}

public final class TimestepEmbedding: Module {
    @ModuleInfo(key: "linear_1") public var linear_1: Linear
    @ModuleInfo(key: "linear_2") public var linear_2: Linear

    public init(inChannels: Int, timeEmbedDim: Int, outDim: Int? = nil) {
        self._linear_1 = ModuleInfo(wrappedValue: zeroLinear(inChannels, timeEmbedDim), key: "linear_1")
        self._linear_2 = ModuleInfo(wrappedValue: zeroLinear(timeEmbedDim, outDim ?? timeEmbedDim), key: "linear_2")
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        linear_2(silu(linear_1(x)))
    }
}

public final class VoxCPMLocDiTV2: Module {
    public let config: LMConfig
    public let inChannels: Int

    @ModuleInfo(key: "in_proj") var inProj: Linear
    @ModuleInfo(key: "cond_proj") var condProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear
    @ModuleInfo var decoder: MiniCPMModel

    public let timeEmbeddings: SinusoidalPosEmb
    @ModuleInfo(key: "time_mlp") public var timeMlp: TimestepEmbedding
    @ModuleInfo(key: "delta_time_mlp") public var deltaTimeMlp: TimestepEmbedding

    public init(config: LMConfig, inChannels: Int = 64) {
        self.config = config
        self.inChannels = inChannels

        self._inProj = ModuleInfo(wrappedValue: zeroLinear(inChannels, config.hiddenSize), key: "in_proj")
        self._condProj = ModuleInfo(wrappedValue: zeroLinear(inChannels, config.hiddenSize), key: "cond_proj")
        self._outProj = ModuleInfo(wrappedValue: zeroLinear(config.hiddenSize, inChannels), key: "out_proj")
        self._decoder = ModuleInfo(wrappedValue: MiniCPMModel(config))
        self.timeEmbeddings = SinusoidalPosEmb(dim: config.hiddenSize)
        self._timeMlp = ModuleInfo(
            wrappedValue: TimestepEmbedding(inChannels: config.hiddenSize, timeEmbedDim: config.hiddenSize),
            key: "time_mlp"
        )
        self._deltaTimeMlp = ModuleInfo(
            wrappedValue: TimestepEmbedding(inChannels: config.hiddenSize, timeEmbedDim: config.hiddenSize),
            key: "delta_time_mlp"
        )

        super.init()
    }

    public func callAsFunction(
        _ x: MLXArray,
        mu: MLXArray,
        t: MLXArray,
        cond: MLXArray,
        dt: MLXArray
    ) -> MLXArray {
        let debugStatsEnabled = ProcessInfo.processInfo.environment["VOXCPM2_DEBUG_VERBOSE"] == "1"

        let xProj = inProj(x.transposed(0, 2, 1).contiguous())
        let condProj = condProj(cond.transposed(0, 2, 1).contiguous())
        let prefix = condProj.dim(1)

        let modelDType = xProj.dtype
        let tEmb = timeMlp(timeEmbeddings(t).asType(modelDType)).asType(modelDType)
        let dtEmb = deltaTimeMlp(timeEmbeddings(dt).asType(modelDType)).asType(modelDType)
        let timeToken = (tEmb + dtEmb).expandedDimensions(axis: 1)

        let hiddenDim = xProj.dim(2)
        let muTokens = mu.asType(modelDType).reshaped(mu.dim(0), -1, hiddenDim)

        let hidden = concatenated([muTokens, timeToken, condProj, xProj], axis: 1)
        if debugStatsEnabled {
            debugRange("LocDiT xProj", xProj)
            debugRange("LocDiT condProj", condProj)
            debugRange("LocDiT tEmb", tEmb)
            debugRange("LocDiT dtEmb", dtEmb)
            debugRange("LocDiT timeToken", timeToken)
            debugRange("LocDiT muTokens", muTokens)
            debugRange("LocDiT hidden", hidden)
        }
        let (decoded, _) = decoder(inputsEmbeds: hidden, isCausal: false)
        let trimmed = decoded[0..., (muTokens.dim(1) + 1 + prefix)..., 0...]
        let projected = outProj(trimmed)
        if debugStatsEnabled {
            debugRange("LocDiT decoded", decoded)
            debugRange("LocDiT trimmed", trimmed)
            debugRange("LocDiT projected", projected)
        }
        return projected.transposed(0, 2, 1).contiguous()
    }
}

public final class UnifiedCFM: Module {
    public let inChannels: Int
    public let cfmParams: CFMConfig
    public let meanMode: Bool

    @ModuleInfo public var estimator: VoxCPMLocDiTV2

    public init(
        inChannels: Int,
        cfmParams: CFMConfig,
        estimator: VoxCPMLocDiTV2,
        meanMode: Bool = false
    ) {
        self.inChannels = inChannels
        self.cfmParams = cfmParams
        self.meanMode = meanMode
        self._estimator = ModuleInfo(wrappedValue: estimator)
        super.init()
    }

    public func solveEuler(
        _ x: MLXArray,
        tSpan: [Float],
        mu: MLXArray,
        cond: MLXArray,
        cfgValue: Float = 1.0,
        useCfgZeroStar: Bool = true
    ) -> MLXArray {
        guard tSpan.count >= 2 else { return x }

        var currentX = x
        var t = tSpan[0]
        var dt = tSpan[0] - tSpan[1]
        let zeroInitSteps = max(1, Int(Double(tSpan.count) * 0.04))

        for step in 1..<tSpan.count {
            let dphiDt: MLXArray
            if useCfgZeroStar && step <= zeroInitSteps {
                dphiDt = MLXArray.zeros(currentX.shape, dtype: currentX.dtype, stream: .cpu)
            } else {
                let batch = currentX.dim(0)
                let xIn = concatenated([currentX, currentX], axis: 0)
                let muIn = concatenated([mu, MLXArray.zeros(mu.shape, dtype: mu.dtype, stream: .cpu)], axis: 0)
                let tVal = MLXArray(Array(repeating: t, count: batch * 2)).reshaped(batch * 2)
                let dtVal = meanMode
                    ? MLXArray(Array(repeating: dt, count: batch * 2)).reshaped(batch * 2)
                    : MLXArray.zeros([batch * 2], dtype: currentX.dtype, stream: .cpu)
                let condIn = concatenated([cond, cond], axis: 0)

                let out = estimator(xIn, mu: muIn, t: tVal, cond: condIn, dt: dtVal)
                let positive = out[0..<batch, 0..., 0...]
                let negative = out[batch..<(batch * 2), 0..., 0...]

                if useCfgZeroStar {
                    let positiveFlat = positive.reshaped(batch, -1)
                    let negativeFlat = negative.reshaped(batch, -1)
                    let dot = (positiveFlat * negativeFlat).sum(axis: 1).reshaped(batch, 1, 1)
                    let sqNorm = ((negativeFlat * negativeFlat).sum(axis: 1) + MLXArray(1e-8)).reshaped(batch, 1, 1)
                    let stStar = dot / sqNorm
                    dphiDt = negative * stStar + MLXArray(cfgValue) * (positive - negative * stStar)
                } else {
                    dphiDt = negative + MLXArray(cfgValue) * (positive - negative)
                }
            }

            currentX = currentX - MLXArray(dt) * dphiDt
            t = tSpan[step]
            if step < tSpan.count - 1 {
                dt = tSpan[step] - tSpan[step + 1]
            }
        }

        return currentX
    }

    public func sample(
        mu: MLXArray,
        nTimesteps: Int,
        patchSize: Int,
        cond: MLXArray,
        temperature: Float = 1.0,
        cfgValue: Float? = nil
    ) -> MLXArray {
        let batch = mu.dim(0)
        let z = MLXRandom.normal([batch, inChannels, patchSize], dtype: mu.dtype) * MLXArray(temperature)
        let tSpan = makeUnifiedCFMTimeSpan(
            timesteps: nTimesteps,
            scheduler: cfmParams.tScheduler,
            sigmaMin: cfmParams.sigmaMin
        )
        return solveEuler(z, tSpan: tSpan, mu: mu, cond: cond, cfgValue: cfgValue ?? cfmParams.inferenceCfgRate)
    }
}
