import Foundation
import MLX
import MLXFast
import MLXNN

// MARK: - Llama3 Scaled RoPE

final class CSMLlama3ScaledRoPE: Module {
    let dims: Int
    private let d2: Int
    let base: Float
    let maxSeqLen: Int
    let scaleFactor: Float
    let lowFreqFactor: Float
    let highFreqFactor: Float
    let oldContextLen: Float

    private var cosF32: MLXArray?
    private var sinF32: MLXArray?
    private var cosByDType: [DType: MLXArray] = [:]
    private var sinByDType: [DType: MLXArray] = [:]
    private var isCacheBuilt = false

    init(dims: Int, maxSeqLen: Int = 2048, base: Float = 500000.0,
         scaleFactor: Float = 32.0, lowFreqFactor: Float = 1.0,
         highFreqFactor: Float = 4.0, oldContextLen: Float = 8192.0) {
        precondition(dims % 2 == 0, "RoPE dim must be even")
        self.dims = dims
        d2 = dims / 2
        self.base = base
        self.maxSeqLen = maxSeqLen
        self.scaleFactor = scaleFactor
        self.lowFreqFactor = lowFreqFactor
        self.highFreqFactor = highFreqFactor
        self.oldContextLen = oldContextLen
        super.init()
        ropeInit()
    }

    convenience init(dims: Int, config: CSMLlamaConfiguration) {
        let base = config.ropeTheta
        let rs = config.ropeScaling
        func num(_ k: String, _ d: Float) -> Float {
            guard let v = rs?[k] else { return d }
            switch v {
            case .float(let x): return x
            case .string(let s): return Float(s) ?? d
            }
        }
        self.init(
            dims: dims, maxSeqLen: config.maxPositionEmbeddings,
            base: base, scaleFactor: num("factor", 32.0),
            lowFreqFactor: num("low_freq_factor", 1.0),
            highFreqFactor: num("high_freq_factor", 4.0),
            oldContextLen: num("original_max_position_embeddings", 8192.0))
    }

    private func ropeInit() {
        let idx = MLXArray(stride(from: 0, to: dims, by: 2).map { Float($0) })
        let exponents = idx / MLXArray(Float(dims))
        let freqs = MLX.pow(MLXArray(base), exponents).asType(.float32)
        let invFreqs = MLXArray(1.0) / freqs
        let theta = applyScaling(freqs: invFreqs)

        let seq = MLXArray((0..<maxSeqLen).map { Float($0) }).reshaped([maxSeqLen, 1])
        let idxTheta = seq * theta.reshaped([1, d2])
        cosF32 = cos(idxTheta)
        sinF32 = sin(idxTheta)
        cosByDType.removeAll()
        sinByDType.removeAll()
        isCacheBuilt = true
    }

    private func applyScaling(freqs: MLXArray) -> MLXArray {
        let twoPi = MLXArray(2.0 * Float.pi)
        let wavelens = twoPi / freqs
        let low = MLXArray(oldContextLen / lowFreqFactor)
        let high = MLXArray(oldContextLen / highFreqFactor)

        var smooth = (MLXArray(oldContextLen) / wavelens - MLXArray(lowFreqFactor))
            / MLXArray(highFreqFactor - lowFreqFactor)
        smooth = MLX.minimum(MLX.maximum(smooth, MLXArray(0.0)), MLXArray(1.0))

        let scaled = freqs / MLXArray(scaleFactor)
        let blended = (MLXArray(1.0) - smooth) * scaled + smooth * freqs

        let condA = wavelens .< high
        let condB = wavelens .> low
        return MLX.where(condA, freqs, MLX.where(condB, scaled, blended)).asType(freqs.dtype)
    }

    private func getCache(dtype: DType, seqLen: Int, offset: Int?) -> (MLXArray, MLXArray) {
        precondition(isCacheBuilt)
        guard let cosF32, let sinF32 else { return (MLXArray(0), MLXArray(0)) }

        let start = max(offset ?? 0, 0)
        let end = start + seqLen
        precondition(end <= maxSeqLen, "RoPE cache length exceeded")

        let cosSrc: MLXArray, sinSrc: MLXArray
        if dtype == .float32 {
            cosSrc = cosF32; sinSrc = sinF32
        } else {
            if cosByDType[dtype] == nil {
                cosByDType[dtype] = cosF32.asType(dtype)
                sinByDType[dtype] = sinF32.asType(dtype)
            }
            cosSrc = cosByDType[dtype]!
            sinSrc = sinByDType[dtype]!
        }

        let cosHead = split(cosSrc, indices: [start], axis: 0)[1]
        let sinHead = split(sinSrc, indices: [start], axis: 0)[1]
        let cosSeg = split(cosHead, indices: [seqLen], axis: 0)[0]
        let sinSeg = split(sinHead, indices: [seqLen], axis: 0)[0]

        return (cosSeg.reshaped([1, seqLen, 1, d2]),
                sinSeg.reshaped([1, seqLen, 1, d2]))
    }

    func callAsFunction(_ x: MLXArray, offset: Int? = nil) -> MLXArray {
        precondition(x.shape.last == dims)

        let seqAxis = (x.ndim == 4) ? 2 : 1
        let seqLen = x.shape[seqAxis]

        let (cosB, sinB) = getCache(dtype: x.dtype, seqLen: seqLen, offset: offset)

        let xShaped = x.reshaped(Array(x.shape.dropLast()) + [d2, 2])

        func splitLast2(_ a: MLXArray) -> (MLXArray, MLXArray) {
            let p = split(a, indices: [1], axis: a.ndim - 1)
            return (p[0], p[1])
        }
        let (xEven, xOdd) = splitLast2(xShaped)

        var ropeShape = [Int](repeating: 1, count: xShaped.ndim - 2)
        ropeShape[seqAxis] = seqLen
        let c = cosB.reshaped(ropeShape + [d2, 1])
        let s = sinB.reshaped(ropeShape + [d2, 1])

        let yEven = xEven * c - xOdd * s
        let yOdd = xOdd * c + xEven * s

        let y = stacked([yEven, yOdd], axis: xShaped.ndim - 1)
        return y.reshaped(x.shape)
    }
}

// MARK: - Attention

final class CSMLlamaAttention: Module {
    let args: CSMLlamaConfiguration
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Module
    @ModuleInfo(key: "k_proj") var kProj: Module
    @ModuleInfo(key: "v_proj") var vProj: Module
    @ModuleInfo(key: "o_proj") var oProj: Module

    let rope: CSMLlama3ScaledRoPE

    init(_ args: CSMLlamaConfiguration, groupSize: Int? = nil, bits: Int? = nil) {
        self.args = args
        let dim = args.hiddenSize
        let heads = args.attentionHeads
        let kvHeads = args.kvHeads
        let headDim = args.resolvedHeadDimensions
        scale = pow(Float(headDim), -0.5)

        _qProj.wrappedValue = makeLinear(dim, heads * headDim, bias: args.attentionBias, groupSize: groupSize, bits: bits)
        _kProj.wrappedValue = makeLinear(dim, kvHeads * headDim, bias: args.attentionBias, groupSize: groupSize, bits: bits)
        _vProj.wrappedValue = makeLinear(dim, kvHeads * headDim, bias: args.attentionBias, groupSize: groupSize, bits: bits)
        _oProj.wrappedValue = makeLinear(heads * headDim, dim, bias: args.attentionBias, groupSize: groupSize, bits: bits)

        rope = CSMLlama3ScaledRoPE(dims: headDim, config: args)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCacheSimple?) -> MLXArray {
        let B = x.shape[0], L = x.shape[1]
        let headDim = args.resolvedHeadDimensions

        var queries = applyLinear(qProj, x)
        var keys = applyLinear(kProj, x)
        var values = applyLinear(vProj, x)

        queries = queries.reshaped([B, L, args.attentionHeads, headDim]).transposed(0, 2, 1, 3)
        keys = keys.reshaped([B, L, args.kvHeads, headDim]).transposed(0, 2, 1, 3)
        values = values.reshaped([B, L, args.kvHeads, headDim]).transposed(0, 2, 1, 3)

        if let cache {
            queries = rope(queries, offset: cache.offset)
            keys = rope(keys, offset: cache.offset)
        } else {
            queries = rope(queries)
            keys = rope(keys)
        }

        if let cache {
            (keys, values) = cache.update(keys: keys, values: values)
        }

        let output = MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values, scale: scale, mask: mask)
            .transposed(0, 2, 1, 3).reshaped([B, L, -1])

        return applyLinear(oProj, output)
    }
}

// MARK: - MLP

final class CSMMLP: Module {
    @ModuleInfo(key: "gate_proj") var gate: Module
    @ModuleInfo(key: "down_proj") var down: Module
    @ModuleInfo(key: "up_proj") var up: Module

    init(_ args: CSMLlamaConfiguration, groupSize: Int? = nil, bits: Int? = nil) {
        _gate.wrappedValue = makeLinear(args.hiddenSize, args.intermediateSize, bias: args.mlpBias, groupSize: groupSize, bits: bits)
        _down.wrappedValue = makeLinear(args.intermediateSize, args.hiddenSize, bias: args.mlpBias, groupSize: groupSize, bits: bits)
        _up.wrappedValue = makeLinear(args.hiddenSize, args.intermediateSize, bias: args.mlpBias, groupSize: groupSize, bits: bits)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let activation = silu(applyLinear(gate, x))
        return applyLinear(down, activation * applyLinear(up, x))
    }
}

// MARK: - Transformer Block

final class CSMTransformerBlock: Module {
    @ModuleInfo(key: "self_attn") var attention: CSMLlamaAttention
    @ModuleInfo(key: "mlp") var mlp: CSMMLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ args: CSMLlamaConfiguration, groupSize: Int? = nil, bits: Int? = nil) {
        _attention.wrappedValue = CSMLlamaAttention(args, groupSize: groupSize, bits: bits)
        _mlp.wrappedValue = CSMMLP(args, groupSize: groupSize, bits: bits)
        _inputLayerNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCacheSimple?) -> MLXArray {
        var r = attention(inputLayerNorm(x), mask: mask, cache: cache)
        let h = x + r
        r = mlp(postAttentionLayerNorm(h))
        return h + r
    }
}

// MARK: - CSMLlamaModel

public final class CSMLlamaModel: Module {
    public let vocabularySize: Int
    let layers: [CSMTransformerBlock]
    let norm: RMSNorm

    public init(_ args: CSMLlamaConfiguration, groupSize: Int? = nil, bits: Int? = nil) {
        precondition(args.vocabularySize > 0)
        vocabularySize = args.vocabularySize
        layers = (0..<args.hiddenLayers).map { _ in CSMTransformerBlock(args, groupSize: groupSize, bits: bits) }
        norm = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCacheSimple]?) -> MLXArray {
        var h = inputs
        let mask = createAttentionMask(h: h, cache: cache)
        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i])
        }
        return norm(h)
    }

    public func makeCache() -> [KVCacheSimple] {
        (0..<layers.count).map { _ in KVCacheSimple() }
    }
}
