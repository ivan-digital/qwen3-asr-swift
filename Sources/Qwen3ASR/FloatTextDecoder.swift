import Foundation
import MLX
import MLXNN
import MLXFast
import MLXCommon
import AudioCommon

/// Protocol abstracting the text decoder for the ForcedAligner,
/// allowing both quantized and float (bf16) implementations.
public protocol ForcedAlignerTextDecoding: AnyObject {
    func embeddings(for inputIds: MLXArray) -> MLXArray
    func decode(
        inputsEmbeds: MLXArray,
        attentionMask: MLXArray?,
        cache: [(MLXArray, MLXArray)]?
    ) -> (MLXArray, [(MLXArray, MLXArray)])
}

extension QuantizedTextModel: ForcedAlignerTextDecoding {
    public func embeddings(for inputIds: MLXArray) -> MLXArray {
        embedTokens(inputIds)
    }

    public func decode(
        inputsEmbeds: MLXArray,
        attentionMask: MLXArray?,
        cache: [(MLXArray, MLXArray)]?
    ) -> (MLXArray, [(MLXArray, MLXArray)]) {
        self(inputIds: nil, inputsEmbeds: inputsEmbeds, attentionMask: attentionMask, cache: cache)
    }
}

// MARK: - Float (non-quantized) text decoder

public class FloatTextAttention: Module {
    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo var qProj: Linear
    @ModuleInfo var kProj: Linear
    @ModuleInfo var vProj: Linear
    @ModuleInfo var oProj: Linear
    @ModuleInfo var qNorm: RMSNorm
    @ModuleInfo var kNorm: RMSNorm

    let rope: MLXNN.RoPE

    public init(config: TextDecoderConfig) {
        self.numHeads = config.numHeads
        self.numKVHeads = config.numKVHeads
        self.headDim = config.headDim
        self.scale = 1.0 / sqrt(Float(headDim))

        let hiddenSize = config.hiddenSize

        self._qProj.wrappedValue = Linear(hiddenSize, numHeads * headDim, bias: false)
        self._kProj.wrappedValue = Linear(hiddenSize, numKVHeads * headDim, bias: false)
        self._vProj.wrappedValue = Linear(hiddenSize, numKVHeads * headDim, bias: false)
        self._oProj.wrappedValue = Linear(numHeads * headDim, hiddenSize, bias: false)

        self._qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        self._kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)

        self.rope = MLXNN.RoPE(dimensions: headDim, traditional: false, base: config.ropeTheta)
        super.init()
    }

    public func callAsFunction(
        _ hiddenStates: MLXArray,
        attentionMask: MLXArray? = nil,
        cache: (MLXArray, MLXArray)? = nil
    ) -> (MLXArray, (MLXArray, MLXArray)) {
        let (batch, seqLen, _) = (hiddenStates.dim(0), hiddenStates.dim(1), hiddenStates.dim(2))

        var queries = qProj(hiddenStates)
        var keys = kProj(hiddenStates)
        var values = vProj(hiddenStates)

        queries = queries.reshaped(batch, seqLen, numHeads, headDim)
        keys = keys.reshaped(batch, seqLen, numKVHeads, headDim)
        values = values.reshaped(batch, seqLen, numKVHeads, headDim)

        queries = qNorm(queries)
        keys = kNorm(keys)

        queries = queries.transposed(0, 2, 1, 3)
        keys = keys.transposed(0, 2, 1, 3)
        values = values.transposed(0, 2, 1, 3)

        let offset = cache?.0.dim(2) ?? 0
        queries = rope(queries, offset: offset)
        keys = rope(keys, offset: offset)

        var cachedKeys = keys
        var cachedValues = values

        if let (prevKeys, prevValues) = cache {
            cachedKeys = concatenated([prevKeys, keys], axis: 2)
            cachedValues = concatenated([prevValues, values], axis: 2)
        }

        let merged = SDPA.attendAndMerge(
            qHeads: queries, kHeads: cachedKeys, vHeads: cachedValues,
            scale: scale, mask: attentionMask)
        let output = oProj(merged)
        return (output, (cachedKeys, cachedValues))
    }
}

public class FloatTextMLP: Module {
    @ModuleInfo var gateProj: Linear
    @ModuleInfo var upProj: Linear
    @ModuleInfo var downProj: Linear

    public init(config: TextDecoderConfig) {
        let hiddenSize = config.hiddenSize
        let intermediateSize = config.intermediateSize

        self._gateProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        self._upProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        self._downProj.wrappedValue = Linear(intermediateSize, hiddenSize, bias: false)

        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let gate = silu(gateProj(x))
        let up = upProj(x)
        return downProj(gate * up)
    }
}

public class FloatTextDecoderLayer: Module {
    @ModuleInfo var selfAttn: FloatTextAttention
    @ModuleInfo var mlp: FloatTextMLP
    @ModuleInfo var inputLayerNorm: RMSNorm
    @ModuleInfo var postAttentionLayerNorm: RMSNorm

    public init(config: TextDecoderConfig) {
        self._selfAttn.wrappedValue = FloatTextAttention(config: config)
        self._mlp.wrappedValue = FloatTextMLP(config: config)
        self._inputLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        super.init()
    }

    public func callAsFunction(
        _ hiddenStates: MLXArray,
        attentionMask: MLXArray? = nil,
        cache: (MLXArray, MLXArray)? = nil
    ) -> (MLXArray, (MLXArray, MLXArray)) {
        let residual = hiddenStates
        var hidden = inputLayerNorm(hiddenStates)
        let (attnOutput, newCache) = selfAttn(hidden, attentionMask: attentionMask, cache: cache)
        hidden = residual + attnOutput

        let residual2 = hidden
        hidden = postAttentionLayerNorm(hidden)
        hidden = mlp(hidden)
        hidden = residual2 + hidden

        return (hidden, newCache)
    }
}

public class FloatTextModel: Module {
    public let config: TextDecoderConfig

    @ModuleInfo public var embedTokens: Embedding
    @ModuleInfo var layers: [FloatTextDecoderLayer]
    @ModuleInfo var norm: RMSNorm

    public init(config: TextDecoderConfig) {
        self.config = config
        self._embedTokens.wrappedValue = Embedding(embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        self._layers.wrappedValue = (0..<config.numLayers).map { _ in FloatTextDecoderLayer(config: config) }
        self._norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        super.init()
    }

    public func callAsFunction(
        inputIds: MLXArray? = nil,
        inputsEmbeds: MLXArray? = nil,
        attentionMask: MLXArray? = nil,
        cache: [(MLXArray, MLXArray)]? = nil
    ) -> (MLXArray, [(MLXArray, MLXArray)]) {
        var hiddenStates: MLXArray
        if let embeds = inputsEmbeds {
            hiddenStates = embeds
        } else if let ids = inputIds {
            hiddenStates = embedTokens(ids)
        } else {
            fatalError("Either inputIds or inputsEmbeds must be provided")
        }

        let seqLen = hiddenStates.dim(1)

        let mask: MLXArray?
        if let providedMask = attentionMask {
            mask = providedMask
        } else if seqLen == 1 {
            mask = nil
        } else {
            let cacheLen = cache?.first?.0.dim(2) ?? 0
            let totalLen = seqLen + cacheLen
            let rows = (MLXArray(0..<Int32(seqLen)) + Int32(cacheLen)).expandedDimensions(axis: 1)
            let cols = MLXArray(0..<Int32(totalLen)).expandedDimensions(axis: 0)
            mask = MLX.where(cols .> rows, MLXArray(Float(-1e9)), MLXArray(Float(0)))
                .expandedDimensions(axes: [0, 1])
                .asType(hiddenStates.dtype)
        }

        var newCache: [(MLXArray, MLXArray)] = []
        for (i, layer) in layers.enumerated() {
            let layerCache = cache?[i]
            let (output, updatedCache) = layer(hiddenStates, attentionMask: mask, cache: layerCache)
            hiddenStates = output
            newCache.append(updatedCache)
        }

        hiddenStates = norm(hiddenStates)
        return (hiddenStates, newCache)
    }
}

extension FloatTextModel: ForcedAlignerTextDecoding {
    public func embeddings(for inputIds: MLXArray) -> MLXArray {
        embedTokens(inputIds)
    }

    public func decode(
        inputsEmbeds: MLXArray,
        attentionMask: MLXArray?,
        cache: [(MLXArray, MLXArray)]?
    ) -> (MLXArray, [(MLXArray, MLXArray)]) {
        self(inputIds: nil, inputsEmbeds: inputsEmbeds, attentionMask: attentionMask, cache: cache)
    }
}
