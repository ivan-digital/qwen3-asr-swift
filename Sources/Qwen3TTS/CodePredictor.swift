import Foundation
import MLX
import MLXNN
import MLXFast
import Qwen3Common

/// Attention for Code Predictor — standard 1D RoPE (same pattern as ASR)
public class CodePredictorAttention: Module {
    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo var qProj: QuantizedLinear
    @ModuleInfo var kProj: QuantizedLinear
    @ModuleInfo var vProj: QuantizedLinear
    @ModuleInfo var oProj: QuantizedLinear
    @ModuleInfo var qNorm: RMSNorm
    @ModuleInfo var kNorm: RMSNorm

    let rope: MLXNN.RoPE

    public init(config: CodePredictorConfig) {
        self.numHeads = config.numHeads
        self.numKVHeads = config.numKVHeads
        self.headDim = config.headDim
        self.scale = 1.0 / sqrt(Float(headDim))

        let hiddenSize = config.hiddenSize

        self._qProj.wrappedValue = QuantizedLinear(
            hiddenSize, numHeads * headDim, bias: false,
            groupSize: config.groupSize, bits: config.bits)
        self._kProj.wrappedValue = QuantizedLinear(
            hiddenSize, numKVHeads * headDim, bias: false,
            groupSize: config.groupSize, bits: config.bits)
        self._vProj.wrappedValue = QuantizedLinear(
            hiddenSize, numKVHeads * headDim, bias: false,
            groupSize: config.groupSize, bits: config.bits)
        self._oProj.wrappedValue = QuantizedLinear(
            numHeads * headDim, hiddenSize, bias: false,
            groupSize: config.groupSize, bits: config.bits)

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

        let attnOutput = MLXFast.scaledDotProductAttention(
            queries: queries, keys: cachedKeys, values: cachedValues,
            scale: scale, mask: attentionMask)

        let output = oProj(attnOutput.transposed(0, 2, 1, 3).reshaped(batch, seqLen, numHeads * headDim))

        return (output, (cachedKeys, cachedValues))
    }
}

/// Code Predictor decoder layer
public class CodePredictorDecoderLayer: Module {
    @ModuleInfo var selfAttn: CodePredictorAttention
    @ModuleInfo var mlp: QuantizedMLP
    @ModuleInfo var inputLayerNorm: RMSNorm
    @ModuleInfo var postAttentionLayerNorm: RMSNorm

    public init(config: CodePredictorConfig) {
        self._selfAttn.wrappedValue = CodePredictorAttention(config: config)
        self._mlp.wrappedValue = QuantizedMLP(
            hiddenSize: config.hiddenSize,
            intermediateSize: config.intermediateSize,
            groupSize: config.groupSize,
            bits: config.bits)
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

/// Code Predictor model — predicts remaining 15 codebooks from first codebook hidden states
public class CodePredictorModel: Module {
    public let config: CodePredictorConfig

    // 15 codec embedding tables (one per remaining codebook group, index 1-15)
    @ModuleInfo var codecEmbeddings: [Embedding]
    @ModuleInfo var layers: [CodePredictorDecoderLayer]
    @ModuleInfo var norm: RMSNorm
    // 15 lm_head projections (one per remaining codebook group, quantized)
    @ModuleInfo var lmHeads: [QuantizedLinear]

    public init(config: CodePredictorConfig) {
        self.config = config

        // 15 embedding tables for codebook groups 2-16 (index 0-14)
        self._codecEmbeddings.wrappedValue = (0..<(config.numCodeGroups - 1)).map { _ in
            Embedding(embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        }

        self._layers.wrappedValue = (0..<config.numLayers).map { _ in
            CodePredictorDecoderLayer(config: config)
        }

        self._norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)

        // 15 lm_heads for codebook groups 2-16 (quantized)
        self._lmHeads.wrappedValue = (0..<(config.numCodeGroups - 1)).map { _ in
            QuantizedLinear(config.hiddenSize, config.vocabSize, bias: false,
                           groupSize: config.groupSize, bits: config.bits)
        }

        super.init()
    }

    /// Predict a specific codebook group
    /// - Parameters:
    ///   - inputsEmbeds: Hidden states from previous step [B, T, D]
    ///   - groupIndex: Which codebook group to predict (0 = codebook 2, 14 = codebook 16)
    ///   - cache: KV cache from previous group steps
    /// - Returns: (logits, newCache)
    public func callAsFunction(
        inputsEmbeds: MLXArray,
        groupIndex: Int,
        cache: [(MLXArray, MLXArray)]? = nil
    ) -> (MLXArray, [(MLXArray, MLXArray)]) {
        var hiddenStates = inputsEmbeds

        let seqLen = hiddenStates.dim(1)
        let mask: MLXArray?
        if seqLen == 1 {
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
        let logits = lmHeads[groupIndex](hiddenStates)

        return (logits, newCache)
    }

    /// Predict all 15 codebook groups in parallel from the same hidden state.
    /// Single forward pass through 5 layers, then apply all 15 lm_heads.
    /// - Parameter inputsEmbeds: [1, 2, D] — [hidden_state, code_0_embed]
    /// - Returns: 15 logit arrays, each [1, 1, vocab]
    public func predictAllGroupsParallel(
        inputsEmbeds: MLXArray
    ) -> [MLXArray] {
        var hiddenStates = inputsEmbeds

        // Build causal mask for length 2
        let mask = MLXArray([Float(0), Float(-1e9), Float(0), Float(0)])
            .reshaped(1, 1, 2, 2)
            .asType(hiddenStates.dtype)

        for layer in layers {
            let (output, _) = layer(hiddenStates, attentionMask: mask, cache: nil)
            hiddenStates = output
        }
        hiddenStates = norm(hiddenStates)

        let lastHidden = hiddenStates[0..., 1..<2, 0...]  // [1, 1, D]
        return lmHeads.map { $0(lastHidden) }
    }

    /// Embed a token for a specific codebook group
    public func embedCodecGroup(_ tokenIds: MLXArray, groupIndex: Int) -> MLXArray {
        codecEmbeddings[groupIndex](tokenIds)
    }

    /// Sum embeddings for all 15 codebook groups in one call.
    /// - Parameter tokens: 15 Int32 tokens (one per group)
    /// - Returns: [1, 1, D] summed embedding
    public func batchEmbedAllGroups(_ tokens: [Int32]) -> MLXArray {
        precondition(tokens.count == config.numCodeGroups - 1)
        var sum = codecEmbeddings[0](MLXArray([tokens[0]]).expandedDimensions(axis: 0))
        for i in 1..<tokens.count {
            sum = sum + codecEmbeddings[i](MLXArray([tokens[i]]).expandedDimensions(axis: 0))
        }
        return sum  // [1, 1, D]
    }

    /// Sum embeddings for all 15 codebook groups for B items.
    /// - Parameter tokens: [B, 15] Int32 tokens
    /// - Returns: [B, 1, D] summed embedding
    public func batchEmbedAllGroupsBatch(_ tokens: MLXArray) -> MLXArray {
        let numGroups = config.numCodeGroups - 1
        // tokens[:, 0:1] → [B, 1], embed → [B, 1, D]
        var sum = codecEmbeddings[0](tokens[0..., 0..<1])  // [B, 1, D]
        for i in 1..<numGroups {
            sum = sum + codecEmbeddings[i](tokens[0..., i..<(i + 1)])  // [B, 1, D]
        }
        return sum
    }
}
