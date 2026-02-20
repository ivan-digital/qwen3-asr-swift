import Foundation
import MLX
import MLXNN
import MLXFast
import Qwen3Common

/// GQA attention for TTS Talker with standard RoPE via MLXFast fused kernel.
///
/// For basic TTS where all 3 MRoPE position dimensions are identical (T=H=W),
/// MRoPE reduces to standard 1D RoPE. Using MLXNN.RoPE (backed by MLXFast.RoPE)
/// replaces 56 manual cos/sin + rotate-half kernel dispatches per step with 56
/// fused Metal kernel calls, reducing command buffer overhead.
public class TalkerAttention: Module {
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

    public init(config: TalkerConfig) {
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

    /// Forward pass with RoPE offset for positional encoding.
    /// Offset is MLXArray to enable compile tracking (compile bakes Swift Ints as constants).
    /// Batch dimension uses -1 in reshapes so compiled graph works for any batch size.
    public func callAsFunction(
        _ hiddenStates: MLXArray,
        offset: MLXArray,
        attentionMask: MLXArray? = nil,
        cache: (MLXArray, MLXArray)? = nil
    ) -> (MLXArray, (MLXArray, MLXArray)) {
        let seqLen = hiddenStates.dim(1)

        var queries = qProj(hiddenStates)
        var keys = kProj(hiddenStates)
        var values = vProj(hiddenStates)

        queries = queries.reshaped(-1, seqLen, numHeads, headDim)
        keys = keys.reshaped(-1, seqLen, numKVHeads, headDim)
        values = values.reshaped(-1, seqLen, numKVHeads, headDim)

        queries = qNorm(queries)
        keys = kNorm(keys)

        // Transpose to [B, N, S, D] for SDPA
        queries = queries.transposed(0, 2, 1, 3)
        keys = keys.transposed(0, 2, 1, 3)
        values = values.transposed(0, 2, 1, 3)

        // Apply RoPE via fused MLXFast kernel (MLXArray offset for compile compatibility)
        queries = rope(queries, offset: offset)
        keys = rope(keys, offset: offset)

        // Update KV cache
        var cachedKeys = keys
        var cachedValues = values

        if let (prevKeys, prevValues) = cache {
            cachedKeys = concatenated([prevKeys, keys], axis: 2)
            cachedValues = concatenated([prevValues, values], axis: 2)
        }

        let attnOutput = MLXFast.scaledDotProductAttention(
            queries: queries, keys: cachedKeys, values: cachedValues,
            scale: scale, mask: attentionMask)

        // [B, N, S, D] -> [B, S, N, D] -> [B, S, N*D]
        let output = oProj(attnOutput.transposed(0, 2, 1, 3).reshaped(-1, seqLen, numHeads * headDim))

        return (output, (cachedKeys, cachedValues))
    }
}

/// Talker decoder layer (pre-norm transformer block)
public class TalkerDecoderLayer: Module {
    @ModuleInfo var selfAttn: TalkerAttention
    @ModuleInfo var mlp: QuantizedMLP
    @ModuleInfo var inputLayerNorm: RMSNorm
    @ModuleInfo var postAttentionLayerNorm: RMSNorm

    public init(config: TalkerConfig) {
        self._selfAttn.wrappedValue = TalkerAttention(config: config)
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
        offset: MLXArray,
        attentionMask: MLXArray? = nil,
        cache: (MLXArray, MLXArray)? = nil
    ) -> (MLXArray, (MLXArray, MLXArray)) {
        let residual = hiddenStates
        var hidden = inputLayerNorm(hiddenStates)
        let (attnOutput, newCache) = selfAttn(
            hidden, offset: offset,
            attentionMask: attentionMask, cache: cache)
        hidden = residual + attnOutput

        let residual2 = hidden
        hidden = postAttentionLayerNorm(hidden)
        hidden = mlp(hidden)
        hidden = residual2 + hidden

        return (hidden, newCache)
    }
}

/// Text projection MLP: Linear(textHidden, textHidden) -> SiLU -> Linear(textHidden, hidden)
public class TextProjectionMLP: Module {
    @ModuleInfo var fc1: QuantizedLinear
    @ModuleInfo var fc2: QuantizedLinear

    public init(textHiddenSize: Int, hiddenSize: Int, groupSize: Int = 64, bits: Int = 4) {
        self._fc1.wrappedValue = QuantizedLinear(
            textHiddenSize, textHiddenSize, bias: true,
            groupSize: groupSize, bits: bits)
        self._fc2.wrappedValue = QuantizedLinear(
            textHiddenSize, hiddenSize, bias: true,
            groupSize: groupSize, bits: bits)

        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = fc1(x)
        h = silu(h)
        h = fc2(h)
        return h
    }
}

/// Full TTS Talker model
public class TalkerModel: Module {
    public let config: TalkerConfig

    @ModuleInfo var codecEmbedding: Embedding
    @ModuleInfo var textEmbedding: Embedding
    @ModuleInfo var textProjection: TextProjectionMLP
    @ModuleInfo var layers: [TalkerDecoderLayer]
    @ModuleInfo var norm: RMSNorm
    @ModuleInfo var codecHead: QuantizedLinear

    public init(config: TalkerConfig) {
        self.config = config

        // Codec embedding: not quantized (float)
        self._codecEmbedding.wrappedValue = Embedding(
            embeddingCount: config.codecVocabSize,
            dimensions: config.hiddenSize)

        // Text embedding: not quantized (float), dim=textHiddenSize (2048)
        self._textEmbedding.wrappedValue = Embedding(
            embeddingCount: config.textVocabSize,
            dimensions: config.textHiddenSize)

        // Text projection: textHiddenSize -> hiddenSize
        self._textProjection.wrappedValue = TextProjectionMLP(
            textHiddenSize: config.textHiddenSize,
            hiddenSize: config.hiddenSize,
            groupSize: config.groupSize,
            bits: config.bits)

        // Transformer layers
        self._layers.wrappedValue = (0..<config.numLayers).map { _ in
            TalkerDecoderLayer(config: config)
        }

        self._norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)

        // Codec head (logits over codec vocabulary)
        self._codecHead.wrappedValue = QuantizedLinear(
            config.hiddenSize, config.codecVocabSize, bias: false,
            groupSize: config.groupSize, bits: config.bits)

        super.init()
    }

    /// Embed text tokens and project to hidden dim
    public func embedText(_ tokenIds: MLXArray) -> MLXArray {
        let embeds = textEmbedding(tokenIds)  // [B, S, textHiddenSize]
        return textProjection(embeds)  // [B, S, hiddenSize]
    }

    /// Embed codec tokens
    public func embedCodec(_ tokenIds: MLXArray) -> MLXArray {
        codecEmbedding(tokenIds)  // [B, S, hiddenSize]
    }

    /// Access the codec embedding layer directly (for use in generation loop)
    public func getInputEmbeddings() -> Embedding {
        codecEmbedding
    }

    /// Forward pass through Talker.
    /// Offset is MLXArray for compile tracking (compile bakes Swift Int as constant).
    /// - Returns: (logits, hiddenStates, newCache)
    public func callAsFunction(
        inputsEmbeds: MLXArray,
        offset: MLXArray,
        attentionMask: MLXArray? = nil,
        cache: [(MLXArray, MLXArray)]? = nil
    ) -> (MLXArray, MLXArray, [(MLXArray, MLXArray)]) {
        var hiddenStates = inputsEmbeds

        // Determine attention mask
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

        // Apply decoder layers with RoPE offset
        var newCache: [(MLXArray, MLXArray)] = []
        for (i, layer) in layers.enumerated() {
            let layerCache = cache?[i]
            let (output, updatedCache) = layer(
                hiddenStates,
                offset: offset,
                attentionMask: mask,
                cache: layerCache)
            hiddenStates = output
            newCache.append(updatedCache)
        }

        hiddenStates = norm(hiddenStates)

        // Compute logits
        let logits = codecHead(hiddenStates)

        return (logits, hiddenStates, newCache)
    }
}
