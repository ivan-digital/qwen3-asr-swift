import Foundation
import MLX
import MLXNN
import MLXFast
import Qwen3Common

// MARK: - Sampling

/// Sample a token from logits using temperature, top-k, top-p, and repetition penalty.
/// EOS protection: the EOS logit is saved before top-k/top-p filtering and restored after.
/// Uses Gumbel-max trick for multinomial sampling to avoid MLX categorical bugs.
func sampleToken(
    logits: MLXArray,
    temperature: Float,
    topK: Int,
    topP: Float,
    repetitionPenalty: Float = 1.0,
    generatedTokens: [Int32] = [],
    suppressRange: (Int, Int)? = nil,
    eosTokenId: Int? = nil
) -> Int32 {
    var logits = logits.squeezed().asType(.float32)
    let vocabSize = logits.dim(0)

    // 1. Token suppression: set range to -inf (except EOS)
    if let (start, end) = suppressRange, start < end, start >= 0, end <= vocabSize {
        let indices = MLXArray(0..<Int32(vocabSize))
        let geStart = indices .>= MLXArray(Int32(start))
        let ltEnd = indices .< MLXArray(Int32(end))
        var suppressMask = logicalAnd(geStart, ltEnd)

        if let eos = eosTokenId, eos >= start, eos < end {
            let notEos = indices .!= MLXArray(Int32(eos))
            suppressMask = logicalAnd(suppressMask, notEos)
        }

        logits = MLX.where(suppressMask, MLXArray(Float(-1e9)), logits)
    }

    // 2. Repetition penalty
    if repetitionPenalty != 1.0 && !generatedTokens.isEmpty {
        let uniqueTokens = Array(Set(generatedTokens))
        let indices = MLXArray(0..<Int32(vocabSize))
        var penaltyMask = indices .== Int32(-1)  // all false
        for token in uniqueTokens {
            penaltyMask = logicalOr(penaltyMask, indices .== token)
        }

        let penalty = MLXArray(repetitionPenalty)
        let penalizedPos = logits / penalty
        let penalizedNeg = logits * penalty
        let penalized = MLX.where(logits .< MLXArray(Float(0)), penalizedNeg, penalizedPos)
        logits = MLX.where(penaltyMask, penalized, logits)
    }

    // 3. Greedy decoding
    if temperature <= 0 {
        return argMax(logits).item(Int32.self)
    }

    // 4. Apply temperature
    logits = logits / MLXArray(temperature)

    // 5. Save EOS logit before top-k/top-p (so it can't be filtered out)
    var savedEosLogit: MLXArray? = nil
    if let eos = eosTokenId, eos >= 0, eos < vocabSize {
        savedEosLogit = logits[eos]
    }

    // 6. Top-k filtering
    if topK > 0 && topK < vocabSize {
        let sorted = MLX.sorted(logits)
        let threshold = sorted[vocabSize - topK]
        logits = MLX.where(logits .< threshold, MLXArray(Float(-1e9)), logits)
    }

    // 7. Top-p (nucleus) filtering
    if topP < 1.0 {
        let sortedIndices = argSort(logits)
        let sortedLogits = logits[sortedIndices]
        let probs = softmax(sortedLogits)
        let cumProbs = cumsum(probs)

        let sortedMask = cumProbs - probs .> MLXArray(topP)
        let filteredLogits = MLX.where(sortedMask, MLXArray(Float(-1e9)), sortedLogits)

        let unsortIndices = argSort(sortedIndices)
        logits = filteredLogits[unsortIndices]
    }

    // 8. Restore EOS logit after top-k/top-p
    if let eos = eosTokenId, let eosLogit = savedEosLogit, eos >= 0, eos < vocabSize {
        let indices = MLXArray(0..<Int32(vocabSize))
        let eosMask = indices .== MLXArray(Int32(eos))
        logits = MLX.where(eosMask, eosLogit, logits)
    }

    // 9. Gumbel-max sampling: argmax(logits + Gumbel) ~ Categorical(softmax(logits))
    let gumbel = MLXRandom.gumbel(logits.shape)
    let perturbedLogits = logits + gumbel
    return argMax(perturbedLogits).item(Int32.self)
}

// MARK: - CosyVoiceAttention

/// GQA attention for CosyVoice LLM (Qwen2.5-0.5B) with RoPE via MLXFast fused kernel.
///
/// Uses 14 query heads, 2 KV heads, head_dim=64, with q_norm and k_norm (Qwen2.5 style).
/// RoPE offset is MLXArray for compile compatibility (compile bakes Swift Ints as constants).
public class CosyVoiceAttention: Module {
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

    public init(config: CosyVoiceLLMConfig) {
        self.numHeads = config.numHeads
        self.numKVHeads = config.numKVHeads
        self.headDim = config.headDim
        self.scale = 1.0 / sqrt(Float(config.headDim))

        let hiddenSize = config.hiddenSize

        self._qProj.wrappedValue = QuantizedLinear(
            hiddenSize, numHeads * headDim, bias: true,
            groupSize: config.groupSize, bits: config.bits)
        self._kProj.wrappedValue = QuantizedLinear(
            hiddenSize, numKVHeads * headDim, bias: true,
            groupSize: config.groupSize, bits: config.bits)
        self._vProj.wrappedValue = QuantizedLinear(
            hiddenSize, numKVHeads * headDim, bias: true,
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

        // SDPA returns [B, N, S, D] -> transpose to [B, S, N, D] -> reshape to [B, S, N*D]
        let output = oProj(attnOutput.transposed(0, 2, 1, 3).reshaped(-1, seqLen, numHeads * headDim))

        return (output, (cachedKeys, cachedValues))
    }
}

// MARK: - CosyVoiceBlock

/// Pre-norm transformer block for CosyVoice LLM.
/// Uses SwiGLU MLP via QuantizedMLP from Qwen3Common.
public class CosyVoiceBlock: Module {
    @ModuleInfo var selfAttn: CosyVoiceAttention
    @ModuleInfo var mlp: QuantizedMLP
    @ModuleInfo var inputLayerNorm: RMSNorm
    @ModuleInfo var postAttentionLayerNorm: RMSNorm

    public init(config: CosyVoiceLLMConfig) {
        self._selfAttn.wrappedValue = CosyVoiceAttention(config: config)
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

// MARK: - CosyVoiceLLM

/// Qwen2.5-0.5B based speech token generator for CosyVoice3.
///
/// Architecture: Standard Qwen2-family decoder-only transformer with separate
/// text and speech embeddings plus a speech token head.
///
/// Input sequence: [sos_embed, text_embeds..., task_id_embed, speech_tokens...]
///
/// Generation: Autoregressive decoding with KV cache. Prefills the text prefix,
/// then generates speech tokens one at a time until EOS or maxTokens.
public class CosyVoiceLLM: Module {
    public let config: CosyVoiceLLMConfig

    @ModuleInfo var textEmbedding: Embedding
    @ModuleInfo var speechEmbedding: Embedding
    @ModuleInfo var layers: [CosyVoiceBlock]
    @ModuleInfo var norm: RMSNorm
    @ModuleInfo var speechHead: QuantizedLinear

    public init(config: CosyVoiceLLMConfig) {
        self.config = config

        // Text embedding: standard Qwen2.5 vocabulary (151936 tokens)
        self._textEmbedding.wrappedValue = Embedding(
            embeddingCount: config.textVocabSize,
            dimensions: config.hiddenSize)

        // Speech embedding: speech tokens + special tokens (6761 total)
        self._speechEmbedding.wrappedValue = Embedding(
            embeddingCount: config.totalSpeechVocabSize,
            dimensions: config.hiddenSize)

        // Transformer layers (24 layers)
        self._layers.wrappedValue = (0..<config.numLayers).map { _ in
            CosyVoiceBlock(config: config)
        }

        // Final layer norm
        self._norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)

        // Speech token head: project hidden states to speech vocabulary logits
        self._speechHead.wrappedValue = QuantizedLinear(
            config.hiddenSize, config.totalSpeechVocabSize, bias: false,
            groupSize: config.groupSize, bits: config.bits)

        super.init()
    }

    /// Build the input embedding sequence for generation.
    ///
    /// Format: [sos_embed, text_embeds..., task_id_embed]
    /// - SOS and task_id come from the speech embedding table (special token indices)
    /// - Text tokens come from the text embedding table
    ///
    /// Returns: [1, prefix_len, hidden_size]
    public func buildInputSequence(textTokens: [Int32]) -> MLXArray {
        // Embed SOS token from speech embedding
        let sosEmbed = speechEmbedding(MLXArray([Int32(config.sosToken)]))  // [1, hidden]

        // Embed text tokens from text embedding
        let textIds = MLXArray(textTokens).expandedDimensions(axis: 0)  // [1, T]
        let textEmbeds = textEmbedding(textIds)  // [1, T, hidden]

        // Embed task_id token from speech embedding
        let taskIdEmbed = speechEmbedding(MLXArray([Int32(config.taskIdToken)]))  // [1, hidden]

        // Concatenate: [sos, text..., task_id] along sequence dimension
        let sosExpanded = sosEmbed.expandedDimensions(axis: 0)      // [1, 1, hidden]
        let taskIdExpanded = taskIdEmbed.expandedDimensions(axis: 0) // [1, 1, hidden]

        return concatenated([sosExpanded, textEmbeds, taskIdExpanded], axis: 1)  // [1, prefix_len, hidden]
    }

    /// Single forward step through the transformer.
    ///
    /// - Parameters:
    ///   - input: Input embeddings [B, S, hidden]
    ///   - offset: RoPE position offset (MLXArray for compile compatibility)
    ///   - cache: KV cache from previous steps (array of tuples, one per layer)
    /// - Returns: (logits [B, S, totalSpeechVocabSize], updated cache)
    public func forwardStep(
        _ input: MLXArray,
        offset: MLXArray,
        cache: [(MLXArray, MLXArray)]? = nil
    ) -> (MLXArray, [(MLXArray, MLXArray)]) {
        var hiddenStates = input

        // Build causal attention mask
        let seqLen = hiddenStates.dim(1)
        let mask: MLXArray?
        if seqLen == 1 {
            // Single token step: no mask needed (attends to all cached positions)
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

        // Apply decoder layers with KV cache
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

        // Final norm and speech head projection
        hiddenStates = norm(hiddenStates)
        let logits = speechHead(hiddenStates)

        return (logits, newCache)
    }

    /// Generate speech tokens autoregressively from text tokens.
    ///
    /// 1. Builds prefix: [sos_embed, text_embeds..., task_id_embed]
    /// 2. Prefills the prefix through the transformer to get initial KV cache
    /// 3. Autoregressively generates speech tokens until EOS or maxTokens
    ///
    /// - Parameters:
    ///   - textTokens: Text token IDs from the tokenizer
    ///   - sampling: Sampling configuration (temperature, topK, topP)
    ///   - maxTokens: Maximum number of speech tokens to generate
    /// - Returns: Array of generated speech token IDs (FSQ codes, 0-6560)
    public func generate(
        textTokens: [Int32],
        sampling: CosyVoiceSamplingConfig = CosyVoiceSamplingConfig(),
        maxTokens: Int = 4096
    ) -> [Int32] {
        let eosToken = config.eosToken

        // Build prefix embeddings: [1, prefix_len, hidden]
        let prefixEmbeds = buildInputSequence(textTokens: textTokens)
        let prefixLen = prefixEmbeds.dim(1)

        // Prefill: forward entire prefix through transformer
        let offset = MLXArray(Int32(0))
        let (prefillLogits, cache) = forwardStep(prefixEmbeds, offset: offset, cache: nil)
        eval(prefillLogits, cache)

        // Sample first speech token from last position of prefill output
        // Suppress tokens above speechTokenSize (6561+) except EOS (6562)
        let suppressStart = config.speechTokenSize  // 6561
        let suppressEnd = config.totalSpeechVocabSize  // 6761
        var currentToken = sampleToken(
            logits: prefillLogits[0..., (prefixLen - 1)..<prefixLen, 0...],
            temperature: 1.0,  // first token: no temperature scaling
            topK: sampling.topK,
            topP: sampling.topP,
            suppressRange: (suppressStart, suppressEnd),
            eosTokenId: eosToken)

        if currentToken == Int32(eosToken) {
            return []
        }

        var generatedTokens: [Int32] = [currentToken]
        var currentCache = cache

        // Autoregressive generation loop
        for step in 0..<(maxTokens - 1) {
            // Embed the last generated speech token
            let tokenEmbed = speechEmbedding(
                MLXArray([currentToken]).expandedDimensions(axis: 0))  // [1, 1, hidden]

            // Forward single token through transformer
            let stepOffset = MLXArray(Int32(prefixLen + step))
            let (stepLogits, newCache) = forwardStep(
                tokenEmbed, offset: stepOffset, cache: currentCache)
            eval(stepLogits, newCache)
            currentCache = newCache

            // Sample next token
            currentToken = sampleToken(
                logits: stepLogits,
                temperature: 1.0,
                topK: sampling.topK,
                topP: sampling.topP,
                generatedTokens: generatedTokens,
                suppressRange: (suppressStart, suppressEnd),
                eosTokenId: eosToken)

            if currentToken == Int32(eosToken) {
                break
            }

            generatedTokens.append(currentToken)
        }

        return generatedTokens
    }
}
