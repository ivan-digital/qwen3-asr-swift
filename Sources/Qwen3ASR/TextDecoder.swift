import Foundation
import MLX
import MLXNN
import MLXFast

/// RoPE (Rotary Position Embedding) implementation
public class RoPE {
    let dims: Int
    let base: Float
    let scale: Float

    public init(dims: Int, base: Float = 10000.0, scale: Float = 1.0) {
        self.dims = dims
        self.base = base
        self.scale = scale
    }

    /// Apply RoPE to queries and keys
    public func callAsFunction(
        _ q: MLXArray,
        _ k: MLXArray,
        offset: Int = 0
    ) -> (MLXArray, MLXArray) {
        let seqLen = q.dim(2)

        // Compute frequencies
        let halfDim = dims / 2
        let freqSeq = MLXArray(0..<halfDim).asType(.float32)
        let invFreq = 1.0 / pow(MLXArray(base), freqSeq / Float(halfDim))

        // Create position sequence
        let positions = MLXArray((offset)..<(offset + seqLen)).asType(.float32) * scale

        // Compute angles: [seq_len, half_dim]
        let angles = positions.expandedDimensions(axis: 1) * invFreq.expandedDimensions(axis: 0)

        // Create rotation matrix components
        let cosAngles = cos(angles)
        let sinAngles = sin(angles)

        // Apply rotation to q and k
        let qRotated = applyRotation(q, cos: cosAngles, sin: sinAngles)
        let kRotated = applyRotation(k, cos: cosAngles, sin: sinAngles)

        return (qRotated, kRotated)
    }

    private func applyRotation(_ x: MLXArray, cos cosAngles: MLXArray, sin sinAngles: MLXArray) -> MLXArray {
        // x shape: [batch, heads, seq, head_dim]
        // Using interleaved rotation (traditional=False in MLX RoPE)
        // This means we take even indices and odd indices, not first half and second half

        // Extract interleaved components: x1 = x[..., 0::2], x2 = x[..., 1::2]
        // MLX doesn't have step slicing, so we need to reshape and extract
        let batch = x.dim(0)
        let heads = x.dim(1)
        let seqLen = x.dim(2)
        let headDim = x.dim(3)
        let halfDim = headDim / 2

        // Reshape to [batch, heads, seq, half_dim, 2] then extract
        let xReshaped = x.reshaped(batch, heads, seqLen, halfDim, 2)
        let x1 = xReshaped[.ellipsis, 0]  // Even indices [batch, heads, seq, half_dim]
        let x2 = xReshaped[.ellipsis, 1]  // Odd indices [batch, heads, seq, half_dim]

        // Reshape cos/sin for broadcasting: [1, 1, seq, half_dim]
        let cosR = cosAngles.expandedDimensions(axes: [0, 1])
        let sinR = sinAngles.expandedDimensions(axes: [0, 1])

        // Apply rotation: [x1 * cos - x2 * sin, x1 * sin + x2 * cos]
        let rotated1 = x1 * cosR - x2 * sinR
        let rotated2 = x1 * sinR + x2 * cosR

        // Interleave back: stack and reshape
        let stacked = stacked([rotated1, rotated2], axis: -1)  // [batch, heads, seq, half_dim, 2]
        return stacked.reshaped(batch, heads, seqLen, headDim)
    }
}

/// Multi-head attention for Qwen3 text decoder with GQA and RoPE
public class TextAttention: Module {
    let config: TextDecoderConfig
    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let scale: Float
    let rope: RoPE

    @ModuleInfo var qProj: Linear
    @ModuleInfo var kProj: Linear
    @ModuleInfo var vProj: Linear
    @ModuleInfo var oProj: Linear
    @ModuleInfo var qNorm: RMSNorm
    @ModuleInfo var kNorm: RMSNorm

    public init(config: TextDecoderConfig) {
        self.config = config
        self.numHeads = config.numHeads
        self.numKVHeads = config.numKVHeads
        self.headDim = config.headDim
        self.scale = 1.0 / sqrt(Float(headDim))
        self.rope = RoPE(dims: headDim, base: config.ropeTheta)

        let hiddenSize = config.hiddenSize
        self._qProj.wrappedValue = Linear(hiddenSize, numHeads * headDim, bias: false)
        self._kProj.wrappedValue = Linear(hiddenSize, numKVHeads * headDim, bias: false)
        self._vProj.wrappedValue = Linear(hiddenSize, numKVHeads * headDim, bias: false)
        self._oProj.wrappedValue = Linear(numHeads * headDim, hiddenSize, bias: false)

        // Q/K normalization (Qwen3 specific)
        self._qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        self._kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)

        super.init()
    }

    public func callAsFunction(
        _ hiddenStates: MLXArray,
        attentionMask: MLXArray? = nil,
        cache: (MLXArray, MLXArray)? = nil
    ) -> (MLXArray, (MLXArray, MLXArray)) {
        let (batch, seqLen, _) = (hiddenStates.dim(0), hiddenStates.dim(1), hiddenStates.dim(2))

        // Project Q, K, V
        var queries = qProj(hiddenStates)
        var keys = kProj(hiddenStates)
        var values = vProj(hiddenStates)

        // Reshape for multi-head attention
        queries = queries.reshaped(batch, seqLen, numHeads, headDim)
        keys = keys.reshaped(batch, seqLen, numKVHeads, headDim)
        values = values.reshaped(batch, seqLen, numKVHeads, headDim)

        // Apply Q/K normalization
        queries = qNorm(queries)
        keys = kNorm(keys)

        // Transpose to [batch, heads, seq, head_dim]
        queries = queries.transposed(0, 2, 1, 3)
        keys = keys.transposed(0, 2, 1, 3)
        values = values.transposed(0, 2, 1, 3)

        // Calculate offset for RoPE based on cache
        let offset = cache?.0.dim(2) ?? 0

        // Apply RoPE
        let (qRotated, kRotated) = rope(queries, keys, offset: offset)

        // Update cache
        var cachedKeys = kRotated
        var cachedValues = values

        if let (prevKeys, prevValues) = cache {
            cachedKeys = concatenated([prevKeys, kRotated], axis: 2)
            cachedValues = concatenated([prevValues, values], axis: 2)
        }

        // Expand KV heads for GQA (repeat KV heads to match Q heads)
        let repeatFactor = numHeads / numKVHeads
        var expandedKeys = cachedKeys
        var expandedValues = cachedValues

        if repeatFactor > 1 {
            // Repeat KV heads
            expandedKeys = expandedKeys.expandedDimensions(axis: 2)
            expandedKeys = tiled(expandedKeys, repetitions: [1, 1, repeatFactor, 1, 1])
            expandedKeys = expandedKeys.reshaped(batch, numHeads, -1, headDim)

            expandedValues = expandedValues.expandedDimensions(axis: 2)
            expandedValues = tiled(expandedValues, repetitions: [1, 1, repeatFactor, 1, 1])
            expandedValues = expandedValues.reshaped(batch, numHeads, -1, headDim)
        }

        // Scaled dot-product attention
        var attnWeights = matmul(qRotated, expandedKeys.transposed(0, 1, 3, 2)) * scale

        if let mask = attentionMask {
            attnWeights = attnWeights + mask
        }

        attnWeights = softmax(attnWeights, axis: -1)

        // Apply attention to values
        var attnOutput = matmul(attnWeights, expandedValues)

        // Reshape back to [batch, seq, hidden]
        attnOutput = attnOutput.transposed(0, 2, 1, 3).reshaped(batch, seqLen, numHeads * headDim)

        let output = oProj(attnOutput)

        return (output, (cachedKeys, cachedValues))
    }
}

/// MLP for Qwen3 text decoder (SwiGLU activation)
public class TextMLP: Module {
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
        // SwiGLU: down(silu(gate(x)) * up(x))
        let gate = silu(gateProj(x))
        let up = upProj(x)
        return downProj(gate * up)
    }
}

/// Decoder layer for Qwen3 text model
public class TextDecoderLayer: Module {
    @ModuleInfo var selfAttn: TextAttention
    @ModuleInfo var mlp: TextMLP
    @ModuleInfo var inputLayerNorm: RMSNorm
    @ModuleInfo var postAttentionLayerNorm: RMSNorm

    public init(config: TextDecoderConfig) {
        self._selfAttn.wrappedValue = TextAttention(config: config)
        self._mlp.wrappedValue = TextMLP(config: config)
        self._inputLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)

        super.init()
    }

    public func callAsFunction(
        _ hiddenStates: MLXArray,
        attentionMask: MLXArray? = nil,
        cache: (MLXArray, MLXArray)? = nil
    ) -> (MLXArray, (MLXArray, MLXArray)) {
        // Self attention with pre-norm
        var residual = hiddenStates
        var hidden = inputLayerNorm(hiddenStates)
        let (attnOutput, newCache) = selfAttn(hidden, attentionMask: attentionMask, cache: cache)
        hidden = residual + attnOutput

        // MLP with pre-norm
        residual = hidden
        hidden = postAttentionLayerNorm(hidden)
        hidden = mlp(hidden)
        hidden = residual + hidden

        return (hidden, newCache)
    }
}

/// Full Qwen3 text decoder model
public class TextModel: Module {
    let config: TextDecoderConfig

    @ModuleInfo var embedTokens: Embedding
    @ModuleInfo var layers: [TextDecoderLayer]
    @ModuleInfo var norm: RMSNorm

    public init(config: TextDecoderConfig) {
        self.config = config

        self._embedTokens.wrappedValue = Embedding(embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        self._layers.wrappedValue = (0..<config.numLayers).map { _ in
            TextDecoderLayer(config: config)
        }
        self._norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)

        super.init()
    }

    /// Forward pass through text decoder
    /// - Parameters:
    ///   - inputIds: Token IDs [batch, seq]
    ///   - inputsEmbeds: Optional pre-computed embeddings [batch, seq, hidden]
    ///   - attentionMask: Attention mask
    ///   - cache: KV cache from previous forward passes
    /// - Returns: Hidden states and updated cache
    public func callAsFunction(
        inputIds: MLXArray? = nil,
        inputsEmbeds: MLXArray? = nil,
        attentionMask: MLXArray? = nil,
        cache: [(MLXArray, MLXArray)]? = nil
    ) -> (MLXArray, [(MLXArray, MLXArray)]) {
        // Get embeddings
        var hiddenStates: MLXArray
        if let embeds = inputsEmbeds {
            hiddenStates = embeds
        } else if let ids = inputIds {
            hiddenStates = embedTokens(ids)
        } else {
            fatalError("Either inputIds or inputsEmbeds must be provided")
        }

        // Create causal attention mask if needed
        let mask = attentionMask

        // Apply decoder layers
        var newCache: [(MLXArray, MLXArray)] = []
        for (i, layer) in layers.enumerated() {
            let layerCache = cache?[i]
            let (output, updatedCache) = layer(hiddenStates, attentionMask: mask, cache: layerCache)
            hiddenStates = output
            newCache.append(updatedCache)
        }

        // Final norm
        hiddenStates = norm(hiddenStates)

        return (hiddenStates, newCache)
    }
}

/// LM head for token prediction
public class LMHead: Module {
    @ModuleInfo var linear: Linear

    public init(hiddenSize: Int, vocabSize: Int) {
        self._linear.wrappedValue = Linear(hiddenSize, vocabSize, bias: false)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        return linear(x)
    }
}
