import Foundation
import MLX
import MLXNN
import MLXFast

/// Pre-quantized embedding that can be loaded directly from safetensors
public class PreQuantizedEmbedding: Module {
    public let groupSize: Int
    public let bits: Int
    public let embeddingCount: Int
    public let dimensions: Int

    @ParameterInfo var weight: MLXArray
    @ParameterInfo var scales: MLXArray
    @ParameterInfo var biases: MLXArray

    public init(embeddingCount: Int, dimensions: Int, groupSize: Int = 64, bits: Int = 4) {
        self.embeddingCount = embeddingCount
        self.dimensions = dimensions
        self.groupSize = groupSize
        self.bits = bits

        // Packed dimensions: 8 values per uint32 for 4-bit
        let packedDim = dimensions / (32 / bits)
        let numGroups = dimensions / groupSize

        // Initialize with zeros - will be loaded from weights
        self._weight.wrappedValue = MLXArray.zeros([embeddingCount, packedDim], dtype: .uint32)
        self._scales.wrappedValue = MLXArray.zeros([embeddingCount, numGroups], dtype: .bfloat16)
        self._biases.wrappedValue = MLXArray.zeros([embeddingCount, numGroups], dtype: .bfloat16)

        super.init()
        self.freeze()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let s = x.shape
        let x = x.flattened()
        let out = dequantized(
            weight[x], scales: scales[x], biases: biases[x],
            groupSize: groupSize, bits: bits)
        return out.reshaped(s + [-1])
    }

    /// For use as LM head (matmul with transposed weight)
    public func asLinear(_ x: MLXArray) -> MLXArray {
        quantizedMatmul(
            x, weight, scales: scales, biases: biases, transpose: true,
            groupSize: groupSize, bits: bits)
    }
}

/// Multi-head attention for Qwen3 text decoder with GQA and RoPE (quantized version)
public class QuantizedTextAttention: Module {
    let config: TextDecoderConfig
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

    public init(config: TextDecoderConfig) {
        self.config = config
        self.numHeads = config.numHeads
        self.numKVHeads = config.numKVHeads
        self.headDim = config.headDim
        self.scale = 1.0 / sqrt(Float(headDim))

        let hiddenSize = config.hiddenSize

        // Create quantized linear layers
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
        let (qRotated, kRotated) = applyRoPE(queries, keys, offset: offset)

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

    private func applyRoPE(_ q: MLXArray, _ k: MLXArray, offset: Int) -> (MLXArray, MLXArray) {
        let seqLen = q.dim(2)
        let halfDim = headDim / 2

        // Compute inverse frequencies: inv_freq[i] = 1.0 / (base ** (2*i / dim))
        // This is equivalent to: exp(-i * log(base) / half_dim) for i in [0, half_dim)
        let freqSeq = MLXArray(0..<halfDim).asType(.float32)
        let invFreq = exp(-freqSeq * (log(MLXArray(config.ropeTheta)) / Float(halfDim)))

        // Create position sequence
        let positions = MLXArray((offset)..<(offset + seqLen)).asType(.float32)

        // Compute angles: [seq_len, half_dim]
        let angles = positions.expandedDimensions(axis: 1) * invFreq.expandedDimensions(axis: 0)

        // Create rotation matrix components
        let cosAngles = cos(angles)
        let sinAngles = sin(angles)

        // Apply split-half rotation (NOT interleaved - this is what mlx-audio/Qwen uses)
        // First half and second half of the head_dim are rotated together
        func rotateSplitHalf(_ x: MLXArray) -> MLXArray {
            // x shape: [batch, heads, seq, head_dim]
            // Split into first half [0:half_dim] and second half [half_dim:dim]
            let x1 = x[0..., 0..., 0..., 0..<halfDim]  // [batch, heads, seq, half_dim]
            let x2 = x[0..., 0..., 0..., halfDim...]   // [batch, heads, seq, half_dim]

            // Expand cos/sin for broadcasting: [seq, half_dim] -> [1, 1, seq, half_dim]
            let cosR = cosAngles.expandedDimensions(axes: [0, 1])
            let sinR = sinAngles.expandedDimensions(axes: [0, 1])

            // Apply rotation: (x1 * cos - x2 * sin, x1 * sin + x2 * cos)
            let rotated1 = x1 * cosR - x2 * sinR
            let rotated2 = x1 * sinR + x2 * cosR

            // Concatenate back: [rotated1, rotated2] along last axis
            return concatenated([rotated1, rotated2], axis: -1)
        }

        return (rotateSplitHalf(q), rotateSplitHalf(k))
    }
}

/// MLP for Qwen3 text decoder (SwiGLU activation, quantized)
public class QuantizedTextMLP: Module {
    @ModuleInfo var gateProj: QuantizedLinear
    @ModuleInfo var upProj: QuantizedLinear
    @ModuleInfo var downProj: QuantizedLinear

    public init(config: TextDecoderConfig) {
        let hiddenSize = config.hiddenSize
        let intermediateSize = config.intermediateSize

        self._gateProj.wrappedValue = QuantizedLinear(
            hiddenSize, intermediateSize, bias: false,
            groupSize: config.groupSize, bits: config.bits)
        self._upProj.wrappedValue = QuantizedLinear(
            hiddenSize, intermediateSize, bias: false,
            groupSize: config.groupSize, bits: config.bits)
        self._downProj.wrappedValue = QuantizedLinear(
            intermediateSize, hiddenSize, bias: false,
            groupSize: config.groupSize, bits: config.bits)

        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // SwiGLU: down(silu(gate(x)) * up(x))
        let gate = silu(gateProj(x))
        let up = upProj(x)
        return downProj(gate * up)
    }
}

/// Decoder layer for Qwen3 text model (quantized)
public class QuantizedTextDecoderLayer: Module {
    @ModuleInfo var selfAttn: QuantizedTextAttention
    @ModuleInfo var mlp: QuantizedTextMLP
    @ModuleInfo var inputLayerNorm: RMSNorm
    @ModuleInfo var postAttentionLayerNorm: RMSNorm

    public init(config: TextDecoderConfig) {
        self._selfAttn.wrappedValue = QuantizedTextAttention(config: config)
        self._mlp.wrappedValue = QuantizedTextMLP(config: config)
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
        let residual = hiddenStates
        var hidden = inputLayerNorm(hiddenStates)
        let (attnOutput, newCache) = selfAttn(hidden, attentionMask: attentionMask, cache: cache)
        hidden = residual + attnOutput

        // MLP with pre-norm
        let residual2 = hidden
        hidden = postAttentionLayerNorm(hidden)
        hidden = mlp(hidden)
        hidden = residual2 + hidden

        return (hidden, newCache)
    }
}

/// Full Qwen3 text decoder model (quantized)
public class QuantizedTextModel: Module {
    public let config: TextDecoderConfig

    @ModuleInfo public var embedTokens: PreQuantizedEmbedding
    @ModuleInfo var layers: [QuantizedTextDecoderLayer]
    @ModuleInfo var norm: RMSNorm

    public init(config: TextDecoderConfig) {
        self.config = config

        self._embedTokens.wrappedValue = PreQuantizedEmbedding(
            embeddingCount: config.vocabSize,
            dimensions: config.hiddenSize,
            groupSize: config.groupSize,
            bits: config.bits)
        self._layers.wrappedValue = (0..<config.numLayers).map { _ in
            QuantizedTextDecoderLayer(config: config)
        }
        self._norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)

        super.init()
    }

    /// Create causal attention mask for autoregressive generation
    private func createCausalMask(seqLen: Int, cacheLen: Int) -> MLXArray {
        // Create mask for attention where each position can only attend to previous positions
        // Shape: [1, 1, seqLen, seqLen + cacheLen]
        let totalLen = seqLen + cacheLen

        // Create the causal mask - positions can attend to positions <= their own
        // Mask is 0 for allowed positions, -inf for masked positions
        var mask = MLXArray.zeros([seqLen, totalLen])

        // Fill upper triangle with -inf (positions that shouldn't be attended)
        for i in 0..<seqLen {
            let queryPos = i + cacheLen
            for j in (queryPos + 1)..<totalLen {
                mask[i, j] = MLXArray(Float(-1e9))
            }
        }

        // Add batch and head dimensions: [seqLen, totalLen] -> [1, 1, seqLen, totalLen]
        return mask.expandedDimensions(axes: [0, 1])
    }

    /// Forward pass through text decoder
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

        let seqLen = hiddenStates.dim(1)
        let cacheLen = cache?.first?.0.dim(2) ?? 0

        // Create causal attention mask
        let mask = attentionMask ?? createCausalMask(seqLen: seqLen, cacheLen: cacheLen)

        // Debug input
        let inputFlat = hiddenStates.flattened()
        print("DEBUG TextDecoder: Input embeds - mean: \(mean(inputFlat).item(Float.self)), std: \(sqrt(variance(inputFlat)).item(Float.self))")

        // Apply decoder layers
        var newCache: [(MLXArray, MLXArray)] = []
        for (i, layer) in layers.enumerated() {
            let layerCache = cache?[i]
            let (output, updatedCache) = layer(hiddenStates, attentionMask: mask, cache: layerCache)
            hiddenStates = output
            newCache.append(updatedCache)

            // Debug first and last layers
            if i == 0 || i == 27 {
                let layerFlat = hiddenStates.flattened()
                print("DEBUG TextDecoder: After layer \(i) - mean: \(mean(layerFlat).item(Float.self)), std: \(sqrt(variance(layerFlat)).item(Float.self))")
            }
        }

        // Final norm
        hiddenStates = norm(hiddenStates)
        let normFlat = hiddenStates.flattened()
        print("DEBUG TextDecoder: After final norm - mean: \(mean(normFlat).item(Float.self)), std: \(sqrt(variance(normFlat)).item(Float.self))")

        return (hiddenStates, newCache)
    }
}
