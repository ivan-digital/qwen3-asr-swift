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

    public init(config: TextDecoderConfig) {
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

        // MLXFast RoPE: split-half rotation (traditional=false), base from config
        self.rope = MLXNN.RoPE(dimensions: headDim, traditional: false, base: config.ropeTheta)

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

        // Apply MLXFast RoPE (handles split-half rotation via optimized Metal kernel)
        queries = rope(queries, offset: offset)
        keys = rope(keys, offset: offset)

        // Update cache
        var cachedKeys = keys
        var cachedValues = values

        if let (prevKeys, prevValues) = cache {
            cachedKeys = concatenated([prevKeys, keys], axis: 2)
            cachedValues = concatenated([prevValues, values], axis: 2)
        }

        // SDPA handles GQA natively (N_q != N_kv), no need to tile KV heads
        let attnOutput = MLXFast.scaledDotProductAttention(
            queries: queries, keys: cachedKeys, values: cachedValues,
            scale: scale, mask: attentionMask)

        // SDPA returns [B, N_q, T_q, D], transpose to [B, T_q, N_q, D] then reshape
        let output = oProj(attnOutput.transposed(0, 2, 1, 3).reshaped(batch, seqLen, numHeads * headDim))

        return (output, (cachedKeys, cachedValues))
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

        // Determine attention mask:
        // - If caller provides a mask, use it
        // - seqLen == 1 (autoregressive step): nil mask — single query attends to all cached positions
        // - seqLen > 1 with no cache (prefill): use SDPA causal mode (handled inside attention via nil mask + SDPA causal)
        // For prefill with seqLen > 1, we create a causal mask using MLX operations
        let mask: MLXArray?
        if let providedMask = attentionMask {
            mask = providedMask
        } else if seqLen == 1 {
            // Autoregressive: single query can attend to all cached positions, no mask needed
            mask = nil
        } else {
            // Prefill: create causal mask using MLX broadcast operations
            let cacheLen = cache?.first?.0.dim(2) ?? 0
            let totalLen = seqLen + cacheLen
            // Row indices [0..seqLen) + cacheLen, column indices [0..totalLen)
            let rows = (MLXArray(0..<Int32(seqLen)) + Int32(cacheLen)).expandedDimensions(axis: 1)
            let cols = MLXArray(0..<Int32(totalLen)).expandedDimensions(axis: 0)
            // mask[i,j] = 0 where col <= row (attend), -1e9 where col > row (block)
            // Cast to match hidden states dtype (bfloat16 from quantized embeddings)
            mask = MLX.where(cols .> rows, MLXArray(Float(-1e9)), MLXArray(Float(0)))
                .expandedDimensions(axes: [0, 1])
                .asType(hiddenStates.dtype)
        }

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
