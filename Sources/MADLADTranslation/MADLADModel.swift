import Foundation
import MLX
import MLXFast
import MLXNN
import MLXCommon

// MARK: - T5 helpers

/// T5 v1.1 "gelu_new" approximate GeLU activation.
///
/// `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`
@inline(__always)
private func t5GeluNew(_ x: MLXArray) -> MLXArray {
    let c = MLXArray(Float(0.7978845608028654))   // sqrt(2 / pi)
    let k = MLXArray(Float(0.044715))
    let inner = c * (x + k * x * x * x)
    return MLXArray(Float(0.5)) * x * (MLXArray(Float(1.0)) + tanh(inner))
}

/// T5 relative position bucketing.
///
/// Maps an integer relative position (q_pos - k_pos) into one of
/// `numBuckets` buckets. Half the buckets handle exact small distances; the
/// other half are log-spaced up to `maxDistance`.
///
/// - Bidirectional (encoder self-attn): buckets distinguish past vs. future
///   so the upper half encodes positive offsets, the lower half negative.
/// - Unidirectional (decoder self-attn): only past positions contribute
///   (offsets are clamped to `<= 0`), and the full bucket budget covers them.
private func relativePositionBucket(
    qLen: Int, kLen: Int, bidirectional: Bool,
    numBuckets: Int, maxDistance: Int
) -> MLXArray {
    // Build a [qLen, kLen] grid of (k_pos - q_pos) — matches HF T5 reference.
    var matrix = [Int32]()
    matrix.reserveCapacity(qLen * kLen)
    for q in 0..<qLen {
        for k in 0..<kLen {
            matrix.append(Int32(k - q))
        }
    }
    var rel = MLXArray(matrix, [qLen, kLen])

    var buckets = MLXArray.zeros([qLen, kLen], dtype: .int32)
    var n = numBuckets

    if bidirectional {
        n /= 2
        // Positive offsets get shifted into the upper half.
        let isPositive = (rel .> MLXArray(Int32(0))).asType(.int32) * MLXArray(Int32(n))
        buckets = buckets + isPositive
        rel = abs(rel)
    } else {
        // Only attend to the past: clamp positive offsets to 0, then negate.
        rel = -minimum(rel, MLXArray(Int32(0)))
    }

    // Small (exact) range: [0, maxExact)
    let maxExact = n / 2
    let isSmall = rel .< MLXArray(Int32(maxExact))

    // Large (log-spaced) range: maxExact + log(rel / maxExact) / log(maxDist / maxExact) * (n - maxExact)
    let relF = rel.asType(.float32)
    let logRatio = log(relF / Float(maxExact)) / log(Float(maxDistance) / Float(maxExact))
    let large = MLXArray(Int32(maxExact)) + (logRatio * Float(n - maxExact)).asType(.int32)
    let largeClamped = minimum(large, MLXArray(Int32(n - 1)))

    let chosen = MLX.where(isSmall, rel, largeClamped)
    return buckets + chosen
}

// MARK: - T5 Attention

/// T5 attention: self-attention (encoder + decoder) and cross-attention.
///
/// Differences from standard SDPA attention:
/// - No `1/sqrt(d_k)` scaling — T5 attention scores are unscaled.
/// - Position information is injected via an additive **relative position
///   bias** computed from a learned `[numBuckets, numHeads]` table. Only the
///   first layer of each stack owns the table; subsequent layers receive the
///   precomputed bias as input and reuse it.
/// - No biases on Q/K/V/O projections.
/// - Cross-attention takes K/V from the encoder output and never uses
///   relative position bias.
public final class T5Attention: Module {
    public let numHeads: Int
    public let headDim: Int
    public let innerDim: Int
    public let dModel: Int
    public let isDecoder: Bool
    public let isCrossAttention: Bool
    public let hasRelativeAttentionBias: Bool
    public let numBuckets: Int
    public let maxDistance: Int

    @ModuleInfo var q: QuantizedLinear
    @ModuleInfo var k: QuantizedLinear
    @ModuleInfo var v: QuantizedLinear
    @ModuleInfo var o: QuantizedLinear

    /// Relative position bias table — only present when `hasRelativeAttentionBias`.
    /// Stored as a quantized embedding so the conversion script can pack it
    /// alongside the rest of the model. The table is small (e.g. 32×16 = 512
    /// floats), but using the same packing keeps weight loading uniform.
    @ModuleInfo(key: "relative_attention_bias") var relativeAttentionBias: Embedding?

    public init(
        config: MADLADTranslationConfig,
        isDecoder: Bool,
        isCrossAttention: Bool,
        hasRelativeAttentionBias: Bool,
        groupSize: Int,
        bits: Int
    ) {
        self.numHeads = config.numHeads
        self.headDim = config.dKv
        self.innerDim = config.numHeads * config.dKv
        self.dModel = config.dModel
        self.isDecoder = isDecoder
        self.isCrossAttention = isCrossAttention
        self.hasRelativeAttentionBias = hasRelativeAttentionBias
        self.numBuckets = config.relativeAttentionNumBuckets
        self.maxDistance = config.relativeAttentionMaxDistance

        self._q = ModuleInfo(wrappedValue: QuantizedLinear(
            dModel, innerDim, bias: false, groupSize: groupSize, bits: bits))
        self._k = ModuleInfo(wrappedValue: QuantizedLinear(
            dModel, innerDim, bias: false, groupSize: groupSize, bits: bits))
        self._v = ModuleInfo(wrappedValue: QuantizedLinear(
            dModel, innerDim, bias: false, groupSize: groupSize, bits: bits))
        self._o = ModuleInfo(wrappedValue: QuantizedLinear(
            innerDim, dModel, bias: false, groupSize: groupSize, bits: bits))

        if hasRelativeAttentionBias {
            self._relativeAttentionBias = ModuleInfo(
                wrappedValue: Embedding(embeddingCount: config.relativeAttentionNumBuckets,
                                         dimensions: config.numHeads),
                key: "relative_attention_bias")
        } else {
            self._relativeAttentionBias = ModuleInfo(wrappedValue: nil,
                                                     key: "relative_attention_bias")
        }

        super.init()
    }

    /// Compute the additive position bias `[1, numHeads, qLen, kLen]`.
    public func computeBias(qLen: Int, kLen: Int, offset: Int = 0) -> MLXArray {
        guard let table = relativeAttentionBias else {
            fatalError("computeBias called on layer without relative_attention_bias")
        }
        // For decoder self-attn during incremental decode, q is at position
        // `offset + i` while k spans `0..<kLen`. We model that by building the
        // bucket grid with q starting at `offset`.
        var matrix = [Int32]()
        matrix.reserveCapacity(qLen * kLen)
        for q in 0..<qLen {
            for kk in 0..<kLen {
                matrix.append(Int32(kk - (q + offset)))
            }
        }
        var rel = MLXArray(matrix, [qLen, kLen])

        var buckets = MLXArray.zeros([qLen, kLen], dtype: .int32)
        var n = numBuckets
        if !isDecoder {
            n /= 2
            let isPositive = (rel .> MLXArray(Int32(0))).asType(.int32) * MLXArray(Int32(n))
            buckets = buckets + isPositive
            rel = abs(rel)
        } else {
            rel = -minimum(rel, MLXArray(Int32(0)))
        }
        let maxExact = n / 2
        let isSmall = rel .< MLXArray(Int32(maxExact))
        let relF = rel.asType(.float32)
        let logRatio = log(relF / Float(maxExact)) / log(Float(maxDistance) / Float(maxExact))
        let large = MLXArray(Int32(maxExact)) + (logRatio * Float(n - maxExact)).asType(.int32)
        let largeClamped = minimum(large, MLXArray(Int32(n - 1)))
        let bucketIds = buckets + MLX.where(isSmall, rel, largeClamped)

        // Look up: [qLen, kLen] → [qLen, kLen, numHeads] → [1, numHeads, qLen, kLen]
        let values = table(bucketIds.flattened()).reshaped(qLen, kLen, numHeads)
        return values.transposed(2, 0, 1).expandedDimensions(axis: 0)
    }

    /// Forward pass.
    ///
    /// - Parameters:
    ///   - x: query input `[B, T_q, dModel]`
    ///   - keyValueStates: K/V input for cross-attention `[B, T_k, dModel]`,
    ///     or `nil` for self-attention (uses `x` for K/V).
    ///   - positionBias: Pre-computed bias `[1, H, T_q, T_k]` from layer 0
    ///     of this stack. Pass `nil` to compute it locally (only valid when
    ///     `hasRelativeAttentionBias` is true).
    ///   - selfAttnCache: Decoder self-attention KV cache `(K, V)` each
    ///     `[B, H, S_past, D]`. Concatenated with the new K/V from `x`.
    ///   - crossAttnCache: Cross-attention KV cache `(K, V)`. Computed once
    ///     from the encoder output and reused for every decode step.
    /// - Returns: `(output, positionBias, newSelfCache, newCrossCache)`.
    public func callAsFunction(
        _ x: MLXArray,
        keyValueStates: MLXArray? = nil,
        positionBias: MLXArray? = nil,
        selfAttnCache: (MLXArray, MLXArray)? = nil,
        crossAttnCache: (MLXArray, MLXArray)? = nil
    ) -> (MLXArray, MLXArray, (MLXArray, MLXArray)?, (MLXArray, MLXArray)?) {
        let batch = x.dim(0)
        let qLen = x.dim(1)

        let queries = q(x)
            .reshaped(batch, qLen, numHeads, headDim)
            .transposed(0, 2, 1, 3)  // [B, H, T_q, D]

        var keys: MLXArray
        var values: MLXArray
        var newSelfCache: (MLXArray, MLXArray)? = nil
        var newCrossCache: (MLXArray, MLXArray)? = nil

        if isCrossAttention {
            if let cached = crossAttnCache {
                keys = cached.0
                values = cached.1
                newCrossCache = cached  // preserve so caller doesn't overwrite with nil
            } else {
                guard let kv = keyValueStates else {
                    fatalError("Cross-attention requires keyValueStates")
                }
                let kLen = kv.dim(1)
                keys = k(kv).reshaped(batch, kLen, numHeads, headDim).transposed(0, 2, 1, 3)
                values = v(kv).reshaped(batch, kLen, numHeads, headDim).transposed(0, 2, 1, 3)
                newCrossCache = (keys, values)
            }
        } else {
            // Self-attention — use x for K/V, optionally concat with cached.
            let kvSource = keyValueStates ?? x
            let newLen = kvSource.dim(1)
            var newK = k(kvSource).reshaped(batch, newLen, numHeads, headDim).transposed(0, 2, 1, 3)
            var newV = v(kvSource).reshaped(batch, newLen, numHeads, headDim).transposed(0, 2, 1, 3)
            if let cache = selfAttnCache {
                newK = concatenated([cache.0, newK], axis: 2)
                newV = concatenated([cache.1, newV], axis: 2)
            }
            keys = newK
            values = newV
            newSelfCache = (keys, values)
        }

        let kLen = keys.dim(2)

        // Position bias — compute or reuse.
        let bias: MLXArray
        if isCrossAttention {
            // Cross-attention has no relative position bias. Use zeros so the
            // SDPA mask path receives a valid additive tensor.
            bias = positionBias ?? MLXArray.zeros([1, numHeads, qLen, kLen], dtype: queries.dtype)
        } else if let pb = positionBias {
            bias = pb
        } else {
            // First layer of stack — compute. Offset = past length so that
            // incremental decoder steps line up with cached positions.
            let offset = (selfAttnCache != nil) ? (kLen - qLen) : 0
            bias = computeBias(qLen: qLen, kLen: kLen, offset: offset).asType(queries.dtype)
        }

        // Decoder self-attention also needs a causal mask. Combine with bias.
        var mask = bias
        if isDecoder && !isCrossAttention {
            let pastLen = kLen - qLen
            let causal = MLXArray.tri(qLen, m: kLen, k: pastLen, type: Float.self) - 1
            let causalAdditive = (causal * Float.greatestFiniteMagnitude).asType(queries.dtype)
            mask = mask + causalAdditive.reshaped(1, 1, qLen, kLen)
        }

        // T5 attention has scale = 1.0 (NO 1/sqrt(d_k)).
        let attn = MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values,
            scale: 1.0, mask: mask)

        let merged = attn.transposed(0, 2, 1, 3).reshaped(batch, qLen, innerDim)
        let out = o(merged)
        return (out, bias, newSelfCache, newCrossCache)
    }
}

// MARK: - T5 Gated FFN

/// T5 v1.1 gated FFN: `wo(gelu(wi_0(x)) * wi_1(x))`.
public final class T5DenseGatedActDense: Module {
    @ModuleInfo(key: "wi_0") var wi0: QuantizedLinear
    @ModuleInfo(key: "wi_1") var wi1: QuantizedLinear
    @ModuleInfo var wo: QuantizedLinear

    public init(config: MADLADTranslationConfig, groupSize: Int, bits: Int) {
        self._wi0 = ModuleInfo(wrappedValue: QuantizedLinear(
            config.dModel, config.dFf, bias: false, groupSize: groupSize, bits: bits),
            key: "wi_0")
        self._wi1 = ModuleInfo(wrappedValue: QuantizedLinear(
            config.dModel, config.dFf, bias: false, groupSize: groupSize, bits: bits),
            key: "wi_1")
        self._wo = ModuleInfo(wrappedValue: QuantizedLinear(
            config.dFf, config.dModel, bias: false, groupSize: groupSize, bits: bits))
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        wo(t5GeluNew(wi0(x)) * wi1(x))
    }
}

// MARK: - Sub-layers (norm + residual)

/// T5 self-attention sublayer: `x + SelfAttn(RMSNorm(x))`.
public final class T5LayerSelfAttention: Module {
    @ModuleInfo(key: "SelfAttention") var selfAttention: T5Attention
    @ModuleInfo(key: "layer_norm") var layerNorm: RMSNorm

    public init(
        config: MADLADTranslationConfig,
        isDecoder: Bool,
        hasRelativeAttentionBias: Bool,
        groupSize: Int, bits: Int
    ) {
        self._selfAttention = ModuleInfo(
            wrappedValue: T5Attention(
                config: config, isDecoder: isDecoder,
                isCrossAttention: false,
                hasRelativeAttentionBias: hasRelativeAttentionBias,
                groupSize: groupSize, bits: bits),
            key: "SelfAttention")
        self._layerNorm = ModuleInfo(
            wrappedValue: RMSNorm(dimensions: config.dModel, eps: Float(config.layerNormEpsilon)),
            key: "layer_norm")
        super.init()
    }

    public func callAsFunction(
        _ x: MLXArray,
        positionBias: MLXArray? = nil,
        selfAttnCache: (MLXArray, MLXArray)? = nil
    ) -> (MLXArray, MLXArray, (MLXArray, MLXArray)?) {
        let normed = layerNorm(x)
        let (attnOut, bias, newCache, _) = selfAttention(
            normed, positionBias: positionBias, selfAttnCache: selfAttnCache)
        return (x + attnOut, bias, newCache)
    }
}

/// T5 cross-attention sublayer (decoder only): `x + CrossAttn(RMSNorm(x), encoder_out)`.
public final class T5LayerCrossAttention: Module {
    @ModuleInfo(key: "EncDecAttention") var encDecAttention: T5Attention
    @ModuleInfo(key: "layer_norm") var layerNorm: RMSNorm

    public init(config: MADLADTranslationConfig, groupSize: Int, bits: Int) {
        self._encDecAttention = ModuleInfo(
            wrappedValue: T5Attention(
                config: config, isDecoder: true, isCrossAttention: true,
                hasRelativeAttentionBias: false,
                groupSize: groupSize, bits: bits),
            key: "EncDecAttention")
        self._layerNorm = ModuleInfo(
            wrappedValue: RMSNorm(dimensions: config.dModel, eps: Float(config.layerNormEpsilon)),
            key: "layer_norm")
        super.init()
    }

    public func callAsFunction(
        _ x: MLXArray,
        encoderOutput: MLXArray,
        crossAttnCache: (MLXArray, MLXArray)? = nil
    ) -> (MLXArray, (MLXArray, MLXArray)?) {
        let normed = layerNorm(x)
        let (attnOut, _, _, newCache) = encDecAttention(
            normed, keyValueStates: encoderOutput,
            crossAttnCache: crossAttnCache)
        return (x + attnOut, newCache)
    }
}

/// T5 FFN sublayer: `x + FFN(RMSNorm(x))`.
public final class T5LayerFF: Module {
    @ModuleInfo(key: "DenseReluDense") var denseReluDense: T5DenseGatedActDense
    @ModuleInfo(key: "layer_norm") var layerNorm: RMSNorm

    public init(config: MADLADTranslationConfig, groupSize: Int, bits: Int) {
        self._denseReluDense = ModuleInfo(
            wrappedValue: T5DenseGatedActDense(config: config, groupSize: groupSize, bits: bits),
            key: "DenseReluDense")
        self._layerNorm = ModuleInfo(
            wrappedValue: RMSNorm(dimensions: config.dModel, eps: Float(config.layerNormEpsilon)),
            key: "layer_norm")
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        x + denseReluDense(layerNorm(x))
    }
}

// MARK: - Blocks

/// One encoder block: SelfAttn + FFN.
///
/// HF stores sub-layers as a `layer` list with indices `[0]`=SelfAttn, `[1]`=FFN.
/// We expose the same key path so weight loading can address them by index.
public final class T5EncoderBlock: Module {
    @ModuleInfo var layer: [Module]

    public var selfAttn: T5LayerSelfAttention { layer[0] as! T5LayerSelfAttention }
    public var ffn: T5LayerFF { layer[1] as! T5LayerFF }

    public init(
        config: MADLADTranslationConfig,
        hasRelativeAttentionBias: Bool,
        groupSize: Int, bits: Int
    ) {
        let sa = T5LayerSelfAttention(
            config: config, isDecoder: false,
            hasRelativeAttentionBias: hasRelativeAttentionBias,
            groupSize: groupSize, bits: bits)
        let ff = T5LayerFF(config: config, groupSize: groupSize, bits: bits)
        self._layer = ModuleInfo(wrappedValue: [sa, ff])
        super.init()
    }

    public func callAsFunction(
        _ x: MLXArray,
        positionBias: MLXArray? = nil
    ) -> (MLXArray, MLXArray) {
        let (h1, bias, _) = selfAttn(x, positionBias: positionBias)
        return (ffn(h1), bias)
    }
}

/// One decoder block: SelfAttn + CrossAttn + FFN.
public final class T5DecoderBlock: Module {
    @ModuleInfo var layer: [Module]

    public var selfAttn: T5LayerSelfAttention { layer[0] as! T5LayerSelfAttention }
    public var crossAttn: T5LayerCrossAttention { layer[1] as! T5LayerCrossAttention }
    public var ffn: T5LayerFF { layer[2] as! T5LayerFF }

    public init(
        config: MADLADTranslationConfig,
        hasRelativeAttentionBias: Bool,
        groupSize: Int, bits: Int
    ) {
        let sa = T5LayerSelfAttention(
            config: config, isDecoder: true,
            hasRelativeAttentionBias: hasRelativeAttentionBias,
            groupSize: groupSize, bits: bits)
        let ca = T5LayerCrossAttention(config: config, groupSize: groupSize, bits: bits)
        let ff = T5LayerFF(config: config, groupSize: groupSize, bits: bits)
        self._layer = ModuleInfo(wrappedValue: [sa, ca, ff])
        super.init()
    }

    public func callAsFunction(
        _ x: MLXArray,
        encoderOutput: MLXArray,
        positionBias: MLXArray? = nil,
        selfAttnCache: (MLXArray, MLXArray)? = nil,
        crossAttnCache: (MLXArray, MLXArray)? = nil
    ) -> (MLXArray, MLXArray, (MLXArray, MLXArray)?, (MLXArray, MLXArray)?) {
        let (h1, bias, newSelfCache) = selfAttn(
            x, positionBias: positionBias, selfAttnCache: selfAttnCache)
        let (h2, newCrossCache) = crossAttn(
            h1, encoderOutput: encoderOutput, crossAttnCache: crossAttnCache)
        return (ffn(h2), bias, newSelfCache, newCrossCache)
    }
}

// MARK: - Full model

/// Per-layer cache for one decoder forward step.
public struct DecoderLayerCache {
    public var selfAttn: (MLXArray, MLXArray)?
    public var crossAttn: (MLXArray, MLXArray)?

    public init() {
        self.selfAttn = nil
        self.crossAttn = nil
    }
}

/// MADLAD-400 T5 v1.1 encoder-decoder translation model.
///
/// Inference flow:
///   1. `encode(inputIds)` → `[1, T_src, dModel]` (run once per source sentence).
///   2. Loop: `decodeStep(token, encoderOutput, caches)` → next-token logits.
///      The first call uses `decoder_start_token_id` (= `<pad>` for MADLAD).
///      Cross-attn KV is computed once and cached; self-attn KV grows.
///   3. Stop on EOS (`</s>` = 1) or `maxTokens`.
public final class MADLADTranslationModel: Module {
    public let config: MADLADTranslationConfig
    public let groupSize: Int
    public let bits: Int

    @ModuleInfo var shared: PreQuantizedEmbedding
    @ModuleInfo var encoder: T5Stack
    @ModuleInfo var decoder: T5Stack
    @ModuleInfo(key: "lm_head") var lmHead: QuantizedLinear

    public init(config: MADLADTranslationConfig, groupSize: Int = 64, bits: Int = 4) {
        self.config = config
        self.groupSize = groupSize
        self.bits = bits

        self._shared = ModuleInfo(wrappedValue: PreQuantizedEmbedding(
            embeddingCount: config.vocabSize,
            dimensions: config.dModel,
            groupSize: groupSize, bits: bits))

        self._encoder = ModuleInfo(wrappedValue: T5Stack(
            config: config, isDecoder: false,
            groupSize: groupSize, bits: bits))
        self._decoder = ModuleInfo(wrappedValue: T5Stack(
            config: config, isDecoder: true,
            groupSize: groupSize, bits: bits))

        self._lmHead = ModuleInfo(wrappedValue: QuantizedLinear(
            config.dModel, config.vocabSize, bias: false,
            groupSize: groupSize, bits: bits), key: "lm_head")

        super.init()
    }

    /// Encode the source token ids.
    ///
    /// - Parameter inputIds: `[1, T_src]` token ids (target language token first).
    /// - Returns: encoder hidden states `[1, T_src, dModel]`.
    public func encode(inputIds: MLXArray) -> MLXArray {
        let embeds = shared(inputIds)
        return encoder.forwardEncoder(embeds)
    }

    /// One decoder forward step.
    ///
    /// - Parameters:
    ///   - inputIds: `[1, T_q]` decoder tokens for this step (usually 1).
    ///   - encoderOutput: cached encoder hidden states.
    ///   - caches: per-layer self-attn + cross-attn caches; mutated in place.
    /// - Returns: logits over the vocab `[1, T_q, vocabSize]`.
    public func decodeStep(
        inputIds: MLXArray,
        encoderOutput: MLXArray,
        caches: inout [DecoderLayerCache]
    ) -> MLXArray {
        let embeds = shared(inputIds)
        let h = decoder.forwardDecoder(
            embeds, encoderOutput: encoderOutput, caches: &caches)
        return lmHead(h)
    }
}

/// A stack of T5 blocks (encoder or decoder) plus a final RMSNorm.
public final class T5Stack: Module {
    public let config: MADLADTranslationConfig
    public let isDecoder: Bool

    @ModuleInfo var block: [Module]
    @ModuleInfo(key: "final_layer_norm") var finalLayerNorm: RMSNorm

    public init(
        config: MADLADTranslationConfig,
        isDecoder: Bool,
        groupSize: Int, bits: Int
    ) {
        self.config = config
        self.isDecoder = isDecoder

        let n = isDecoder ? config.numDecoderLayers : config.numLayers
        let blocks: [Module] = (0..<n).map { i in
            isDecoder
                ? T5DecoderBlock(
                    config: config, hasRelativeAttentionBias: i == 0,
                    groupSize: groupSize, bits: bits) as Module
                : T5EncoderBlock(
                    config: config, hasRelativeAttentionBias: i == 0,
                    groupSize: groupSize, bits: bits) as Module
        }
        self._block = ModuleInfo(wrappedValue: blocks)
        self._finalLayerNorm = ModuleInfo(
            wrappedValue: RMSNorm(dimensions: config.dModel, eps: Float(config.layerNormEpsilon)),
            key: "final_layer_norm")
        super.init()
    }

    public func forwardEncoder(_ embeds: MLXArray) -> MLXArray {
        var h = embeds
        var bias: MLXArray? = nil
        for b in block {
            let blk = b as! T5EncoderBlock
            let (newH, newBias) = blk(h, positionBias: bias)
            h = newH
            if bias == nil { bias = newBias }
        }
        return finalLayerNorm(h)
    }

    public func forwardDecoder(
        _ embeds: MLXArray,
        encoderOutput: MLXArray,
        caches: inout [DecoderLayerCache]
    ) -> MLXArray {
        var h = embeds
        var bias: MLXArray? = nil
        for (i, b) in block.enumerated() {
            let blk = b as! T5DecoderBlock
            let (newH, newBias, newSelf, newCross) = blk(
                h, encoderOutput: encoderOutput,
                positionBias: bias,
                selfAttnCache: caches[i].selfAttn,
                crossAttnCache: caches[i].crossAttn)
            h = newH
            if bias == nil { bias = newBias }
            caches[i].selfAttn = newSelf
            caches[i].crossAttn = newCross
        }
        return finalLayerNorm(h)
    }
}
