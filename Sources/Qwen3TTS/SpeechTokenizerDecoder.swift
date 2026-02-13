import Foundation
import MLX
import MLXNN
import MLXFast
import Qwen3Common

// MARK: - Causal Conv1d

/// Conv1d with left padding for causality
public class CausalConv1d: Module {
    @ModuleInfo var conv: Conv1d
    let padAmount: Int

    public init(
        inputChannels: Int,
        outputChannels: Int,
        kernelSize: Int,
        stride: Int = 1,
        dilation: Int = 1,
        groups: Int = 1,
        bias: Bool = true
    ) {
        // For causal: pad = (kernel_size - 1) * dilation on the left only
        self.padAmount = (kernelSize - 1) * dilation
        self._conv.wrappedValue = Conv1d(
            inputChannels: inputChannels,
            outputChannels: outputChannels,
            kernelSize: kernelSize,
            stride: stride,
            padding: 0,
            dilation: dilation,
            groups: groups,
            bias: bias)

        super.init()
    }

    /// Input: [B, T, C] (NLC format)
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Left pad along time axis
        let padded = padAmount > 0
            ? padded(x, widths: [.init((low: 0, high: 0)), .init((low: padAmount, high: 0)), .init((low: 0, high: 0))])
            : x
        return conv(padded)
    }
}

// MARK: - Causal Transposed Conv1d

/// ConvTransposed1d with right trimming for causality
public class CausalTransposeConv1d: Module {
    @ModuleInfo var conv: ConvTransposed1d
    let trimRight: Int

    public init(
        inputChannels: Int,
        outputChannels: Int,
        kernelSize: Int,
        stride: Int = 1,
        bias: Bool = true
    ) {
        // trim = kernel_size - stride
        self.trimRight = kernelSize - stride
        self._conv.wrappedValue = ConvTransposed1d(
            inputChannels: inputChannels,
            outputChannels: outputChannels,
            kernelSize: kernelSize,
            stride: stride,
            padding: 0,
            bias: bias)

        super.init()
    }

    /// Input: [B, T, C] (NLC format)
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = conv(x)
        // Trim the right side to maintain causality
        if trimRight > 0 {
            let outLen = out.dim(1)
            out = out[0..., 0..<(outLen - trimRight), 0...]
        }
        return out
    }
}

// MARK: - SnakeBeta Activation

/// SnakeBeta activation: x + (1/exp(beta)) * sin^2(exp(alpha) * x)
/// Learnable params stored in log-space
public class SnakeBeta: Module {
    @ParameterInfo var alpha: MLXArray
    @ParameterInfo var beta: MLXArray

    public init(channels: Int) {
        // Initialize in log-space (exp(0) = 1.0)
        self._alpha.wrappedValue = MLXArray.zeros([1, 1, channels])
        self._beta.wrappedValue = MLXArray.zeros([1, 1, channels])

        super.init()
    }

    /// Input: [B, T, C] (NLC format)
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let a = exp(alpha)  // [1, 1, C]
        let b = exp(beta)   // [1, 1, C]
        let sinTerm = sin(a * x)
        return x + (1.0 / b) * (sinTerm * sinTerm)
    }
}

// MARK: - LayerScale

/// Per-channel learnable scale factor
public class LayerScale: Module {
    @ParameterInfo var scale: MLXArray

    public init(channels: Int, initValue: Float = 0.01) {
        self._scale.wrappedValue = MLXArray(Array(repeating: initValue, count: channels))
            .reshaped([1, 1, channels])

        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        x * scale
    }
}

// MARK: - ConvNeXt Block

/// ConvNeXt block: depthwise conv -> LayerNorm -> Linear -> GELU -> Linear -> LayerScale -> residual
public class ConvNeXtBlock: Module {
    @ModuleInfo var dwConv: CausalConv1d
    @ModuleInfo var norm: LayerNorm
    @ModuleInfo var pwConv1: Linear
    @ModuleInfo var pwConv2: Linear
    @ModuleInfo var layerScale: LayerScale

    public init(dim: Int, intermediateScale: Int = 4, kernelSize: Int = 7) {
        let intermediateDim = dim * intermediateScale

        // Depthwise (groups=dim)
        self._dwConv.wrappedValue = CausalConv1d(
            inputChannels: dim, outputChannels: dim,
            kernelSize: kernelSize, groups: dim)
        self._norm.wrappedValue = LayerNorm(dimensions: dim)
        self._pwConv1.wrappedValue = Linear(dim, intermediateDim)
        self._pwConv2.wrappedValue = Linear(intermediateDim, dim)
        self._layerScale.wrappedValue = LayerScale(channels: dim)

        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let residual = x
        var h = dwConv(x)
        h = norm(h)
        h = pwConv1(h)
        h = gelu(h)
        h = pwConv2(h)
        h = layerScale(h)
        return h + residual
    }
}

// MARK: - Decoder Residual Unit

/// Dilated residual unit: SnakeBeta -> CausalConv1d(dilated) -> SnakeBeta -> CausalConv1d(1x1) -> residual
public class DecoderResidualUnit: Module {
    @ModuleInfo var snake1: SnakeBeta
    @ModuleInfo var conv1: CausalConv1d
    @ModuleInfo var snake2: SnakeBeta
    @ModuleInfo var conv2: CausalConv1d

    public init(dim: Int, dilation: Int) {
        self._snake1.wrappedValue = SnakeBeta(channels: dim)
        self._conv1.wrappedValue = CausalConv1d(
            inputChannels: dim, outputChannels: dim,
            kernelSize: 7, dilation: dilation)
        self._snake2.wrappedValue = SnakeBeta(channels: dim)
        self._conv2.wrappedValue = CausalConv1d(
            inputChannels: dim, outputChannels: dim,
            kernelSize: 1)

        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let residual = x
        var h = snake1(x)
        h = conv1(h)
        h = snake2(h)
        h = conv2(h)
        return h + residual
    }
}

// MARK: - Decoder Block (Upsample)

/// Upsample block: SnakeBeta -> CausalTransposeConv1d -> 3x DecoderResidualUnit
public class DecoderBlock: Module {
    @ModuleInfo var snake: SnakeBeta
    @ModuleInfo var upsample: CausalTransposeConv1d
    @ModuleInfo var residualUnits: [DecoderResidualUnit]

    public init(inputDim: Int, outputDim: Int, stride: Int) {
        self._snake.wrappedValue = SnakeBeta(channels: inputDim)
        self._upsample.wrappedValue = CausalTransposeConv1d(
            inputChannels: inputDim, outputChannels: outputDim,
            kernelSize: stride * 2, stride: stride)
        // 3 residual units with dilations [1, 3, 9]
        self._residualUnits.wrappedValue = [1, 3, 9].map { dilation in
            DecoderResidualUnit(dim: outputDim, dilation: dilation)
        }

        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = snake(x)
        h = upsample(h)
        for unit in residualUnits {
            h = unit(h)
        }
        return h
    }
}

// MARK: - Decoder Transformer (Pre-Transformer)

/// Attention for codec decoder transformer — operates at hiddenSize=512
public class DecoderTransformerAttention: Module {
    let numHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo var qProj: Linear
    @ModuleInfo var kProj: Linear
    @ModuleInfo var vProj: Linear
    @ModuleInfo var oProj: Linear

    let rope: MLXNN.RoPE

    public init(hiddenSize: Int, numHeads: Int, headDim: Int, ropeTheta: Float = 10000.0) {
        self.numHeads = numHeads
        self.headDim = headDim
        self.scale = 1.0 / sqrt(Float(headDim))

        // Projections: hidden(512) -> numHeads*headDim(1024) for q/k/v
        self._qProj.wrappedValue = Linear(hiddenSize, numHeads * headDim, bias: false)
        self._kProj.wrappedValue = Linear(hiddenSize, numHeads * headDim, bias: false)
        self._vProj.wrappedValue = Linear(hiddenSize, numHeads * headDim, bias: false)
        self._oProj.wrappedValue = Linear(numHeads * headDim, hiddenSize, bias: false)

        self.rope = MLXNN.RoPE(dimensions: headDim, traditional: false, base: ropeTheta)

        super.init()
    }

    public func callAsFunction(
        _ x: MLXArray,
        attentionMask: MLXArray? = nil,
        cache: (MLXArray, MLXArray)? = nil
    ) -> (MLXArray, (MLXArray, MLXArray)) {
        let (batch, seqLen, _) = (x.dim(0), x.dim(1), x.dim(2))

        var q = qProj(x).reshaped(batch, seqLen, numHeads, headDim).transposed(0, 2, 1, 3)
        var k = kProj(x).reshaped(batch, seqLen, numHeads, headDim).transposed(0, 2, 1, 3)
        let v = vProj(x).reshaped(batch, seqLen, numHeads, headDim).transposed(0, 2, 1, 3)

        let offset = cache?.0.dim(2) ?? 0
        q = rope(q, offset: offset)
        k = rope(k, offset: offset)

        var cachedK = k
        var cachedV = v
        if let (prevK, prevV) = cache {
            cachedK = concatenated([prevK, k], axis: 2)
            cachedV = concatenated([prevV, v], axis: 2)
        }

        let attnOut = MLXFast.scaledDotProductAttention(
            queries: q, keys: cachedK, values: cachedV,
            scale: scale, mask: attentionMask)

        let out = oProj(attnOut.transposed(0, 2, 1, 3).reshaped(batch, seqLen, numHeads * headDim))
        return (out, (cachedK, cachedV))
    }
}

/// Decoder transformer layer — SwiGLU MLP + LayerScale
public class DecoderTransformerLayer: Module {
    @ModuleInfo var selfAttn: DecoderTransformerAttention
    @ModuleInfo var gateProj: Linear
    @ModuleInfo var upProj: Linear
    @ModuleInfo var downProj: Linear
    @ModuleInfo var norm1: RMSNorm  // input_layernorm
    @ModuleInfo var norm2: RMSNorm  // post_attention_layernorm
    @ModuleInfo var attnLayerScale: LayerScale
    @ModuleInfo var mlpLayerScale: LayerScale

    public init(config: SpeechTokenizerDecoderConfig) {
        let hidden = config.hiddenSize  // 512
        let intermediate = config.hiddenSize * 2  // 1024

        self._selfAttn.wrappedValue = DecoderTransformerAttention(
            hiddenSize: hidden,
            numHeads: config.numHeads,
            headDim: config.headDim)

        // SwiGLU MLP
        self._gateProj.wrappedValue = Linear(hidden, intermediate, bias: false)
        self._upProj.wrappedValue = Linear(hidden, intermediate, bias: false)
        self._downProj.wrappedValue = Linear(intermediate, hidden, bias: false)

        self._norm1.wrappedValue = RMSNorm(dimensions: hidden, eps: config.rmsNormEps)
        self._norm2.wrappedValue = RMSNorm(dimensions: hidden, eps: config.rmsNormEps)

        // LayerScale for attention and MLP residuals
        self._attnLayerScale.wrappedValue = LayerScale(channels: hidden)
        self._mlpLayerScale.wrappedValue = LayerScale(channels: hidden)

        super.init()
    }

    public func callAsFunction(
        _ x: MLXArray,
        attentionMask: MLXArray? = nil,
        cache: (MLXArray, MLXArray)? = nil
    ) -> (MLXArray, (MLXArray, MLXArray)) {
        let residual = x
        var h = norm1(x)
        let (attnOut, newCache) = selfAttn(h, attentionMask: attentionMask, cache: cache)
        h = residual + attnLayerScale(attnOut)

        let residual2 = h
        h = norm2(h)
        // SwiGLU: gate(x) * up(x), then down
        h = silu(gateProj(h)) * upProj(h)
        h = downProj(h)
        h = residual2 + mlpLayerScale(h)

        return (h, newCache)
    }
}

/// Full pre-transformer with input/output projections
public class DecoderTransformer: Module {
    @ModuleInfo var inputProj: Linear
    @ModuleInfo var layers: [DecoderTransformerLayer]
    @ModuleInfo var norm: RMSNorm
    @ModuleInfo var outputProj: Linear

    public init(config: SpeechTokenizerDecoderConfig) {
        // Project from latentDim(1024) to hiddenSize(512) for transformer
        self._inputProj.wrappedValue = Linear(config.latentDim, config.hiddenSize)
        self._layers.wrappedValue = (0..<config.numLayers).map { _ in
            DecoderTransformerLayer(config: config)
        }
        self._norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        // Project back from hiddenSize(512) to latentDim(1024)
        self._outputProj.wrappedValue = Linear(config.hiddenSize, config.latentDim)

        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = inputProj(x)  // [B, T, 1024] -> [B, T, 512]

        // Build causal mask for sequences longer than 1 (matches Python's create_additive_causal_mask)
        let seqLen = h.dim(1)
        let mask: MLXArray?
        if seqLen > 1 {
            let rows = MLXArray(0..<Int32(seqLen)).expandedDimensions(axis: 1)
            let cols = MLXArray(0..<Int32(seqLen)).expandedDimensions(axis: 0)
            mask = MLX.where(cols .> rows, MLXArray(Float(-1e9)), MLXArray(Float(0)))
                .expandedDimensions(axes: [0, 1])
                .asType(h.dtype)
        } else {
            mask = nil
        }

        for layer in layers {
            let (out, _) = layer(h, attentionMask: mask)
            h = out
        }
        h = norm(h)
        h = outputProj(h)  // [B, T, 512] -> [B, T, 1024]
        return h
    }
}

// MARK: - Residual Vector Quantizer

/// Single vector quantizer codebook
public class VectorQuantizerCodebook: Module {
    @ModuleInfo var embedding: Embedding

    public init(codebookSize: Int, codebookDim: Int) {
        self._embedding.wrappedValue = Embedding(
            embeddingCount: codebookSize, dimensions: codebookDim)

        super.init()
    }

    public func decode(_ indices: MLXArray) -> MLXArray {
        embedding(indices)
    }
}

/// Residual Vector Quantizer with output projection
public class ResidualVectorQuantizer: Module {
    @ModuleInfo var quantizers: [VectorQuantizerCodebook]
    @ModuleInfo var outputProj: Conv1d  // codebookDim -> hiddenSize via 1x1 conv
    let numQuantizers: Int

    public init(numQuantizers: Int, codebookSize: Int, codebookDim: Int, outputDim: Int) {
        self.numQuantizers = numQuantizers
        self._quantizers.wrappedValue = (0..<numQuantizers).map { _ in
            VectorQuantizerCodebook(codebookSize: codebookSize, codebookDim: codebookDim)
        }
        // Output projection: Conv1d(codebookDim, outputDim, kernel_size=1)
        self._outputProj.wrappedValue = Conv1d(
            inputChannels: codebookDim, outputChannels: outputDim,
            kernelSize: 1, bias: false)

        super.init()
    }

    /// Decode: sum embeddings from all quantizers, then project
    /// - Parameter codes: [B, numQuantizers, T] — codebook indices
    /// - Returns: [B, T, outputDim]
    public func decode(_ codes: MLXArray) -> MLXArray {
        var result: MLXArray?
        for i in 0..<numQuantizers {
            let quantizerCodes = codes[0..., i, 0...]  // [B, T]
            let decoded = quantizers[i].decode(quantizerCodes)  // [B, T, codebookDim]
            if let r = result {
                result = r + decoded
            } else {
                result = decoded
            }
        }
        // Apply output projection
        return outputProj(result!)  // [B, T, outputDim]
    }
}

/// Split RVQ: rvq_first (1 semantic) + rvq_rest (15 acoustic)
public class SplitResidualVectorQuantizer: Module {
    @ModuleInfo var rvqFirst: ResidualVectorQuantizer
    @ModuleInfo var rvqRest: ResidualVectorQuantizer

    public init(config: SpeechTokenizerDecoderConfig) {
        // rvq_first: 1 semantic quantizer
        self._rvqFirst.wrappedValue = ResidualVectorQuantizer(
            numQuantizers: 1,
            codebookSize: config.semanticCodebookSize,
            codebookDim: config.codebookDim,
            outputDim: config.hiddenSize)
        // rvq_rest: 15 acoustic quantizers
        self._rvqRest.wrappedValue = ResidualVectorQuantizer(
            numQuantizers: config.numQuantizers - 1,
            codebookSize: config.acousticCodebookSize,
            codebookDim: config.codebookDim,
            outputDim: config.hiddenSize)

        super.init()
    }

    /// Decode all 16 codebook indices to embeddings
    /// - Parameter codes: [B, 16, T] — all codebook indices
    /// - Returns: [B, T, hiddenSize=512]
    public func decode(_ codes: MLXArray) -> MLXArray {
        let firstCodes = codes[0..., 0..<1, 0...]   // [B, 1, T]
        let restCodes = codes[0..., 1..., 0...]      // [B, 15, T]

        let firstEmbed = rvqFirst.decode(firstCodes)    // [B, T, 512]
        let restEmbed = rvqRest.decode(restCodes)        // [B, T, 512]

        return firstEmbed + restEmbed
    }
}

// MARK: - Full Speech Tokenizer Decoder

/// Full Mimi-based speech tokenizer decoder
/// Converts 16-codebook indices to 24kHz audio waveform
public class SpeechTokenizerDecoder: Module {
    public let config: SpeechTokenizerDecoderConfig

    @ModuleInfo var splitRVQ: SplitResidualVectorQuantizer

    // Pre-conv: hiddenSize(512) -> latentDim(1024)
    @ModuleInfo var preConv: CausalConv1d

    // Pre-transformer: 1024 -> 512 bottleneck -> 1024
    @ModuleInfo var transformer: DecoderTransformer

    // Pre-upsample stages (upsampling_ratios = [2, 2])
    @ModuleInfo var preUpsample1: CausalTransposeConv1d
    @ModuleInfo var preConvNeXt1: ConvNeXtBlock
    @ModuleInfo var preUpsample2: CausalTransposeConv1d
    @ModuleInfo var preConvNeXt2: ConvNeXtBlock

    // Input conv to decoder dim
    @ModuleInfo var inputConv: CausalConv1d

    // Main decoder blocks (upsample_rates = [8, 5, 4, 3])
    @ModuleInfo var decoderBlocks: [DecoderBlock]

    // Final output
    @ModuleInfo var finalSnake: SnakeBeta
    @ModuleInfo var finalConv: CausalConv1d

    public init(config: SpeechTokenizerDecoderConfig) {
        self.config = config

        self._splitRVQ.wrappedValue = SplitResidualVectorQuantizer(config: config)

        // Pre-conv: 512 -> 1024, kernel_size=3
        self._preConv.wrappedValue = CausalConv1d(
            inputChannels: config.hiddenSize,
            outputChannels: config.latentDim,
            kernelSize: 3)

        // Transformer (8 layers with 512-dim bottleneck)
        self._transformer.wrappedValue = DecoderTransformer(config: config)

        // Pre-upsample stages: latentDim -> latentDim (2x each)
        let latent = config.latentDim
        self._preUpsample1.wrappedValue = CausalTransposeConv1d(
            inputChannels: latent, outputChannels: latent,
            kernelSize: config.upsamplingRatios[0] * 2, stride: config.upsamplingRatios[0])
        self._preConvNeXt1.wrappedValue = ConvNeXtBlock(dim: latent)
        self._preUpsample2.wrappedValue = CausalTransposeConv1d(
            inputChannels: latent, outputChannels: latent,
            kernelSize: config.upsamplingRatios[1] * 2, stride: config.upsamplingRatios[1])
        self._preConvNeXt2.wrappedValue = ConvNeXtBlock(dim: latent)

        // Input conv: latentDim -> decoderDim
        self._inputConv.wrappedValue = CausalConv1d(
            inputChannels: latent, outputChannels: config.decoderDim,
            kernelSize: 7)

        // Decoder blocks with reducing channel sizes
        // 1536 -> 768 -> 384 -> 192 -> 96
        var dims: [Int] = [config.decoderDim]
        for _ in config.upsampleRates {
            dims.append(dims.last! / 2)
        }

        self._decoderBlocks.wrappedValue = config.upsampleRates.enumerated().map { i, rate in
            DecoderBlock(inputDim: dims[i], outputDim: dims[i + 1], stride: rate)
        }

        // Final: SnakeBeta + Conv1d(7, 1) -> tanh
        let finalDim = dims.last!
        self._finalSnake.wrappedValue = SnakeBeta(channels: finalDim)
        self._finalConv.wrappedValue = CausalConv1d(
            inputChannels: finalDim, outputChannels: 1,
            kernelSize: 7)

        super.init()
    }

    /// Decode codebook indices to audio waveform
    public func callAsFunction(_ codes: MLXArray) -> MLXArray {
        // RVQ decode: [B, 16, T] -> [B, T, 512]
        var h = splitRVQ.decode(codes)

        // Pre-conv: [B, T, 512] -> [B, T, 1024]
        h = preConv(h)

        // Transformer: [B, T, 1024] -> input_proj -> 8 layers -> output_proj -> [B, T, 1024]
        h = transformer(h)

        // Pre-upsample (2x, 2x = 4x total)
        h = preUpsample1(h)
        h = preConvNeXt1(h)
        h = preUpsample2(h)
        h = preConvNeXt2(h)

        // Input conv to decoder dim
        h = inputConv(h)

        // Main decoder blocks (8x, 5x, 4x, 3x = 480x)
        for block in decoderBlocks {
            h = block(h)
        }

        // Final output
        h = finalSnake(h)
        h = finalConv(h)
        h = clip(h, min: -1.0, max: 1.0)

        return h  // [B, T*1920, 1]
    }

    /// Chunked decode: process codec frames in overlapping chunks to reduce O(T²) attention cost.
    /// Mirrors Python's `chunked_decode(codes, chunk_size=25, left_context=10)`.
    ///
    /// Each chunk processes `leftContext + chunkSize` frames through the full decoder pipeline.
    /// Only the last `chunkSize * samplesPerFrame` samples are kept (overlap is trimmed).
    /// All convolutions are causal (left-padded), so chunks produce correct output with left context.
    public func chunkedDecode(codes: MLXArray, chunkSize: Int = 25, leftContext: Int = 10) -> MLXArray {
        let numFrames = codes.dim(2)  // [B, 16, T]
        let samplesPerFrame = 1920    // 24000 / 12.5

        if numFrames <= chunkSize + leftContext {
            // Short enough to decode in one pass
            return callAsFunction(codes)
        }

        var audioChunks: [MLXArray] = []

        var offset = 0
        while offset < numFrames {
            let chunkEnd = min(offset + chunkSize, numFrames)
            let contextStart = max(offset - leftContext, 0)
            let actualContext = offset - contextStart

            let chunkCodes = codes[0..., 0..., contextStart..<chunkEnd]  // [B, 16, contextFrames + chunkFrames]
            let chunkWaveform = callAsFunction(chunkCodes)  // [B, T_samples, 1]

            // Trim left context samples
            let trimSamples = actualContext * samplesPerFrame
            let totalSamples = chunkWaveform.dim(1)
            let kept = chunkWaveform[0..., trimSamples..<totalSamples, 0...]

            audioChunks.append(kept)
            offset = chunkEnd
        }

        return concatenated(audioChunks, axis: 1)
    }

    /// Convert codes to float audio samples using chunked decoding + bulk extraction
    public func decode(codes: MLXArray) -> [Float] {
        let waveform = chunkedDecode(codes: codes)
        let flat = waveform.squeezed()
        eval(flat)
        return flat.asArray(Float.self)
    }

    /// Decode multiple items sequentially via existing `decode()`.
    /// Each item has its own length, so batching the decoder itself is deferred.
    /// - Parameter codesList: Array of `[1, 16, Ti]` codebook arrays
    /// - Returns: Array of float audio samples per item
    public func decodeBatch(codesList: [MLXArray]) -> [[Float]] {
        codesList.map { decode(codes: $0) }
    }
}
