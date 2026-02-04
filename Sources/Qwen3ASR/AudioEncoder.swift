import Foundation
import MLX
import MLXNN
import MLXFast

/// Audio encoder configuration matching Qwen3-ASR HuggingFace model
public struct Qwen3AudioEncoderConfig {
    public let dModel: Int           // 896
    public let encoderAttentionHeads: Int  // 14
    public let encoderFFNDim: Int    // 3584
    public let encoderLayers: Int    // 18
    public let numMelBins: Int       // 128
    public let maxSourcePositions: Int  // 1500
    public let outputDim: Int        // 1024
    public let downsampleHiddenSize: Int  // 480
    public let convChunksize: Int    // 500
    public let nWindow: Int          // 50 (chunk size = n_window * 2 = 100)
    public let nWindowInfer: Int     // 800
    public let dropout: Float        // 0.0
    public let attentionDropout: Float  // 0.0
    public let activationDropout: Float // 0.0
    public let layerNormEps: Float   // 1e-5
    public let convOutInputDim: Int  // 7680 (480 channels * 16 spatial positions)

    public static let `default` = Qwen3AudioEncoderConfig(
        dModel: 896,
        encoderAttentionHeads: 14,
        encoderFFNDim: 3584,
        encoderLayers: 18,
        numMelBins: 128,
        maxSourcePositions: 1500,
        outputDim: 1024,
        downsampleHiddenSize: 480,
        convChunksize: 500,
        nWindow: 50,
        nWindowInfer: 800,
        dropout: 0.0,
        attentionDropout: 0.0,
        activationDropout: 0.0,
        layerNormEps: 1e-5,
        convOutInputDim: 7680
    )
}

/// Multi-head self-attention for audio encoder layers
/// Weight names: self_attn.q_proj, k_proj, v_proj, out_proj
public class AudioSelfAttention: Module {
    let numHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") public var qProj: Linear
    @ModuleInfo(key: "k_proj") public var kProj: Linear
    @ModuleInfo(key: "v_proj") public var vProj: Linear
    @ModuleInfo(key: "out_proj") public var outProj: Linear

    public init(hiddenSize: Int, numHeads: Int) {
        self.numHeads = numHeads
        self.headDim = hiddenSize / numHeads
        self.scale = 1.0 / sqrt(Float(headDim))

        self._qProj.wrappedValue = Linear(hiddenSize, hiddenSize, bias: true)
        self._kProj.wrappedValue = Linear(hiddenSize, hiddenSize, bias: true)
        self._vProj.wrappedValue = Linear(hiddenSize, hiddenSize, bias: true)
        self._outProj.wrappedValue = Linear(hiddenSize, hiddenSize, bias: true)

        super.init()
    }

    public func callAsFunction(_ x: MLXArray, attentionMask: MLXArray? = nil) -> MLXArray {
        let (batch, seqLen, _) = (x.dim(0), x.dim(1), x.dim(2))

        var q = qProj(x)
        var k = kProj(x)
        var v = vProj(x)

        // Reshape to [batch, seq, numHeads, headDim] -> [batch, numHeads, seq, headDim]
        q = q.reshaped(batch, seqLen, numHeads, headDim).transposed(0, 2, 1, 3)
        k = k.reshaped(batch, seqLen, numHeads, headDim).transposed(0, 2, 1, 3)
        v = v.reshaped(batch, seqLen, numHeads, headDim).transposed(0, 2, 1, 3)

        // Scaled dot-product attention
        var attnWeights = matmul(q, k.transposed(0, 1, 3, 2)) * scale

        if let mask = attentionMask {
            attnWeights = attnWeights + mask
        }

        attnWeights = softmax(attnWeights, axis: -1)

        var out = matmul(attnWeights, v)
        out = out.transposed(0, 2, 1, 3).reshaped(batch, seqLen, numHeads * headDim)

        return outProj(out)
    }
}

/// Audio encoder transformer layer
/// Weight names: self_attn, self_attn_layer_norm, fc1, fc2, final_layer_norm
public class AudioEncoderLayer: Module {
    @ModuleInfo(key: "self_attn") public var selfAttn: AudioSelfAttention
    @ModuleInfo(key: "self_attn_layer_norm") public var selfAttnLayerNorm: LayerNorm
    @ModuleInfo public var fc1: Linear
    @ModuleInfo public var fc2: Linear
    @ModuleInfo(key: "final_layer_norm") public var finalLayerNorm: LayerNorm

    public init(hiddenSize: Int, numHeads: Int, ffnDim: Int, layerNormEps: Float) {
        self._selfAttn.wrappedValue = AudioSelfAttention(hiddenSize: hiddenSize, numHeads: numHeads)
        self._selfAttnLayerNorm.wrappedValue = LayerNorm(dimensions: hiddenSize, eps: layerNormEps)
        self._fc1.wrappedValue = Linear(hiddenSize, ffnDim, bias: true)
        self._fc2.wrappedValue = Linear(ffnDim, hiddenSize, bias: true)
        self._finalLayerNorm.wrappedValue = LayerNorm(dimensions: hiddenSize, eps: layerNormEps)

        super.init()
    }

    public func callAsFunction(_ x: MLXArray, attentionMask: MLXArray? = nil) -> MLXArray {
        // Self attention with residual
        var residual = x
        var hidden = selfAttnLayerNorm(x)
        hidden = selfAttn(hidden, attentionMask: attentionMask)
        hidden = residual + hidden

        // FFN with residual
        residual = hidden
        hidden = finalLayerNorm(hidden)
        hidden = fc1(hidden)
        hidden = gelu(hidden)
        hidden = fc2(hidden)
        hidden = residual + hidden

        return hidden
    }
}

/// Create sinusoidal position embeddings matching Python mlx-audio implementation
/// - Parameters:
///   - seqLen: Sequence length
///   - dModel: Model dimension (channels)
/// - Returns: Position embeddings [1, seqLen, dModel]
private func createSinusoidalPositionEmbeddings(seqLen: Int, dModel: Int) -> MLXArray {
    let halfDim = dModel / 2
    let maxTimescale: Float = 10000.0

    // Python formula: log_timescale_increment = log(max_timescale) / (channels // 2 - 1)
    // inv_timescales = exp(-log_timescale_increment * arange(channels // 2))
    let logTimescaleIncrement = log(maxTimescale) / Float(halfDim - 1)
    let invTimescales = exp(
        MLXArray(0..<halfDim).asType(.float32) * (-logTimescaleIncrement)
    )

    // Position indices: [0, 1, 2, ..., seqLen-1]
    let positions = MLXArray(0..<seqLen).asType(.float32)

    // Compute scaled_time: positions[:, None] * inv_timescales[None, :]
    // [seqLen, 1] * [1, halfDim] -> [seqLen, halfDim]
    let scaledTime = positions.expandedDimensions(axis: 1) * invTimescales.expandedDimensions(axis: 0)

    // Sin and cos embeddings
    let sinEmbed = sin(scaledTime)  // [seqLen, halfDim]
    let cosEmbed = cos(scaledTime)  // [seqLen, halfDim]

    // Concatenate [sin, cos] along axis 1 (NOT interleave!)
    // Python: concatenate([sin(scaled_time), cos(scaled_time)], axis=1)
    let posEmbed = concatenated([sinEmbed, cosEmbed], axis: 1)  // [seqLen, dModel]

    // Add batch dimension
    return posEmbed.expandedDimensions(axis: 0)  // [1, seqLen, dModel]
}

/// Full Qwen3-ASR Audio Encoder (audio_tower)
/// Matches HuggingFace weight structure exactly
public class Qwen3AudioEncoder: Module {
    public let config: Qwen3AudioEncoderConfig

    // Conv frontend - using 2D convolutions
    // Input: [batch, 1, mel=128, time] (single channel mel spectrogram image)
    // Weight format in safetensors: [out, in, kH, kW] -> transpose to MLX [out, kH, kW, in]
    @ModuleInfo public var conv2d1: Conv2d  // 1 -> 480 channels, 3x3 kernel
    @ModuleInfo public var conv2d2: Conv2d  // 480 -> 480 channels, 3x3, stride 2
    @ModuleInfo public var conv2d3: Conv2d  // 480 -> 480 channels, 3x3, stride 2

    // Output projection: flattened conv features -> d_model
    // Weight: [896, 7680] -> Linear(7680, 896)
    @ModuleInfo(key: "conv_out") public var convOut: Linear

    // Post layer norm
    @ModuleInfo(key: "ln_post") public var lnPost: LayerNorm

    // Projector to text model dimension
    // proj1: [896, 896] -> Linear(896, 896)
    // proj2: [1024, 896] -> Linear(896, 1024)
    @ModuleInfo public var proj1: Linear
    @ModuleInfo public var proj2: Linear

    // Transformer layers
    @ModuleInfo public var layers: [AudioEncoderLayer]

    public init(config: Qwen3AudioEncoderConfig = .default) {
        self.config = config

        // Conv2D layers for mel spectrogram processing
        // All three convs have stride 2 for 8x downsampling
        // Input: [batch, mel_bins=128, time, 1] in NHWC
        self._conv2d1.wrappedValue = Conv2d(
            inputChannels: 1,
            outputChannels: config.downsampleHiddenSize,  // 480
            kernelSize: IntOrPair(3),
            stride: IntOrPair(2),  // Changed from 1 to 2
            padding: IntOrPair(1)
        )
        self._conv2d2.wrappedValue = Conv2d(
            inputChannels: config.downsampleHiddenSize,
            outputChannels: config.downsampleHiddenSize,
            kernelSize: IntOrPair(3),
            stride: IntOrPair(2),
            padding: IntOrPair(1)
        )
        self._conv2d3.wrappedValue = Conv2d(
            inputChannels: config.downsampleHiddenSize,
            outputChannels: config.downsampleHiddenSize,
            kernelSize: IntOrPair(3),
            stride: IntOrPair(2),
            padding: IntOrPair(1)
        )

        // Output conv projection: flattened features (7680) -> d_model (896)
        self._convOut.wrappedValue = Linear(config.convOutInputDim, config.dModel, bias: false)

        // Post layer norm
        self._lnPost.wrappedValue = LayerNorm(dimensions: config.dModel, eps: config.layerNormEps)

        // Projector to text model dimension
        // proj1: 896 -> 896
        // proj2: 896 -> 1024
        self._proj1.wrappedValue = Linear(config.dModel, config.dModel, bias: true)
        self._proj2.wrappedValue = Linear(config.dModel, config.outputDim, bias: true)

        // Transformer layers
        self._layers.wrappedValue = (0..<config.encoderLayers).map { _ in
            AudioEncoderLayer(
                hiddenSize: config.dModel,
                numHeads: config.encoderAttentionHeads,
                ffnDim: config.encoderFFNDim,
                layerNormEps: config.layerNormEps
            )
        }

        super.init()
    }

    /// Calculate output length from input length using the chunking formula
    /// This matches the Python _get_feat_extract_output_lengths function
    private func getOutputLength(_ inputLength: Int) -> Int {
        let chunkSize = config.nWindow * 2  // 100
        let remainder = inputLength % chunkSize

        // Process remainder through conv downsampling formula
        var featLen = (remainder - 1) / 2 + 1  // First stride-2
        featLen = (featLen - 1) / 2 + 1        // Second stride-2
        featLen = (featLen - 1) / 2 + 1        // Third stride-2

        // Full chunks each produce 13 tokens
        let fullChunkTokens = (inputLength / chunkSize) * 13

        // Handle edge case when remainder is 0
        let remainderTokens = remainder > 0 ? max(featLen, 1) : 0

        return fullChunkTokens + remainderTokens
    }

    /// Process a single chunk through conv layers
    /// Input: [batch, mel=128, time, 1] in NHWC format
    /// Output: [batch, time_tokens, features=7680]
    private func processConvChunk(_ chunk: MLXArray) -> MLXArray {
        var x = chunk

        // Apply conv layers: 128 -> 64 -> 32 -> 16 in mel dimension
        // Time dimension also downsampled 8x
        x = conv2d1(x)
        x = gelu(x)
        x = conv2d2(x)
        x = gelu(x)
        x = conv2d3(x)
        x = gelu(x)

        // Shape after conv: [batch, mel/8=16, time/8, 480]
        let batch = x.dim(0)
        let height = x.dim(1)    // 16 (mel after 3x stride-2)
        let width = x.dim(2)     // time/8
        let channels = x.dim(3)  // 480

        // Flatten mel*channels: [batch, 16, time_tokens, 480] -> [batch, time_tokens, 16*480]
        // Transpose to [batch, time, mel, channels] then flatten last two
        x = x.transposed(0, 2, 1, 3)  // [batch, time, mel, channels]
        x = x.reshaped(batch, width, height * channels)  // [batch, time_tokens, 7680]

        return x
    }

    /// Create block attention mask for preventing cross-chunk attention
    /// Each block in cu_seqlens can only attend to itself
    private func createBlockAttentionMask(seqLen: Int, cuSeqlens: [Int]) -> MLXArray {
        // Create a mask filled with -1e9 (masked)
        var maskData = [Float](repeating: -1e9, count: seqLen * seqLen)

        // Allow attention within each block
        for i in 0..<(cuSeqlens.count - 1) {
            let start = cuSeqlens[i]
            let end = cuSeqlens[i + 1]
            for row in start..<end {
                for col in start..<end {
                    maskData[row * seqLen + col] = 0.0
                }
            }
        }

        // Return mask with shape [1, 1, seqLen, seqLen] for broadcasting
        return MLXArray(maskData, [seqLen, seqLen]).expandedDimensions(axes: [0, 1])
    }

    /// Process mel spectrogram with time chunking (matching Python mlx-audio exactly)
    /// Input: [batch, mel_bins, time]
    /// Output: [time', output_dim] (no batch dim, matching Python)
    public func callAsFunction(_ melFeatures: MLXArray) -> MLXArray {
        let batch = melFeatures.dim(0)
        let melBins = melFeatures.dim(1)
        let timeFrames = melFeatures.dim(2)
        let chunkSize = config.nWindow * 2  // 100

        print("DEBUG AudioEncoder: Input shape [batch=\(batch), mel=\(melBins), time=\(timeFrames)]")
        print("DEBUG AudioEncoder: Chunk size = \(chunkSize), expected output tokens = \(getOutputLength(timeFrames))")

        // Calculate number of chunks
        let numChunks = (timeFrames + chunkSize - 1) / chunkSize  // ceil division

        // Compute chunk lengths (all full except possibly last)
        var chunkLengths = [Int]()
        for i in 0..<numChunks {
            if i == numChunks - 1 {
                let remainder = timeFrames % chunkSize
                chunkLengths.append(remainder == 0 ? chunkSize : remainder)
            } else {
                chunkLengths.append(chunkSize)
            }
        }

        let maxChunkLen = chunkLengths.max() ?? chunkSize

        // Extract and pad chunks - stack as batch dimension for parallel processing
        var paddedChunks: [MLXArray] = []
        var pos = 0
        for i in 0..<numChunks {
            let clen = chunkLengths[i]
            // Extract chunk from first (and only) batch item
            let feat = melFeatures[0, 0..., pos..<(pos + clen)]  // [mel, clen]
            pos += clen

            // Pad if needed
            var chunk: MLXArray
            if clen < maxChunkLen {
                let padWidth = maxChunkLen - clen
                chunk = padded(feat, widths: [.init((low: 0, high: 0)), .init((low: 0, high: padWidth))])
            } else {
                chunk = feat
            }
            paddedChunks.append(chunk)
        }

        // Stack chunks as batch: [numChunks, mel, maxChunkLen]
        var paddedFeature = stacked(paddedChunks, axis: 0)

        // Add channel dim: [numChunks, mel, time, 1] for Conv2d (NHWC)
        var x = paddedFeature.expandedDimensions(axis: -1)

        print("DEBUG AudioEncoder: Padded feature shape [\(x.dim(0)), \(x.dim(1)), \(x.dim(2)), \(x.dim(3))]")

        // Process through conv layers
        x = conv2d1(x)
        x = gelu(x)
        x = conv2d2(x)
        x = gelu(x)
        x = conv2d3(x)
        x = gelu(x)

        // Shape after conv: [numChunks, freq=16, time', channels=480]
        let numChunksBatch = x.dim(0)
        let freq = x.dim(1)      // 16
        let timeAfterConv = x.dim(2)  // ~13 for 100 input frames
        let channels = x.dim(3)  // 480

        print("DEBUG AudioEncoder: After conv - shape [\(numChunksBatch), \(freq), \(timeAfterConv), \(channels)]")

        // Transpose and reshape: [numChunks, freq, time, channels] -> [numChunks, time, channels*freq]
        x = x.transposed(0, 2, 3, 1)  // [numChunks, time, channels, freq]
        x = x.reshaped(numChunksBatch, timeAfterConv, channels * freq)  // [numChunks, time, 7680]

        print("DEBUG AudioEncoder: After reshape - shape [\(x.dim(0)), \(x.dim(1)), \(x.dim(2))]")

        // Project through conv_out (7680 -> 896)
        x = convOut(x)

        print("DEBUG AudioEncoder: After conv_out - shape [\(x.dim(0)), \(x.dim(1)), \(x.dim(2))]")

        // Add sinusoidal position embeddings - same for each chunk!
        // This is the key difference from before: positions 0 to timeAfterConv-1 for EACH chunk
        let posEmbed = createSinusoidalPositionEmbeddings(seqLen: timeAfterConv, dModel: config.dModel)
        x = x + posEmbed  // Broadcasting: [numChunks, time, 896] + [1, time, 896]

        print("DEBUG AudioEncoder: After pos embed - mean: \(mean(x.flattened()).item(Float.self)), std: \(sqrt(variance(x.flattened())).item(Float.self))")

        // Calculate valid lengths after CNN for each chunk
        var featureLensAfterCnn = [Int]()
        for clen in chunkLengths {
            // Formula from Python: (((clen-1)//2 + 1 - 1)//2 + 1 - 1)//2 + 1
            var featLen = (clen - 1) / 2 + 1
            featLen = (featLen - 1) / 2 + 1
            featLen = (featLen - 1) / 2 + 1
            featureLensAfterCnn.append(featLen)
        }

        print("DEBUG AudioEncoder: Feature lens after CNN: \(featureLensAfterCnn)")

        // Extract valid portions and concatenate
        var hiddenList: [MLXArray] = []
        for i in 0..<numChunks {
            let validLen = featureLensAfterCnn[i]
            let chunkHidden = x[i, 0..<validLen, 0...]  // [validLen, 896]
            hiddenList.append(chunkHidden)
        }

        // Concatenate all valid hidden states: [totalTokens, 896]
        var hiddenStates = concatenated(hiddenList, axis: 0)
        let totalTokens = hiddenStates.dim(0)

        print("DEBUG AudioEncoder: Hidden states (concatenated) - shape [\(totalTokens), \(hiddenStates.dim(1))]")

        // Build cumulative sequence lengths for block attention mask
        let maxLenAfterCnn = featureLensAfterCnn.max() ?? 13
        let windowAfterCnn = maxLenAfterCnn * (config.nWindowInfer / (config.nWindow * 2))  // 13 * 8 = 104

        let totalCnnLen = getOutputLength(timeFrames)  // 260

        // Build chunk lengths for windowed attention (NOT starting with 0)
        var cuChunkLens = [Int]()
        let numFullWindows = totalCnnLen / windowAfterCnn
        for _ in 0..<numFullWindows {
            cuChunkLens.append(windowAfterCnn)
        }
        let windowRemainder = totalCnnLen % windowAfterCnn
        if windowRemainder != 0 {
            cuChunkLens.append(windowRemainder)
        }

        // Compute cumulative sums: [0, 104, 208, 260]
        var cuSeqlens = [0]
        var cumsum = 0
        for len in cuChunkLens {
            cumsum += len
            cuSeqlens.append(cumsum)
        }

        print("DEBUG AudioEncoder: cu_seqlens: \(cuSeqlens)")

        // Create block attention mask
        let attentionMask = createBlockAttentionMask(seqLen: totalTokens, cuSeqlens: cuSeqlens)

        // Add batch dimension for transformer: [1, totalTokens, 896]
        hiddenStates = hiddenStates.expandedDimensions(axis: 0)

        // Apply transformer layers with attention mask
        for layer in layers {
            hiddenStates = layer(hiddenStates, attentionMask: attentionMask)
        }

        // Remove batch dimension: [totalTokens, 896]
        hiddenStates = hiddenStates.squeezed(axis: 0)

        // Post processing
        hiddenStates = lnPost(hiddenStates)

        let postLnFlat = hiddenStates.flattened()
        print("DEBUG AudioEncoder: After ln_post - mean: \(mean(postLnFlat).item(Float.self)), std: \(sqrt(variance(postLnFlat)).item(Float.self))")

        // Project to text model dimension (GELU activation)
        hiddenStates = proj1(hiddenStates)
        hiddenStates = gelu(hiddenStates)
        hiddenStates = proj2(hiddenStates)

        let finalFlat = hiddenStates.flattened()
        print("DEBUG AudioEncoder: Final output - mean: \(mean(finalFlat).item(Float.self)), std: \(sqrt(variance(finalFlat)).item(Float.self)), shape [\(hiddenStates.dim(0)), \(hiddenStates.dim(1))]")

        return hiddenStates
    }
}
