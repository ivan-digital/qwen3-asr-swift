import Foundation
import MLX
import MLXNN
import MLXFast

// MARK: - SinusoidalPositionEmbedding

/// Generates sinusoidal position embeddings for diffusion timesteps.
/// Maps a scalar timestep t in [0, 1] to a `dim`-dimensional embedding.
public class SinusoidalPositionEmbedding: Module {
    let dim: Int

    public init(dim: Int) {
        self.dim = dim
        super.init()
    }

    /// - Parameter t: `[B]` float timestep values in [0, 1]
    /// - Returns: `[B, dim]` sinusoidal embedding
    public func callAsFunction(_ t: MLXArray) -> MLXArray {
        let halfDim = dim / 2
        let logFactor = MLXArray(Float(log(10000.0)) / Float(halfDim - 1))
        let freqs = exp(MLXArray(0 ..< Int32(halfDim)).asType(.float32) * (-logFactor))
        // t: [B] -> [B, 1], freqs: [halfDim] -> [1, halfDim]
        let angles = t.expandedDimensions(axis: 1) * freqs.expandedDimensions(axis: 0)
        return concatenated([sin(angles), cos(angles)], axis: -1)
    }
}

// MARK: - TimestepEmbedding

/// MLP that projects sinusoidal timestep embedding to model dimension.
/// sinEmbed -> linear1 -> SiLU -> linear2
public class TimestepEmbedding: Module {
    @ModuleInfo(key: "sin_embed") var sinEmbed: SinusoidalPositionEmbedding
    @ModuleInfo var linear1: Linear
    @ModuleInfo var linear2: Linear

    public init(freqEmbedDim: Int, dim: Int) {
        self._sinEmbed.wrappedValue = SinusoidalPositionEmbedding(dim: freqEmbedDim)
        self._linear1.wrappedValue = Linear(freqEmbedDim, dim, bias: true)
        self._linear2.wrappedValue = Linear(dim, dim, bias: true)
        super.init()
    }

    /// - Parameter t: `[B]` timestep values
    /// - Returns: `[B, dim]` time conditioning embedding
    public func callAsFunction(_ t: MLXArray) -> MLXArray {
        var h = sinEmbed(t)       // [B, freqEmbedDim]
        h = linear1(h)            // [B, dim]
        h = silu(h)
        h = linear2(h)            // [B, dim]
        return h
    }
}

// MARK: - AdaLayerNormZero

/// Adaptive layer normalization that produces 6 modulation parameters
/// for both the attention and feedforward sub-layers.
///
/// Uses non-affine LayerNorm (no learnable weight/bias) followed by
/// conditioning-dependent scale and shift.
public class AdaLayerNormZero: Module {
    @ModuleInfo var linear: Linear
    let norm: LayerNorm

    public init(dim: Int) {
        self._linear.wrappedValue = Linear(dim, dim * 6, bias: true)
        self.norm = LayerNorm(dimensions: dim, eps: 1e-6, affine: false)
        super.init()
    }

    /// - Parameters:
    ///   - x: `[B, T, dim]` input hidden states
    ///   - emb: `[B, dim]` timestep conditioning embedding
    /// - Returns: (norm_x, gate_msa, shift_mlp, scale_mlp, gate_mlp)
    ///   where norm_x is `[B, T, dim]` and the rest are `[B, dim]`
    public func callAsFunction(
        _ x: MLXArray, emb: MLXArray
    ) -> (MLXArray, MLXArray, MLXArray, MLXArray, MLXArray) {
        // silu(emb) -> linear -> [B, dim*6] -> 6 chunks of [B, dim]
        let params = linear(silu(emb))  // [B, dim*6]
        let chunks = split(params, parts: 6, axis: -1)
        let shiftMSA = chunks[0]   // [B, dim]
        let scaleMSA = chunks[1]   // [B, dim]
        let gateMSA = chunks[2]    // [B, dim]
        let shiftMLP = chunks[3]   // [B, dim]
        let scaleMLP = chunks[4]   // [B, dim]
        let gateMLP = chunks[5]    // [B, dim]

        // Apply non-affine layer norm then modulate
        let normX = norm(x) * (1 + scaleMSA.expandedDimensions(axis: 1)) + shiftMSA.expandedDimensions(axis: 1)
        return (normX, gateMSA, shiftMLP, scaleMLP, gateMLP)
    }
}

// MARK: - AdaLayerNormZeroFinal

/// Final adaptive layer norm with only 2 modulation params (scale + shift).
public class AdaLayerNormZeroFinal: Module {
    @ModuleInfo var linear: Linear
    let norm: LayerNorm

    public init(dim: Int) {
        self._linear.wrappedValue = Linear(dim, dim * 2, bias: true)
        self.norm = LayerNorm(dimensions: dim, eps: 1e-6, affine: false)
        super.init()
    }

    /// - Parameters:
    ///   - x: `[B, T, dim]` hidden states
    ///   - emb: `[B, dim]` timestep embedding
    /// - Returns: `[B, T, dim]` normalized and modulated output
    public func callAsFunction(_ x: MLXArray, emb: MLXArray) -> MLXArray {
        let params = linear(silu(emb))  // [B, dim*2]
        let chunks = split(params, parts: 2, axis: -1)
        let scale = chunks[0]  // [B, dim]
        let shift = chunks[1]  // [B, dim]
        return norm(x) * (1 + scale.expandedDimensions(axis: 1)) + shift.expandedDimensions(axis: 1)
    }
}

// MARK: - DiTAttention

/// Self-attention for DiT (full attention, NOT GQA -- all heads are query heads).
/// Expects input in [B, T, dim] format.
public class DiTAttention: Module {
    let heads: Int
    let dimHead: Int
    let scale: Float

    @ModuleInfo(key: "to_q") var toQ: Linear
    @ModuleInfo(key: "to_k") var toK: Linear
    @ModuleInfo(key: "to_v") var toV: Linear
    @ModuleInfo(key: "to_out") var toOut: Linear

    public init(dim: Int, heads: Int, dimHead: Int) {
        self.heads = heads
        self.dimHead = dimHead
        self.scale = 1.0 / sqrt(Float(dimHead))

        let innerDim = heads * dimHead
        self._toQ.wrappedValue = Linear(dim, innerDim, bias: true)
        self._toK.wrappedValue = Linear(dim, innerDim, bias: true)
        self._toV.wrappedValue = Linear(dim, innerDim, bias: true)
        self._toOut.wrappedValue = Linear(innerDim, dim, bias: true)

        super.init()
    }

    /// - Parameters:
    ///   - x: `[B, T, dim]` input
    ///   - mask: optional attention mask `[B, 1, 1, T]` or broadcastable, additive (0 = attend, -inf = mask)
    ///   - rope: optional RoPE module (applied to q, k in [B, heads, T, dimHead] layout)
    /// - Returns: `[B, T, dim]`
    public func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil, rope: MLXNN.RoPE? = nil) -> MLXArray {
        let B = x.dim(0)
        let T = x.dim(1)

        var q = toQ(x)  // [B, T, heads*dimHead]
        var k = toK(x)  // [B, T, heads*dimHead]
        var v = toV(x)  // [B, T, heads*dimHead]

        // Reshape to [B, T, heads, dimHead] then transpose to [B, heads, T, dimHead]
        q = q.reshaped(B, T, heads, dimHead).transposed(0, 2, 1, 3)
        k = k.reshaped(B, T, heads, dimHead).transposed(0, 2, 1, 3)
        v = v.reshaped(B, T, heads, dimHead).transposed(0, 2, 1, 3)

        // Apply RoPE if provided
        if let rope = rope {
            q = rope(q)  // offset=0 default
            k = rope(k)
        }

        // Scaled dot-product attention: [B, heads, T, dimHead]
        let attnOut = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v,
            scale: scale, mask: mask)

        // Transpose back: [B, heads, T, dimHead] -> [B, T, heads, dimHead] -> [B, T, heads*dimHead]
        let output = attnOut.transposed(0, 2, 1, 3).reshaped(B, T, heads * dimHead)

        return toOut(output)
    }
}

// MARK: - DiTFeedForward

/// Feedforward network with GELU (tanh approximate) activation.
/// linear1 -> GELU(tanh) -> linear2
public class DiTFeedForward: Module {
    @ModuleInfo var linear1: Linear
    @ModuleInfo var linear2: Linear

    public init(dim: Int, ffDim: Int) {
        self._linear1.wrappedValue = Linear(dim, ffDim, bias: true)
        self._linear2.wrappedValue = Linear(ffDim, dim, bias: true)
        super.init()
    }

    /// - Parameter x: `[B, T, dim]`
    /// - Returns: `[B, T, dim]`
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = linear1(x)           // [B, T, ffDim]
        h = geluApproximate(h)       // GELU with tanh approximation
        h = linear2(h)               // [B, T, dim]
        return h
    }
}

// MARK: - DiTBlock

/// Single DiT transformer layer with AdaLN conditioning.
///
/// Applies adaptive layer norm before attention and feedforward, with gating.
public class DiTBlock: Module {
    @ModuleInfo(key: "attn_norm") var attnNorm: AdaLayerNormZero
    @ModuleInfo var attn: DiTAttention
    @ModuleInfo(key: "ff_norm") var ffNorm: LayerNorm
    @ModuleInfo var ff: DiTFeedForward

    public init(dim: Int, heads: Int, dimHead: Int, ffMult: Int) {
        self._attnNorm.wrappedValue = AdaLayerNormZero(dim: dim)
        self._attn.wrappedValue = DiTAttention(dim: dim, heads: heads, dimHead: dimHead)
        // Non-affine LayerNorm for feedforward pre-norm (modulated externally)
        self._ffNorm.wrappedValue = LayerNorm(dimensions: dim, eps: 1e-6, affine: false)
        self._ff.wrappedValue = DiTFeedForward(dim: dim, ffDim: dim * ffMult)
        super.init()
    }

    /// - Parameters:
    ///   - x: `[B, T, dim]` hidden states
    ///   - t: `[B, dim]` time embedding
    ///   - mask: optional attention mask
    ///   - rope: optional RoPE for positional encoding
    /// - Returns: `[B, T, dim]`
    public func callAsFunction(
        _ x: MLXArray, t: MLXArray, mask: MLXArray? = nil, rope: MLXNN.RoPE? = nil
    ) -> MLXArray {
        // 1. AdaLN for attention
        let (normX, gateMSA, shiftMLP, scaleMLP, gateMLP) = attnNorm(x, emb: t)

        // 2. Self-attention
        let attnOut = attn(normX, mask: mask, rope: rope)

        // 3. Residual with gating
        var h = x + gateMSA.expandedDimensions(axis: 1) * attnOut

        // 4. Feedforward with external modulation (non-affine norm + scale/shift from AdaLN)
        let ffNormX = ffNorm(h) * (1 + scaleMLP.expandedDimensions(axis: 1)) + shiftMLP.expandedDimensions(axis: 1)

        // 5. Feedforward
        let ffOut = ff(ffNormX)

        // 6. Residual with gating
        h = h + gateMLP.expandedDimensions(axis: 1) * ffOut

        return h
    }
}

// MARK: - ConvPositionEmbedding

/// Convolutional position embedding with two grouped convolutions + GELU.
///
/// Two Conv1d layers (groups=dim//64, kernel=31) with symmetric padding=15
/// inject local positional information. Each followed by GELU activation.
/// Residual connection adds the output back to the input.
public class ConvPositionEmbedding: Module {
    let kernelSize: Int
    let padding: Int
    @ModuleInfo var conv1: MLXNN.Conv1d
    @ModuleInfo var conv2: MLXNN.Conv1d

    public init(dim: Int, kernelSize: Int = 31) {
        self.kernelSize = kernelSize
        self.padding = kernelSize / 2  // 15 for k=31 (symmetric)
        let groups = dim / 64  // 1024 / 64 = 16
        self._conv1.wrappedValue = MLXNN.Conv1d(
            inputChannels: dim, outputChannels: dim,
            kernelSize: kernelSize, stride: 1, padding: padding,
            groups: groups, bias: true)
        self._conv2.wrappedValue = MLXNN.Conv1d(
            inputChannels: dim, outputChannels: dim,
            kernelSize: kernelSize, stride: 1, padding: padding,
            groups: groups, bias: true)
        super.init()
    }

    /// - Parameter x: `[B, T, dim]` (NLC format, matching MLXNN.Conv1d expectation)
    /// - Returns: `[B, T, dim]` with local positional information added
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let T = x.dim(1)
        var h = conv1(x)
        // Trim to original length if padding added extra
        if h.dim(1) > T { h = h[0..., 0..<T, 0...] }
        h = geluApproximate(h)
        h = conv2(h)
        if h.dim(1) > T { h = h[0..., 0..<T, 0...] }
        h = geluApproximate(h)
        return x + h
    }
}

// MARK: - InputEmbedding

/// Combines all conditioning inputs (noised mel, conditioning mel, text embedding,
/// speaker embedding) and projects to model dimension.
public class InputEmbedding: Module {
    let spkDim: Int
    @ModuleInfo var proj: Linear
    @ModuleInfo(key: "conv_pos_embed") var convPosEmbed: ConvPositionEmbedding

    public init(melDim: Int, muDim: Int, spkDim: Int, dim: Int) {
        self.spkDim = spkDim
        // Input: concatenation of [x, cond, textEmbed, spks] = [melDim, melDim, muDim, spkDim]
        let inputDim = melDim + melDim + muDim + spkDim
        self._proj.wrappedValue = Linear(inputDim, dim, bias: true)
        self._convPosEmbed.wrappedValue = ConvPositionEmbedding(dim: dim)
        super.init()
    }

    /// - Parameters:
    ///   - x: `[B, T, melDim]` noised mel spectrogram
    ///   - cond: `[B, T, melDim]` conditioning mel (zeros for non-streaming)
    ///   - textEmbed: `[B, T, muDim]` text/token conditioning (mu)
    ///   - spks: `[B, spkDim]` speaker embedding (already projected), or nil
    /// - Returns: `[B, T, dim]` combined and projected embedding
    public func callAsFunction(
        _ x: MLXArray, cond: MLXArray, textEmbed: MLXArray, spks: MLXArray?
    ) -> MLXArray {
        let B = x.dim(0)
        let T = x.dim(1)

        // Always include speaker embedding (zeros if nil) to match proj input dim
        let spksExpanded: MLXArray
        if let spks = spks {
            spksExpanded = repeated(spks.expandedDimensions(axis: 1), count: T, axis: 1)
        } else {
            spksExpanded = MLXArray.zeros([B, T, spkDim]).asType(x.dtype)
        }

        // Concatenate along feature dimension: [B, T, inputDim]
        var h = concatenated([x, cond, textEmbed, spksExpanded], axis: -1)

        // Project to model dimension
        h = proj(h)  // [B, T, dim]

        // Add convolutional position embedding
        h = convPosEmbed(h)  // [B, T, dim]

        return h
    }
}

// MARK: - DiT (Main Class)

/// Diffusion Transformer for CosyVoice3 flow matching decoder.
///
/// 22-layer pure DiT with AdaLN conditioning. Takes noised mel spectrograms
/// and conditioning signals, outputs predicted velocity field for flow matching.
public class DiT: Module {
    public let config: CosyVoiceDiTConfig

    @ModuleInfo(key: "time_embed") var timeEmbed: TimestepEmbedding
    @ModuleInfo(key: "input_embed") var inputEmbed: InputEmbedding
    let rotaryEmbed: MLXNN.RoPE
    @ModuleInfo(key: "transformer_blocks") var transformerBlocks: [DiTBlock]
    @ModuleInfo(key: "norm_out") var normOut: AdaLayerNormZeroFinal
    @ModuleInfo(key: "proj_out") var projOut: Linear

    public init(config: CosyVoiceDiTConfig) {
        self.config = config

        self._timeEmbed.wrappedValue = TimestepEmbedding(
            freqEmbedDim: config.freqEmbedDim, dim: config.dim)

        self._inputEmbed.wrappedValue = InputEmbedding(
            melDim: config.melDim, muDim: config.muDim,
            spkDim: config.spkDim, dim: config.dim)

        // RoPE: dimensions = dimHead (64), traditional=false (split-half), base=10000
        self.rotaryEmbed = MLXNN.RoPE(
            dimensions: config.dimHead, traditional: false, base: 10000)

        self._transformerBlocks.wrappedValue = (0 ..< config.depth).map { _ in
            DiTBlock(
                dim: config.dim, heads: config.heads,
                dimHead: config.dimHead, ffMult: config.ffMult)
        }

        self._normOut.wrappedValue = AdaLayerNormZeroFinal(dim: config.dim)
        self._projOut.wrappedValue = Linear(config.dim, config.melDim, bias: true)

        super.init()
    }

    /// Forward pass through the Diffusion Transformer.
    ///
    /// - Parameters:
    ///   - x: `[B, melDim, T]` noised mel spectrogram (transposed internally to `[B, T, melDim]`)
    ///   - mask: `[B, 1, T]` sequence mask (1 = valid, 0 = padding)
    ///   - mu: `[B, melDim, T]` text conditioning (transposed internally)
    ///   - t: `[B]` diffusion timestep in [0, 1]
    ///   - spks: `[B, spkDim]` speaker embedding (already projected from 192->80), or nil
    ///   - cond: `[B, melDim, T]` additional conditioning mel, or nil
    /// - Returns: `[B, melDim, T]` predicted velocity field
    public func callAsFunction(
        _ x: MLXArray,
        mask: MLXArray,
        mu: MLXArray,
        t: MLXArray,
        spks: MLXArray?,
        cond: MLXArray?
    ) -> MLXArray {
        // 1. Transpose from [B, melDim, T] to [B, T, melDim]
        let xT = x.transposed(0, 2, 1)
        let muT = mu.transposed(0, 2, 1)
        let condT: MLXArray
        if let cond = cond {
            condT = cond.transposed(0, 2, 1)
        } else {
            condT = MLXArray.zeros(like: xT)
        }

        // 2. Compute timestep embedding: [B] -> [B, dim]
        let tEmbed = timeEmbed(t)

        // 3. Combine inputs and project to model dim: [B, T, dim]
        var h = inputEmbed(xT, cond: condT, textEmbed: muT, spks: spks)

        // 4. Build attention mask from sequence mask
        // mask: [B, 1, T] -> convert to additive mask for SDPA [B, 1, 1, T]
        // where 1 = attend (0.0), 0 = ignore (-inf)
        let attnMask = MLX.where(
            mask.expandedDimensions(axis: 2),  // [B, 1, 1, T]
            MLXArray(Float(0.0)),
            MLXArray(Float(-1e9))
        ).asType(h.dtype)

        // 5. Run through transformer blocks
        for block in transformerBlocks {
            h = block(h, t: tEmbed, mask: attnMask, rope: rotaryEmbed)
        }

        // 6. Final norm and projection
        h = normOut(h, emb: tEmbed)   // [B, T, dim]
        h = projOut(h)                // [B, T, melDim]

        // 7. Transpose back to [B, melDim, T]
        return h.transposed(0, 2, 1)
    }
}
