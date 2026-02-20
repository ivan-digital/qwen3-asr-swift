import Foundation
import MLX
import MLXNN
import MLXFast

// MARK: - RepeatInterleaveUpsampler

/// Upsamples token embeddings from token rate (25 Hz) to mel rate (50 Hz) via repeat-interleave.
///
/// This is a pure function with no learnable parameters. Each input frame is repeated `ratio`
/// times along the time axis, matching Python's `torch.repeat_interleave(ratio, dim=1)`.
///
/// For `ratio=2`: `[a, b, c]` → `[a, a, b, b, c, c]`
public enum RepeatInterleaveUpsampler {

    /// Upsample by repeating each frame `ratio` times along the time axis.
    ///
    /// - Parameters:
    ///   - x: `[B, T, D]` input tensor
    ///   - ratio: integer repeat factor (e.g. 2 for 25 Hz -> 50 Hz)
    /// - Returns: `[B, T*ratio, D]` upsampled tensor
    public static func upsample(_ x: MLXArray, ratio: Int) -> MLXArray {
        guard ratio > 1 else { return x }

        // [B, T, D] → [B, T, 1, D] → repeat along axis 2 → [B, T, ratio, D] → [B, T*ratio, D]
        let expanded = x.expandedDimensions(axis: 2)             // [B, T, 1, D]
        let rep = repeated(expanded, count: ratio, axis: 2)      // [B, T, ratio, D]
        var shape = x.shape
        shape[1] *= ratio
        return rep.reshaped(shape)                                // [B, T*ratio, D]
    }
}

// MARK: - ConditionalFlowMatching

/// ODE solver with classifier-free guidance for flow matching.
///
/// Uses the Euler method to integrate the velocity field predicted by the DiT decoder,
/// applying classifier-free guidance (CFG) to improve sample quality. The ODE evolves
/// from pure noise (`t=0`) to the target distribution (`t=1`) over `nTimesteps` steps.
public class ConditionalFlowMatching: Module {
    public let config: CosyVoiceFlowConfig

    @ModuleInfo var decoder: DiT

    public init(config: CosyVoiceFlowConfig) {
        self.config = config
        self._decoder.wrappedValue = DiT(config: config.dit)
        super.init()
    }

    /// Solve the flow matching ODE to generate a mel spectrogram.
    ///
    /// The solver starts from Gaussian noise scaled by `temperature` and integrates
    /// using the Euler method with classifier-free guidance. At each timestep, the
    /// DiT is called with a doubled batch (conditioned + unconditioned) and the
    /// velocity is blended using `cfgRate`.
    ///
    /// - Parameters:
    ///   - mu: `[B, 80, T]` conditioning mel from the encoder
    ///   - mask: `[B, 1, T]` validity mask (1 = valid, 0 = padding)
    ///   - nTimesteps: number of ODE integration steps (default 10)
    ///   - temperature: noise scaling factor (default 1.0)
    ///   - spks: `[B, 80]` projected speaker embedding, or nil
    ///   - cond: `[B, 80, T]` additional conditioning, or nil
    /// - Returns: `[B, 80, T]` generated mel spectrogram
    public func forward(
        mu: MLXArray,
        mask: MLXArray,
        nTimesteps: Int = 10,
        temperature: Float = 1.0,
        spks: MLXArray? = nil,
        cond: MLXArray? = nil
    ) -> MLXArray {
        // 1. Sample initial noise: z ~ N(0, temperature^2 * I)
        let z = MLXRandom.normal(mu.shape).asType(mu.dtype) * MLXArray(temperature)

        // 2. Create time schedule with cosine mapping
        // Python: t_span = torch.linspace(0, 1, n_timesteps + 1)
        //         t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
        let tSchedule: [Float] = (0 ... nTimesteps).map { i in
            let t = Float(i) / Float(nTimesteps)
            return 1.0 - cos(t * 0.5 * .pi)
        }

        // 3. Euler solver with classifier-free guidance
        var x = z

        let cfgRate = MLXArray(config.cfgRate).asType(mu.dtype)

        for i in 0 ..< nTimesteps {
            let t = tSchedule[i]
            let dt = tSchedule[i + 1] - tSchedule[i]

            let dtScalar = MLXArray(dt).asType(mu.dtype)

            // Two separate forward passes for CFG (matching Python reference)
            let batchSize = x.dim(0)
            let tArr = MLXArray([Float](repeating: t, count: batchSize)).asType(mu.dtype)

            // Conditioned pass
            let vCond = decoder(x, mask: mask, mu: mu, t: tArr, spks: spks, cond: cond)

            // Unconditioned pass
            let vUncond = decoder(
                x, mask: mask,
                mu: MLXArray.zeros(mu.shape, dtype: mu.dtype),
                t: tArr,
                spks: spks.map { MLXArray.zeros($0.shape, dtype: $0.dtype) },
                cond: cond.map { MLXArray.zeros($0.shape, dtype: $0.dtype) }
            )

            // Apply classifier-free guidance:
            //   v = (1 + cfg_rate) * v_cond - cfg_rate * v_uncond
            let v = (1.0 + cfgRate) * vCond - cfgRate * vUncond

            // Euler step: x_{t+dt} = x_t + dt * v
            x = x + dtScalar * v

            // Evaluate to avoid building too large a computation graph
            eval(x)
        }

        return x
    }
}

// MARK: - PreLookaheadLayer

/// Causal convolution encoder before DiT.
/// Two Conv1d layers: conv1(80→1024, k=4) → ReLU → conv2(1024→80, k=3).
/// Adds local context to token embeddings before flow matching.
public class PreLookaheadLayer: Module {
    @ModuleInfo var conv1: CausalDilatedConv1d
    @ModuleInfo var conv2: CausalDilatedConv1d

    public init(inputDim: Int = 80, hiddenDim: Int = 1024) {
        // conv1: right-padding (look-ahead), kernel_size=4
        // Python: CausalConv1d(input_dim, hidden_dim, kernel_size, causal_type='right')
        self._conv1.wrappedValue = CausalDilatedConv1d(
            inputChannels: inputDim, outputChannels: hiddenDim, kernelSize: 4, causalType: .right)
        // conv2: left-padding (causal), kernel_size=3
        // Python: CausalConv1d(hidden_dim, input_dim, kernel_size - 1, causal_type='left')
        self._conv2.wrappedValue = CausalDilatedConv1d(
            inputChannels: hiddenDim, outputChannels: inputDim, kernelSize: 3)
        super.init()
    }

    /// Input: [B, C, T] (NCL) → Output: [B, C, T] (NCL)
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = conv1(x)
        h = relu(h)
        h = conv2(h)
        return h
    }
}

// MARK: - CosyVoiceFlowModel

/// Complete flow matching module for CosyVoice3.
///
/// Combines the speech token encoder (embedding → pre-lookahead → upsample) with
/// the conditional flow matching decoder (DiT + ODE solver) to produce mel spectrograms.
///
/// Pipeline:
/// 1. Embed speech tokens: `[B, T]` → `[B, T, 80]`
/// 2. Pre-lookahead conv encoder: `[B, 80, T]` → `[B, 80, T]`
/// 3. Upsample to mel rate: `[B, T, 80]` → `[B, T*2, 80]` (25 Hz → 50 Hz)
/// 4. Run flow matching ODE with DiT: `[B, 80, T*2]` → `[B, 80, T*2]`
public class CosyVoiceFlowModel: Module {
    public let config: CosyVoiceFlowConfig

    @ModuleInfo(key: "input_embedding") var inputEmbedding: Embedding
    @ModuleInfo(key: "spk_embed_affine_layer") var spkEmbedAffineLayer: Linear
    @ModuleInfo(key: "pre_lookahead_layer") var preLookaheadLayer: PreLookaheadLayer
    @ModuleInfo var decoder: ConditionalFlowMatching

    public init(config: CosyVoiceFlowConfig) {
        self.config = config

        // FSQ vocabulary embedding: 6561 tokens → 80 dims (mel dim directly)
        self._inputEmbedding.wrappedValue = Embedding(
            embeddingCount: config.vocabSize, dimensions: config.outputSize)

        // Speaker embedding projection: 192 → 80
        self._spkEmbedAffineLayer.wrappedValue = Linear(
            config.spkEmbedDim, config.outputSize, bias: true)

        // Pre-lookahead causal conv encoder: 80 → 1024 → 80
        self._preLookaheadLayer.wrappedValue = PreLookaheadLayer(
            inputDim: config.outputSize, hiddenDim: 1024)

        // Flow matching decoder (contains DiT)
        self._decoder.wrappedValue = ConditionalFlowMatching(config: config)

        super.init()
    }

    /// Generate a mel spectrogram from speech tokens.
    ///
    /// - Parameters:
    ///   - tokens: `[B, T]` speech token IDs (FSQ codes 0-6560)
    ///   - spkEmbedding: `[B, 192]` raw speaker embedding, or nil for single-speaker
    ///   - nTimesteps: ODE solver steps (default from config, typically 10)
    ///   - temperature: noise temperature for sampling (default 1.0)
    /// - Returns: `[B, 80, T_mel]` mel spectrogram where `T_mel = T * tokenMelRatio`
    public func callAsFunction(
        tokens: MLXArray,
        spkEmbedding: MLXArray? = nil,
        nTimesteps: Int? = nil,
        temperature: Float = 1.0
    ) -> MLXArray {
        let steps = nTimesteps ?? config.nTimesteps

        // 1. Embed tokens: [B, T] → [B, T, 80]
        var mu = inputEmbedding(tokens)

        // 2. Pre-lookahead conv encoder: [B, T, 80] → NCL → [B, 80, T] → NLC → [B, T, 80]
        mu = preLookaheadLayer(mu.transposed(0, 2, 1)).transposed(0, 2, 1)

        // 3. Upsample from token rate (25 Hz) to mel rate (50 Hz)
        //    [B, T, 80] → [B, T*2, 80]  (repeat_interleave, each frame duplicated)
        let muUpsampled = RepeatInterleaveUpsampler.upsample(mu, ratio: config.tokenMelRatio)
        let melLen = muUpsampled.dim(1)

        // 4. Transpose to [B, 80, T_mel] for DiT (expects channel-first)
        let muTransposed = muUpsampled.transposed(0, 2, 1)

        // 5. Create mask [B, 1, T_mel] (all ones — no padding)
        let batchSize = tokens.dim(0)
        let mask = MLXArray.ones([batchSize, 1, melLen]).asType(muTransposed.dtype)

        // 6. Project speaker embedding if provided
        //    L2-normalize first, then affine projection: [B, 192] → [B, 80]
        let spks: MLXArray? = spkEmbedding.map { emb in
            let norm = sqrt(sum(emb * emb, axis: -1, keepDims: true)) + 1e-8
            let normalized = emb / norm
            return spkEmbedAffineLayer(normalized)
        }

        // 7. Run flow matching ODE solver
        let mel = decoder.forward(
            mu: muTransposed,
            mask: mask,
            nTimesteps: steps,
            temperature: temperature,
            spks: spks,
            cond: nil
        )

        return mel  // [B, 80, T_mel]
    }
}
