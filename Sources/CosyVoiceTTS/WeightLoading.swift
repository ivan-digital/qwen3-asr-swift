import Foundation
import MLX
import MLXNN
import Qwen3Common

/// Weight loading for CosyVoice3 TTS components (LLM, Flow/DiT, HiFi-GAN).
///
/// Loads from three separate safetensors files produced by the conversion script:
/// - `llm.safetensors`: Qwen2.5-0.5B based speech token generator
/// - `flow.safetensors`: Conditional flow matching with DiT decoder
/// - `hifigan.safetensors`: Neural source filter vocoder
///
/// The conversion script already remaps Python keys to match our module hierarchy
/// and transposes Conv1d weights to MLX format `[out, kernel, in]`.
public enum CosyVoiceWeightLoader {

    // MARK: - LLM

    /// Load LLM weights from llm.safetensors into the CosyVoiceLLM module.
    ///
    /// Expected keys (after conversion script remapping):
    /// - text_embedding.weight
    /// - speech_embedding.weight
    /// - layers.{i}.self_attn.q_proj.weight/.scales/.biases (quantized)
    /// - layers.{i}.self_attn.q_proj.bias (non-quantized bias)
    /// - layers.{i}.self_attn.k_proj/v_proj/o_proj (same pattern)
    /// - layers.{i}.self_attn.q_norm.weight
    /// - layers.{i}.self_attn.k_norm.weight
    /// - layers.{i}.input_layernorm.weight
    /// - layers.{i}.post_attention_layernorm.weight
    /// - layers.{i}.mlp.gate_proj/up_proj/down_proj (quantized)
    /// - norm.weight
    /// - speech_head.weight/.scales/.biases (quantized)
    public static func loadLLM(_ llm: CosyVoiceLLM, from url: URL) throws {
        let weights = try CommonWeightLoader.loadSafetensors(url: url)

        // Text embedding (not quantized)
        CommonWeightLoader.applyEmbeddingWeights(
            to: llm.textEmbedding, prefix: "text_embedding", from: weights)

        // Speech embedding (not quantized)
        CommonWeightLoader.applyEmbeddingWeights(
            to: llm.speechEmbedding, prefix: "speech_embedding", from: weights)

        // Transformer layers
        for (i, layer) in llm.layers.enumerated() {
            let prefix = "layers.\(i)"

            // Attention projections (quantized)
            CommonWeightLoader.applyQuantizedLinearWeights(
                to: layer.selfAttn.qProj, prefix: "\(prefix).self_attn.q_proj", from: weights)
            CommonWeightLoader.applyQuantizedLinearWeights(
                to: layer.selfAttn.kProj, prefix: "\(prefix).self_attn.k_proj", from: weights)
            CommonWeightLoader.applyQuantizedLinearWeights(
                to: layer.selfAttn.vProj, prefix: "\(prefix).self_attn.v_proj", from: weights)
            CommonWeightLoader.applyQuantizedLinearWeights(
                to: layer.selfAttn.oProj, prefix: "\(prefix).self_attn.o_proj", from: weights)

            // Layer norms
            CommonWeightLoader.applyRMSNormWeights(
                to: layer.inputLayerNorm, prefix: "\(prefix).input_layernorm", from: weights)
            CommonWeightLoader.applyRMSNormWeights(
                to: layer.postAttentionLayerNorm, prefix: "\(prefix).post_attention_layernorm", from: weights)

            // MLP (quantized SwiGLU)
            CommonWeightLoader.applyQuantizedMLPWeights(
                to: layer.mlp, prefix: "\(prefix).mlp", from: weights)
        }

        // Final norm
        CommonWeightLoader.applyRMSNormWeights(
            to: llm.norm, prefix: "norm", from: weights)

        // Speech head (quantized)
        CommonWeightLoader.applyQuantizedLinearWeights(
            to: llm.speechHead, prefix: "speech_head", from: weights)
    }

    // MARK: - Flow (DiT Decoder)

    /// Load flow weights from flow.safetensors into the CosyVoiceFlowModel.
    ///
    /// Expected keys (after conversion script remapping):
    /// - input_embedding.weight
    /// - spk_embed_affine_layer.weight/bias
    /// - encoder.* (pre-lookahead encoder layers)
    /// - encoder_proj.weight/bias
    /// - decoder.time_embed.*
    /// - decoder.input_embed.*
    /// - decoder.transformer_blocks.{i}.*
    /// - decoder.norm_out.*
    /// - decoder.proj_out.*
    public static func loadFlow(_ flow: CosyVoiceFlowModel, from url: URL) throws {
        let weights = try CommonWeightLoader.loadSafetensors(url: url)

        // Input embedding (speech token -> 80 mel dims directly)
        CommonWeightLoader.applyEmbeddingWeights(
            to: flow.inputEmbedding, prefix: "input_embedding", from: weights)

        // Speaker embedding affine projection (192 -> 80)
        CommonWeightLoader.applyLinearWeights(
            to: flow.spkEmbedAffineLayer, prefix: "spk_embed_affine_layer", from: weights)

        // Pre-lookahead layer: conv1 (80→1024, k=4) + conv2 (1024→80, k=3)
        CommonWeightLoader.applyConv1dWeights(
            to: flow.preLookaheadLayer.conv1.conv,
            prefix: "pre_lookahead_layer.conv1", from: weights, transpose: false)
        CommonWeightLoader.applyConv1dWeights(
            to: flow.preLookaheadLayer.conv2.conv,
            prefix: "pre_lookahead_layer.conv2", from: weights, transpose: false)

        // DiT decoder weights
        loadDiT(flow.decoder.decoder, prefix: "decoder", from: weights)
    }

    /// Load DiT weights from a weight dictionary with a given prefix.
    ///
    /// The DiT has: timestep embedding MLP, input embedding (with conv pos embed),
    /// RoPE (no learnable params), transformer blocks, final adaptive norm, output projection.
    static func loadDiT(_ dit: DiT, prefix: String, from weights: [String: MLXArray]) {
        // Time embedding MLP
        // SinusoidalPositionEmbedding has no learnable parameters.
        // TimestepEmbedding: sin_embed (no params) -> linear1 -> SiLU -> linear2
        // Python keys: time_embed.time_mlp.0 (linear1), time_embed.time_mlp.2 (linear2)
        CommonWeightLoader.applyQuantizedLinearWeights(
            to: dit.timeEmbed.linear1, prefix: "\(prefix).time_embed.time_mlp.0", from: weights)
        CommonWeightLoader.applyQuantizedLinearWeights(
            to: dit.timeEmbed.linear2, prefix: "\(prefix).time_embed.time_mlp.2", from: weights)

        // Input embedding: projection + causal conv position embedding
        CommonWeightLoader.applyQuantizedLinearWeights(
            to: dit.inputEmbed.proj, prefix: "\(prefix).input_embed.proj", from: weights)

        // Conv position embedding: two grouped Conv1d layers
        // Keys are conv1.0 and conv2.0 (from nn.Sequential wrapping)
        CommonWeightLoader.applyConv1dWeights(
            to: dit.inputEmbed.convPosEmbed.conv1,
            prefix: "\(prefix).input_embed.conv_pos_embed.conv1.0",
            from: weights, transpose: false)
        CommonWeightLoader.applyConv1dWeights(
            to: dit.inputEmbed.convPosEmbed.conv2,
            prefix: "\(prefix).input_embed.conv_pos_embed.conv2.0",
            from: weights, transpose: false)

        // Transformer blocks
        for (i, block) in dit.transformerBlocks.enumerated() {
            let blockPrefix = "\(prefix).transformer_blocks.\(i)"

            // AdaLN attention norm: projects timestep embedding to 6 modulation params
            CommonWeightLoader.applyQuantizedLinearWeights(
                to: block.attnNorm.linear, prefix: "\(blockPrefix).attn_norm.linear", from: weights)

            // Self-attention (full attention, not GQA)
            // Python keys use to_q, to_k, to_v, to_out.0
            CommonWeightLoader.applyQuantizedLinearWeights(
                to: block.attn.toQ, prefix: "\(blockPrefix).attn.to_q", from: weights)
            CommonWeightLoader.applyQuantizedLinearWeights(
                to: block.attn.toK, prefix: "\(blockPrefix).attn.to_k", from: weights)
            CommonWeightLoader.applyQuantizedLinearWeights(
                to: block.attn.toV, prefix: "\(blockPrefix).attn.to_v", from: weights)
            CommonWeightLoader.applyQuantizedLinearWeights(
                to: block.attn.toOut, prefix: "\(blockPrefix).attn.to_out.0", from: weights)

            // Feedforward: GELU(tanh) MLP
            // Python keys: ff.ff.0.0 (linear1 after GELU wrapper), ff.ff.2 (linear2)
            CommonWeightLoader.applyQuantizedLinearWeights(
                to: block.ff.linear1, prefix: "\(blockPrefix).ff.ff.0.0", from: weights)
            CommonWeightLoader.applyQuantizedLinearWeights(
                to: block.ff.linear2, prefix: "\(blockPrefix).ff.ff.2", from: weights)
        }

        // Final adaptive norm: projects to 2 modulation params (scale + shift)
        CommonWeightLoader.applyQuantizedLinearWeights(
            to: dit.normOut.linear, prefix: "\(prefix).norm_out.linear", from: weights)

        // Output projection: model dim -> mel dim (NOT quantized: 80 % 64 != 0)
        CommonWeightLoader.applyLinearWeights(
            to: dit.projOut, prefix: "\(prefix).proj_out", from: weights)
    }

    // MARK: - HiFi-GAN

    /// Load HiFi-GAN weights from hifigan.safetensors.
    ///
    /// Expected keys (after conversion script remapping):
    /// - conv_pre.weight/bias (already transposed by conversion script)
    /// - ups.{i}.weight/bias
    /// - resblocks.{stage*3+kernel_idx}.convs1.{dilation_idx}.weight/bias
    /// - resblocks.{stage*3+kernel_idx}.convs2.{dilation_idx}.weight/bias
    /// - resblocks.{...}.activations1.{...}.alpha/beta
    /// - resblocks.{...}.activations2.{...}.alpha/beta
    /// - source_downs.{i}.weight/bias
    /// - source_resblocks.{i}.*
    /// - conv_post.weight/bias
    /// - m_source.l_linear.weight/bias
    /// - f0_predictor.condnet.{i}.weight/bias
    /// - f0_predictor.classifier.weight/bias
    /// - up_activations.{i}.alpha/beta
    /// - final_activation.alpha/beta
    ///
    /// Note: Conv1d weights are already in MLX format [out, kernel, in] from conversion script.
    public static func loadHiFiGAN(_ hifigan: HiFiGANGenerator, from url: URL) throws {
        let weights = try CommonWeightLoader.loadSafetensors(url: url)

        // conv_pre (CausalDilatedConv1d wraps Conv1d)
        CommonWeightLoader.applyConv1dWeights(
            to: hifigan.convPre.conv, prefix: "conv_pre", from: weights, transpose: false)

        // Upsample convolutions (CausalConv1dUpsample: nn.Upsample + CausalDilatedConv1d)
        for (i, up) in hifigan.ups.enumerated() {
            CommonWeightLoader.applyConv1dWeights(
                to: up.conv.conv, prefix: "ups.\(i)", from: weights, transpose: false)
        }

        // Source downsampling convolutions (CausalConv1dDownSample or CausalDilatedConv1d)
        for (i, down) in hifigan.sourceDowns.enumerated() {
            if let downSample = down as? CausalConv1dDownSample {
                CommonWeightLoader.applyConv1dWeights(
                    to: downSample.conv, prefix: "source_downs.\(i)", from: weights, transpose: false)
            } else if let causalConv = down as? CausalDilatedConv1d {
                CommonWeightLoader.applyConv1dWeights(
                    to: causalConv.conv, prefix: "source_downs.\(i)", from: weights, transpose: false)
            }
        }

        // Main resblocks: flattened indexing (stage * numKernels + kernelIdx)
        // hifigan.resblocks is [[ResBlock]] (stages x kernels)
        var flatIdx = 0
        for stage in hifigan.resblocks {
            for resblock in stage {
                loadResBlock(resblock, prefix: "resblocks.\(flatIdx)", from: weights)
                flatIdx += 1
            }
        }

        // Source resblocks
        for (i, resblock) in hifigan.sourceResblocks.enumerated() {
            loadResBlock(resblock, prefix: "source_resblocks.\(i)", from: weights)
        }

        // conv_post (CausalDilatedConv1d wraps Conv1d)
        CommonWeightLoader.applyConv1dWeights(
            to: hifigan.convPost.conv, prefix: "conv_post", from: weights, transpose: false)

        // Source module: harmonic linear merge
        CommonWeightLoader.applyLinearWeights(
            to: hifigan.source.linearMerge, prefix: "m_source.l_linear", from: weights)

        // F0 predictor: condnet conv layers + classifier
        // F0Predictor.condnet is [CausalDilatedConv1d], each wrapping Conv1d.
        // Python condnet is Sequential of (Conv1d, ELU) pairs, so even indices are convs.
        for (i, conv) in hifigan.f0Predictor.condnet.enumerated() {
            CommonWeightLoader.applyConv1dWeights(
                to: conv.conv, prefix: "f0_predictor.condnet.\(i * 2)", from: weights, transpose: false)
        }
        CommonWeightLoader.applyLinearWeights(
            to: hifigan.f0Predictor.classifier, prefix: "f0_predictor.classifier", from: weights)
    }

    // MARK: - ResBlock Weight Loading

    /// Load weights into a ResBlock (SnakeActivation + CausalDilatedConv1d layers).
    ///
    /// ResBlock structure per dilation stage:
    ///   activations1[j] -> convs1[j] -> activations2[j] -> convs2[j] -> residual add
    ///
    /// Each CausalDilatedConv1d wraps a Conv1d, so we load into `.conv`.
    static func loadResBlock(_ resblock: ResBlock, prefix: String, from weights: [String: MLXArray]) {
        for (j, conv) in resblock.convs1.enumerated() {
            CommonWeightLoader.applyConv1dWeights(
                to: conv.conv, prefix: "\(prefix).convs1.\(j)", from: weights, transpose: false)
        }
        for (j, conv) in resblock.convs2.enumerated() {
            CommonWeightLoader.applyConv1dWeights(
                to: conv.conv, prefix: "\(prefix).convs2.\(j)", from: weights, transpose: false)
        }

        // Snake activation parameters (alpha/beta in log-space)
        for (j, act) in resblock.activations1.enumerated() {
            loadSnakeActivation(act, prefix: "\(prefix).activations1.\(j)", from: weights)
        }
        for (j, act) in resblock.activations2.enumerated() {
            loadSnakeActivation(act, prefix: "\(prefix).activations2.\(j)", from: weights)
        }
    }

    // MARK: - SnakeActivation Weight Loading

    /// Load SnakeActivation parameters (alpha only, CosyVoice3 uses alpha=beta).
    /// Stored as 1D [channels] tensors in log-space.
    static func loadSnakeActivation(
        _ act: SnakeActivation, prefix: String, from weights: [String: MLXArray]
    ) {
        if let alpha = weights["\(prefix).alpha"] {
            act.update(parameters: ModuleParameters(values: ["alpha": .value(alpha)]))
        }
    }
}
