import Foundation
import MLX
import MLXNN
import MLXCommon

/// Weight loader for MADLAD-400 (T5 v1.1) MLX model.
///
/// Expected HuggingFace safetensors keys (matches our `MADLADTranslationModel`
/// 1:1 — no prefix stripping needed):
///
/// ```
/// shared.{weight,scales,biases}                                          ── input embedding (PreQuantizedEmbedding)
/// encoder.block.{i}.layer.0.SelfAttention.{q,k,v,o}.{weight,scales,biases}
/// encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight   ── only block 0 (FP16, not quantized)
/// encoder.block.{i}.layer.0.layer_norm.weight
/// encoder.block.{i}.layer.1.DenseReluDense.{wi_0,wi_1,wo}.{weight,scales,biases}
/// encoder.block.{i}.layer.1.layer_norm.weight
/// encoder.final_layer_norm.weight
/// decoder.block.{i}.layer.0.SelfAttention.{q,k,v,o}.{weight,scales,biases}
/// decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight   ── only block 0
/// decoder.block.{i}.layer.0.layer_norm.weight
/// decoder.block.{i}.layer.1.EncDecAttention.{q,k,v,o}.{weight,scales,biases}
/// decoder.block.{i}.layer.1.layer_norm.weight
/// decoder.block.{i}.layer.2.DenseReluDense.{wi_0,wi_1,wo}.{weight,scales,biases}
/// decoder.block.{i}.layer.2.layer_norm.weight
/// decoder.final_layer_norm.weight
/// lm_head.{weight,scales,biases}
/// ```
///
/// `encoder.embed_tokens.*` and `decoder.embed_tokens.*` are duplicates of
/// `shared.*` in HF MADLAD weights and are ignored if present.
public enum MADLADWeightLoader {

    public static func loadWeights(
        into model: MADLADTranslationModel,
        from directory: URL,
        progressHandler: ((Double, String) -> Void)? = nil
    ) throws {
        progressHandler?(0.05, "Loading weight files...")
        let weights = try CommonWeightLoader.loadAllSafetensors(from: directory)
        progressHandler?(0.3, "Loaded \(weights.count) tensors")

        // Shared embedding (used by both encoder and decoder via `shared`).
        CommonWeightLoader.applyQuantizedEmbeddingWeights(
            to: model.shared, prefix: "shared", from: weights)

        // Encoder
        progressHandler?(0.35, "Loading encoder...")
        try loadEncoderStack(model.encoder, from: weights)

        // Decoder
        progressHandler?(0.7, "Loading decoder...")
        try loadDecoderStack(model.decoder, from: weights)

        // LM head
        CommonWeightLoader.applyQuantizedLinearWeights(
            to: model.lmHead, prefix: "lm_head", from: weights)

        progressHandler?(1.0, "Weights loaded")
    }

    // MARK: - Encoder

    private static func loadEncoderStack(
        _ stack: T5Stack, from weights: [String: MLXArray]
    ) throws {
        for (i, b) in stack.block.enumerated() {
            guard let block = b as? T5EncoderBlock else { continue }
            let prefix = "encoder.block.\(i)"
            try loadSelfAttentionLayer(
                block.selfAttn, prefix: "\(prefix).layer.0",
                from: weights, isFirstBlock: i == 0)
            loadFeedForwardLayer(
                block.ffn, prefix: "\(prefix).layer.1", from: weights)
        }
        CommonWeightLoader.applyRMSNormWeights(
            to: stack.finalLayerNorm,
            prefix: "encoder.final_layer_norm", from: weights)
    }

    // MARK: - Decoder

    private static func loadDecoderStack(
        _ stack: T5Stack, from weights: [String: MLXArray]
    ) throws {
        for (i, b) in stack.block.enumerated() {
            guard let block = b as? T5DecoderBlock else { continue }
            let prefix = "decoder.block.\(i)"
            try loadSelfAttentionLayer(
                block.selfAttn, prefix: "\(prefix).layer.0",
                from: weights, isFirstBlock: i == 0)
            loadCrossAttentionLayer(
                block.crossAttn, prefix: "\(prefix).layer.1", from: weights)
            loadFeedForwardLayer(
                block.ffn, prefix: "\(prefix).layer.2", from: weights)
        }
        CommonWeightLoader.applyRMSNormWeights(
            to: stack.finalLayerNorm,
            prefix: "decoder.final_layer_norm", from: weights)
    }

    // MARK: - Sub-layer loaders

    private static func loadSelfAttentionLayer(
        _ layer: T5LayerSelfAttention, prefix: String,
        from weights: [String: MLXArray], isFirstBlock: Bool
    ) throws {
        let attnPrefix = "\(prefix).SelfAttention"
        loadAttentionProjections(layer.selfAttention, prefix: attnPrefix, from: weights)

        if isFirstBlock, let table = layer.selfAttention.relativeAttentionBias {
            CommonWeightLoader.applyEmbeddingWeights(
                to: table,
                prefix: "\(attnPrefix).relative_attention_bias", from: weights)
        }
        CommonWeightLoader.applyRMSNormWeights(
            to: layer.layerNorm, prefix: "\(prefix).layer_norm", from: weights)
    }

    private static func loadCrossAttentionLayer(
        _ layer: T5LayerCrossAttention, prefix: String,
        from weights: [String: MLXArray]
    ) {
        let attnPrefix = "\(prefix).EncDecAttention"
        loadAttentionProjections(layer.encDecAttention, prefix: attnPrefix, from: weights)
        CommonWeightLoader.applyRMSNormWeights(
            to: layer.layerNorm, prefix: "\(prefix).layer_norm", from: weights)
    }

    private static func loadFeedForwardLayer(
        _ layer: T5LayerFF, prefix: String, from weights: [String: MLXArray]
    ) {
        let ffPrefix = "\(prefix).DenseReluDense"
        CommonWeightLoader.applyQuantizedLinearWeights(
            to: layer.denseReluDense.wi0, prefix: "\(ffPrefix).wi_0", from: weights)
        CommonWeightLoader.applyQuantizedLinearWeights(
            to: layer.denseReluDense.wi1, prefix: "\(ffPrefix).wi_1", from: weights)
        CommonWeightLoader.applyQuantizedLinearWeights(
            to: layer.denseReluDense.wo, prefix: "\(ffPrefix).wo", from: weights)
        CommonWeightLoader.applyRMSNormWeights(
            to: layer.layerNorm, prefix: "\(prefix).layer_norm", from: weights)
    }

    private static func loadAttentionProjections(
        _ attn: T5Attention, prefix: String, from weights: [String: MLXArray]
    ) {
        CommonWeightLoader.applyQuantizedLinearWeights(
            to: attn.q, prefix: "\(prefix).q", from: weights)
        CommonWeightLoader.applyQuantizedLinearWeights(
            to: attn.k, prefix: "\(prefix).k", from: weights)
        CommonWeightLoader.applyQuantizedLinearWeights(
            to: attn.v, prefix: "\(prefix).v", from: weights)
        CommonWeightLoader.applyQuantizedLinearWeights(
            to: attn.o, prefix: "\(prefix).o", from: weights)
    }
}
