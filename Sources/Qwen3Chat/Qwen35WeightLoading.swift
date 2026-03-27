import Foundation
import MLXCommon
import MLX
import MLXNN

// MARK: - Weight Loading for Qwen3.5-0.8B MLX Model

/// Loads quantized safetensors weights into the Qwen3.5 MLX model.
///
/// Expected weight key structure (HuggingFace format):
///   - `model.embed_tokens.*`           -> embed_tokens (PreQuantizedEmbedding)
///   - `model.layers.{i}.self_attn.*`   -> layers.{i}.self_attn (DeltaNet or GatedAttention)
///   - `model.layers.{i}.mlp.*`         -> layers.{i}.mlp
///   - `model.layers.{i}.input_layernorm.*`
///   - `model.layers.{i}.post_attention_layernorm.*`
///   - `model.norm.*`                   -> norm (final RMSNorm)
///   - `lm_head.*`                      -> ignored (tied to embed_tokens)
///
/// DeltaNet (linear_attention) layer weights under `self_attn.`:
///   - `in_proj_qkv.weight`: [6144, 1024]
///   - `in_proj_z.weight`: [2048, 1024]
///   - `in_proj_b.weight`: [16, 1024]
///   - `in_proj_a.weight`: [16, 1024]
///   - `conv1d.weight`: [6144, 1, 4]
///   - `dt_bias`: [16]
///   - `A_log`: [16]
///   - `norm.weight`: [128]
///   - `out_proj.weight`: [1024, 2048]
///
/// GatedAttention (full_attention) layer weights under `self_attn.`:
///   - `q_proj.{weight,scales,biases}`: quantized [4096, 1024]
///   - `k_proj.{weight,scales,biases}`: quantized [512, 1024]
///   - `v_proj.{weight,scales,biases}`: quantized [512, 1024]
///   - `o_proj.{weight,scales,biases}`: quantized [1024, 2048]
///   - `q_norm.weight`: [256]
///   - `k_norm.weight`: [256]
///
/// MLP weights under `mlp.`:
///   - `gate_proj.{weight,scales,biases}`: quantized [3584, 1024]
///   - `up_proj.{weight,scales,biases}`: quantized [3584, 1024]
///   - `down_proj.{weight,scales,biases}`: quantized [1024, 3584]
public enum Qwen35WeightLoader {

    /// Load weights from a directory containing safetensors files.
    ///
    /// - Parameters:
    ///   - model: The Qwen3.5 MLX model to load weights into
    ///   - directory: Directory containing safetensors files
    ///   - progressHandler: Optional progress callback
    public static func loadWeights(
        into model: Qwen35MLXModel,
        from directory: URL,
        progressHandler: ((Double, String) -> Void)? = nil
    ) throws {
        progressHandler?(0.05, "Loading weight files...")

        // Load all safetensors files from the directory
        let allWeights = try CommonWeightLoader.loadAllSafetensors(from: directory)
        progressHandler?(0.3, "Loaded \(allWeights.count) tensors")

        // Strip "model." prefix from keys
        var modelWeights: [String: MLXArray] = [:]
        for (key, value) in allWeights {
            if key.hasPrefix("model.") {
                modelWeights[String(key.dropFirst("model.".count))] = value
            } else if key.hasPrefix("lm_head.") {
                // Skip lm_head — tied to embed_tokens
                continue
            } else {
                modelWeights[key] = value
            }
        }

        progressHandler?(0.4, "Applying embedding weights...")

        // Load embed_tokens (PreQuantizedEmbedding)
        CommonWeightLoader.applyQuantizedEmbeddingWeights(
            to: model.embedTokens,
            prefix: "embed_tokens",
            from: modelWeights)

        // Load final norm
        CommonWeightLoader.applyRMSNormWeights(
            to: model.norm, prefix: "norm", from: modelWeights)

        progressHandler?(0.5, "Loading transformer layers...")

        // Load each layer
        let numLayers = model.config.numHiddenLayers
        for i in 0..<numLayers {
            let prefix = "layers.\(i)"
            let layer = model.layers[i]

            // Layer norms
            CommonWeightLoader.applyRMSNormWeights(
                to: layer.inputLayerNorm,
                prefix: "\(prefix).input_layernorm",
                from: modelWeights)
            CommonWeightLoader.applyRMSNormWeights(
                to: layer.postAttentionLayerNorm,
                prefix: "\(prefix).post_attention_layernorm",
                from: modelWeights)

            // MLP
            applyQuantizedMLPWeights(
                to: layer.mlp,
                prefix: "\(prefix).mlp",
                from: modelWeights)

            // Attention (type-specific)
            let attnPrefix = "\(prefix).self_attn"
            if layer.layerType == "linear_attention" {
                try applyDeltaNetWeights(
                    to: layer.deltaNet!,
                    prefix: attnPrefix,
                    from: modelWeights)
            } else {
                applyGatedAttentionWeights(
                    to: layer.gatedAttn!,
                    prefix: attnPrefix,
                    from: modelWeights)
            }

            let pct = 0.5 + 0.45 * Double(i + 1) / Double(numLayers)
            progressHandler?(pct, "Layer \(i + 1)/\(numLayers)")
        }

        // Evaluate all parameters
        eval(model)
        progressHandler?(1.0, "Weights loaded")
    }

    // MARK: - DeltaNet Weight Loading

    /// Apply weights to a DeltaNet (linear attention) layer.
    ///
    /// DeltaNet layers use float16/bfloat16 Linear (NOT quantized) for most projections,
    /// since these are relatively small. The conv1d weight and scalar parameters
    /// (dt_bias, A_log) are loaded directly.
    private static func applyDeltaNetWeights(
        to layer: DeltaNetLayer,
        prefix: String,
        from weights: [String: MLXArray]
    ) throws {
        // Linear projections
        CommonWeightLoader.applyLinearWeights(
            to: layer.inProjQKV, prefix: "\(prefix).in_proj_qkv", from: weights)
        CommonWeightLoader.applyLinearWeights(
            to: layer.inProjZ, prefix: "\(prefix).in_proj_z", from: weights)
        CommonWeightLoader.applyLinearWeights(
            to: layer.inProjB, prefix: "\(prefix).in_proj_b", from: weights)
        CommonWeightLoader.applyLinearWeights(
            to: layer.inProjA, prefix: "\(prefix).in_proj_a", from: weights)

        // Conv1d weight: stored as [C, 1, K]
        if let w = weights["\(prefix).conv1d.weight"] {
            layer.convWeight = w
        }

        // Scalar parameters
        if let dtb = weights["\(prefix).dt_bias"] {
            layer.dtBias = dtb
        }
        if let alog = weights["\(prefix).A_log"] {
            layer.aLog = alog
        }

        // Per-head norm
        CommonWeightLoader.applyRMSNormWeights(
            to: layer.norm, prefix: "\(prefix).norm", from: weights)

        // Output projection
        CommonWeightLoader.applyLinearWeights(
            to: layer.outProj, prefix: "\(prefix).out_proj", from: weights)
    }

    // MARK: - GatedAttention Weight Loading

    /// Apply weights to a GatedAttention (full attention) layer.
    ///
    /// All projections are quantized (INT4 with group_size=64).
    private static func applyGatedAttentionWeights(
        to layer: GatedAttentionLayer,
        prefix: String,
        from weights: [String: MLXArray]
    ) {
        CommonWeightLoader.applyQuantizedLinearWeights(
            to: layer.qProj, prefix: "\(prefix).q_proj", from: weights)
        CommonWeightLoader.applyQuantizedLinearWeights(
            to: layer.kProj, prefix: "\(prefix).k_proj", from: weights)
        CommonWeightLoader.applyQuantizedLinearWeights(
            to: layer.vProj, prefix: "\(prefix).v_proj", from: weights)
        CommonWeightLoader.applyQuantizedLinearWeights(
            to: layer.oProj, prefix: "\(prefix).o_proj", from: weights)

        CommonWeightLoader.applyRMSNormWeights(
            to: layer.qNorm, prefix: "\(prefix).q_norm", from: weights)
        CommonWeightLoader.applyRMSNormWeights(
            to: layer.kNorm, prefix: "\(prefix).k_norm", from: weights)
    }

    // MARK: - MLP Weight Loading

    /// Apply quantized MLP weights (SwiGLU: gate_proj, up_proj, down_proj).
    private static func applyQuantizedMLPWeights(
        to mlp: Qwen35MLP,
        prefix: String,
        from weights: [String: MLXArray]
    ) {
        CommonWeightLoader.applyQuantizedLinearWeights(
            to: mlp.gateProj, prefix: "\(prefix).gate_proj", from: weights)
        CommonWeightLoader.applyQuantizedLinearWeights(
            to: mlp.upProj, prefix: "\(prefix).up_proj", from: weights)
        CommonWeightLoader.applyQuantizedLinearWeights(
            to: mlp.downProj, prefix: "\(prefix).down_proj", from: weights)
    }
}
