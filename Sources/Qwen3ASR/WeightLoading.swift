import Foundation
import MLX
import MLXNN

/// Weight loading utilities for Qwen3-ASR
/// Uses direct HuggingFace key paths - model structure must match exactly
public enum WeightLoader {

    /// Load weights from safetensors file
    public static func loadSafetensors(url: URL) throws -> [String: MLXArray] {
        try MLX.loadArrays(url: url)
    }

    /// Load and apply weights to model using HuggingFace key paths directly
    public static func loadWeights(
        into audioEncoder: Qwen3AudioEncoder,
        from directory: URL
    ) throws {
        // Find all safetensors files
        let fileManager = FileManager.default
        let contents = try fileManager.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
        let safetensorFiles = contents.filter { $0.pathExtension == "safetensors" }

        guard !safetensorFiles.isEmpty else {
            throw WeightLoadingError.noWeightsFound(directory)
        }

        print("Found \(safetensorFiles.count) safetensor files")

        // Load all weight files
        var allWeights: [String: MLXArray] = [:]
        for file in safetensorFiles {
            print("Loading: \(file.lastPathComponent)")
            let weights = try loadSafetensors(url: file)
            allWeights.merge(weights) { _, new in new }
        }

        print("Loaded \(allWeights.count) weight tensors from files")

        // Filter to audio_tower weights and strip prefix
        var audioTowerWeights: [String: MLXArray] = [:]
        for (key, value) in allWeights {
            if key.hasPrefix("audio_tower.") {
                let strippedKey = String(key.dropFirst("audio_tower.".count))
                audioTowerWeights[strippedKey] = value
            }
        }

        print("Found \(audioTowerWeights.count) audio_tower weights")

        // Apply weights to each component using update(parameters:)
        // This avoids the "Unable to set layers" issue with array-indexed modules

        // Conv layers - use Conv2d format transpose
        applyConv2dWeights(to: audioEncoder.conv2d1, prefix: "conv2d1", from: audioTowerWeights)
        applyConv2dWeights(to: audioEncoder.conv2d2, prefix: "conv2d2", from: audioTowerWeights)
        applyConv2dWeights(to: audioEncoder.conv2d3, prefix: "conv2d3", from: audioTowerWeights)

        // Output linear (no bias)
        applyLinearWeights(to: audioEncoder.convOut, prefix: "conv_out", from: audioTowerWeights)

        // Post layer norm
        applyLayerNormWeights(to: audioEncoder.lnPost, prefix: "ln_post", from: audioTowerWeights)

        // Projector layers
        applyLinearWeights(to: audioEncoder.proj1, prefix: "proj1", from: audioTowerWeights)
        applyLinearWeights(to: audioEncoder.proj2, prefix: "proj2", from: audioTowerWeights)

        // Transformer layers (indexed as layers.0, layers.1, etc.)
        for (index, layer) in audioEncoder.layers.enumerated() {
            let prefix = "layers.\(index)"
            applyEncoderLayerWeights(to: layer, prefix: prefix, from: audioTowerWeights)
        }

        print("Applied weights to audio encoder (\(audioEncoder.layers.count) layers)")
    }

    /// Load and apply weights to quantized text decoder
    public static func loadTextDecoderWeights(
        into textModel: QuantizedTextModel,
        from directory: URL
    ) throws {
        // Find all safetensors files
        let fileManager = FileManager.default
        let contents = try fileManager.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
        let safetensorFiles = contents.filter { $0.pathExtension == "safetensors" }

        guard !safetensorFiles.isEmpty else {
            throw WeightLoadingError.noWeightsFound(directory)
        }

        // Load all weight files
        var allWeights: [String: MLXArray] = [:]
        for file in safetensorFiles {
            let weights = try loadSafetensors(url: file)
            allWeights.merge(weights) { _, new in new }
        }

        // Filter to model.* weights (text decoder)
        var textWeights: [String: MLXArray] = [:]
        for (key, value) in allWeights {
            if key.hasPrefix("model.") {
                let strippedKey = String(key.dropFirst("model.".count))
                textWeights[strippedKey] = value
            }
        }

        print("Found \(textWeights.count) text decoder weights")

        // Load embedding weights (quantized)
        applyQuantizedEmbeddingWeights(
            to: textModel.embedTokens,
            prefix: "embed_tokens",
            from: textWeights
        )

        // Load final layer norm
        applyRMSNormWeights(to: textModel.norm, prefix: "norm", from: textWeights)

        // Load each decoder layer
        for (index, layer) in textModel.layers.enumerated() {
            let prefix = "layers.\(index)"
            applyQuantizedDecoderLayerWeights(to: layer, prefix: prefix, from: textWeights)
        }

        print("Applied weights to text decoder (\(textModel.layers.count) layers)")
    }

    // MARK: - Quantized Weight Application Helpers

    private static func applyQuantizedEmbeddingWeights(
        to embedding: PreQuantizedEmbedding,
        prefix: String,
        from weights: [String: MLXArray]
    ) {
        var params: [String: NestedItem<String, MLXArray>] = [:]

        if let weight = weights["\(prefix).weight"] {
            params["weight"] = .value(weight)
        }
        if let scales = weights["\(prefix).scales"] {
            params["scales"] = .value(scales)
        }
        if let biases = weights["\(prefix).biases"] {
            params["biases"] = .value(biases)
        }

        if !params.isEmpty {
            embedding.update(parameters: ModuleParameters(values: params))
        }
    }

    private static func applyQuantizedLinearWeights(
        to linear: QuantizedLinear,
        prefix: String,
        from weights: [String: MLXArray]
    ) {
        var params: [String: NestedItem<String, MLXArray>] = [:]

        if let weight = weights["\(prefix).weight"] {
            params["weight"] = .value(weight)
        }
        if let scales = weights["\(prefix).scales"] {
            params["scales"] = .value(scales)
        }
        if let biases = weights["\(prefix).biases"] {
            params["biases"] = .value(biases)
        }

        if !params.isEmpty {
            linear.update(parameters: ModuleParameters(values: params))
        }
    }

    private static func applyRMSNormWeights(
        to norm: RMSNorm,
        prefix: String,
        from weights: [String: MLXArray]
    ) {
        var params: [String: NestedItem<String, MLXArray>] = [:]

        if let weight = weights["\(prefix).weight"] {
            params["weight"] = .value(weight)
        }

        if !params.isEmpty {
            norm.update(parameters: ModuleParameters(values: params))
        }
    }

    private static func applyQuantizedDecoderLayerWeights(
        to layer: QuantizedTextDecoderLayer,
        prefix: String,
        from weights: [String: MLXArray]
    ) {
        // Self attention
        applyQuantizedLinearWeights(to: layer.selfAttn.qProj, prefix: "\(prefix).self_attn.q_proj", from: weights)
        applyQuantizedLinearWeights(to: layer.selfAttn.kProj, prefix: "\(prefix).self_attn.k_proj", from: weights)
        applyQuantizedLinearWeights(to: layer.selfAttn.vProj, prefix: "\(prefix).self_attn.v_proj", from: weights)
        applyQuantizedLinearWeights(to: layer.selfAttn.oProj, prefix: "\(prefix).self_attn.o_proj", from: weights)

        // Q/K norms
        applyRMSNormWeights(to: layer.selfAttn.qNorm, prefix: "\(prefix).self_attn.q_norm", from: weights)
        applyRMSNormWeights(to: layer.selfAttn.kNorm, prefix: "\(prefix).self_attn.k_norm", from: weights)

        // Layer norms
        applyRMSNormWeights(to: layer.inputLayerNorm, prefix: "\(prefix).input_layernorm", from: weights)
        applyRMSNormWeights(to: layer.postAttentionLayerNorm, prefix: "\(prefix).post_attention_layernorm", from: weights)

        // MLP
        applyQuantizedLinearWeights(to: layer.mlp.gateProj, prefix: "\(prefix).mlp.gate_proj", from: weights)
        applyQuantizedLinearWeights(to: layer.mlp.upProj, prefix: "\(prefix).mlp.up_proj", from: weights)
        applyQuantizedLinearWeights(to: layer.mlp.downProj, prefix: "\(prefix).mlp.down_proj", from: weights)
    }

    // MARK: - Weight Application Helpers

    private static func applyConv2dWeights(
        to conv: Conv2d,
        prefix: String,
        from weights: [String: MLXArray]
    ) {
        var params: [String: NestedItem<String, MLXArray>] = [:]

        // The mlx-community model weights are already in MLX format: [out_channels, kH, kW, in_channels]
        // No transpose needed
        if let weight = weights["\(prefix).weight"] {
            params["weight"] = .value(weight)
        }
        if let bias = weights["\(prefix).bias"] {
            params["bias"] = .value(bias)
        }

        if !params.isEmpty {
            conv.update(parameters: ModuleParameters(values: params))
        }
    }

    private static func applyLinearWeights(
        to linear: Linear,
        prefix: String,
        from weights: [String: MLXArray]
    ) {
        var params: [String: NestedItem<String, MLXArray>] = [:]

        if let weight = weights["\(prefix).weight"] {
            params["weight"] = .value(weight)
        }
        if let bias = weights["\(prefix).bias"] {
            params["bias"] = .value(bias)
        }

        if !params.isEmpty {
            linear.update(parameters: ModuleParameters(values: params))
        }
    }

    private static func applyLayerNormWeights(
        to layerNorm: LayerNorm,
        prefix: String,
        from weights: [String: MLXArray]
    ) {
        var params: [String: NestedItem<String, MLXArray>] = [:]

        if let weight = weights["\(prefix).weight"] {
            params["weight"] = .value(weight)
        }
        if let bias = weights["\(prefix).bias"] {
            params["bias"] = .value(bias)
        }

        if !params.isEmpty {
            layerNorm.update(parameters: ModuleParameters(values: params))
        }
    }

    private static func applyEncoderLayerWeights(
        to layer: AudioEncoderLayer,
        prefix: String,
        from weights: [String: MLXArray]
    ) {
        // Self attention
        applyLinearWeights(to: layer.selfAttn.qProj, prefix: "\(prefix).self_attn.q_proj", from: weights)
        applyLinearWeights(to: layer.selfAttn.kProj, prefix: "\(prefix).self_attn.k_proj", from: weights)
        applyLinearWeights(to: layer.selfAttn.vProj, prefix: "\(prefix).self_attn.v_proj", from: weights)
        applyLinearWeights(to: layer.selfAttn.outProj, prefix: "\(prefix).self_attn.out_proj", from: weights)

        // Layer norms
        applyLayerNormWeights(to: layer.selfAttnLayerNorm, prefix: "\(prefix).self_attn_layer_norm", from: weights)
        applyLayerNormWeights(to: layer.finalLayerNorm, prefix: "\(prefix).final_layer_norm", from: weights)

        // FFN
        applyLinearWeights(to: layer.fc1, prefix: "\(prefix).fc1", from: weights)
        applyLinearWeights(to: layer.fc2, prefix: "\(prefix).fc2", from: weights)
    }
}

/// Weight loading errors
public enum WeightLoadingError: Error, LocalizedError {
    case noWeightsFound(URL)
    case incompatibleWeights(String)
    case missingRequiredWeight(String)

    public var errorDescription: String? {
        switch self {
        case .noWeightsFound(let url):
            return "No safetensors files found in: \(url.path)"
        case .incompatibleWeights(let reason):
            return "Incompatible weights: \(reason)"
        case .missingRequiredWeight(let key):
            return "Missing required weight: \(key)"
        }
    }
}
