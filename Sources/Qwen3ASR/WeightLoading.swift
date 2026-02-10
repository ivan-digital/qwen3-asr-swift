import Foundation
import MLX
import MLXNN
import Qwen3Common

/// Weight loading utilities for Qwen3-ASR
/// Uses direct HuggingFace key paths - model structure must match exactly
public enum WeightLoader {

    /// Load weights from safetensors file
    public static func loadSafetensors(url: URL) throws -> [String: MLXArray] {
        try CommonWeightLoader.loadSafetensors(url: url)
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
        applyConv2dWeights(to: audioEncoder.conv2d1, prefix: "conv2d1", from: audioTowerWeights)
        applyConv2dWeights(to: audioEncoder.conv2d2, prefix: "conv2d2", from: audioTowerWeights)
        applyConv2dWeights(to: audioEncoder.conv2d3, prefix: "conv2d3", from: audioTowerWeights)

        // Output linear (no bias)
        CommonWeightLoader.applyLinearWeights(to: audioEncoder.convOut, prefix: "conv_out", from: audioTowerWeights)

        // Post layer norm
        CommonWeightLoader.applyLayerNormWeights(to: audioEncoder.lnPost, prefix: "ln_post", from: audioTowerWeights)

        // Projector layers
        CommonWeightLoader.applyLinearWeights(to: audioEncoder.proj1, prefix: "proj1", from: audioTowerWeights)
        CommonWeightLoader.applyLinearWeights(to: audioEncoder.proj2, prefix: "proj2", from: audioTowerWeights)

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
        CommonWeightLoader.applyQuantizedEmbeddingWeights(
            to: textModel.embedTokens,
            prefix: "embed_tokens",
            from: textWeights
        )

        // Load final layer norm
        CommonWeightLoader.applyRMSNormWeights(to: textModel.norm, prefix: "norm", from: textWeights)

        // Load each decoder layer
        for (index, layer) in textModel.layers.enumerated() {
            let prefix = "layers.\(index)"
            applyQuantizedDecoderLayerWeights(to: layer, prefix: prefix, from: textWeights)
        }

        print("Applied weights to text decoder (\(textModel.layers.count) layers)")
    }

    // MARK: - ASR-specific Weight Application Helpers

    private static func applyQuantizedDecoderLayerWeights(
        to layer: QuantizedTextDecoderLayer,
        prefix: String,
        from weights: [String: MLXArray]
    ) {
        // Self attention
        CommonWeightLoader.applyQuantizedLinearWeights(to: layer.selfAttn.qProj, prefix: "\(prefix).self_attn.q_proj", from: weights)
        CommonWeightLoader.applyQuantizedLinearWeights(to: layer.selfAttn.kProj, prefix: "\(prefix).self_attn.k_proj", from: weights)
        CommonWeightLoader.applyQuantizedLinearWeights(to: layer.selfAttn.vProj, prefix: "\(prefix).self_attn.v_proj", from: weights)
        CommonWeightLoader.applyQuantizedLinearWeights(to: layer.selfAttn.oProj, prefix: "\(prefix).self_attn.o_proj", from: weights)

        // Q/K norms
        CommonWeightLoader.applyRMSNormWeights(to: layer.selfAttn.qNorm, prefix: "\(prefix).self_attn.q_norm", from: weights)
        CommonWeightLoader.applyRMSNormWeights(to: layer.selfAttn.kNorm, prefix: "\(prefix).self_attn.k_norm", from: weights)

        // Layer norms
        CommonWeightLoader.applyRMSNormWeights(to: layer.inputLayerNorm, prefix: "\(prefix).input_layernorm", from: weights)
        CommonWeightLoader.applyRMSNormWeights(to: layer.postAttentionLayerNorm, prefix: "\(prefix).post_attention_layernorm", from: weights)

        // MLP
        CommonWeightLoader.applyQuantizedLinearWeights(to: layer.mlp.gateProj, prefix: "\(prefix).mlp.gate_proj", from: weights)
        CommonWeightLoader.applyQuantizedLinearWeights(to: layer.mlp.upProj, prefix: "\(prefix).mlp.up_proj", from: weights)
        CommonWeightLoader.applyQuantizedLinearWeights(to: layer.mlp.downProj, prefix: "\(prefix).mlp.down_proj", from: weights)
    }

    // MARK: - Audio Encoder Weight Helpers

    private static func applyConv2dWeights(
        to conv: Conv2d,
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
            conv.update(parameters: ModuleParameters(values: params))
        }
    }

    private static func applyEncoderLayerWeights(
        to layer: AudioEncoderLayer,
        prefix: String,
        from weights: [String: MLXArray]
    ) {
        // Self attention
        CommonWeightLoader.applyLinearWeights(to: layer.selfAttn.qProj, prefix: "\(prefix).self_attn.q_proj", from: weights)
        CommonWeightLoader.applyLinearWeights(to: layer.selfAttn.kProj, prefix: "\(prefix).self_attn.k_proj", from: weights)
        CommonWeightLoader.applyLinearWeights(to: layer.selfAttn.vProj, prefix: "\(prefix).self_attn.v_proj", from: weights)
        CommonWeightLoader.applyLinearWeights(to: layer.selfAttn.outProj, prefix: "\(prefix).self_attn.out_proj", from: weights)

        // Layer norms
        CommonWeightLoader.applyLayerNormWeights(to: layer.selfAttnLayerNorm, prefix: "\(prefix).self_attn_layer_norm", from: weights)
        CommonWeightLoader.applyLayerNormWeights(to: layer.finalLayerNorm, prefix: "\(prefix).final_layer_norm", from: weights)

        // FFN
        CommonWeightLoader.applyLinearWeights(to: layer.fc1, prefix: "\(prefix).fc1", from: weights)
        CommonWeightLoader.applyLinearWeights(to: layer.fc2, prefix: "\(prefix).fc2", from: weights)
    }
}
