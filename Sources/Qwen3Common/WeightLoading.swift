import Foundation
import MLX
import MLXNN

/// Generic weight loading utilities shared between ASR and TTS
public enum CommonWeightLoader {

    /// Load weights from safetensors file
    public static func loadSafetensors(url: URL) throws -> [String: MLXArray] {
        try MLX.loadArrays(url: url)
    }

    /// Load all safetensors from a directory, optionally filtering by prefix
    public static func loadAllSafetensors(
        from directory: URL,
        prefix: String? = nil,
        stripPrefix: Bool = true
    ) throws -> [String: MLXArray] {
        let fileManager = FileManager.default
        let contents = try fileManager.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
        let safetensorFiles = contents.filter { $0.pathExtension == "safetensors" }

        guard !safetensorFiles.isEmpty else {
            throw WeightLoadingError.noWeightsFound(directory)
        }

        var allWeights: [String: MLXArray] = [:]
        for file in safetensorFiles {
            let weights = try loadSafetensors(url: file)
            allWeights.merge(weights) { _, new in new }
        }

        // Filter and strip prefix if specified
        guard let prefix = prefix else { return allWeights }

        var filtered: [String: MLXArray] = [:]
        for (key, value) in allWeights {
            if key.hasPrefix(prefix) {
                let strippedKey = stripPrefix ? String(key.dropFirst(prefix.count)) : key
                filtered[strippedKey] = value
            }
        }
        return filtered
    }

    // MARK: - Quantized Weight Application Helpers

    public static func applyQuantizedEmbeddingWeights(
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

    public static func applyQuantizedLinearWeights(
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

    public static func applyRMSNormWeights(
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

    public static func applyLinearWeights(
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

    public static func applyLayerNormWeights(
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

    public static func applyEmbeddingWeights(
        to embedding: Embedding,
        prefix: String,
        from weights: [String: MLXArray]
    ) {
        var params: [String: NestedItem<String, MLXArray>] = [:]

        if let weight = weights["\(prefix).weight"] {
            params["weight"] = .value(weight)
        }

        if !params.isEmpty {
            embedding.update(parameters: ModuleParameters(values: params))
        }
    }

    public static func applyConv1dWeights(
        to conv: Conv1d,
        prefix: String,
        from weights: [String: MLXArray],
        transpose: Bool = false
    ) {
        var params: [String: NestedItem<String, MLXArray>] = [:]

        if let weight = weights["\(prefix).weight"] {
            // PyTorch Conv1d: [out, in, kernel] -> MLX Conv1d: [out, kernel, in]
            let w = transpose ? weight.transposed(0, 2, 1) : weight
            params["weight"] = .value(w)
        }
        if let bias = weights["\(prefix).bias"] {
            params["bias"] = .value(bias)
        }

        if !params.isEmpty {
            conv.update(parameters: ModuleParameters(values: params))
        }
    }

    public static func applyConvTransposed1dWeights(
        to conv: ConvTransposed1d,
        prefix: String,
        from weights: [String: MLXArray],
        transpose: Bool = false
    ) {
        var params: [String: NestedItem<String, MLXArray>] = [:]

        if let weight = weights["\(prefix).weight"] {
            // PyTorch ConvTranspose1d: [in, out, kernel] -> MLX ConvTransposed1d: [out, kernel, in]
            let w = transpose ? weight.transposed(1, 2, 0) : weight
            params["weight"] = .value(w)
        }
        if let bias = weights["\(prefix).bias"] {
            params["bias"] = .value(bias)
        }

        if !params.isEmpty {
            conv.update(parameters: ModuleParameters(values: params))
        }
    }

    /// Apply QuantizedMLP weights (SwiGLU)
    public static func applyQuantizedMLPWeights(
        to mlp: QuantizedMLP,
        prefix: String,
        from weights: [String: MLXArray]
    ) {
        applyQuantizedLinearWeights(to: mlp.gateProj, prefix: "\(prefix).gate_proj", from: weights)
        applyQuantizedLinearWeights(to: mlp.upProj, prefix: "\(prefix).up_proj", from: weights)
        applyQuantizedLinearWeights(to: mlp.downProj, prefix: "\(prefix).down_proj", from: weights)
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
