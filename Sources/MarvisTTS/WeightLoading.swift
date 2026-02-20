import Foundation
import MLX
import MLXNN

// MARK: - CSM Weight Loading

func loadMarvisWeights(from directory: URL) throws -> [String: MLXArray] {
    let fm = FileManager.default
    let singleFile = directory.appendingPathComponent("model.safetensors")
    if fm.fileExists(atPath: singleFile.path) {
        return try MLX.loadArrays(url: singleFile)
    }
    let files = try fm.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
    let safetensorFiles = files.filter { $0.pathExtension == "safetensors" }
    var weights: [String: MLXArray] = [:]
    for file in safetensorFiles {
        let fileWeights = try MLX.loadArrays(url: file)
        weights.merge(fileWeights) { _, new in new }
    }
    return weights
}

// MARK: - CSM Weight Sanitization

func sanitizeCSMWeights(_ weights: [String: MLXArray]) -> [String: MLXArray] {
    var out: [String: MLXArray] = [:]
    out.reserveCapacity(weights.count)

    for (rawKey, v) in weights {
        var k = rawKey

        // Strip "model." prefix — CSMModel IS the root module
        if k.hasPrefix("model.") {
            k = String(k.dropFirst("model.".count))
        }

        // Legacy format: attn → self_attn, output_proj → o_proj
        if k.contains("attn") && !k.contains("self_attn") {
            k = k.replacingOccurrences(of: "attn", with: "self_attn")
            k = k.replacingOccurrences(of: "output_proj", with: "o_proj")
        }

        // Legacy format: w1/w2/w3 → gate/down/up_proj
        if k.contains("mlp") {
            k = k.replacingOccurrences(of: "w1", with: "gate_proj")
            k = k.replacingOccurrences(of: "w2", with: "down_proj")
            k = k.replacingOccurrences(of: "w3", with: "up_proj")
        }

        // Legacy format: sa_norm/mlp_norm → layernorm names, scale → weight
        if k.contains("sa_norm") || k.contains("mlp_norm") {
            k = k.replacingOccurrences(of: "sa_norm", with: "input_layernorm")
            k = k.replacingOccurrences(of: "mlp_norm", with: "post_attention_layernorm")
            k = k.replacingOccurrences(of: "scale", with: "weight")
        }

        if k.contains("decoder.norm") || k.contains("backbone.norm") {
            k = k.replacingOccurrences(of: "scale", with: "weight")
        }

        out[k] = v
    }

    return out
}
