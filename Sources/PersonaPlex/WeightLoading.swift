import Foundation
import MLX
import MLXNN
import Qwen3Common

// MARK: - Weight Loading

public enum PersonaPlexWeightLoader {

    /// Load all weights from a model directory containing split safetensors files.
    public static func loadWeights(
        model: PersonaPlexModel,
        from directory: URL,
        progressHandler: ((Double, String) -> Void)? = nil
    ) throws {
        let numSteps = model.cfg.depformer.numSteps

        // Load temporal transformer weights (4-bit quantized)
        progressHandler?(0.1, "Loading temporal transformer...")
        let temporalFile = directory.appendingPathComponent("temporal.safetensors")
        if FileManager.default.fileExists(atPath: temporalFile.path) {
            let weights = try MLX.loadArrays(url: temporalFile)
            let sanitized = sanitizeTemporalWeights(weights)
            let params = ModuleParameters.unflattened(sanitized)
            try model.temporal.update(parameters: params, verify: .noUnusedKeys)
        }

        // Load embeddings (mixed temporal + depformer keys)
        progressHandler?(0.3, "Loading embeddings...")
        let embFile = directory.appendingPathComponent("embeddings.safetensors")
        if FileManager.default.fileExists(atPath: embFile.path) {
            let weights = try MLX.loadArrays(url: embFile)
            let (temporalEmb, depformerEmb) = splitEmbeddingWeights(weights)

            let tParams = ModuleParameters.unflattened(temporalEmb)
            try model.temporal.update(parameters: tParams, verify: .noUnusedKeys)

            let dParams = ModuleParameters.unflattened(depformerEmb)
            try model.depformer.update(parameters: dParams, verify: .noUnusedKeys)
        }

        // Load depformer weights (BF16)
        progressHandler?(0.5, "Loading depformer...")
        let depFile = directory.appendingPathComponent("depformer.safetensors")
        if FileManager.default.fileExists(atPath: depFile.path) {
            let weights = try MLX.loadArrays(url: depFile)
            let sanitized = sanitizeDepformerWeights(weights, numSteps: numSteps)
            let params = ModuleParameters.unflattened(sanitized)
            try model.depformer.update(parameters: params, verify: .noUnusedKeys)
        }

        eval(model.temporal, model.depformer)
        progressHandler?(0.7, "Model weights loaded")
    }

    /// Load Mimi codec weights
    public static func loadMimi(
        model: Mimi,
        from directory: URL,
        progressHandler: ((Double, String) -> Void)? = nil
    ) throws {
        progressHandler?(0.0, "Loading Mimi codec...")
        let mimiFile = directory.appendingPathComponent("mimi.safetensors")
        guard FileManager.default.fileExists(atPath: mimiFile.path) else {
            throw PersonaPlexError.missingWeightFile("mimi.safetensors")
        }

        var weights = try MLX.loadArrays(url: mimiFile)
        weights = model.sanitize(weights: weights)

        let params = ModuleParameters.unflattened(weights)
        // Use .noUnusedKeys (not .all) — some Mimi model parameters may not have
        // corresponding weights (e.g. cache state, streaming buffers)
        try model.update(parameters: params, verify: .noUnusedKeys)

        // Update codebooks
        func updateCodebooks(_ module: Module) {
            if let codebook = module as? EuclideanCodebook {
                codebook.updateInPlace()
            }
            for (_, child) in module.children().flattened() {
                updateCodebooks(child)
            }
        }
        updateCodebooks(model)
        eval(model)

        progressHandler?(1.0, "Mimi codec loaded")
    }

    /// Load voice prompt embeddings
    public static func loadVoice(
        _ voice: PersonaPlexVoice,
        from directory: URL
    ) throws -> MLXArray {
        let voiceDir = directory.appendingPathComponent("voices")
        let voiceFile = voiceDir.appendingPathComponent("\(voice.rawValue).safetensors")

        guard FileManager.default.fileExists(atPath: voiceFile.path) else {
            throw PersonaPlexError.missingWeightFile("voices/\(voice.rawValue).safetensors")
        }

        let weights = try MLX.loadArrays(url: voiceFile)
        guard let embeddings = weights["embeddings"] else {
            throw PersonaPlexError.missingKey("embeddings", in: voiceFile.lastPathComponent)
        }

        return embeddings
    }

    // MARK: - Temporal Weight Sanitization

    /// Sanitize temporal transformer weights:
    /// - Rename `*.alpha` (1,1,D) → `*.weight` (D)  (RMSNorm)
    /// - Rename `*.in_proj_weight` → `*.in_proj.weight`  (packed QKV)
    private static func sanitizeTemporalWeights(
        _ weights: [String: MLXArray]
    ) -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        for (key, value) in weights {
            var newKey = key
            var newValue = value

            // RMSNorm: alpha (1,1,D) → weight (D)
            if key.hasSuffix(".alpha") {
                newKey = String(key.dropLast(6)) + ".weight"
                if newValue.ndim == 3 {
                    newValue = newValue.squeezed(axes: [0, 1])
                }
            }

            // Attention in_proj: flat param → submodule
            if key.hasSuffix(".in_proj_weight") {
                newKey = String(key.dropLast("_weight".count)) + ".weight"
            }

            out[newKey] = newValue
        }
        return out
    }

    // MARK: - Depformer Weight Sanitization

    /// Sanitize depformer weights:
    /// - Rename `*.alpha` (1,1,D) → `*.weight` (D)
    /// - Rename `*.in_proj_weight` → `*.in_proj.weight`
    /// - Pack per-step FFN weights into MultiLinear format:
    ///   `gating.{step}.linear_in.weight` → concatenated `gating.linear_in.weight`
    private static func sanitizeDepformerWeights(
        _ weights: [String: MLXArray],
        numSteps: Int
    ) -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        var perStepWeights: [String: [(Int, MLXArray)]] = [:]

        for (key, value) in weights {
            var newKey = key
            var newValue = value

            // RMSNorm: alpha (1,1,D) → weight (D)
            if key.hasSuffix(".alpha") {
                newKey = String(key.dropLast(6)) + ".weight"
                if newValue.ndim == 3 {
                    newValue = newValue.squeezed(axes: [0, 1])
                }
                out[newKey] = newValue
                continue
            }

            // Attention in_proj: flat param → submodule
            if key.hasSuffix(".in_proj_weight") {
                newKey = String(key.dropLast("_weight".count)) + ".weight"
                out[newKey] = newValue
                continue
            }

            // Per-step FFN: detect gating.{step}.linear_in/out.weight pattern
            if let match = parsePerStepGatingKey(key) {
                perStepWeights[match.packedKey, default: []].append((match.step, value))
                continue
            }

            out[newKey] = newValue
        }

        // Pack per-step weights into MultiLinear format
        for (packedKey, stepWeights) in perStepWeights {
            let sorted = stepWeights.sorted { $0.0 < $1.0 }
            let packed = concatenated(sorted.map { $0.1 }, axis: 0)
            out[packedKey] = packed
        }

        return out
    }

    // MARK: - Embedding Weight Splitting

    private static func splitEmbeddingWeights(
        _ weights: [String: MLXArray]
    ) -> (temporal: [String: MLXArray], depformer: [String: MLXArray]) {
        var temporal: [String: MLXArray] = [:]
        var depformer: [String: MLXArray] = [:]

        for (key, value) in weights {
            if key.hasPrefix("text_emb.") || key.hasPrefix("emb.") || key.hasPrefix("text_linear.") {
                temporal[key] = value
            } else if key.hasPrefix("depformer_emb.") || key.hasPrefix("depformer_text_emb.") || key.hasPrefix("linears.") {
                depformer[key] = value
            }
        }

        return (temporal, depformer)
    }

    // MARK: - Per-step Gating Key Parser

    private struct PerStepGatingMatch {
        let packedKey: String
        let step: Int
    }

    private static func parsePerStepGatingKey(_ key: String) -> PerStepGatingMatch? {
        let parts = key.split(separator: ".")
        guard parts.count == 6,
              parts[0] == "layers",
              parts[2] == "gating",
              let step = Int(parts[3]),
              (parts[4] == "linear_in" || parts[4] == "linear_out"),
              parts[5] == "weight"
        else { return nil }

        let packedKey = "\(parts[0]).\(parts[1]).\(parts[2]).\(parts[4]).\(parts[5])"
        return PerStepGatingMatch(packedKey: packedKey, step: step)
    }
}

// MARK: - Errors

public enum PersonaPlexError: Error, LocalizedError {
    case missingWeightFile(String)
    case missingKey(String, in: String)
    case invalidAudio(String)
    case generationFailed(String)

    public var errorDescription: String? {
        switch self {
        case .missingWeightFile(let file):
            return "Missing weight file: \(file)"
        case .missingKey(let key, let file):
            return "Missing key '\(key)' in \(file)"
        case .invalidAudio(let msg):
            return "Invalid audio: \(msg)"
        case .generationFailed(let msg):
            return "Generation failed: \(msg)"
        }
    }
}
