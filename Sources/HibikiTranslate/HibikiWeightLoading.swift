import Foundation
import MLX
import MLXNN
import PersonaPlex   // Mimi, EuclideanCodebook, sanitize helpers

// MARK: - Weight Loader

public enum HibikiWeightLoader {

    /// Load all four safetensors from a Hibiki Zero-3B model directory:
    /// - `temporal.safetensors`  (4-bit / 8-bit quantized temporal transformer)
    /// - `depformer.safetensors` (4-bit / 8-bit quantized scheduled depformer)
    /// - `embeddings.safetensors` (BF16 — text/audio embeddings + output heads)
    /// - `mimi.safetensors`      (BF16 → cast to fp32 internally for codec)
    public static func loadWeights(
        model: HibikiTranslateModel,
        from directory: URL,
        progressHandler: ((Double, String) -> Void)? = nil
    ) throws {
        // Temporal
        progressHandler?(0.1, "Loading temporal transformer...")
        let temporalFile = directory.appendingPathComponent("temporal.safetensors")
        if FileManager.default.fileExists(atPath: temporalFile.path) {
            let weights = try MLX.loadArrays(url: temporalFile)
            let sanitized = sanitizeTemporalWeights(weights)
            let params = ModuleParameters.unflattened(sanitized)
            try model.temporal.update(parameters: params, verify: .noUnusedKeys)
        } else {
            throw HibikiError.missingWeightFile("temporal.safetensors")
        }

        // Embeddings (mixed temporal + depformer keys)
        progressHandler?(0.3, "Loading embeddings...")
        let embFile = directory.appendingPathComponent("embeddings.safetensors")
        if FileManager.default.fileExists(atPath: embFile.path) {
            let weights = try MLX.loadArrays(url: embFile)
            let (temporalEmb, depformerEmb) = splitEmbeddingWeights(weights)

            let tParams = ModuleParameters.unflattened(temporalEmb)
            try model.temporal.update(parameters: tParams, verify: .noUnusedKeys)

            let dParams = ModuleParameters.unflattened(depformerEmb)
            try model.depformer.update(parameters: dParams, verify: .noUnusedKeys)
        } else {
            throw HibikiError.missingWeightFile("embeddings.safetensors")
        }

        // Depformer
        progressHandler?(0.5, "Loading depformer...")
        let depFile = directory.appendingPathComponent("depformer.safetensors")
        if FileManager.default.fileExists(atPath: depFile.path) {
            let weights = try MLX.loadArrays(url: depFile)
            let sanitized = sanitizeDepformerWeights(weights)
            let params = ModuleParameters.unflattened(sanitized)
            try model.depformer.update(parameters: params, verify: .noUnusedKeys)
        } else {
            throw HibikiError.missingWeightFile("depformer.safetensors")
        }

        eval(model.temporal, model.depformer)
        progressHandler?(0.7, "Model weights loaded")
    }

    /// Load the Mimi codec from `mimi.safetensors`. Same protocol as PersonaPlex.
    public static func loadMimi(
        model: Mimi,
        from directory: URL,
        progressHandler: ((Double, String) -> Void)? = nil
    ) throws {
        progressHandler?(0.0, "Loading Mimi codec...")
        let mimiFile = directory.appendingPathComponent("mimi.safetensors")
        guard FileManager.default.fileExists(atPath: mimiFile.path) else {
            throw HibikiError.missingWeightFile("mimi.safetensors")
        }
        var weights = try MLX.loadArrays(url: mimiFile)
        weights = model.sanitize(weights: weights)
        let params = ModuleParameters.unflattened(weights)
        try model.update(parameters: params, verify: .noUnusedKeys)

        // Initialize EuclideanCodebook running stats (shared with PersonaPlex Mimi).
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

    // MARK: - Temporal sanitization

    /// Sanitize temporal-transformer keys to match Swift module hierarchy.
    /// - `*.alpha` (1,1,D) → `*.weight` (D)
    /// - `*.in_proj_weight` → `*.in_proj.weight` (+ `_scales`/`_biases` for quantized)
    private static func sanitizeTemporalWeights(
        _ weights: [String: MLXArray]
    ) -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        for (key, value) in weights {
            var newKey = key
            var newValue = value

            if key.hasSuffix(".alpha") {
                newKey = String(key.dropLast(6)) + ".weight"
                if newValue.ndim == 3 { newValue = newValue.squeezed(axes: [0, 1]) }
            }

            for suffix in ["_weight", "_scales", "_biases"] {
                let needle = ".in_proj" + suffix
                if key.hasSuffix(needle) {
                    let dotSuffix = "." + String(suffix.dropFirst())
                    newKey = String(key.dropLast(needle.count)) + ".in_proj" + dotSuffix
                    break
                }
            }

            out[newKey] = newValue
        }
        return out
    }

    // MARK: - Depformer sanitization

    /// Sanitize depformer keys:
    /// - `*.alpha` → `*.weight` (RMSNorm)
    /// - `*.in_proj_weight` / `*.out_proj_weight` → `*.in_proj.weight` / `*.out_proj.weight`
    /// - Per-step `gating.{step}.linear_in/out.{weight,scales,biases}` →
    ///   packed `gating.linear_in/out.{weight,scales,biases}` of
    ///   `[numUniqueSlices * outDim, packedInDim]` (concatenated along axis 0,
    ///   sorted by step). For Hibiki Zero-3B/2B this packs **9** slices.
    private static func sanitizeDepformerWeights(
        _ weights: [String: MLXArray]
    ) -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        var perStepWeights: [String: [(Int, MLXArray)]] = [:]

        for (key, value) in weights {
            var newKey = key
            var newValue = value

            if key.hasSuffix(".alpha") {
                newKey = String(key.dropLast(6)) + ".weight"
                if newValue.ndim == 3 { newValue = newValue.squeezed(axes: [0, 1]) }
                out[newKey] = newValue
                continue
            }

            var matchedProj = false
            for projName in ["in_proj", "out_proj"] {
                for suffix in ["_weight", "_scales", "_biases"] {
                    let needle = "." + projName + suffix
                    if key.hasSuffix(needle) {
                        let dotSuffix = "." + String(suffix.dropFirst())
                        newKey = String(key.dropLast(needle.count)) + "." + projName + dotSuffix
                        out[newKey] = newValue
                        matchedProj = true
                        break
                    }
                }
                if matchedProj { break }
            }
            if matchedProj { continue }

            if let match = parsePerStepGatingKey(key) {
                perStepWeights[match.packedKey, default: []].append((match.step, value))
                continue
            }

            out[newKey] = newValue
        }

        // Concatenate per-step slices in step order (axis 0).
        // For Hibiki Zero-3B this yields 9 slices stacked.
        for (packedKey, stepWeights) in perStepWeights {
            let sorted = stepWeights.sorted { $0.0 < $1.0 }
            let packed = concatenated(sorted.map { $0.1 }, axis: 0)
            out[packedKey] = packed
        }

        return out
    }

    // MARK: - Embedding split

    /// Embeddings safetensors mixes temporal-side keys (text_emb, emb, text_linear)
    /// with depformer-side keys (depformer_emb, depformer_text_emb, linears).
    /// Hibiki Zero-3B has neither `condition_provider.*` nor `.low_rank.*` keys.
    private static func splitEmbeddingWeights(
        _ weights: [String: MLXArray]
    ) -> (temporal: [String: MLXArray], depformer: [String: MLXArray]) {
        var temporal: [String: MLXArray] = [:]
        var depformer: [String: MLXArray] = [:]
        for (key, value) in weights {
            if key.hasPrefix("text_emb.") || key.hasPrefix("emb.") || key.hasPrefix("text_linear.") {
                temporal[key] = value
            } else if key.hasPrefix("depformer_emb.") || key.hasPrefix("depformer_text_emb.")
                       || key.hasPrefix("linears.") {
                depformer[key] = value
            }
        }
        return (temporal, depformer)
    }

    // MARK: - Per-step gating parser

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
              (parts[5] == "weight" || parts[5] == "scales" || parts[5] == "biases")
        else { return nil }
        let packedKey = "\(parts[0]).\(parts[1]).\(parts[2]).\(parts[4]).\(parts[5])"
        return PerStepGatingMatch(packedKey: packedKey, step: step)
    }
}
