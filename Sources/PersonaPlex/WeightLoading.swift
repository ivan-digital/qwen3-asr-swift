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
        // Load temporal transformer weights (4-bit quantized)
        progressHandler?(0.1, "Loading temporal transformer...")
        let temporalFile = directory.appendingPathComponent("temporal.safetensors")
        if FileManager.default.fileExists(atPath: temporalFile.path) {
            let weights = try MLX.loadArrays(url: temporalFile)
            let params = ModuleParameters.unflattened(weights)
            try model.temporal.update(parameters: params, verify: .noUnusedKeys)
        }

        // Load embeddings (text + audio embeddings, output heads)
        progressHandler?(0.3, "Loading embeddings...")
        let embFile = directory.appendingPathComponent("embeddings.safetensors")
        if FileManager.default.fileExists(atPath: embFile.path) {
            let weights = try MLX.loadArrays(url: embFile)
            let remapped = remapEmbeddingKeys(weights)
            let params = ModuleParameters.unflattened(remapped)
            try model.temporal.update(parameters: params, verify: .noUnusedKeys)
        }

        // Load depformer weights (BF16, small)
        progressHandler?(0.5, "Loading depformer...")
        let depFile = directory.appendingPathComponent("depformer.safetensors")
        if FileManager.default.fileExists(atPath: depFile.path) {
            let weights = try MLX.loadArrays(url: depFile)
            let params = ModuleParameters.unflattened(weights)
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
        try model.update(parameters: params, verify: .all)

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

    // MARK: - Key Remapping

    /// Remap embedding weight keys from flat conversion format to model hierarchy
    private static func remapEmbeddingKeys(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        for (key, value) in weights {
            // text_emb.weight -> text_emb.weight (no change)
            // emb.{i}.weight -> emb.{i}.weight (no change)
            // text_linear.weight -> text_linear.weight (no change)
            out[key] = value
        }
        return out
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
