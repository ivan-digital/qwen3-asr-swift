import Foundation
import MLX
import MLXNN
import PersonaPlex
import AudioCommon   // HuggingFaceDownloader, SentencePieceModel

// MARK: - Errors

public enum HibikiError: Error, LocalizedError {
    case missingWeightFile(String)
    case missingKey(String, in: String)
    case invalidAudio(String)
    case generationFailed(String)

    public var errorDescription: String? {
        switch self {
        case .missingWeightFile(let file): return "Missing weight file: \(file)"
        case .missingKey(let k, let f):    return "Missing key '\(k)' in \(f)"
        case .invalidAudio(let m):         return "Invalid audio: \(m)"
        case .generationFailed(let m):     return "Generation failed: \(m)"
        }
    }
}

// MARK: - Model

/// Hibiki streaming speech-to-speech translation model (Zero-3B preset by default).
///
/// Pipeline:
/// 1. Encode source-language audio with Mimi → `[1, 16, T]` codebook tokens.
/// 2. Per Mimi frame (12.5 Hz), feed source tokens into the temporal transformer's
///    audio embeddings for streams 1..16, and feed previously generated target
///    tokens into streams 17..32.
/// 3. Sample text token from temporal logits.
/// 4. Run depformer for 16 generation steps to produce the next target codebook.
/// 5. Decode accumulated target codebooks via Mimi → English audio.
///
/// The driver (translate/translateStream) lives in `HibikiTranslate.swift`. This
/// type is the `Module` shell that `HibikiWeightLoader` inflates.
public final class HibikiTranslateModel: Module {
    public static let defaultModelId  = "aufklarer/Hibiki-Zero-3B-MLX-4bit"
    public static let modelId8bit     = "aufklarer/Hibiki-Zero-3B-MLX-8bit"

    public var cfg: HibikiConfig
    public private(set) var modelId: String

    @ModuleInfo public var temporal: HibikiTemporalTransformer
    @ModuleInfo public var depformer: HibikiDepformer
    public let mimi: Mimi

    public init(cfg: HibikiConfig = .zero3B, modelId: String = HibikiTranslateModel.defaultModelId) {
        self.cfg = cfg
        self.modelId = modelId
        self._temporal = ModuleInfo(wrappedValue: HibikiTemporalTransformer(cfg: cfg.temporal))
        self._depformer = ModuleInfo(wrappedValue: HibikiDepformer(
            cfg: cfg.depformer, temporalDim: cfg.temporal.dim))
        self.mimi = Mimi(cfg: cfg.mimi)
    }

    /// Pre-compile temporal transformer (~30% faster after warmup).
    public func warmUp() {
        temporal.setupCompilation()
    }

    // MARK: - Model loading

    /// Download model artifacts from HuggingFace (if not cached) and load weights.
    /// Reads `config.json` for quantization settings.
    public static func fromPretrained(
        modelId: String = defaultModelId,
        cacheDir: URL? = nil,
        offlineMode: Bool = false,
        progressHandler: ((Double, String) -> Void)? = nil
    ) async throws -> HibikiTranslateModel {
        progressHandler?(0.05, "Downloading Hibiki Zero-3B weights...")
        let modelDir = try cacheDir ?? HuggingFaceDownloader.getCacheDirectory(for: modelId)

        let weightFiles = [
            "temporal.safetensors",
            "depformer.safetensors",
            "embeddings.safetensors",
            "mimi.safetensors",
            "tokenizer_spm_48k_multi6_2.model",
            "config.json",
        ]

        try await HuggingFaceDownloader.downloadWeights(
            modelId: modelId,
            to: modelDir,
            additionalFiles: weightFiles,
            offlineMode: offlineMode
        ) { progress in
            progressHandler?(0.05 + progress * 0.6, "Downloading...")
        }

        // Read config.json (quantization, dims, schedule).
        let configFile = modelDir.appendingPathComponent("config.json")
        let cfg: HibikiConfig
        if FileManager.default.fileExists(atPath: configFile.path) {
            cfg = try HibikiConfig.load(from: configFile)
        } else {
            cfg = .zero3B
        }

        let model = HibikiTranslateModel(cfg: cfg, modelId: modelId)

        progressHandler?(0.7, "Loading model weights...")
        try HibikiWeightLoader.loadWeights(model: model, from: modelDir) { p, msg in
            progressHandler?(0.7 + p * 0.15, msg)
        }
        try HibikiWeightLoader.loadMimi(model: model.mimi, from: modelDir) { p, msg in
            progressHandler?(0.85 + p * 0.1, msg)
        }

        progressHandler?(1.0, "Hibiki Zero-3B ready")
        return model
    }
}
