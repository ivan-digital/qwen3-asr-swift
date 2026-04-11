import Foundation
import MLX
import MLXNN
import MLXCommon
import AudioCommon

/// Meta Omnilingual ASR — CTC variant, MLX/Metal backend.
///
/// This is the MLX counterpart to ``OmnilingualASRModel`` (CoreML/ANE). It
/// loads any of the published `aufklarer/Omnilingual-ASR-CTC-{300M,1B,3B,7B}-MLX-{4,8}bit`
/// repos. The encoder is wav2vec 2.0 with quantised attention/FFN projections;
/// the CTC head is a single quantised linear over the shared 10288-entry
/// SentencePiece vocabulary covering 1,672 languages.
///
/// Like the CoreML model, the CTC variant is language-agnostic — the
/// `language` parameter on the `SpeechRecognitionModel` protocol is intentionally
/// ignored. The 40 s reference cap and utterance-level layer-norm preprocessing
/// match the CoreML path exactly.
public final class OmnilingualASRMLXModel {
    public let config: OmnilingualMLXConfig
    public let frontend: Wav2Vec2Frontend
    public let encoder: Wav2Vec2Encoder
    public let ctcHead: CTCHead
    private let vocabulary: OmnilingualVocabulary
    var _isLoaded: Bool = true

    public static let layerNormEpsilon: Float = 1e-5
    public static let maxAudioSeconds: Double = 40.0

    init(
        config: OmnilingualMLXConfig,
        vocabulary: OmnilingualVocabulary
    ) {
        self.config = config
        self.frontend = Wav2Vec2Frontend(config: config)
        self.encoder = Wav2Vec2Encoder(config: config)
        self.ctcHead = CTCHead(config: config)
        self.vocabulary = vocabulary
    }

    public var sampleRate: Int { config.sampleRate }

    // MARK: - Loading

    /// Download and load the model from HuggingFace.
    public static func fromPretrained(
        variant: OmnilingualMLXConfig.Variant = .m300,
        bits: Int = 4,
        modelId: String? = nil,
        cacheDir: URL? = nil,
        offlineMode: Bool = false,
        progressHandler: ((Double, String) -> Void)? = nil
    ) async throws -> OmnilingualASRMLXModel {
        let resolvedModelId = modelId ?? OmnilingualMLXConfig.defaultModelId(
            variant: variant, bits: bits)
        let detectedVariant = detectVariant(from: resolvedModelId) ?? variant
        let detectedBits = detectBits(from: resolvedModelId) ?? bits
        let mlxConfig = OmnilingualMLXConfig.variant(detectedVariant, bits: detectedBits)

        AudioLog.modelLoading.info("Loading Omnilingual MLX model: \(resolvedModelId)")

        let resolvedCacheDir: URL
        do {
            resolvedCacheDir = try cacheDir ?? HuggingFaceDownloader.getCacheDirectory(for: resolvedModelId)
        } catch {
            throw AudioModelError.modelLoadFailed(
                modelId: resolvedModelId, reason: "Failed to resolve cache directory", underlying: error)
        }

        progressHandler?(0.0, "Downloading model...")
        do {
            try await HuggingFaceDownloader.downloadWeights(
                modelId: resolvedModelId,
                to: resolvedCacheDir,
                additionalFiles: ["config.json", "tokenizer.model", "model.safetensors"],
                offlineMode: offlineMode
            ) { fraction in
                progressHandler?(fraction * 0.85, "Downloading weights...")
            }
        } catch {
            throw AudioModelError.modelLoadFailed(
                modelId: resolvedModelId, reason: "Download failed", underlying: error)
        }

        progressHandler?(0.85, "Loading tokenizer...")
        let tokenizerPath = resolvedCacheDir.appendingPathComponent("tokenizer.model")
        guard FileManager.default.fileExists(atPath: tokenizerPath.path) else {
            throw AudioModelError.modelLoadFailed(
                modelId: resolvedModelId,
                reason: "tokenizer.model not found at \(tokenizerPath.path)",
                underlying: nil)
        }
        let tokenizerConfig = OmnilingualConfig.Tokenizer(
            kind: "sentencepiece", file: "tokenizer.model",
            bosIdx: 0, padIdx: 1, eosIdx: 2, unkIdx: 3)
        let vocabulary = try OmnilingualVocabulary.load(from: tokenizerPath, tokenizer: tokenizerConfig)
        AudioLog.modelLoading.debug("Loaded SentencePiece vocabulary: \(vocabulary.count) pieces")

        progressHandler?(0.88, "Building modules...")
        let model = OmnilingualASRMLXModel(config: mlxConfig, vocabulary: vocabulary)

        progressHandler?(0.92, "Loading weights...")
        try OmnilingualMLXWeightLoader.loadWeights(into: model, from: resolvedCacheDir)

        // Force materialisation so the first inference doesn't pay the lazy
        // weight-evaluation cost.
        eval(model.frontend, model.encoder, model.ctcHead)

        MetalBudget.pinMemory()
        progressHandler?(1.0, "Model loaded")
        AudioLog.modelLoading.info("Omnilingual MLX model loaded successfully")
        return model
    }

    private static func detectVariant(from modelId: String) -> OmnilingualMLXConfig.Variant? {
        for v in OmnilingualMLXConfig.Variant.allCases where modelId.contains("CTC-\(v.rawValue)-") {
            return v
        }
        return nil
    }

    private static func detectBits(from modelId: String) -> Int? {
        if modelId.contains("4bit") { return 4 }
        if modelId.contains("8bit") { return 8 }
        return nil
    }

    // MARK: - Warmup

    /// Run a single forward pass on silence to materialise compute graphs.
    public func warmUp() throws {
        let dummySamples = config.sampleRate  // 1 s
        let dummy = [Float](repeating: 0, count: dummySamples)
        _ = try transcribeAudio(dummy, sampleRate: config.sampleRate)
    }

    // MARK: - Inference

    /// Transcribe an audio buffer to text. Resamples to 16 kHz internally.
    /// Hard-capped at 40 s, matching the reference Python pipeline.
    public func transcribeAudio(
        _ audio: [Float], sampleRate: Int, language: String? = nil
    ) throws -> String {
        guard _isLoaded else {
            throw AudioModelError.inferenceFailed(operation: "transcribe", reason: "Model not loaded")
        }

        let samples: [Float]
        if sampleRate != config.sampleRate {
            samples = AudioFileLoader.resample(audio, from: sampleRate, to: config.sampleRate)
        } else {
            samples = audio
        }
        let durationSec = Double(samples.count) / Double(config.sampleRate)
        if durationSec > Self.maxAudioSeconds {
            throw AudioModelError.inferenceFailed(
                operation: "transcribe",
                reason: "Input \(String(format: "%.1f", durationSec))s exceeds Omnilingual cap of \(Int(Self.maxAudioSeconds))s. Segment with SpeechVAD or use ParakeetStreamingASR.")
        }
        if samples.isEmpty {
            return ""
        }

        // Layer-norm normalise the raw waveform (matches reference apply_audio_normalization).
        let normalized = OmnilingualASRModel.layerNormalize(samples, eps: Self.layerNormEpsilon)

        // [B=1, T, C=1]
        let input = MLXArray(normalized).reshaped([1, normalized.count, 1])

        let tEnc0 = CFAbsoluteTimeGetCurrent()
        let frontendOut = frontend(input)
        let encoderOut = encoder(frontendOut)
        let logits = ctcHead(encoderOut)
        eval(logits)
        let tEnc1 = CFAbsoluteTimeGetCurrent()

        // logits shape: [1, T', vocab]
        let shape = logits.shape
        precondition(shape.count == 3 && shape[0] == 1, "Unexpected logits shape \(shape)")
        let T = shape[1]
        let V = shape[2]

        // Argmax → [1, T'] then collapse duplicates in Swift land.
        let argmax = logits.argMax(axis: -1)
        // Materialise to [Int]
        let ids = argmax.reshaped([T]).asArray(Int32.self).map { Int($0) }
        let collapsed = collapseConsecutiveDuplicates(ids)

        let text = vocabulary.decode(collapsed)
        let dt = (tEnc1 - tEnc0) * 1000
        AudioLog.inference.info("Omnilingual MLX (\(self.config.variant.rawValue), \(self.config.bits)bit): forward=\(String(format: "%.1f", dt))ms T=\(T) V=\(V)")
        _ = T
        _ = V
        return text
    }

    private func collapseConsecutiveDuplicates(_ ids: [Int]) -> [Int] {
        guard !ids.isEmpty else { return [] }
        var out: [Int] = []
        out.reserveCapacity(ids.count)
        var prev = -1
        for id in ids {
            if id != prev {
                out.append(id)
                prev = id
            }
        }
        return out
    }
}
