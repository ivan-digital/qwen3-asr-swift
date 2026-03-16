#if canImport(CoreML)
import CoreML
import Foundation
import AVFoundation
import AudioCommon

/// CAM++ speaker embedding model (CoreML, Neural Engine).
///
/// Produces 192-dim speaker embeddings compatible with CosyVoice3 voice cloning.
/// The model runs on Apple Neural Engine via CoreML for efficient inference.
///
/// - Note: Download from HuggingFace on first use (~14 MB).
public final class CamPlusPlusSpeaker {
    private let model: MLModel
    private let melExtractor: CamPlusPlusMelExtractor

    /// Default HuggingFace model ID
    public static let defaultModelId = "aufklarer/CamPlusPlus-Speaker-CoreML"

    /// Embedding dimensionality
    public static let embeddingDim = 192

    init(model: MLModel) {
        self.model = model
        self.melExtractor = CamPlusPlusMelExtractor()
    }

    /// Load CAM++ from HuggingFace.
    ///
    /// Downloads the CoreML model on first use, caches locally.
    public static func fromPretrained(
        modelId: String = CamPlusPlusSpeaker.defaultModelId,
        progressHandler: ((Double, String) -> Void)? = nil
    ) async throws -> CamPlusPlusSpeaker {
        let cacheDir = try HuggingFaceDownloader.getCacheDirectory(for: modelId)

        let modelURL = cacheDir.appendingPathComponent("CamPlusPlus.mlmodelc", isDirectory: true)
        if !FileManager.default.fileExists(atPath: modelURL.path) {
            progressHandler?(0.0, "Downloading CAM++ speaker model...")
            try await HuggingFaceDownloader.downloadWeights(
                modelId: modelId,
                to: cacheDir,
                additionalFiles: ["CamPlusPlus.mlmodelc/**"],
                progressHandler: { progress in
                    progressHandler?(progress * 0.8, "Downloading CAM++ speaker model...")
                }
            )
        }

        progressHandler?(0.8, "Loading CoreML model...")

        guard FileManager.default.fileExists(atPath: modelURL.path) else {
            throw AudioModelError.modelLoadFailed(
                modelId: modelId,
                reason: "CoreML model not found at \(modelURL.path)")
        }

        let mlConfig = MLModelConfiguration()
        mlConfig.computeUnits = .cpuAndNeuralEngine

        let mlModel: MLModel
        do {
            mlModel = try MLModel(contentsOf: modelURL, configuration: mlConfig)
        } catch {
            throw AudioModelError.modelLoadFailed(
                modelId: modelId,
                reason: "Failed to load CoreML model: \(error.localizedDescription)")
        }

        progressHandler?(1.0, "Ready")
        return CamPlusPlusSpeaker(model: mlModel)
    }

    /// Extract a 192-dim speaker embedding from audio.
    ///
    /// - Parameters:
    ///   - audio: PCM Float32 samples
    ///   - sampleRate: Sample rate of input audio (resampled to 16kHz if needed)
    /// - Returns: 192-dim speaker embedding (not L2-normalized; flow model normalizes internally)
    public func embed(audio: [Float], sampleRate: Int = 16000) throws -> [Float] {
        // Resample to 16kHz if needed
        let samples: [Float]
        if sampleRate != 16000 {
            samples = AudioFileLoader.resample(audio, from: sampleRate, to: 16000)
        } else {
            samples = audio
        }

        guard samples.count >= 1600 else {  // minimum ~0.1s
            throw CosyVoiceTTSError.invalidInput(
                "Audio too short for speaker embedding (\(samples.count) samples, need >= 1600)")
        }

        // Extract 80-dim log-mel features
        let (melSpec, nFrames) = melExtractor.extractRaw(samples)

        guard nFrames >= 10 else {
            throw CosyVoiceTTSError.invalidInput(
                "Too few mel frames for speaker embedding (\(nFrames), need >= 10)")
        }

        // Fixed input size: 500 frames (~5s at 16kHz).
        // The CoreML model uses fixed shape to avoid dynamic reshape issues.
        // Pad short audio with zeros, truncate long audio.
        let targetFrames = 500

        // Create input: [1, 500, 80] float16
        let melArray = try MLMultiArray(
            shape: [1, targetFrames as NSNumber, 80],
            dataType: .float16
        )
        let melPtr = melArray.dataPointer.assumingMemoryBound(to: Float16.self)

        let copyFrames = min(nFrames, targetFrames)
        let copyCount = copyFrames * 80
        for i in 0..<copyCount {
            melPtr[i] = Float16(melSpec[i])
        }
        // Zero-pad remaining frames
        let totalElements = targetFrames * 80
        for i in copyCount..<totalElements {
            melPtr[i] = 0
        }

        let input = try MLDictionaryFeatureProvider(dictionary: [
            "mel_features": MLFeatureValue(multiArray: melArray),
        ])

        let result = try model.prediction(from: input)

        // Extract "embedding" output: [1, 192]
        guard let embArray = result.featureValue(for: "embedding")?.multiArrayValue else {
            throw CosyVoiceTTSError.generationFailed("Missing 'embedding' output from CAM++ model")
        }

        var embedding = [Float](repeating: 0, count: Self.embeddingDim)
        let embPtr = embArray.dataPointer.assumingMemoryBound(to: Float16.self)
        for i in 0..<Self.embeddingDim {
            embedding[i] = Float(embPtr[i])
        }

        return embedding
    }

}
#endif
