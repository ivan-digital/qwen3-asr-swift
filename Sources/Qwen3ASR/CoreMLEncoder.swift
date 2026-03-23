#if canImport(CoreML)
import CoreML
import Foundation
import MLX
import AudioCommon

/// CoreML audio encoder for Qwen3-ASR.
///
/// Runs the audio encoder on Neural Engine via CoreML instead of GPU via MLX.
/// Produces audio embeddings that feed into the MLX text decoder. This enables
/// lower power consumption on macOS and is a step toward full iOS deployment.
///
/// The encoder processes mel spectrograms without chunking/block attention.
/// For typical audio lengths (< 30s), this produces equivalent results to the
/// chunked MLX encoder.
public class CoreMLASREncoder {
    private let model: MLModel
    private let enumeratedMelLengths: [Int]

    public static let defaultModelId = "aufklarer/Qwen3-ASR-CoreML"

    public init(model: MLModel, enumeratedMelLengths: [Int] = [100, 200, 400, 600, 800, 1000, 1500, 2000, 3000]) {
        self.model = model
        self.enumeratedMelLengths = enumeratedMelLengths
    }

    /// Load encoder from a directory containing `encoder.mlmodelc`.
    public static func load(
        from directory: URL,
        computeUnits: MLComputeUnits = .all
    ) throws -> CoreMLASREncoder {
        let modelURL = directory.appendingPathComponent("encoder.mlmodelc", isDirectory: true)
        guard FileManager.default.fileExists(atPath: modelURL.path) else {
            throw AudioModelError.modelLoadFailed(
                modelId: "encoder",
                reason: "CoreML encoder not found at \(modelURL.path)")
        }

        let config = MLModelConfiguration()
        config.computeUnits = computeUnits
        let model = try MLModel(contentsOf: modelURL, configuration: config)
        return CoreMLASREncoder(model: model)
    }

    /// Load encoder from HuggingFace.
    public static func fromPretrained(
        modelId: String = defaultModelId,
        computeUnits: MLComputeUnits = .all,
        progressHandler: ((Double, String) -> Void)? = nil
    ) async throws -> CoreMLASREncoder {
        let cacheDir = try HuggingFaceDownloader.getCacheDirectory(for: modelId)

        progressHandler?(0.0, "Downloading CoreML encoder...")
        try await HuggingFaceDownloader.downloadWeights(
            modelId: modelId,
            to: cacheDir,
            additionalFiles: ["encoder.mlmodelc/**", "config.json"]
        ) { fraction in
            progressHandler?(fraction * 0.8, "Downloading CoreML encoder...")
        }

        progressHandler?(0.9, "Loading CoreML encoder...")
        let encoder = try load(from: cacheDir, computeUnits: computeUnits)
        progressHandler?(1.0, "Ready")
        return encoder
    }

    /// Warm up the encoder with a short dummy input to trigger CoreML compilation.
    public func warmUp() throws {
        let minT = enumeratedMelLengths.first ?? 100
        let dummy = try MLMultiArray(shape: [1, 128, minT as NSNumber], dataType: .float32)
        let input = try MLDictionaryFeatureProvider(dictionary: [
            "mel": MLFeatureValue(multiArray: dummy),
        ])
        _ = try model.prediction(from: input)
    }

    /// Encode mel spectrogram to audio embeddings.
    ///
    /// - Parameter melFeatures: Mel spectrogram as MLXArray `[128, T]`
    /// - Returns: Audio embeddings as MLXArray `[1, T/8, 1024]`
    public func encode(_ melFeatures: MLXArray) throws -> MLXArray {
        let melBins = melFeatures.dim(0)
        let melTime = melFeatures.dim(1)

        guard let targetLength = enumeratedMelLengths.first(where: { $0 >= melTime }) else {
            throw AudioModelError.inferenceFailed(
                operation: "CoreML encoder",
                reason: "Audio too long: \(melTime) mel frames exceeds max \(enumeratedMelLengths.last!)")
        }

        // Convert MLXArray [128, T] → MLMultiArray [1, 128, targetLength]
        let melData: [Float] = melFeatures.asArray(Float.self)
        let melArray = try MLMultiArray(
            shape: [1, melBins as NSNumber, targetLength as NSNumber],
            dataType: .float32)
        let ptr = melArray.dataPointer.assumingMemoryBound(to: Float.self)

        for bin in 0..<melBins {
            let srcOffset = bin * melTime
            let dstOffset = bin * targetLength
            for t in 0..<melTime {
                ptr[dstOffset + t] = melData[srcOffset + t]
            }
            for t in melTime..<targetLength {
                ptr[dstOffset + t] = 0
            }
        }

        // Run prediction
        let input = try MLDictionaryFeatureProvider(dictionary: [
            "mel": MLFeatureValue(multiArray: melArray),
        ])
        let output = try model.prediction(from: input)

        guard let embeddings = output.featureValue(for: "audio_embeddings")?.multiArrayValue else {
            throw AudioModelError.inferenceFailed(
                operation: "CoreML encoder", reason: "Missing audio_embeddings output")
        }

        // Convert MLMultiArray → MLXArray
        return multiArrayToMLXArray(embeddings)
    }

    // MARK: - MLX-free encoding (for iOS background / pure CoreML path)

    /// Encode mel spectrogram to audio embeddings without any MLXArray dependency.
    ///
    /// Accepts raw `[Float]` mel data in `[melBins, timeFrames]` layout (the same
    /// layout produced by `WhisperFeatureExtractor.extractFeaturesRaw`).
    /// Returns the encoder output as `MLMultiArray` directly, avoiding the
    /// Metal GPU eval that `MLXArray` would trigger.
    ///
    /// - Parameters:
    ///   - melData: Flat float array in row-major `[melBins, timeFrames]` order
    ///   - melBins: Number of mel frequency bins (typically 128)
    ///   - timeFrames: Number of time frames
    /// - Returns: Audio embeddings as `MLMultiArray` with shape `[1, T/8, 1024]`
    public func encode(melData: [Float], melBins: Int, timeFrames: Int) throws -> MLMultiArray {
        guard let targetLength = enumeratedMelLengths.first(where: { $0 >= timeFrames }) else {
            throw AudioModelError.inferenceFailed(
                operation: "CoreML encoder",
                reason: "Audio too long: \(timeFrames) mel frames exceeds max \(enumeratedMelLengths.last!)")
        }

        // Create MLMultiArray [1, melBins, targetLength] directly from [Float]
        let melArray = try MLMultiArray(
            shape: [1, melBins as NSNumber, targetLength as NSNumber],
            dataType: .float32)
        let ptr = melArray.dataPointer.assumingMemoryBound(to: Float.self)

        // melData layout is [melBins, timeFrames] (row-major)
        // MLMultiArray layout is [1, melBins, targetLength] (row-major, batch dim = 1)
        for bin in 0..<melBins {
            let srcOffset = bin * timeFrames
            let dstOffset = bin * targetLength
            for t in 0..<timeFrames {
                ptr[dstOffset + t] = melData[srcOffset + t]
            }
            // Zero-pad remaining time steps
            for t in timeFrames..<targetLength {
                ptr[dstOffset + t] = 0
            }
        }

        // Run prediction
        let input = try MLDictionaryFeatureProvider(dictionary: [
            "mel": MLFeatureValue(multiArray: melArray),
        ])
        let output = try model.prediction(from: input)

        guard let embeddings = output.featureValue(for: "audio_embeddings")?.multiArrayValue else {
            throw AudioModelError.inferenceFailed(
                operation: "CoreML encoder", reason: "Missing audio_embeddings output")
        }

        return embeddings
    }

    /// Convenience: encode a `MelFeatures` struct directly.
    public func encode(melFeatures: MelFeatures) throws -> MLMultiArray {
        return try encode(melData: melFeatures.data, melBins: melFeatures.melBins, timeFrames: melFeatures.timeFrames)
    }

    private func multiArrayToMLXArray(_ array: MLMultiArray) -> MLXArray {
        let shape = array.shape.map { $0.intValue }
        let count = array.count

        switch array.dataType {
        case .float16:
            let src = array.dataPointer.assumingMemoryBound(to: Float16.self)
            var floats = [Float](repeating: 0, count: count)
            for i in 0..<count { floats[i] = Float(src[i]) }
            return MLXArray(floats, shape)
        case .float32:
            let src = array.dataPointer.assumingMemoryBound(to: Float.self)
            return MLXArray(Array(UnsafeBufferPointer(start: src, count: count)), shape)
        default:
            let src = array.dataPointer.assumingMemoryBound(to: Float.self)
            return MLXArray(Array(UnsafeBufferPointer(start: src, count: count)), shape)
        }
    }
}

// MARK: - Qwen3ASRModel Integration

extension Qwen3ASRModel {
    /// Transcribe audio using CoreML encoder + MLX text decoder.
    ///
    /// This hybrid approach runs the encoder on Neural Engine (CoreML) and the
    /// text decoder on GPU (MLX), combining the power efficiency of ANE with
    /// the flexibility of MLX for autoregressive decoding.
    public func transcribe(
        audio: [Float],
        sampleRate: Int = 16000,
        language: String? = nil,
        maxTokens: Int = 448,
        coremlEncoder: CoreMLASREncoder
    ) throws -> String {
        let melFeatures = featureExtractor.process(audio, sampleRate: sampleRate)

        // CoreML encoder returns [1, T/8, 1024] (batch dim included)
        let audioEmbeds = try coremlEncoder.encode(melFeatures)

        guard let textDecoder = textDecoder else {
            return "[CoreML encoded: \(audioEmbeds.shape)] - Text decoder not loaded"
        }

        return generateText(
            audioEmbeds: audioEmbeds,
            textDecoder: textDecoder,
            language: language,
            maxTokens: maxTokens
        )
    }
}
#endif
