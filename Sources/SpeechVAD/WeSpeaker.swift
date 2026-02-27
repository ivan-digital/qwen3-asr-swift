import Foundation
import MLX
import AudioCommon

/// Speaker embedding model using WeSpeaker ResNet34-LM.
///
/// Produces 256-dimensional L2-normalized speaker embeddings from audio.
/// Uses 80-dim log-mel features at 16kHz.
///
/// This class is thread-safe: all properties are `let` and inference is pure computation
/// with no mutable state.
///
/// ```swift
/// let model = try await WeSpeakerModel.fromPretrained()
/// let embedding = model.embed(audio: samples, sampleRate: 16000)
/// // embedding: [Float] of length 256
/// ```
public final class WeSpeakerModel {

    /// The ResNet34 network
    let network: WeSpeakerNetwork

    /// Mel feature extractor
    let melExtractor: MelFeatureExtractor

    /// Default HuggingFace model ID
    public static let defaultModelId = "aufklarer/WeSpeaker-ResNet34-LM-MLX"

    /// Embedding dimension
    public let embeddingDimension: Int = 256

    /// Expected input sample rate
    public let inputSampleRate: Int = 16000

    init(network: WeSpeakerNetwork) {
        self.network = network
        self.melExtractor = MelFeatureExtractor()
    }

    /// Load a pre-trained speaker embedding model from HuggingFace.
    ///
    /// Downloads model weights on first use, then caches locally.
    ///
    /// - Parameters:
    ///   - modelId: HuggingFace model ID
    ///   - progressHandler: callback for download progress
    /// - Returns: ready-to-use speaker embedding model
    public static func fromPretrained(
        modelId: String = defaultModelId,
        progressHandler: ((Double, String) -> Void)? = nil
    ) async throws -> WeSpeakerModel {
        progressHandler?(0.0, "Downloading speaker embedding model...")

        let cacheDir = try HuggingFaceDownloader.getCacheDirectory(for: modelId)

        try await HuggingFaceDownloader.downloadWeights(
            modelId: modelId,
            to: cacheDir,
            progressHandler: { progress in
                progressHandler?(progress * 0.8, "Downloading weights...")
            }
        )

        progressHandler?(0.8, "Loading model...")

        let network = WeSpeakerNetwork()
        try WeSpeakerWeightLoader.loadWeights(model: network, from: cacheDir)

        progressHandler?(1.0, "Ready")

        return WeSpeakerModel(network: network)
    }

    /// Extract a 256-dimensional speaker embedding from audio.
    ///
    /// - Parameters:
    ///   - audio: PCM Float32 audio samples
    ///   - sampleRate: sample rate of the input audio
    /// - Returns: 256-dim L2-normalized speaker embedding
    public func embed(audio: [Float], sampleRate: Int) -> [Float] {
        let samples: [Float]
        if sampleRate != inputSampleRate {
            samples = resample(audio, from: sampleRate, to: inputSampleRate)
        } else {
            samples = audio
        }

        // Extract mel features: [T, 80]
        let mel = melExtractor.extract(samples)

        // Add batch and channel dims: [1, T, 80, 1]
        let input = mel.reshaped(1, mel.dim(0), mel.dim(1), 1)

        // Forward pass: [1, 256]
        let emb = network(input)
        eval(emb)

        return emb[0].asArray(Float.self)
    }

    /// Compute cosine similarity between two embeddings.
    public static func cosineSimilarity(_ a: [Float], _ b: [Float]) -> Float {
        guard a.count == b.count, !a.isEmpty else { return 0 }
        var dot: Float = 0
        var normA: Float = 0
        var normB: Float = 0
        for i in 0..<a.count {
            dot += a[i] * b[i]
            normA += a[i] * a[i]
            normB += b[i] * b[i]
        }
        let denom = sqrt(normA) * sqrt(normB)
        return denom > 0 ? dot / denom : 0
    }

    private func resample(_ audio: [Float], from sourceSR: Int, to targetSR: Int) -> [Float] {
        guard sourceSR != targetSR else { return audio }
        let ratio = Double(targetSR) / Double(sourceSR)
        let outputLen = Int(Double(audio.count) * ratio)
        var output = [Float](repeating: 0, count: outputLen)

        for i in 0..<outputLen {
            let srcPos = Double(i) / ratio
            let srcIdx = Int(srcPos)
            let frac = Float(srcPos - Double(srcIdx))

            if srcIdx + 1 < audio.count {
                output[i] = audio[srcIdx] * (1 - frac) + audio[srcIdx + 1] * frac
            } else if srcIdx < audio.count {
                output[i] = audio[srcIdx]
            }
        }

        return output
    }
}
