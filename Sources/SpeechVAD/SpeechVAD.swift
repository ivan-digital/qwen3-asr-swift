import Foundation
import MLX
import AudioCommon

/// Voice Activity Detection using pyannote PyanNet segmentation.
///
/// Detects speech regions in audio using a sliding-window segmentation model
/// with hysteresis thresholding and duration filtering.
///
/// - Warning: This class is not thread-safe. Create separate instances for concurrent use.
///
/// ```swift
/// let vad = try await PyannoteVADModel.fromPretrained()
/// let segments = vad.detectSpeech(audio: samples, sampleRate: 16000)
/// for seg in segments {
///     print("Speech: \(seg.startTime)s - \(seg.endTime)s")
/// }
/// ```
public final class PyannoteVADModel {
    /// The segmentation model
    let model: SegmentationModel

    /// VAD pipeline configuration
    public let vadConfig: VADConfig

    /// Segmentation model configuration
    public let segConfig: SegmentationConfig

    /// Default HuggingFace model ID
    public static let defaultModelId = "aufklarer/Pyannote-Segmentation-MLX"

    init(model: SegmentationModel, segConfig: SegmentationConfig, vadConfig: VADConfig) {
        self.model = model
        self.segConfig = segConfig
        self.vadConfig = vadConfig
    }

    /// Load a pre-trained VAD model from HuggingFace.
    ///
    /// Downloads model weights on first use, then caches locally.
    ///
    /// - Parameters:
    ///   - modelId: HuggingFace model ID
    ///   - vadConfig: VAD pipeline configuration (thresholds, durations)
    ///   - progressHandler: callback for download progress
    /// - Returns: ready-to-use VAD model
    public static func fromPretrained(
        modelId: String = defaultModelId,
        vadConfig: VADConfig = .default,
        progressHandler: ((Double, String) -> Void)? = nil
    ) async throws -> PyannoteVADModel {
        progressHandler?(0.0, "Downloading model...")

        let cacheDir = try HuggingFaceDownloader.getCacheDirectory(for: modelId)

        try await HuggingFaceDownloader.downloadWeights(
            modelId: modelId,
            to: cacheDir,
            progressHandler: { progress in
                progressHandler?(progress * 0.8, "Downloading weights...")
            }
        )

        progressHandler?(0.8, "Loading model...")

        let segConfig = SegmentationConfig.default
        let model = SegmentationModel(config: segConfig)

        try SegmentationWeightLoader.loadWeights(model: model, from: cacheDir)

        progressHandler?(1.0, "Ready")

        return PyannoteVADModel(model: model, segConfig: segConfig, vadConfig: vadConfig)
    }

    /// Detect speech segments in audio.
    ///
    /// - Parameters:
    ///   - audio: PCM Float32 audio samples
    ///   - sampleRate: sample rate of the input audio (will resample to 16kHz if needed)
    /// - Returns: array of speech segments with start/end times in seconds
    public func detectSpeech(audio: [Float], sampleRate: Int) -> [SpeechSegment] {
        let samples: [Float]
        if sampleRate != segConfig.sampleRate {
            samples = resample(audio, from: sampleRate, to: segConfig.sampleRate)
        } else {
            samples = audio
        }

        let pipeline = VADPipeline(
            config: vadConfig,
            sampleRate: segConfig.sampleRate,
            framesPerChunk: 589
        )

        let positions = pipeline.windowPositions(numSamples: samples.count)

        guard !positions.isEmpty else { return [] }

        let windowSamples = Int(vadConfig.windowDuration * Float(segConfig.sampleRate))

        // Run segmentation on each window
        var windowProbs = [[Float]]()

        for (start, end) in positions {
            // Extract window, zero-pad if needed
            var window = Array(samples[start ..< end])
            if window.count < windowSamples {
                window.append(contentsOf: [Float](repeating: 0, count: windowSamples - window.count))
            }

            // Run model: [1, 1, samples] → [1, frames, 7]
            let input = MLXArray(window).reshaped(1, 1, windowSamples)
            let posteriors = model(input)

            // Extract speech probability: [1, frames] → [frames]
            let speechProb = SegmentationModel.speechProbability(from: posteriors)
            eval(speechProb)

            let probArray = speechProb[0].asArray(Float.self)
            windowProbs.append(probArray)
        }

        // Aggregate overlapping windows
        let aggregated = pipeline.aggregateFrames(
            windowProbs: windowProbs,
            positions: positions,
            numSamples: samples.count
        )

        // Binarize with hysteresis
        return pipeline.binarize(probs: aggregated)
    }

    /// Simple linear resampling.
    private func resample(_ audio: [Float], from sourceSR: Int, to targetSR: Int) -> [Float] {
        guard sourceSR != targetSR else { return audio }
        let ratio = Double(targetSR) / Double(sourceSR)
        let outputLen = Int(Double(audio.count) * ratio)
        var output = [Float](repeating: 0, count: outputLen)

        for i in 0 ..< outputLen {
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
