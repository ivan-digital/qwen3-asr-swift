import Foundation
import MLX
import AudioCommon

/// Streaming Voice Activity Detection using Silero VAD v5.
///
/// A lightweight (~260K params) VAD model that processes 512-sample chunks
/// (32ms @ 16kHz) with sub-millisecond latency. Carries LSTM state across
/// chunks for streaming operation.
///
/// - Warning: This class is not thread-safe. Create separate instances for concurrent use.
///
/// ```swift
/// let vad = try await SileroVADModel.fromPretrained()
///
/// // Streaming: process one chunk at a time
/// let prob = vad.processChunk(samples512)  // → 0.0...1.0
///
/// // Batch: detect all speech segments
/// let segments = vad.detectSpeech(audio: samples, sampleRate: 16000)
/// ```
public final class SileroVADModel {

    /// The neural network
    let network: SileroVADNetwork

    /// LSTM hidden state (carried across chunks)
    private var h: MLXArray?
    /// LSTM cell state (carried across chunks)
    private var c: MLXArray?
    /// Context buffer: last 64 samples from previous chunk
    private var context: [Float]

    /// Default HuggingFace model ID
    public static let defaultModelId = "aufklarer/Silero-VAD-v5-MLX"

    /// Number of audio samples per chunk (32ms @ 16kHz)
    public static let chunkSize = 512

    /// Number of context samples prepended from previous chunk
    public static let contextSize = 64

    /// Expected input sample rate
    public static let sampleRate = 16000

    init(network: SileroVADNetwork) {
        self.network = network
        self.context = [Float](repeating: 0, count: Self.contextSize)
    }

    /// Process a single 512-sample audio chunk and return speech probability.
    ///
    /// Maintains internal LSTM state across calls. Call `resetState()` between
    /// different audio streams.
    ///
    /// - Parameter samples: exactly 512 PCM Float32 samples at 16kHz
    /// - Returns: speech probability in `[0, 1]`
    public func processChunk(_ samples: [Float]) -> Float {
        precondition(samples.count == Self.chunkSize,
                     "Chunk must be \(Self.chunkSize) samples, got \(samples.count)")

        // Prepend 64-sample context from previous chunk
        let fullSamples = context + samples

        // Save last 64 samples as context for next chunk
        context = Array(samples.suffix(Self.contextSize))

        // Run network: [1, 576] → probability
        let input = MLXArray(fullSamples).reshaped(1, fullSamples.count)
        let (prob, newH, newC) = network.forward(input, h: h, c: c)

        h = newH
        c = newC

        eval(prob)
        return prob.item(Float.self)
    }

    /// Reset LSTM and context state.
    ///
    /// Call this between processing different audio streams to prevent
    /// state leakage.
    public func resetState() {
        h = nil
        c = nil
        context = [Float](repeating: 0, count: Self.contextSize)
    }

    /// Detect speech segments in complete audio (batch mode).
    ///
    /// Processes the entire audio in 512-sample chunks, collects per-chunk
    /// probabilities, then applies hysteresis thresholding and duration filtering.
    ///
    /// - Parameters:
    ///   - audio: PCM Float32 audio samples
    ///   - sampleRate: sample rate of input audio (resampled to 16kHz if needed)
    ///   - config: VAD configuration (defaults to Silero-tuned thresholds)
    /// - Returns: array of speech segments with start/end times in seconds
    public func detectSpeech(
        audio: [Float],
        sampleRate: Int,
        config: VADConfig = .sileroDefault
    ) -> [SpeechSegment] {
        let samples: [Float]
        if sampleRate != Self.sampleRate {
            samples = resample(audio, from: sampleRate, to: Self.sampleRate)
        } else {
            samples = audio
        }

        resetState()

        // Collect per-chunk probabilities
        var probs = [Float]()
        var offset = 0

        while offset + Self.chunkSize <= samples.count {
            let chunk = Array(samples[offset ..< (offset + Self.chunkSize)])
            probs.append(processChunk(chunk))
            offset += Self.chunkSize
        }

        // Handle remaining samples with zero-padding
        if offset < samples.count {
            var lastChunk = Array(samples[offset...])
            lastChunk.append(contentsOf: [Float](repeating: 0, count: Self.chunkSize - lastChunk.count))
            probs.append(processChunk(lastChunk))
        }

        guard !probs.isEmpty else { return [] }

        // Use VADPipeline for hysteresis binarization.
        // Set windowDuration = probs.count * chunkDuration so frameDuration = 0.032s
        let chunkDuration: Float = Float(Self.chunkSize) / Float(Self.sampleRate)
        let batchPipeline = VADPipeline(
            config: VADConfig(
                onset: config.onset,
                offset: config.offset,
                minSpeechDuration: config.minSpeechDuration,
                minSilenceDuration: config.minSilenceDuration,
                windowDuration: Float(probs.count) * chunkDuration,
                stepRatio: 1.0
            ),
            sampleRate: Self.sampleRate,
            framesPerChunk: probs.count
        )

        return batchPipeline.binarize(probs: probs)
    }

    /// Load a pre-trained Silero VAD model from HuggingFace.
    ///
    /// Downloads model weights on first use, then caches locally at
    /// `~/Library/Caches/qwen3-speech/aufklarer_Silero-VAD-v5-MLX/`.
    ///
    /// - Parameters:
    ///   - modelId: HuggingFace model ID
    ///   - progressHandler: callback for download progress
    /// - Returns: ready-to-use VAD model
    public static func fromPretrained(
        modelId: String = defaultModelId,
        progressHandler: ((Double, String) -> Void)? = nil
    ) async throws -> SileroVADModel {
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

        let network = SileroVADNetwork()
        try SileroWeightLoader.loadWeights(model: network, from: cacheDir)

        progressHandler?(1.0, "Ready")

        return SileroVADModel(network: network)
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

// MARK: - VoiceActivityDetectionModel

extension SileroVADModel: VoiceActivityDetectionModel {
    public var inputSampleRate: Int { Self.sampleRate }

    /// Protocol conformance — uses default Silero config.
    public func detectSpeech(audio: [Float], sampleRate: Int) -> [SpeechSegment] {
        detectSpeech(audio: audio, sampleRate: sampleRate, config: .sileroDefault)
    }
}
