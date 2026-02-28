import Foundation
import MLX
import AudioCommon

#if canImport(CoreML)
import CoreML
#endif

/// Inference engine for Silero VAD.
public enum SileroVADEngine: String, Sendable {
    /// MLX backend — runs on GPU via Metal shaders.
    case mlx
    /// CoreML backend — runs on Neural Engine + CPU, freeing the GPU.
    case coreml
}

/// Streaming Voice Activity Detection using Silero VAD v5.
///
/// A lightweight (~260K params) VAD model that processes 512-sample chunks
/// (32ms @ 16kHz) with sub-millisecond latency. Carries LSTM state across
/// chunks for streaming operation.
///
/// Supports two backends:
/// - `.mlx`: GPU-based inference via MLX (default)
/// - `.coreml`: Neural Engine inference via CoreML (lower power, frees GPU)
///
/// - Warning: This class is not thread-safe. Create separate instances for concurrent use.
///
/// ```swift
/// let vad = try await SileroVADModel.fromPretrained(engine: .coreml)
///
/// // Streaming: process one chunk at a time
/// let prob = vad.processChunk(samples512)  // → 0.0...1.0
///
/// // Batch: detect all speech segments
/// let segments = vad.detectSpeech(audio: samples, sampleRate: 16000)
/// ```
public final class SileroVADModel {

    /// The inference engine in use.
    public let engine: SileroVADEngine

    /// The MLX neural network (nil when using CoreML engine).
    let network: SileroVADNetwork?

    // MARK: - MLX State

    /// LSTM hidden state (carried across chunks) — MLX engine
    private var h: MLXArray?
    /// LSTM cell state (carried across chunks) — MLX engine
    private var c: MLXArray?

    // MARK: - CoreML State

    #if canImport(CoreML)
    /// CoreML compiled model (nil when using MLX engine).
    var coremlModel: MLModel?
    /// CoreML LSTM hidden state
    var coremlH: MLMultiArray?
    /// CoreML LSTM cell state
    var coremlC: MLMultiArray?
    #endif

    /// Context buffer: last 64 samples from previous chunk
    private var context: [Float]

    /// Default HuggingFace model ID (MLX weights)
    public static let defaultModelId = "aufklarer/Silero-VAD-v5-MLX"

    /// Default HuggingFace model ID (CoreML weights)
    public static let defaultCoreMLModelId = "aufklarer/Silero-VAD-v5-CoreML"

    /// Number of audio samples per chunk (32ms @ 16kHz)
    public static let chunkSize = 512

    /// Number of context samples prepended from previous chunk
    public static let contextSize = 64

    /// Expected input sample rate
    public static let sampleRate = 16000

    init(network: SileroVADNetwork) {
        self.engine = .mlx
        self.network = network
        self.context = [Float](repeating: 0, count: Self.contextSize)
    }

    #if canImport(CoreML)
    init(coremlModel: MLModel) {
        self.engine = .coreml
        self.network = nil
        self.coremlModel = coremlModel
        self.context = [Float](repeating: 0, count: Self.contextSize)
    }
    #endif

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

        switch engine {
        case .mlx:
            return processChunkMLX(fullSamples)
        case .coreml:
            #if canImport(CoreML)
            return (try? processChunkCoreML(fullSamples)) ?? 0.0
            #else
            fatalError("CoreML not available on this platform")
            #endif
        }
    }

    /// MLX inference path.
    private func processChunkMLX(_ fullSamples: [Float]) -> Float {
        guard let network else { fatalError("MLX network not loaded") }

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
        #if canImport(CoreML)
        coremlH = nil
        coremlC = nil
        #endif
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
    /// Downloads model weights on first use, then caches locally.
    ///
    /// - Parameters:
    ///   - modelId: HuggingFace model ID (auto-selected by engine if not specified)
    ///   - engine: inference backend (`.mlx` or `.coreml`)
    ///   - progressHandler: callback for download progress
    /// - Returns: ready-to-use VAD model
    public static func fromPretrained(
        modelId: String? = nil,
        engine: SileroVADEngine = .mlx,
        progressHandler: ((Double, String) -> Void)? = nil
    ) async throws -> SileroVADModel {
        let resolvedModelId = modelId ?? (engine == .coreml ? defaultCoreMLModelId : defaultModelId)

        progressHandler?(0.0, "Downloading model...")

        let cacheDir = try HuggingFaceDownloader.getCacheDirectory(for: resolvedModelId)

        switch engine {
        case .mlx:
            try await HuggingFaceDownloader.downloadWeights(
                modelId: resolvedModelId,
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

        case .coreml:
            #if canImport(CoreML)
            try await HuggingFaceDownloader.downloadWeights(
                modelId: resolvedModelId,
                to: cacheDir,
                additionalFiles: ["silero_vad.mlmodelc/**", "config.json"],
                progressHandler: { progress in
                    progressHandler?(progress * 0.8, "Downloading CoreML model...")
                }
            )

            progressHandler?(0.8, "Loading CoreML model...")

            let modelURL = cacheDir.appendingPathComponent("silero_vad.mlmodelc", isDirectory: true)
            guard FileManager.default.fileExists(atPath: modelURL.path) else {
                throw AudioModelError.modelLoadFailed(
                    modelId: resolvedModelId,
                    reason: "CoreML model not found at \(modelURL.path)")
            }

            let mlConfig = MLModelConfiguration()
            mlConfig.computeUnits = .cpuAndNeuralEngine

            let model: MLModel
            do {
                model = try MLModel(contentsOf: modelURL, configuration: mlConfig)
            } catch {
                throw AudioModelError.modelLoadFailed(
                    modelId: resolvedModelId,
                    reason: "Failed to load CoreML model",
                    underlying: error)
            }

            progressHandler?(1.0, "Ready")
            return SileroVADModel(coremlModel: model)
            #else
            throw AudioModelError.invalidConfiguration(
                model: "SileroVAD", reason: "CoreML not available on this platform")
            #endif
        }
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
