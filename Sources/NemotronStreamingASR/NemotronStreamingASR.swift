import CoreML
import Foundation
import AudioCommon

/// Nemotron Speech Streaming 0.6B — low-latency English streaming ASR on CoreML.
///
/// Cache-aware FastConformer encoder + RNN-T decoder. 600M parameters (INT8
/// palettized encoder). Native punctuation and capitalization emitted as
/// regular BPE tokens — no EOU/EOB heads; caller signals end of stream via
/// `finalize()`.
///
/// - Warning: This class is not thread-safe. Create separate instances for concurrent use.
public class NemotronStreamingASRModel {
    public let config: NemotronStreamingConfig

    public static let defaultModelId = "aufklarer/Nemotron-Speech-Streaming-0.6B-CoreML-INT8"

    var _isLoaded = true
    private let melPreprocessor: StreamingMelPreprocessor
    var encoder: MLModel?
    var decoder: MLModel?
    var joint: MLModel?
    private let vocabulary: NemotronVocabulary

    private init(
        config: NemotronStreamingConfig,
        encoder: MLModel?,
        decoder: MLModel?,
        joint: MLModel?,
        vocabulary: NemotronVocabulary
    ) {
        self.config = config
        self.melPreprocessor = StreamingMelPreprocessor(config: config)
        self.encoder = encoder
        self.decoder = decoder
        self.joint = joint
        self.vocabulary = vocabulary
    }

    /// A partial transcript from streaming recognition.
    public struct PartialTranscript: Sendable {
        public let text: String
        public let isFinal: Bool
        public let confidence: Float
        public let segmentIndex: Int
    }

    /// Create a new streaming session.
    public func createSession() throws -> StreamingSession {
        guard _isLoaded, let encoder, let decoder, let joint else {
            throw AudioModelError.inferenceFailed(operation: "createSession", reason: "Model not loaded")
        }
        return try StreamingSession(
            config: config,
            encoder: encoder,
            decoder: decoder,
            joint: joint,
            vocabulary: vocabulary,
            melPreprocessor: melPreprocessor
        )
    }

    /// Convenience: stream transcription from a buffer, yielding partial results.
    public func transcribeStream(
        audio: [Float],
        sampleRate: Int,
        chunkDuration: Float? = nil
    ) -> AsyncStream<PartialTranscript> {
        let chunkMs = chunkDuration.map { Int($0 * 1000) } ?? config.streaming.chunkMs

        return AsyncStream { continuation in
            Task {
                do {
                    let samples: [Float]
                    if sampleRate != self.config.sampleRate {
                        samples = AudioFileLoader.resample(audio, from: sampleRate, to: self.config.sampleRate)
                    } else {
                        samples = audio
                    }
                    let actualSamplesPerChunk = chunkMs * self.config.sampleRate / 1000
                    let session = try self.createSession()
                    var offset = 0
                    while offset < samples.count {
                        let end = min(offset + actualSamplesPerChunk, samples.count)
                        let chunk = Array(samples[offset..<end])
                        let partials = try session.pushAudio(chunk)
                        for partial in partials { continuation.yield(partial) }
                        offset = end
                    }
                    let finals = try session.finalize()
                    for partial in finals { continuation.yield(partial) }
                    continuation.finish()
                } catch {
                    AudioLog.inference.error("Nemotron streaming transcription failed: \(error)")
                    continuation.finish()
                }
            }
        }
    }

    /// Transcribe a full audio buffer (non-streaming fallback).
    ///
    /// The streaming encoder starts each session with an all-zero attention /
    /// convolution cache and an all-zero mel pre-cache. For natural speech with
    /// ambient lead-in that's fine, but short TTS-generated audio with sharp
    /// onset and offset can lose the first and last words at chunk boundaries.
    /// Padding 0.1 s of silence at both ends primes the cache with a single
    /// encoder output frame of "silent" context — enough to eliminate edge
    /// clipping on TTS input while staying short enough not to perturb chunk
    /// alignment on natural audio. Empirically recovers full word accuracy on
    /// Kokoro-generated test phrases without degrading natural-speech tests.
    public func transcribeAudio(_ audio: [Float], sampleRate: Int, language: String? = nil) throws -> String {
        var samples: [Float]
        if sampleRate != config.sampleRate {
            samples = AudioFileLoader.resample(audio, from: sampleRate, to: config.sampleRate)
        } else {
            samples = audio
        }
        let padSamples = config.sampleRate / 10  // 100 ms
        samples = [Float](repeating: 0, count: padSamples) + samples + [Float](repeating: 0, count: padSamples)

        let session = try createSession()
        var allPartials = try session.pushAudio(samples)
        allPartials.append(contentsOf: try session.finalize())
        if let lastFinal = allPartials.last(where: { $0.isFinal }) {
            return lastFinal.text
        }
        return allPartials.last?.text ?? ""
    }

    /// Warm up CoreML models on a dummy input.
    public func warmUp() throws {
        let dummy = [Float](repeating: 0, count: config.sampleRate)
        _ = try transcribeAudio(dummy, sampleRate: config.sampleRate)
    }

    // MARK: - Model Loading

    public static func fromPretrained(
        modelId: String? = nil,
        progressHandler: ((Double, String) -> Void)? = nil
    ) async throws -> NemotronStreamingASRModel {
        let effectiveModelId = modelId ?? defaultModelId
        AudioLog.modelLoading.info("Loading Nemotron Streaming model: \(effectiveModelId)")

        let cacheDir: URL
        do {
            cacheDir = try HuggingFaceDownloader.getCacheDirectory(for: effectiveModelId)
        } catch {
            throw AudioModelError.modelLoadFailed(
                modelId: effectiveModelId, reason: "Failed to resolve cache directory", underlying: error)
        }

        progressHandler?(0.0, "Downloading model...")
        do {
            try await HuggingFaceDownloader.downloadWeights(
                modelId: effectiveModelId,
                to: cacheDir,
                additionalFiles: [
                    "encoder.mlmodelc/**",
                    "decoder.mlmodelc/**",
                    "joint.mlmodelc/**",
                    "vocab.json",
                    "config.json",
                ]
            ) { fraction in
                progressHandler?(fraction * 0.7, "Downloading model...")
            }
        } catch {
            throw AudioModelError.modelLoadFailed(
                modelId: effectiveModelId, reason: "Download failed", underlying: error)
        }

        progressHandler?(0.70, "Loading configuration...")
        let config: NemotronStreamingConfig
        let configURL = cacheDir.appendingPathComponent("config.json")
        if FileManager.default.fileExists(atPath: configURL.path) {
            let data = try Data(contentsOf: configURL)
            config = try JSONDecoder().decode(NemotronStreamingConfig.self, from: data)
        } else {
            config = .default
        }

        progressHandler?(0.75, "Loading vocabulary...")
        let vocabURL = cacheDir.appendingPathComponent("vocab.json")
        let vocabulary = try NemotronVocabulary.load(from: vocabURL)

        progressHandler?(0.80, "Loading CoreML models...")
        let encoder = try loadCoreMLModel(name: "encoder", from: cacheDir, computeUnits: .cpuAndGPU)
        progressHandler?(0.90, "Loading decoder...")
        let decoder = try loadCoreMLModel(name: "decoder", from: cacheDir, computeUnits: .cpuAndGPU)
        progressHandler?(0.95, "Loading joint network...")
        let joint = try loadCoreMLModel(name: "joint", from: cacheDir, computeUnits: .cpuAndGPU)

        progressHandler?(1.0, "Model loaded")
        AudioLog.modelLoading.info("Nemotron Streaming model loaded (\(vocabulary.count) tokens)")

        return NemotronStreamingASRModel(
            config: config,
            encoder: encoder,
            decoder: decoder,
            joint: joint,
            vocabulary: vocabulary
        )
    }

    private static func loadCoreMLModel(
        name: String,
        from directory: URL,
        computeUnits: MLComputeUnits
    ) throws -> MLModel {
        let modelURL = directory.appendingPathComponent("\(name).mlmodelc", isDirectory: true)
        guard FileManager.default.fileExists(atPath: modelURL.path) else {
            throw AudioModelError.modelLoadFailed(
                modelId: name, reason: "CoreML model not found at \(modelURL.path)")
        }
        let mlConfig = MLModelConfiguration()
        mlConfig.computeUnits = computeUnits
        return try MLModel(contentsOf: modelURL, configuration: mlConfig)
    }
}
