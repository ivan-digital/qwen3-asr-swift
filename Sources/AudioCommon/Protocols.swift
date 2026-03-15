import Foundation

// MARK: - Model Memory Management

/// Memory statistics for a loaded model.
public struct ModelMemoryStats: Sendable {
    /// Estimated weight memory in bytes
    public let weightMemory: Int
    /// Current active GPU memory in bytes (MLX only)
    public let activeMemory: Int

    public init(weightMemory: Int, activeMemory: Int = 0) {
        self.weightMemory = weightMemory
        self.activeMemory = activeMemory
    }
}

/// A model that supports explicit memory management.
///
/// Call `unload()` to release model weights and free GPU memory.
/// After unloading, the model cannot be used for inference until re-loaded.
public protocol ModelMemoryManageable: AnyObject {
    /// Whether the model is currently loaded and ready for inference.
    var isLoaded: Bool { get }

    /// Release model weights and free GPU memory.
    ///
    /// After calling this, `isLoaded` returns false and inference methods will fail.
    /// To use the model again, create a new instance via `fromPretrained()`.
    func unload()

    /// Estimated memory footprint of the loaded model weights in bytes.
    /// Returns 0 if the model is not loaded.
    var memoryFootprint: Int { get }
}

// MARK: - Unified Audio Chunk

/// A chunk of audio produced during streaming synthesis or generation.
public struct AudioChunk: Sendable {
    /// PCM audio samples (Float32)
    public let samples: [Float]
    /// Sample rate in Hz (e.g. 24000)
    public let sampleRate: Int
    /// Index of the first frame in this chunk
    public let frameIndex: Int
    /// True if this is the last chunk
    public let isFinal: Bool
    /// Wall-clock seconds since generation started (nil if not tracked)
    public let elapsedTime: Double?
    /// Text tokens generated alongside audio (populated on final chunk if available)
    public let textTokens: [Int32]

    public init(
        samples: [Float],
        sampleRate: Int,
        frameIndex: Int,
        isFinal: Bool,
        elapsedTime: Double? = nil,
        textTokens: [Int32] = []
    ) {
        self.samples = samples
        self.sampleRate = sampleRate
        self.frameIndex = frameIndex
        self.isFinal = isFinal
        self.elapsedTime = elapsedTime
        self.textTokens = textTokens
    }
}

// MARK: - Aligned Word

/// A word with its aligned start and end timestamps (in seconds).
public struct AlignedWord: Sendable {
    public let text: String
    public let startTime: Float
    public let endTime: Float

    public init(text: String, startTime: Float, endTime: Float) {
        self.text = text
        self.startTime = startTime
        self.endTime = endTime
    }
}

// MARK: - Speech Generation (TTS)

/// A text-to-speech model that generates audio from text.
public protocol SpeechGenerationModel: AnyObject {
    /// Output sample rate in Hz
    var sampleRate: Int { get }
    /// Synthesize audio from text (returns full waveform)
    func generate(text: String, language: String?) async throws -> [Float]
    /// Synthesize audio from text with streaming output.
    /// Default implementation wraps `generate()` as a single chunk.
    func generateStream(text: String, language: String?) -> AsyncThrowingStream<AudioChunk, Error>
}

extension SpeechGenerationModel {
    /// Default: wraps `generate()` as a single-chunk stream.
    public func generateStream(text: String, language: String?) -> AsyncThrowingStream<AudioChunk, Error> {
        let rate = sampleRate
        return AsyncThrowingStream { continuation in
            Task {
                do {
                    let samples = try await self.generate(text: text, language: language)
                    continuation.yield(AudioChunk(samples: samples, sampleRate: rate, frameIndex: 0, isFinal: true))
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }
}

// MARK: - Speech Recognition (STT)

/// Result of speech recognition including detected language.
public struct TranscriptionResult: Sendable {
    public let text: String
    /// Detected language (e.g. "english", "russian"). Nil if model doesn't detect.
    public let language: String?

    public init(text: String, language: String? = nil) {
        self.text = text
        self.language = language
    }
}

/// A speech-to-text model that transcribes audio.
public protocol SpeechRecognitionModel: AnyObject {
    /// Expected input sample rate in Hz
    var inputSampleRate: Int { get }
    /// Transcribe audio to text
    func transcribe(audio: [Float], sampleRate: Int, language: String?) -> String
    /// Transcribe audio to text with language detection
    func transcribeWithLanguage(audio: [Float], sampleRate: Int, language: String?) -> TranscriptionResult
}

/// Default implementation: delegates to transcribe() with no language detection.
public extension SpeechRecognitionModel {
    func transcribeWithLanguage(audio: [Float], sampleRate: Int, language: String?) -> TranscriptionResult {
        TranscriptionResult(text: transcribe(audio: audio, sampleRate: sampleRate, language: language))
    }
}

// MARK: - Forced Alignment

/// A model that aligns text to audio at the word level.
public protocol ForcedAlignmentModel: AnyObject {
    /// Align text to audio, returning word-level timestamps
    func align(audio: [Float], text: String, sampleRate: Int, language: String?) -> [AlignedWord]
}

// MARK: - Speech-to-Speech

/// A speech-to-speech model that generates a spoken response to spoken input.
public protocol SpeechToSpeechModel: AnyObject {
    /// Output sample rate in Hz
    var sampleRate: Int { get }
    /// Generate response audio from input audio (blocking)
    func respond(userAudio: [Float]) -> [Float]
    /// Generate response audio from input audio with streaming output
    func respondStream(userAudio: [Float]) -> AsyncThrowingStream<AudioChunk, Error>
}

// MARK: - Voice Activity Detection

/// A time segment where speech was detected.
public struct SpeechSegment: Sendable {
    /// Start time in seconds
    public let startTime: Float
    /// End time in seconds
    public let endTime: Float

    public init(startTime: Float, endTime: Float) {
        self.startTime = startTime
        self.endTime = endTime
    }

    /// Duration in seconds
    public var duration: Float { endTime - startTime }
}

/// A model that detects speech activity regions in audio.
public protocol VoiceActivityDetectionModel: AnyObject {
    /// Expected input sample rate in Hz
    var inputSampleRate: Int { get }
    /// Detect speech segments in audio
    func detectSpeech(audio: [Float], sampleRate: Int) -> [SpeechSegment]
}

/// A streaming VAD that processes fixed-size audio chunks and returns speech probability.
///
/// Maps directly to speech-core's `sc_vad_vtable_t` for pipeline integration.
public protocol StreamingVADProvider: AnyObject {
    /// Expected input sample rate in Hz
    var inputSampleRate: Int { get }
    /// Number of samples per chunk
    var chunkSize: Int { get }
    /// Process a single audio chunk, returns speech probability in [0, 1]
    func processChunk(_ samples: [Float]) -> Float
    /// Reset internal state (LSTM hidden state, context buffer, etc.)
    func resetState()
}

// MARK: - Speaker Diarization

/// A speech segment with an assigned speaker identity.
public struct DiarizedSegment: Sendable {
    /// Start time in seconds
    public let startTime: Float
    /// End time in seconds
    public let endTime: Float
    /// Speaker identifier (0-based)
    public let speakerId: Int

    public init(startTime: Float, endTime: Float, speakerId: Int) {
        self.startTime = startTime
        self.endTime = endTime
        self.speakerId = speakerId
    }

    /// Duration in seconds
    public var duration: Float { endTime - startTime }
}

/// A model that produces speaker embeddings from audio.
public protocol SpeakerEmbeddingModel: AnyObject {
    /// Expected input sample rate in Hz
    var inputSampleRate: Int { get }
    /// Embedding vector dimension
    var embeddingDimension: Int { get }
    /// Extract a speaker embedding from audio
    func embed(audio: [Float], sampleRate: Int) -> [Float]
}

// MARK: - Speech Enhancement

/// A model that enhances speech by removing noise.
public protocol SpeechEnhancementModel: AnyObject {
    /// Expected input sample rate in Hz
    var inputSampleRate: Int { get }
    /// Enhance audio by removing noise
    func enhance(audio: [Float], sampleRate: Int) throws -> [Float]
}

/// A model that assigns speaker identities to speech segments.
public protocol SpeakerDiarizationModel: AnyObject {
    /// Expected input sample rate in Hz
    var inputSampleRate: Int { get }
    /// Diarize audio into speaker-labeled segments
    func diarize(audio: [Float], sampleRate: Int) -> [DiarizedSegment]
}

/// A diarization model that also supports extracting a specific speaker's segments
/// using a reference embedding. Not all engines support this (e.g. Sortformer is
/// end-to-end and does not produce speaker embeddings).
public protocol SpeakerExtractionCapable: SpeakerDiarizationModel {
    /// Extract segments belonging to a target speaker identified by a reference embedding.
    func extractSpeaker(audio: [Float], sampleRate: Int, targetEmbedding: [Float]) -> [SpeechSegment]
}
