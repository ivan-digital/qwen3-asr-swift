import Foundation

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
    /// Synthesize audio from text (blocking, returns full waveform)
    func generate(text: String, language: String?) async throws -> [Float]
    /// Synthesize audio from text with streaming output
    func generateStream(text: String, language: String?) -> AsyncThrowingStream<AudioChunk, Error>
}

// MARK: - Speech Recognition (STT)

/// A speech-to-text model that transcribes audio.
public protocol SpeechRecognitionModel: AnyObject {
    /// Expected input sample rate in Hz
    var inputSampleRate: Int { get }
    /// Transcribe audio to text
    func transcribe(audio: [Float], sampleRate: Int, language: String?) -> String
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

/// A model that assigns speaker identities to speech segments.
public protocol SpeakerDiarizationModel: AnyObject {
    /// Expected input sample rate in Hz
    var inputSampleRate: Int { get }
    /// Diarize audio into speaker-labeled segments
    func diarize(audio: [Float], sampleRate: Int) -> [DiarizedSegment]
}
