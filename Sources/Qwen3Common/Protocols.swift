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

    public init(
        samples: [Float],
        sampleRate: Int,
        frameIndex: Int,
        isFinal: Bool,
        elapsedTime: Double? = nil
    ) {
        self.samples = samples
        self.sampleRate = sampleRate
        self.frameIndex = frameIndex
        self.isFinal = isFinal
        self.elapsedTime = elapsedTime
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
