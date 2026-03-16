import Foundation

/// Configuration for the Sortformer diarization model.
///
/// Sortformer is NVIDIA's end-to-end neural diarization model that directly
/// predicts speaker activity without requiring separate embedding extraction
/// or clustering stages.
public struct SortformerConfig: Sendable {

    // MARK: - Mel Feature Extraction

    /// Number of mel frequency bins
    public let nMels: Int
    /// FFT window size in samples
    public let nFFT: Int
    /// Hop length in samples
    public let hopLength: Int
    /// Expected input sample rate in Hz
    public let sampleRate: Int

    // MARK: - Streaming Chunking

    /// Chunk length in seconds for streaming inference
    public let chunkLenSeconds: Float
    /// Left context in seconds (prepended from previous chunk)
    public let leftContextSeconds: Float
    /// Right context in seconds (lookahead)
    public let rightContextSeconds: Float
    /// Subsampling factor of the encoder (frames → mel frames)
    public let subsamplingFactor: Int

    // MARK: - State Dimensions

    /// Speaker cache length (number of frames)
    public let spkcacheLen: Int
    /// FIFO buffer length (number of frames)
    public let fifoLen: Int
    /// Feature/hidden dimension of the model
    public let fcDModel: Int

    // MARK: - Model I/O Shapes

    /// Maximum number of speakers the model can predict
    public let maxSpeakers: Int

    // MARK: - Post-processing

    /// Onset threshold for speaker activity binarization
    public var onset: Float
    /// Offset threshold for speaker activity binarization
    public var offset: Float
    /// Minimum speech segment duration in seconds
    public var minSpeechDuration: Float
    /// Minimum silence gap to split segments, in seconds
    public var minSilenceDuration: Float

    // MARK: - Presets

    /// Default streaming configuration matching the NeMo checkpoint.
    public static let `default` = SortformerConfig(
        nMels: 128,
        nFFT: 400,
        hopLength: 160,
        sampleRate: 16000,
        chunkLenSeconds: 6.0,
        leftContextSeconds: 1.0,
        rightContextSeconds: 7.0,
        subsamplingFactor: 8,
        spkcacheLen: 188,
        fifoLen: 40,
        fcDModel: 512,
        maxSpeakers: 4,
        onset: 0.5,
        offset: 0.3,
        minSpeechDuration: 0.3,
        minSilenceDuration: 0.15
    )

    public init(
        nMels: Int = 128,
        nFFT: Int = 400,
        hopLength: Int = 160,
        sampleRate: Int = 16000,
        chunkLenSeconds: Float = 6.0,
        leftContextSeconds: Float = 1.0,
        rightContextSeconds: Float = 7.0,
        subsamplingFactor: Int = 8,
        spkcacheLen: Int = 188,
        fifoLen: Int = 40,
        fcDModel: Int = 512,
        maxSpeakers: Int = 4,
        onset: Float = 0.5,
        offset: Float = 0.3,
        minSpeechDuration: Float = 0.3,
        minSilenceDuration: Float = 0.15
    ) {
        self.nMels = nMels
        self.nFFT = nFFT
        self.hopLength = hopLength
        self.sampleRate = sampleRate
        self.chunkLenSeconds = chunkLenSeconds
        self.leftContextSeconds = leftContextSeconds
        self.rightContextSeconds = rightContextSeconds
        self.subsamplingFactor = subsamplingFactor
        self.spkcacheLen = spkcacheLen
        self.fifoLen = fifoLen
        self.fcDModel = fcDModel
        self.maxSpeakers = maxSpeakers
        self.onset = onset
        self.offset = offset
        self.minSpeechDuration = minSpeechDuration
        self.minSilenceDuration = minSilenceDuration
    }
}
