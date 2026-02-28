import Foundation

/// Configuration for Parakeet TDT ASR model.
public struct ParakeetConfig: Codable, Sendable {
    /// Number of mel-spectrogram frequency bins.
    public let numMelBins: Int
    /// Expected input audio sample rate in Hz.
    public let sampleRate: Int
    /// FFT window size.
    public let nFFT: Int
    /// Hop length between successive STFT frames.
    public let hopLength: Int
    /// Window length for STFT.
    public let winLength: Int
    /// Pre-emphasis coefficient applied to the raw waveform.
    public let preEmphasis: Float
    /// Hidden dimension of the FastConformer encoder.
    public let encoderHidden: Int
    /// Number of FastConformer encoder layers.
    public let encoderLayers: Int
    /// Subsampling factor (encoder output frames = input frames / factor).
    public let subsamplingFactor: Int
    /// Hidden dimension of the LSTM prediction network.
    public let decoderHidden: Int
    /// Number of LSTM layers in the prediction network.
    public let decoderLayers: Int
    /// Vocabulary size (excluding blank).
    public let vocabSize: Int
    /// Token ID used for blank in the TDT decoder.
    public let blankTokenId: Int
    /// Number of duration bins for TDT.
    public let numDurationBins: Int
    /// Duration values corresponding to each bin index.
    public let durationBins: [Int]

    public init(
        numMelBins: Int = 128,
        sampleRate: Int = 16000,
        nFFT: Int = 512,
        hopLength: Int = 160,
        winLength: Int = 400,
        preEmphasis: Float = 0.97,
        encoderHidden: Int = 1024,
        encoderLayers: Int = 24,
        subsamplingFactor: Int = 8,
        decoderHidden: Int = 640,
        decoderLayers: Int = 2,
        vocabSize: Int = 8192,
        blankTokenId: Int = 8192,
        numDurationBins: Int = 5,
        durationBins: [Int] = [0, 1, 2, 3, 4]
    ) {
        self.numMelBins = numMelBins
        self.sampleRate = sampleRate
        self.nFFT = nFFT
        self.hopLength = hopLength
        self.winLength = winLength
        self.preEmphasis = preEmphasis
        self.encoderHidden = encoderHidden
        self.encoderLayers = encoderLayers
        self.subsamplingFactor = subsamplingFactor
        self.decoderHidden = decoderHidden
        self.decoderLayers = decoderLayers
        self.vocabSize = vocabSize
        self.blankTokenId = blankTokenId
        self.numDurationBins = numDurationBins
        self.durationBins = durationBins
    }

    /// Default configuration matching Parakeet-TDT 0.6B v3.
    public static let `default` = ParakeetConfig()
}
