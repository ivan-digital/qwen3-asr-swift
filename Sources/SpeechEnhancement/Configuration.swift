import Foundation

/// Configuration for DeepFilterNet3 speech enhancement model.
public struct DeepFilterNet3Config {
    /// FFT window size
    public let fftSize: Int
    /// Hop size (frame shift) in samples
    public let hopSize: Int
    /// Number of ERB frequency bands
    public let erbBands: Int
    /// Number of frequency bins for deep filtering (lowest bins)
    public let dfBins: Int
    /// Deep filter order (number of taps)
    public let dfOrder: Int
    /// Deep filter lookahead in frames
    public let dfLookahead: Int
    /// Number of convolution channels
    public let convCh: Int
    /// Embedding/GRU hidden dimension
    public let embHidden: Int
    /// Number of GRU layers in encoder
    public let encGruLayers: Int
    /// Number of GRU layers in ERB decoder
    public let erbDecGruLayers: Int
    /// DF decoder hidden dimension
    public let dfHidden: Int
    /// Number of GRU layers in DF decoder
    public let dfGruLayers: Int
    /// Groups for encoder linear layer
    public let encLinGroups: Int
    /// Groups for linear layers in SqueezedGRU
    public let linGroups: Int
    /// Sample rate
    public let sampleRate: Int
    /// Convolution lookahead in frames
    public let convLookahead: Int
    /// LSNR max value
    public let lsnrMax: Float
    /// LSNR min value
    public let lsnrMin: Float
    /// Exponential normalization time constant
    public let normTau: Float

    /// Number of FFT frequency bins (fftSize / 2 + 1)
    public var freqBins: Int { fftSize / 2 + 1 }

    /// Normalization alpha (exponential decay)
    public var normAlpha: Float {
        exp(-Float(hopSize) / Float(sampleRate) / normTau)
    }

    /// Default configuration matching the pretrained DeepFilterNet3 model.
    public static let `default` = DeepFilterNet3Config(
        fftSize: 960,
        hopSize: 480,
        erbBands: 32,
        dfBins: 96,
        dfOrder: 5,
        dfLookahead: 2,
        convCh: 64,
        embHidden: 256,
        encGruLayers: 1,
        erbDecGruLayers: 2,
        dfHidden: 256,
        dfGruLayers: 2,
        encLinGroups: 32,
        linGroups: 16,
        sampleRate: 48000,
        convLookahead: 2,
        lsnrMax: 35,
        lsnrMin: -15,
        normTau: 1.0
    )
}
