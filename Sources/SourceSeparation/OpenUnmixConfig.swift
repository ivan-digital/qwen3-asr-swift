import Foundation

/// Configuration for Open-Unmix source separation.
public struct OpenUnmixConfig: Codable, Sendable {
    /// Model variant (umxhq or umxl)
    public let model: String
    /// Hidden size for FC/LSTM layers
    public let hiddenSize: Int
    /// Number of FFT bins
    public let nbBins: Int
    /// Max frequency bin (bandwidth limit)
    public let maxBin: Int
    /// Number of audio channels (2 = stereo)
    public let nbChannels: Int
    /// Sample rate
    public let sampleRate: Int
    /// FFT window size
    public let nFFT: Int
    /// FFT hop size
    public let nHop: Int
    /// Target stems
    public let targets: [String]

    public static let umxhq = OpenUnmixConfig(
        model: "umxhq",
        hiddenSize: 512,
        nbBins: 2049,
        maxBin: 1487,
        nbChannels: 2,
        sampleRate: 44100,
        nFFT: 4096,
        nHop: 1024,
        targets: ["vocals", "drums", "bass", "other"]
    )

    public static let umxl = OpenUnmixConfig(
        model: "umxl",
        hiddenSize: 1024,
        nbBins: 2049,
        maxBin: 1487,
        nbChannels: 2,
        sampleRate: 44100,
        nFFT: 4096,
        nHop: 1024,
        targets: ["vocals", "drums", "bass", "other"]
    )
}

/// Available target stems for separation.
public enum SeparationTarget: String, CaseIterable, Sendable {
    case vocals
    case drums
    case bass
    case other
}
