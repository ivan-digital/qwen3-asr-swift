import Foundation

/// Configuration for Kokoro-82M TTS model.
public struct KokoroConfig: Codable, Sendable {
    /// Output audio sample rate in Hz.
    public let sampleRate: Int
    /// Maximum phoneme input length.
    public let maxPhonemeLength: Int
    /// Style embedding dimension (ref_s input to CoreML model).
    public let styleDim: Int
    /// Supported languages.
    public let languages: [String]
    /// Audio samples produced per alignment frame.
    public let samplesPerFrame: Int

    public init(
        sampleRate: Int = 24000,
        maxPhonemeLength: Int = 510,
        styleDim: Int = 256,
        languages: [String] = ["en", "fr", "es", "ja", "zh", "hi", "pt", "ko"],
        samplesPerFrame: Int = 600
    ) {
        self.sampleRate = sampleRate
        self.maxPhonemeLength = maxPhonemeLength
        self.styleDim = styleDim
        self.languages = languages
        self.samplesPerFrame = samplesPerFrame
    }

    /// Default configuration matching Kokoro-82M.
    public static let `default` = KokoroConfig()
}

/// Phoneme input length buckets for the duration model.
///
/// The duration model uses enumerated shapes to minimize backward LSTM
/// contamination from padding. Input is padded to the nearest bucket.
public enum PhonemeBucket: Int, CaseIterable, Sendable {
    case p16 = 16
    case p32 = 32
    case p64 = 64
    case p128 = 128

    /// Select the smallest bucket that fits the given token count.
    public static func select(forTokenCount count: Int) -> PhonemeBucket? {
        for bucket in allCases {
            if count <= bucket.rawValue { return bucket }
        }
        return nil
    }
}

/// Decoder model buckets for different maximum output lengths.
///
/// CoreML decoder models have fixed output shapes. Each bucket handles a
/// different maximum frame count, trading memory for maximum output length.
public enum DecoderBucket: CaseIterable, Sendable, Hashable {
    /// Up to 200 frames → 120,000 samples ≈ 5.0s
    case d5s
    /// Up to 400 frames → 240,000 samples ≈ 10.0s
    case d10s
    /// Up to 600 frames → 360,000 samples ≈ 15.0s
    case d15s

    /// CoreML model filename (without extension).
    public var modelName: String {
        switch self {
        case .d5s:  return "decoder_5s"
        case .d10s: return "decoder_10s"
        case .d15s: return "decoder_15s"
        }
    }

    /// Maximum alignment frames for this bucket.
    public var maxFrames: Int {
        switch self {
        case .d5s:  return 200
        case .d10s: return 400
        case .d15s: return 600
        }
    }

    /// Maximum output audio samples (frames × 600).
    public var maxSamples: Int { maxFrames * 600 }

    /// Maximum duration in seconds.
    public var maxDuration: Double { Double(maxSamples) / 24000.0 }

    /// Select the smallest decoder bucket that fits the given frame count.
    public static func select(forFrameCount frames: Int) -> DecoderBucket? {
        for bucket in allCases {
            if frames <= bucket.maxFrames { return bucket }
        }
        return nil
    }
}

// MARK: - Legacy ModelBucket (for old end-to-end models)

/// Legacy model variants for old end-to-end Kokoro models.
/// Kept for backward compatibility with aufklarer/Kokoro-82M-CoreML v2.1/v2.4 models.
public enum ModelBucket: CaseIterable, Sendable, Hashable {
    case v21_5s, v21_10s, v21_15s, v24_10s, v24_15s

    public var modelName: String {
        switch self {
        case .v21_5s:  return "kokoro_21_5s"
        case .v21_10s: return "kokoro_21_10s"
        case .v21_15s: return "kokoro_21_15s"
        case .v24_10s: return "kokoro_24_10s"
        case .v24_15s: return "kokoro_24_15s"
        }
    }

    public var maxTokens: Int {
        switch self {
        case .v21_5s:  return 124
        case .v21_10s: return 168
        case .v21_15s: return 249
        case .v24_10s: return 242
        case .v24_15s: return 242
        }
    }

    public var maxSamples: Int {
        switch self {
        case .v21_5s:  return 175_800
        case .v21_10s: return 253_200
        case .v21_15s: return 372_600
        case .v24_10s: return 240_000
        case .v24_15s: return 360_000
        }
    }

    public var maxDuration: Double { Double(maxSamples) / 24000.0 }

    public static func select(forTokenCount tokens: Int, preferV24: Bool = true) -> ModelBucket? {
        if preferV24 {
            if tokens <= ModelBucket.v24_10s.maxTokens { return .v24_10s }
        }
        let v21Buckets: [ModelBucket] = [.v21_5s, .v21_10s, .v21_15s]
        for bucket in v21Buckets {
            if tokens <= bucket.maxTokens { return bucket }
        }
        return nil
    }
}
