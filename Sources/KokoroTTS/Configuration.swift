import Foundation

/// Configuration for Kokoro-82M TTS model.
public struct KokoroConfig: Codable, Sendable {
    /// Output audio sample rate in Hz.
    public let sampleRate: Int
    /// Maximum phoneme input length.
    public let maxPhonemeLength: Int
    /// Style embedding dimension (ref_s input to CoreML model).
    public let styleDim: Int
    /// Number of random phases for iSTFTNet vocoder.
    public let numPhases: Int
    /// Supported languages.
    public let languages: [String]

    public init(
        sampleRate: Int = 24000,
        maxPhonemeLength: Int = 510,
        styleDim: Int = 256,
        numPhases: Int = 9,
        languages: [String] = ["en", "fr", "es", "ja", "zh", "hi", "pt", "ko"]
    ) {
        self.sampleRate = sampleRate
        self.maxPhonemeLength = maxPhonemeLength
        self.styleDim = styleDim
        self.numPhases = numPhases
        self.languages = languages
    }

    /// Default configuration matching Kokoro-82M.
    public static let `default` = KokoroConfig()
}

/// Available model variants for different maximum output lengths.
///
/// CoreML models on the Neural Engine require fixed output shapes. The model
/// is compiled into multiple variants for different maximum durations.
/// aufklarer/Kokoro-82M-CoreML provides v2.1 (iOS 16+) and v2.4 (iOS 17+) variants.
public enum ModelBucket: CaseIterable, Sendable, Hashable {
    /// v2.1, max ~5s output, 124 token input
    case v21_5s
    /// v2.1, max ~10s output, 168 token input
    case v21_10s
    /// v2.1, max ~15s output, 249 token input
    case v21_15s
    /// v2.4, max 10s output, 242 token input (iOS 17+)
    case v24_10s
    /// v2.4, max 15s output, 242 token input (iOS 17+)
    case v24_15s

    /// CoreML model filename (without extension).
    public var modelName: String {
        switch self {
        case .v21_5s:  return "kokoro_21_5s"
        case .v21_10s: return "kokoro_21_10s"
        case .v21_15s: return "kokoro_21_15s"
        case .v24_10s: return "kokoro_24_10s"
        case .v24_15s: return "kokoro_24_15s"
        }
    }

    /// Maximum input token length for this variant.
    public var maxTokens: Int {
        switch self {
        case .v21_5s:  return 124
        case .v21_10s: return 168
        case .v21_15s: return 249
        case .v24_10s: return 242
        case .v24_15s: return 242
        }
    }

    /// Maximum output audio samples.
    public var maxSamples: Int {
        switch self {
        case .v21_5s:  return 175_800
        case .v21_10s: return 253_200
        case .v21_15s: return 372_600
        case .v24_10s: return 240_000
        case .v24_15s: return 360_000
        }
    }

    /// Maximum duration in seconds.
    public var maxDuration: Double { Double(maxSamples) / 24000.0 }

    /// Select the smallest bucket that fits the given token count.
    ///
    /// When `available` is provided, only selects from loaded buckets.
    /// This ensures selection matches what `KokoroNetwork` actually loaded
    /// (e.g., with `maxBuckets: 1`, only the smallest bucket is available).
    public static func select(
        forTokenCount tokens: Int,
        preferV24: Bool = true,
        available: Set<ModelBucket>? = nil
    ) -> ModelBucket? {
        let candidates: [ModelBucket]
        if preferV24 {
            // v2.4 first (iOS 17+), then v2.1 fallback, sorted by maxTokens
            candidates = [.v24_10s, .v21_5s, .v21_10s, .v24_15s, .v21_15s]
        } else {
            candidates = [.v21_5s, .v21_10s, .v21_15s]
        }

        for bucket in candidates {
            if let avail = available, !avail.contains(bucket) { continue }
            if tokens <= bucket.maxTokens {
                return bucket
            }
        }

        // If nothing fits, return the largest available bucket (truncated output)
        if let avail = available {
            return candidates.filter { avail.contains($0) }.last
        }
        return nil
    }
}
