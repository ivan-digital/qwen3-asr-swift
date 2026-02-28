import Foundation

// MARK: - Talker Config

public struct TalkerConfig: Codable, Sendable {
    public var hiddenSize: Int = 1024
    public var numLayers: Int = 28
    public var numHeads: Int = 16
    public var numKVHeads: Int = 8
    public var headDim: Int = 128
    public var intermediateSize: Int = 3072
    public var ropeTheta: Float = 1_000_000.0
    public var mropeSections: [Int] = [24, 20, 20]
    public var rmsNormEps: Float = 1e-6
    public var textVocabSize: Int = 151936
    public var textHiddenSize: Int = 2048
    public var codecVocabSize: Int = 3072
    public var groupSize: Int = 64
    public var bits: Int = 4

    public init() {}

    public static var base06B: TalkerConfig { TalkerConfig() }
}

// MARK: - Code Predictor Config

public struct CodePredictorConfig: Codable, Sendable {
    public var hiddenSize: Int = 1024
    public var numLayers: Int = 5
    public var numHeads: Int = 16
    public var numKVHeads: Int = 8
    public var headDim: Int = 128
    public var intermediateSize: Int = 3072
    public var ropeTheta: Float = 1_000_000.0
    public var rmsNormEps: Float = 1e-6
    public var vocabSize: Int = 2048
    public var numCodeGroups: Int = 16
    public var groupSize: Int = 64
    public var bits: Int = 4

    public init() {}
}

// MARK: - Speech Tokenizer Decoder Config

public struct SpeechTokenizerDecoderConfig: Codable, Sendable {
    public var latentDim: Int = 1024
    public var decoderDim: Int = 1536
    public var hiddenSize: Int = 512
    public var numHeads: Int = 16
    public var numKVHeads: Int = 16
    public var headDim: Int = 64
    public var numLayers: Int = 8
    public var upsampleRates: [Int] = [8, 5, 4, 3]
    public var upsamplingRatios: [Int] = [2, 2]
    public var numQuantizers: Int = 16
    public var semanticCodebookSize: Int = 2048
    public var acousticCodebookSize: Int = 2048
    public var codebookDim: Int = 256
    public var slidingWindow: Int = 72
    public var sampleRate: Int = 24000
    public var frameRate: Double = 12.5
    public var rmsNormEps: Float = 1e-8

    public init() {}
}

// MARK: - Special Codec Tokens

public struct CodecTokens {
    public static let codecPad: Int = 2148
    public static let codecBos: Int = 2149
    public static let codecEos: Int = 2150
    public static let codecThink: Int = 2154
    public static let codecNothink: Int = 2155
    public static let codecThinkBos: Int = 2156
    public static let codecThinkEos: Int = 2157
    public static let ttsPad: Int = 151671
    public static let ttsBos: Int = 151672
    public static let ttsEos: Int = 151673
    public static let languageEnglish: Int = 2050
    public static let languageGerman: Int = 2052
    public static let languageChinese: Int = 2055
    public static let languageJapanese: Int = 2058
    public static let languageSpanish: Int = 2054
    public static let languageFrench: Int = 2061
    public static let languageKorean: Int = 2064
    public static let languageRussian: Int = 2069
    public static let languageItalian: Int = 2070
    public static let languagePortuguese: Int = 2071
    public static let languageBeijingDialect: Int = 2074
    public static let languageSichuanDialect: Int = 2062

    public static func languageId(for language: String) -> Int? {
        switch language.lowercased() {
        case "english", "en": return languageEnglish
        case "german", "de": return languageGerman
        case "chinese", "zh": return languageChinese
        case "japanese", "ja": return languageJapanese
        case "spanish", "es": return languageSpanish
        case "french", "fr": return languageFrench
        case "korean", "ko": return languageKorean
        case "russian", "ru": return languageRussian
        case "italian", "it": return languageItalian
        case "portuguese", "pt": return languagePortuguese
        case "beijing_dialect": return languageBeijingDialect
        case "sichuan_dialect": return languageSichuanDialect
        default: return nil
        }
    }
}

// MARK: - Speaker Config

/// Parsed speaker data from CustomVoice model config.json
public struct SpeakerConfig: Sendable {
    /// Speaker name → codec token ID mapping
    public let speakerIds: [String: Int]
    /// Speaker name → dialect name mapping (e.g., "eric" → "sichuan_dialect")
    public let speakerDialects: [String: String]
    /// Dynamic language ID mapping from config.json codec_language_id
    public let codecLanguageIds: [String: Int]

    public var availableSpeakers: [String] { Array(speakerIds.keys).sorted() }

    public init(speakerIds: [String: Int], speakerDialects: [String: String], codecLanguageIds: [String: Int] = [:]) {
        self.speakerIds = speakerIds
        self.speakerDialects = speakerDialects
        self.codecLanguageIds = codecLanguageIds
    }
}

// MARK: - Streaming Config

/// Configuration for streaming TTS synthesis with chunked audio output.
public struct StreamingConfig: Sendable {
    /// Number of codec frames in the first emitted chunk (lower = lower latency).
    /// At 12.5 Hz, each frame = 80ms audio. Default 3 = 240ms audio.
    public var firstChunkFrames: Int

    /// Number of codec frames per subsequent chunk. Default 25 = 2s audio.
    public var chunkFrames: Int

    /// Left context frames for decoder quality (overlapping decode window). Default 10.
    public var decoderLeftContext: Int

    public init(firstChunkFrames: Int = 3, chunkFrames: Int = 25, decoderLeftContext: Int = 10) {
        self.firstChunkFrames = firstChunkFrames
        self.chunkFrames = chunkFrames
        self.decoderLeftContext = decoderLeftContext
    }

    /// Balanced defaults: ~225ms first-packet latency, 2s subsequent chunks.
    public static var `default`: StreamingConfig { .init() }

    /// Low-latency preset: ~120ms first-packet latency, smaller chunks.
    public static var lowLatency: StreamingConfig { .init(firstChunkFrames: 1, chunkFrames: 15) }
}

// MARK: - Model Variant

/// Well-known TTS model variants
public enum TTSModelVariant: String, CaseIterable, Sendable {
    case base = "aufklarer/Qwen3-TTS-12Hz-0.6B-Base-MLX-4bit"
    case customVoice = "aufklarer/Qwen3-TTS-12Hz-0.6B-CustomVoice-MLX-4bit"
}

// MARK: - Combined TTS Config

public struct Qwen3TTSConfig: Codable, Sendable {
    public var talker: TalkerConfig
    public var codePredictor: CodePredictorConfig
    public var speechTokenizerDecoder: SpeechTokenizerDecoderConfig

    public init(
        talker: TalkerConfig = TalkerConfig(),
        codePredictor: CodePredictorConfig = CodePredictorConfig(),
        speechTokenizerDecoder: SpeechTokenizerDecoderConfig = SpeechTokenizerDecoderConfig()
    ) {
        self.talker = talker
        self.codePredictor = codePredictor
        self.speechTokenizerDecoder = speechTokenizerDecoder
    }

    public static var base06B: Qwen3TTSConfig {
        Qwen3TTSConfig()
    }
}
