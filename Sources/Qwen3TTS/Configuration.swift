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

    public static func languageId(for language: String) -> Int? {
        switch language.lowercased() {
        case "english", "en": return languageEnglish
        case "german", "de": return languageGerman
        case "chinese", "zh": return languageChinese
        case "japanese", "ja": return languageJapanese
        default: return nil
        }
    }
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
