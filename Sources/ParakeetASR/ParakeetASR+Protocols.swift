import AudioCommon
import NaturalLanguage

extension ParakeetASRModel: SpeechRecognitionModel {
    public var inputSampleRate: Int { config.sampleRate }

    public func transcribe(audio: [Float], sampleRate: Int, language: String?) -> String {
        do {
            return try transcribeAudio(audio, sampleRate: sampleRate, language: language)
        } catch {
            AudioLog.inference.error("Parakeet transcription failed: \(error)")
            return ""
        }
    }

    public func transcribeWithLanguage(audio: [Float], sampleRate: Int, language: String?) -> TranscriptionResult {
        let text = transcribe(audio: audio, sampleRate: sampleRate, language: language)
        guard !text.isEmpty else { return TranscriptionResult(text: text) }
        let ttsLang = detectLanguage(text)
        return TranscriptionResult(text: text, language: ttsLang, confidence: lastConfidence, words: lastWordConfidences)
    }

    /// Use Apple NLLanguageRecognizer to detect language, mapped to TTS language name.
    private func detectLanguage(_ text: String) -> String? {
        let recognizer = NLLanguageRecognizer()
        recognizer.processString(text)
        guard let lang = recognizer.dominantLanguage else { return nil }
        return Self.ttsLanguageName(for: lang)
    }

    /// Map NLLanguage to Qwen3-TTS language names.
    /// Only includes languages the TTS model actually supports (codec_language_id).
    private static func ttsLanguageName(for lang: NLLanguage) -> String? {
        switch lang {
        case .english:              return "english"
        case .russian:              return "russian"
        case .simplifiedChinese,
             .traditionalChinese:   return "chinese"
        case .japanese:             return "japanese"
        case .korean:               return "korean"
        case .german:               return "german"
        case .french:               return "french"
        case .italian:              return "italian"
        case .spanish:              return "spanish"
        case .portuguese:           return "portuguese"
        default:                    return "english"
        }
    }
}
