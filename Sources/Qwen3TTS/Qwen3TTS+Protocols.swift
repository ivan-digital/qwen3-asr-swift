import AudioCommon

// MARK: - SpeechGenerationModel

extension Qwen3TTSModel: SpeechGenerationModel {
    public var sampleRate: Int { 24000 }

    public func generate(text: String, language: String?) async throws -> [Float] {
        let lang = language ?? "english"
        let speaker = speakerForLanguage(lang)
        return synthesize(text: text, language: lang, speaker: speaker)
    }

    public func generateStream(text: String, language: String?) -> AsyncThrowingStream<AudioChunk, Error> {
        let lang = language ?? "english"
        let speaker = speakerForLanguage(lang)
        return synthesizeStream(text: text, language: lang, speaker: speaker)
    }

    /// Select the best speaker for a given language.
    /// Uses native-language speakers per Qwen3-TTS docs for optimal quality.
    /// Falls back to `defaultSpeaker` if set, otherwise Ryan (English native).
    private func speakerForLanguage(_ language: String) -> String? {
        // If no speaker config (Base model), return nil
        guard speakerConfig != nil else { return nil }

        switch language.lowercased() {
        case "english":    return defaultSpeaker ?? "ryan"
        case "chinese":    return "vivian"
        case "japanese":   return "ono_anna"
        case "korean":     return "sohee"
        default:
            // Russian, German, French, Italian, Spanish, Portuguese —
            // no dedicated native speaker. Use Ryan (English) as best general voice.
            return defaultSpeaker ?? "ryan"
        }
    }
}
