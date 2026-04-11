import AudioCommon

extension OmnilingualASRModel: SpeechRecognitionModel {
    public var inputSampleRate: Int { config.sampleRate }

    /// Conforms to `SpeechRecognitionModel`. The CTC variant is language-agnostic
    /// and ignores the `language` hint (it's a 1600+ language model — any
    /// hint is a no-op). Pass non-nil if you want to be explicit about intent.
    public func transcribe(audio: [Float], sampleRate: Int, language: String?) -> String {
        do {
            return try transcribeAudio(audio, sampleRate: sampleRate, language: language)
        } catch {
            AudioLog.inference.error("Omnilingual transcription failed: \(error)")
            return ""
        }
    }
}
