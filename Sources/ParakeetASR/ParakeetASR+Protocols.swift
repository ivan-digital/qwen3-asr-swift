import AudioCommon

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
}
