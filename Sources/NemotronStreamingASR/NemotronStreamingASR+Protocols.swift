import Foundation
import AudioCommon

extension NemotronStreamingASRModel: SpeechRecognitionModel {
    public var inputSampleRate: Int { config.sampleRate }

    public func transcribe(audio: [Float], sampleRate: Int, language: String?) -> String {
        do {
            return try transcribeAudio(audio, sampleRate: sampleRate, language: language)
        } catch {
            AudioLog.inference.error("Nemotron streaming transcription failed: \(error)")
            return ""
        }
    }

    public func transcribeWithLanguage(audio: [Float], sampleRate: Int, language: String?) -> TranscriptionResult {
        let text = transcribe(audio: audio, sampleRate: sampleRate, language: language)
        // Nemotron is English-only.
        return TranscriptionResult(text: text, language: text.isEmpty ? nil : "english", confidence: 0)
    }
}
