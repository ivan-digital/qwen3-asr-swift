import Qwen3Common

// MARK: - SpeechRecognitionModel

extension Qwen3ASRModel: SpeechRecognitionModel {
    public var inputSampleRate: Int { 16000 }

    public func transcribe(audio: [Float], sampleRate: Int, language: String?) -> String {
        transcribe(audio: audio, sampleRate: sampleRate, language: language, maxTokens: 448)
    }
}
