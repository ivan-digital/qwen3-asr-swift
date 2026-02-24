import Qwen3Common

// MARK: - SpeechGenerationModel

extension Qwen3TTSModel: SpeechGenerationModel {
    public var sampleRate: Int { 24000 }

    public func generate(text: String, language: String?) async throws -> [Float] {
        synthesize(text: text, language: language ?? "english")
    }

    public func generateStream(text: String, language: String?) -> AsyncThrowingStream<AudioChunk, Error> {
        synthesizeStream(text: text, language: language ?? "english")
    }
}
