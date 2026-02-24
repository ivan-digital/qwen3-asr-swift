import Qwen3Common

// MARK: - SpeechGenerationModel

extension CosyVoiceTTSModel: SpeechGenerationModel {
    public var sampleRate: Int { config.sampleRate }

    public func generate(text: String, language: String?) async throws -> [Float] {
        synthesize(text: text, language: language ?? "english")
    }

    public func generateStream(text: String, language: String?) -> AsyncThrowingStream<AudioChunk, Error> {
        synthesizeStream(text: text, language: language ?? "english")
    }
}
