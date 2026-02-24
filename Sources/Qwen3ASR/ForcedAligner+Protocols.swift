import Qwen3Common

// MARK: - ForcedAlignmentModel

extension Qwen3ForcedAligner: ForcedAlignmentModel {
    public func align(audio: [Float], text: String, sampleRate: Int, language: String?) -> [AlignedWord] {
        align(audio: audio, text: text, sampleRate: sampleRate, language: language ?? "English")
    }
}
