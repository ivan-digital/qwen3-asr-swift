#if canImport(CoreML)
import Foundation
import AudioCommon

extension Qwen3TTSCoreMLModel: SpeechGenerationModel {
    public var sampleRate: Int { 24000 }

    public func generate(text: String, language: String?) async throws -> [Float] {
        try synthesize(text: text, language: language ?? "english")
    }
}
#endif
