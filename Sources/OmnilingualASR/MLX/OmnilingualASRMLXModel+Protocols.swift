import AudioCommon

extension OmnilingualASRMLXModel: SpeechRecognitionModel {
    public var inputSampleRate: Int { config.sampleRate }

    /// Conforms to `SpeechRecognitionModel`. The CTC variant is language-agnostic
    /// and ignores the `language` hint, just like the CoreML model.
    public func transcribe(audio: [Float], sampleRate: Int, language: String?) -> String {
        do {
            return try transcribeAudio(audio, sampleRate: sampleRate, language: language)
        } catch {
            AudioLog.inference.error("Omnilingual MLX transcription failed: \(error)")
            return ""
        }
    }
}

extension OmnilingualASRMLXModel: ModelMemoryManageable {
    public var isLoaded: Bool { _isLoaded }

    public func unload() {
        guard _isLoaded else { return }
        _isLoaded = false
    }

    public var memoryFootprint: Int {
        guard _isLoaded else { return 0 }
        // Rough estimate: published file sizes for INT4 / INT8 variants.
        let mb: Int
        switch (config.variant, config.bits) {
        case (.m300, 4): mb = 193
        case (.m300, 8): mb = 342
        case (.b1, 4):   mb = 549
        case (.b1, 8):   mb = 1006
        case (.b3, 4):   mb = 1709
        case (.b3, 8):   mb = 3159
        case (.b7, 4):   mb = 3550
        case (.b7, 8):   mb = 6630
        default:         mb = 312
        }
        return mb * 1024 * 1024
    }
}
