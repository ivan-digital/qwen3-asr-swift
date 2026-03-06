import AudioCommon

extension Qwen3TTSModel: ModelMemoryManageable {
    public var isLoaded: Bool { _isLoaded }

    public func unload() {
        guard _isLoaded else { return }
        talker.clearParameters()
        codePredictor.clearParameters()
        codecDecoder.clearParameters()
        speakerEncoder.clearParameters()
        _isLoaded = false
    }

    public var memoryFootprint: Int {
        guard _isLoaded else { return 0 }
        return talker.parameterMemoryBytes()
            + codePredictor.parameterMemoryBytes()
            + codecDecoder.parameterMemoryBytes()
            + speakerEncoder.parameterMemoryBytes()
    }
}
