import AudioCommon

extension CosyVoiceTTSModel: ModelMemoryManageable {
    public var isLoaded: Bool { _isLoaded }

    public func unload() {
        guard _isLoaded else { return }
        llm.clearParameters()
        flow.clearParameters()
        hifigan.clearParameters()
        _isLoaded = false
    }

    public var memoryFootprint: Int {
        guard _isLoaded else { return 0 }
        return llm.parameterMemoryBytes()
            + flow.parameterMemoryBytes()
            + hifigan.parameterMemoryBytes()
    }
}
