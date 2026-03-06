import AudioCommon

extension PyannoteVADModel: ModelMemoryManageable {
    public var isLoaded: Bool { _isLoaded }

    public func unload() {
        guard _isLoaded else { return }
        model.clearParameters()
        _isLoaded = false
    }

    public var memoryFootprint: Int {
        guard _isLoaded else { return 0 }
        return model.parameterMemoryBytes()
    }
}
