import AudioCommon

extension PersonaPlexModel: ModelMemoryManageable {
    public var isLoaded: Bool { _isLoaded }

    public func unload() {
        guard _isLoaded else { return }
        temporal.clearParameters()
        depformer.clearParameters()
        mimi.clearParameters()
        _isLoaded = false
    }

    public var memoryFootprint: Int {
        guard _isLoaded else { return 0 }
        return temporal.parameterMemoryBytes()
            + depformer.parameterMemoryBytes()
            + mimi.parameterMemoryBytes()
    }
}
