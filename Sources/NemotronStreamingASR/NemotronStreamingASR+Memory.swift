import AudioCommon

extension NemotronStreamingASRModel: ModelMemoryManageable {
    public var isLoaded: Bool { _isLoaded }

    public func unload() {
        guard _isLoaded else { return }
        encoder = nil
        decoder = nil
        joint = nil
        _isLoaded = false
    }

    public var memoryFootprint: Int {
        guard _isLoaded else { return 0 }
        return 600 * 1024 * 1024  // ~600 MB for INT8 0.6B encoder + small decoder/joint
    }
}
