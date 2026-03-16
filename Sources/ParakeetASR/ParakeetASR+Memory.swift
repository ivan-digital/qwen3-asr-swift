import AudioCommon

extension ParakeetASRModel: ModelMemoryManageable {
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
        // Parakeet-TDT CoreML INT4: ~315 MB encoder + decoder + joint
        return 315 * 1024 * 1024
    }
}
