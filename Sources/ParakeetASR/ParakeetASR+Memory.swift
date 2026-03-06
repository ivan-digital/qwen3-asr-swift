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
        // CoreML models don't expose weight memory easily; return 0
        guard _isLoaded else { return 0 }
        return 0
    }
}
