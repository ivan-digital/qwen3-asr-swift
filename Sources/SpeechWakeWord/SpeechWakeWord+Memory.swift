import AudioCommon

extension WakeWordDetector: ModelMemoryManageable {
    public var isLoaded: Bool { _isLoaded }

    public func unload() {
        guard _isLoaded else { return }
        encoder = nil
        decoder = nil
        joiner = nil
        _isLoaded = false
    }

    public var memoryFootprint: Int {
        // 3.3 MB encoder + 525 KB decoder + 160 KB joiner .mlmodelc, plus
        // ~2 MB encoder state (36 per-layer caches + ConvNeXt left-pad).
        // Round to 6 MB to leave headroom for MLModel's internal buffers.
        guard _isLoaded else { return 0 }
        return 6 * 1024 * 1024
    }
}
