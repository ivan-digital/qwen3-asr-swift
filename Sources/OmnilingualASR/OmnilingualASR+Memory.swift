import AudioCommon

extension OmnilingualASRModel: ModelMemoryManageable {
    public var isLoaded: Bool { _isLoaded }

    public func unload() {
        guard _isLoaded else { return }
        model = nil
        _isLoaded = false
    }

    public var memoryFootprint: Int {
        guard _isLoaded else { return 0 }
        // Omnilingual CTC-300M CoreML INT8: ~312 MB on disk (matches the
        // published repo size). CoreML runtime overhead bumps this modestly
        // during inference.
        return 312 * 1024 * 1024
    }
}
