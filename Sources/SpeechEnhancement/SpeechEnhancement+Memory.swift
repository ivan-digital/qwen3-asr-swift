import AudioCommon

extension SpeechEnhancer: ModelMemoryManageable {
    public var isLoaded: Bool { _isLoaded }

    public func unload() {
        guard _isLoaded else { return }
        network = nil
        _isLoaded = false
    }

    public var memoryFootprint: Int {
        guard _isLoaded else { return 0 }
        // DeepFilterNet3 CoreML FP16: ~4.2 MB
        return 4_200_000
    }
}
