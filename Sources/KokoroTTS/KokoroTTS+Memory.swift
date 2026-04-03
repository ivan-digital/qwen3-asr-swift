import AudioCommon

extension KokoroTTSModel: ModelMemoryManageable {
    public var isLoaded: Bool { _isLoaded }

    public func unload() {
        guard _isLoaded else { return }
        network = nil
        voiceEmbeddings = [:]
        _isLoaded = false
    }

    public var memoryFootprint: Int {
        guard _isLoaded else { return 0 }
        // ~163 MB for 3-stage pipeline (duration 39 + prosody 17 + decoder 107)
        return 163 * 1024 * 1024
    }
}
