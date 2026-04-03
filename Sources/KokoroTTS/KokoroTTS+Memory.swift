import AudioCommon

extension KokoroTTSModel: ModelMemoryManageable {
    public var isLoaded: Bool { _isLoaded }

    public func unload() {
        network = nil
        voiceEmbeddings = [:]
    }

    public var memoryFootprint: Int {
        guard _isLoaded else { return 0 }
        // ~200 MB for E2E model (kokoro_5s.mlmodelc)
        return 200 * 1024 * 1024
    }
}
