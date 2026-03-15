import AudioCommon

extension Qwen3ChatModel: ModelMemoryManageable {
    public var isLoaded: Bool { _isLoaded }

    public func unload() {
        guard _isLoaded else { return }
        // CoreMLGenerator holds MLModel references which are the main memory consumers.
        // We can't nil out a `let` property, but we can clear caches and mark as unloaded.
        // To fully release memory, discard this instance and create a new one via fromPretrained().
        conversationHistory.removeAll()
        systemPromptCached = false
        generator.resetCache()
        generator.clearPromptCache()
        _isLoaded = false
    }

    public var memoryFootprint: Int {
        guard _isLoaded else { return 0 }
        // INT4 Qwen3-0.6B CoreML ≈ 318 MB on disk
        return 318 * 1024 * 1024
    }
}
