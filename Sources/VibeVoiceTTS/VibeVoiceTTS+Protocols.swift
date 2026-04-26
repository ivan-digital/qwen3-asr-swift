import AudioCommon

extension VibeVoiceTTSModel: SpeechGenerationModel {
    public var sampleRate: Int { AudioConstants.sampleRate }

    public func generate(text: String, language: String?) async throws -> [Float] {
        // VibeVoice-Realtime-0.5B is EN/ZH only — `language` is accepted for
        // protocol conformance but not used for language dispatch. Non-EN/ZH
        // input will produce unintelligible audio per upstream model card.
        try await generate(text: text)
    }

    /// Override the default single-chunk stream with true chunk-streaming —
    /// each yielded `AudioChunk` is one acoustic-decoder output (~100-250 ms),
    /// so callers can start playback well before generation finishes.
    public func generateStream(text: String, language: String?) -> AsyncThrowingStream<AudioChunk, Error> {
        let rate = sampleRate
        let inner = generateChunkStream(text: text)
        return AsyncThrowingStream { continuation in
            Task {
                do {
                    var frame = 0
                    var lastSamples: [Float]? = nil
                    for try await samples in inner {
                        if let prev = lastSamples {
                            continuation.yield(AudioChunk(
                                samples: prev,
                                sampleRate: rate,
                                frameIndex: frame,
                                isFinal: false
                            ))
                            frame += prev.count
                        }
                        lastSamples = samples
                    }
                    if let last = lastSamples {
                        continuation.yield(AudioChunk(
                            samples: last,
                            sampleRate: rate,
                            frameIndex: frame,
                            isFinal: true
                        ))
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }
}
