import AudioCommon

// MARK: - SpeechToSpeechModel

extension PersonaPlexModel: SpeechToSpeechModel {
    public var sampleRate: Int { cfg.sampleRate }

    public func respond(userAudio: [Float]) -> [Float] {
        respond(userAudio: userAudio, voice: .NATM0).audio
    }

    public func respondStream(userAudio: [Float]) -> AsyncThrowingStream<AudioChunk, Error> {
        respondStream(userAudio: userAudio, voice: .NATM0)
    }
}
