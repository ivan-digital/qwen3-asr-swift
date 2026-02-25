import AudioCommon

// MARK: - VoiceActivityDetectionModel

extension PyannoteVADModel: VoiceActivityDetectionModel {
    public var inputSampleRate: Int { segConfig.sampleRate }
}

// MARK: - SpeakerEmbeddingModel

extension WeSpeakerModel: SpeakerEmbeddingModel {}

// MARK: - SpeakerDiarizationModel

extension DiarizationPipeline: SpeakerDiarizationModel {
    public var inputSampleRate: Int { segConfig.sampleRate }

    public func diarize(audio: [Float], sampleRate: Int) -> [DiarizedSegment] {
        diarize(audio: audio, sampleRate: sampleRate, config: .default).segments
    }
}
