import AudioCommon

// MARK: - VoiceActivityDetectionModel

extension PyannoteVADModel: VoiceActivityDetectionModel {
    public var inputSampleRate: Int { segConfig.sampleRate }
}

// MARK: - SpeakerEmbeddingModel

extension WeSpeakerModel: SpeakerEmbeddingModel {}

// MARK: - SpeakerDiarizationModel

extension PyannoteDiarizationPipeline: SpeakerDiarizationModel {
    public var inputSampleRate: Int { segConfig.sampleRate }

    public func diarize(audio: [Float], sampleRate: Int) -> [DiarizedSegment] {
        diarize(audio: audio, sampleRate: sampleRate, config: .default).segments
    }
}

// MARK: - SpeakerExtractionCapable

extension PyannoteDiarizationPipeline: SpeakerExtractionCapable {
    public func extractSpeaker(audio: [Float], sampleRate: Int, targetEmbedding: [Float]) -> [SpeechSegment] {
        extractSpeaker(audio: audio, sampleRate: sampleRate, targetEmbedding: targetEmbedding, config: .default)
    }
}
