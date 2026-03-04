import AudioCommon

extension SpeechEnhancer: SpeechEnhancementModel {
    public var inputSampleRate: Int { Self.sampleRate }
}
