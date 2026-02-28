import Foundation
import AudioCommon
import SpeechVAD

// MARK: - TranscriptionSegment

public struct TranscriptionSegment: Sendable {
    public let text: String
    public let startTime: Float
    public let endTime: Float
    public let isFinal: Bool
    public let segmentIndex: Int

    public init(text: String, startTime: Float, endTime: Float, isFinal: Bool, segmentIndex: Int) {
        self.text = text
        self.startTime = startTime
        self.endTime = endTime
        self.isFinal = isFinal
        self.segmentIndex = segmentIndex
    }
}

// MARK: - StreamingASRConfig

public struct StreamingASRConfig: Sendable {
    public var maxSegmentDuration: Float
    public var vadConfig: VADConfig
    public var language: String?
    public var maxTokens: Int
    public var emitPartialResults: Bool
    public var partialResultInterval: Float

    public init(
        maxSegmentDuration: Float = 10.0,
        vadConfig: VADConfig = .sileroDefault,
        language: String? = nil,
        maxTokens: Int = 448,
        emitPartialResults: Bool = false,
        partialResultInterval: Float = 1.0
    ) {
        self.maxSegmentDuration = maxSegmentDuration
        self.vadConfig = vadConfig
        self.language = language
        self.maxTokens = maxTokens
        self.emitPartialResults = emitPartialResults
        self.partialResultInterval = partialResultInterval
    }

    public static let `default` = StreamingASRConfig()
}

// MARK: - StreamingASR

/// Streaming ASR with VAD-guided segmentation.
///
/// - Warning: This class is not thread-safe. Create separate instances for concurrent use.
public class StreamingASR {
    private let asrModel: Qwen3ASRModel
    private let vadModel: SileroVADModel

    public init(asrModel: Qwen3ASRModel, vadModel: SileroVADModel) {
        self.asrModel = asrModel
        self.vadModel = vadModel
    }

    public static func fromPretrained(
        asrModelId: String = "aufklarer/Qwen3-ASR-0.6B-MLX-4bit",
        vadModelId: String = SileroVADModel.defaultModelId,
        progressHandler: ((Double, String) -> Void)? = nil
    ) async throws -> StreamingASR {
        let asr = try await Qwen3ASRModel.fromPretrained(
            modelId: asrModelId, progressHandler: progressHandler)
        let vad = try await SileroVADModel.fromPretrained(
            modelId: vadModelId, progressHandler: progressHandler)
        return StreamingASR(asrModel: asr, vadModel: vad)
    }

    /// Streaming transcription — emits TranscriptionSegments as speech is detected.
    public func transcribeStream(
        audio: [Float],
        sampleRate: Int = 16000,
        config: StreamingASRConfig = .default
    ) -> AsyncThrowingStream<TranscriptionSegment, Error> {
        AsyncThrowingStream { continuation in
                let samples: [Float]
                if sampleRate != 16000 {
                    samples = AudioFileLoader.resample(audio, from: sampleRate, to: 16000)
                } else {
                    samples = audio
                }

                let processor = StreamingVADProcessor(model: vadModel, config: config.vadConfig)
                let chunkSize = SileroVADModel.chunkSize
                var segmentIndex = 0
                var speechStartSample: Int?

                // Phase 2 state
                var lastPartialTime: Float = 0

                var offset = 0
                while offset < samples.count {
                    let end = min(offset + chunkSize, samples.count)
                    let chunk = Array(samples[offset..<end])
                    let events = processor.process(samples: chunk)

                    for event in events {
                        switch event {
                        case .speechStarted(let time):
                            speechStartSample = Int(time * 16000)
                            lastPartialTime = time

                        case .speechEnded(let segment):
                            if let startSample = speechStartSample {
                                let endSample = min(Int(segment.endTime * 16000), samples.count)
                                let segmentAudio = Array(samples[startSample..<endSample])
                                let text = asrModel.transcribe(
                                    audio: segmentAudio, sampleRate: 16000,
                                    language: config.language, maxTokens: config.maxTokens)
                                let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
                                if !trimmed.isEmpty {
                                    continuation.yield(TranscriptionSegment(
                                        text: trimmed,
                                        startTime: segment.startTime,
                                        endTime: segment.endTime,
                                        isFinal: true,
                                        segmentIndex: segmentIndex))
                                    segmentIndex += 1
                                }
                                speechStartSample = nil
                            }
                        }
                    }

                    // Phase 2: emit partial results during speech
                    if config.emitPartialResults, let startSample = speechStartSample {
                        let currentTime = processor.currentTime
                        let speechStart = Float(startSample) / 16000
                        let speechDuration = currentTime - speechStart

                        if currentTime - lastPartialTime >= config.partialResultInterval {
                            let endSample = min(Int(currentTime * 16000), samples.count)
                            let segmentAudio = Array(samples[startSample..<endSample])
                            let text = asrModel.transcribe(
                                audio: segmentAudio, sampleRate: 16000,
                                language: config.language, maxTokens: config.maxTokens)
                            let currentWords = text.trimmingCharacters(in: .whitespacesAndNewlines)
                                .split(separator: " ").map(String.init)

                            if !currentWords.isEmpty {
                                continuation.yield(TranscriptionSegment(
                                    text: currentWords.joined(separator: " "),
                                    startTime: speechStart,
                                    endTime: currentTime,
                                    isFinal: false,
                                    segmentIndex: segmentIndex))
                            }
                            lastPartialTime = currentTime
                        }

                        // Force-split if speech exceeds maxSegmentDuration
                        if speechDuration >= config.maxSegmentDuration {
                            let endSample = min(Int(currentTime * 16000), samples.count)
                            let segmentAudio = Array(samples[startSample..<endSample])
                            let text = asrModel.transcribe(
                                audio: segmentAudio, sampleRate: 16000,
                                language: config.language, maxTokens: config.maxTokens)
                            let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
                            if !trimmed.isEmpty {
                                continuation.yield(TranscriptionSegment(
                                    text: trimmed,
                                    startTime: speechStart,
                                    endTime: currentTime,
                                    isFinal: true,
                                    segmentIndex: segmentIndex))
                                segmentIndex += 1
                            }
                            speechStartSample = Int(currentTime * 16000)
                            lastPartialTime = currentTime
                        }
                    } else if !config.emitPartialResults, let startSample = speechStartSample {
                        // Force-split without partial results
                        let currentTime = processor.currentTime
                        let speechStart = Float(startSample) / 16000
                        let speechDuration = currentTime - speechStart

                        if speechDuration >= config.maxSegmentDuration {
                            let endSample = min(Int(currentTime * 16000), samples.count)
                            let segmentAudio = Array(samples[startSample..<endSample])
                            let text = asrModel.transcribe(
                                audio: segmentAudio, sampleRate: 16000,
                                language: config.language, maxTokens: config.maxTokens)
                            let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
                            if !trimmed.isEmpty {
                                continuation.yield(TranscriptionSegment(
                                    text: trimmed,
                                    startTime: speechStart,
                                    endTime: currentTime,
                                    isFinal: true,
                                    segmentIndex: segmentIndex))
                                segmentIndex += 1
                            }
                            speechStartSample = Int(currentTime * 16000)
                        }
                    }

                    offset = end
                }

                // Flush any remaining speech
                let flushEvents = processor.flush()
                for event in flushEvents {
                    if case .speechEnded(let segment) = event, let startSample = speechStartSample {
                        let endSample = min(Int(segment.endTime * 16000), samples.count)
                        let segmentAudio = Array(samples[startSample..<endSample])
                        let text = asrModel.transcribe(
                            audio: segmentAudio, sampleRate: 16000,
                            language: config.language, maxTokens: config.maxTokens)
                        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
                        if !trimmed.isEmpty {
                            continuation.yield(TranscriptionSegment(
                                text: trimmed,
                                startTime: segment.startTime,
                                endTime: segment.endTime,
                                isFinal: true,
                                segmentIndex: segmentIndex))
                        }
                    }
                }

                continuation.finish()
        }
    }
}

// MARK: - LocalAgreement Helper

/// Returns the longest common prefix of two word arrays (case-insensitive).
public func longestCommonPrefix(_ a: [String], _ b: [String]) -> [String] {
    var result: [String] = []
    for i in 0..<min(a.count, b.count) {
        if a[i].lowercased() == b[i].lowercased() {
            result.append(b[i])
        } else {
            break
        }
    }
    return result
}
