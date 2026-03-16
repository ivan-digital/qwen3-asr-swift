import Foundation

/// Configuration for multi-segment dialogue synthesis.
public struct DialogueSynthesisConfig: Sendable {
    /// Silence gap between speaker turns in seconds
    public var turnGapSeconds: Float
    /// Crossfade overlap between segments in seconds (0 = no crossfade)
    public var crossfadeSeconds: Float
    /// Default instruction when no emotion tag is present
    public var defaultInstruction: String
    /// Maximum tokens per segment
    public var maxTokensPerSegment: Int

    public init(
        turnGapSeconds: Float = 0.2,
        crossfadeSeconds: Float = 0.0,
        defaultInstruction: String = "You are a helpful assistant.",
        maxTokensPerSegment: Int = 500
    ) {
        self.turnGapSeconds = turnGapSeconds
        self.crossfadeSeconds = crossfadeSeconds
        self.defaultInstruction = defaultInstruction
        self.maxTokensPerSegment = maxTokensPerSegment
    }
}

/// Audio output from a single dialogue segment.
public struct DialogueAudioSegment: Sendable {
    /// The parsed dialogue segment
    public let segment: DialogueSegment
    /// Audio samples at 24kHz
    public let samples: [Float]
    /// Index in the dialogue sequence
    public let segmentIndex: Int
}

/// Orchestrates multi-segment dialogue synthesis using CosyVoice3.
public enum DialogueSynthesizer {

    /// Synthesize all dialogue segments into a single audio buffer.
    ///
    /// For each segment:
    /// 1. Resolve instruction from emotion tag or default
    /// 2. Resolve speaker embedding (if available)
    /// 3. Synthesize via CosyVoiceTTSModel
    /// 4. Append with gap/crossfade between segments
    public static func synthesize(
        segments: [DialogueSegment],
        speakerEmbeddings: [String: [Float]],
        model: CosyVoiceTTSModel,
        language: String,
        config: DialogueSynthesisConfig = DialogueSynthesisConfig(),
        verbose: Bool = false
    ) -> [Float] {
        guard !segments.isEmpty else { return [] }

        var result: [Float] = []
        let gapSamples = silenceGap(seconds: config.turnGapSeconds, sampleRate: 24000)
        let crossfadeSampleCount = Int(config.crossfadeSeconds * 24000)

        for (i, segment) in segments.enumerated() {
            let instruction = segment.emotion.map {
                DialogueParser.emotionToInstruction($0)
            } ?? config.defaultInstruction

            let embedding = segment.speaker.flatMap { speakerEmbeddings[$0] }

            if verbose {
                var info = "  Segment \(i + 1)/\(segments.count): \"\(segment.text)\""
                if let spk = segment.speaker { info += " [speaker: \(spk)]" }
                if let emo = segment.emotion { info += " [emotion: \(emo)]" }
                print(info)
            }

            let samples = model.synthesize(
                text: segment.text,
                language: language,
                instruction: instruction,
                speakerEmbedding: embedding,
                verbose: verbose
            )

            guard !samples.isEmpty else { continue }

            if result.isEmpty {
                result = samples
            } else if crossfadeSampleCount > 0 {
                result = crossfade(left: result, right: samples, overlapSamples: crossfadeSampleCount)
            } else {
                result.append(contentsOf: gapSamples)
                result.append(contentsOf: samples)
            }
        }

        return result
    }

    /// Synthesize dialogue segments as an async stream of per-segment audio.
    public static func synthesizeStream(
        segments: [DialogueSegment],
        speakerEmbeddings: [String: [Float]],
        model: CosyVoiceTTSModel,
        language: String,
        config: DialogueSynthesisConfig = DialogueSynthesisConfig(),
        verbose: Bool = false
    ) -> AsyncThrowingStream<DialogueAudioSegment, Error> {
        let (stream, continuation) = AsyncThrowingStream<DialogueAudioSegment, Error>.makeStream()

        Task {
            for (i, segment) in segments.enumerated() {
                let instruction = segment.emotion.map {
                    DialogueParser.emotionToInstruction($0)
                } ?? config.defaultInstruction

                let embedding = segment.speaker.flatMap { speakerEmbeddings[$0] }

                let samples = model.synthesize(
                    text: segment.text,
                    language: language,
                    instruction: instruction,
                    speakerEmbedding: embedding,
                    verbose: verbose
                )

                let audioSegment = DialogueAudioSegment(
                    segment: segment,
                    samples: samples,
                    segmentIndex: i
                )
                continuation.yield(audioSegment)
            }
            continuation.finish()
        }

        return stream
    }

    // MARK: - Audio Utilities

    /// Generate silence of the specified duration.
    public static func silenceGap(seconds: Float, sampleRate: Int) -> [Float] {
        let count = Int(seconds * Float(sampleRate))
        return [Float](repeating: 0, count: max(count, 0))
    }

    /// Linear crossfade between two audio buffers.
    ///
    /// The last `overlapSamples` of `left` are blended with the first `overlapSamples` of `right`.
    /// If overlap exceeds either buffer, it is clamped.
    public static func crossfade(left: [Float], right: [Float], overlapSamples: Int) -> [Float] {
        let overlap = min(overlapSamples, left.count, right.count)
        guard overlap > 0 else { return left + right }

        var result = Array(left.dropLast(overlap))

        // Linear blend in the overlap region
        for i in 0..<overlap {
            let t = Float(i) / Float(overlap)
            let blended = left[left.count - overlap + i] * (1 - t) + right[i] * t
            result.append(blended)
        }

        result.append(contentsOf: right.dropFirst(overlap))
        return result
    }
}
