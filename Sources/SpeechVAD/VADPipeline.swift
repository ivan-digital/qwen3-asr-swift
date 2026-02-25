import Foundation
import MLX
import AudioCommon

/// VAD pipeline: sliding window segmentation → aggregation → binarization.
///
/// Processes audio in overlapping 10-second windows, runs the segmentation model
/// on each window, aggregates overlapping frame predictions, then applies
/// hysteresis thresholding and duration filtering to produce speech segments.
public struct VADPipeline {

    /// Configuration for the pipeline
    public let config: VADConfig

    /// Sample rate expected by the segmentation model
    public let sampleRate: Int

    /// Number of output frames per 10s chunk (from the segmentation model)
    public let framesPerChunk: Int

    public init(config: VADConfig = .default, sampleRate: Int = 16000, framesPerChunk: Int = 589) {
        self.config = config
        self.sampleRate = sampleRate
        self.framesPerChunk = framesPerChunk
    }

    /// Duration of one frame in seconds.
    public var frameDuration: Float {
        config.windowDuration / Float(framesPerChunk)
    }

    // MARK: - Sliding Window

    /// Generate sliding window positions for the given audio length.
    /// - Parameter numSamples: total audio samples
    /// - Returns: array of (start, end) sample indices
    func windowPositions(numSamples: Int) -> [(start: Int, end: Int)] {
        let windowSamples = Int(config.windowDuration * Float(sampleRate))
        let stepSamples = Int(config.windowDuration * config.stepRatio * Float(sampleRate))

        guard numSamples > 0 else { return [] }

        // If audio is shorter than one window, just use one window (zero-padded)
        if numSamples <= windowSamples {
            return [(0, numSamples)]
        }

        var positions = [(start: Int, end: Int)]()
        var start = 0
        while start + windowSamples <= numSamples {
            positions.append((start, start + windowSamples))
            start += stepSamples
        }
        // Handle the last partial window
        if positions.isEmpty || positions.last!.end < numSamples {
            positions.append((numSamples - windowSamples, numSamples))
        }

        return positions
    }

    // MARK: - Frame Aggregation

    /// Aggregate overlapping frame-level speech probabilities from multiple windows.
    ///
    /// Each window produces `framesPerChunk` frames. Overlapping regions are
    /// averaged across windows.
    ///
    /// - Parameters:
    ///   - windowProbs: array of per-window speech probability arrays (each `framesPerChunk` long)
    ///   - positions: corresponding window positions (sample indices)
    ///   - numSamples: total audio length in samples
    /// - Returns: aggregated speech probability per frame for the entire audio
    func aggregateFrames(
        windowProbs: [[Float]],
        positions: [(start: Int, end: Int)],
        numSamples: Int
    ) -> [Float] {
        let totalDuration = Float(numSamples) / Float(sampleRate)
        let numFrames = Int(ceil(totalDuration / frameDuration))

        guard numFrames > 0 else { return [] }

        var sumProbs = [Float](repeating: 0, count: numFrames)
        var counts = [Float](repeating: 0, count: numFrames)

        for (windowIdx, (start, _)) in positions.enumerated() {
            let probs = windowProbs[windowIdx]
            let windowStartTime = Float(start) / Float(sampleRate)

            for (frameIdx, prob) in probs.enumerated() {
                let frameTime = windowStartTime + Float(frameIdx) * frameDuration
                let globalFrame = Int(frameTime / frameDuration)

                if globalFrame >= 0 && globalFrame < numFrames {
                    sumProbs[globalFrame] += prob
                    counts[globalFrame] += 1
                }
            }
        }

        // Average where we have overlapping windows
        return zip(sumProbs, counts).map { sum, count in
            count > 0 ? sum / count : 0
        }
    }

    // MARK: - Binarization (Hysteresis Thresholding)

    /// Apply hysteresis thresholding to speech probabilities.
    ///
    /// Speech starts when probability exceeds `onset` and ends when it drops
    /// below `offset`. This two-threshold approach prevents rapid toggling.
    ///
    /// - Parameter probs: per-frame speech probabilities
    /// - Returns: array of `SpeechSegment` with start/end times
    public func binarize(probs: [Float]) -> [SpeechSegment] {
        var segments = [SpeechSegment]()
        var inSpeech = false
        var speechStart: Float = 0

        for (i, prob) in probs.enumerated() {
            let time = Float(i) * frameDuration

            if !inSpeech && prob >= config.onset {
                inSpeech = true
                speechStart = time
            } else if inSpeech && prob < config.offset {
                inSpeech = false
                let segment = SpeechSegment(startTime: speechStart, endTime: time)
                segments.append(segment)
            }
        }

        // Close any open segment
        if inSpeech {
            let endTime = Float(probs.count) * frameDuration
            segments.append(SpeechSegment(startTime: speechStart, endTime: endTime))
        }

        return filterDurations(segments)
    }

    // MARK: - Duration Filtering

    /// Filter segments by minimum speech and silence durations.
    ///
    /// 1. Remove speech segments shorter than `minSpeechDuration`
    /// 2. Merge segments separated by silence shorter than `minSilenceDuration`
    func filterDurations(_ segments: [SpeechSegment]) -> [SpeechSegment] {
        guard !segments.isEmpty else { return [] }

        // Filter short speech segments
        let filtered = segments.filter { $0.duration >= config.minSpeechDuration }

        guard !filtered.isEmpty else { return [] }

        // Merge segments separated by short silence
        var merged = [SpeechSegment]()
        var current = filtered[0]

        for i in 1 ..< filtered.count {
            let next = filtered[i]
            let gap = next.startTime - current.endTime

            if gap < config.minSilenceDuration {
                // Merge: extend current segment
                current = SpeechSegment(
                    startTime: current.startTime,
                    endTime: next.endTime
                )
            } else {
                merged.append(current)
                current = next
            }
        }
        merged.append(current)

        return merged
    }
}
