import Foundation
import MLX
import AudioCommon

// MARK: - Configuration

/// Configuration for speaker diarization.
public struct DiarizationConfig: Sendable {
    /// Onset threshold for speaker activity
    public var onset: Float
    /// Offset threshold for speaker activity
    public var offset: Float
    /// Minimum speech segment duration in seconds
    public var minSpeechDuration: Float
    /// Minimum silence duration between segments in seconds
    public var minSilenceDuration: Float
    /// Minimum number of speakers (0 = automatic)
    public var minSpeakers: Int
    /// Maximum number of speakers (0 = automatic)
    public var maxSpeakers: Int

    public init(
        onset: Float = 0.5,
        offset: Float = 0.3,
        minSpeechDuration: Float = 0.3,
        minSilenceDuration: Float = 0.15,
        minSpeakers: Int = 0,
        maxSpeakers: Int = 0
    ) {
        self.onset = onset
        self.offset = offset
        self.minSpeechDuration = minSpeechDuration
        self.minSilenceDuration = minSilenceDuration
        self.minSpeakers = minSpeakers
        self.maxSpeakers = maxSpeakers
    }

    public static let `default` = DiarizationConfig()
}

// MARK: - Result

/// Result of speaker diarization.
public struct DiarizationResult: Sendable {
    /// Diarized speech segments with speaker IDs
    public let segments: [DiarizedSegment]
    /// Number of distinct speakers found
    public let numSpeakers: Int
    /// Centroid embedding for each speaker (speaker ID → 256-dim embedding)
    public let speakerEmbeddings: [[Float]]

    public init(segments: [DiarizedSegment], numSpeakers: Int, speakerEmbeddings: [[Float]]) {
        self.segments = segments
        self.numSpeakers = numSpeakers
        self.speakerEmbeddings = speakerEmbeddings
    }
}

// MARK: - Pipeline

/// Speaker diarization pipeline: segmentation + activity-based speaker chaining.
///
/// - Warning: This class is not thread-safe. Create separate instances for concurrent use.
///
/// Pipeline (with optional VAD pre-filter):
/// 0. **VAD Pre-filter** (optional): Silero VAD masks non-speech regions → reduces false alarms
/// 1. **Segmentation + Speaker Chaining**: Pyannote on 10s sliding windows (50% overlap) →
///    per-speaker probability tracks → Pearson correlation in overlap zones → greedy exclusive
///    matching → global speaker IDs
/// 2. **Post-hoc Embedding**: WeSpeaker 256-dim centroid per speaker (for target speaker extraction)
///
/// ```swift
/// let pipeline = try await DiarizationPipeline.fromPretrained(useVADFilter: true)
/// let result = pipeline.diarize(audio: samples, sampleRate: 16000)
/// for seg in result.segments {
///     print("Speaker \(seg.speakerId): [\(seg.startTime)s - \(seg.endTime)s]")
/// }
/// ```
public final class DiarizationPipeline {

    /// Pyannote segmentation model
    let segmentationModel: SegmentationModel

    /// Segmentation config
    let segConfig: SegmentationConfig

    /// WeSpeaker embedding model
    public let embeddingModel: WeSpeakerModel

    /// Optional Silero VAD for pre-filtering non-speech
    let vadModel: SileroVADModel?

    init(
        segmentationModel: SegmentationModel,
        segConfig: SegmentationConfig,
        embeddingModel: WeSpeakerModel,
        vadModel: SileroVADModel? = nil
    ) {
        self.segmentationModel = segmentationModel
        self.segConfig = segConfig
        self.embeddingModel = embeddingModel
        self.vadModel = vadModel
    }

    /// Load pre-trained models for diarization.
    ///
    /// Downloads both the pyannote segmentation model and WeSpeaker embedding model.
    ///
    /// - Parameters:
    ///   - segModelId: HuggingFace model ID for segmentation
    ///   - embModelId: HuggingFace model ID for speaker embeddings (auto-selected by engine if nil)
    ///   - embeddingEngine: inference backend for speaker embeddings (`.mlx` or `.coreml`)
    ///   - progressHandler: callback for download progress
    /// - Returns: ready-to-use diarization pipeline
    public static func fromPretrained(
        segModelId: String = PyannoteVADModel.defaultModelId,
        embModelId: String? = nil,
        embeddingEngine: WeSpeakerEngine = .mlx,
        useVADFilter: Bool = false,
        progressHandler: ((Double, String) -> Void)? = nil
    ) async throws -> DiarizationPipeline {
        progressHandler?(0.0, "Downloading segmentation model...")

        // Load segmentation model
        let segCacheDir = try HuggingFaceDownloader.getCacheDirectory(for: segModelId)
        try await HuggingFaceDownloader.downloadWeights(
            modelId: segModelId,
            to: segCacheDir,
            progressHandler: { progress in
                progressHandler?(progress * 0.3, "Downloading segmentation weights...")
            }
        )

        let segConfig = SegmentationConfig.default
        let segModel = SegmentationModel(config: segConfig)
        try SegmentationWeightLoader.loadWeights(model: segModel, from: segCacheDir)

        progressHandler?(0.3, "Downloading speaker embedding model...")

        // Load embedding model
        let embModel = try await WeSpeakerModel.fromPretrained(
            modelId: embModelId,
            engine: embeddingEngine,
            progressHandler: { progress, status in
                progressHandler?(0.3 + progress * 0.4, status)
            }
        )

        // Optionally load Silero VAD for pre-filtering
        var vadModel: SileroVADModel? = nil
        if useVADFilter {
            progressHandler?(0.7, "Downloading VAD filter model...")
            vadModel = try await SileroVADModel.fromPretrained(
                engine: .mlx,
                progressHandler: { progress, status in
                    progressHandler?(0.7 + progress * 0.25, status)
                }
            )
        }

        progressHandler?(1.0, "Ready")

        return DiarizationPipeline(
            segmentationModel: segModel,
            segConfig: segConfig,
            embeddingModel: embModel,
            vadModel: vadModel
        )
    }

    /// Run speaker diarization on audio.
    ///
    /// - Parameters:
    ///   - audio: PCM Float32 audio samples
    ///   - sampleRate: sample rate of the input audio
    ///   - config: diarization configuration
    /// - Returns: diarization result with speaker-labeled segments
    public func diarize(
        audio: [Float],
        sampleRate: Int,
        config: DiarizationConfig = .default
    ) -> DiarizationResult {
        let samples: [Float]
        if sampleRate != segConfig.sampleRate {
            samples = resample(audio, from: sampleRate, to: segConfig.sampleRate)
        } else {
            samples = audio
        }

        // Stage 0 (optional): VAD pre-filter — mask non-speech to reduce false alarms
        let speechMask: [SpeechSegment]?
        if let vadModel {
            speechMask = vadModel.detectSpeech(
                audio: samples, sampleRate: segConfig.sampleRate)
        } else {
            speechMask = nil
        }

        if let speechMask, speechMask.isEmpty {
            return DiarizationResult(segments: [], numSpeakers: 0, speakerEmbeddings: [])
        }

        // Stage 1: Per-window segmentation with probability-based speaker chaining
        let (segments, numSpeakers) = runActivityChainedDiarization(
            samples: samples, config: config, speechMask: speechMask)

        guard !segments.isEmpty else {
            return DiarizationResult(segments: [], numSpeakers: 0, speakerEmbeddings: [])
        }

        let merged = mergeAdjacentSegments(
            segments, minSilence: config.minSilenceDuration)

        // Stage 2: Compute per-speaker embeddings from final segments
        var speakerEmbeddings = [[Float]]()
        for spk in 0..<numSpeakers {
            var spkAudio = [Float]()
            for seg in merged where seg.speakerId == spk {
                let s = max(0, Int(seg.startTime * Float(segConfig.sampleRate)))
                let e = min(samples.count, Int(seg.endTime * Float(segConfig.sampleRate)))
                if e > s { spkAudio.append(contentsOf: samples[s..<e]) }
            }
            if spkAudio.count >= segConfig.sampleRate / 4 {
                speakerEmbeddings.append(embeddingModel.embed(
                    audio: spkAudio, sampleRate: segConfig.sampleRate))
            } else {
                speakerEmbeddings.append([Float](repeating: 0, count: 256))
            }
        }

        return DiarizationResult(
            segments: merged,
            numSpeakers: numSpeakers,
            speakerEmbeddings: speakerEmbeddings
        )
    }

    /// Extract segments of a target speaker from audio.
    ///
    /// Given a reference embedding (from `WeSpeakerModel.embed()`), finds the
    /// speaker with highest cosine similarity and returns their segments.
    ///
    /// - Parameters:
    ///   - audio: PCM Float32 audio samples
    ///   - sampleRate: sample rate of the input audio
    ///   - targetEmbedding: 256-dim reference embedding of the target speaker
    ///   - config: diarization configuration
    /// - Returns: speech segments belonging to the target speaker
    public func extractSpeaker(
        audio: [Float],
        sampleRate: Int,
        targetEmbedding: [Float],
        config: DiarizationConfig = .default
    ) -> [SpeechSegment] {
        let result = diarize(audio: audio, sampleRate: sampleRate, config: config)

        guard result.numSpeakers > 0 else { return [] }

        // Find speaker with highest cosine similarity to target
        var bestSpeaker = 0
        var bestSimilarity: Float = -1

        for (i, centroid) in result.speakerEmbeddings.enumerated() {
            let sim = WeSpeakerModel.cosineSimilarity(centroid, targetEmbedding)
            if sim > bestSimilarity {
                bestSimilarity = sim
                bestSpeaker = i
            }
        }

        return result.segments
            .filter { $0.speakerId == bestSpeaker }
            .map { SpeechSegment(startTime: $0.startTime, endTime: $0.endTime) }
    }

    // MARK: - Activity-Based Speaker Chaining

    /// Per-window raw probability tracks (3 speakers × nFrames).
    private struct WindowProbs {
        let startSample: Int
        let endSample: Int
        /// Speaker probability tracks [3][nFrames]
        let tracks: [[Float]]
    }

    /// Run diarization by chaining speakers across windows using activity correlation
    /// in the overlap region (no embedding-based matching).
    ///
    /// With 50% overlapping windows, adjacent windows share a half-window overlap region.
    /// The activity patterns of the 3 local speakers in this overlap are compared via
    /// Pearson correlation to find the best permutation mapping between windows.
    private func runActivityChainedDiarization(
        samples: [Float],
        config: DiarizationConfig,
        speechMask: [SpeechSegment]?
    ) -> (segments: [DiarizedSegment], numSpeakers: Int) {
        let windowDuration: Float = 10.0
        let sampleRate = segConfig.sampleRate
        let windowSamples = Int(windowDuration * Float(sampleRate))
        let framesPerChunk = 589
        let frameDuration = windowDuration / Float(framesPerChunk)
        let stepSamples = windowSamples / 2  // 50% overlap
        let halfFrames = framesPerChunk / 2  // ~294 frames in overlap

        let numSamples = samples.count
        guard numSamples > 0 else { return ([], 0) }

        // Generate window positions with 50% overlap
        var positions = [(start: Int, end: Int)]()
        if numSamples <= windowSamples {
            positions.append((0, numSamples))
        } else {
            var start = 0
            while start + windowSamples <= numSamples {
                positions.append((start, start + windowSamples))
                start += stepSamples
            }
            if positions.isEmpty || positions.last!.end < numSamples {
                positions.append((numSamples - windowSamples, numSamples))
            }
        }

        // Step 1: Run segmentation on all windows, collect probability tracks
        var windowProbs = [WindowProbs]()

        for (_, (start, end)) in positions.enumerated() {
            var window = Array(samples[start..<end])
            if window.count < windowSamples {
                window.append(contentsOf: [Float](repeating: 0, count: windowSamples - window.count))
            }

            let input = MLXArray(window).reshaped(1, 1, windowSamples)
            let posteriors = segmentationModel(input)
            let speakerProbs = PowersetDecoder.speakerProbabilities(from: posteriors)
            eval(speakerProbs)

            var tracks = [[Float]]()
            for spk in 0..<3 {
                tracks.append(speakerProbs[0, 0..., spk].asArray(Float.self))
            }
            windowProbs.append(WindowProbs(startSample: start, endSample: end, tracks: tracks))
        }

        guard !windowProbs.isEmpty else { return ([], 0) }

        // Step 2: Chain speakers across adjacent windows using overlap correlation
        // globalPermutation[w][localSpk] = global speaker ID for window w, local speaker localSpk
        var globalPermutation = [[Int]](repeating: [0, 1, 2], count: windowProbs.count)
        var maxGlobalSpk = 2  // max global speaker ID used so far

        for w in 1..<windowProbs.count {
            let prev = windowProbs[w - 1]
            let curr = windowProbs[w]

            // The overlap region: second half of prev == first half of curr
            // prev frames [halfFrames...589) overlap with curr frames [0...halfFrames)
            let overlapFrames = min(halfFrames, prev.tracks[0].count - halfFrames,
                                     curr.tracks[0].count)
            guard overlapFrames > 10 else {
                // No meaningful overlap — assign new IDs
                globalPermutation[w] = [maxGlobalSpk + 1, maxGlobalSpk + 2, maxGlobalSpk + 3]
                maxGlobalSpk += 3
                continue
            }

            // Determine which tracks are "active" (have meaningful speech) in the overlap
            let activityThreshold: Float = 0.05  // mean prob > 5% → active
            var prevActive = [Int]()
            var currActive = [Int]()
            var prevSlices = [[Float]](repeating: [], count: 3)
            var currSlices = [[Float]](repeating: [], count: 3)

            for p in 0..<3 {
                let slice = Array(prev.tracks[p][halfFrames..<(halfFrames + overlapFrames)])
                prevSlices[p] = slice
                let mean = slice.reduce(0, +) / Float(overlapFrames)
                if mean > activityThreshold { prevActive.append(p) }
            }
            for c in 0..<3 {
                let slice = Array(curr.tracks[c][0..<overlapFrames])
                currSlices[c] = slice
                let mean = slice.reduce(0, +) / Float(overlapFrames)
                if mean > activityThreshold { currActive.append(c) }
            }

            let prevMapping = globalPermutation[w - 1]
            var currMapping = [Int](repeating: -1, count: 3)

            if prevActive.isEmpty || currActive.isEmpty {
                // No active speakers in overlap — preserve order from previous window
                currMapping = prevMapping
            } else {
                // Compute correlation matrix only for active tracks
                // Then use greedy exclusive matching
                var pairs = [(corr: Float, prevLocal: Int, currLocal: Int)]()
                for p in prevActive {
                    for c in currActive {
                        let corr = activityCorrelation(prevSlices[p], currSlices[c])
                        pairs.append((corr, p, c))
                    }
                }
                pairs.sort { $0.corr > $1.corr }

                var usedPrev = Set<Int>()
                var usedCurr = Set<Int>()
                for (_, p, c) in pairs {
                    if usedPrev.contains(p) || usedCurr.contains(c) { continue }
                    currMapping[c] = prevMapping[p]
                    usedPrev.insert(p)
                    usedCurr.insert(c)
                }

                // Inactive curr tracks: assign to unmatched prev globals (by order)
                var unmatchedPrevGlobals = [Int]()
                for p in 0..<3 {
                    if !usedPrev.contains(p) {
                        unmatchedPrevGlobals.append(prevMapping[p])
                    }
                }
                var unmatchedIdx = 0
                for c in 0..<3 {
                    if currMapping[c] < 0 {
                        if unmatchedIdx < unmatchedPrevGlobals.count {
                            currMapping[c] = unmatchedPrevGlobals[unmatchedIdx]
                            unmatchedIdx += 1
                        } else {
                            maxGlobalSpk += 1
                            currMapping[c] = maxGlobalSpk
                        }
                    }
                }
            }

            globalPermutation[w] = currMapping
        }

        // Step 3: Binarize and build segments with global speaker IDs
        var diarizedSegments = [DiarizedSegment]()
        let numGlobalSpeakers = maxGlobalSpk + 1

        for (w, wp) in windowProbs.enumerated() {
            let windowStartTime = Float(wp.startSample) / Float(sampleRate)
            let windowEndTime = Float(wp.endSample) / Float(sampleRate)

            // Center zone ownership
            let prevEnd = w > 0 ?
                Float(positions[w - 1].end) / Float(sampleRate) : 0
            let nextStart = w + 1 < positions.count ?
                Float(positions[w + 1].start) / Float(sampleRate) : Float(numSamples) / Float(sampleRate)
            let ownStart = w > 0 ? (windowStartTime + prevEnd) / 2 : 0
            let ownEnd = w + 1 < positions.count ?
                (windowEndTime + nextStart) / 2 : Float(numSamples) / Float(sampleRate)

            for localSpk in 0..<3 {
                let globalSpk = globalPermutation[w][localSpk]
                let probs = wp.tracks[localSpk]
                let segments = PowersetDecoder.binarize(
                    probs: probs, onset: config.onset,
                    offset: config.offset, frameDuration: frameDuration)

                for seg in segments {
                    let absStart = windowStartTime + seg.startTime
                    let absEnd = min(windowStartTime + seg.endTime, windowEndTime)

                    // Clip to center zone
                    let clippedStart = max(absStart, ownStart)
                    let clippedEnd = min(absEnd, ownEnd)
                    guard clippedEnd - clippedStart >= config.minSpeechDuration else { continue }

                    // Apply VAD mask if present
                    if let speechMask {
                        if let trimmed = trimToSpeechMask(
                            start: clippedStart, end: clippedEnd,
                            speechRegions: speechMask,
                            minDuration: config.minSpeechDuration
                        ) {
                            diarizedSegments.append(DiarizedSegment(
                                startTime: trimmed.startTime,
                                endTime: trimmed.endTime,
                                speakerId: globalSpk
                            ))
                        }
                    } else {
                        diarizedSegments.append(DiarizedSegment(
                            startTime: clippedStart,
                            endTime: clippedEnd,
                            speakerId: globalSpk
                        ))
                    }
                }
            }
        }

        diarizedSegments.sort { $0.startTime < $1.startTime }

        // Compact speaker IDs (remove gaps)
        let usedIds = Set(diarizedSegments.map(\.speakerId)).sorted()
        let idMap = Dictionary(uniqueKeysWithValues: usedIds.enumerated().map { ($1, $0) })
        diarizedSegments = diarizedSegments.map {
            DiarizedSegment(startTime: $0.startTime, endTime: $0.endTime,
                           speakerId: idMap[$0.speakerId] ?? $0.speakerId)
        }

        return (diarizedSegments, usedIds.count)
    }

    /// Correlation between two activity probability vectors.
    /// Returns value in [-1, 1]; high correlation means similar activity patterns.
    private func activityCorrelation(_ a: [Float], _ b: [Float]) -> Float {
        let n = min(a.count, b.count)
        guard n > 1 else { return 0 }

        var sumA: Float = 0, sumB: Float = 0
        for i in 0..<n { sumA += a[i]; sumB += b[i] }
        let meanA = sumA / Float(n)
        let meanB = sumB / Float(n)

        var cov: Float = 0, varA: Float = 0, varB: Float = 0
        for i in 0..<n {
            let da = a[i] - meanA
            let db = b[i] - meanB
            cov += da * db
            varA += da * da
            varB += db * db
        }

        let denom = sqrt(varA * varB)
        return denom > 1e-10 ? cov / denom : 0
    }


    /// Trim a segment to intersect with speech regions.
    private func trimToSpeechMask(
        start: Float, end: Float,
        speechRegions: [SpeechSegment],
        minDuration: Float
    ) -> (startTime: Float, endTime: Float)? {
        let segDuration = end - start
        guard segDuration > 0 else { return nil }

        var totalOverlap: Float = 0
        var trimStart: Float = end
        var trimEnd: Float = start

        for vad in speechRegions {
            let oStart = max(start, vad.startTime)
            let oEnd = min(end, vad.endTime)
            if oStart < oEnd {
                totalOverlap += oEnd - oStart
                trimStart = min(trimStart, oStart)
                trimEnd = max(trimEnd, oEnd)
            }
        }

        guard totalOverlap / segDuration >= 0.5,
              trimEnd - trimStart >= minDuration else { return nil }
        return (trimStart, trimEnd)
    }

    // MARK: - Segment Merging

    private func mergeAdjacentSegments(
        _ segments: [DiarizedSegment],
        minSilence: Float
    ) -> [DiarizedSegment] {
        guard !segments.isEmpty else { return [] }

        var merged = [DiarizedSegment]()
        var current = segments[0]

        for i in 1..<segments.count {
            let next = segments[i]
            let gap = next.startTime - current.endTime

            if next.speakerId == current.speakerId && gap < minSilence {
                current = DiarizedSegment(
                    startTime: current.startTime,
                    endTime: next.endTime,
                    speakerId: current.speakerId
                )
            } else {
                merged.append(current)
                current = next
            }
        }
        merged.append(current)

        return merged
    }

    // MARK: - Resampling

    private func resample(_ audio: [Float], from sourceSR: Int, to targetSR: Int) -> [Float] {
        guard sourceSR != targetSR else { return audio }
        let ratio = Double(targetSR) / Double(sourceSR)
        let outputLen = Int(Double(audio.count) * ratio)
        var output = [Float](repeating: 0, count: outputLen)

        for i in 0..<outputLen {
            let srcPos = Double(i) / ratio
            let srcIdx = Int(srcPos)
            let frac = Float(srcPos - Double(srcIdx))

            if srcIdx + 1 < audio.count {
                output[i] = audio[srcIdx] * (1 - frac) + audio[srcIdx + 1] * frac
            } else if srcIdx < audio.count {
                output[i] = audio[srcIdx]
            }
        }

        return output
    }
}
