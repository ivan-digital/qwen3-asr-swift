import Foundation
import MLXCommon
import MLX
import AudioCommon

// MARK: - Configuration

/// Configuration for speaker diarization thresholds.
///
/// Shared by all diarization engines (Pyannote and Sortformer).
public struct DiarizationConfig: Sendable {
    /// Onset threshold for speaker activity
    public var onset: Float
    /// Offset threshold for speaker activity
    public var offset: Float
    /// Minimum speech segment duration in seconds
    public var minSpeechDuration: Float
    /// Minimum silence duration between segments in seconds
    public var minSilenceDuration: Float
    /// Cosine similarity threshold for merging speaker clusters (0.0-1.0)
    /// Higher = fewer merges (more speakers). Default 0.5.
    public var clusteringThreshold: Float

    public init(
        onset: Float = 0.5,
        offset: Float = 0.3,
        minSpeechDuration: Float = 0.3,
        minSilenceDuration: Float = 0.15,
        clusteringThreshold: Float = 1.0  // disabled by default (WeSpeaker similarities too high for conversational audio)
    ) {
        self.onset = onset
        self.offset = offset
        self.minSpeechDuration = minSpeechDuration
        self.minSilenceDuration = minSilenceDuration
        self.clusteringThreshold = clusteringThreshold
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

/// Pyannote-based speaker diarization: segmentation + activity-based speaker chaining.
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
/// let pipeline = try await PyannoteDiarizationPipeline.fromPretrained(useVADFilter: true)
/// let result = pipeline.diarize(audio: samples, sampleRate: 16000)
/// for seg in result.segments {
///     print("Speaker \(seg.speakerId): [\(seg.startTime)s - \(seg.endTime)s]")
/// }
/// ```
public final class PyannoteDiarizationPipeline {

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
    ) async throws -> PyannoteDiarizationPipeline {
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

        return PyannoteDiarizationPipeline(
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
        let samples = DiarizationHelpers.resample(audio, from: sampleRate, to: segConfig.sampleRate)

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

        let merged = DiarizationHelpers.mergeSegments(
            segments, minSilence: config.minSilenceDuration)

        // Stage 2: Compute per-speaker embeddings
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

        // Stage 3: Embedding-based agglomerative clustering
        // Merge speaker clusters with high cosine similarity to fix
        // speaker confusion from activity-based chaining.
        let (refinedSegments, refinedEmbeddings, refinedNumSpeakers) =
            refineWithEmbeddings(
                segments: merged,
                embeddings: speakerEmbeddings,
                numSpeakers: numSpeakers,
                threshold: config.clusteringThreshold)

        return DiarizationResult(
            segments: refinedSegments,
            numSpeakers: refinedNumSpeakers,
            speakerEmbeddings: refinedEmbeddings
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
    /// Refine speaker assignments using embedding-based agglomerative clustering.
    ///
    /// Activity-based chaining can produce fragmented speaker IDs (same person
    /// gets different IDs in different windows). This step computes cosine
    /// similarity between speaker embeddings and merges clusters above threshold.
    private func refineWithEmbeddings(
        segments: [DiarizedSegment],
        embeddings: [[Float]],
        numSpeakers: Int,
        threshold: Float
    ) -> ([DiarizedSegment], [[Float]], Int) {
        guard numSpeakers > 1 else {
            return (segments, embeddings, numSpeakers)
        }

        // Build speaker ID mapping: start with identity
        var idMap = [Int](0..<numSpeakers)

        // Agglomerative clustering: merge most similar pair until below threshold
        var active = Set(0..<numSpeakers)
        // Track merged embeddings (mean of merged clusters)
        var clusterEmbeddings = embeddings

        while active.count > 1 {
            // Find most similar pair
            var bestSim: Float = -1
            var bestI = -1, bestJ = -1

            let activeList = active.sorted()
            for ai in 0..<activeList.count {
                for aj in (ai + 1)..<activeList.count {
                    let i = activeList[ai], j = activeList[aj]
                    // Skip zero embeddings (insufficient audio)
                    let normI = clusterEmbeddings[i].reduce(Float(0)) { $0 + $1 * $1 }
                    let normJ = clusterEmbeddings[j].reduce(Float(0)) { $0 + $1 * $1 }
                    guard normI > 0.01 && normJ > 0.01 else { continue }

                    let sim = WeSpeakerModel.cosineSimilarity(
                        clusterEmbeddings[i], clusterEmbeddings[j])
                    if sim > bestSim {
                        bestSim = sim
                        bestI = i
                        bestJ = j
                    }
                }
            }

            // Stop if best similarity is below threshold
            guard bestSim >= threshold && bestI >= 0 else { break }

            // Merge bestJ into bestI
            // Update embedding: average of the two
            for d in 0..<clusterEmbeddings[bestI].count {
                clusterEmbeddings[bestI][d] = (clusterEmbeddings[bestI][d] + clusterEmbeddings[bestJ][d]) / 2.0
            }

            // Remap all segments with bestJ to bestI
            for k in 0..<numSpeakers {
                if idMap[k] == bestJ { idMap[k] = bestI }
            }

            active.remove(bestJ)
        }

        // Apply remapping and compact IDs
        let remapped = segments.map { seg in
            DiarizedSegment(
                startTime: seg.startTime,
                endTime: seg.endTime,
                speakerId: idMap[seg.speakerId])
        }

        let compacted = DiarizationHelpers.compactSpeakerIds(remapped)
        let finalNumSpeakers = Set(compacted.map(\.speakerId)).count

        // Collect final embeddings
        let finalEmbeddings: [[Float]]
        if finalNumSpeakers > 0 {
            var embMap = [[Float]](repeating: [], count: finalNumSpeakers)
            let uniqueIds = active.sorted()
            for (newId, oldId) in uniqueIds.enumerated() where newId < finalNumSpeakers {
                embMap[newId] = clusterEmbeddings[oldId]
            }
            finalEmbeddings = embMap
        } else {
            finalEmbeddings = []
        }

        return (compacted, finalEmbeddings, finalNumSpeakers)
    }

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
        diarizedSegments = DiarizationHelpers.compactSpeakerIds(diarizedSegments)
        let numCompacted = Set(diarizedSegments.map(\.speakerId)).count

        return (diarizedSegments, numCompacted)
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

}

/// Backwards-compatible alias for the renamed pipeline.
public typealias DiarizationPipeline = PyannoteDiarizationPipeline
