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
    /// Clustering distance threshold (cosine distance, 0-2 range)
    public var clusteringThreshold: Float
    /// Maximum number of speakers (0 = automatic)
    public var maxSpeakers: Int

    public init(
        onset: Float = 0.5,
        offset: Float = 0.3,
        minSpeechDuration: Float = 0.3,
        minSilenceDuration: Float = 0.15,
        clusteringThreshold: Float = 0.5,
        maxSpeakers: Int = 0
    ) {
        self.onset = onset
        self.offset = offset
        self.minSpeechDuration = minSpeechDuration
        self.minSilenceDuration = minSilenceDuration
        self.clusteringThreshold = clusteringThreshold
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

/// Speaker diarization pipeline: segmentation → embedding → clustering.
///
/// - Warning: This class is not thread-safe. Create separate instances for concurrent use.
///
/// Three-stage pipeline:
/// 1. **Segmentation**: Pyannote segmentation on 10s sliding windows → per-speaker local segments
/// 2. **Embedding**: For each local segment, crop audio → mel → WeSpeaker → 256-dim embedding
/// 3. **Clustering**: Agglomerative clustering on embeddings → global speaker IDs
///
/// ```swift
/// let pipeline = try await DiarizationPipeline.fromPretrained()
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

    init(
        segmentationModel: SegmentationModel,
        segConfig: SegmentationConfig,
        embeddingModel: WeSpeakerModel
    ) {
        self.segmentationModel = segmentationModel
        self.segConfig = segConfig
        self.embeddingModel = embeddingModel
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
        progressHandler: ((Double, String) -> Void)? = nil
    ) async throws -> DiarizationPipeline {
        progressHandler?(0.0, "Downloading segmentation model...")

        // Load segmentation model
        let segCacheDir = try HuggingFaceDownloader.getCacheDirectory(for: segModelId)
        try await HuggingFaceDownloader.downloadWeights(
            modelId: segModelId,
            to: segCacheDir,
            progressHandler: { progress in
                progressHandler?(progress * 0.4, "Downloading segmentation weights...")
            }
        )

        let segConfig = SegmentationConfig.default
        let segModel = SegmentationModel(config: segConfig)
        try SegmentationWeightLoader.loadWeights(model: segModel, from: segCacheDir)

        progressHandler?(0.4, "Downloading speaker embedding model...")

        // Load embedding model
        let embModel = try await WeSpeakerModel.fromPretrained(
            modelId: embModelId,
            engine: embeddingEngine,
            progressHandler: { progress, status in
                progressHandler?(0.4 + progress * 0.5, status)
            }
        )

        progressHandler?(1.0, "Ready")

        return DiarizationPipeline(
            segmentationModel: segModel,
            segConfig: segConfig,
            embeddingModel: embModel
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

        // Stage 1: Segmentation — get per-speaker local segments from sliding windows
        let localSegments = runSegmentation(samples: samples, config: config)

        guard !localSegments.isEmpty else {
            return DiarizationResult(segments: [], numSpeakers: 0, speakerEmbeddings: [])
        }

        // Stage 2: Embedding — extract speaker embedding for each local segment
        var embeddings = [[Float]]()
        for seg in localSegments {
            let startSample = max(0, Int(seg.startTime * Float(segConfig.sampleRate)))
            let endSample = min(samples.count, Int(seg.endTime * Float(segConfig.sampleRate)))

            guard endSample > startSample else {
                embeddings.append([Float](repeating: 0, count: embeddingModel.embeddingDimension))
                continue
            }

            let cropAudio = Array(samples[startSample..<endSample])
            let emb = embeddingModel.embed(audio: cropAudio, sampleRate: segConfig.sampleRate)
            embeddings.append(emb)
        }

        // Stage 3: Clustering — assign global speaker IDs
        let (clusterIds, centroids) = agglomerativeClustering(
            embeddings: embeddings,
            threshold: config.clusteringThreshold,
            maxClusters: config.maxSpeakers > 0 ? config.maxSpeakers : Int.max
        )

        // Build diarized segments
        var diarizedSegments = [DiarizedSegment]()
        for (i, seg) in localSegments.enumerated() {
            diarizedSegments.append(DiarizedSegment(
                startTime: seg.startTime,
                endTime: seg.endTime,
                speakerId: clusterIds[i]
            ))
        }

        // Merge adjacent segments with same speaker and short gaps
        let merged = mergeAdjacentSegments(
            diarizedSegments,
            minSilence: config.minSilenceDuration
        )

        let numSpeakers = Set(clusterIds).count

        return DiarizationResult(
            segments: merged,
            numSpeakers: numSpeakers,
            speakerEmbeddings: centroids
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

    // MARK: - Stage 1: Segmentation

    private func runSegmentation(
        samples: [Float],
        config: DiarizationConfig
    ) -> [(startTime: Float, endTime: Float, localSpeaker: Int)] {
        let vadConfig = VADConfig.default
        let pipeline = VADPipeline(
            config: vadConfig,
            sampleRate: segConfig.sampleRate,
            framesPerChunk: 589
        )

        let positions = pipeline.windowPositions(numSamples: samples.count)
        guard !positions.isEmpty else { return [] }

        let windowSamples = Int(vadConfig.windowDuration * Float(segConfig.sampleRate))
        let frameDuration = vadConfig.windowDuration / Float(589)

        var allSegments = [(startTime: Float, endTime: Float, localSpeaker: Int)]()

        for (start, end) in positions {
            var window = Array(samples[start..<end])
            if window.count < windowSamples {
                window.append(contentsOf: [Float](repeating: 0, count: windowSamples - window.count))
            }

            // Run segmentation: [1, 1, samples] → [1, frames, 7]
            let input = MLXArray(window).reshaped(1, 1, windowSamples)
            let posteriors = segmentationModel(input)

            // Decode powerset to per-speaker: [1, frames, 3]
            let speakerProbs = PowersetDecoder.speakerProbabilities(from: posteriors)
            eval(speakerProbs)

            let windowStartTime = Float(start) / Float(segConfig.sampleRate)

            // Extract segments for each local speaker
            for spk in 0..<3 {
                let probs = speakerProbs[0, 0..., spk].asArray(Float.self)

                let segments = PowersetDecoder.binarize(
                    probs: probs,
                    onset: config.onset,
                    offset: config.offset,
                    frameDuration: frameDuration
                )

                for seg in segments {
                    let absStart = windowStartTime + seg.startTime
                    let absEnd = windowStartTime + seg.endTime
                    let duration = absEnd - absStart

                    if duration >= config.minSpeechDuration {
                        allSegments.append((absStart, absEnd, spk))
                    }
                }
            }
        }

        // Sort by start time
        return allSegments.sorted { $0.startTime < $1.startTime }
    }

    // MARK: - Stage 3: Agglomerative Clustering

    private func agglomerativeClustering(
        embeddings: [[Float]],
        threshold: Float,
        maxClusters: Int
    ) -> (clusterIds: [Int], centroids: [[Float]]) {
        let n = embeddings.count
        guard n > 0 else { return ([], []) }
        if n == 1 { return ([0], embeddings) }

        // Initialize: each embedding is its own cluster
        var clusterAssignment = Array(0..<n)
        var centroids = embeddings
        var activeClusterIds = Set(0..<n)

        while activeClusterIds.count > 1 {
            // Find closest pair of clusters
            var bestDist: Float = Float.infinity
            var bestI = -1
            var bestJ = -1

            let activeList = Array(activeClusterIds).sorted()
            for ai in 0..<activeList.count {
                for aj in (ai + 1)..<activeList.count {
                    let ci = activeList[ai]
                    let cj = activeList[aj]
                    let dist = cosineDistance(centroids[ci], centroids[cj])
                    if dist < bestDist {
                        bestDist = dist
                        bestI = ci
                        bestJ = cj
                    }
                }
            }

            // Stop if minimum distance exceeds threshold
            if bestDist > threshold { break }

            // Stop if we've reached max clusters
            if activeClusterIds.count <= maxClusters { break }

            // Merge bestJ into bestI
            let membersI = clusterAssignment.enumerated().filter { $0.element == bestI }.map(\.offset)
            let membersJ = clusterAssignment.enumerated().filter { $0.element == bestJ }.map(\.offset)

            // Update centroid (average linkage)
            let totalMembers = membersI.count + membersJ.count
            var newCentroid = [Float](repeating: 0, count: embeddings[0].count)
            for idx in membersI + membersJ {
                for d in 0..<newCentroid.count {
                    newCentroid[d] += embeddings[idx][d]
                }
            }
            for d in 0..<newCentroid.count {
                newCentroid[d] /= Float(totalMembers)
            }

            // L2 normalize centroid
            var norm: Float = 0
            for d in 0..<newCentroid.count { norm += newCentroid[d] * newCentroid[d] }
            norm = sqrt(norm + 1e-10)
            for d in 0..<newCentroid.count { newCentroid[d] /= norm }

            centroids[bestI] = newCentroid

            // Reassign members of bestJ to bestI
            for idx in membersJ {
                clusterAssignment[idx] = bestI
            }
            activeClusterIds.remove(bestJ)
        }

        // Remap cluster IDs to contiguous 0-based
        let uniqueClusters = Array(Set(clusterAssignment)).sorted()
        var remap = [Int: Int]()
        for (newId, oldId) in uniqueClusters.enumerated() {
            remap[oldId] = newId
        }

        let remappedIds = clusterAssignment.map { remap[$0]! }
        let finalCentroids = uniqueClusters.map { centroids[$0] }

        return (remappedIds, finalCentroids)
    }

    private func cosineDistance(_ a: [Float], _ b: [Float]) -> Float {
        return 1.0 - WeSpeakerModel.cosineSimilarity(a, b)
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
