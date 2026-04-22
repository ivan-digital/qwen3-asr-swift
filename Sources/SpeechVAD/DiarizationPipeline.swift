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
    /// Cosine distance threshold for merging speaker clusters (0.0-2.0).
    /// Lower = more merges (fewer speakers). Default 0.715.
    public var clusteringThreshold: Float

    public init(
        onset: Float = 0.5,
        offset: Float = 0.3,
        minSpeechDuration: Float = 0.3,
        minSilenceDuration: Float = 0.15,
        clusteringThreshold: Float = 0.715
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

/// Pyannote-based speaker diarization: segmentation + per-window embedding + constrained clustering.
///
/// - Warning: This class is not thread-safe. Create separate instances for concurrent use.
///
/// Pipeline (with optional VAD pre-filter):
/// 0. **VAD Pre-filter** (optional): Silero VAD masks non-speech regions → reduces false alarms
/// 1. **Segmentation**: Pyannote on 10s sliding windows (50% overlap) → per-speaker probability tracks
/// 2. **Per-window Embedding**: WeSpeaker 256-dim embedding per local speaker from non-overlapping speech
/// 3. **Constrained Clustering**: Agglomerative clustering with same-window constraint → global speaker IDs
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
        cacheBaseDir: URL? = nil,
        offlineMode: Bool = false,
        progressHandler: ((Double, String) -> Void)? = nil
    ) async throws -> PyannoteDiarizationPipeline {
        progressHandler?(0.0, "Downloading segmentation model...")

        // Load segmentation model
        let segCacheDir = try HuggingFaceDownloader.getCacheDirectory(for: segModelId, basePath: cacheBaseDir)
        try await HuggingFaceDownloader.downloadWeightsWithSourceSelection(
            modelId: segModelId,
            to: segCacheDir,
            offlineMode: offlineMode,
            progressHandler: { progress in
                progressHandler?(progress * 0.3, "Downloading segmentation weights...")
            }
        )

        let segConfig = SegmentationConfig.default
        let segModel = SegmentationModel(config: segConfig)
        try SegmentationWeightLoader.loadWeights(model: segModel, from: segCacheDir)

        progressHandler?(0.3, "Downloading speaker embedding model...")

        // Load embedding model
        let resolvedEmbModelId = embModelId ?? (embeddingEngine == .coreml ? WeSpeakerModel.defaultCoreMLModelId : WeSpeakerModel.defaultModelId)
        let embCacheDir: URL? = if let cacheBaseDir { try HuggingFaceDownloader.getCacheDirectory(for: resolvedEmbModelId, basePath: cacheBaseDir) } else { nil }
        let embModel = try await WeSpeakerModel.fromPretrained(
            modelId: embModelId,
            engine: embeddingEngine,
            cacheDir: embCacheDir,
            offlineMode: offlineMode,
            progressHandler: { progress, status in
                progressHandler?(0.3 + progress * 0.4, status)
            }
        )

        // Optionally load Silero VAD for pre-filtering
        var vadModel: SileroVADModel? = nil
        if useVADFilter {
            progressHandler?(0.7, "Downloading VAD filter model...")
            let vadCacheDir: URL? = if let cacheBaseDir { try HuggingFaceDownloader.getCacheDirectory(for: SileroVADModel.defaultModelId, basePath: cacheBaseDir) } else { nil }
            vadModel = try await SileroVADModel.fromPretrained(
                engine: .mlx,
                cacheDir: vadCacheDir,
                offlineMode: offlineMode,
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
        diarize(audio: audio, sampleRate: sampleRate, config: config, progressHandler: nil)
    }

    /// Diarize audio with progress reporting and optional cancellation.
    ///
    /// Same as `diarize(audio:sampleRate:config:)` but reports progress during
    /// the two most expensive stages (segmentation and embedding extraction).
    /// The handler returns a `Bool`: `true` to continue, `false` to cancel.
    /// When cancelled, an empty `DiarizationResult` is returned immediately.
    ///
    /// - Parameters:
    ///   - audio: PCM Float32 audio samples
    ///   - sampleRate: sample rate of the input audio
    ///   - config: diarization configuration
    ///   - progressHandler: called with (progress 0.0–1.0, stage description);
    ///     return `true` to continue or `false` to cancel
    /// - Returns: diarization result with speaker-labeled segments
    public func diarize(
        audio: [Float],
        sampleRate: Int,
        config: DiarizationConfig = .default,
        progressHandler: ((Float, String) -> Bool)?
    ) -> DiarizationResult {
        let emptyResult = DiarizationResult(segments: [], numSpeakers: 0, speakerEmbeddings: [])
        let samples = DiarizationHelpers.resample(audio, from: sampleRate, to: segConfig.sampleRate)

        // Stage 0 (optional): VAD pre-filter — mask non-speech to reduce false alarms
        let speechMask: [SpeechSegment]?
        if let vadModel {
            if progressHandler?(0, "VAD pre-filtering") == false {
                return emptyResult
            }
            speechMask = vadModel.detectSpeech(
                audio: samples, sampleRate: segConfig.sampleRate)
        } else {
            speechMask = nil
        }

        if let speechMask, speechMask.isEmpty {
            return emptyResult
        }

        // Run embedding-clustered diarization pipeline
        return runEmbeddingClusteredDiarization(
            samples: samples, config: config, speechMask: speechMask,
            progressHandler: progressHandler)
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

    // MARK: - Embedding-Clustered Diarization

    /// Per-window raw probability tracks (3 speakers × nFrames).
    private struct WindowProbs {
        let startSample: Int
        let endSample: Int
        /// Speaker probability tracks [3][nFrames]
        let tracks: [[Float]]
    }

    /// Per-window per-speaker embedding for clustering.
    private struct WindowSpeakerEmbedding {
        let windowIndex: Int
        let localSpeakerId: Int
        let embedding: [Float]
    }

    /// Run diarization using per-window speaker embeddings + constrained agglomerative clustering.
    ///
    /// 1. Segment all windows → per-speaker probability tracks
    /// 2. Extract per-window per-speaker embeddings from non-overlapping speech
    /// 3. Constrained clustering (same-window items never merge) → global speaker IDs
    /// 4. Map cluster IDs back to binarized segments
    private func runEmbeddingClusteredDiarization(
        samples: [Float],
        config: DiarizationConfig,
        speechMask: [SpeechSegment]?,
        progressHandler: ((Float, String) -> Bool)? = nil
    ) -> DiarizationResult {
        let windowDuration: Float = 10.0
        let sampleRate = segConfig.sampleRate
        let windowSamples = Int(windowDuration * Float(sampleRate))
        let framesPerChunk = 589
        let frameDuration = windowDuration / Float(framesPerChunk)
        let stepSamples = windowSamples / 2  // 50% overlap

        let numSamples = samples.count
        guard numSamples > 0 else {
            return DiarizationResult(segments: [], numSpeakers: 0, speakerEmbeddings: [])
        }

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
        // Progress: both steps iterate over all windows, so total = 2 * windowCount
        let totalUnits = positions.count * 2
        var completedUnits = 0
        var windowProbs = [WindowProbs]()

        let emptyResult = DiarizationResult(segments: [], numSpeakers: 0, speakerEmbeddings: [])

        for (posIdx, (start, end)) in positions.enumerated() {
            completedUnits += 1
            if progressHandler?(Float(completedUnits) / Float(totalUnits), "Segmenting \(posIdx + 1)/\(positions.count)") == false {
                return emptyResult
            }

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

        guard !windowProbs.isEmpty else {
            return DiarizationResult(segments: [], numSpeakers: 0, speakerEmbeddings: [])
        }

        // Step 2: Extract per-window per-speaker embeddings from non-overlapping speech
        let minEmbeddingSamples = sampleRate / 2  // 0.5s minimum for embedding
        var windowEmbeddings = [WindowSpeakerEmbedding]()

        for (wIdx, wp) in windowProbs.enumerated() {
            completedUnits += 1
            if progressHandler?(Float(completedUnits) / Float(totalUnits), "Embedding \(wIdx + 1)/\(windowProbs.count)") == false {
                return emptyResult
            }

            let windowStartSample = wp.startSample

            for localSpk in 0..<3 {
                let probs = wp.tracks[localSpk]

                // Binarize this speaker's track
                let binarySegments = PowersetDecoder.binarize(
                    probs: probs, onset: config.onset,
                    offset: config.offset, frameDuration: frameDuration)

                guard !binarySegments.isEmpty else { continue }

                // Collect audio from frames where ONLY this speaker is active (non-overlapping)
                var spkAudio = [Float]()

                for seg in binarySegments {
                    let segStartFrame = Int(seg.startTime / frameDuration)
                    let segEndFrame = min(Int(seg.endTime / frameDuration), probs.count)

                    for frame in segStartFrame..<segEndFrame {
                        // Check if other speakers are below offset at this frame
                        var otherActive = false
                        for otherSpk in 0..<3 where otherSpk != localSpk {
                            if wp.tracks[otherSpk][frame] >= config.offset {
                                otherActive = true
                                break
                            }
                        }
                        if otherActive { continue }

                        // Extract audio samples for this frame
                        let frameStartSample = windowStartSample + Int(Float(frame) * frameDuration * Float(sampleRate))
                        let frameEndSample = min(
                            windowStartSample + Int(Float(frame + 1) * frameDuration * Float(sampleRate)),
                            samples.count
                        )
                        if frameEndSample > frameStartSample {
                            spkAudio.append(contentsOf: samples[frameStartSample..<frameEndSample])
                        }
                    }
                }

                // Need minimum 0.5s of audio for a reliable embedding
                guard spkAudio.count >= minEmbeddingSamples else { continue }

                let embedding = embeddingModel.embed(audio: spkAudio, sampleRate: sampleRate)
                windowEmbeddings.append(WindowSpeakerEmbedding(
                    windowIndex: wIdx, localSpeakerId: localSpk, embedding: embedding))
            }
        }

        // Handle edge case: no embeddings could be extracted
        guard !windowEmbeddings.isEmpty else {
            return DiarizationResult(segments: [], numSpeakers: 0, speakerEmbeddings: [])
        }

        // Step 3: Constrained agglomerative clustering
        let clusterItems = windowEmbeddings.map {
            DiarizationHelpers.ClusterItem(
                windowIndex: $0.windowIndex,
                localSpeakerId: $0.localSpeakerId,
                embedding: $0.embedding)
        }

        let (clusterAssignment, centroids) = DiarizationHelpers.constrainedAgglomerativeClustering(
            items: clusterItems, threshold: config.clusteringThreshold)

        // Build mapping: (windowIndex, localSpeakerId) → global cluster ID
        var localToGlobal = [Int: [Int: Int]]()  // windowIndex → (localSpeakerId → globalId)
        for (i, we) in windowEmbeddings.enumerated() {
            localToGlobal[we.windowIndex, default: [:]][we.localSpeakerId] = clusterAssignment[i]
        }

        // Step 4: Build segments with global speaker IDs
        var diarizedSegments = [DiarizedSegment]()

        for (w, wp) in windowProbs.enumerated() {
            let windowStartTime = Float(wp.startSample) / Float(sampleRate)
            let windowEndTime = Float(wp.endSample) / Float(sampleRate)

            // Center zone ownership (same as before)
            let prevEnd = w > 0 ?
                Float(positions[w - 1].end) / Float(sampleRate) : 0
            let nextStart = w + 1 < positions.count ?
                Float(positions[w + 1].start) / Float(sampleRate) : Float(numSamples) / Float(sampleRate)
            let ownStart = w > 0 ? (windowStartTime + prevEnd) / 2 : 0
            let ownEnd = w + 1 < positions.count ?
                (windowEndTime + nextStart) / 2 : Float(numSamples) / Float(sampleRate)

            for localSpk in 0..<3 {
                guard let globalSpk = localToGlobal[w]?[localSpk] else {
                    continue  // No embedding for this speaker — skip (insufficient audio)
                }

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

        // Compact speaker IDs and merge
        diarizedSegments = DiarizationHelpers.compactSpeakerIds(diarizedSegments)
        let merged = DiarizationHelpers.mergeSegments(
            diarizedSegments, minSilence: config.minSilenceDuration)
        let numSpeakers = Set(merged.map(\.speakerId)).count

        // Re-compact centroids to match compacted speaker IDs
        let finalCentroids: [[Float]]
        if numSpeakers <= centroids.count {
            finalCentroids = Array(centroids.prefix(numSpeakers))
        } else {
            // Pad with zero embeddings for speakers that had no embedding
            var padded = centroids
            while padded.count < numSpeakers {
                padded.append([Float](repeating: 0, count: 256))
            }
            finalCentroids = padded
        }

        return DiarizationResult(
            segments: merged,
            numSpeakers: numSpeakers,
            speakerEmbeddings: finalCentroids
        )
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
