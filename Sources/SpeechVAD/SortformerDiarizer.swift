#if canImport(CoreML)
import CoreML
import Foundation
import AudioCommon

/// End-to-end neural speaker diarization using NVIDIA Sortformer (CoreML).
///
/// Sortformer directly predicts per-frame speaker activity for up to 4 speakers
/// without requiring separate embedding extraction or clustering. Runs on
/// Neural Engine at ~120x real-time.
///
/// ```swift
/// let diarizer = try await SortformerDiarizer.fromPretrained()
/// let result = diarizer.diarize(audio: samples, sampleRate: 16000)
/// for seg in result.segments {
///     print("Speaker \(seg.speakerId): [\(seg.startTime)s - \(seg.endTime)s]")
/// }
/// ```
public final class SortformerDiarizer {

    /// Default HuggingFace model ID for the CoreML Sortformer model
    public static let defaultModelId = "aufklarer/Sortformer-Diarization-CoreML"

    private let model: SortformerCoreMLModel
    private let melExtractor: SortformerMelExtractor
    let config: SortformerConfig

    /// Frame duration from model metadata (0.08s = 80ms per diarization frame)
    private let frameDuration: Float = 0.08

    // MARK: - Streaming State

    /// Speaker cache buffer, flat `[spkcacheLen * fcDModel]`
    private var spkcache: [Float]
    /// Number of valid frames in speaker cache
    private var spkcacheLength: Int = 0
    /// FIFO buffer, flat `[fifoLen * fcDModel]`
    private var fifo: [Float]
    /// Number of valid frames in FIFO
    private var fifoLength: Int = 0
    init(model: SortformerCoreMLModel, config: SortformerConfig = .default) {
        self.model = model
        self.config = config
        self.melExtractor = SortformerMelExtractor(config: config)
        self.spkcache = [Float](repeating: 0, count: config.spkcacheLen * config.fcDModel)
        self.fifo = [Float](repeating: 0, count: config.fifoLen * config.fcDModel)
    }

    /// Reset streaming state between different audio files.
    public func resetState() {
        spkcache = [Float](repeating: 0, count: config.spkcacheLen * config.fcDModel)
        spkcacheLength = 0
        fifo = [Float](repeating: 0, count: config.fifoLen * config.fcDModel)
        fifoLength = 0
    }

    // MARK: - Loading

    /// Load a pre-trained Sortformer model from HuggingFace.
    ///
    /// - Parameters:
    ///   - modelId: HuggingFace model ID
    ///   - progressHandler: callback for download progress
    /// - Returns: ready-to-use diarizer
    public static func fromPretrained(
        modelId: String = defaultModelId,
        progressHandler: ((Double, String) -> Void)? = nil
    ) async throws -> SortformerDiarizer {
        progressHandler?(0.0, "Downloading Sortformer model...")

        let cacheDir = try HuggingFaceDownloader.getCacheDirectory(for: modelId)

        try await HuggingFaceDownloader.downloadWeights(
            modelId: modelId,
            to: cacheDir,
            additionalFiles: ["Sortformer.mlmodelc/**", "config.json"],
            progressHandler: { progress in
                progressHandler?(progress * 0.8, "Downloading Sortformer model...")
            }
        )

        progressHandler?(0.8, "Loading CoreML model...")

        let modelURL = cacheDir.appendingPathComponent("Sortformer.mlmodelc", isDirectory: true)
        guard FileManager.default.fileExists(atPath: modelURL.path) else {
            throw AudioModelError.modelLoadFailed(
                modelId: modelId,
                reason: "CoreML model not found at \(modelURL.path)")
        }

        let mlConfig = MLModelConfiguration()
        mlConfig.computeUnits = .cpuAndNeuralEngine

        let mlModel: MLModel
        do {
            mlModel = try MLModel(contentsOf: modelURL, configuration: mlConfig)
        } catch {
            throw AudioModelError.modelLoadFailed(
                modelId: modelId,
                reason: "Failed to load CoreML model",
                underlying: error)
        }

        let config = SortformerConfig.default
        let coremlModel = SortformerCoreMLModel(model: mlModel, config: config)

        progressHandler?(1.0, "Ready")
        return SortformerDiarizer(model: coremlModel, config: config)
    }

    // MARK: - Diarization

    /// Run speaker diarization on complete audio.
    ///
    /// Processes audio in streaming chunks matching NeMo's streaming_feat_loader:
    /// each chunk is 112 mel frames = (leftCtx + coreChunk + rightCtx) × subsampling.
    /// Core predictions are extracted per chunk and concatenated.
    ///
    /// - Parameters:
    ///   - audio: PCM Float32 audio samples
    ///   - sampleRate: sample rate of the input audio
    ///   - config: optional override for diarization thresholds
    /// - Returns: diarization result with speaker-labeled segments
    public func diarize(
        audio: [Float],
        sampleRate: Int,
        config: DiarizationConfig = .default
    ) -> DiarizationResult {
        let samples = DiarizationHelpers.resample(audio, from: sampleRate, to: self.config.sampleRate)

        guard !samples.isEmpty else {
            return DiarizationResult(segments: [], numSpeakers: 0, speakerEmbeddings: [])
        }

        resetState()

        // Extract mel features for the entire audio: [totalMelFrames, 128]
        let (melSpec, totalMelFrames) = melExtractor.extract(samples)

        guard totalMelFrames > 0 else {
            return DiarizationResult(segments: [], numSpeakers: 0, speakerEmbeddings: [])
        }

        // Streaming chunking parameters (matching NeMo)
        let subFactor = self.config.subsamplingFactor
        let chunkLen = Int(self.config.chunkLenSeconds)  // 6 encoder output frames
        let leftCtx = Int(self.config.leftContextSeconds)  // 1
        let rightCtx = Int(self.config.rightContextSeconds) // 7
        let coreMelFrames = chunkLen * subFactor  // 48 mel frames per core chunk
        let coreMLInputFrames = 112  // Fixed CoreML input size
        let nMels = self.config.nMels
        let numSpeakers = self.config.maxSpeakers

        // Collect core predictions from each chunk
        var allChunkProbs = [[Float]]()  // Each entry: [coreFrames * numSpeakers]

        var sttFeat = 0
        var endFeat = 0

        while endFeat < totalMelFrames {
            let leftOffset = min(leftCtx * subFactor, sttFeat)
            endFeat = min(sttFeat + coreMelFrames, totalMelFrames)
            let rightOffset = min(rightCtx * subFactor, totalMelFrames - endFeat)

            let chunkStart = sttFeat - leftOffset
            let chunkEnd = endFeat + rightOffset
            let actualLen = chunkEnd - chunkStart

            // Build padded mel chunk [coreMLInputFrames, nMels]
            var chunkMel = [Float](repeating: 0, count: coreMLInputFrames * nMels)
            let framesToCopy = min(actualLen, coreMLInputFrames)
            for fi in 0..<framesToCopy {
                let srcBase = (chunkStart + fi) * nMels
                let dstBase = fi * nMels
                for di in 0..<nMels {
                    chunkMel[dstBase + di] = melSpec[srcBase + di]
                }
            }

            do {
                let output = try model.predict(
                    chunk: chunkMel,
                    chunkLength: actualLen,
                    spkcache: spkcache,
                    spkcacheLength: spkcacheLength,
                    fifo: fifo,
                    fifoLength: fifoLength
                )

                // Extract core predictions (skip spkcache + fifo + left context,
                // trim right context)
                let validEmbs: Int = output.validEmbFrames
                let lcFrames: Int = Int(Float(leftOffset) / Float(subFactor) + 0.5)
                let rcFrames: Int = Int(ceil(Float(rightOffset) / Float(subFactor)))
                let coreLen: Int = validEmbs - lcFrames - rcFrames
                let corePredLen = coreLen > 0 ? coreLen : 0

                let predOffset = spkcacheLength + fifoLength + lcFrames
                let totalPredFrames = output.predsFrames

                var chunkProbs = [Float]()
                for f in 0..<corePredLen {
                    let predFrame = predOffset + f
                    guard predFrame < totalPredFrames else { break }
                    for s in 0..<numSpeakers {
                        chunkProbs.append(output.pred(frame: predFrame, speaker: s))
                    }
                }
                allChunkProbs.append(chunkProbs)

                // Update streaming state (FIFO overflow → spkcache)
                updateState(from: output)
            } catch {
                print("Warning: Sortformer inference failed on chunk at mel frame \(sttFeat): \(error)")
            }

            sttFeat = endFeat
        }

        guard !allChunkProbs.isEmpty else {
            return DiarizationResult(segments: [], numSpeakers: 0, speakerEmbeddings: [])
        }

        // Concatenate all core predictions
        let audioDuration = Float(samples.count) / Float(self.config.sampleRate)
        let segments = binarizeCorePredictions(
            allChunkProbs: allChunkProbs,
            audioDuration: audioDuration,
            numSpeakers: numSpeakers,
            onset: config.onset,
            offset: config.offset,
            minSpeechDuration: config.minSpeechDuration,
            minSilenceDuration: config.minSilenceDuration
        )

        let usedSpeakers = Set(segments.map(\.speakerId))
        return DiarizationResult(
            segments: segments,
            numSpeakers: usedSpeakers.count,
            speakerEmbeddings: []  // End-to-end model, no separate embeddings
        )
    }

    // MARK: - State Management (NeMo FIFO→spkcache pattern)

    /// Update spkcache and fifo buffers from encoder embeddings.
    ///
    /// Follows NeMo's streaming_update: new embeddings go into FIFO.
    /// When FIFO overflows, oldest frames move to spkcache.
    private func updateState(from output: SortformerOutput) {
        let validFrames = output.validEmbFrames
        guard validFrames > 0 else { return }
        let dim = config.fcDModel
        let fifoCapacity = config.fifoLen
        let cacheCapacity = config.spkcacheLen

        if fifoLength + validFrames <= fifoCapacity {
            // FIFO has room — just append
            for f in 0..<validFrames {
                let srcBase = f * dim
                let dstBase = (fifoLength + f) * dim
                for d in 0..<dim {
                    fifo[dstBase + d] = output.encoderEmbs[srcBase + d]
                }
            }
            fifoLength += validFrames
        } else {
            // FIFO overflow: move oldest frames to spkcache
            let overflow = fifoLength + validFrames - fifoCapacity

            // Move overflow frames from front of FIFO to spkcache
            if spkcacheLength + overflow <= cacheCapacity {
                // Append to spkcache
                for f in 0..<overflow {
                    let srcBase = f * dim
                    let dstBase = (spkcacheLength + f) * dim
                    for d in 0..<dim {
                        spkcache[dstBase + d] = fifo[srcBase + d]
                    }
                }
                spkcacheLength += overflow
            } else {
                // Spkcache also overflows — shift left and append
                let cacheOverflow = spkcacheLength + overflow - cacheCapacity
                let keep = spkcacheLength - cacheOverflow
                if keep > 0 {
                    for f in 0..<keep {
                        let srcBase = (f + cacheOverflow) * dim
                        let dstBase = f * dim
                        for d in 0..<dim {
                            spkcache[dstBase + d] = spkcache[srcBase + d]
                        }
                    }
                }
                for f in 0..<overflow {
                    let srcBase = f * dim
                    let dstBase = (keep + f) * dim
                    for d in 0..<dim {
                        spkcache[dstBase + d] = fifo[srcBase + d]
                    }
                }
                spkcacheLength = min(cacheCapacity, keep + overflow)
            }

            // Shift FIFO left by overflow, then append new frames
            let remaining = fifoLength - overflow
            if remaining > 0 {
                for f in 0..<remaining {
                    let srcBase = (f + overflow) * dim
                    let dstBase = f * dim
                    for d in 0..<dim {
                        fifo[dstBase + d] = fifo[srcBase + d]
                    }
                }
            }
            fifoLength = remaining
            for f in 0..<validFrames {
                let srcBase = f * dim
                let dstBase = (fifoLength + f) * dim
                for d in 0..<dim {
                    fifo[dstBase + d] = output.encoderEmbs[srcBase + d]
                }
            }
            fifoLength += validFrames
        }
    }

    // MARK: - Binarization

    /// Concatenate per-chunk core predictions and binarize into segments.
    private func binarizeCorePredictions(
        allChunkProbs: [[Float]],
        audioDuration: Float,
        numSpeakers: Int,
        onset: Float,
        offset: Float,
        minSpeechDuration: Float,
        minSilenceDuration: Float
    ) -> [DiarizedSegment] {
        // Concatenate all chunk predictions into one flat array
        var allProbs = [Float]()
        for chunkProbs in allChunkProbs {
            allProbs.append(contentsOf: chunkProbs)
        }

        let totalFrames = allProbs.count / numSpeakers
        guard totalFrames > 0 else { return [] }

        // Apply sigmoid if predictions are logits
        for i in 0..<allProbs.count {
            if allProbs[i] > 1.0 || allProbs[i] < 0.0 {
                allProbs[i] = 1.0 / (1.0 + exp(-allProbs[i]))
            }
        }

        // Binarize each speaker track
        var allSegments = [DiarizedSegment]()

        for spk in 0..<numSpeakers {
            var probs = [Float](repeating: 0, count: totalFrames)
            for f in 0..<totalFrames {
                probs[f] = allProbs[f * numSpeakers + spk]
            }

            let rawSegments = PowersetDecoder.binarize(
                probs: probs,
                onset: onset,
                offset: offset,
                frameDuration: frameDuration
            )

            for seg in rawSegments {
                let duration = seg.endTime - seg.startTime
                guard duration >= minSpeechDuration else { continue }
                allSegments.append(DiarizedSegment(
                    startTime: seg.startTime,
                    endTime: min(seg.endTime, audioDuration),
                    speakerId: spk
                ))
            }
        }

        allSegments.sort { $0.startTime < $1.startTime }
        let merged = DiarizationHelpers.mergeSegments(allSegments, minSilence: minSilenceDuration)
        return DiarizationHelpers.compactSpeakerIds(merged)
    }
}

// MARK: - SpeakerDiarizationModel

extension SortformerDiarizer: SpeakerDiarizationModel {
    public var inputSampleRate: Int { config.sampleRate }

    public func diarize(audio: [Float], sampleRate: Int) -> [DiarizedSegment] {
        diarize(audio: audio, sampleRate: sampleRate, config: .default).segments
    }
}
#endif
