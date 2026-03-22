import CoreML
import Foundation
import AudioCommon

/// CoreML wrapper for Kokoro-82M end-to-end TTS inference.
///
/// Each model variant is a single end-to-end model that takes:
/// - `input_ids` [1, N]: phoneme token IDs
/// - `attention_mask` [1, N]: 1 for real tokens, 0 for padding
/// - `ref_s` [1, 256]: voice style embedding
/// - `random_phases` [1, 9]: random phases for iSTFTNet vocoder
///
/// And outputs:
/// - `audio` [1, 1, S]: generated waveform at 24kHz
/// - `audio_length_samples` [1]: actual valid sample count
/// - `pred_dur` [1, N]: predicted phoneme durations
class KokoroNetwork {

    private var models: [ModelBucket: MLModel]

    /// Load CoreML models from cache directory.
    ///
    /// - Parameters:
    ///   - directory: Path to cached model files
    ///   - computeUnits: CoreML compute units
    ///   - maxBuckets: Maximum number of buckets to load (smallest first).
    ///     Loading fewer buckets saves memory. Default 0 = load all available.
    init(directory: URL, computeUnits: MLComputeUnits = .all, maxBuckets: Int = 0) throws {
        let config = MLModelConfiguration()
        config.computeUnits = computeUnits

        var loaded = [ModelBucket: MLModel]()
        // Load smallest buckets first — they use least memory
        let sortedBuckets = ModelBucket.allCases.sorted { $0.maxTokens < $1.maxTokens }
        for bucket in sortedBuckets {
            if maxBuckets > 0 && loaded.count >= maxBuckets { break }
            let url = directory.appendingPathComponent("\(bucket.modelName).mlmodelc", isDirectory: true)
            if FileManager.default.fileExists(atPath: url.path) {
                loaded[bucket] = try MLModel(contentsOf: url, configuration: config)
            }
        }

        guard !loaded.isEmpty else {
            throw AudioModelError.modelLoadFailed(
                modelId: "kokoro",
                reason: "No Kokoro CoreML models found in \(directory.path)")
        }

        self.models = loaded
    }

    /// Run end-to-end TTS inference.
    ///
    /// - Parameters:
    ///   - inputIds: Phoneme token IDs [1, N]
    ///   - attentionMask: Attention mask [1, N]
    ///   - refS: Voice style embedding [1, 256]
    ///   - randomPhases: Random phases [1, 9]
    ///   - bucket: Which model variant to use
    /// - Returns: Inference output with audio, length, and predicted durations
    func predict(
        inputIds: MLMultiArray,
        attentionMask: MLMultiArray,
        refS: MLMultiArray,
        randomPhases: MLMultiArray,
        bucket: ModelBucket
    ) throws -> InferenceOutput {
        guard let model = models[bucket] else {
            throw AudioModelError.inferenceFailed(
                operation: "kokoro",
                reason: "No model loaded for bucket \(bucket.modelName)")
        }

        let input = try MLDictionaryFeatureProvider(dictionary: [
            "input_ids": MLFeatureValue(multiArray: inputIds),
            "attention_mask": MLFeatureValue(multiArray: attentionMask),
            "ref_s": MLFeatureValue(multiArray: refS),
            "random_phases": MLFeatureValue(multiArray: randomPhases),
        ])

        let output = try model.prediction(from: input)

        guard let audio = output.featureValue(for: "audio")?.multiArrayValue else {
            throw AudioModelError.inferenceFailed(
                operation: "kokoro", reason: "Missing audio output")
        }

        let audioLength = output.featureValue(for: "audio_length_samples")?.multiArrayValue
        let predDur = output.featureValue(for: "pred_dur")?.multiArrayValue

        return InferenceOutput(
            audio: audio,
            audioLengthSamples: audioLength,
            predictedDurations: predDur
        )
    }

    /// Available model buckets.
    var availableBuckets: [ModelBucket] {
        ModelBucket.allCases.filter { models[$0] != nil }
    }

    /// Inference output from the end-to-end model.
    struct InferenceOutput {
        /// Audio waveform [1, 1, S].
        let audio: MLMultiArray
        /// Actual valid sample count [1] (optional).
        let audioLengthSamples: MLMultiArray?
        /// Predicted phoneme durations [1, N] (optional).
        let predictedDurations: MLMultiArray?
    }
}
