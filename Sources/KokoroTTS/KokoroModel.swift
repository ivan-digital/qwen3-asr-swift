import CoreML
import Foundation
import AudioCommon

/// CoreML wrapper for Kokoro-82M 3-stage TTS inference.
///
/// Stage 1 - Duration:  input_ids + attention_mask + ref_s + speed → pred_dur, d_transposed, t_en
/// Stage 2 - Prosody:   en + s → F0_pred, N_pred
/// Stage 3 - Decoder:   asr + F0_pred + N_pred + ref_s → audio
///
/// Between stages 1 and 2, Swift builds an alignment matrix from predicted durations.
class KokoroNetwork {

    private var durationModel: MLModel?
    private var prosodyModel: MLModel?
    private var decoderModels: [DecoderBucket: MLModel]

    /// Load CoreML models from cache directory.
    init(directory: URL, computeUnits: MLComputeUnits = .all) throws {
        let config = MLModelConfiguration()
        config.computeUnits = computeUnits

        // Load duration model
        let durURL = directory.appendingPathComponent("duration.mlmodelc", isDirectory: true)
        if FileManager.default.fileExists(atPath: durURL.path) {
            durationModel = try MLModel(contentsOf: durURL, configuration: config)
        }

        // Load prosody model
        let prosURL = directory.appendingPathComponent("prosody.mlmodelc", isDirectory: true)
        if FileManager.default.fileExists(atPath: prosURL.path) {
            prosodyModel = try MLModel(contentsOf: prosURL, configuration: config)
        }

        // Load decoder buckets
        var decoders = [DecoderBucket: MLModel]()
        for bucket in DecoderBucket.allCases {
            let url = directory.appendingPathComponent("\(bucket.modelName).mlmodelc", isDirectory: true)
            if FileManager.default.fileExists(atPath: url.path) {
                decoders[bucket] = try MLModel(contentsOf: url, configuration: config)
            }
        }
        decoderModels = decoders

        // Require at least the duration model
        guard durationModel != nil else {
            throw AudioModelError.modelLoadFailed(
                modelId: "kokoro",
                reason: "Duration model not found in \(directory.path)")
        }
    }

    /// Available decoder buckets.
    var availableDecoderBuckets: [DecoderBucket] {
        DecoderBucket.allCases.filter { decoderModels[$0] != nil }
    }

    /// Whether the 3-stage pipeline is available.
    var hasThreeStagePipeline: Bool {
        durationModel != nil && prosodyModel != nil && !decoderModels.isEmpty
    }

    // MARK: - Stage 1: Duration

    struct DurationOutput {
        let predDur: MLMultiArray     // [1, N] predicted durations
        let dTransposed: MLMultiArray // [1, 640, N] prosody features
        let tEn: MLMultiArray         // [1, 512, N] text encoding
    }

    func predictDuration(
        inputIds: MLMultiArray,
        attentionMask: MLMultiArray,
        refS: MLMultiArray,
        speed: MLMultiArray
    ) throws -> DurationOutput {
        guard let model = durationModel else {
            throw AudioModelError.inferenceFailed(
                operation: "kokoro-duration", reason: "Duration model not loaded")
        }

        let input = try MLDictionaryFeatureProvider(dictionary: [
            "input_ids": MLFeatureValue(multiArray: inputIds),
            "attention_mask": MLFeatureValue(multiArray: attentionMask),
            "ref_s": MLFeatureValue(multiArray: refS),
            "speed": MLFeatureValue(multiArray: speed),
        ])
        let output = try model.prediction(from: input)

        guard let predDur = output.featureValue(for: "pred_dur")?.multiArrayValue,
              let dTransposed = output.featureValue(for: "d_transposed")?.multiArrayValue,
              let tEn = output.featureValue(for: "t_en")?.multiArrayValue else {
            throw AudioModelError.inferenceFailed(
                operation: "kokoro-duration", reason: "Missing output tensors")
        }
        return DurationOutput(predDur: predDur, dTransposed: dTransposed, tEn: tEn)
    }

    // MARK: - Stage 2: Prosody

    struct ProsodyOutput {
        let f0Pred: MLMultiArray  // [1, F*2]
        let nPred: MLMultiArray   // [1, F*2]
    }

    func predictProsody(
        en: MLMultiArray,
        s: MLMultiArray
    ) throws -> ProsodyOutput {
        guard let model = prosodyModel else {
            throw AudioModelError.inferenceFailed(
                operation: "kokoro-prosody", reason: "Prosody model not loaded")
        }

        let input = try MLDictionaryFeatureProvider(dictionary: [
            "en": MLFeatureValue(multiArray: en),
            "s": MLFeatureValue(multiArray: s),
        ])
        let output = try model.prediction(from: input)

        guard let f0 = output.featureValue(for: "F0_pred")?.multiArrayValue,
              let n = output.featureValue(for: "N_pred")?.multiArrayValue else {
            throw AudioModelError.inferenceFailed(
                operation: "kokoro-prosody", reason: "Missing F0/N output")
        }
        return ProsodyOutput(f0Pred: f0, nPred: n)
    }

    // MARK: - Stage 3: Decoder

    func decode(
        asr: MLMultiArray,
        f0Pred: MLMultiArray,
        nPred: MLMultiArray,
        refS: MLMultiArray,
        bucket: DecoderBucket
    ) throws -> MLMultiArray {
        guard let model = decoderModels[bucket] else {
            throw AudioModelError.inferenceFailed(
                operation: "kokoro-decoder",
                reason: "No decoder model for bucket \(bucket.modelName)")
        }

        let input = try MLDictionaryFeatureProvider(dictionary: [
            "asr": MLFeatureValue(multiArray: asr),
            "F0_pred": MLFeatureValue(multiArray: f0Pred),
            "N_pred": MLFeatureValue(multiArray: nPred),
            "ref_s": MLFeatureValue(multiArray: refS),
        ])
        let output = try model.prediction(from: input)

        guard let audio = output.featureValue(for: "audio")?.multiArrayValue else {
            throw AudioModelError.inferenceFailed(
                operation: "kokoro-decoder", reason: "Missing audio output")
        }
        return audio
    }

    // MARK: - Legacy single-model support

    private var legacyModels: [ModelBucket: MLModel]?

    /// Load legacy end-to-end models (v2.1/v2.4).
    func loadLegacyModels(directory: URL, computeUnits: MLComputeUnits = .all) throws {
        let config = MLModelConfiguration()
        config.computeUnits = computeUnits

        var loaded = [ModelBucket: MLModel]()
        for bucket in ModelBucket.allCases {
            let url = directory.appendingPathComponent("\(bucket.modelName).mlmodelc", isDirectory: true)
            if FileManager.default.fileExists(atPath: url.path) {
                loaded[bucket] = try MLModel(contentsOf: url, configuration: config)
            }
        }
        if !loaded.isEmpty {
            legacyModels = loaded
        }
    }

    /// Available legacy model buckets.
    var availableBuckets: [ModelBucket] {
        ModelBucket.allCases.filter { legacyModels?[$0] != nil }
    }

    /// Legacy end-to-end prediction.
    func predict(
        inputIds: MLMultiArray,
        attentionMask: MLMultiArray,
        refS: MLMultiArray,
        randomPhases: MLMultiArray,
        bucket: ModelBucket
    ) throws -> LegacyInferenceOutput {
        guard let model = legacyModels?[bucket] else {
            throw AudioModelError.inferenceFailed(
                operation: "kokoro", reason: "No legacy model for bucket \(bucket.modelName)")
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
        return LegacyInferenceOutput(
            audio: audio,
            audioLengthSamples: output.featureValue(for: "audio_length_samples")?.multiArrayValue,
            predictedDurations: output.featureValue(for: "pred_dur")?.multiArrayValue
        )
    }

    struct LegacyInferenceOutput {
        let audio: MLMultiArray
        let audioLengthSamples: MLMultiArray?
        let predictedDurations: MLMultiArray?
    }
}
