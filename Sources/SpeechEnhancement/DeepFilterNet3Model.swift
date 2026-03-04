import Foundation
import CoreML

/// Core ML wrapper for the DeepFilterNet3 neural network.
///
/// Runs the encoder + ERB decoder + DF decoder via Core ML.
/// Input: normalized ERB and spec features (computed by the signal processing pipeline).
/// Output: ERB gain mask and deep filter coefficients.
class DeepFilterNet3Network {

    private let model: MLModel

    /// Load a compiled Core ML model from a .mlpackage or .mlmodelc directory.
    init(modelURL: URL, computeUnits: MLComputeUnits = .all) throws {
        let config = MLModelConfiguration()
        config.computeUnits = computeUnits

        // Check for .mlmodelc (compiled) first, then compile from .mlpackage
        let compiledURL: URL
        if modelURL.pathExtension == "mlmodelc" {
            compiledURL = modelURL
        } else {
            compiledURL = try MLModel.compileModel(at: modelURL)
        }

        self.model = try MLModel(contentsOf: compiledURL, configuration: config)
    }

    /// Run inference.
    ///
    /// - Parameters:
    ///   - featErb: ERB features `[1, 1, T, 32]` (NCHW, channels=1)
    ///   - featSpec: Spec features `[1, 2, T, 96]` (NCHW, channels=2: real, imag)
    /// - Returns: `(erbMask [1, 1, T, 32], dfCoefs [1, 5, T, 96, 2])`
    func predict(featErb: MLMultiArray, featSpec: MLMultiArray) throws -> (MLMultiArray, MLMultiArray) {
        let input = try MLDictionaryFeatureProvider(
            dictionary: [
                "feat_erb": MLFeatureValue(multiArray: featErb),
                "feat_spec": MLFeatureValue(multiArray: featSpec),
            ])

        let output = try model.prediction(from: input)

        guard let erbMask = output.featureValue(for: "erb_mask")?.multiArrayValue,
              let dfCoefs = output.featureValue(for: "df_coefs")?.multiArrayValue else {
            throw DeepFilterNet3Error.predictionFailed
        }

        return (erbMask, dfCoefs)
    }

    enum DeepFilterNet3Error: Error {
        case predictionFailed
    }
}
