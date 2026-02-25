import Foundation
import MLX
import MLXNN
import AudioCommon

/// Weight loading for the Silero VAD v5 model.
///
/// Loads from safetensors files produced by `scripts/convert_silero_vad.py`.
/// The conversion script transposes Conv1d weights and sums LSTM biases,
/// so loading is a straightforward parameter tree update.
enum SileroWeightLoader {

    /// Load weights from a directory containing model.safetensors.
    static func loadWeights(
        model: SileroVADNetwork,
        from directory: URL
    ) throws {
        let weightsURL = directory.appendingPathComponent("model.safetensors")

        guard FileManager.default.fileExists(atPath: weightsURL.path) else {
            throw WeightLoadingError.noWeightsFound(directory)
        }

        let weights = try MLX.loadArrays(url: weightsURL)

        // Build nested parameter tree from flat keys
        let parameters = ModuleParameters.unflattened(weights)

        // Apply to model
        try model.update(parameters: parameters, verify: .noUnusedKeys)

        // Materialize all parameters
        MLX.eval(model.parameters())
    }
}
