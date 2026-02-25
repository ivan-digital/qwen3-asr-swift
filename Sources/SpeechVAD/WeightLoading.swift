import Foundation
import MLX
import MLXNN
import AudioCommon

/// Weight loading for the PyanNet segmentation model.
///
/// Loads from safetensors files produced by `scripts/convert_pyannote.py`.
/// The conversion script pre-computes sinc filters and transposes Conv1d weights,
/// so loading is a straightforward parameter tree update.
enum SegmentationWeightLoader {

    /// Load weights from a directory containing model.safetensors.
    static func loadWeights(
        model: SegmentationModel,
        from directory: URL
    ) throws {
        let weightsURL = directory.appendingPathComponent("model.safetensors")

        guard FileManager.default.fileExists(atPath: weightsURL.path) else {
            throw WeightLoadingError.noWeightsFound(directory)
        }

        let weights = try MLX.loadArrays(url: weightsURL)

        // The conversion script produces keys matching our module structure directly.
        // Use ModuleParameters.unflattened to build the nested parameter tree.
        let parameters = ModuleParameters.unflattened(weights)

        // Apply to model
        try model.update(parameters: parameters, verify: .noUnusedKeys)

        // Evaluate all parameters to ensure they're materialized
        MLX.eval(model.parameters())
    }
}
