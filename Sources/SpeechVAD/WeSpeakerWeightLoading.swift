import Foundation
import MLX
import MLXNN
import AudioCommon

/// Weight loading for the WeSpeaker ResNet34-LM speaker embedding model.
///
/// Loads from safetensors files produced by `scripts/convert_wespeaker.py`.
/// The conversion script fuses BatchNorm into Conv2d and transposes weights,
/// so loading is a straightforward parameter tree update.
enum WeSpeakerWeightLoader {

    /// Load weights from a directory containing model.safetensors.
    static func loadWeights(
        model: WeSpeakerNetwork,
        from directory: URL
    ) throws {
        let weightsURL = directory.appendingPathComponent("model.safetensors")

        guard FileManager.default.fileExists(atPath: weightsURL.path) else {
            throw WeightLoadingError.noWeightsFound(directory)
        }

        let weights = try MLX.loadArrays(url: weightsURL)

        let parameters = ModuleParameters.unflattened(weights)

        try model.update(parameters: parameters, verify: .noUnusedKeys)

        MLX.eval(model.parameters())
    }
}
