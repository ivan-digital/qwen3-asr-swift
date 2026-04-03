#if canImport(CoreML)
import AudioCommon
import CoreML
import Foundation

extension SileroVADModel {

    /// Run CoreML inference for one 576-sample chunk (64 context + 512 new).
    ///
    /// Uses MLState for LSTM h/c state — managed internally by CoreML.
    ///
    /// - Parameter fullSamples: 576 Float32 samples (context prepended)
    /// - Returns: speech probability in `[0, 1]`
    func processChunkCoreML(_ fullSamples: [Float]) throws -> Float {
        guard let model = coremlModel else {
            throw AudioModelError.inferenceFailed(
                operation: "VAD", reason: "CoreML model not loaded")
        }

        if coremlState == nil {
            coremlState = model.makeState()
        }

        // Create audio input: [1, 1, 576] float16
        let audioArray = try MLMultiArray(shape: [1, 1, 576], dataType: .float16)
        let audioPtr = audioArray.dataPointer.assumingMemoryBound(to: Float16.self)
        for i in 0..<576 {
            audioPtr[i] = Float16(fullSamples[i])
        }

        let input = try MLDictionaryFeatureProvider(dictionary: [
            "audio": MLFeatureValue(multiArray: audioArray),
        ])
        let result = try model.prediction(from: input, using: coremlState!)

        // Extract probability scalar
        let probArray = result.featureValue(for: "probability")!.multiArrayValue!
        let probPtr = probArray.dataPointer.assumingMemoryBound(to: Float16.self)
        return Float(probPtr[0])
    }
}
#endif
