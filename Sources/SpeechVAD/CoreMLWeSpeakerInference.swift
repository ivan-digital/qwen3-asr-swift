#if canImport(CoreML)
import AudioCommon
import CoreML
import Foundation

extension WeSpeakerModel {

    /// Run CoreML inference to extract a 256-dim speaker embedding.
    ///
    /// Pads the mel spectrogram to the nearest enumerated shape (required by
    /// the CoreML model which uses EnumeratedShapes for the time dimension),
    /// runs prediction, and extracts the embedding.
    ///
    /// - Parameters:
    ///   - melSpec: flat Float array of log-mel features `[nFrames * 80]`
    ///   - nFrames: number of mel frames
    /// - Returns: 256-dim L2-normalized speaker embedding
    func embedCoreML(melSpec: [Float], nFrames: Int) throws -> [Float] {
        guard let model = coremlModel else {
            throw AudioModelError.inferenceFailed(
                operation: "SpeakerEmbedding", reason: "CoreML model not loaded")
        }

        // Find nearest enumerated length >= nFrames
        let targetLength = Self.enumeratedMelLengths.first { $0 >= nFrames }
            ?? Self.enumeratedMelLengths.last!

        // Create input: [1, 1, targetLength, 80] float16
        let melArray = try MLMultiArray(
            shape: [1, 1, targetLength as NSNumber, 80],
            dataType: .float16
        )
        let melPtr = melArray.dataPointer.assumingMemoryBound(to: Float16.self)

        // Fill with mel data (row-major: frame-major, 80 mels per frame)
        let copyCount = min(nFrames, targetLength) * 80
        for i in 0..<copyCount {
            melPtr[i] = Float16(melSpec[i])
        }
        // Zero-pad remaining frames
        let totalElements = targetLength * 80
        for i in copyCount..<totalElements {
            melPtr[i] = 0
        }

        let input = try MLDictionaryFeatureProvider(dictionary: [
            "mel": MLFeatureValue(multiArray: melArray),
        ])

        let result = try model.prediction(from: input)

        // Extract "embedding" output: [1, 256]
        guard let embArray = result.featureValue(for: "embedding")?.multiArrayValue else {
            throw AudioModelError.inferenceFailed(
                operation: "SpeakerEmbedding", reason: "Missing 'embedding' output")
        }

        // Read 256 float16 values
        var embedding = [Float](repeating: 0, count: 256)
        let embPtr = embArray.dataPointer.assumingMemoryBound(to: Float16.self)
        for i in 0..<256 {
            embedding[i] = Float(embPtr[i])
        }

        return embedding
    }
}
#endif
