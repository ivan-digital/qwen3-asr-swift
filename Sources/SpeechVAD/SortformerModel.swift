#if canImport(CoreML)
import CoreML
import Foundation
import AudioCommon

/// CoreML wrapper for the Sortformer streaming diarization model.
///
/// Runs on Neural Engine via CoreML. The model takes a chunk of mel features
/// plus streaming state buffers (spkcache, fifo) and outputs per-frame
/// speaker predictions for up to 4 speakers.
final class SortformerCoreMLModel {

    private let model: MLModel
    let config: SortformerConfig

    /// Input shape constants derived from the CoreML model.
    /// chunk: [1, 112, 128], spkcache: [1, 188, 512], fifo: [1, 40, 512]
    private let chunkFrames: Int = 112
    private let spkcacheFrames: Int
    private let fifoFrames: Int
    private let featureDim: Int

    init(model: MLModel, config: SortformerConfig = .default) {
        self.model = model
        self.config = config
        self.spkcacheFrames = config.spkcacheLen
        self.fifoFrames = config.fifoLen
        self.featureDim = config.fcDModel
    }

    /// Run one streaming inference step.
    ///
    /// - Parameters:
    ///   - chunk: Mel features for this chunk, flat `[chunkFrames * nMels]` float array.
    ///            If fewer frames available, zero-pad on the right.
    ///   - chunkLength: Actual number of valid mel frames in the chunk
    ///   - spkcache: Speaker cache state, flat `[spkcacheFrames * fcDModel]`
    ///   - spkcacheLength: Number of valid frames in speaker cache
    ///   - fifo: FIFO buffer state, flat `[fifoFrames * fcDModel]`
    ///   - fifoLength: Number of valid frames in FIFO
    /// - Returns: `SortformerOutput` with predictions and updated state
    func predict(
        chunk: [Float],
        chunkLength: Int,
        spkcache: [Float],
        spkcacheLength: Int,
        fifo: [Float],
        fifoLength: Int
    ) throws -> SortformerOutput {
        // Create input arrays
        let chunkArray = try makeMultiArray(
            shape: [1, NSNumber(value: chunkFrames), NSNumber(value: config.nMels)],
            from: chunk)
        let chunkLenArray = try makeScalarInt32Array(value: Int32(chunkLength))

        let spkcacheArray = try makeMultiArray(
            shape: [1, NSNumber(value: spkcacheFrames), NSNumber(value: featureDim)],
            from: spkcache)
        let spkcacheLenArray = try makeScalarInt32Array(value: Int32(spkcacheLength))

        let fifoArray = try makeMultiArray(
            shape: [1, NSNumber(value: fifoFrames), NSNumber(value: featureDim)],
            from: fifo)
        let fifoLenArray = try makeScalarInt32Array(value: Int32(fifoLength))

        let input = try MLDictionaryFeatureProvider(dictionary: [
            "chunk": MLFeatureValue(multiArray: chunkArray),
            "chunk_lengths": MLFeatureValue(multiArray: chunkLenArray),
            "spkcache": MLFeatureValue(multiArray: spkcacheArray),
            "spkcache_lengths": MLFeatureValue(multiArray: spkcacheLenArray),
            "fifo": MLFeatureValue(multiArray: fifoArray),
            "fifo_lengths": MLFeatureValue(multiArray: fifoLenArray),
        ])

        let result = try model.prediction(from: input)

        // Extract outputs
        let predsArray = result.featureValue(for: "speaker_preds")!.multiArrayValue!
        let embsArray = result.featureValue(for: "chunk_pre_encoder_embs")!.multiArrayValue!
        let embsLenArray = result.featureValue(for: "chunk_pre_encoder_lengths")!.multiArrayValue!

        let predsShape = (0..<predsArray.shape.count).map { predsArray.shape[$0].intValue }
        let embsShape = (0..<embsArray.shape.count).map { embsArray.shape[$0].intValue }

        let totalPreds = predsShape.reduce(1, *)
        let totalEmbs = embsShape.reduce(1, *)

        var preds = [Float](repeating: 0, count: totalPreds)
        let predsPtr = predsArray.dataPointer.assumingMemoryBound(to: Float.self)
        for i in 0..<totalPreds { preds[i] = predsPtr[i] }

        var embs = [Float](repeating: 0, count: totalEmbs)
        let embsPtr = embsArray.dataPointer.assumingMemoryBound(to: Float.self)
        for i in 0..<totalEmbs { embs[i] = embsPtr[i] }

        let embsLenPtr = embsLenArray.dataPointer.assumingMemoryBound(to: Int32.self)
        let validEmbFrames = Int(embsLenPtr[0])

        return SortformerOutput(
            speakerPreds: preds,
            predsFrames: predsShape.count >= 2 ? predsShape[predsShape.count - 2] : totalPreds / config.maxSpeakers,
            numSpeakers: config.maxSpeakers,
            encoderEmbs: embs,
            encoderEmbFrames: embsShape.count >= 2 ? embsShape[embsShape.count - 2] : validEmbFrames,
            embDim: featureDim,
            validEmbFrames: validEmbFrames
        )
    }

    // MARK: - Helpers

    private func makeMultiArray(shape: [NSNumber], from data: [Float]) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: shape, dataType: .float32)
        let ptr = array.dataPointer.assumingMemoryBound(to: Float.self)
        let count = min(data.count, array.count)
        for i in 0..<count {
            ptr[i] = data[i]
        }
        // Zero-fill remainder
        for i in count..<array.count {
            ptr[i] = 0
        }
        return array
    }

    private func makeScalarInt32Array(value: Int32) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: [1], dataType: .int32)
        let ptr = array.dataPointer.assumingMemoryBound(to: Int32.self)
        ptr[0] = value
        return array
    }
}

/// Output from one Sortformer inference step.
struct SortformerOutput {
    /// Speaker predictions, flat `[predsFrames * numSpeakers]`, sigmoid probabilities
    let speakerPreds: [Float]
    /// Number of prediction frames
    let predsFrames: Int
    /// Number of speaker channels
    let numSpeakers: Int
    /// Pre-encoder embeddings for state update, flat `[encoderEmbFrames * embDim]`
    let encoderEmbs: [Float]
    /// Total encoder embedding frames
    let encoderEmbFrames: Int
    /// Embedding dimension
    let embDim: Int
    /// Number of valid (non-padding) embedding frames
    let validEmbFrames: Int

    /// Get speaker prediction probability at (frame, speaker).
    func pred(frame: Int, speaker: Int) -> Float {
        speakerPreds[frame * numSpeakers + speaker]
    }

    /// Get encoder embedding at (frame, dim).
    func emb(frame: Int, dim: Int) -> Float {
        encoderEmbs[frame * embDim + dim]
    }
}
#endif
