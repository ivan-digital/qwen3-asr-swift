import CoreML
import Foundation

/// Greedy decoder for Token-and-Duration Transducer (TDT) models.
///
/// TDT extends standard transducers with a duration prediction head.
/// When a non-blank token is emitted, the duration head predicts how many
/// encoder frames to advance (from `durationBins`), enabling variable-rate
/// alignment between audio and text.
struct TDTGreedyDecoder {
    let config: ParakeetConfig
    let decoder: MLModel
    let joint: MLModel

    /// Decode encoded audio representations into token IDs.
    ///
    /// - Parameters:
    ///   - encoded: Encoder output as MLMultiArray, shape `[1, T, encoderHidden]`
    ///   - encodedLength: Number of valid encoder frames
    /// - Returns: Array of decoded token IDs (excluding blank)
    func decode(encoded: MLMultiArray, encodedLength: Int) throws -> [Int] {
        var tokens = [Int]()

        // Initialize LSTM state
        let hShape = [config.decoderLayers, 1, config.decoderHidden] as [NSNumber]
        let h = try MLMultiArray(shape: hShape, dataType: .float16)
        let c = try MLMultiArray(shape: hShape, dataType: .float16)
        zeroFill(h)
        zeroFill(c)

        // Initial decoder step with blank token
        let blankInput = try makeDecoderInput(tokenId: config.blankTokenId, h: h, c: c)
        let initialOut = try decoder.prediction(from: blankInput)

        var decoderOutput = initialOut.featureValue(for: "decoder_output")!.multiArrayValue!
        var hState = initialOut.featureValue(for: "h_out")!.multiArrayValue!
        var cState = initialOut.featureValue(for: "c_out")!.multiArrayValue!

        // Encoder slice buffer: [1, 1, encoderHidden]
        let encSlice = try MLMultiArray(shape: [1, 1, config.encoderHidden as NSNumber], dataType: .float16)

        var t = 0
        while t < encodedLength {
            // Extract encoder frame at position t
            copyEncoderFrame(from: encoded, at: t, to: encSlice)

            // Joint network: (encoder_slice, decoder_output) → (token_logits, duration_logits)
            let jointInput = try makeJointInput(encoderSlice: encSlice, decoderOutput: decoderOutput)
            let jointOut = try joint.prediction(from: jointInput)

            let tokenLogits = jointOut.featureValue(for: "token_logits")!.multiArrayValue!
            let durationLogits = jointOut.featureValue(for: "duration_logits")!.multiArrayValue!

            let tokenId = argmax(tokenLogits, count: config.vocabSize + 1)

            if tokenId == config.blankTokenId {
                // Blank: advance one frame
                t += 1
            } else {
                // Non-blank: emit token and advance by predicted duration
                tokens.append(tokenId)

                let durationIdx = argmax(durationLogits, count: config.numDurationBins)
                let duration = config.durationBins[durationIdx]
                t += max(duration, 1)

                // Update decoder state with the emitted token
                let decInput = try makeDecoderInput(tokenId: tokenId, h: hState, c: cState)
                let decOut = try decoder.prediction(from: decInput)
                decoderOutput = decOut.featureValue(for: "decoder_output")!.multiArrayValue!
                hState = decOut.featureValue(for: "h_out")!.multiArrayValue!
                cState = decOut.featureValue(for: "c_out")!.multiArrayValue!
            }
        }

        return tokens
    }

    // MARK: - Input Construction

    private func makeDecoderInput(tokenId: Int, h: MLMultiArray, c: MLMultiArray) throws -> MLDictionaryFeatureProvider {
        let tokenArray = try MLMultiArray(shape: [1, 1], dataType: .int32)
        tokenArray[0] = NSNumber(value: Int32(tokenId))

        return try MLDictionaryFeatureProvider(dictionary: [
            "token": MLFeatureValue(multiArray: tokenArray),
            "h": MLFeatureValue(multiArray: h),
            "c": MLFeatureValue(multiArray: c),
        ])
    }

    private func makeJointInput(encoderSlice: MLMultiArray, decoderOutput: MLMultiArray) throws -> MLDictionaryFeatureProvider {
        try MLDictionaryFeatureProvider(dictionary: [
            "encoder_output": MLFeatureValue(multiArray: encoderSlice),
            "decoder_output": MLFeatureValue(multiArray: decoderOutput),
        ])
    }

    // MARK: - Array Operations

    /// Copy encoder frame at time `t` into the slice buffer using dataPointer.
    private func copyEncoderFrame(from encoded: MLMultiArray, at t: Int, to slice: MLMultiArray) {
        let hidden = config.encoderHidden
        let srcPtr = UnsafeBufferPointer(
            start: encoded.dataPointer.assumingMemoryBound(to: Float16.self),
            count: encoded.count)
        let dstPtr = UnsafeMutableBufferPointer(
            start: slice.dataPointer.assumingMemoryBound(to: Float16.self),
            count: slice.count)

        let srcOffset = t * hidden
        for i in 0..<hidden {
            dstPtr[i] = srcPtr[srcOffset + i]
        }
    }

    /// Find the index of the maximum value in the first `count` elements.
    private func argmax(_ array: MLMultiArray, count: Int) -> Int {
        let ptr = UnsafeBufferPointer(
            start: array.dataPointer.assumingMemoryBound(to: Float16.self),
            count: array.count)

        var maxIdx = 0
        var maxVal = ptr[0]
        for i in 1..<count {
            if ptr[i] > maxVal {
                maxVal = ptr[i]
                maxIdx = i
            }
        }
        return maxIdx
    }

    /// Zero-fill an MLMultiArray.
    private func zeroFill(_ array: MLMultiArray) {
        let ptr = UnsafeMutableBufferPointer(
            start: array.dataPointer.assumingMemoryBound(to: Float16.self),
            count: array.count)
        for i in 0..<ptr.count {
            ptr[i] = 0
        }
    }
}
