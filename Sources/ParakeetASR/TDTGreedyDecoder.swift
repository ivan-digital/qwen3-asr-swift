import Accelerate
import CoreML
import Foundation

/// Reusable MLFeatureProvider that avoids dictionary allocation on every CoreML prediction.
/// Backing MLMultiArray references can be updated via `update(_:_:)` when the underlying
/// array changes (e.g. new decoder hidden state after a non-blank token).
private class ReusableFeatureProvider: MLFeatureProvider {
    let featureNames: Set<String>
    private var values: [String: MLFeatureValue]

    init(_ dict: [String: MLMultiArray]) {
        self.featureNames = Set(dict.keys)
        self.values = dict.mapValues { MLFeatureValue(multiArray: $0) }
    }

    func featureValue(for name: String) -> MLFeatureValue? { values[name] }

    func update(_ name: String, _ array: MLMultiArray) {
        values[name] = MLFeatureValue(multiArray: array)
    }
}

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

        // Pre-allocate token array — reused every decoder step
        let tokenArray = try MLMultiArray(shape: [1, 1], dataType: .int32)
        let tokenPtr = tokenArray.dataPointer.assumingMemoryBound(to: Int32.self)

        // Pre-allocate Float buffer for vDSP argmax on token logits
        let argmaxBuf = UnsafeMutablePointer<Float>.allocate(capacity: config.vocabSize + 1)
        defer { argmaxBuf.deallocate() }

        // Initial decoder step with blank token
        tokenPtr.pointee = Int32(config.blankTokenId)
        let decoderProvider = ReusableFeatureProvider([
            "token": tokenArray, "h": h, "c": c,
        ])
        let initialOut = try decoder.prediction(from: decoderProvider)

        var decoderOutput = initialOut.featureValue(for: "decoder_output")!.multiArrayValue!
        var hState = initialOut.featureValue(for: "h_out")!.multiArrayValue!
        var cState = initialOut.featureValue(for: "c_out")!.multiArrayValue!

        // Encoder slice buffer: [1, 1, encoderHidden]
        let encSlice = try MLMultiArray(shape: [1, 1, config.encoderHidden as NSNumber], dataType: .float16)

        // Pre-allocate reusable feature providers for the decode loop
        let jointProvider = ReusableFeatureProvider([
            "encoder_output": encSlice, "decoder_output": decoderOutput,
        ])
        decoderProvider.update("h", hState)
        decoderProvider.update("c", cState)

        var t = 0
        while t < encodedLength {
            // Extract encoder frame at position t (mutates encSlice data in-place)
            copyEncoderFrame(from: encoded, at: t, to: encSlice)

            // Joint network: (encoder_slice, decoder_output) → (token_logits, duration_logits)
            let jointOut = try joint.prediction(from: jointProvider)

            let tokenLogits = jointOut.featureValue(for: "token_logits")!.multiArrayValue!
            let durationLogits = jointOut.featureValue(for: "duration_logits")!.multiArrayValue!

            let tokenId = argmax(tokenLogits, count: config.vocabSize + 1, floatBuf: argmaxBuf)

            if tokenId == config.blankTokenId {
                // Blank: advance one frame
                t += 1
            } else {
                // Non-blank: emit token and advance by predicted duration
                tokens.append(tokenId)

                let durationIdx = argmax(durationLogits, count: config.numDurationBins, floatBuf: nil)
                let duration = config.durationBins[durationIdx]
                t += max(duration, 1)

                // Update decoder state with the emitted token
                tokenPtr.pointee = Int32(tokenId)
                decoderProvider.update("h", hState)
                decoderProvider.update("c", cState)
                let decOut = try decoder.prediction(from: decoderProvider)
                decoderOutput = decOut.featureValue(for: "decoder_output")!.multiArrayValue!
                hState = decOut.featureValue(for: "h_out")!.multiArrayValue!
                cState = decOut.featureValue(for: "c_out")!.multiArrayValue!

                // Update joint provider with new decoder output
                jointProvider.update("decoder_output", decoderOutput)
            }
        }

        return tokens
    }

    // MARK: - Array Operations

    /// Copy encoder frame at time `t` into the slice buffer using memcpy.
    private func copyEncoderFrame(from encoded: MLMultiArray, at t: Int, to slice: MLMultiArray) {
        let hidden = config.encoderHidden
        let src = encoded.dataPointer.advanced(by: t * hidden * MemoryLayout<Float16>.stride)
        memcpy(slice.dataPointer, src, hidden * MemoryLayout<Float16>.stride)
    }

    /// Find the index of the maximum value in the first `count` elements.
    /// Uses vDSP for large arrays (token logits); scalar for small arrays (duration logits).
    private func argmax(_ array: MLMultiArray, count: Int, floatBuf: UnsafeMutablePointer<Float>?) -> Int {
        let ptr = array.dataPointer.assumingMemoryBound(to: Float16.self)

        // Small arrays or no buffer: scalar path
        if count <= 16 || floatBuf == nil {
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

        // Large arrays: convert Float16→Float, then vDSP_maxvi
        for i in 0..<count { floatBuf![i] = Float(ptr[i]) }
        var maxVal: Float = 0
        var maxIdx: vDSP_Length = 0
        vDSP_maxvi(floatBuf!, 1, &maxVal, &maxIdx, vDSP_Length(count))
        return Int(maxIdx)
    }

    /// Zero-fill an MLMultiArray using memset.
    private func zeroFill(_ array: MLMultiArray) {
        memset(array.dataPointer, 0, array.count * MemoryLayout<Float16>.stride)
    }
}
