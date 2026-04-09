import Accelerate
import AudioCommon
import CoreML
import Foundation

/// Reusable MLFeatureProvider that avoids dictionary allocation on every CoreML prediction.
class ReusableFeatureProvider: MLFeatureProvider {
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

/// Result of RNNT greedy decoding over a set of encoder frames.
struct RNNTDecodeResult {
    let tokens: [Int]
    let tokenLogProbs: [Float]
    let eouDetected: Bool
}

/// Greedy RNNT decoder for Parakeet EOU streaming ASR.
///
/// Matches reference implementation:
/// - Encoder output: [B, D, T] (channels-first)
/// - Decoder: float32 h/c, output [B, D, 1]
/// - Joint: argmax baked in, outputs token_id directly
struct RNNTGreedyDecoder {
    let config: ParakeetEOUConfig
    let decoder: MLModel
    let joint: MLModel

    /// Maximum non-blank tokens per encoder frame.
    private let maxSymbolsPerStep = 2  // Matches reference

    /// Decode encoder output with persistent LSTM state.
    func decode(
        encoded: MLMultiArray,
        encodedLength: Int,
        startFrame: Int = 0,
        h: inout MLMultiArray,
        c: inout MLMultiArray,
        decoderOutput: inout MLMultiArray,
        decoderProvider: ReusableFeatureProvider,
        jointProvider: ReusableFeatureProvider,
        tokenArray: MLMultiArray,
        encSlice: MLMultiArray
    ) throws -> RNNTDecodeResult {
        var tokens = [Int]()
        var tokenLogProbs = [Float]()
        var eouDetected = false

        let tokenPtr = tokenArray.dataPointer.assumingMemoryBound(to: Int32.self)

        for t in startFrame..<(startFrame + encodedLength) {
            // Extract encoder frame t from [B, D, T] → [B, D, 1]
            copyEncoderFrame(from: encoded, at: t, to: encSlice)

            for _ in 0..<maxSymbolsPerStep {
                // Joint expects float32 inputs — convert decoder output if needed
                let decStep: MLMultiArray
                if decoderOutput.dataType == .float16 {
                    decStep = try convertToFloat32(decoderOutput)
                } else {
                    decStep = decoderOutput
                }
                jointProvider.update("encoder_step", encSlice)
                jointProvider.update("decoder_step", decStep)
                let jointOut = try joint.prediction(from: jointProvider)
                let tokenIdArray = jointOut.featureValue(for: "token_id")!.multiArrayValue!
                let tokenId = Int(tokenIdArray[0].int32Value)
                if t == startFrame {
                    let probArray = jointOut.featureValue(for: "token_prob")!.multiArrayValue!
                    print("[DBG] joint[t=0]: token=\(tokenId) prob=\(probArray[0]) blank=\(config.blankTokenId)")
                    print("[DBG] encSlice: shape=\(encSlice.shape) dtype=\(encSlice.dataType.rawValue)")
                    print("[DBG] decOut: shape=\(decoderOutput.shape) dtype=\(decoderOutput.dataType.rawValue)")
                }

                if tokenId == config.blankTokenId {
                    break  // Advance to next encoder frame
                }

                if tokenId == config.eouTokenId {
                    eouDetected = true
                    break
                }

                // Emit token
                tokens.append(tokenId)
                // Get probability for confidence
                let probArray = jointOut.featureValue(for: "token_prob")!.multiArrayValue!
                tokenLogProbs.append(Float(truncating: probArray[0]))

                // Update decoder LSTM with emitted token
                tokenPtr.pointee = Int32(tokenId)
                decoderProvider.update("h_in", h)
                decoderProvider.update("c_in", c)
                let decOut = try decoder.prediction(from: decoderProvider)
                decoderOutput = decOut.featureValue(for: "decoder")!.multiArrayValue!
                h = decOut.featureValue(for: "h_out")!.multiArrayValue!
                c = decOut.featureValue(for: "c_out")!.multiArrayValue!
            }

            if eouDetected { break }
        }

        return RNNTDecodeResult(tokens: tokens, tokenLogProbs: tokenLogProbs, eouDetected: eouDetected)
    }

    // MARK: - Array Operations

    /// Convert float16 MLMultiArray to float32.
    private func convertToFloat32(_ array: MLMultiArray) throws -> MLMultiArray {
        let result = try MLMultiArray(shape: array.shape, dataType: .float32)
        let src = array.dataPointer.assumingMemoryBound(to: Float16.self)
        let dst = result.dataPointer.assumingMemoryBound(to: Float.self)
        for i in 0..<array.count { dst[i] = Float(src[i]) }
        return result
    }

    /// Copy encoder frame at time `t` from [B, D, T] layout to [B, D, 1].
    private func copyEncoderFrame(from encoded: MLMultiArray, at t: Int, to slice: MLMultiArray) {
        let hidden = config.encoderHidden
        let totalFrames = encoded.shape[2].intValue
        // encoded is [1, D, T] float16 from CoreML — copy with stride
        let src = encoded.dataPointer.assumingMemoryBound(to: Float16.self)
        let dst = slice.dataPointer.assumingMemoryBound(to: Float.self)
        for d in 0..<hidden {
            dst[d] = Float(src[d * totalFrames + t])
        }
    }
}
