import Accelerate
import AudioCommon
import CoreML
import Foundation

/// Reusable MLFeatureProvider that avoids dictionary allocation per prediction.
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

struct RNNTDecodeResult {
    let tokens: [Int]
    let tokenLogProbs: [Float]
}

/// Greedy RNNT decoder for Nemotron Streaming. Blank advances to the next encoder
/// frame; any non-blank token is emitted directly (punctuation and capitalization
/// tokens are part of the vocab, so there is no special EOU/EOB handling).
struct RNNTGreedyDecoder {
    let config: NemotronStreamingConfig
    let decoder: MLModel
    let joint: MLModel

    private let maxSymbolsPerStep = 10

    func decode(
        encoded: MLMultiArray,
        encodedLength: Int,
        frameOffset: Int = 0,
        h: inout MLMultiArray,
        c: inout MLMultiArray,
        decoderOutput: inout MLMultiArray,
        decoderProvider: ReusableFeatureProvider,
        jointProvider: ReusableFeatureProvider,
        tokenArray: MLMultiArray,
        encSlice: MLMultiArray,
        argmaxBuf: UnsafeMutablePointer<Float>
    ) throws -> RNNTDecodeResult {
        var tokens = [Int]()
        var tokenLogProbs = [Float]()

        let tokenPtr = tokenArray.dataPointer.assumingMemoryBound(to: Int32.self)
        let totalClasses = config.vocabSize + 1

        for i in 0..<encodedLength {
            let t = i + frameOffset
            copyEncoderFrameFP16(from: encoded, at: t, toFP32: encSlice)

            for _ in 0..<maxSymbolsPerStep {
                jointProvider.update("encoder_output", encSlice)
                jointProvider.update("decoder_output", decoderOutput)
                let jointOut = try joint.prediction(from: jointProvider)
                let logits = jointOut.featureValue(for: "logits")!.multiArrayValue!

                let tokenId = argmax(logits, count: totalClasses, floatBuf: argmaxBuf)

                if tokenId == config.blankTokenId { break }

                tokens.append(tokenId)
                let logProb = logSoftmax(logits, tokenId: tokenId, count: totalClasses, floatBuf: argmaxBuf)
                tokenLogProbs.append(logProb)

                tokenPtr.pointee = Int32(tokenId)
                decoderProvider.update("h", h)
                decoderProvider.update("c", c)
                let decOut = try decoder.prediction(from: decoderProvider)
                StreamingSession.copyCastFP16ToFP32(
                    decOut.featureValue(for: "decoder_output")!.multiArrayValue!, into: decoderOutput)
                StreamingSession.copyCastFP16ToFP32(
                    decOut.featureValue(for: "h_out")!.multiArrayValue!, into: h)
                StreamingSession.copyCastFP16ToFP32(
                    decOut.featureValue(for: "c_out")!.multiArrayValue!, into: c)
            }
        }

        return RNNTDecodeResult(tokens: tokens, tokenLogProbs: tokenLogProbs)
    }

    private func copyEncoderFrameFP16(from encoded: MLMultiArray, at t: Int, toFP32 slice: MLMultiArray) {
        let hidden = config.encoderHidden
        let src = encoded.dataPointer.assumingMemoryBound(to: Float16.self).advanced(by: t * hidden)
        let dst = slice.dataPointer.assumingMemoryBound(to: Float.self)
        for i in 0..<hidden { dst[i] = Float(src[i]) }
    }

    private func logSoftmax(_ array: MLMultiArray, tokenId: Int, count: Int, floatBuf: UnsafeMutablePointer<Float>) -> Float {
        loadFP16AsFloat(array, count: count, into: floatBuf)

        var maxVal: Float = 0
        var maxIdx: vDSP_Length = 0
        vDSP_maxvi(floatBuf, 1, &maxVal, &maxIdx, vDSP_Length(count))

        let logitForToken = floatBuf[tokenId]

        var negMax = -maxVal
        vDSP_vsadd(floatBuf, 1, &negMax, floatBuf, 1, vDSP_Length(count))
        var n = Int32(count)
        vvexpf(floatBuf, floatBuf, &n)
        var sumExp: Float = 0
        vDSP_sve(floatBuf, 1, &sumExp, vDSP_Length(count))

        let logSumExp = log(sumExp) + maxVal
        return logitForToken - logSumExp
    }

    private func argmax(_ array: MLMultiArray, count: Int, floatBuf: UnsafeMutablePointer<Float>) -> Int {
        loadFP16AsFloat(array, count: count, into: floatBuf)
        var maxVal: Float = 0
        var maxIdx: vDSP_Length = 0
        vDSP_maxvi(floatBuf, 1, &maxVal, &maxIdx, vDSP_Length(count))
        return Int(maxIdx)
    }

    private func loadFP16AsFloat(_ array: MLMultiArray, count: Int, into buf: UnsafeMutablePointer<Float>) {
        let ptr = array.dataPointer.assumingMemoryBound(to: Float16.self)
        for i in 0..<count { buf[i] = Float(ptr[i]) }
    }
}
