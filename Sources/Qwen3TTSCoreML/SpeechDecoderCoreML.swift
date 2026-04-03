#if canImport(CoreML)
import CoreML
import Foundation

/// CoreML batch speech decoder: 16 codebooks → 24kHz audio.
/// Fixed T=125 frames (10s max). Pads shorter sequences, trims output.
final class SpeechDecoderCoreML {
    private let model: MLModel
    private let batchFrames = 125
    private let samplesPerFrame = 1920

    init(model: MLModel) { self.model = model }

    /// Decode codebook indices to audio.
    /// - Parameter codes: [16][T] — 16 codebook indices for T frames
    /// - Returns: Audio samples at 24kHz, mono Float32
    func decode(codes: [[Int32]]) throws -> [Float] {
        let numCodebooks = codes.count
        let numFrames = codes[0].count
        guard numCodebooks == 16, numFrames > 0 else {
            throw SpeechDecoderError.invalidInput("Expected 16 codebooks, got \(numCodebooks)")
        }

        // Build [1, 16, 125] input, zero-padded
        let input = try MLMultiArray(shape: [1, 16, NSNumber(value: batchFrames)], dataType: .int32)
        let ptr = input.dataPointer.assumingMemoryBound(to: Int32.self)
        memset(ptr, 0, 16 * batchFrames * 4)
        for cb in 0..<16 {
            for t in 0..<min(numFrames, batchFrames) {
                ptr[cb * batchFrames + t] = codes[cb][t]
            }
        }

        let provider = try MLDictionaryFeatureProvider(dictionary: [
            "audio_codes": MLFeatureValue(multiArray: input)])
        let result = try model.prediction(from: provider)
        let audioArray = result.featureValue(for: "audio")!.multiArrayValue!

        // Extract and trim to actual frame count
        let totalSamples = min(numFrames * samplesPerFrame, audioArray.count)
        var audio = [Float](repeating: 0, count: totalSamples)
        if audioArray.dataType == .float16 {
            let src = audioArray.dataPointer.assumingMemoryBound(to: Float16.self)
            for i in 0..<totalSamples { audio[i] = Float(src[i]) }
        } else {
            let src = audioArray.dataPointer.assumingMemoryBound(to: Float.self)
            for i in 0..<totalSamples { audio[i] = src[i] }
        }
        return audio
    }

    enum SpeechDecoderError: Error {
        case invalidInput(String)
    }
}
#endif
