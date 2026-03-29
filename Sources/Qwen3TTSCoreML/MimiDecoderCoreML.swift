#if canImport(CoreML)
import CoreML
import Foundation

/// CoreML-based Mimi speech decoder for Qwen3-TTS.
///
/// Converts 16-codebook indices to 24kHz mono audio waveform.
/// Non-autoregressive: single forward pass per chunk.
final class MimiDecoderCoreML {

    private let model: MLModel
    private let samplesPerFrame: Int = 1920  // 24000 / 12.5 Hz

    init(model: MLModel) {
        self.model = model
    }

    /// Allowed frame sizes for the EnumeratedShapes model
    private let allowedFrames = [4, 14, 35, 50]

    /// Decode codebook indices to audio using chunked processing.
    ///
    /// Splits into chunks that fit the model's EnumeratedShapes,
    /// with left context overlap for continuity.
    ///
    /// - Parameter codes: [16][T] — 16 codebook indices for T frames
    /// - Returns: Audio samples at 24kHz, mono Float32
    func decode(codes: [[Int32]]) throws -> [Float] {
        let numCodebooks = codes.count
        let numFrames = codes[0].count
        guard numCodebooks == 16, numFrames > 0 else {
            throw MimiError.invalidInput("Expected 16 codebooks, got \(numCodebooks)")
        }

        // If fits in one allowed shape, decode directly
        if let shape = allowedFrames.first(where: { $0 >= numFrames }) {
            return try decodeSingleChunk(codes: codes, padTo: shape)
        }

        // Chunked decode with overlap
        let chunkSize = 35  // Use 35-frame chunks
        let leftContext = 10
        var allAudio = [Float]()
        var offset = 0

        while offset < numFrames {
            let chunkEnd = min(offset + chunkSize, numFrames)
            let contextStart = max(0, offset - leftContext)
            let totalFrames = chunkEnd - contextStart

            // Pad to nearest allowed shape
            let padTo = allowedFrames.first(where: { $0 >= totalFrames }) ?? allowedFrames.last!

            // Extract chunk codes
            var chunkCodes = [[Int32]]()
            for cb in 0..<16 {
                var slice = Array(codes[cb][contextStart..<chunkEnd])
                // Zero-pad to padTo
                while slice.count < padTo { slice.append(0) }
                chunkCodes.append(slice)
            }

            let chunkAudio = try decodeSingleChunk(codes: chunkCodes, padTo: padTo)

            // Trim context and padding
            let contextSamples = (offset - contextStart) * samplesPerFrame
            let chunkSamples = (chunkEnd - offset) * samplesPerFrame
            let end = min(contextSamples + chunkSamples, chunkAudio.count)
            if contextSamples < chunkAudio.count {
                allAudio.append(contentsOf: chunkAudio[contextSamples..<end])
            }

            offset = chunkEnd
        }

        return allAudio
    }

    /// Decode a single chunk (must be an allowed frame size).
    private func decodeSingleChunk(codes: [[Int32]], padTo: Int) throws -> [Float] {
        let codesArray = try MLMultiArray(
            shape: [1, 16, NSNumber(value: padTo)], dataType: .int32)
        let ptr = codesArray.dataPointer.assumingMemoryBound(to: Int32.self)
        let numFrames = codes[0].count
        for cb in 0..<16 {
            for t in 0..<min(numFrames, padTo) {
                ptr[cb * padTo + t] = codes[cb][t]
            }
        }

        let input = try MLDictionaryFeatureProvider(dictionary: [
            "codes": MLFeatureValue(multiArray: codesArray)
        ])

        let result = try model.prediction(from: input)
        let audioArray = result.featureValue(for: "waveform")!.multiArrayValue!

        let totalSamples = audioArray.shape.map { $0.intValue }.reduce(1, *)
        var audio = [Float](repeating: 0, count: totalSamples)

        if audioArray.dataType == .float16 {
            let srcPtr = audioArray.dataPointer.assumingMemoryBound(to: Float16.self)
            for i in 0..<totalSamples { audio[i] = Float(srcPtr[i]) }
        } else {
            let srcPtr = audioArray.dataPointer.assumingMemoryBound(to: Float.self)
            for i in 0..<totalSamples { audio[i] = srcPtr[i] }
        }

        return audio
    }

    /// Chunked decode with left context for streaming.
    ///
    /// - Parameters:
    ///   - allCodebooks: [16][totalFrames] — all generated codebook indices
    ///   - chunkStart: Start frame for this chunk
    ///   - chunkEnd: End frame for this chunk
    ///   - leftContext: Number of frames of left context to include
    /// - Returns: Audio samples for this chunk only (context trimmed)
    func decodeChunk(
        allCodebooks: [[Int32]],
        chunkStart: Int,
        chunkEnd: Int,
        leftContext: Int = 10
    ) throws -> [Float] {
        let contextStart = max(0, chunkStart - leftContext)
        let chunkFrames = chunkEnd - chunkStart
        let totalFrames = chunkEnd - contextStart

        // Extract chunk with context
        var chunkCodes = [[Int32]]()
        for cb in 0..<16 {
            chunkCodes.append(Array(allCodebooks[cb][contextStart..<chunkEnd]))
        }

        let fullAudio = try decode(codes: chunkCodes)

        // Trim context from output
        let contextSamples = (chunkStart - contextStart) * samplesPerFrame
        let chunkSamples = chunkFrames * samplesPerFrame
        let end = min(contextSamples + chunkSamples, fullAudio.count)

        if contextSamples < fullAudio.count {
            return Array(fullAudio[contextSamples..<end])
        }
        return Array(fullAudio.suffix(chunkSamples))
    }

    enum MimiError: Error {
        case invalidInput(String)
    }
}
#endif
