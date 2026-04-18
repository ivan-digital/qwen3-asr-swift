import CoreML
import Foundation
import AudioCommon

/// Stateful wrapper over encoder/decoder/joiner + the streaming KWS decoder.
///
/// One ``WakeWordSession`` = one independent streaming audio source. Not
/// thread-safe; push audio from a single queue.
public final class WakeWordSession {
    public let config: KWSZipformerConfig
    private let encoder: MLModel
    private let decoder: MLModel
    private let joiner: MLModel
    private let fbank: KaldiFbank
    private let kwsDecoder: StreamingKwsDecoder

    // fbank accumulation buffer
    private var audioBuffer: [Float] = []
    // per-frame mel features buffered for the encoder sliding window
    private var melBuffer: [[Float]] = []
    // encoder state tensors, keyed by ALL_STATE_NAMES order from the export
    private var layerStates: [String: MLMultiArray]
    private var cachedEmbedLeftPad: MLMultiArray
    private var processedLens: MLMultiArray

    public init(
        config: KWSZipformerConfig,
        encoder: MLModel,
        decoder: MLModel,
        joiner: MLModel,
        fbank: KaldiFbank,
        contextGraph: ContextGraph
    ) throws {
        self.config = config
        self.encoder = encoder
        self.decoder = decoder
        self.joiner = joiner
        self.fbank = fbank

        var states = [String: MLMultiArray]()
        for (name, shape) in zip(config.encoder.layerStateNames, config.encoder.layerStateShapes) {
            states[name] = try MLMultiArray(
                shape: shape.map { NSNumber(value: $0) }, dataType: .float32
            )
        }
        self.layerStates = states
        self.cachedEmbedLeftPad = try MLMultiArray(
            shape: config.encoder.cachedEmbedLeftPadShape.map { NSNumber(value: $0) },
            dataType: .float32
        )
        self.processedLens = try MLMultiArray(shape: [1], dataType: .int32)
        self.processedLens[0] = 0

        let decoderModel = decoder
        let joinerModel = joiner

        // Closure-based backends for the pure-Swift decoder.
        self.kwsDecoder = StreamingKwsDecoder(
            decoderFn: { ctx in
                Self.runDecoder(model: decoderModel, contextTokens: ctx, contextSize: config.decoder.contextSize)
            },
            joinerFn: { enc, dec in
                Self.runJoiner(model: joinerModel, encoderFrame: enc, decoderOut: dec)
            },
            contextGraph: contextGraph,
            blankId: config.decoder.blankId,
            unkId: nil,
            contextSize: config.decoder.contextSize,
            beam: 4,
            numTrailingBlanks: config.kws.defaultNumTrailingBlanks,
            blankPenalty: 0,
            frameShiftSeconds: 0.04,
            autoResetSeconds: config.kws.autoResetSeconds
        )
    }

    /// Reset all streaming state (audio buffer, encoder caches, decoder beam).
    public func reset() throws {
        audioBuffer.removeAll(keepingCapacity: true)
        melBuffer.removeAll(keepingCapacity: true)
        for name in layerStates.keys {
            let array = layerStates[name]!
            memset(array.dataPointer, 0, array.count * MemoryLayout<Float>.stride)
        }
        memset(cachedEmbedLeftPad.dataPointer, 0,
               cachedEmbedLeftPad.count * MemoryLayout<Float>.stride)
        processedLens[0] = 0
        kwsDecoder.reset()
    }

    /// Push raw PCM and return any keyword detections that fired.
    public func pushAudio(_ samples: [Float]) throws -> [KeywordDetection] {
        audioBuffer.append(contentsOf: samples)

        // Build full mel feature buffer each time (streaming fbank inside a
        // single stream can be done in-place, but KaldiFbank stays stateless
        // and we simply re-compute over the accumulated buffer). We slice the
        // frames already consumed by the encoder to keep this bounded.
        let allMels = fbank.compute(audioBuffer)
        let numBins = config.feature.numMelBins
        var newFrames: [[Float]] = []
        newFrames.reserveCapacity(allMels.count / numBins)
        for f in 0..<(allMels.count / numBins) {
            let base = f * numBins
            newFrames.append(Array(allMels[base..<(base + numBins)]))
        }
        // Only keep mel frames not yet fed to the encoder.
        if newFrames.count < melBuffer.count {
            // fbank output shrank (shouldn't happen without reset)
            return []
        }
        let additions = Array(newFrames[melBuffer.count..<newFrames.count])
        melBuffer.append(contentsOf: additions)

        var emissions: [KeywordDetection] = []
        let totalIn = config.encoder.totalInputFrames  // 45
        let chunkSize = config.encoder.chunkSize       // 16
        while melBuffer.count >= totalIn {
            let window = Array(melBuffer.prefix(totalIn))
            let encoderFrames = try runEncoder(melWindow: window)
            emissions.append(contentsOf: kwsDecoder.stepChunk(encoderFrames))
            melBuffer.removeFirst(chunkSize)
        }
        return emissions
    }

    /// Flush remaining audio and surface any final detections.
    public func finalize() throws -> [KeywordDetection] {
        // Pad enough silence to flush one more encoder chunk.
        let shift = config.feature.frameShiftMs * 1e-3 * Double(config.feature.sampleRate)
        let missingFrames = max(0, config.encoder.totalInputFrames - melBuffer.count)
        let padSamples = Int(Double(missingFrames) * shift)
        if padSamples > 0 {
            return try pushAudio([Float](repeating: 0, count: padSamples))
        }
        return []
    }

    // MARK: - CoreML adapters

    private func runEncoder(melWindow: [[Float]]) throws -> [[Float]] {
        let numBins = config.feature.numMelBins
        let totalIn = config.encoder.totalInputFrames
        precondition(melWindow.count == totalIn, "expected \(totalIn) mel frames")

        let x = try MLMultiArray(
            shape: [1, totalIn as NSNumber, numBins as NSNumber], dataType: .float32
        )
        let ptr = x.dataPointer.assumingMemoryBound(to: Float.self)
        for (i, row) in melWindow.enumerated() {
            for (j, v) in row.enumerated() {
                ptr[i * numBins + j] = v
            }
        }

        var features: [String: Any] = [
            "x": x,
            "cached_embed_left_pad": cachedEmbedLeftPad,
            "processed_lens": processedLens
        ]
        for (name, array) in layerStates {
            features[name] = array
        }
        let input = try MLDictionaryFeatureProvider(dictionary: features)
        let prediction = try encoder.prediction(from: input)

        // Consume new state outputs — names prefixed with `new_`.
        for name in layerStates.keys {
            let outName = "new_\(name)"
            if let next = prediction.featureValue(for: outName)?.multiArrayValue {
                layerStates[name] = next
            }
        }
        if let nextPad = prediction.featureValue(for: "new_cached_embed_left_pad")?.multiArrayValue {
            cachedEmbedLeftPad = nextPad
        }
        if let nextProc = prediction.featureValue(for: "new_processed_lens")?.multiArrayValue {
            processedLens = nextProc
        }

        guard let encOut = prediction.featureValue(for: "encoder_out")?.multiArrayValue else {
            throw AudioModelError.inferenceFailed(operation: "kws_encoder", reason: "missing encoder_out")
        }
        return decodeEncoderOutput(encOut)
    }

    private func decodeEncoderOutput(_ array: MLMultiArray) -> [[Float]] {
        // encoder_out: (1, outputFrames, joinerDim)
        let outputFrames = config.encoder.outputFrames
        let joinerDim = config.encoder.joinerDim
        let ptr = array.dataPointer.assumingMemoryBound(to: Float.self)
        var frames: [[Float]] = []
        frames.reserveCapacity(outputFrames)
        for f in 0..<outputFrames {
            let base = f * joinerDim
            var row = [Float](repeating: 0, count: joinerDim)
            for j in 0..<joinerDim { row[j] = ptr[base + j] }
            frames.append(row)
        }
        return frames
    }

    private static func runDecoder(
        model: MLModel, contextTokens: [Int], contextSize: Int
    ) -> [Float] {
        do {
            let y = try MLMultiArray(shape: [1, contextSize as NSNumber], dataType: .int32)
            for i in 0..<contextSize {
                let raw = i < contextTokens.count ? contextTokens[i] : 0
                y[i] = NSNumber(value: Int32(max(raw, 0)))
            }
            let input = try MLDictionaryFeatureProvider(dictionary: ["y": y])
            let out = try model.prediction(from: input)
            guard let arr = out.featureValue(for: "decoder_out")?.multiArrayValue else { return [] }
            let ptr = arr.dataPointer.assumingMemoryBound(to: Float.self)
            return Array(UnsafeBufferPointer(start: ptr, count: arr.count))
        } catch {
            AudioLog.inference.error("kws decoder step failed: \(error)")
            return []
        }
    }

    private static func runJoiner(
        model: MLModel, encoderFrame: [Float], decoderOut: [Float]
    ) -> [Float] {
        do {
            let enc = try MLMultiArray(shape: [1, encoderFrame.count as NSNumber], dataType: .float16)
            let dec = try MLMultiArray(shape: [1, decoderOut.count as NSNumber], dataType: .float16)
            copyFloatsToFloat16(encoderFrame, into: enc)
            copyFloatsToFloat16(decoderOut, into: dec)
            let input = try MLDictionaryFeatureProvider(
                dictionary: ["encoder_out": enc, "decoder_out": dec]
            )
            let out = try model.prediction(from: input)
            guard let arr = out.featureValue(for: "logits")?.multiArrayValue else { return [] }
            return floatArray(from: arr)
        } catch {
            AudioLog.inference.error("kws joiner step failed: \(error)")
            return []
        }
    }

    private static func copyFloatsToFloat16(_ src: [Float], into array: MLMultiArray) {
        let count = min(src.count, array.count)
        let floats = array.dataPointer.assumingMemoryBound(to: UInt16.self)
        src.withUnsafeBufferPointer { buf in
            for i in 0..<count {
                floats[i] = floatToHalf(buf[i])
            }
        }
    }

    private static func floatArray(from array: MLMultiArray) -> [Float] {
        let count = array.count
        switch array.dataType {
        case .float32:
            let ptr = array.dataPointer.assumingMemoryBound(to: Float.self)
            return Array(UnsafeBufferPointer(start: ptr, count: count))
        case .float16:
            let ptr = array.dataPointer.assumingMemoryBound(to: UInt16.self)
            var out = [Float](repeating: 0, count: count)
            for i in 0..<count { out[i] = halfToFloat(ptr[i]) }
            return out
        default:
            return []
        }
    }

    // Minimal IEEE 754 half <-> single conversion (matching `__fp16`). CoreML
    // expects fp16 inputs on the joiner; we don't take a hard dependency on
    // `_Float16` to keep the module buildable on older Xcodes.
    private static func floatToHalf(_ f: Float) -> UInt16 {
        let bits = f.bitPattern
        let sign = UInt16((bits >> 16) & 0x8000)
        var exp = Int((bits >> 23) & 0xFF) - 127 + 15
        var mant = UInt32(bits & 0x007FFFFF)
        if exp <= 0 {
            if exp < -10 { return sign }
            mant = (mant | 0x00800000) >> UInt32(1 - exp)
            if (mant & 0x00001000) != 0 { mant += 0x00002000 }
            return sign | UInt16(mant >> 13)
        }
        if exp >= 31 { return sign | 0x7C00 }
        if (mant & 0x00001000) != 0 {
            mant += 0x00002000
            if (mant & 0x00800000) != 0 { mant = 0; exp += 1 }
        }
        if exp >= 31 { return sign | 0x7C00 }
        return sign | UInt16(exp << 10) | UInt16(mant >> 13)
    }

    private static func halfToFloat(_ h: UInt16) -> Float {
        let sign = UInt32(h & 0x8000) << 16
        let exp = UInt32(h & 0x7C00) >> 10
        let mant = UInt32(h & 0x03FF)
        if exp == 0 {
            if mant == 0 { return Float(bitPattern: sign) }
            var m = mant
            var e: Int32 = 1
            while (m & 0x0400) == 0 { m <<= 1; e -= 1 }
            m &= 0x03FF
            let bits = sign | UInt32(Int32(127 - 15) + e) << 23 | (m << 13)
            return Float(bitPattern: bits)
        }
        if exp == 31 {
            return Float(bitPattern: sign | 0x7F800000 | (mant << 13))
        }
        let bits = sign | UInt32(Int32(exp) - 15 + 127) << 23 | (mant << 13)
        return Float(bitPattern: bits)
    }
}
