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
    private let fbankSession: KaldiFbank.StreamingSession
    private let kwsDecoder: StreamingKwsDecoder

    // Per-frame mel features buffered for the encoder sliding window. Stores
    // only frames that have not yet been consumed by an encoder chunk.
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
        self.fbankSession = KaldiFbank.StreamingSession(fbank)

        var states = [String: MLMultiArray]()
        for (name, shape) in zip(config.encoder.layerStateNames, config.encoder.layerStateShapes) {
            let array = try MLMultiArray(
                shape: shape.map { NSNumber(value: $0) }, dataType: .float32
            )
            memset(array.dataPointer, 0, array.count * MemoryLayout<Float>.stride)
            states[name] = array
        }
        self.layerStates = states
        let embedPad = try MLMultiArray(
            shape: config.encoder.cachedEmbedLeftPadShape.map { NSNumber(value: $0) },
            dataType: .float32
        )
        memset(embedPad.dataPointer, 0, embedPad.count * MemoryLayout<Float>.stride)
        self.cachedEmbedLeftPad = embedPad
        self.processedLens = try MLMultiArray(shape: [1], dataType: .int32)
        processedLens.dataPointer.assumingMemoryBound(to: Int32.self)[0] = 0

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
        fbankSession.reset()
        melBuffer.removeAll(keepingCapacity: true)
        for name in layerStates.keys {
            let array = layerStates[name]!
            memset(array.dataPointer, 0, array.count * MemoryLayout<Float>.stride)
        }
        memset(cachedEmbedLeftPad.dataPointer, 0,
               cachedEmbedLeftPad.count * MemoryLayout<Float>.stride)
        processedLens.dataPointer.assumingMemoryBound(to: Int32.self)[0] = 0
        kwsDecoder.reset()
    }

    /// Push raw PCM and return any keyword detections that fired.
    public func pushAudio(_ samples: [Float]) throws -> [KeywordDetection] {
        appendMelFrames(fbankSession.accept(samples))
        return try drainEncoderChunks()
    }

    /// Flush remaining audio and surface any final detections.
    public func finalize() throws -> [KeywordDetection] {
        // Release any trailing mel frames the streaming fbank held back for
        // mirror-padding, then zero-pad the mel buffer up to a full encoder
        // window so the last chunk can be processed.
        appendMelFrames(fbankSession.flush())
        var emissions = try drainEncoderChunks()
        let totalIn = config.encoder.totalInputFrames
        if melBuffer.count > 0 && melBuffer.count < totalIn {
            let numBins = config.feature.numMelBins
            let pad = [Float](repeating: -15.0, count: numBins)  // ~kaldi silence
            while melBuffer.count < totalIn {
                melBuffer.append(pad)
            }
            emissions.append(contentsOf: try drainEncoderChunks())
        }
        return emissions
    }

    private func appendMelFrames(_ rowMajor: [Float]) {
        guard !rowMajor.isEmpty else { return }
        let numBins = config.feature.numMelBins
        let count = rowMajor.count / numBins
        melBuffer.reserveCapacity(melBuffer.count + count)
        for f in 0..<count {
            let base = f * numBins
            melBuffer.append(Array(rowMajor[base..<(base + numBins)]))
        }
    }

    private func drainEncoderChunks() throws -> [KeywordDetection] {
        var emissions: [KeywordDetection] = []
        let totalIn = config.encoder.totalInputFrames      // 45
        // Each encoder chunk consumes ``chunkSize * 2`` fresh mel frames; the
        // trailing 13 frames (``PAD_LENGTH`` in the Python export) overlap with
        // the next call and are reabsorbed by ``cached_embed_left_pad`` state.
        let stride = config.encoder.chunkSize * 2          // 32
        while melBuffer.count >= totalIn {
            let window = Array(melBuffer.prefix(totalIn))
            let encoderFrames = try runEncoder(melWindow: window)
            emissions.append(contentsOf: kwsDecoder.stepChunk(encoderFrames))
            melBuffer.removeFirst(stride)
        }
        return emissions
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
        // encoder_out: (1, outputFrames, joinerDim). Some CoreML backends
        // return fp16 here even though the input spec is fp32 — handle both.
        let outputFrames = config.encoder.outputFrames
        let joinerDim = config.encoder.joinerDim
        var frames: [[Float]] = []
        frames.reserveCapacity(outputFrames)

        let count = array.count
        let floats: [Float]
        switch array.dataType {
        case .float32:
            let ptr = array.dataPointer.assumingMemoryBound(to: Float.self)
            floats = Array(UnsafeBufferPointer(start: ptr, count: count))
        case .float16:
            let ptr = array.dataPointer.assumingMemoryBound(to: Float16.self)
            var out = [Float](repeating: 0, count: count)
            for i in 0..<count { out[i] = Float(ptr[i]) }
            floats = out
        default:
            floats = []
        }
        for f in 0..<outputFrames {
            let base = f * joinerDim
            frames.append(Array(floats[base..<(base + joinerDim)]))
        }
        return frames
    }

    static func runDecoder(
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
            // decoder_out is fp16 in the shipped CoreML bundle even though the
            // export spec declared fp32. Use the generic reader that handles both.
            return floatArray(from: arr)
        } catch {
            AudioLog.inference.error("kws decoder step failed: \(error)")
            return []
        }
    }

    static func runJoiner(
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
        let halves = array.dataPointer.assumingMemoryBound(to: Float16.self)
        src.withUnsafeBufferPointer { buf in
            for i in 0..<count {
                halves[i] = Float16(buf[i])
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
            let ptr = array.dataPointer.assumingMemoryBound(to: Float16.self)
            var out = [Float](repeating: 0, count: count)
            for i in 0..<count { out[i] = Float(ptr[i]) }
            return out
        default:
            return []
        }
    }

    fileprivate static func halfToFloat(_ h: Float16) -> Float { Float(h) }
}
