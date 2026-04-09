import CoreML
import Foundation
import AudioCommon

/// A streaming ASR session that processes audio chunks incrementally.
///
/// Matches reference implementation I/O spec:
/// - Encoder: separate pre_cache input/output, [B,D,T] output
/// - Decoder: float32 h/c, [B,D,1] output
/// - Joint: argmax baked in, outputs token_id
public class StreamingSession {
    private let config: ParakeetEOUConfig
    private let encoder: MLModel
    private let decoder: MLModel
    private let joint: MLModel
    private let vocabulary: ParakeetEOUVocabulary
    private let melPreprocessor: StreamingMelPreprocessor
    private let rnntDecoder: RNNTGreedyDecoder

    // Encoder cache state
    private var preCache: MLMultiArray      // [1, 128, preCacheSize]
    private var cacheLastChannel: MLMultiArray
    private var cacheLastTime: MLMultiArray
    private var cacheLastChannelLen: MLMultiArray

    // Decoder LSTM state (float32)
    private var h: MLMultiArray
    private var c: MLMultiArray
    private var decoderOutput: MLMultiArray  // [1, D, 1]

    // Pre-allocated buffers
    private let tokenArray: MLMultiArray
    private let encSlice: MLMultiArray       // [1, D, 1] float32
    private let decoderProvider: ReusableFeatureProvider
    private let jointProvider: ReusableFeatureProvider

    // Accumulated state
    private var allTokens: [Int] = []
    private var allLogProbs: [Float] = []
    private var segmentIndex: Int = 0
    private var eouDetected = false
    private var sampleBuffer: [Float] = []
    private var eouTokenOffset: Int = 0

    /// Whether to use raw mel (no normalization) for streaming.
    public var useRunningNormalization = true

    /// Number of mel frames accumulated for running normalization.
    public var melRunningCount: Int { melPreprocessor.runningCount }

    init(
        config: ParakeetEOUConfig,
        encoder: MLModel,
        decoder: MLModel,
        joint: MLModel,
        vocabulary: ParakeetEOUVocabulary,
        melPreprocessor: StreamingMelPreprocessor
    ) throws {
        self.config = config
        self.encoder = encoder
        self.decoder = decoder
        self.joint = joint
        self.vocabulary = vocabulary
        self.melPreprocessor = melPreprocessor
        self.rnntDecoder = RNNTGreedyDecoder(config: config, decoder: decoder, joint: joint)

        let layers = config.encoderLayers
        let hidden = config.encoderHidden
        let attCtx = config.attentionContext
        let convCache = config.convCacheSize
        let preCacheSize = config.streaming.preCacheSize

        // Encoder caches
        preCache = try MLMultiArray(
            shape: [1, config.numMelBins as NSNumber, preCacheSize as NSNumber], dataType: .float32)
        memset(preCache.dataPointer, 0,
               config.numMelBins * preCacheSize * MemoryLayout<Float>.stride)
        cacheLastChannel = try MLMultiArray(
            shape: [layers, 1, attCtx, hidden] as [NSNumber], dataType: .float32)
        cacheLastTime = try MLMultiArray(
            shape: [layers, 1, hidden, convCache] as [NSNumber], dataType: .float32)
        cacheLastChannelLen = try MLMultiArray(shape: [1], dataType: .int32)
        memset(cacheLastChannel.dataPointer, 0,
               layers * attCtx * hidden * MemoryLayout<Float>.stride)
        memset(cacheLastTime.dataPointer, 0,
               layers * hidden * convCache * MemoryLayout<Float>.stride)
        cacheLastChannelLen[0] = NSNumber(value: Int32(0))

        // Decoder LSTM (float32)
        h = try MLMultiArray(shape: [1, 1, config.decoderHidden] as [NSNumber], dataType: .float32)
        c = try MLMultiArray(shape: [1, 1, config.decoderHidden] as [NSNumber], dataType: .float32)
        memset(h.dataPointer, 0, config.decoderHidden * MemoryLayout<Float>.stride)
        memset(c.dataPointer, 0, config.decoderHidden * MemoryLayout<Float>.stride)

        // Prime decoder with blank token
        tokenArray = try MLMultiArray(shape: [1, 1], dataType: .int32)
        tokenArray.dataPointer.assumingMemoryBound(to: Int32.self).pointee = Int32(config.blankTokenId)

        decoderProvider = ReusableFeatureProvider([
            "targets": tokenArray, "h_in": h, "c_in": c,
        ])
        let initOut = try decoder.prediction(from: decoderProvider)
        decoderOutput = initOut.featureValue(for: "decoder")!.multiArrayValue!
        h = initOut.featureValue(for: "h_out")!.multiArrayValue!
        c = initOut.featureValue(for: "c_out")!.multiArrayValue!

        // Encoder slice [1, D, 1] float32 and joint provider
        encSlice = try MLMultiArray(shape: [1, hidden as NSNumber, 1], dataType: .float32)
        jointProvider = ReusableFeatureProvider([
            "encoder_step": encSlice, "decoder_step": decoderOutput,
        ])
    }

    // MARK: - Push Audio

    public func pushAudio(_ samples: [Float]) throws -> [ParakeetStreamingASRModel.PartialTranscript] {
        sampleBuffer.append(contentsOf: samples)

        let samplesPerChunk = (config.streaming.melFrames - 1) * config.hopLength
        var results: [ParakeetStreamingASRModel.PartialTranscript] = []

        while sampleBuffer.count >= samplesPerChunk {
            let chunk = Array(sampleBuffer.prefix(samplesPerChunk))
            sampleBuffer.removeFirst(samplesPerChunk)
            if let partial = try processChunk(chunk) { results.append(partial) }
        }

        return results
    }

    public func finalize() throws -> [ParakeetStreamingASRModel.PartialTranscript] {
        var results: [ParakeetStreamingASRModel.PartialTranscript] = []

        if !sampleBuffer.isEmpty {
            let samplesPerChunk = (config.streaming.melFrames - 1) * config.hopLength
            let padded = sampleBuffer + [Float](repeating: 0, count: max(0, samplesPerChunk - sampleBuffer.count))
            sampleBuffer.removeAll()
            if let partial = try processChunk(Array(padded.prefix(samplesPerChunk))) {
                results.append(partial)
            }
        }

        if !allTokens.isEmpty {
            let currentTokens = Array(allTokens[eouTokenOffset...])
            let text = vocabulary.decode(currentTokens)
            if !text.isEmpty {
                let currentLogProbs = Array(allLogProbs[eouTokenOffset...])
                let confidence: Float = currentLogProbs.isEmpty ? 0 :
                    min(1.0, exp(currentLogProbs.reduce(0, +) / Float(currentLogProbs.count)))
                results.append(ParakeetStreamingASRModel.PartialTranscript(
                    text: text, isFinal: true, confidence: confidence,
                    eouDetected: eouDetected, segmentIndex: segmentIndex))
            }
        }

        return results
    }

    // MARK: - Internal

    private func processChunk(_ audio: [Float]) throws -> ParakeetStreamingASRModel.PartialTranscript? {
        let (rawMel, melLength): (MLMultiArray, Int)
        if useRunningNormalization {
            (rawMel, melLength) = try melPreprocessor.extractRaw(audio)
        } else {
            (rawMel, melLength) = try melPreprocessor.extract(audio)
        }
        guard melLength > 0 else { return nil }

        let expectedFrames = config.streaming.melFrames
        let actualMelFrames = rawMel.shape[2].intValue
        let mel: MLMultiArray
        if actualMelFrames > expectedFrames {
            mel = try truncateMel(rawMel, to: expectedFrames)
        } else if actualMelFrames < expectedFrames {
            mel = try padMel(rawMel, actualLength: actualMelFrames, targetLength: expectedFrames)
        } else {
            mel = rawMel
        }

        // Run encoder with separate pre_cache input
        let encoderInput = try MLDictionaryFeatureProvider(dictionary: [
            "audio_signal": MLFeatureValue(multiArray: mel),
            "audio_length": MLFeatureValue(multiArray: makeInt32Array(value: Int32(expectedFrames))),
            "pre_cache": MLFeatureValue(multiArray: preCache),
            "cache_last_channel": MLFeatureValue(multiArray: cacheLastChannel),
            "cache_last_time": MLFeatureValue(multiArray: cacheLastTime),
            "cache_last_channel_len": MLFeatureValue(multiArray: cacheLastChannelLen),
        ])

        let encoderOutput = try encoder.prediction(from: encoderInput)

        // Encoder output is [B, D, T]
        let encoded = encoderOutput.featureValue(for: "encoded_output")!.multiArrayValue!
        let reportedLength = encoderOutput.featureValue(for: "encoded_length")!.multiArrayValue![0].intValue
        let actualFrames = encoded.shape[2].intValue
        // Only decode the LAST validOutputFrames — first frames are from pre-cache context
        let validOut = config.streaming.outputFrames
        let encodedLength = min(validOut, actualFrames)
        let encodedStartFrame = max(0, actualFrames - validOut)

        // Update all caches including pre_cache loopback
        preCache = encoderOutput.featureValue(for: "new_pre_cache")!.multiArrayValue!
        cacheLastChannel = encoderOutput.featureValue(for: "new_cache_last_channel")!.multiArrayValue!
        cacheLastTime = encoderOutput.featureValue(for: "new_cache_last_time")!.multiArrayValue!
        cacheLastChannelLen = encoderOutput.featureValue(for: "new_cache_last_channel_len")!.multiArrayValue!

        // Debug encoder output
        print("[DBG] encoder: shape=\(encoded.shape) dtype=\(encoded.dataType.rawValue) len=\(encodedLength) reported=\(reportedLength)")
        if encoded.dataType == .float16 {
            let ptr = encoded.dataPointer.assumingMemoryBound(to: Float16.self)
            let total = min(10, encoded.count)
            let vals = (0..<total).map { Float(ptr[$0]) }
            print("[DBG] first10: \(vals.map { String(format: "%.3f", $0) })")
        }

        guard encodedLength > 0 else { return nil }

        // RNNT decode — start from encodedStartFrame (skip pre-cache context frames)
        let result = try rnntDecoder.decode(
            encoded: encoded,
            encodedLength: encodedLength,
            startFrame: encodedStartFrame,
            h: &h, c: &c,
            decoderOutput: &decoderOutput,
            decoderProvider: decoderProvider,
            jointProvider: jointProvider,
            tokenArray: tokenArray,
            encSlice: encSlice
        )

        allTokens.append(contentsOf: result.tokens)
        allLogProbs.append(contentsOf: result.tokenLogProbs)
        if result.eouDetected { eouDetected = true }

        let currentTokens = Array(allTokens[eouTokenOffset...])
        let text = vocabulary.decode(currentTokens)
        guard !text.isEmpty else { return nil }

        let currentLogProbs = Array(allLogProbs[eouTokenOffset...])
        let confidence: Float = currentLogProbs.isEmpty ? 0 :
            min(1.0, exp(currentLogProbs.reduce(0, +) / Float(currentLogProbs.count)))

        if eouDetected {
            let partial = ParakeetStreamingASRModel.PartialTranscript(
                text: text, isFinal: true, confidence: confidence,
                eouDetected: true, segmentIndex: segmentIndex)
            eouTokenOffset = allTokens.count
            segmentIndex += 1
            eouDetected = false
            return partial
        }

        return ParakeetStreamingASRModel.PartialTranscript(
            text: text, isFinal: false, confidence: confidence,
            eouDetected: false, segmentIndex: segmentIndex)
    }

    // MARK: - Helpers

    private func makeInt32Array(value: Int32) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: [1], dataType: .int32)
        array[0] = NSNumber(value: value)
        return array
    }

    private func truncateMel(_ mel: MLMultiArray, to targetFrames: Int) throws -> MLMultiArray {
        let numMelBins = config.numMelBins
        let stride = MemoryLayout<Float>.stride
        let truncated = try MLMultiArray(
            shape: [1, numMelBins as NSNumber, targetFrames as NSNumber], dataType: mel.dataType)
        let actualFrames = mel.shape[2].intValue
        for bin in 0..<numMelBins {
            memcpy(truncated.dataPointer.advanced(by: bin * targetFrames * stride),
                   mel.dataPointer.advanced(by: bin * actualFrames * stride),
                   targetFrames * stride)
        }
        return truncated
    }

    private func padMel(_ mel: MLMultiArray, actualLength: Int, targetLength: Int) throws -> MLMultiArray {
        let numMelBins = config.numMelBins
        let stride = MemoryLayout<Float>.stride
        let padded = try MLMultiArray(
            shape: [1, numMelBins as NSNumber, targetLength as NSNumber], dataType: mel.dataType)
        for bin in 0..<numMelBins {
            memcpy(padded.dataPointer.advanced(by: bin * targetLength * stride),
                   mel.dataPointer.advanced(by: bin * actualLength * stride),
                   actualLength * stride)
            memset(padded.dataPointer.advanced(by: (bin * targetLength + actualLength) * stride), 0,
                   (targetLength - actualLength) * stride)
        }
        return padded
    }
}
