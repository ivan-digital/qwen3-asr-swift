import CoreML
import Foundation
import AudioCommon

/// A streaming ASR session that processes audio chunks incrementally.
///
/// Maintains encoder cache state and LSTM decoder state between chunks.
/// Emits partial transcripts as tokens are decoded, and detects end-of-utterance
/// via the `<EOU>` token from the RNNT joint network.
public class StreamingSession {
    private let config: ParakeetEOUConfig
    private let encoder: MLModel
    private let decoder: MLModel
    private let joint: MLModel
    private let vocabulary: ParakeetEOUVocabulary
    private let melPreprocessor: StreamingMelPreprocessor
    private let rnntDecoder: RNNTGreedyDecoder

    // Encoder cache state
    private var preCache: MLMultiArray
    private var cacheLastChannel: MLMultiArray
    private var cacheLastTime: MLMultiArray
    private var cacheLastChannelLen: MLMultiArray

    // Decoder LSTM state
    private var h: MLMultiArray
    private var c: MLMultiArray
    private var decoderOutput: MLMultiArray

    // Pre-allocated buffers
    private let tokenArray: MLMultiArray
    private let encSlice: MLMultiArray
    private let argmaxBuf: UnsafeMutablePointer<Float>
    private let decoderProvider: ReusableFeatureProvider
    private let jointProvider: ReusableFeatureProvider

    // Accumulated state
    private var allTokens: [Int] = []
    private var allLogProbs: [Float] = []
    private var segmentIndex: Int = 0
    private var eouDetected = false
    private var sampleBuffer: [Float] = []

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

        // Initialize encoder caches to zero
        let layers = config.encoderLayers
        let hidden = config.encoderHidden
        let attCtx = config.attentionContext
        let convCache = config.convCacheSize
        let preCacheSize = config.streaming.preCacheSize

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
               layers * 1 * attCtx * hidden * MemoryLayout<Float>.stride)
        memset(cacheLastTime.dataPointer, 0,
               layers * 1 * hidden * convCache * MemoryLayout<Float>.stride)
        cacheLastChannelLen[0] = NSNumber(value: Int32(0))

        // Initialize LSTM state
        let decLayers = config.decoderLayers
        let decHidden = config.decoderHidden

        h = try MLMultiArray(shape: [decLayers, 1, decHidden] as [NSNumber], dataType: .float16)
        c = try MLMultiArray(shape: [decLayers, 1, decHidden] as [NSNumber], dataType: .float16)
        memset(h.dataPointer, 0, decLayers * decHidden * MemoryLayout<Float16>.stride)
        memset(c.dataPointer, 0, decLayers * decHidden * MemoryLayout<Float16>.stride)

        // Prime decoder with blank token
        tokenArray = try MLMultiArray(shape: [1, 1], dataType: .int32)
        let tokenPtr = tokenArray.dataPointer.assumingMemoryBound(to: Int32.self)
        tokenPtr.pointee = Int32(config.blankTokenId)

        decoderProvider = ReusableFeatureProvider(["token": tokenArray, "h": h, "c": c])
        let initOut = try decoder.prediction(from: decoderProvider)
        decoderOutput = initOut.featureValue(for: "decoder_output")!.multiArrayValue!
        h = initOut.featureValue(for: "h_out")!.multiArrayValue!
        c = initOut.featureValue(for: "c_out")!.multiArrayValue!

        // Encoder slice and joint provider
        encSlice = try MLMultiArray(shape: [1, 1, hidden as NSNumber], dataType: .float16)
        jointProvider = ReusableFeatureProvider([
            "encoder_output": encSlice, "decoder_output": decoderOutput,
        ])

        // Argmax buffer
        argmaxBuf = .allocate(capacity: config.vocabSize + 1)
    }

    deinit {
        argmaxBuf.deallocate()
    }

    // MARK: - Push Audio

    /// Push a chunk of audio samples and get any new partial transcripts.
    ///
    /// Samples are buffered internally. When enough samples accumulate for a
    /// full mel chunk, the encoder and decoder run and partial results are returned.
    public func pushAudio(_ samples: [Float]) throws -> [ParakeetStreamingASRModel.PartialTranscript] {
        guard !eouDetected else { return [] }

        sampleBuffer.append(contentsOf: samples)

        let samplesPerChunk = config.streaming.melFrames * config.hopLength
        var results: [ParakeetStreamingASRModel.PartialTranscript] = []

        while sampleBuffer.count >= samplesPerChunk {
            let chunk = Array(sampleBuffer.prefix(samplesPerChunk))
            sampleBuffer.removeFirst(samplesPerChunk)

            let partial = try processChunk(chunk)
            if let partial { results.append(partial) }

            if eouDetected { break }
        }

        return results
    }

    /// Signal end of audio stream and return any remaining transcription.
    public func finalize() throws -> [ParakeetStreamingASRModel.PartialTranscript] {
        var results: [ParakeetStreamingASRModel.PartialTranscript] = []

        // Process remaining buffered samples
        if !sampleBuffer.isEmpty && !eouDetected {
            // Pad to full chunk size
            let samplesPerChunk = config.streaming.melFrames * config.hopLength
            let padded = sampleBuffer + [Float](repeating: 0, count: max(0, samplesPerChunk - sampleBuffer.count))
            sampleBuffer.removeAll()
            if let partial = try processChunk(Array(padded.prefix(samplesPerChunk))) {
                results.append(partial)
            }
        }

        // Emit final transcript
        if !allTokens.isEmpty {
            let text = vocabulary.decode(allTokens)
            let confidence: Float
            if !allLogProbs.isEmpty {
                let mean = allLogProbs.reduce(0, +) / Float(allLogProbs.count)
                confidence = min(1.0, exp(mean))
            } else {
                confidence = 0
            }
            results.append(ParakeetStreamingASRModel.PartialTranscript(
                text: text,
                isFinal: true,
                confidence: confidence,
                eouDetected: eouDetected,
                segmentIndex: segmentIndex
            ))
        }

        return results
    }

    // MARK: - Internal

    private func processChunk(_ audio: [Float]) throws -> ParakeetStreamingASRModel.PartialTranscript? {
        // Extract mel spectrogram
        let (rawMel, melLength) = try melPreprocessor.extract(audio)
        guard melLength > 0 else { return nil }

        // Truncate mel to exact expected frame count (encoder has fixed input shape)
        let expectedFrames = config.streaming.melFrames
        let mel: MLMultiArray
        if melLength > expectedFrames {
            mel = try truncateMel(rawMel, to: expectedFrames)
        } else if melLength < expectedFrames {
            mel = try padMel(rawMel, actualLength: melLength, targetLength: expectedFrames)
        } else {
            mel = rawMel
        }

        // Run cache-aware encoder (with pre_cache for mel subsampling context)
        let encoderInput = try MLDictionaryFeatureProvider(dictionary: [
            "audio_signal": MLFeatureValue(multiArray: mel),
            "audio_length": MLFeatureValue(multiArray: makeInt32Array(value: Int32(melLength))),
            "pre_cache": MLFeatureValue(multiArray: preCache),
            "cache_last_channel": MLFeatureValue(multiArray: cacheLastChannel),
            "cache_last_time": MLFeatureValue(multiArray: cacheLastTime),
            "cache_last_channel_len": MLFeatureValue(multiArray: cacheLastChannelLen),
        ])

        let encoderOutput = try encoder.prediction(from: encoderInput)

        let encoded = encoderOutput.featureValue(for: "encoded_output")!.multiArrayValue!
        let reportedLength = encoderOutput.featureValue(for: "encoded_length")!.multiArrayValue![0].intValue
        // Encoder output is [B, T, D] — frame count is shape[1]
        let actualFrames = encoded.shape[1].intValue
        let encodedLength = min(reportedLength, actualFrames)

        // Update encoder caches
        cacheLastChannel = encoderOutput.featureValue(for: "new_cache_last_channel")!.multiArrayValue!
        cacheLastTime = encoderOutput.featureValue(for: "new_cache_last_time")!.multiArrayValue!
        cacheLastChannelLen = encoderOutput.featureValue(for: "new_cache_last_channel_len")!.multiArrayValue!

        AudioLog.inference.debug("EOU encoder: encodedLength=\(encodedLength), shape=\(encoded.shape)")

        guard encodedLength > 0 else { return nil }

        // RNNT greedy decode
        let result = try rnntDecoder.decode(
            encoded: encoded,
            encodedLength: encodedLength,
            h: &h,
            c: &c,
            decoderOutput: &decoderOutput,
            decoderProvider: decoderProvider,
            jointProvider: jointProvider,
            tokenArray: tokenArray,
            encSlice: encSlice,
            argmaxBuf: argmaxBuf
        )

        allTokens.append(contentsOf: result.tokens)
        allLogProbs.append(contentsOf: result.tokenLogProbs)

        if result.eouDetected {
            eouDetected = true
        }

        // Emit partial transcript
        let text = vocabulary.decode(allTokens)
        guard !text.isEmpty else { return nil }

        let confidence: Float
        if !allLogProbs.isEmpty {
            let mean = allLogProbs.reduce(0, +) / Float(allLogProbs.count)
            confidence = min(1.0, exp(mean))
        } else {
            confidence = 0
        }

        if eouDetected {
            let partial = ParakeetStreamingASRModel.PartialTranscript(
                text: text,
                isFinal: true,
                confidence: confidence,
                eouDetected: true,
                segmentIndex: segmentIndex
            )
            // Reset for next utterance
            allTokens.removeAll()
            allLogProbs.removeAll()
            segmentIndex += 1
            eouDetected = false
            return partial
        }

        return ParakeetStreamingASRModel.PartialTranscript(
            text: text,
            isFinal: false,
            confidence: confidence,
            eouDetected: false,
            segmentIndex: segmentIndex
        )
    }

    private func makeInt32Array(value: Int32) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: [1], dataType: .int32)
        array[0] = NSNumber(value: value)
        return array
    }

    /// Truncate mel to exactly `targetFrames` frames.
    private func truncateMel(_ mel: MLMultiArray, to targetFrames: Int) throws -> MLMultiArray {
        let numMelBins = config.numMelBins
        let stride = mel.dataType == .float16 ? MemoryLayout<Float16>.stride : MemoryLayout<Float>.stride
        let truncated = try MLMultiArray(
            shape: [1, numMelBins as NSNumber, targetFrames as NSNumber], dataType: mel.dataType)
        let actualFrames = mel.shape[2].intValue
        for bin in 0..<numMelBins {
            let srcOffset = bin * actualFrames * stride
            let dstOffset = bin * targetFrames * stride
            memcpy(truncated.dataPointer.advanced(by: dstOffset),
                   mel.dataPointer.advanced(by: srcOffset),
                   targetFrames * stride)
        }
        return truncated
    }

    /// Pad mel to `targetLength` frames with zeros.
    private func padMel(_ mel: MLMultiArray, actualLength: Int, targetLength: Int) throws -> MLMultiArray {
        let numMelBins = config.numMelBins
        let stride = mel.dataType == .float16 ? MemoryLayout<Float16>.stride : MemoryLayout<Float>.stride
        let padded = try MLMultiArray(
            shape: [1, numMelBins as NSNumber, targetLength as NSNumber], dataType: mel.dataType)
        for bin in 0..<numMelBins {
            let srcOffset = bin * actualLength * stride
            let dstOffset = bin * targetLength * stride
            memcpy(padded.dataPointer.advanced(by: dstOffset),
                   mel.dataPointer.advanced(by: srcOffset),
                   actualLength * stride)
            memset(padded.dataPointer.advanced(by: dstOffset + actualLength * stride), 0,
                   (targetLength - actualLength) * stride)
        }
        return padded
    }
}
