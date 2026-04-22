import CoreML
import Foundation
import AudioCommon

/// A Nemotron streaming ASR session that processes audio chunks incrementally
/// and emits partial transcripts as tokens are decoded. End of stream is
/// signaled by calling `finalize()`.
public class StreamingSession {
    private let config: NemotronStreamingConfig
    private let encoder: MLModel
    private let decoder: MLModel
    private let joint: MLModel
    private let vocabulary: NemotronVocabulary
    private let melPreprocessor: StreamingMelPreprocessor
    private let rnntDecoder: RNNTGreedyDecoder

    private var cacheLastChannel: MLMultiArray
    private var cacheLastTime: MLMultiArray
    private var cacheLastChannelLen: MLMultiArray
    private var preCache: MLMultiArray

    private var h: MLMultiArray
    private var c: MLMultiArray
    private var decoderOutput: MLMultiArray

    private let tokenArray: MLMultiArray
    private let encSlice: MLMultiArray
    private let argmaxBuf: UnsafeMutablePointer<Float>
    private let decoderProvider: ReusableFeatureProvider
    private let jointProvider: ReusableFeatureProvider

    private var allTokens: [Int] = []
    private var allLogProbs: [Float] = []
    private var sampleBuffer: [Float] = []
    private var segmentIndex: Int = 0

    init(
        config: NemotronStreamingConfig,
        encoder: MLModel,
        decoder: MLModel,
        joint: MLModel,
        vocabulary: NemotronVocabulary,
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
        let numMelBins = config.numMelBins

        preCache = try MLMultiArray(
            shape: [1, numMelBins as NSNumber, preCacheSize as NSNumber], dataType: .float32)
        memset(preCache.dataPointer, 0, numMelBins * preCacheSize * MemoryLayout<Float>.stride)

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

        let decLayers = config.decoderLayers
        let decHidden = config.decoderHidden
        h = try MLMultiArray(shape: [decLayers, 1, decHidden] as [NSNumber], dataType: .float32)
        c = try MLMultiArray(shape: [decLayers, 1, decHidden] as [NSNumber], dataType: .float32)
        memset(h.dataPointer, 0, decLayers * decHidden * MemoryLayout<Float>.stride)
        memset(c.dataPointer, 0, decLayers * decHidden * MemoryLayout<Float>.stride)

        decoderOutput = try MLMultiArray(
            shape: [1, 1, decHidden as NSNumber], dataType: .float32)
        memset(decoderOutput.dataPointer, 0, decHidden * MemoryLayout<Float>.stride)

        tokenArray = try MLMultiArray(shape: [1, 1], dataType: .int32)
        let tokenPtr = tokenArray.dataPointer.assumingMemoryBound(to: Int32.self)
        tokenPtr.pointee = Int32(config.blankTokenId)

        decoderProvider = ReusableFeatureProvider(["token": tokenArray, "h": h, "c": c])
        let initOut = try decoder.prediction(from: decoderProvider)
        Self.copyCastFP16ToFP32(initOut.featureValue(for: "decoder_output")!.multiArrayValue!,
                                into: decoderOutput)
        Self.copyCastFP16ToFP32(initOut.featureValue(for: "h_out")!.multiArrayValue!, into: h)
        Self.copyCastFP16ToFP32(initOut.featureValue(for: "c_out")!.multiArrayValue!, into: c)

        encSlice = try MLMultiArray(shape: [1, 1, hidden as NSNumber], dataType: .float32)
        jointProvider = ReusableFeatureProvider([
            "encoder_output": encSlice, "decoder_output": decoderOutput,
        ])

        argmaxBuf = .allocate(capacity: config.vocabSize + 1)
    }

    deinit {
        argmaxBuf.deallocate()
    }

    public func pushAudio(_ samples: [Float]) throws -> [NemotronStreamingASRModel.PartialTranscript] {
        sampleBuffer.append(contentsOf: samples)

        let samplesPerChunk = config.streaming.melFrames * config.hopLength
        let shiftMelFrames = config.streaming.outputFrames * config.subsamplingFactor
        let shiftSamples = shiftMelFrames * config.hopLength
        var results: [NemotronStreamingASRModel.PartialTranscript] = []

        while sampleBuffer.count >= samplesPerChunk {
            let chunk = Array(sampleBuffer.prefix(samplesPerChunk))
            let drop = min(shiftSamples, sampleBuffer.count)
            sampleBuffer.removeFirst(drop)

            if let partial = try processChunk(chunk) {
                results.append(partial)
            }
        }

        return results
    }

    public func finalize() throws -> [NemotronStreamingASRModel.PartialTranscript] {
        if !sampleBuffer.isEmpty {
            let samplesPerChunk = config.streaming.melFrames * config.hopLength
            let padded = sampleBuffer + [Float](repeating: 0, count: max(0, samplesPerChunk - sampleBuffer.count))
            sampleBuffer.removeAll()
            _ = try processChunk(Array(padded.prefix(samplesPerChunk)))
        }

        guard !allTokens.isEmpty else { return [] }

        let text = vocabulary.decode(allTokens)
        let confidence: Float
        if !allLogProbs.isEmpty {
            let mean = allLogProbs.reduce(0, +) / Float(allLogProbs.count)
            confidence = min(1.0, exp(mean))
        } else {
            confidence = 0
        }
        return [NemotronStreamingASRModel.PartialTranscript(
            text: text,
            isFinal: true,
            confidence: confidence,
            segmentIndex: segmentIndex
        )]
    }

    private func processChunk(_ audio: [Float]) throws -> NemotronStreamingASRModel.PartialTranscript? {
        let (rawMel, melLength) = try melPreprocessor.extractRaw(audio)
        guard melLength > 0 else { return nil }

        let expectedFrames = config.streaming.melFrames
        let actualMelFrames = rawMel.shape[2].intValue
        let chunkMel: MLMultiArray
        if actualMelFrames > expectedFrames {
            chunkMel = try truncateMel(rawMel, to: expectedFrames)
        } else if actualMelFrames < expectedFrames {
            chunkMel = try padMel(rawMel, actualLength: actualMelFrames, targetLength: expectedFrames)
        } else {
            chunkMel = rawMel
        }

        let encoderInput = try MLDictionaryFeatureProvider(dictionary: [
            "audio_signal": MLFeatureValue(multiArray: chunkMel),
            "audio_length": MLFeatureValue(multiArray: makeInt32Array(value: Int32(expectedFrames))),
            "pre_cache": MLFeatureValue(multiArray: preCache),
            "cache_last_channel": MLFeatureValue(multiArray: cacheLastChannel),
            "cache_last_time": MLFeatureValue(multiArray: cacheLastTime),
            "cache_last_channel_len": MLFeatureValue(multiArray: cacheLastChannelLen),
        ])
        let encoderOutput = try encoder.prediction(from: encoderInput)

        let encoded = encoderOutput.featureValue(for: "encoded_output")!.multiArrayValue!
        let reportedLength = encoderOutput.featureValue(for: "encoded_length")!.multiArrayValue![0].intValue
        let actualFrames = encoded.shape[1].intValue
        let totalFrames = min(reportedLength, actualFrames)
        let encodedLength = min(config.streaming.outputFrames, totalFrames)

        preCache = encoderOutput.featureValue(for: "new_pre_cache")!.multiArrayValue!
        cacheLastChannel = encoderOutput.featureValue(for: "new_cache_last_channel")!.multiArrayValue!
        cacheLastTime = encoderOutput.featureValue(for: "new_cache_last_time")!.multiArrayValue!
        cacheLastChannelLen = encoderOutput.featureValue(for: "new_cache_last_channel_len")!.multiArrayValue!

        guard encodedLength > 0 else { return nil }

        let result = try rnntDecoder.decode(
            encoded: encoded,
            encodedLength: encodedLength,
            frameOffset: 0,
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

        let text = vocabulary.decode(allTokens)
        guard !text.isEmpty else { return nil }

        let confidence: Float
        if !allLogProbs.isEmpty {
            let mean = allLogProbs.reduce(0, +) / Float(allLogProbs.count)
            confidence = min(1.0, exp(mean))
        } else {
            confidence = 0
        }

        return NemotronStreamingASRModel.PartialTranscript(
            text: text,
            isFinal: false,
            confidence: confidence,
            segmentIndex: segmentIndex
        )
    }

    private func makeInt32Array(value: Int32) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: [1], dataType: .int32)
        array[0] = NSNumber(value: value)
        return array
    }

    static func copyCastFP16ToFP32(_ src: MLMultiArray, into dst: MLMultiArray) {
        let count = src.count
        let srcPtr = src.dataPointer.assumingMemoryBound(to: Float16.self)
        let dstPtr = dst.dataPointer.assumingMemoryBound(to: Float.self)
        for i in 0..<count { dstPtr[i] = Float(srcPtr[i]) }
    }

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
