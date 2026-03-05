import Foundation
import CoreML
import AudioCommon

/// Speech enhancement using DeepFilterNet3 on Core ML (FP16, Neural Engine).
///
/// Removes background noise from speech audio. The neural network runs on
/// Apple's Neural Engine via Core ML, while signal processing (STFT, ERB
/// filterbank, deep filtering) runs on CPU via Accelerate/vDSP.
///
/// ```swift
/// let enhancer = try await SpeechEnhancer.fromPretrained()
/// let cleanAudio = try enhancer.enhance(audio: noisyAudio, sampleRate: 48000)
/// ```
public final class SpeechEnhancer {

    /// Default HuggingFace model ID
    public static let defaultModelId = "aufklarer/DeepFilterNet3-CoreML"

    /// Native sample rate (48kHz)
    public static let sampleRate = 48000

    let config: DeepFilterNet3Config
    let network: DeepFilterNet3Network
    let stft: STFTProcessor

    // ERB filterbank matrices
    let erbFb: [Float]       // [freqBins, erbBands]
    let erbInvFb: [Float]    // [erbBands, freqBins]

    // Normalization state
    var meanNormState: [Float]
    var unitNormState: [Float]
    let meanNormStateInit: [Float]
    let unitNormStateInit: [Float]

    // STFT streaming state
    var analysisMem: [Float]
    var synthesisMem: [Float]

    init(
        network: DeepFilterNet3Network,
        config: DeepFilterNet3Config,
        auxData: DeepFilterNet3WeightLoader.AuxiliaryData
    ) {
        self.network = network
        self.config = config

        self.stft = STFTProcessor(
            fftSize: config.fftSize,
            hopSize: config.hopSize,
            window: auxData.window)

        self.erbFb = auxData.erbFb
        self.erbInvFb = auxData.erbInvFb

        self.meanNormState = auxData.meanNormState
        self.unitNormState = auxData.unitNormState
        self.meanNormStateInit = auxData.meanNormState
        self.unitNormStateInit = auxData.unitNormState

        self.analysisMem = [Float](repeating: 0, count: config.fftSize - config.hopSize)
        self.synthesisMem = [Float](repeating: 0, count: config.fftSize - config.hopSize)
    }

    /// Enhance audio by removing noise (batch mode).
    ///
    /// Processes the entire audio at once. Signal processing runs on CPU (vDSP),
    /// neural network inference runs via Core ML.
    ///
    /// - Parameters:
    ///   - audio: Input audio samples (mono, Float32)
    ///   - sampleRate: Sample rate of input audio (resampled to 48kHz if needed)
    /// - Returns: Enhanced audio at 48kHz
    public func enhance(audio: [Float], sampleRate: Int) throws -> [Float] {
        var samples = audio
        if sampleRate != Self.sampleRate {
            samples = resample(audio, from: sampleRate, to: Self.sampleRate)
        }

        resetState()

        let hop = config.hopSize
        let freqBins = config.freqBins  // 481 (960/2 + 1)

        // Pad audio so inverse STFT captures the full signal
        let paddedSamples = samples + [Float](repeating: 0, count: hop)

        // STFT — 960-point DFT producing 481 frequency bins
        let (specReal, specImag) = stft.forward(audio: paddedSamples, analysisMem: &analysisMem)
        let numFrames = specReal.count / freqBins
        guard numFrames > 0 else { return [] }

        // Compute ERB features
        var erbFeats = computeERBFeatures(
            real: specReal, imag: specImag,
            erbFb: erbFb,
            freqBins: freqBins, erbBands: config.erbBands, numFrames: numFrames)

        // Apply mean normalization
        applyMeanNormalization(
            &erbFeats, state: &meanNormState,
            alpha: config.normAlpha,
            erbBands: config.erbBands, numFrames: numFrames)

        // Extract first dfBins for spec features
        var specFeatReal = [Float](repeating: 0, count: numFrames * config.dfBins)
        var specFeatImag = [Float](repeating: 0, count: numFrames * config.dfBins)
        for t in 0..<numFrames {
            for f in 0..<config.dfBins {
                specFeatReal[t * config.dfBins + f] = specReal[t * freqBins + f]
                specFeatImag[t * config.dfBins + f] = specImag[t * freqBins + f]
            }
        }

        // Apply unit normalization
        applyUnitNormalization(
            real: &specFeatReal, imag: &specFeatImag,
            state: &unitNormState,
            alpha: config.normAlpha,
            dfBins: config.dfBins, numFrames: numFrames)

        // No lookahead padding here — the Core ML model applies it internally

        // Create Core ML input arrays
        // ERB: [1, 1, T, 32] (NCHW)
        let erbInput = try MLMultiArray(shape: [1, 1, numFrames as NSNumber, config.erbBands as NSNumber], dataType: .float32)
        let erbPtr = erbInput.dataPointer.assumingMemoryBound(to: Float.self)
        erbFeats.withUnsafeBufferPointer { src in
            erbPtr.update(from: src.baseAddress!, count: numFrames * config.erbBands)
        }

        // Spec: [1, 2, T, 96] (NCHW: channel 0=real, channel 1=imag)
        let specInput = try MLMultiArray(shape: [1, 2, numFrames as NSNumber, config.dfBins as NSNumber], dataType: .float32)
        let specPtr = specInput.dataPointer.assumingMemoryBound(to: Float.self)
        let specChannelStride = numFrames * config.dfBins
        specFeatReal.withUnsafeBufferPointer { src in
            specPtr.update(from: src.baseAddress!, count: specChannelStride)
        }
        specFeatImag.withUnsafeBufferPointer { src in
            (specPtr + specChannelStride).update(from: src.baseAddress!, count: specChannelStride)
        }

        // Run neural network (single pass — GRU state is sequential, can't be chunked)
        let (erbMaskArray, coefsArray) = try network.predict(featErb: erbInput, featSpec: specInput)

        // Extract ERB mask [1, 1, T, 32] — handle float16 output from Core ML
        let erbMaskCount = numFrames * config.erbBands
        var erbMaskFlat = [Float](repeating: 0, count: erbMaskCount)
        extractMLMultiArrayFlat(erbMaskArray, into: &erbMaskFlat, count: erbMaskCount)

        // Extract DF coefficients [1, 5, T, 96, 2] → [T, F, O, C=2]
        let dfOrder = config.dfOrder
        let coefsCount = dfOrder * numFrames * config.dfBins * 2
        var coefsRaw = [Float](repeating: 0, count: coefsCount)
        extractMLMultiArrayFlat(coefsArray, into: &coefsRaw, count: coefsCount)

        // Reshape from Core ML layout [O, T, F, 2] to [T, F, O, 2]
        var coefsFlat = [Float](repeating: 0, count: coefsCount)
        for t in 0..<numFrames {
            for f in 0..<config.dfBins {
                for o in 0..<dfOrder {
                    let srcIdx = ((o * numFrames + t) * config.dfBins + f) * 2
                    let dstIdx = ((t * config.dfBins + f) * dfOrder + o) * 2
                    coefsFlat[dstIdx] = coefsRaw[srcIdx]
                    coefsFlat[dstIdx + 1] = coefsRaw[srcIdx + 1]
                }
            }
        }

        // Apply ERB mask to full spectrum
        var enhancedReal = specReal
        var enhancedImag = specImag
        applyERBMask(
            specReal: &enhancedReal, specImag: &enhancedImag,
            erbMask: erbMaskFlat,
            erbInvFb: erbInvFb,
            erbBands: config.erbBands, freqBins: freqBins, numFrames: numFrames)

        // Apply deep filtering to lowest dfBins
        let (dfReal, dfImag) = applyDeepFiltering(
            specReal: specReal, specImag: specImag,
            coefs: coefsFlat,
            dfBins: config.dfBins, dfOrder: config.dfOrder,
            dfLookahead: config.dfLookahead,
            numFrames: numFrames, freqBins: freqBins)

        // Combine: DF-enhanced for bins 0..<dfBins, ERB-masked for rest
        for t in 0..<numFrames {
            for f in 0..<config.dfBins {
                enhancedReal[t * freqBins + f] = dfReal[t * config.dfBins + f]
                enhancedImag[t * freqBins + f] = dfImag[t * config.dfBins + f]
            }
        }

        // Inverse STFT
        let rawOutput = stft.inverse(real: enhancedReal, imag: enhancedImag, synthesisMem: &synthesisMem)

        // Trim the hop-size latency from prepended analysis memory
        let trimStart = hop
        let trimEnd = min(trimStart + samples.count, rawOutput.count)
        guard trimEnd > trimStart else { return [] }
        return Array(rawOutput[trimStart..<trimEnd])
    }

    /// Reset all streaming state (STFT buffers, normalization).
    public func resetState() {
        analysisMem = [Float](repeating: 0, count: config.fftSize - config.hopSize)
        synthesisMem = [Float](repeating: 0, count: config.fftSize - config.hopSize)
        meanNormState = meanNormStateInit
        unitNormState = unitNormStateInit
    }

    /// Load a pretrained DeepFilterNet3 model.
    ///
    /// Downloads Core ML model and auxiliary data on first use, then caches locally.
    ///
    /// - Parameters:
    ///   - modelId: HuggingFace model ID
    ///   - progressHandler: Callback for download progress
    /// - Returns: Ready-to-use speech enhancer
    public static func fromPretrained(
        modelId: String = defaultModelId,
        progressHandler: ((Double, String) -> Void)? = nil
    ) async throws -> SpeechEnhancer {
        progressHandler?(0.0, "Downloading model...")

        let cacheDir = try HuggingFaceDownloader.getCacheDirectory(for: modelId)

        try await HuggingFaceDownloader.downloadWeights(
            modelId: modelId,
            to: cacheDir,
            additionalFiles: [
                "no_weights.safetensors",  // suppress default *.safetensors glob
                "DeepFilterNet3.mlpackage/**",
                "auxiliary.npz",
            ],
            progressHandler: { progress in
                progressHandler?(progress * 0.8, "Downloading model...")
            }
        )

        progressHandler?(0.8, "Loading model...")

        let config = DeepFilterNet3Config.default
        let (network, auxData) = try DeepFilterNet3WeightLoader.load(from: cacheDir)

        progressHandler?(1.0, "Ready")

        return SpeechEnhancer(network: network, config: config, auxData: auxData)
    }

    /// Extract float32 data from an MLMultiArray, handling float16 output from Core ML.
    private func extractMLMultiArrayFlat(_ array: MLMultiArray, into output: inout [Float], count: Int) {
        if array.dataType == .float16 {
            let ptr = array.dataPointer.assumingMemoryBound(to: Float16.self)
            for i in 0..<count {
                output[i] = Float(ptr[i])
            }
        } else {
            let ptr = array.dataPointer.assumingMemoryBound(to: Float.self)
            output.withUnsafeMutableBufferPointer { dst in
                dst.baseAddress!.update(from: ptr, count: count)
            }
        }
    }

    /// Simple linear resampling.
    private func resample(_ audio: [Float], from sourceSR: Int, to targetSR: Int) -> [Float] {
        guard sourceSR != targetSR else { return audio }
        let ratio = Double(targetSR) / Double(sourceSR)
        let outputLen = Int(Double(audio.count) * ratio)
        var output = [Float](repeating: 0, count: outputLen)

        for i in 0..<outputLen {
            let srcPos = Double(i) / ratio
            let srcIdx = Int(srcPos)
            let frac = Float(srcPos - Double(srcIdx))

            if srcIdx + 1 < audio.count {
                output[i] = audio[srcIdx] * (1 - frac) + audio[srcIdx + 1] * frac
            } else if srcIdx < audio.count {
                output[i] = audio[srcIdx]
            }
        }

        return output
    }
}
