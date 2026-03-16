import Foundation
import Accelerate
import AudioCommon

#if canImport(CoreML)
import CoreML
#endif

/// Voice Activity Detection using FireRedVAD (DFSMN, CoreML).
///
/// A lightweight 588K-param model using DFSMN blocks (depthwise Conv1d)
/// for temporal context. Runs on Neural Engine + CPU via CoreML.
///
/// - Warning: This class is not thread-safe. Create separate instances for concurrent use.
///
/// ```swift
/// let vad = try await FireRedVADModel.fromPretrained()
/// let segments = vad.detectSpeech(audio: samples, sampleRate: 16000)
/// ```
public final class FireRedVADModel {

    /// Default HuggingFace model ID
    public static let defaultModelId = "aufklarer/FireRedVAD-CoreML"

    /// Whether the model weights are loaded and ready for inference.
    var _isLoaded = true

    #if canImport(CoreML)
    /// CoreML compiled model
    private let coremlModel: MLModel
    #endif

    /// Feature extractor: 80-dim log Mel fbank (Kaldi-compatible)
    private let featureExtractor: KaldiFbankExtractor

    /// Post-processing config
    public var speechThreshold: Float = 0.4
    public var smoothWindowSize: Int = 5
    public var minSpeechDuration: Float = 0.2
    public var minSilenceDuration: Float = 0.2

    #if canImport(CoreML)
    init(coremlModel: MLModel) {
        self.coremlModel = coremlModel
        self.featureExtractor = KaldiFbankExtractor()
    }
    #endif

    // MARK: - Model Loading

    /// Load FireRedVAD from HuggingFace.
    public static func fromPretrained(
        modelId: String = defaultModelId,
        progressHandler: ((Double, String) -> Void)? = nil
    ) async throws -> FireRedVADModel {
        #if canImport(CoreML)
        progressHandler?(0.0, "Downloading model...")

        let cacheDir = try HuggingFaceDownloader.getCacheDirectory(for: modelId)

        try await HuggingFaceDownloader.downloadWeights(
            modelId: modelId,
            to: cacheDir,
            additionalFiles: [
                "fireredvad.mlmodelc/**",
                "config.json",
                "cmvn.json",
            ],
            progressHandler: { progress in
                progressHandler?(progress * 0.8, "Downloading model...")
            }
        )

        progressHandler?(0.8, "Loading CoreML model...")

        let modelURL = cacheDir.appendingPathComponent(
            "fireredvad.mlmodelc", isDirectory: true)
        guard FileManager.default.fileExists(atPath: modelURL.path) else {
            throw AudioModelError.modelLoadFailed(
                modelId: modelId,
                reason: "CoreML model not found at \(modelURL.path)")
        }

        let mlConfig = MLModelConfiguration()
        mlConfig.computeUnits = .cpuAndNeuralEngine

        let model = try MLModel(contentsOf: modelURL, configuration: mlConfig)

        progressHandler?(1.0, "Ready")
        return FireRedVADModel(coremlModel: model)
        #else
        throw AudioModelError.invalidConfiguration(
            model: "FireRedVAD", reason: "CoreML not available on this platform")
        #endif
    }

    // MARK: - Inference

    /// Detect speech segments in audio.
    ///
    /// - Parameters:
    ///   - audio: Float32 PCM samples
    ///   - sampleRate: Sample rate of the audio
    /// - Returns: Array of speech segments with start/end times
    public func detectSpeech(
        audio: [Float],
        sampleRate: Int = 16000
    ) -> [SpeechSegment] {
        // Resample if needed
        let samples: [Float]
        if sampleRate != 16000 {
            samples = AudioFileLoader.resample(audio, from: sampleRate, to: 16000)
        } else {
            samples = audio
        }

        // Extract features
        let features = featureExtractor.extract(samples)
        guard features.count > 0 else { return [] }

        let numFrames = features.count / 80

        // Run CoreML inference (chunk if > 6000 frames / 60s)
        #if canImport(CoreML)
        let maxFrames = 6000
        let probs: [Float]
        if numFrames <= maxFrames {
            probs = runCoreML(features: features, numFrames: numFrames)
        } else {
            // Process in chunks
            var allProbs = [Float]()
            var offset = 0
            while offset < numFrames {
                let chunkFrames = min(maxFrames, numFrames - offset)
                let chunkStart = offset * 80
                let chunkEnd = chunkStart + chunkFrames * 80
                let chunkFeatures = Array(features[chunkStart..<chunkEnd])
                let chunkProbs = runCoreML(features: chunkFeatures, numFrames: chunkFrames)
                allProbs.append(contentsOf: chunkProbs)
                offset += chunkFrames
            }
            probs = allProbs
        }
        #else
        return []
        #endif

        // Post-process: smooth → threshold → segment
        let smoothed = smoothProbabilities(probs, windowSize: smoothWindowSize)
        return extractSegments(
            smoothed,
            threshold: speechThreshold,
            frameShift: 0.01,
            minSpeechDuration: minSpeechDuration,
            minSilenceDuration: minSilenceDuration
        )
    }

    #if canImport(CoreML)
    private func runCoreML(features: [Float], numFrames: Int) -> [Float] {
        // Create MLMultiArray [1, T, 80]
        guard let input = try? MLMultiArray(
            shape: [1, NSNumber(value: numFrames), 80],
            dataType: .float32
        ) else { return [] }

        let ptr = input.dataPointer.assumingMemoryBound(to: Float.self)
        features.withUnsafeBufferPointer { src in
            ptr.update(from: src.baseAddress!, count: min(features.count, numFrames * 80))
        }

        guard let prediction = try? coremlModel.prediction(
            from: FireRedVADInput(features: input)
        ) else { return [] }

        guard let outputArray = prediction.featureValue(
            for: "probabilities"
        )?.multiArrayValue else { return [] }

        // Extract probabilities — output is [1, T, 1]
        // Use MLMultiArray subscript to handle strides and float16 conversion
        var probs = [Float](repeating: 0, count: numFrames)
        for i in 0..<numFrames {
            probs[i] = outputArray[[0, NSNumber(value: i), 0]].floatValue
        }
        return probs
    }
    #endif

    // MARK: - Post-processing

    private func smoothProbabilities(_ probs: [Float], windowSize: Int) -> [Float] {
        guard windowSize > 1, probs.count > windowSize else { return probs }
        var smoothed = [Float](repeating: 0, count: probs.count)
        let half = windowSize / 2

        for i in 0..<probs.count {
            let lo = max(0, i - half)
            let hi = min(probs.count, i + half + 1)
            var sum: Float = 0
            for j in lo..<hi { sum += probs[j] }
            smoothed[i] = sum / Float(hi - lo)
        }
        return smoothed
    }

    private func extractSegments(
        _ probs: [Float],
        threshold: Float,
        frameShift: Float,
        minSpeechDuration: Float,
        minSilenceDuration: Float
    ) -> [SpeechSegment] {
        // Binary decisions
        let decisions = probs.map { $0 >= threshold }

        // Find contiguous speech regions
        var segments = [SpeechSegment]()
        var speechStart: Int?

        for i in 0...decisions.count {
            let isSpeech = i < decisions.count ? decisions[i] : false
            if isSpeech && speechStart == nil {
                speechStart = i
            } else if !isSpeech, let start = speechStart {
                let startTime = Float(start) * frameShift
                let endTime = Float(i) * frameShift
                if endTime - startTime >= minSpeechDuration {
                    segments.append(SpeechSegment(
                        startTime: startTime, endTime: endTime))
                }
                speechStart = nil
            }
        }

        // Merge segments with short silence gaps
        guard segments.count > 1 else { return segments }
        var merged = [segments[0]]
        for i in 1..<segments.count {
            let gap = segments[i].startTime - merged.last!.endTime
            if gap < minSilenceDuration {
                let last = merged.removeLast()
                merged.append(SpeechSegment(
                    startTime: last.startTime,
                    endTime: segments[i].endTime))
            } else {
                merged.append(segments[i])
            }
        }
        return merged
    }
}

// MARK: - CoreML Input Wrapper

#if canImport(CoreML)
private class FireRedVADInput: MLFeatureProvider {
    let features: MLMultiArray

    init(features: MLMultiArray) {
        self.features = features
    }

    var featureNames: Set<String> { ["features"] }

    func featureValue(for featureName: String) -> MLFeatureValue? {
        if featureName == "features" {
            return MLFeatureValue(multiArray: features)
        }
        return nil
    }
}
#endif

// MARK: - Kaldi-compatible Fbank Extractor

/// 80-dim log Mel filterbank extractor matching Kaldi's default settings.
///
/// Configuration: 16kHz, 25ms window, 10ms hop, 80 mel bins, snip_edges=true.
/// Uses Povey window (Hann-like) and log energy.
final class KaldiFbankExtractor {
    let sampleRate: Int = 16000
    let frameLength: Int = 400   // 25ms at 16kHz
    let frameShift: Int = 160    // 10ms at 16kHz
    let nMels: Int = 80
    let nFFT: Int = 512          // next power of 2 from 400
    let log2FFT: vDSP_Length = 9 // log2(512)
    let preemphCoeff: Float = 0.97

    private let window: [Float]
    private let fftSetup: FFTSetup
    private let melFilterbank: [Float] // [nMels, nBins]
    private let nBins: Int

    init() {
        // Povey window (Hann raised to 0.85 power)
        var w = [Float](repeating: 0, count: 400)
        for i in 0..<400 {
            let hann = 0.5 - 0.5 * cos(2.0 * Float.pi * Float(i) / Float(400))
            w[i] = pow(hann, 0.85)
        }
        self.window = w

        self.fftSetup = vDSP_create_fftsetup(9, FFTRadix(kFFTRadix2))!
        self.nBins = 257 // nFFT/2 + 1

        // Build mel filterbank
        self.melFilterbank = KaldiFbankExtractor.buildMelFilterbank(
            nMels: 80, nBins: 257, sampleRate: 16000, nFFT: 512)
    }

    deinit {
        vDSP_destroy_fftsetup(fftSetup)
    }

    /// Extract 80-dim log Mel features from audio samples.
    /// Returns flat array [T * 80] where T is the number of frames.
    func extract(_ samples: [Float]) -> [Float] {
        // snip_edges=true: first frame starts at 0, last frame ends within audio
        let numFrames = max(0, (samples.count - frameLength) / frameShift + 1)
        guard numFrames > 0 else { return [] }

        var allFeatures = [Float]()
        allFeatures.reserveCapacity(numFrames * nMels)

        var paddedFrame = [Float](repeating: 0, count: nFFT)
        var realPart = [Float](repeating: 0, count: nFFT / 2)
        var imagPart = [Float](repeating: 0, count: nFFT / 2)

        for frame in 0..<numFrames {
            let start = frame * frameShift

            // Extract raw frame, scale to int16 range
            var rawFrame = [Float](repeating: 0, count: frameLength)
            for i in 0..<frameLength {
                rawFrame[i] = samples[start + i] * 32768.0
            }

            // Remove DC offset (Kaldi default)
            var dcMean: Float = 0
            vDSP_meanv(rawFrame, 1, &dcMean, vDSP_Length(frameLength))
            var negMean = -dcMean
            vDSP_vsadd(rawFrame, 1, &negMean, &rawFrame, 1, vDSP_Length(frameLength))

            // Pre-emphasis: y[n] = x[n] - 0.97 * x[n-1]
            for i in stride(from: frameLength - 1, through: 1, by: -1) {
                rawFrame[i] -= preemphCoeff * rawFrame[i - 1]
            }
            rawFrame[0] -= preemphCoeff * rawFrame[0]  // Kaldi: first sample uses itself

            // Apply window
            for i in 0..<frameLength {
                paddedFrame[i] = rawFrame[i] * window[i]
            }
            for i in frameLength..<nFFT {
                paddedFrame[i] = 0
            }

            // FFT
            realPart.withUnsafeMutableBufferPointer { rp in
                imagPart.withUnsafeMutableBufferPointer { ip in
                    paddedFrame.withUnsafeBufferPointer { input in
                        // Deinterleave
                        var splitComplex = DSPSplitComplex(
                            realp: rp.baseAddress!,
                            imagp: ip.baseAddress!)
                        input.baseAddress!.withMemoryRebound(
                            to: DSPComplex.self,
                            capacity: nFFT / 2
                        ) { complexPtr in
                            vDSP_ctoz(complexPtr, 2, &splitComplex, 1, vDSP_Length(nFFT / 2))
                        }
                        vDSP_fft_zrip(fftSetup, &splitComplex, 1, log2FFT, FFTDirection(kFFTDirection_Forward))
                    }
                }
            }

            // Power spectrum: |X(k)|² (Kaldi does NOT divide by N)
            // vDSP_fft_zrip output is 2x the true FFT, so scale by 0.5
            let fftScale: Float = 0.5
            var powerSpec = [Float](repeating: 0, count: nBins)
            // DC (stored in realPart[0])
            powerSpec[0] = (realPart[0] * fftScale) * (realPart[0] * fftScale)
            // Nyquist (stored in imagPart[0])
            powerSpec[nBins - 1] = (imagPart[0] * fftScale) * (imagPart[0] * fftScale)
            // Remaining bins
            for i in 1..<(nBins - 1) {
                let r = realPart[i] * fftScale
                let im = imagPart[i] * fftScale
                powerSpec[i] = r * r + im * im
            }

            // Apply mel filterbank
            var melEnergies = [Float](repeating: 0, count: nMels)
            for m in 0..<nMels {
                var energy: Float = 0
                let offset = m * nBins
                for b in 0..<nBins {
                    energy += melFilterbank[offset + b] * powerSpec[b]
                }
                melEnergies[m] = log(max(energy, Float.ulpOfOne))
            }

            allFeatures.append(contentsOf: melEnergies)
        }

        return allFeatures
    }

    /// Build mel filterbank matrix [nMels, nBins] matching Kaldi's computation.
    ///
    /// Key differences from naive implementation:
    /// - Uses `floor()` for bin indices (Kaldi convention)
    /// - Uses `(nFFT + 1) / sampleRate` for Hz-to-bin conversion (Kaldi convention)
    private static func buildMelFilterbank(
        nMels: Int, nBins: Int, sampleRate: Int, nFFT: Int
    ) -> [Float] {
        func hzToMel(_ hz: Float) -> Float {
            return 1127.0 * log(1.0 + hz / 700.0)
        }
        func melToHz(_ mel: Float) -> Float {
            return 700.0 * (exp(mel / 1127.0) - 1.0)
        }

        let fMin: Float = 20.0
        let fMax = Float(sampleRate) / 2.0
        let melMin = hzToMel(fMin)
        let melMax = hzToMel(fMax)

        // Mel center frequencies
        var melPoints = [Float](repeating: 0, count: nMels + 2)
        for i in 0..<(nMels + 2) {
            melPoints[i] = melMin + Float(i) * (melMax - melMin) / Float(nMels + 1)
        }

        // Convert to FFT bin indices (Kaldi: floor, (nFFT+1)/sr)
        var binCenters = [Int](repeating: 0, count: nMels + 2)
        for i in 0..<(nMels + 2) {
            binCenters[i] = Int(floor(melToHz(melPoints[i]) * Float(nFFT + 1) / Float(sampleRate)))
        }

        // Build triangular filters using integer bin centers
        var filterbank = [Float](repeating: 0, count: nMels * nBins)
        for m in 0..<nMels {
            let left = binCenters[m]
            let center = binCenters[m + 1]
            let right = binCenters[m + 2]

            if center > left {
                for b in max(0, left)...min(nBins - 1, center) {
                    filterbank[m * nBins + b] = Float(b - left) / Float(center - left)
                }
            }
            if right > center {
                for b in max(0, center + 1)...min(nBins - 1, right) {
                    filterbank[m * nBins + b] = Float(right - b) / Float(right - center)
                }
            }
        }

        return filterbank
    }
}
