import Accelerate
import AudioCommon
import CoreML
import Foundation

/// Mel spectrogram preprocessor for Parakeet EOU streaming ASR.
///
/// Implements the NeMo `AudioToMelSpectrogramPreprocessor` pipeline in Swift:
/// pre-emphasis → STFT → power spectrum → mel filterbank → log → per-feature normalization.
///
/// Ported from ParakeetASR/MelPreprocessor (proven to match NeMo output).
/// Outputs float32 (EOU encoder expects float32 input).
class StreamingMelPreprocessor {
    let config: ParakeetEOUConfig

    private let paddedFFT: Int = 512
    private let log2PaddedFFT: vDSP_Length = 9
    private let nBins: Int = 257          // paddedFFT / 2 + 1
    private let reflectPad: Int = 256     // n_fft / 2
    private let logGuard: Float = 5.960464477539063e-08  // 2^{-24}

    private let fftSetup: FFTSetup
    private let hannWindow: [Float]
    private let melFilterbank: [Float]    // [nMels, nBins]

    // Running normalization state (Welford online algorithm)
    private var runningSum: [Float]       // per mel bin
    private var runningSumSq: [Float]     // per mel bin
    private(set) var runningCount: Int = 0

    init(config: ParakeetEOUConfig) {
        self.config = config

        // Symmetric Hann window: periodic=False, formula uses (N-1)
        // Verified against NeMo: torch.hann_window(400, periodic=False)
        var window = [Float](repeating: 0, count: config.winLength)
        for i in 0..<config.winLength {
            window[i] = 0.5 * (1.0 - cos(2.0 * Float.pi * Float(i) / Float(config.winLength - 1)))
        }
        self.hannWindow = window

        guard let setup = vDSP_create_fftsetup(9, FFTRadix(kFFTRadix2)) else {
            fatalError("Failed to create vDSP FFT setup")
        }
        self.fftSetup = setup

        self.melFilterbank = Self.buildMelFilterbank(
            nMels: config.numMelBins,
            nBins: 257,
            sampleRate: config.sampleRate,
            paddedFFT: 512
        )

        self.runningSum = [Float](repeating: 0, count: config.numMelBins)
        self.runningSumSq = [Float](repeating: 0, count: config.numMelBins)
    }

    /// Extract mel spectrogram from audio samples.
    ///
    /// - Parameter audio: PCM Float32 samples at config.sampleRate
    /// - Returns: Mel MLMultiArray [1, numMelBins, nFrames] in float32 + valid frame count
    func extract(_ audio: [Float]) throws -> (mel: MLMultiArray, melLength: Int) {
        guard !audio.isEmpty else {
            let mel = try MLMultiArray(shape: [1, config.numMelBins as NSNumber, 1], dataType: .float32)
            return (mel, 0)
        }

        // Pre-emphasis: x[n] = audio[n] - 0.97 * audio[n-1]
        var preemphasized = [Float](repeating: 0, count: audio.count)
        preemphasized[0] = audio[0]
        audio.withUnsafeBufferPointer { src in
            preemphasized.withUnsafeMutableBufferPointer { dst in
                var negCoeff = -config.preEmphasis
                vDSP_vsma(src.baseAddress!, 1, &negCoeff,
                          src.baseAddress! + 1, 1,
                          dst.baseAddress! + 1, 1,
                          vDSP_Length(audio.count - 1))
            }
        }

        // Reflect padding (center=True)
        let totalLen = reflectPad + preemphasized.count + reflectPad
        var padded = [Float](repeating: 0, count: totalLen)
        for i in 0..<reflectPad {
            padded[i] = preemphasized[reflectPad - i]
        }
        for i in 0..<preemphasized.count {
            padded[reflectPad + i] = preemphasized[i]
        }
        for i in 0..<reflectPad {
            let srcIdx = preemphasized.count - 2 - i
            padded[reflectPad + preemphasized.count + i] = preemphasized[max(0, srcIdx)]
        }

        let nFrames = (padded.count - paddedFFT) / config.hopLength + 1
        let melLength = audio.count / config.hopLength

        let halfPadded = paddedFFT / 2

        // STFT + power spectrum
        var powerSpec = [Float](repeating: 0, count: nFrames * nBins)
        var splitReal = [Float](repeating: 0, count: halfPadded)
        var splitImag = [Float](repeating: 0, count: halfPadded)
        var paddedFrame = [Float](repeating: 0, count: paddedFFT)

        for frame in 0..<nFrames {
            let start = frame * config.hopLength

            padded.withUnsafeBufferPointer { buf in
                vDSP_vmul(buf.baseAddress! + start, 1, hannWindow, 1,
                          &paddedFrame, 1, vDSP_Length(config.winLength))
            }
            for i in config.winLength..<paddedFFT {
                paddedFrame[i] = 0
            }

            for i in 0..<halfPadded {
                splitReal[i] = paddedFrame[2 * i]
                splitImag[i] = paddedFrame[2 * i + 1]
            }

            splitReal.withUnsafeMutableBufferPointer { realBuf in
                splitImag.withUnsafeMutableBufferPointer { imagBuf in
                    var splitComplex = DSPSplitComplex(
                        realp: realBuf.baseAddress!,
                        imagp: imagBuf.baseAddress!)
                    vDSP_fft_zrip(fftSetup, &splitComplex, 1,
                                  log2PaddedFFT, FFTDirection(kFFTDirection_Forward))
                }
            }

            let base = frame * nBins
            powerSpec[base] = splitReal[0] * splitReal[0]
            powerSpec[base + halfPadded] = splitImag[0] * splitImag[0]
            for k in 1..<halfPadded {
                powerSpec[base + k] = splitReal[k] * splitReal[k] + splitImag[k] * splitImag[k]
            }
        }

        // Mel filterbank: [nMels, nBins] × [nBins, nFrames] → [nMels, nFrames]
        var powerSpecT = [Float](repeating: 0, count: nBins * nFrames)
        vDSP_mtrans(powerSpec, 1, &powerSpecT, 1,
                    vDSP_Length(nBins), vDSP_Length(nFrames))

        var melSpec = [Float](repeating: 0, count: config.numMelBins * nFrames)
        vDSP_mmul(melFilterbank, 1, powerSpecT, 1, &melSpec, 1,
                  vDSP_Length(config.numMelBins), vDSP_Length(nFrames), vDSP_Length(nBins))

        // Log
        var guardVal = logGuard
        vDSP_vsadd(melSpec, 1, &guardVal, &melSpec, 1, vDSP_Length(melSpec.count))
        var countN = Int32(melSpec.count)
        vvlogf(&melSpec, melSpec, &countN)

        // Per-feature normalization over valid frames
        melSpec.withUnsafeMutableBufferPointer { buf in
            for bin in 0..<config.numMelBins {
                let base = buf.baseAddress! + bin * nFrames

                var mean: Float = 0
                vDSP_meanv(base, 1, &mean, vDSP_Length(melLength))

                var negMean = -mean
                vDSP_vsadd(base, 1, &negMean, base, 1, vDSP_Length(melLength))

                var meanSq: Float = 0
                vDSP_measqv(base, 1, &meanSq, vDSP_Length(melLength))
                let std = sqrt(Float(melLength) * meanSq / Float(max(melLength - 1, 1)))
                var invStd = 1.0 / (std + 1e-5)
                vDSP_vsmul(base, 1, &invStd, base, 1, vDSP_Length(melLength))

                // Zero pad frames beyond melLength
                memset(base + melLength, 0, (nFrames - melLength) * MemoryLayout<Float>.stride)
            }
        }

        // Create MLMultiArray [1, numMelBins, nFrames] in float32
        let mel = try MLMultiArray(
            shape: [1, config.numMelBins as NSNumber, nFrames as NSNumber],
            dataType: .float32)
        let melPtr = mel.dataPointer.assumingMemoryBound(to: Float.self)
        memcpy(melPtr, melSpec, config.numMelBins * nFrames * MemoryLayout<Float>.stride)

        return (mel, melLength)
    }

    /// Extract mel WITHOUT normalization — raw log mel values.
    /// Some streaming models handle normalization internally or don't need it.
    /// Extract mel WITHOUT normalization — raw log mel values.
    /// Uses NeMo EOU streaming config: symmetric Hann window, zero center padding,
    /// no per-feature normalization.
    func extractRaw(_ audio: [Float]) throws -> (mel: MLMultiArray, melLength: Int) {
        guard !audio.isEmpty else {
            let mel = try MLMultiArray(shape: [1, config.numMelBins as NSNumber, 1], dataType: .float32)
            return (mel, 0)
        }

        // Pre-emphasis: y[n] = x[n] - 0.97 * x[n-1]
        var preemphasized = [Float](repeating: 0, count: audio.count)
        preemphasized[0] = audio[0]
        audio.withUnsafeBufferPointer { src in
            preemphasized.withUnsafeMutableBufferPointer { dst in
                var negCoeff = -config.preEmphasis
                vDSP_vsma(src.baseAddress!, 1, &negCoeff,
                          src.baseAddress! + 1, 1,
                          dst.baseAddress! + 1, 1,
                          vDSP_Length(audio.count - 1))
            }
        }

        // Zero center padding: pad_mode="constant" (NeMo default)
        // Verified: torch.stft center=True uses n_fft//2 zeros on each side
        let totalLen = reflectPad + preemphasized.count + reflectPad
        var padded = [Float](repeating: 0, count: totalLen)
        for i in 0..<preemphasized.count { padded[reflectPad + i] = preemphasized[i] }

        let nFrames = (padded.count - paddedFFT) / config.hopLength + 1
        let melLength = audio.count / config.hopLength
        let halfPadded = paddedFFT / 2

        var powerSpec = [Float](repeating: 0, count: nFrames * nBins)
        var splitReal = [Float](repeating: 0, count: halfPadded)
        var splitImag = [Float](repeating: 0, count: halfPadded)
        var paddedFrame = [Float](repeating: 0, count: paddedFFT)

        // torch.stft centers the window in the n_fft frame:
        // frame[0..55] = 0, frame[56..455] = signal * hann, frame[456..511] = 0
        let winOffset = (paddedFFT - config.winLength) / 2

        for frame in 0..<nFrames {
            let start = frame * config.hopLength

            // Zero the FFT frame
            memset(&paddedFrame, 0, paddedFFT * MemoryLayout<Float>.stride)
            // Apply window centered at offset 56
            padded.withUnsafeBufferPointer { buf in
                vDSP_vmul(buf.baseAddress! + start + winOffset, 1, hannWindow, 1,
                          &paddedFrame[winOffset], 1, vDSP_Length(config.winLength))
            }

            for i in 0..<halfPadded {
                splitReal[i] = paddedFrame[2 * i]
                splitImag[i] = paddedFrame[2 * i + 1]
            }
            splitReal.withUnsafeMutableBufferPointer { realBuf in
                splitImag.withUnsafeMutableBufferPointer { imagBuf in
                    var splitComplex = DSPSplitComplex(realp: realBuf.baseAddress!, imagp: imagBuf.baseAddress!)
                    vDSP_fft_zrip(fftSetup, &splitComplex, 1, log2PaddedFFT, FFTDirection(kFFTDirection_Forward))
                }
            }
            // vDSP_fft_zrip scales by 2x vs standard DFT (torch.stft).
            // Divide power by 4 (2^2) to match training pipeline.
            let base = frame * nBins
            powerSpec[base] = splitReal[0] * splitReal[0] * 0.25
            powerSpec[base + halfPadded] = splitImag[0] * splitImag[0] * 0.25
            for k in 1..<halfPadded {
                powerSpec[base + k] = (splitReal[k] * splitReal[k] + splitImag[k] * splitImag[k]) * 0.25
            }
        }

        var powerSpecT = [Float](repeating: 0, count: nBins * nFrames)
        vDSP_mtrans(powerSpec, 1, &powerSpecT, 1, vDSP_Length(nBins), vDSP_Length(nFrames))
        var melSpec = [Float](repeating: 0, count: config.numMelBins * nFrames)
        vDSP_mmul(melFilterbank, 1, powerSpecT, 1, &melSpec, 1,
                  vDSP_Length(config.numMelBins), vDSP_Length(nFrames), vDSP_Length(nBins))

        var guardVal = logGuard
        vDSP_vsadd(melSpec, 1, &guardVal, &melSpec, 1, vDSP_Length(melSpec.count))
        var countN = Int32(melSpec.count)
        vvlogf(&melSpec, melSpec, &countN)

        let mel = try MLMultiArray(
            shape: [1, config.numMelBins as NSNumber, nFrames as NSNumber], dataType: .float32)
        memcpy(mel.dataPointer.assumingMemoryBound(to: Float.self),
               melSpec, config.numMelBins * nFrames * MemoryLayout<Float>.stride)
        return (mel, melLength)
    }

    /// Extract mel with running normalization for streaming.
    ///
    /// Same DSP as `extract()` but uses accumulated mean/std across all chunks
    /// instead of per-chunk normalization. This matches NeMo's whole-utterance
    /// normalization behavior as more audio accumulates.
    func extractStreaming(_ audio: [Float]) throws -> (mel: MLMultiArray, melLength: Int) {
        guard !audio.isEmpty else {
            let mel = try MLMultiArray(shape: [1, config.numMelBins as NSNumber, 1], dataType: .float32)
            return (mel, 0)
        }

        // Pre-emphasis
        var preemphasized = [Float](repeating: 0, count: audio.count)
        preemphasized[0] = audio[0]
        audio.withUnsafeBufferPointer { src in
            preemphasized.withUnsafeMutableBufferPointer { dst in
                var negCoeff = -config.preEmphasis
                vDSP_vsma(src.baseAddress!, 1, &negCoeff,
                          src.baseAddress! + 1, 1,
                          dst.baseAddress! + 1, 1,
                          vDSP_Length(audio.count - 1))
            }
        }

        // Reflect padding
        let totalLen = reflectPad + preemphasized.count + reflectPad
        var padded = [Float](repeating: 0, count: totalLen)
        for i in 0..<reflectPad { padded[i] = preemphasized[reflectPad - i] }
        for i in 0..<preemphasized.count { padded[reflectPad + i] = preemphasized[i] }
        for i in 0..<reflectPad {
            padded[reflectPad + preemphasized.count + i] = preemphasized[max(0, preemphasized.count - 2 - i)]
        }

        let nFrames = (padded.count - paddedFFT) / config.hopLength + 1
        let melLength = audio.count / config.hopLength
        let halfPadded = paddedFFT / 2

        // STFT + power spectrum (same as extract)
        var powerSpec = [Float](repeating: 0, count: nFrames * nBins)
        var splitReal = [Float](repeating: 0, count: halfPadded)
        var splitImag = [Float](repeating: 0, count: halfPadded)
        var paddedFrame = [Float](repeating: 0, count: paddedFFT)

        for frame in 0..<nFrames {
            let start = frame * config.hopLength
            padded.withUnsafeBufferPointer { buf in
                vDSP_vmul(buf.baseAddress! + start, 1, hannWindow, 1,
                          &paddedFrame, 1, vDSP_Length(config.winLength))
            }
            for i in config.winLength..<paddedFFT { paddedFrame[i] = 0 }

            for i in 0..<halfPadded {
                splitReal[i] = paddedFrame[2 * i]
                splitImag[i] = paddedFrame[2 * i + 1]
            }
            splitReal.withUnsafeMutableBufferPointer { realBuf in
                splitImag.withUnsafeMutableBufferPointer { imagBuf in
                    var splitComplex = DSPSplitComplex(realp: realBuf.baseAddress!, imagp: imagBuf.baseAddress!)
                    vDSP_fft_zrip(fftSetup, &splitComplex, 1, log2PaddedFFT, FFTDirection(kFFTDirection_Forward))
                }
            }
            let base = frame * nBins
            powerSpec[base] = splitReal[0] * splitReal[0]
            powerSpec[base + halfPadded] = splitImag[0] * splitImag[0]
            for k in 1..<halfPadded {
                powerSpec[base + k] = splitReal[k] * splitReal[k] + splitImag[k] * splitImag[k]
            }
        }

        // Mel filterbank
        var powerSpecT = [Float](repeating: 0, count: nBins * nFrames)
        vDSP_mtrans(powerSpec, 1, &powerSpecT, 1, vDSP_Length(nBins), vDSP_Length(nFrames))
        var melSpec = [Float](repeating: 0, count: config.numMelBins * nFrames)
        vDSP_mmul(melFilterbank, 1, powerSpecT, 1, &melSpec, 1,
                  vDSP_Length(config.numMelBins), vDSP_Length(nFrames), vDSP_Length(nBins))

        // Log
        var guardVal = logGuard
        vDSP_vsadd(melSpec, 1, &guardVal, &melSpec, 1, vDSP_Length(melSpec.count))
        var countN = Int32(melSpec.count)
        vvlogf(&melSpec, melSpec, &countN)

        // Update running stats and normalize using accumulated mean/std
        let validFrames = min(melLength, nFrames)
        melSpec.withUnsafeMutableBufferPointer { buf in
            for bin in 0..<config.numMelBins {
                let base = buf.baseAddress! + bin * nFrames

                // Accumulate sum and sum-of-squares for this bin
                var chunkSum: Float = 0
                var chunkSumSq: Float = 0
                vDSP_sve(base, 1, &chunkSum, vDSP_Length(validFrames))
                vDSP_svesq(base, 1, &chunkSumSq, vDSP_Length(validFrames))

                runningSum[bin] += chunkSum
                runningSumSq[bin] += chunkSumSq
            }
            runningCount += validFrames

            // Normalize using running mean/std (Bessel-corrected)
            let n = Float(max(runningCount, 1))
            for bin in 0..<config.numMelBins {
                let base = buf.baseAddress! + bin * nFrames

                let mean = runningSum[bin] / n
                var negMean = -mean
                vDSP_vsadd(base, 1, &negMean, base, 1, vDSP_Length(validFrames))

                let variance = max(runningSumSq[bin] / n - mean * mean, 0)
                let std = sqrt(variance * n / max(n - 1, 1))
                var invStd = 1.0 / (std + 1e-5)
                vDSP_vsmul(base, 1, &invStd, base, 1, vDSP_Length(validFrames))

                // Zero pad
                if validFrames < nFrames {
                    memset(base + validFrames, 0, (nFrames - validFrames) * MemoryLayout<Float>.stride)
                }
            }
        }

        let mel = try MLMultiArray(
            shape: [1, config.numMelBins as NSNumber, nFrames as NSNumber], dataType: .float32)
        memcpy(mel.dataPointer.assumingMemoryBound(to: Float.self),
               melSpec, config.numMelBins * nFrames * MemoryLayout<Float>.stride)

        return (mel, melLength)
    }

    /// Reset running normalization state (call when starting a new session).
    func resetRunningStats() {
        runningSum = [Float](repeating: 0, count: config.numMelBins)
        runningSumSq = [Float](repeating: 0, count: config.numMelBins)
        runningCount = 0
    }

    // MARK: - Mel Filterbank

    private static func buildMelFilterbank(
        nMels: Int, nBins: Int, sampleRate: Int, paddedFFT: Int
    ) -> [Float] {
        let fMax: Float = Float(sampleRate) / 2.0
        let minLogHertz: Float = 1000.0
        let minLogMel: Float = 15.0
        let logstepHzToMel: Float = 27.0 / log(6.4)
        let logstepMelToHz: Float = log(6.4) / 27.0

        func hzToMel(_ hz: Float) -> Float {
            hz < minLogHertz ? 3.0 * hz / 200.0 : minLogMel + log(hz / minLogHertz) * logstepHzToMel
        }
        func melToHz(_ mel: Float) -> Float {
            mel < minLogMel ? 200.0 * mel / 3.0 : minLogHertz * exp((mel - minLogMel) * logstepMelToHz)
        }

        var fftFreqs = [Float](repeating: 0, count: nBins)
        for i in 0..<nBins {
            fftFreqs[i] = Float(i) * Float(sampleRate) / Float(paddedFFT)
        }

        let melMin = hzToMel(0)
        let melMax = hzToMel(fMax)
        let nMelPoints = nMels + 2
        var melPoints = [Float](repeating: 0, count: nMelPoints)
        for i in 0..<nMelPoints {
            melPoints[i] = melMin + Float(i) * (melMax - melMin) / Float(nMelPoints - 1)
        }
        let filterFreqs = melPoints.map { melToHz($0) }

        var filterDiff = [Float](repeating: 0, count: nMelPoints - 1)
        for i in 0..<(nMelPoints - 1) {
            filterDiff[i] = filterFreqs[i + 1] - filterFreqs[i]
        }

        var filterbank = [Float](repeating: 0, count: nBins * nMels)
        for bin in 0..<nBins {
            let freq = fftFreqs[bin]
            for mel in 0..<nMels {
                let downSlope = (freq - filterFreqs[mel]) / filterDiff[mel]
                let upSlope = (filterFreqs[mel + 2] - freq) / filterDiff[mel + 1]
                filterbank[bin * nMels + mel] = max(0.0, min(downSlope, upSlope))
            }
        }

        for mel in 0..<nMels {
            let enorm = 2.0 / (filterFreqs[mel + 2] - filterFreqs[mel])
            for bin in 0..<nBins {
                filterbank[bin * nMels + mel] *= enorm
            }
        }

        var transposed = [Float](repeating: 0, count: nMels * nBins)
        for mel in 0..<nMels {
            for bin in 0..<nBins {
                transposed[mel * nBins + bin] = filterbank[bin * nMels + mel]
            }
        }

        return transposed
    }
}
