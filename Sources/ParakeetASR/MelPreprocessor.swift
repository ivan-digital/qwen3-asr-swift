import Accelerate
import CoreML
import Foundation

/// Mel spectrogram preprocessor for Parakeet TDT models.
///
/// Implements the NeMo `AudioToMelSpectrogramPreprocessor` pipeline in Swift:
/// pre-emphasis → STFT → power spectrum → mel filterbank → log → per-feature normalization.
///
/// Uses vDSP/Accelerate for performance. No CoreML model needed — pure signal processing.
struct MelPreprocessor {
    let config: ParakeetConfig

    private let paddedFFT: Int = 512      // n_fft (already power of 2)
    private let log2PaddedFFT: vDSP_Length = 9
    private let nBins: Int = 257          // paddedFFT / 2 + 1
    private let reflectPad: Int = 256     // n_fft / 2
    private let logGuard: Float = 5.960464477539063e-08  // 2^{-24}

    private let fftSetup: FFTSetup
    private let hannWindow: [Float]       // win_length = 400
    private let melFilterbank: [Float]    // [nMels, nBins] = [128, 257]

    init(config: ParakeetConfig) {
        self.config = config

        // Hann window (periodic): 0.5 * (1 - cos(2π*i/N))
        var window = [Float](repeating: 0, count: config.winLength)
        for i in 0..<config.winLength {
            window[i] = 0.5 * (1.0 - cos(2.0 * Float.pi * Float(i) / Float(config.winLength)))
        }
        self.hannWindow = window

        guard let setup = vDSP_create_fftsetup(9, FFTRadix(kFFTRadix2)) else {
            fatalError("Failed to create vDSP FFT setup")
        }
        self.fftSetup = setup

        self.melFilterbank = MelPreprocessor.buildMelFilterbank(
            nMels: config.numMelBins,
            nBins: 257,
            sampleRate: config.sampleRate,
            paddedFFT: 512
        )
    }

    /// Extract mel spectrogram from audio samples.
    ///
    /// - Parameters:
    ///   - audio: PCM Float32 samples at 16kHz
    /// - Returns: Tuple of (mel MLMultiArray [1, 128, T], melLength Int)
    func extract(_ audio: [Float]) throws -> (mel: MLMultiArray, melLength: Int) {
        // Pre-emphasis: x[n] = audio[n] - 0.97 * audio[n-1]
        var preemphasized = [Float](repeating: 0, count: audio.count)
        preemphasized[0] = audio[0]
        for i in 1..<audio.count {
            preemphasized[i] = audio[i] - config.preEmphasis * audio[i - 1]
        }

        // Reflect padding (n_fft // 2 = 256 on each side, matching torch.stft center=True)
        let totalLen = reflectPad + preemphasized.count + reflectPad
        var padded = [Float](repeating: 0, count: totalLen)
        // Left reflect pad
        for i in 0..<reflectPad {
            padded[i] = preemphasized[reflectPad - i]
        }
        // Center (original signal)
        for i in 0..<preemphasized.count {
            padded[reflectPad + i] = preemphasized[i]
        }
        // Right reflect pad
        for i in 0..<reflectPad {
            let srcIdx = preemphasized.count - 2 - i
            padded[reflectPad + preemphasized.count + i] = preemphasized[max(0, srcIdx)]
        }

        let nFrames = (padded.count - paddedFFT) / config.hopLength + 1
        let melLength = audio.count / config.hopLength  // NeMo: floor(samples / hop)

        let halfPadded = paddedFFT / 2

        // STFT + power spectrum
        var powerSpec = [Float](repeating: 0, count: nFrames * nBins)
        var splitReal = [Float](repeating: 0, count: halfPadded)
        var splitImag = [Float](repeating: 0, count: halfPadded)
        var paddedFrame = [Float](repeating: 0, count: paddedFFT)

        for frame in 0..<nFrames {
            let start = frame * config.hopLength

            // Apply window to first win_length samples
            padded.withUnsafeBufferPointer { buf in
                vDSP_vmul(buf.baseAddress! + start, 1, hannWindow, 1,
                          &paddedFrame, 1, vDSP_Length(config.winLength))
            }
            // Zero-pad from win_length to paddedFFT
            for i in config.winLength..<paddedFFT {
                paddedFrame[i] = 0
            }

            // Pack into split-complex format for vDSP
            for i in 0..<halfPadded {
                splitReal[i] = paddedFrame[2 * i]
                splitImag[i] = paddedFrame[2 * i + 1]
            }

            // FFT
            splitReal.withUnsafeMutableBufferPointer { realBuf in
                splitImag.withUnsafeMutableBufferPointer { imagBuf in
                    var splitComplex = DSPSplitComplex(
                        realp: realBuf.baseAddress!,
                        imagp: imagBuf.baseAddress!)
                    vDSP_fft_zrip(fftSetup, &splitComplex, 1,
                                  log2PaddedFFT, FFTDirection(kFFTDirection_Forward))
                }
            }

            // Power spectrum: |X[k]|^2 = real^2 + imag^2
            let base = frame * nBins
            // DC component (packed in realp[0])
            powerSpec[base] = splitReal[0] * splitReal[0]
            // Nyquist component (packed in imagp[0])
            powerSpec[base + halfPadded] = splitImag[0] * splitImag[0]
            // Other bins
            for k in 1..<halfPadded {
                powerSpec[base + k] = splitReal[k] * splitReal[k] + splitImag[k] * splitImag[k]
            }
        }

        // Mel filterbank: [nMels, nBins] × [nBins, nFrames] → [nMels, nFrames]
        // powerSpec is [nFrames, nBins], need to transpose for matmul
        var powerSpecT = [Float](repeating: 0, count: nBins * nFrames)
        vDSP_mtrans(powerSpec, 1, &powerSpecT, 1,
                    vDSP_Length(nBins), vDSP_Length(nFrames))

        var melSpec = [Float](repeating: 0, count: config.numMelBins * nFrames)
        vDSP_mmul(melFilterbank, 1, powerSpecT, 1, &melSpec, 1,
                  vDSP_Length(config.numMelBins), vDSP_Length(nFrames), vDSP_Length(nBins))

        // Log: log(mel + guard)
        let count = melSpec.count
        var guardVal = logGuard
        vDSP_vsadd(melSpec, 1, &guardVal, &melSpec, 1, vDSP_Length(count))
        var countN = Int32(count)
        vvlogf(&melSpec, melSpec, &countN)

        // Per-feature normalization: for each mel bin, normalize over valid frames
        // NeMo: mean = x[:, :melLength].mean(dim=1), std = x[:, :melLength].std(dim=1)
        for bin in 0..<config.numMelBins {
            let offset = bin * nFrames

            // Mean
            var sum: Float = 0
            for t in 0..<melLength {
                sum += melSpec[offset + t]
            }
            let mean = sum / Float(melLength)

            // Std (Bessel's correction: N-1, matching torch.std)
            var varSum: Float = 0
            for t in 0..<melLength {
                let diff = melSpec[offset + t] - mean
                varSum += diff * diff
            }
            let std = sqrt(varSum / Float(melLength - 1))
            let invStd = 1.0 / (std + 1e-5)

            // Normalize valid frames
            for t in 0..<melLength {
                melSpec[offset + t] = (melSpec[offset + t] - mean) * invStd
            }
            // Zero out frames beyond melLength
            for t in melLength..<nFrames {
                melSpec[offset + t] = 0
            }
        }

        // Create MLMultiArray [1, numMelBins, nFrames] in float16
        let mel = try MLMultiArray(
            shape: [1, config.numMelBins as NSNumber, nFrames as NSNumber],
            dataType: .float16)
        let melPtr = mel.dataPointer.assumingMemoryBound(to: Float16.self)

        // melSpec is [numMelBins, nFrames] row-major, same as MLMultiArray [1, 128, T]
        for i in 0..<(config.numMelBins * nFrames) {
            melPtr[i] = Float16(melSpec[i])
        }

        return (mel, melLength)
    }

    // MARK: - Mel Filterbank Construction

    /// Build Slaney-normalized mel filterbank matrix.
    private static func buildMelFilterbank(
        nMels: Int, nBins: Int, sampleRate: Int, paddedFFT: Int
    ) -> [Float] {
        let fMin: Float = 0.0
        let fMax: Float = Float(sampleRate) / 2.0

        // Slaney mel scale (piecewise: linear <1000 Hz, log ≥1000 Hz)
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

        // FFT bin frequencies
        var fftFreqs = [Float](repeating: 0, count: nBins)
        for i in 0..<nBins {
            fftFreqs[i] = Float(i) * Float(sampleRate) / Float(paddedFFT)
        }

        // Mel-spaced center frequencies
        let melMin = hzToMel(fMin)
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

        // Build filterbank [nBins, nMels] then transpose
        var filterbank = [Float](repeating: 0, count: nBins * nMels)
        for bin in 0..<nBins {
            let freq = fftFreqs[bin]
            for mel in 0..<nMels {
                let downSlope = (freq - filterFreqs[mel]) / filterDiff[mel]
                let upSlope = (filterFreqs[mel + 2] - freq) / filterDiff[mel + 1]
                filterbank[bin * nMels + mel] = max(0.0, min(downSlope, upSlope))
            }
        }

        // Slaney normalization
        for mel in 0..<nMels {
            let enorm = 2.0 / (filterFreqs[mel + 2] - filterFreqs[mel])
            for bin in 0..<nBins {
                filterbank[bin * nMels + mel] *= enorm
            }
        }

        // Transpose to [nMels, nBins]
        var transposed = [Float](repeating: 0, count: nMels * nBins)
        for mel in 0..<nMels {
            for bin in 0..<nBins {
                transposed[mel * nBins + bin] = filterbank[bin * nMels + mel]
            }
        }

        return transposed
    }
}
