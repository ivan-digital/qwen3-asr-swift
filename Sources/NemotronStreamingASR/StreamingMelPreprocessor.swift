import Accelerate
import AudioCommon
import CoreML
import Foundation

/// Mel spectrogram preprocessor for Nemotron Streaming ASR.
///
/// Implements the NeMo `AudioToMelSpectrogramPreprocessor` pipeline:
/// pre-emphasis → STFT → power spectrum → mel filterbank → log.
/// Streaming variant keeps running mean/std across chunks.
class StreamingMelPreprocessor {
    let config: NemotronStreamingConfig

    private let paddedFFT: Int = 512
    private let log2PaddedFFT: vDSP_Length = 9
    private let nBins: Int = 257
    private let reflectPad: Int = 256
    private let logGuard: Float = 5.960464477539063e-08

    private let fftSetup: FFTSetup
    private let hannWindow: [Float]
    private let melFilterbank: [Float]

    private var runningSum: [Float]
    private var runningSumSq: [Float]
    private(set) var runningCount: Int = 0

    init(config: NemotronStreamingConfig) {
        self.config = config

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

    /// Extract raw log-mel (no normalization) — the Nemotron encoder handles
    /// normalization internally (matches NeMo `normalize: "NA"`).
    func extractRaw(_ audio: [Float]) throws -> (mel: MLMultiArray, melLength: Int) {
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

        // Zero center padding (pad_mode=constant, NeMo default)
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

        let winOffset = (paddedFFT - config.winLength) / 2

        for frame in 0..<nFrames {
            let start = frame * config.hopLength

            memset(&paddedFrame, 0, paddedFFT * MemoryLayout<Float>.stride)
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
            // vDSP_fft_zrip scales 2x vs torch.stft — divide power by 4.
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
