import Foundation
import Accelerate
import MLX

/// 80-dim log-mel feature extractor for WeSpeaker (speaker embeddings).
///
/// Uses vDSP for FFT and mel filterbank computation.
/// Parameters: nFFT=400, hop=160, 80 mel bins, Hamming window, 16kHz.
/// Simple log-mel: `log(max(mel, 1e-10))` — no Whisper-specific normalization.
class MelFeatureExtractor {
    let sampleRate: Int = 16000
    let nFFT: Int = 400
    let hopLength: Int = 160
    let nMels: Int = 80

    private let paddedFFT: Int = 512
    private let log2PaddedFFT: vDSP_Length = 9
    private var fftSetup: FFTSetup
    private var hammingWindow: [Float]
    private var melFilterbank: [Float]  // [nMels, nBins]

    init() {
        // Hamming window: 0.54 - 0.46 * cos(2π*i/N)
        hammingWindow = [Float](repeating: 0, count: 400)
        for i in 0..<400 {
            hammingWindow[i] = 0.54 - 0.46 * cos(2.0 * Float.pi * Float(i) / Float(400))
        }

        guard let setup = vDSP_create_fftsetup(9, FFTRadix(kFFTRadix2)) else {
            fatalError("Failed to create vDSP FFT setup")
        }
        fftSetup = setup

        // Initialize filterbank (will be replaced in setupMelFilterbank)
        melFilterbank = []
        setupMelFilterbank()
    }

    deinit {
        vDSP_destroy_fftsetup(fftSetup)
    }

    private func setupMelFilterbank() {
        let fMin: Float = 0.0
        let fMax: Float = Float(sampleRate) / 2.0

        // Slaney mel scale
        let minLogHertz: Float = 1000.0
        let minLogMel: Float = 15.0
        let logstepHzToMel: Float = 27.0 / log(6.4)
        let logstepMelToHz: Float = log(6.4) / 27.0

        func hzToMel(_ hz: Float) -> Float {
            if hz < minLogHertz {
                return 3.0 * hz / 200.0
            } else {
                return minLogMel + log(hz / minLogHertz) * logstepHzToMel
            }
        }

        func melToHz(_ mel: Float) -> Float {
            if mel < minLogMel {
                return 200.0 * mel / 3.0
            } else {
                return minLogHertz * exp((mel - minLogMel) * logstepMelToHz)
            }
        }

        let nBins = paddedFFT / 2 + 1  // 257

        var fftFreqs = [Float](repeating: 0, count: nBins)
        for i in 0..<nBins {
            fftFreqs[i] = Float(i) * Float(sampleRate) / Float(paddedFFT)
        }

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

        // Build filterbank [nBins, nMels]
        var filterbank = [Float](repeating: 0, count: nBins * nMels)
        for bin in 0..<nBins {
            let freq = fftFreqs[bin]
            for mel in 0..<nMels {
                let lowFreq = filterFreqs[mel]
                let highFreq = filterFreqs[mel + 2]
                let downSlope = (freq - lowFreq) / filterDiff[mel]
                let upSlope = (highFreq - freq) / filterDiff[mel + 1]
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

        self.melFilterbank = transposed
    }

    /// Extract raw 80-dim log-mel features as a flat Float array.
    ///
    /// This avoids creating an MLXArray, useful for CoreML paths that need
    /// raw floats without an MLX round-trip.
    ///
    /// - Parameter audio: PCM Float32 samples at 16kHz
    /// - Returns: `(melSpec, nFrames)` where melSpec is a flat `[nFrames * 80]` array
    func extractRaw(_ audio: [Float]) -> (melSpec: [Float], nFrames: Int) {
        let nBins = paddedFFT / 2 + 1
        let halfPadded = paddedFFT / 2

        // Reflect padding
        let padLength = nFFT / 2
        var paddedAudio = [Float](repeating: 0, count: padLength + audio.count + padLength)

        for i in 0..<padLength {
            let srcIdx = min(padLength - i, audio.count - 1)
            paddedAudio[i] = audio[max(0, srcIdx)]
        }
        for i in 0..<audio.count {
            paddedAudio[padLength + i] = audio[i]
        }
        for i in 0..<padLength {
            let srcIdx = audio.count - 2 - i
            paddedAudio[padLength + audio.count + i] = audio[max(0, srcIdx)]
        }

        let nFrames = (paddedAudio.count - nFFT) / hopLength + 1

        var splitReal = [Float](repeating: 0, count: halfPadded)
        var splitImag = [Float](repeating: 0, count: halfPadded)
        var paddedFrame = [Float](repeating: 0, count: paddedFFT)
        var magnitude = [Float](repeating: 0, count: nFrames * nBins)

        for frame in 0..<nFrames {
            let start = frame * hopLength

            paddedAudio.withUnsafeBufferPointer { buf in
                vDSP_vmul(buf.baseAddress! + start, 1, hammingWindow, 1, &paddedFrame, 1, vDSP_Length(nFFT))
            }
            for i in nFFT..<paddedFFT {
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
                    vDSP_fft_zrip(fftSetup, &splitComplex, 1, log2PaddedFFT, FFTDirection(kFFTDirection_Forward))
                }
            }

            let baseIdx = frame * nBins
            magnitude[baseIdx] = splitReal[0] * splitReal[0]
            magnitude[baseIdx + halfPadded] = splitImag[0] * splitImag[0]
            for k in 1..<halfPadded {
                magnitude[baseIdx + k] = splitReal[k] * splitReal[k] + splitImag[k] * splitImag[k]
            }
        }

        // Mel filterbank matmul
        var melSpec = [Float](repeating: 0, count: nFrames * nMels)
        var filterbankT = [Float](repeating: 0, count: nBins * nMels)
        vDSP_mtrans(melFilterbank, 1, &filterbankT, 1, vDSP_Length(nBins), vDSP_Length(nMels))

        vDSP_mmul(magnitude, 1, filterbankT, 1, &melSpec, 1,
                  vDSP_Length(nFrames), vDSP_Length(nMels), vDSP_Length(nBins))

        // Simple log-mel: log(max(x, 1e-10))
        let count = melSpec.count
        var countN = Int32(count)

        var epsilon: Float = 1e-10
        vDSP_vclip(melSpec, 1, &epsilon, [Float.greatestFiniteMagnitude], &melSpec, 1, vDSP_Length(count))
        vvlogf(&melSpec, melSpec, &countN)

        return (melSpec, nFrames)
    }

    /// Extract 80-dim log-mel features from audio.
    /// - Parameter audio: PCM Float32 samples at 16kHz
    /// - Returns: `MLXArray [T, 80]` — time-major mel features
    func extract(_ audio: [Float]) -> MLXArray {
        let (melSpec, nFrames) = extractRaw(audio)
        return MLXArray(melSpec, [nFrames, nMels])
    }
}
