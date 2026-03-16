import Foundation
import Accelerate

/// 80-dim log-mel feature extractor for CAM++ speaker embeddings.
///
/// Matches Kaldi FBank defaults used by 3D-Speaker/CAM++:
/// nFFT=400, hop=160, 80 mel bins, 16kHz, pre-emphasis=0.97, Povey window,
/// HTK mel scale, Slaney normalization.
final class CamPlusPlusMelExtractor {
    let sampleRate: Int = 16000
    let nFFT: Int = 400
    let hopLength: Int = 160
    let nMels: Int = 80

    private let preEmphasis: Float = 0.97
    private let paddedFFT: Int = 512
    private let log2PaddedFFT: vDSP_Length = 9
    private var fftSetup: FFTSetup
    private var window: [Float]
    private var melFilterbank: [Float]  // [nMels, nBins]

    init() {
        // Povey window (Kaldi default): pow(0.5 - 0.5*cos(2π*i/(N-1)), 0.85)
        window = [Float](repeating: 0, count: 400)
        for i in 0..<400 {
            let hann = 0.5 - 0.5 * cos(2.0 * Float.pi * Float(i) / Float(399))
            window[i] = pow(hann, 0.85)
        }

        guard let setup = vDSP_create_fftsetup(9, FFTRadix(kFFTRadix2)) else {
            fatalError("Failed to create vDSP FFT setup")
        }
        fftSetup = setup

        melFilterbank = []
        setupMelFilterbank()
    }

    deinit {
        vDSP_destroy_fftsetup(fftSetup)
    }

    private func setupMelFilterbank() {
        let fMin: Float = 20.0
        let fMax: Float = Float(sampleRate) / 2.0

        func hzToMel(_ hz: Float) -> Float {
            return 2595.0 * log10(1.0 + hz / 700.0)
        }

        func melToHz(_ mel: Float) -> Float {
            return 700.0 * (pow(10.0, mel / 2595.0) - 1.0)
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
    /// - Parameter audio: PCM Float32 samples at 16kHz
    /// - Returns: `(melSpec, nFrames)` where melSpec is a flat `[nFrames * 80]` array
    func extractRaw(_ audio: [Float]) -> (melSpec: [Float], nFrames: Int) {
        let nBins = paddedFFT / 2 + 1
        let halfPadded = paddedFFT / 2

        // Pre-emphasis: y[n] = x[n] - coeff * x[n-1]
        var emphasized = [Float](repeating: 0, count: audio.count)
        if !audio.isEmpty {
            emphasized[0] = audio[0]
            for i in 1..<audio.count {
                emphasized[i] = audio[i] - preEmphasis * audio[i - 1]
            }
        }

        // Reflect padding
        let padLength = nFFT / 2
        var paddedAudio = [Float](repeating: 0, count: padLength + emphasized.count + padLength)

        for i in 0..<padLength {
            let srcIdx = min(padLength - i, emphasized.count - 1)
            paddedAudio[i] = emphasized[max(0, srcIdx)]
        }
        for i in 0..<emphasized.count {
            paddedAudio[padLength + i] = emphasized[i]
        }
        for i in 0..<padLength {
            let srcIdx = emphasized.count - 2 - i
            paddedAudio[padLength + emphasized.count + i] = emphasized[max(0, srcIdx)]
        }

        let nFrames = (paddedAudio.count - nFFT) / hopLength + 1

        var splitReal = [Float](repeating: 0, count: halfPadded)
        var splitImag = [Float](repeating: 0, count: halfPadded)
        var paddedFrame = [Float](repeating: 0, count: paddedFFT)
        var magnitude = [Float](repeating: 0, count: nFrames * nBins)

        for frame in 0..<nFrames {
            let start = frame * hopLength

            paddedAudio.withUnsafeBufferPointer { buf in
                vDSP_vmul(buf.baseAddress! + start, 1, window, 1, &paddedFrame, 1, vDSP_Length(nFFT))
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

        // Log-mel: log(max(x, 1e-10))
        let count = melSpec.count
        var countN = Int32(count)
        var epsilon: Float = 1e-10
        vDSP_vclip(melSpec, 1, &epsilon, [Float.greatestFiniteMagnitude], &melSpec, 1, vDSP_Length(count))
        vvlogf(&melSpec, melSpec, &countN)

        return (melSpec, nFrames)
    }
}
