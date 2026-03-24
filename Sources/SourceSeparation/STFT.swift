import Accelerate
import Foundation

/// STFT processor for stereo 44.1kHz audio (Open-Unmix configuration).
///
/// Uses vDSP for FFT and overlap-add synthesis. Supports center padding
/// with reflect mode for first/last frames.
struct STFTProcessor {
    let nFFT: Int       // 4096
    let nHop: Int       // 1024
    let nBins: Int      // nFFT/2 + 1 = 2049
    let window: [Float] // Hann window

    private let fftSetup: vDSP_DFT_Setup
    private let ifftSetup: vDSP_DFT_Setup

    init(nFFT: Int = 4096, nHop: Int = 1024) {
        self.nFFT = nFFT
        self.nHop = nHop
        self.nBins = nFFT / 2 + 1

        // Hann window (periodic)
        var w = [Float](repeating: 0, count: nFFT)
        vDSP_hann_window(&w, vDSP_Length(nFFT), Int32(vDSP_HANN_NORM))
        self.window = w

        // vDSP DFT setup (supports non-power-of-2, but 4096 is power-of-2)
        self.fftSetup = vDSP_DFT_zop_CreateSetup(
            nil, vDSP_Length(nFFT), .FORWARD)!
        self.ifftSetup = vDSP_DFT_zop_CreateSetup(
            nil, vDSP_Length(nFFT), .INVERSE)!
    }

    /// Forward STFT on a single channel.
    /// - Parameter audio: Mono audio samples
    /// - Returns: (real, imag) each of shape [nBins, T]
    func forward(_ audio: [Float]) -> (real: [[Float]], imag: [[Float]]) {
        // Center padding (reflect)
        let padLen = nFFT / 2
        var padded = [Float](repeating: 0, count: padLen + audio.count + padLen)

        // Reflect pad left
        for i in 0..<padLen {
            let srcIdx = min(padLen - i, audio.count - 1)
            padded[i] = audio[max(0, srcIdx)]
        }
        // Copy audio
        for i in 0..<audio.count {
            padded[padLen + i] = audio[i]
        }
        // Reflect pad right
        for i in 0..<padLen {
            let srcIdx = audio.count - 2 - i
            padded[padLen + audio.count + i] = audio[max(0, srcIdx)]
        }

        let nFrames = (padded.count - nFFT) / nHop + 1
        var realFrames = [[Float]](repeating: [Float](repeating: 0, count: nBins), count: nFrames)
        var imagFrames = [[Float]](repeating: [Float](repeating: 0, count: nBins), count: nFrames)

        var windowed = [Float](repeating: 0, count: nFFT)
        var inReal = [Float](repeating: 0, count: nFFT)
        var inImag = [Float](repeating: 0, count: nFFT)
        var outReal = [Float](repeating: 0, count: nFFT)
        var outImag = [Float](repeating: 0, count: nFFT)

        for frame in 0..<nFrames {
            let start = frame * nHop

            // Apply window
            vDSP_vmul(Array(padded[start..<(start + nFFT)]), 1, window, 1, &windowed, 1, vDSP_Length(nFFT))

            // Copy to real, zero imag
            inReal = windowed
            inImag = [Float](repeating: 0, count: nFFT)

            // Forward DFT
            vDSP_DFT_Execute(fftSetup, inReal, inImag, &outReal, &outImag)

            // Take first nBins (one-sided)
            realFrames[frame] = Array(outReal[0..<nBins])
            imagFrames[frame] = Array(outImag[0..<nBins])
        }

        return (realFrames, imagFrames)
    }

    /// Compute magnitude from STFT.
    /// - Returns: [T, nBins] magnitude spectrogram
    func magnitude(real: [[Float]], imag: [[Float]]) -> [[Float]] {
        let T = real.count
        var mag = [[Float]](repeating: [Float](repeating: 0, count: nBins), count: T)
        for t in 0..<T {
            for f in 0..<nBins {
                mag[t][f] = sqrt(real[t][f] * real[t][f] + imag[t][f] * imag[t][f])
            }
        }
        return mag
    }

    /// Phase angle from STFT.
    /// - Returns: [T, nBins] phase
    func phase(real: [[Float]], imag: [[Float]]) -> [[Float]] {
        let T = real.count
        var ph = [[Float]](repeating: [Float](repeating: 0, count: nBins), count: T)
        for t in 0..<T {
            for f in 0..<nBins {
                ph[t][f] = atan2(imag[t][f], real[t][f])
            }
        }
        return ph
    }

    /// Inverse STFT (overlap-add synthesis).
    /// - Parameters:
    ///   - real: [T, nBins] real part of complex spectrogram
    ///   - imag: [T, nBins] imaginary part
    ///   - length: Expected output length in samples
    /// - Returns: Reconstructed audio
    func inverse(real: [[Float]], imag: [[Float]], length: Int) -> [Float] {
        let T = real.count
        let padLen = nFFT / 2
        let paddedLen = padLen + length + padLen
        var output = [Float](repeating: 0, count: paddedLen)
        var windowSum = [Float](repeating: 0, count: paddedLen)

        var fullReal = [Float](repeating: 0, count: nFFT)
        var fullImag = [Float](repeating: 0, count: nFFT)
        var outReal = [Float](repeating: 0, count: nFFT)
        var outImag = [Float](repeating: 0, count: nFFT)

        for frame in 0..<T {
            // Mirror one-sided spectrum to full
            for f in 0..<nBins {
                fullReal[f] = real[frame][f]
                fullImag[f] = imag[frame][f]
            }
            for f in 1..<(nFFT / 2) {
                fullReal[nFFT - f] = real[frame][f]
                fullImag[nFFT - f] = -imag[frame][f]
            }

            // Inverse DFT
            vDSP_DFT_Execute(ifftSetup, fullReal, fullImag, &outReal, &outImag)

            // Scale by 1/N
            var scale = Float(1.0) / Float(nFFT)
            vDSP_vsmul(outReal, 1, &scale, &outReal, 1, vDSP_Length(nFFT))

            // Apply window and overlap-add
            let start = frame * nHop
            for i in 0..<nFFT where (start + i) < paddedLen {
                output[start + i] += outReal[i] * window[i]
                windowSum[start + i] += window[i] * window[i]
            }
        }

        // Normalize by window sum
        for i in 0..<paddedLen {
            if windowSum[i] > 1e-8 {
                output[i] /= windowSum[i]
            }
        }

        // Remove center padding
        return Array(output[padLen..<(padLen + length)])
    }

    /// Apply magnitude mask with original phase, then iSTFT.
    /// - Parameters:
    ///   - maskedMag: [T, nBins] masked magnitude from the model
    ///   - origReal: Original STFT real part
    ///   - origImag: Original STFT imaginary part
    ///   - length: Output audio length
    /// - Returns: Separated audio
    func applyMaskAndInvert(
        maskedMag: [[Float]],
        origReal: [[Float]],
        origImag: [[Float]],
        length: Int
    ) -> [Float] {
        let T = maskedMag.count
        let ph = phase(real: origReal, imag: origImag)

        var newReal = [[Float]](repeating: [Float](repeating: 0, count: nBins), count: T)
        var newImag = [[Float]](repeating: [Float](repeating: 0, count: nBins), count: T)

        for t in 0..<T {
            for f in 0..<nBins {
                newReal[t][f] = maskedMag[t][f] * cos(ph[t][f])
                newImag[t][f] = maskedMag[t][f] * sin(ph[t][f])
            }
        }

        return inverse(real: newReal, imag: newImag, length: length)
    }
}
