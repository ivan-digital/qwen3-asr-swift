import Foundation
import Accelerate
import MLX

// MARK: - Vorbis Window

/// Compute the Vorbis window used by DeepFilterNet3.
///
/// Formula: `w[n] = sin(pi/2 * sin^2(pi * (n + 0.5) / N))`
func computeVorbisWindow(size: Int) -> [Float] {
    var window = [Float](repeating: 0, count: size)
    let n = Float(size)
    for i in 0..<size {
        let x = Float.pi * (Float(i) + 0.5) / n
        let sinSq = sin(x) * sin(x)
        window[i] = sin(Float.pi / 2.0 * sinSq)
    }
    return window
}

// MARK: - ERB Filterbank

/// Compute the ERB (Equivalent Rectangular Bandwidth) filterbank.
///
/// Returns a `[freqBins, erbBands]` matrix mapping FFT bins to ERB bands,
/// and the inverse `[erbBands, freqBins]` matrix for decompression.
func computeERBFilterbank(
    config: DeepFilterNet3Config
) -> (forward: [Float], inverse: [Float], widths: [Int]) {
    let sr = Float(config.sampleRate)
    let freqBins = config.freqBins  // 481
    let nBands = config.erbBands    // 32
    let minFreqsPerBand = 2

    // ERB scale conversion
    func freq2erb(_ f: Float) -> Float {
        9.265 * log(1.0 + f / (24.7 * 9.265))
    }
    func erb2freq(_ e: Float) -> Float {
        24.7 * 9.265 * (exp(e / 9.265) - 1.0)
    }

    let nyquist = sr / 2.0
    let erbLow = freq2erb(0)
    let erbHigh = freq2erb(nyquist)
    let step = (erbHigh - erbLow) / Float(nBands)

    // Compute band widths (number of FFT bins per ERB band)
    var widths = [Int](repeating: 0, count: nBands)
    var totalBins = 0

    for band in 0..<nBands {
        let erbCenter = erbLow + (Float(band) + 0.5) * step
        let freqLow = erb2freq(erbLow + Float(band) * step)
        let freqHigh = erb2freq(erbLow + Float(band + 1) * step)
        _ = erbCenter  // used for center frequency reference

        let binLow = Int(round(freqLow * Float(config.fftSize) / sr))
        let binHigh = Int(round(freqHigh * Float(config.fftSize) / sr))
        var width = max(minFreqsPerBand, binHigh - binLow)

        if band == nBands - 1 {
            // Last band gets remaining bins
            width = freqBins - totalBins
        }

        widths[band] = width
        totalBins += width
    }

    // Adjust to ensure total == freqBins
    if totalBins != freqBins {
        widths[nBands - 1] += (freqBins - totalBins)
    }

    // Build forward filterbank [freqBins, nBands]
    // Each column has 1/width for its frequency bins (normalized)
    var forward = [Float](repeating: 0, count: freqBins * nBands)
    var binOffset = 0
    for band in 0..<nBands {
        let w = widths[band]
        let norm = 1.0 / Float(w)
        for bin in binOffset..<(binOffset + w) {
            if bin < freqBins {
                forward[bin * nBands + band] = norm
            }
        }
        binOffset += w
    }

    // Build inverse filterbank [nBands, freqBins]
    // Each row repeats 1.0 across its frequency bins (no normalization)
    var inverse = [Float](repeating: 0, count: nBands * freqBins)
    binOffset = 0
    for band in 0..<nBands {
        let w = widths[band]
        for bin in binOffset..<(binOffset + w) {
            if bin < freqBins {
                inverse[band * freqBins + bin] = 1.0
            }
        }
        binOffset += w
    }

    return (forward, inverse, widths)
}

// MARK: - STFT / iSTFT

/// Forward STFT using vDSP Accelerate.
///
/// Computes the Short-Time Fourier Transform of the input audio.
///
/// - Parameters:
///   - audio: Input audio samples
///   - window: Analysis window (Vorbis)
///   - fftSize: FFT window size (960)
///   - hopSize: Hop size (480)
/// - Returns: Complex spectrum as interleaved [real, imag] per bin,
///   shape `[numFrames, freqBins, 2]` flattened
final class STFTProcessor {
    let fftSize: Int
    let hopSize: Int
    let paddedFFT: Int
    let log2N: vDSP_Length
    let freqBins: Int
    let window: [Float]
    let fftSetup: FFTSetup
    /// Forward FFT scale: 1/2 (vDSP correction for real-to-complex)
    let forwardScale: Float
    /// Inverse FFT scale: 1/paddedFFT (vDSP correction for complex-to-real)
    let inverseScale: Float

    init(fftSize: Int, hopSize: Int, window: [Float]) {
        self.fftSize = fftSize
        self.hopSize = hopSize
        self.window = window

        // Next power of 2 for FFT
        var padded = 1
        var log2 = 0
        while padded < fftSize {
            padded *= 2
            log2 += 1
        }
        self.paddedFFT = padded
        self.log2N = vDSP_Length(log2)
        // Use all bins from padded FFT for perfect reconstruction
        self.freqBins = padded / 2 + 1

        // vDSP real-FFT scales: forward result is 2x, inverse result needs /N
        self.forwardScale = 0.5
        self.inverseScale = 1.0 / Float(padded)

        guard let setup = vDSP_create_fftsetup(self.log2N, FFTRadix(kFFTRadix2)) else {
            fatalError("Failed to create vDSP FFT setup for paddedFFT=\(padded)")
        }
        self.fftSetup = setup
    }

    deinit {
        vDSP_destroy_fftsetup(fftSetup)
    }

    /// Analysis STFT: audio → complex spectrum [numFrames][freqBins][2]
    ///
    /// Maintains analysis memory for streaming (overlap region from previous call).
    /// Returns (real, imaginary) arrays each of shape [numFrames * freqBins].
    func forward(audio: [Float], analysisMem: inout [Float]) -> (real: [Float], imag: [Float]) {
        let halfPadded = paddedFFT / 2
        let overlapSize = fftSize - hopSize

        // Build full buffer: [analysisMem | audio]
        let buffer = analysisMem + audio

        let numFrames = max(0, (buffer.count - fftSize) / hopSize + 1)
        guard numFrames > 0 else {
            analysisMem = Array(buffer.suffix(overlapSize))
            return ([], [])
        }

        var real = [Float](repeating: 0, count: numFrames * freqBins)
        var imag = [Float](repeating: 0, count: numFrames * freqBins)

        var paddedFrame = [Float](repeating: 0, count: paddedFFT)
        var splitReal = [Float](repeating: 0, count: halfPadded)
        var splitImag = [Float](repeating: 0, count: halfPadded)

        for frame in 0..<numFrames {
            let start = frame * hopSize

            // Apply window
            buffer.withUnsafeBufferPointer { buf in
                vDSP_vmul(buf.baseAddress! + start, 1, window, 1, &paddedFrame, 1, vDSP_Length(fftSize))
            }
            // Zero-pad
            for i in fftSize..<paddedFFT {
                paddedFrame[i] = 0
            }

            // Pack into split-complex
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
                    vDSP_fft_zrip(fftSetup, &splitComplex, 1, log2N, FFTDirection(kFFTDirection_Forward))
                }
            }

            let baseIdx = frame * freqBins

            // DC component (purely real, stored in realp[0])
            real[baseIdx] = splitReal[0] * forwardScale
            imag[baseIdx] = 0

            // Bins 1 to halfPadded-1
            for k in 1..<halfPadded {
                real[baseIdx + k] = splitReal[k] * forwardScale
                imag[baseIdx + k] = splitImag[k] * forwardScale
            }

            // Nyquist bin (purely real, packed in imagp[0])
            real[baseIdx + halfPadded] = splitImag[0] * forwardScale
            imag[baseIdx + halfPadded] = 0
        }

        // Update analysis memory
        let consumed = numFrames * hopSize
        analysisMem = Array(buffer.suffix(buffer.count - consumed))
        // Ensure it's exactly overlapSize
        if analysisMem.count > overlapSize {
            analysisMem = Array(analysisMem.suffix(overlapSize))
        } else if analysisMem.count < overlapSize {
            analysisMem = [Float](repeating: 0, count: overlapSize - analysisMem.count) + analysisMem
        }

        return (real, imag)
    }

    /// Inverse STFT: complex spectrum → audio
    ///
    /// - Parameters:
    ///   - real: Real parts [numFrames * freqBins]
    ///   - imag: Imaginary parts [numFrames * freqBins]
    ///   - synthesisMem: Overlap-add buffer (mutated)
    /// - Returns: Reconstructed audio samples
    func inverse(real: [Float], imag: [Float], synthesisMem: inout [Float]) -> [Float] {
        let halfPadded = paddedFFT / 2
        let numFrames = real.count / freqBins
        guard numFrames > 0 else { return [] }

        let outputLen = numFrames * hopSize
        var output = [Float](repeating: 0, count: outputLen)

        var splitReal = [Float](repeating: 0, count: halfPadded)
        var splitImag = [Float](repeating: 0, count: halfPadded)
        var timeFrame = [Float](repeating: 0, count: paddedFFT)

        for frame in 0..<numFrames {
            let baseIdx = frame * freqBins

            // Pack into split-complex format
            splitReal[0] = real[baseIdx]  // DC
            splitImag[0] = real[baseIdx + halfPadded]  // Nyquist

            // Bins 1 to halfPadded-1
            for k in 1..<halfPadded {
                splitReal[k] = real[baseIdx + k]
                splitImag[k] = imag[baseIdx + k]
            }

            // Inverse FFT
            splitReal.withUnsafeMutableBufferPointer { realBuf in
                splitImag.withUnsafeMutableBufferPointer { imagBuf in
                    var splitComplex = DSPSplitComplex(
                        realp: realBuf.baseAddress!,
                        imagp: imagBuf.baseAddress!)
                    vDSP_fft_zrip(fftSetup, &splitComplex, 1, log2N, FFTDirection(kFFTDirection_Inverse))
                }
            }

            // Unpack from split-complex to interleaved
            for i in 0..<halfPadded {
                timeFrame[2 * i] = splitReal[i]
                timeFrame[2 * i + 1] = splitImag[i]
            }

            // vDSP real-FFT inverse scale: 1/paddedFFT
            var scale = inverseScale
            vDSP_vsmul(timeFrame, 1, &scale, &timeFrame, 1, vDSP_Length(paddedFFT))

            // Apply window
            var windowed = [Float](repeating: 0, count: fftSize)
            vDSP_vmul(timeFrame, 1, window, 1, &windowed, 1, vDSP_Length(fftSize))

            // Overlap-add with synthesis memory
            let outStart = frame * hopSize
            for i in 0..<fftSize {
                if i < synthesisMem.count {
                    windowed[i] += synthesisMem[i]
                }
            }

            // Copy hop-size samples to output
            for i in 0..<hopSize {
                if outStart + i < outputLen {
                    output[outStart + i] = windowed[i]
                }
            }

            // Update synthesis memory (remaining overlap)
            let overlapSize = fftSize - hopSize
            synthesisMem = Array(windowed[hopSize..<fftSize])
            // Pad to full overlap size if needed
            if synthesisMem.count < overlapSize {
                synthesisMem.append(contentsOf: [Float](repeating: 0, count: overlapSize - synthesisMem.count))
            }
        }

        return output
    }
}

// MARK: - Feature Extraction

/// Compute ERB power features from complex spectrum.
///
/// - Parameters:
///   - real: Real parts [numFrames * freqBins]
///   - imag: Imaginary parts [numFrames * freqBins]
///   - erbFb: Forward ERB filterbank [freqBins, erbBands]
///   - freqBins: Number of frequency bins
///   - erbBands: Number of ERB bands
///   - numFrames: Number of time frames
/// - Returns: ERB features [numFrames * erbBands] in dB, not yet normalized
func computeERBFeatures(
    real: [Float], imag: [Float],
    erbFb: [Float],
    freqBins: Int, erbBands: Int, numFrames: Int
) -> [Float] {
    // Power spectrum: |X|^2
    var power = [Float](repeating: 0, count: numFrames * freqBins)
    for i in 0..<power.count {
        power[i] = real[i] * real[i] + imag[i] * imag[i]
    }

    // ERB compression: power[T, F] @ erbFb[F, B] → erb[T, B]
    var erb = [Float](repeating: 0, count: numFrames * erbBands)
    vDSP_mmul(power, 1, erbFb, 1, &erb, 1,
              vDSP_Length(numFrames), vDSP_Length(erbBands), vDSP_Length(freqBins))

    // Convert to dB: 10 * log10(erb + 1e-10)
    var count32 = Int32(erb.count)
    var epsilon: Float = 1e-10
    vDSP_vsadd(erb, 1, &epsilon, &erb, 1, vDSP_Length(erb.count))
    vvlog10f(&erb, erb, &count32)
    var scale: Float = 10.0
    vDSP_vsmul(erb, 1, &scale, &erb, 1, vDSP_Length(erb.count))

    return erb
}

/// Apply exponential mean normalization to ERB features.
///
/// For each ERB band: `state = x * (1 - alpha) + state * alpha; x_norm = (x - state) / 40`
///
/// - Parameters:
///   - erb: ERB features [numFrames * erbBands], modified in place
///   - state: Running mean state [erbBands], modified in place
///   - alpha: Exponential decay factor
///   - erbBands: Number of ERB bands
///   - numFrames: Number of time frames
func applyMeanNormalization(
    _ erb: inout [Float],
    state: inout [Float],
    alpha: Float,
    erbBands: Int,
    numFrames: Int
) {
    let oneMinusAlpha = 1.0 - alpha
    for t in 0..<numFrames {
        let baseIdx = t * erbBands
        for b in 0..<erbBands {
            let x = erb[baseIdx + b]
            state[b] = x * oneMinusAlpha + state[b] * alpha
            erb[baseIdx + b] = (x - state[b]) / 40.0
        }
    }
}

/// Apply exponential unit normalization to complex spec features.
///
/// For each frequency bin: `state = |x| * (1 - alpha) + state * alpha; x_norm = x / sqrt(state)`
///
/// - Parameters:
///   - real: Real parts [numFrames * dfBins], modified in place
///   - imag: Imaginary parts [numFrames * dfBins], modified in place
///   - state: Running magnitude state [dfBins], modified in place
///   - alpha: Exponential decay factor
///   - dfBins: Number of deep filtering bins
///   - numFrames: Number of time frames
func applyUnitNormalization(
    real: inout [Float], imag: inout [Float],
    state: inout [Float],
    alpha: Float,
    dfBins: Int,
    numFrames: Int
) {
    let oneMinusAlpha = 1.0 - alpha
    for t in 0..<numFrames {
        let baseIdx = t * dfBins
        for f in 0..<dfBins {
            let re = real[baseIdx + f]
            let im = imag[baseIdx + f]
            let mag = sqrt(re * re + im * im)
            state[f] = mag * oneMinusAlpha + state[f] * alpha
            let norm = sqrt(max(state[f], 1e-10))
            real[baseIdx + f] = re / norm
            imag[baseIdx + f] = im / norm
        }
    }
}

// MARK: - Deep Filtering

/// Apply deep filtering coefficients to the spectrum.
///
/// For each time `t` and frequency `f` (0..<dfBins):
/// ```
/// Y(t, f) = sum_{n=0}^{order-1} X(t + n - padBefore, f) * W(n, t, f)
/// ```
/// where multiplication is complex.
///
/// - Parameters:
///   - specReal: Full spectrum real [numFrames * freqBins]
///   - specImag: Full spectrum imag [numFrames * freqBins]
///   - coefs: Deep filter coefficients [numFrames * dfBins * dfOrder * 2] (real, imag interleaved per tap)
///   - dfBins: Number of frequency bins to filter
///   - dfOrder: Number of filter taps
///   - dfLookahead: Lookahead in frames
///   - numFrames: Number of time frames
///   - freqBins: Total frequency bins
/// - Returns: Filtered spectrum (real, imag) for the first dfBins, [numFrames * dfBins] each
func applyDeepFiltering(
    specReal: [Float], specImag: [Float],
    coefs: [Float],
    dfBins: Int, dfOrder: Int, dfLookahead: Int,
    numFrames: Int, freqBins: Int
) -> (real: [Float], imag: [Float]) {
    let padBefore = dfOrder - 1 - dfLookahead  // 5-1-2 = 2

    var outReal = [Float](repeating: 0, count: numFrames * dfBins)
    var outImag = [Float](repeating: 0, count: numFrames * dfBins)

    for t in 0..<numFrames {
        for f in 0..<dfBins {
            var sumRe: Float = 0
            var sumIm: Float = 0

            for n in 0..<dfOrder {
                let srcT = t + n - padBefore
                let coefIdx = (t * dfBins * dfOrder + f * dfOrder + n) * 2

                let wRe = coefs[coefIdx]
                let wIm = coefs[coefIdx + 1]

                // Clamp source frame index
                let clampedT = max(0, min(numFrames - 1, srcT))
                let srcIdx = clampedT * freqBins + f

                let xRe = specReal[srcIdx]
                let xIm = specImag[srcIdx]

                // Complex multiply: (xRe + j*xIm) * (wRe + j*wIm)
                sumRe += xRe * wRe - xIm * wIm
                sumIm += xIm * wRe + xRe * wIm
            }

            let outIdx = t * dfBins + f
            outReal[outIdx] = sumRe
            outImag[outIdx] = sumIm
        }
    }

    return (outReal, outImag)
}

// MARK: - ERB Mask Application

/// Apply ERB mask to the full spectrum.
///
/// Expands the 32-band ERB mask to 481 frequency bins using the inverse filterbank,
/// then multiplies the spectrum by the mask.
///
/// - Parameters:
///   - specReal: Spectrum real [numFrames * freqBins], modified in place
///   - specImag: Spectrum imag [numFrames * freqBins], modified in place
///   - erbMask: ERB mask [numFrames * erbBands] (sigmoid output, 0..1)
///   - erbInvFb: Inverse ERB filterbank [erbBands, freqBins]
///   - erbBands: Number of ERB bands
///   - freqBins: Number of frequency bins
///   - numFrames: Number of time frames
func applyERBMask(
    specReal: inout [Float], specImag: inout [Float],
    erbMask: [Float],
    erbInvFb: [Float],
    erbBands: Int, freqBins: Int, numFrames: Int
) {
    // Expand mask: mask[T, B] @ invFb[B, F] → fullMask[T, F]
    var fullMask = [Float](repeating: 0, count: numFrames * freqBins)
    vDSP_mmul(erbMask, 1, erbInvFb, 1, &fullMask, 1,
              vDSP_Length(numFrames), vDSP_Length(freqBins), vDSP_Length(erbBands))

    // Apply mask to spectrum
    vDSP_vmul(specReal, 1, fullMask, 1, &specReal, 1, vDSP_Length(specReal.count))
    vDSP_vmul(specImag, 1, fullMask, 1, &specImag, 1, vDSP_Length(specImag.count))
}

// MARK: - Lookahead Padding

/// Apply lookahead padding: trim `lookahead` frames from start, add `lookahead` zero frames at end.
///
/// This shifts features forward in time to provide the model with future context.
func applyLookaheadPad<T: Numeric>(
    _ data: [T],
    featuresPerFrame: Int,
    numFrames: Int,
    lookahead: Int
) -> [T] {
    guard lookahead > 0, numFrames > lookahead else { return data }

    // Trim `lookahead` frames from start
    let trimmedStart = lookahead * featuresPerFrame
    var result = Array(data[trimmedStart...])

    // Add `lookahead` zero frames at end
    result.append(contentsOf: [T](repeating: 0, count: lookahead * featuresPerFrame))

    return result
}
