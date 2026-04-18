import Accelerate
import Foundation

/// Kaldi-compatible online fbank feature extractor.
///
/// Matches the defaults used by ``kaldi-native-fbank`` for sherpa-onnx KWS
/// (see ``models/kws-zipformer/export/streaming_fbank.py``):
///
/// - 25 ms / 10 ms frames (400 / 160 samples @ 16 kHz)
/// - Povey window (``pow(0.5 - 0.5*cos(2π n/(N-1)), 0.85)``)
/// - per-frame DC removal + preemphasis ``y[n] = x[n] - 0.97·x[n-1]``
/// - FFT size 512 (next power of two ≥ win length)
/// - 80 mel bins, triangular filters in ``mel = 1127·log(1 + hz/700)`` space
/// - low_freq = 20 Hz, high_freq = Nyquist + config.highFreq (default −400 Hz)
/// - power spectrum (``|FFT|^2``), natural log, energy floor ``FLT_EPSILON``
/// - no CMVN (icefall KWS recipe does not apply it)
/// - ``snip_edges = false``: the first frame is centred at ``shift/2``, earlier
///   samples mirror-padded, matching ``lhotse`` / kaldi-native-fbank
///
/// Byte-exact parity with kaldi is verified by the ``fbank_reference.bin``
/// resource generated from ``streaming_fbank.waveform_to_fbank`` — see
/// ``Tests/SpeechWakeWordTests``.
public final class KaldiFbank {
    public struct Options: Sendable {
        public let sampleRate: Int
        public let frameLength: Int   // samples
        public let frameShift: Int    // samples
        public let numMelBins: Int
        public let lowFreq: Double
        public let highFreq: Double   // negative = nyquist + value (kaldi convention)
        public let preemphCoeff: Double
        public let removeDCOffset: Bool
        public let snipEdges: Bool
        public let usePower: Bool

        public init(
            sampleRate: Int = 16000,
            frameLengthMs: Double = 25.0,
            frameShiftMs: Double = 10.0,
            numMelBins: Int = 80,
            lowFreq: Double = 20.0,
            highFreq: Double = -400.0,
            preemphCoeff: Double = 0.97,
            removeDCOffset: Bool = true,
            snipEdges: Bool = false,
            usePower: Bool = true
        ) {
            self.sampleRate = sampleRate
            self.frameLength = Int((frameLengthMs * 1e-3 * Double(sampleRate)).rounded())
            self.frameShift = Int((frameShiftMs * 1e-3 * Double(sampleRate)).rounded())
            self.numMelBins = numMelBins
            self.lowFreq = lowFreq
            self.highFreq = highFreq
            self.preemphCoeff = preemphCoeff
            self.removeDCOffset = removeDCOffset
            self.snipEdges = snipEdges
            self.usePower = usePower
        }
    }

    public let options: Options
    private let paddedSize: Int            // next power of two ≥ frameLength
    private let log2Padded: vDSP_Length
    private let numBins: Int               // paddedSize/2 + 1
    private let fftSetup: FFTSetup
    private let poveyWindow: [Float]
    private let melFilterbank: [Float]     // [numMelBins, numBins] row-major
    private let logFloor: Float = .leastNormalMagnitude  // matches kaldi FLT_EPSILON floor

    public init(_ options: Options = Options()) {
        self.options = options

        var size = 1
        while size < options.frameLength { size *= 2 }
        self.paddedSize = size
        self.log2Padded = vDSP_Length(log2(Float(size)))
        self.numBins = size / 2 + 1

        guard let setup = vDSP_create_fftsetup(log2Padded, FFTRadix(kFFTRadix2)) else {
            fatalError("Failed to create vDSP FFT setup (paddedSize=\(size))")
        }
        self.fftSetup = setup

        // Povey window
        let n = options.frameLength
        var w = [Float](repeating: 0, count: n)
        let denom = Double(n - 1)
        for i in 0..<n {
            let raw = 0.5 - 0.5 * cos(2.0 * .pi * Double(i) / denom)
            w[i] = Float(pow(raw, 0.85))
        }
        self.poveyWindow = w

        self.melFilterbank = Self.buildMelFilterbank(
            numMelBins: options.numMelBins,
            numBins: numBins,
            sampleRate: options.sampleRate,
            paddedSize: size,
            lowFreq: options.lowFreq,
            highFreq: options.highFreq
        )
    }

    deinit { vDSP_destroy_fftsetup(fftSetup) }

    /// Number of full frames available for ``numSamples`` at the current options.
    public func numFrames(for numSamples: Int) -> Int {
        if numSamples == 0 { return 0 }
        if options.snipEdges {
            if numSamples < options.frameLength { return 0 }
            return (numSamples - options.frameLength) / options.frameShift + 1
        }
        // kaldi snip_edges=false: N = floor((numSamples + shift/2) / shift)
        return Int((Double(numSamples) + Double(options.frameShift) / 2.0) / Double(options.frameShift))
    }

    /// Extract fbank over a whole utterance.
    /// - Parameter samples: Float32 PCM in [-1, 1] at ``options.sampleRate``.
    /// - Returns: ``[numFrames * numMelBins]`` row-major mel frames (log energies).
    public func compute(_ samples: [Float]) -> [Float] {
        let frames = numFrames(for: samples.count)
        guard frames > 0 else { return [] }

        var output = [Float](repeating: 0, count: frames * options.numMelBins)
        var frame = [Float](repeating: 0, count: paddedSize)
        var splitReal = [Float](repeating: 0, count: paddedSize / 2)
        var splitImag = [Float](repeating: 0, count: paddedSize / 2)
        var power = [Float](repeating: 0, count: numBins)
        var mels = [Float](repeating: 0, count: options.numMelBins)

        for f in 0..<frames {
            // 1. extract raw window with snip_edges handling
            extractWindow(samples: samples, frameIndex: f, into: &frame)

            // 2. DC-remove + preemph + window on the first frameLength samples
            preprocessWindow(&frame)

            // 3. zero remainder to paddedSize
            for i in options.frameLength..<paddedSize { frame[i] = 0 }

            // 4. FFT → power spectrum
            fftPower(frame: &frame, real: &splitReal, imag: &splitImag, power: &power)

            // 5. mel filterbank
            applyMelFilterbank(power: power, into: &mels)

            // 6. log floor + natural log
            for m in 0..<options.numMelBins {
                output[f * options.numMelBins + m] = log(max(mels[m], logFloor))
            }
        }
        return output
    }

    // MARK: - window extraction

    private func extractWindow(samples: [Float], frameIndex: Int, into frame: inout [Float]) {
        let shift = options.frameShift
        let length = options.frameLength
        let n = samples.count

        let start: Int
        if options.snipEdges {
            start = frameIndex * shift
        } else {
            // kaldi snip_edges=false: first frame centred at 0 in "virtual" coords.
            // ``FirstSampleOfFrame = f·shift - (length-shift)/2``
            start = frameIndex * shift - (length - shift) / 2
        }

        for i in 0..<length {
            var idx = start + i
            if idx < 0 || idx >= n {
                // kaldi mirrors samples: `Mirror(idx, num_samples)`
                idx = mirror(idx, total: n)
            }
            frame[i] = samples[idx]
        }
    }

    @inline(__always)
    private func mirror(_ index: Int, total: Int) -> Int {
        if total == 0 { return 0 }
        var i = index
        // Clamp/reflect: kaldi's implementation keeps bouncing until in range.
        while i < 0 || i >= total {
            if i < 0 { i = -i - 1 }
            if i >= total { i = 2 * total - i - 1 }
        }
        return i
    }

    // MARK: - per-frame preprocessing

    private func preprocessWindow(_ frame: inout [Float]) {
        let n = options.frameLength

        if options.removeDCOffset {
            var mean: Float = 0
            vDSP_meanv(frame, 1, &mean, vDSP_Length(n))
            var negMean = -mean
            vDSP_vsadd(frame, 1, &negMean, &frame, 1, vDSP_Length(n))
        }

        // Preemphasis (kaldi: iterate from end to start so each y[i] uses the
        // original x[i-1], and y[0] = x[0]·(1 - preemph)).
        let preemph = Float(options.preemphCoeff)
        if preemph != 0 {
            for i in stride(from: n - 1, through: 1, by: -1) {
                frame[i] -= preemph * frame[i - 1]
            }
            frame[0] -= preemph * frame[0]
        }

        vDSP_vmul(frame, 1, poveyWindow, 1, &frame, 1, vDSP_Length(n))
    }

    // MARK: - FFT → power

    private func fftPower(
        frame: inout [Float],
        real: inout [Float],
        imag: inout [Float],
        power: inout [Float]
    ) {
        let half = paddedSize / 2
        for i in 0..<half {
            real[i] = frame[2 * i]
            imag[i] = frame[2 * i + 1]
        }
        real.withUnsafeMutableBufferPointer { r in
            imag.withUnsafeMutableBufferPointer { im in
                var split = DSPSplitComplex(realp: r.baseAddress!, imagp: im.baseAddress!)
                vDSP_fft_zrip(fftSetup, &split, 1, log2Padded,
                              FFTDirection(kFFTDirection_Forward))
            }
        }
        // vDSP packs Nyquist bin in imag[0]. Unpack + scale (vDSP returns 2× DFT).
        let scale: Float = 0.5
        let dcReal = real[0] * scale
        let nyquistReal = imag[0] * scale
        power[0] = options.usePower ? dcReal * dcReal : abs(dcReal)
        for k in 1..<half {
            let re = real[k] * scale
            let im = imag[k] * scale
            let p = re * re + im * im
            power[k] = options.usePower ? p : sqrt(p)
        }
        let nyq = nyquistReal * nyquistReal
        power[half] = options.usePower ? nyq : sqrt(nyq)
    }

    // MARK: - mel filterbank

    private func applyMelFilterbank(power: [Float], into mels: inout [Float]) {
        vDSP_mmul(
            melFilterbank, 1,
            power, 1,
            &mels, 1,
            vDSP_Length(options.numMelBins),
            1,
            vDSP_Length(numBins)
        )
    }

    private static func buildMelFilterbank(
        numMelBins: Int,
        numBins: Int,
        sampleRate: Int,
        paddedSize: Int,
        lowFreq: Double,
        highFreq: Double
    ) -> [Float] {
        let nyquist = Double(sampleRate) / 2.0
        let effectiveHigh = highFreq < 0 ? nyquist + highFreq : highFreq
        precondition(effectiveHigh > lowFreq, "Invalid fbank freq range")

        func hzToMel(_ hz: Double) -> Double { 1127.0 * log1p(hz / 700.0) }
        func melToHz(_ mel: Double) -> Double { 700.0 * (exp(mel / 1127.0) - 1.0) }

        let melLow = hzToMel(lowFreq)
        let melHigh = hzToMel(effectiveHigh)
        let melDelta = (melHigh - melLow) / Double(numMelBins + 1)

        // FFT bin centre frequencies.
        var fftFreqs = [Double](repeating: 0, count: numBins)
        for k in 0..<numBins {
            fftFreqs[k] = Double(k) * Double(sampleRate) / Double(paddedSize)
        }

        // filterbank[m, k]
        var filters = [Float](repeating: 0, count: numMelBins * numBins)
        for m in 0..<numMelBins {
            let leftMel = melLow + Double(m) * melDelta
            let centreMel = leftMel + melDelta
            let rightMel = leftMel + 2 * melDelta
            for k in 0..<numBins {
                let mel = hzToMel(fftFreqs[k])
                var weight = 0.0
                if mel > leftMel && mel < rightMel {
                    if mel <= centreMel {
                        weight = (mel - leftMel) / (centreMel - leftMel)
                    } else {
                        weight = (rightMel - mel) / (rightMel - centreMel)
                    }
                }
                filters[m * numBins + k] = Float(weight)
            }
            // Kaldi does not apply Slaney normalization — keep raw triangles.
            _ = melToHz(centreMel)
        }
        return filters
    }
}
