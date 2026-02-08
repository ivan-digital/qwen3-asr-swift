import Foundation
import Accelerate
import MLX

/// Whisper-style feature extractor for Qwen3-ASR
/// Converts raw audio to mel spectrograms
/// Parameters from HuggingFace preprocessor_config.json
public class WhisperFeatureExtractor {
    public let sampleRate: Int = 16000      // HF WhisperFeatureExtractor uses 16kHz
    public let nFFT: Int = 400              // FFT size (from config)
    public let hopLength: Int = 160         // 10ms hop at 16kHz (from config)
    public let nMels: Int = 128             // Mel filterbank bins (feature_size)
    public let chunkLength: Int = 30        // Max audio chunk in seconds

    private var melFilterbank: [Float]?

    // Cached Hann window and FFT setup for Accelerate
    private var hannWindow: [Float]
    // Power-of-2 FFT: zero-pad nFFT=400 to paddedFFT=512 for vDSP compatibility
    private let paddedFFT: Int = 512
    private let log2PaddedFFT: vDSP_Length = 9  // log2(512) = 9
    private var fftSetup: FFTSetup

    public init() {
        // Precompute periodic Hann window
        hannWindow = [Float](repeating: 0, count: 400) // nFFT
        for i in 0..<400 {
            hannWindow[i] = 0.5 * (1.0 - cos(2.0 * Float.pi * Float(i) / Float(400)))
        }

        // Create power-of-2 FFT setup (512-point)
        guard let setup = vDSP_create_fftsetup(9, FFTRadix(kFFTRadix2)) else {
            fatalError("Failed to create vDSP FFT setup for paddedFFT=512")
        }
        fftSetup = setup

        setupMelFilterbank()
    }

    deinit {
        vDSP_destroy_fftsetup(fftSetup)
    }

    /// Setup mel filterbank matrix with slaney normalization
    /// Matches HuggingFace transformers.audio_utils.mel_filter_bank exactly
    private func setupMelFilterbank() {
        let fMin: Float = 0.0
        let fMax: Float = Float(sampleRate) / 2.0  // Nyquist frequency (8000 Hz for 16kHz)

        // Slaney mel scale conversion functions (HuggingFace style)
        // This is a piecewise function: linear below 1000 Hz, logarithmic above
        let minLogHertz: Float = 1000.0
        let minLogMel: Float = 15.0
        let logstepHzToMel: Float = 27.0 / log(6.4)  // For Hz->Mel
        let logstepMelToHz: Float = log(6.4) / 27.0  // For Mel->Hz

        func hzToMel(_ hz: Float) -> Float {
            if hz < minLogHertz {
                return 3.0 * hz / 200.0  // Linear region
            } else {
                return minLogMel + log(hz / minLogHertz) * logstepHzToMel  // Log region
            }
        }

        func melToHz(_ mel: Float) -> Float {
            if mel < minLogMel {
                return 200.0 * mel / 3.0  // Linear region
            } else {
                return minLogHertz * exp((mel - minLogMel) * logstepMelToHz)  // Exp region
            }
        }

        // Use paddedFFT for bin count since we zero-pad to 512 for FFT
        let nBins = paddedFFT / 2 + 1  // 257 for paddedFFT=512

        // FFT bin frequencies: k * fs / paddedFFT (not nFFT, since we zero-pad)
        var fftFreqs = [Float](repeating: 0, count: nBins)
        for i in 0..<nBins {
            fftFreqs[i] = Float(i) * Float(sampleRate) / Float(paddedFFT)
        }

        // Create mel filter center frequencies
        let melMin = hzToMel(fMin)
        let melMax = hzToMel(fMax)

        // nMels + 2 points for triangular filters (includes low and high edges)
        let nMelPoints = nMels + 2
        var melPoints = [Float](repeating: 0, count: nMelPoints)
        for i in 0..<nMelPoints {
            melPoints[i] = melMin + Float(i) * (melMax - melMin) / Float(nMelPoints - 1)
        }

        // Convert mel points to Hz - these are the filter edge frequencies
        let filterFreqs = melPoints.map { melToHz($0) }

        // Calculate filter frequency differences for normalization
        var filterDiff = [Float](repeating: 0, count: nMelPoints - 1)
        for i in 0..<(nMelPoints - 1) {
            filterDiff[i] = filterFreqs[i + 1] - filterFreqs[i]
        }

        // Create filterbank using HuggingFace's _create_triangular_filter_bank approach
        // This creates smooth triangular filters in frequency space
        // Output shape: [nBins, nMels] - we'll transpose later for our use
        var filterbank = [Float](repeating: 0, count: nBins * nMels)

        for bin in 0..<nBins {
            let fftFreq = fftFreqs[bin]

            for mel in 0..<nMels {
                // Filter edges: filterFreqs[mel], filterFreqs[mel+1], filterFreqs[mel+2]
                let lowFreq = filterFreqs[mel]
                let centerFreq = filterFreqs[mel + 1]
                let highFreq = filterFreqs[mel + 2]

                // Calculate slopes (HuggingFace formula)
                // slopes = filter_freqs - fft_freqs (broadcast)
                // down_slopes = -slopes[:, :-2] / filter_diff[:-1]
                // up_slopes = slopes[:, 2:] / filter_diff[1:]

                let downSlope = (fftFreq - lowFreq) / filterDiff[mel]     // Rising edge
                let upSlope = (highFreq - fftFreq) / filterDiff[mel + 1]  // Falling edge

                // Triangular filter: max(0, min(down_slope, up_slope))
                let filterValue = max(0.0, min(downSlope, upSlope))

                // Store in [nBins, nMels] layout
                filterbank[bin * nMels + mel] = filterValue
            }
        }

        // Apply slaney normalization: 2.0 / (high_freq - low_freq) for each mel filter
        for mel in 0..<nMels {
            let enorm = 2.0 / (filterFreqs[mel + 2] - filterFreqs[mel])
            for bin in 0..<nBins {
                filterbank[bin * nMels + mel] *= enorm
            }
        }

        // Transpose to [nMels, nBins] for our matrix multiplication
        var filterbankTransposed = [Float](repeating: 0, count: nMels * nBins)
        for mel in 0..<nMels {
            for bin in 0..<nBins {
                filterbankTransposed[mel * nBins + bin] = filterbank[bin * nMels + mel]
            }
        }

        self.melFilterbank = filterbankTransposed
    }

    /// Extract mel spectrogram features from audio samples
    /// - Parameter audio: Raw audio samples (Float array, mono, at sampleRate)
    /// - Returns: Mel spectrogram [mel_bins, time_frames]
    public func extractFeatures(_ audio: [Float]) -> MLXArray {
        let nBins = paddedFFT / 2 + 1  // 257 bins for 512-point FFT
        let halfPadded = paddedFFT / 2  // 256

        // Pad audio with reflect padding (like Whisper/librosa)
        let padLength = nFFT / 2
        var paddedAudio = [Float](repeating: 0, count: padLength + audio.count + padLength)

        // Reflect pad left side
        for i in 0..<padLength {
            let srcIdx = min(padLength - i, audio.count - 1)
            paddedAudio[i] = audio[max(0, srcIdx)]
        }

        // Copy original audio
        for i in 0..<audio.count {
            paddedAudio[padLength + i] = audio[i]
        }

        // Reflect pad right side
        for i in 0..<padLength {
            let srcIdx = audio.count - 2 - i
            paddedAudio[padLength + audio.count + i] = audio[max(0, srcIdx)]
        }

        // Calculate number of frames
        let nFrames = (paddedAudio.count - nFFT) / hopLength + 1

        // --- Accelerate FFT-based STFT using vDSP_fft_zrip ---
        // Zero-pad nFFT=400 samples to paddedFFT=512 for power-of-2 FFT
        // vDSP_fft_zrip uses split-complex in-place: even-indexed in realp, odd-indexed in imagp
        // Output packing: DC in realp[0], Nyquist in imagp[0], bins 1..N/2-1 in realp[k]+j*imagp[k]

        // Preallocate buffers outside the frame loop
        var splitReal = [Float](repeating: 0, count: halfPadded)
        var splitImag = [Float](repeating: 0, count: halfPadded)
        var paddedFrame = [Float](repeating: 0, count: paddedFFT)

        var magnitude = [Float](repeating: 0, count: nFrames * nBins)

        for frame in 0..<nFrames {
            let start = frame * hopLength

            // Apply window and write into paddedFrame (first nFFT elements)
            paddedAudio.withUnsafeBufferPointer { buf in
                vDSP_vmul(buf.baseAddress! + start, 1, hannWindow, 1, &paddedFrame, 1, vDSP_Length(nFFT))
            }
            // Zero-pad the rest (nFFT..<paddedFFT)
            for i in nFFT..<paddedFFT {
                paddedFrame[i] = 0
            }

            // Pack into split-complex: realp[i] = frame[2*i], imagp[i] = frame[2*i+1]
            for i in 0..<halfPadded {
                splitReal[i] = paddedFrame[2 * i]
                splitImag[i] = paddedFrame[2 * i + 1]
            }

            // Execute in-place real FFT
            splitReal.withUnsafeMutableBufferPointer { realBuf in
                splitImag.withUnsafeMutableBufferPointer { imagBuf in
                    var splitComplex = DSPSplitComplex(
                        realp: realBuf.baseAddress!,
                        imagp: imagBuf.baseAddress!)
                    vDSP_fft_zrip(fftSetup, &splitComplex, 1, log2PaddedFFT, FFTDirection(kFFTDirection_Forward))
                }
            }

            // Extract power spectrum
            let baseIdx = frame * nBins

            // DC component: realp[0]^2 (DC is purely real, stored in realp[0])
            magnitude[baseIdx] = splitReal[0] * splitReal[0]

            // Nyquist component: imagp[0]^2 (Nyquist is purely real, packed in imagp[0])
            magnitude[baseIdx + halfPadded] = splitImag[0] * splitImag[0]

            // Bins 1 to N/2-1: realp[k]^2 + imagp[k]^2
            for k in 1..<halfPadded {
                magnitude[baseIdx + k] = splitReal[k] * splitReal[k] + splitImag[k] * splitImag[k]
            }
        }

        // --- BLAS sgemm for mel filterbank ---
        guard let filterbank = melFilterbank else {
            fatalError("Mel filterbank not initialized")
        }

        var melSpec = [Float](repeating: 0, count: nFrames * nMels)

        // melSpec[nFrames, nMels] = magnitude[nFrames, nBins] * filterbankT[nBins, nMels]
        // filterbank is [nMels, nBins]. We need A * B^T.
        // vDSP_mmul computes C = A * B, so we need to pre-transpose filterbank.
        // Transpose filterbank [nMels, nBins] -> filterbankT [nBins, nMels]
        var filterbankT = [Float](repeating: 0, count: nBins * nMels)
        vDSP_mtrans(filterbank, 1, &filterbankT, 1, vDSP_Length(nBins), vDSP_Length(nMels))

        // C[nFrames, nMels] = A[nFrames, nBins] * B[nBins, nMels]
        vDSP_mmul(magnitude, 1, filterbankT, 1, &melSpec, 1,
                  vDSP_Length(nFrames), vDSP_Length(nMels), vDSP_Length(nBins))

        // --- Vectorized log10, clamp, normalize ---
        let count = melSpec.count
        var countN = Int32(count)

        // Clamp minimum to epsilon before log
        var epsilon: Float = 1e-10
        vDSP_vclip(melSpec, 1, &epsilon, [Float.greatestFiniteMagnitude], &melSpec, 1, vDSP_Length(count))

        // log10 using vForce
        vvlog10f(&melSpec, melSpec, &countN)

        // Find max value for dynamic range compression
        var maxVal: Float = -Float.infinity
        vDSP_maxv(melSpec, 1, &maxVal, vDSP_Length(count))

        // Clamp minimum to max - 8.0
        var minClamp = maxVal - 8.0
        var maxClamp = Float.greatestFiniteMagnitude
        vDSP_vclip(melSpec, 1, &minClamp, &maxClamp, &melSpec, 1, vDSP_Length(count))

        // Normalize: (x + 4.0) / 4.0 = x * 0.25 + 1.0
        var scale: Float = 0.25
        var offset: Float = 1.0
        vDSP_vsmsa(melSpec, 1, &scale, &offset, &melSpec, 1, vDSP_Length(count))

        // CRITICAL: HuggingFace WhisperFeatureExtractor removes the last frame: log_spec[:, :-1]
        let trimmedFrames = nFrames - 1
        let trimmedMelSpec = Array(melSpec.prefix(trimmedFrames * nMels))

        // DON'T pad to 3000 frames - let the audio encoder handle the actual length
        let maxFrames = chunkLength * sampleRate / hopLength  // 30 * 16000 / 160 = 3000
        var finalMelSpec = trimmedMelSpec

        if trimmedFrames > maxFrames {
            finalMelSpec = Array(trimmedMelSpec.prefix(maxFrames * nMels))
        }

        let finalFrames = finalMelSpec.count / nMels

        let array = MLXArray(finalMelSpec, [finalFrames, nMels])
        return array.transposed(1, 0)  // [mel_bins, time_frames]
    }

    /// Process audio for Qwen3-ASR model
    /// - Parameter audio: Raw audio samples (any sample rate)
    /// - Parameter inputSampleRate: Sample rate of input audio
    /// - Returns: Preprocessed mel features ready for the model
    public func process(_ audio: [Float], sampleRate inputSampleRate: Int) -> MLXArray {
        var processedAudio = audio

        // Resample if needed
        if inputSampleRate != sampleRate {
            processedAudio = resample(audio, from: inputSampleRate, to: sampleRate)
        }

        // NOTE: HuggingFace WhisperFeatureExtractor does NOT normalize audio amplitude
        // The model expects raw audio values (typically in [-1, 1] range from int16 conversion)
        // Do NOT divide by max absolute value!

        // Extract features
        return extractFeatures(processedAudio)
    }

    /// Simple linear resampling
    private func resample(_ audio: [Float], from inputRate: Int, to outputRate: Int) -> [Float] {
        let ratio = Double(outputRate) / Double(inputRate)
        let outputLength = Int(Double(audio.count) * ratio)

        guard outputLength > 0 else { return [] }

        var output = [Float](repeating: 0, count: outputLength)

        for i in 0..<outputLength {
            let srcIndex = Double(i) / ratio
            let srcIndexFloor = Int(srcIndex)
            let srcIndexCeil = min(srcIndexFloor + 1, audio.count - 1)
            let fraction = Float(srcIndex - Double(srcIndexFloor))

            output[i] = audio[srcIndexFloor] * (1 - fraction) + audio[srcIndexCeil] * fraction
        }

        return output
    }
}
