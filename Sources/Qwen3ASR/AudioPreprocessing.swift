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

    public init() {
        setupMelFilterbank()
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

        let nBins = nFFT / 2 + 1  // 201 for nFFT=400

        // Create linearly spaced FFT bin frequencies (0 to Nyquist)
        // This matches: np.linspace(0, sampling_rate // 2, num_frequency_bins)
        var fftFreqs = [Float](repeating: 0, count: nBins)
        for i in 0..<nBins {
            fftFreqs[i] = Float(i) * fMax / Float(nBins - 1)
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
        let nBins = nFFT / 2 + 1

        // Pad audio with reflect padding (like Whisper/librosa)
        let padLength = nFFT / 2
        var paddedAudio = [Float](repeating: 0, count: padLength + audio.count + padLength)

        // Reflect pad left side: audio[padLength], audio[padLength-1], ..., audio[1]
        // torch.nn.functional.pad reflect mode mirrors around the edge element
        for i in 0..<padLength {
            let srcIdx = min(padLength - i, audio.count - 1)
            paddedAudio[i] = audio[max(0, srcIdx)]
        }

        // Copy original audio
        for i in 0..<audio.count {
            paddedAudio[padLength + i] = audio[i]
        }

        // Reflect pad right side: audio[n-2], audio[n-3], ..., audio[n-padLength-1]
        for i in 0..<padLength {
            let srcIdx = audio.count - 2 - i
            paddedAudio[padLength + audio.count + i] = audio[max(0, srcIdx)]
        }

        // Calculate number of frames
        let nFrames = (paddedAudio.count - nFFT) / hopLength + 1

        // Create PERIODIC Hann window (like PyTorch/librosa, not symmetric)
        // Formula: 0.5 * (1 - cos(2πn/N)) for n=0 to N-1
        var window = [Float](repeating: 0, count: nFFT)
        for i in 0..<nFFT {
            window[i] = 0.5 * (1.0 - cos(2.0 * Float.pi * Float(i) / Float(nFFT)))
        }

        // Precompute twiddle factors for DFT (only need first nBins frequencies for real input)
        var twiddleReal = [Float](repeating: 0, count: nBins * nFFT)
        var twiddleImag = [Float](repeating: 0, count: nBins * nFFT)

        for k in 0..<nBins {
            for n in 0..<nFFT {
                let angle = -2.0 * Float.pi * Float(k) * Float(n) / Float(nFFT)
                twiddleReal[k * nFFT + n] = cos(angle)
                twiddleImag[k * nFFT + n] = sin(angle)
            }
        }

        // Compute STFT
        var magnitude = [Float](repeating: 0, count: nFrames * nBins)

        // Temporary buffer for windowed frame
        var windowedFrame = [Float](repeating: 0, count: nFFT)

        for frame in 0..<nFrames {
            let start = frame * hopLength

            // Extract and window the frame
            for i in 0..<nFFT {
                if start + i < paddedAudio.count {
                    windowedFrame[i] = paddedAudio[start + i] * window[i]
                } else {
                    windowedFrame[i] = 0
                }
            }

            // Compute DFT for each frequency bin (only positive frequencies up to Nyquist)
            for k in 0..<nBins {
                var sumReal: Float = 0
                var sumImag: Float = 0

                // Use vDSP for vectorized dot product
                let twiddleRealStart = k * nFFT
                vDSP_dotpr(windowedFrame, 1, Array(twiddleReal[twiddleRealStart..<twiddleRealStart + nFFT]), 1, &sumReal, vDSP_Length(nFFT))
                vDSP_dotpr(windowedFrame, 1, Array(twiddleImag[twiddleRealStart..<twiddleRealStart + nFFT]), 1, &sumImag, vDSP_Length(nFFT))

                // Power spectrum: |X[k]|^2 = real^2 + imag^2
                magnitude[frame * nBins + k] = sumReal * sumReal + sumImag * sumImag
            }
        }

        // Apply mel filterbank
        guard let filterbank = melFilterbank else {
            fatalError("Mel filterbank not initialized")
        }

        // Debug: check magnitude (power spectrum) stats
        var magMean: Float = 0
        var magMax: Float = -Float.infinity
        vDSP_meanv(magnitude, 1, &magMean, vDSP_Length(magnitude.count))
        vDSP_maxv(magnitude, 1, &magMax, vDSP_Length(magnitude.count))
        print("DEBUG FeatureExtractor: Power spectrum - mean: \(magMean), max: \(magMax)")

        // Debug: check filterbank stats
        var fbMax: Float = -Float.infinity
        vDSP_maxv(filterbank, 1, &fbMax, vDSP_Length(filterbank.count))
        print("DEBUG FeatureExtractor: Filterbank max: \(fbMax)")

        var melSpec = [Float](repeating: 0, count: nFrames * nMels)

        // Matrix multiply: melSpec = magnitude * filterbank^T
        for frame in 0..<nFrames {
            for mel in 0..<nMels {
                var sum: Float = 0
                for bin in 0..<nBins {
                    sum += magnitude[frame * nBins + bin] * filterbank[mel * nBins + bin]
                }
                melSpec[frame * nMels + mel] = sum
            }
        }

        // Debug: check mel spec before log
        var melMean: Float = 0
        var melMax: Float = -Float.infinity
        vDSP_meanv(melSpec, 1, &melMean, vDSP_Length(melSpec.count))
        vDSP_maxv(melSpec, 1, &melMax, vDSP_Length(melSpec.count))
        print("DEBUG FeatureExtractor: Mel spec (before log) - mean: \(melMean), max: \(melMax)")

        // Apply log10-mel transformation with small epsilon (Whisper-style)
        let epsilon: Float = 1e-10
        for i in 0..<melSpec.count {
            melSpec[i] = log10(max(melSpec[i], epsilon))
        }

        // Debug: check log10 mel values before normalization
        var logMax: Float = -Float.infinity
        var logMin: Float = Float.infinity
        vDSP_maxv(melSpec, 1, &logMax, vDSP_Length(melSpec.count))
        vDSP_minv(melSpec, 1, &logMin, vDSP_Length(melSpec.count))
        print("DEBUG FeatureExtractor: log10 mel - min: \(logMin), max: \(logMax)")

        // Whisper-style normalization:
        // 1. Clamp to max - 8.0 (dynamic range compression)
        // 2. Normalize: (log_spec + 4.0) / 4.0
        var maxVal: Float = -Float.infinity
        vDSP_maxv(melSpec, 1, &maxVal, vDSP_Length(melSpec.count))

        // Clamp minimum to max - 8.0
        let minClamp = maxVal - 8.0
        for i in 0..<melSpec.count {
            melSpec[i] = max(melSpec[i], minClamp)
        }

        // Debug: check after clipping
        var clippedMin: Float = Float.infinity
        vDSP_minv(melSpec, 1, &clippedMin, vDSP_Length(melSpec.count))
        print("DEBUG FeatureExtractor: After clipping - min: \(clippedMin), max: \(maxVal)")

        // Normalize: (x + 4.0) / 4.0
        for i in 0..<melSpec.count {
            melSpec[i] = (melSpec[i] + 4.0) / 4.0
        }

        // Debug: check final values
        var finalMax: Float = -Float.infinity
        var finalMin: Float = Float.infinity
        vDSP_maxv(melSpec, 1, &finalMax, vDSP_Length(melSpec.count))
        vDSP_minv(melSpec, 1, &finalMin, vDSP_Length(melSpec.count))
        print("DEBUG FeatureExtractor: Final normalized - min: \(finalMin), max: \(finalMax)")

        // CRITICAL: HuggingFace WhisperFeatureExtractor removes the last frame: log_spec[:, :-1]
        // This is needed to match the exact frame count that the model expects
        var trimmedFrames = nFrames - 1  // Remove last frame
        var trimmedMelSpec = Array(melSpec.prefix(trimmedFrames * nMels))
        print("DEBUG FeatureExtractor: Trimmed last frame: \(nFrames) -> \(trimmedFrames)")

        // DON'T pad to 3000 frames - let the audio encoder handle the actual length
        // The Python reference only pads minimally for chunk alignment, not to a fixed length
        let maxFrames = chunkLength * sampleRate / hopLength  // 30 * 16000 / 160 = 3000
        var finalMelSpec = trimmedMelSpec

        if trimmedFrames > maxFrames {
            // Truncate to 3000 frames if longer than 30 seconds
            finalMelSpec = Array(trimmedMelSpec.prefix(maxFrames * nMels))
            print("DEBUG FeatureExtractor: Truncated from \(trimmedFrames) to \(maxFrames) frames")
        } else {
            print("DEBUG FeatureExtractor: Using actual \(trimmedFrames) frames (no padding)")
        }

        let finalFrames = finalMelSpec.count / nMels

        // Reshape to [mel_bins, time_frames] and convert to MLXArray
        // First create [time_frames, mel_bins] then transpose
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
