import Accelerate
import Foundation
import MLX
import MLXNN
import MLXFast
import AudioCommon

// MARK: - Squeeze-Excitation Block

class SEBlock: Module {
    @ModuleInfo var conv1: Conv1d
    @ModuleInfo var conv2: Conv1d

    init(channels: Int, bottleneck: Int = 128) {
        self._conv1.wrappedValue = Conv1d(
            inputChannels: channels, outputChannels: bottleneck,
            kernelSize: 1, bias: true)
        self._conv2.wrappedValue = Conv1d(
            inputChannels: bottleneck, outputChannels: channels,
            kernelSize: 1, bias: true)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x: [B, T, C]
        let s = x.mean(axis: 1, keepDims: true)  // [B, 1, C]
        let h = sigmoid(conv2(relu(conv1(s))))    // [B, 1, C]
        return x * h
    }
}

// MARK: - Res2Net Block

class Res2NetBlock: Module {
    @ModuleInfo var blocks: [Conv1d]
    let scaleCount: Int

    init(channels: Int, scale: Int = 8, kernelSize: Int = 3, dilation: Int = 1) {
        let width = channels / scale
        self.scaleCount = scale
        let pad = ((kernelSize - 1) * dilation) / 2
        self._blocks.wrappedValue = (0..<(scale - 1)).map { _ in
            Conv1d(inputChannels: width, outputChannels: width,
                   kernelSize: kernelSize, padding: pad,
                   dilation: dilation, bias: true)
        }
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x: [B, T, C]
        let width = x.dim(2) / scaleCount
        var outputs: [MLXArray] = []

        for i in 0..<scaleCount {
            let chunk = x[0..., 0..., (i * width)..<((i + 1) * width)]
            if i == 0 {
                outputs.append(chunk)
            } else {
                let input = i == 1 ? chunk : chunk + outputs[i - 1]
                // Each sub-block is a TimeDelayNetBlock (Conv1d + ReLU)
                outputs.append(relu(blocks[i - 1](input)))
            }
        }

        return concatenated(outputs, axis: 2)
    }
}

// MARK: - ECAPA-TDNN SE-Res2Net Block

class ECAPABlock: Module {
    @ModuleInfo var tdnn1: Conv1d
    @ModuleInfo var res2netBlock: Res2NetBlock
    @ModuleInfo var tdnn2: Conv1d
    @ModuleInfo var seBlock: SEBlock

    init(channels: Int, kernelSize: Int = 3, dilation: Int = 1) {
        self._tdnn1.wrappedValue = Conv1d(
            inputChannels: channels, outputChannels: channels,
            kernelSize: 1, bias: true)
        self._res2netBlock.wrappedValue = Res2NetBlock(
            channels: channels, scale: 8, kernelSize: kernelSize, dilation: dilation)
        self._tdnn2.wrappedValue = Conv1d(
            inputChannels: channels, outputChannels: channels,
            kernelSize: 1, bias: true)
        self._seBlock.wrappedValue = SEBlock(channels: channels, bottleneck: 128)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let residual = x
        // tdnn1 and tdnn2 are TimeDelayNetBlocks (Conv1d + ReLU) in Python
        var h = relu(tdnn1(x))
        h = res2netBlock(h)
        h = relu(tdnn2(h))
        h = seBlock(h)
        return h + residual
    }
}

// MARK: - Attentive Statistics Pooling

/// Attentive statistics pooling matching Python's implementation.
/// Internally computes global mean/std, concatenates [x, mean, std] for attention context,
/// then uses attention weights to compute weighted mean/std statistics.
class AttentiveStatisticsPooling: Module {
    @ModuleInfo var tdnn: Conv1d   // attention: channels*3 → attention_channels
    @ModuleInfo var conv: Conv1d   // attention_channels → channels

    init(channels: Int, attention: Int = 128) {
        // tdnn takes [x, global_mean, global_std] concatenated = 3*channels
        self._tdnn.wrappedValue = Conv1d(
            inputChannels: channels * 3, outputChannels: attention,
            kernelSize: 1, bias: true)
        // conv projects back to channels (NOT channels*3)
        self._conv.wrappedValue = Conv1d(
            inputChannels: attention, outputChannels: channels,
            kernelSize: 1, bias: true)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x: [B, T, C] where C = 1536 (concatenated block outputs after MFA)

        // Compute global mean and std over time
        let globalMean = x.mean(axis: 1, keepDims: true)  // [B, 1, C]
        let globalVar = ((x - globalMean) * (x - globalMean)).mean(axis: 1, keepDims: true)
        let globalStd = sqrt(clip(globalVar, min: 1e-12))  // [B, 1, C]

        // Expand to time dimension
        let meanExpanded = broadcast(globalMean, to: x.shape)  // [B, T, C]
        let stdExpanded = broadcast(globalStd, to: x.shape)    // [B, T, C]

        // Concatenate [x, mean, std] for attention context: [B, T, 3C]
        let attnInput = concatenated([x, meanExpanded, stdExpanded], axis: 2)

        // Compute attention weights: [B, T, 3C] → [B, T, C]
        let alpha = softmax(conv(tanh(tdnn(attnInput))), axis: 1)  // softmax over T

        // Weighted statistics
        let weightedMean = (alpha * x).sum(axis: 1)  // [B, C]
        let weightedVar = (alpha * (x - weightedMean.expandedDimensions(axis: 1)) * (x - weightedMean.expandedDimensions(axis: 1))).sum(axis: 1)
        let weightedStd = sqrt(clip(weightedVar, min: 1e-12))  // [B, C]

        // Concatenate mean and std: [B, 2C]
        return concatenated([weightedMean, weightedStd], axis: 1)
    }
}

// MARK: - Multi-Feature Aggregation (TimeDelayNetBlock: Conv1d + ReLU)

class MFA: Module {
    @ModuleInfo var conv: Conv1d

    init(channels: Int) {
        self._conv.wrappedValue = Conv1d(
            inputChannels: channels, outputChannels: channels,
            kernelSize: 1, bias: true)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        relu(conv(x))
    }
}

// MARK: - ECAPA-TDNN Speaker Encoder

/// ECAPA-TDNN speaker encoder for extracting 1024-dim speaker embeddings.
/// Used for voice cloning in Qwen3-TTS Base model.
///
/// Architecture: TDNN(128→512) → 3x SE-Res2Net blocks(512) → cat(1536) → MFA(1536) → ASP → FC(3072→1024)
/// Input: 128-bin mel spectrogram (24kHz, n_fft=1024, hop=256)
/// Output: 1024-dim speaker embedding
public class SpeakerEncoder: Module {

    @ModuleInfo(key: "blocks") var initialConv: Conv1d
    @ModuleInfo var block1: ECAPABlock
    @ModuleInfo var block2: ECAPABlock
    @ModuleInfo var block3: ECAPABlock
    @ModuleInfo var mfa: MFA
    @ModuleInfo var asp: AttentiveStatisticsPooling
    @ModuleInfo var fc: Conv1d

    public override init() {
        // Initial TDNN: mel(128) → 512 channels, kernel=5
        self._initialConv.wrappedValue = Conv1d(
            inputChannels: 128, outputChannels: 512,
            kernelSize: 5, padding: 2, bias: true)

        // 3 SE-Res2Net blocks with increasing dilation
        self._block1.wrappedValue = ECAPABlock(channels: 512, kernelSize: 3, dilation: 2)
        self._block2.wrappedValue = ECAPABlock(channels: 512, kernelSize: 3, dilation: 3)
        self._block3.wrappedValue = ECAPABlock(channels: 512, kernelSize: 3, dilation: 4)

        // Multi-feature aggregation: cat of 3 block outputs (1536) → 1536
        self._mfa.wrappedValue = MFA(channels: 1536)

        // Attentive statistics pooling: 1536 channels
        self._asp.wrappedValue = AttentiveStatisticsPooling(channels: 1536, attention: 128)

        // Final projection: 3072 (1536*2 from ASP mean+std) → 1024
        self._fc.wrappedValue = Conv1d(
            inputChannels: 3072, outputChannels: 1024,
            kernelSize: 1, bias: true)

        super.init()
    }

    /// Extract speaker embedding from mel spectrogram
    /// - Parameter mels: [B, T, 128] mel spectrogram
    /// - Returns: [B, 1024] speaker embedding
    public func callAsFunction(_ mels: MLXArray) -> MLXArray {
        // Initial TDNN (Conv1d + ReLU)
        let h0 = relu(initialConv(mels))  // [B, T, 512]

        // 3 SE-Res2Net blocks
        let out1 = block1(h0)
        let out2 = block2(out1)
        let out3 = block3(out2)

        // Concatenate block outputs (skip initial conv)
        var h = concatenated([out1, out2, out3], axis: 2)  // [B, T, 1536]

        // Multi-feature aggregation (Conv1d + ReLU)
        h = mfa(h)  // [B, T, 1536]

        // Attentive statistics pooling → [B, 3072]
        h = asp(h)

        // Final projection → [B, 1024]
        h = h.expandedDimensions(axis: 1)  // [B, 1, 3072] for Conv1d
        h = fc(h)  // [B, 1, 1024]
        h = h.squeezed(axis: 1)  // [B, 1024]

        return h
    }
}

// MARK: - Mel Spectrogram for Speaker Encoder

/// Compute 128-bin mel spectrogram matching Python's mel_spectrogram() used by Qwen3-TTS speaker encoder.
/// Parameters: n_fft=1024, hop=256, win=1024, fmin=0, fmax=12000, 24kHz
public enum SpeakerMel {

    public static func compute(audio: [Float], sampleRate: Int = 24000) -> MLXArray {
        var samples = audio
        if sampleRate != 24000 {
            samples = AudioFileLoader.resample(samples, from: sampleRate, to: 24000)
        }

        let nFFT = 1024
        let hopLength = 256
        let winLength = 1024
        let nMels = 128
        let fMin: Float = 0
        let fMax: Float = 12000
        let sr = 24000

        // Build mel filterbank
        let filterbank = melFilterbank(
            nMels: nMels, nFFT: nFFT, sampleRate: sr, fMin: fMin, fMax: fMax)

        // STFT
        let window = hannWindow(size: winLength)
        let magnitudes = stftMagnitudes(
            samples: samples, nFFT: nFFT, hopLength: hopLength, window: window)

        // Apply mel filterbank: [n_frames, n_fft/2+1] x [n_fft/2+1, n_mels] → [n_frames, n_mels]
        let melSpec = matmul(magnitudes, filterbank)

        // Log mel (clamp for numerical stability)
        let logMel = log(clip(melSpec, min: 1e-5))

        // Add batch dimension: [1, T, 128]
        return logMel.expandedDimensions(axis: 0)
    }

    // MARK: - DSP Helpers

    private static func hannWindow(size: Int) -> [Float] {
        (0..<size).map { i in
            0.5 * (1.0 - cos(2.0 * Float.pi * Float(i) / Float(size)))
        }
    }

    private static func stftMagnitudes(
        samples: [Float], nFFT: Int, hopLength: Int, window: [Float]
    ) -> MLXArray {
        let numBins = nFFT / 2 + 1
        let padAmount = nFFT / 2
        // Reflect pad
        var padded = [Float](repeating: 0, count: padAmount + samples.count + padAmount)
        for i in 0..<padAmount {
            padded[padAmount - 1 - i] = samples[min(i + 1, samples.count - 1)]
        }
        for i in 0..<samples.count {
            padded[padAmount + i] = samples[i]
        }
        for i in 0..<padAmount {
            padded[padAmount + samples.count + i] = samples[max(samples.count - 2 - i, 0)]
        }

        let numFrames = (padded.count - nFFT) / hopLength + 1
        var magnitudes = [Float](repeating: 0, count: numFrames * numBins)

        let log2n = vDSP_Length(log2(Float(nFFT)))
        guard let fftSetup = vDSP_create_fftsetup(log2n, FFTRadix(kFFTRadix2)) else {
            fatalError("Failed to create FFT setup")
        }
        defer { vDSP_destroy_fftsetup(fftSetup) }

        var realPart = [Float](repeating: 0, count: nFFT / 2)
        var imagPart = [Float](repeating: 0, count: nFFT / 2)
        var windowed = [Float](repeating: 0, count: nFFT)

        for frame in 0..<numFrames {
            let start = frame * hopLength
            // Apply window
            for i in 0..<nFFT {
                windowed[i] = padded[start + i] * window[i]
            }

            // Real FFT using vDSP
            windowed.withUnsafeMutableBufferPointer { winBuf in
                realPart.withUnsafeMutableBufferPointer { realBuf in
                    imagPart.withUnsafeMutableBufferPointer { imagBuf in
                        var splitComplex = DSPSplitComplex(
                            realp: realBuf.baseAddress!, imagp: imagBuf.baseAddress!)
                        winBuf.baseAddress!.withMemoryRebound(to: DSPComplex.self, capacity: nFFT / 2) { complexBuf in
                            vDSP_ctoz(complexBuf, 2, &splitComplex, 1, vDSP_Length(nFFT / 2))
                        }
                        vDSP_fft_zrip(fftSetup, &splitComplex, 1, log2n, FFTDirection(kFFTDirection_Forward))

                        // Extract magnitudes: DC, bins 1..N/2-1, Nyquist
                        let dc = splitComplex.realp[0] / 2.0
                        let nyquist = splitComplex.imagp[0] / 2.0
                        magnitudes[frame * numBins] = abs(dc)
                        for bin in 1..<(nFFT / 2) {
                            let r = splitComplex.realp[bin] / 2.0
                            let im = splitComplex.imagp[bin] / 2.0
                            magnitudes[frame * numBins + bin] = sqrt(r * r + im * im)
                        }
                        magnitudes[frame * numBins + nFFT / 2] = abs(nyquist)
                    }
                }
            }
        }

        return MLXArray(magnitudes, [numFrames, numBins])
    }

    private static func melFilterbank(
        nMels: Int, nFFT: Int, sampleRate: Int, fMin: Float, fMax: Float
    ) -> MLXArray {
        let numBins = nFFT / 2 + 1

        func hzToMel(_ hz: Float) -> Float { 2595.0 * log10(1.0 + hz / 700.0) }
        func melToHz(_ mel: Float) -> Float { 700.0 * (pow(10.0, mel / 2595.0) - 1.0) }

        let melMin = hzToMel(fMin)
        let melMax = hzToMel(fMax)
        let melPoints = (0...(nMels + 1)).map { i in
            melToHz(melMin + Float(i) * (melMax - melMin) / Float(nMels + 1))
        }

        let fftFreqs = (0..<numBins).map { Float($0) * Float(sampleRate) / Float(nFFT) }

        var filterbank = [Float](repeating: 0, count: numBins * nMels)
        for m in 0..<nMels {
            let fLow = melPoints[m]
            let fCenter = melPoints[m + 1]
            let fHigh = melPoints[m + 2]

            for k in 0..<numBins {
                let freq = fftFreqs[k]
                if freq >= fLow && freq <= fCenter && fCenter > fLow {
                    filterbank[k * nMels + m] = (freq - fLow) / (fCenter - fLow)
                } else if freq > fCenter && freq <= fHigh && fHigh > fCenter {
                    filterbank[k * nMels + m] = (fHigh - freq) / (fHigh - fCenter)
                }
            }
        }

        return MLXArray(filterbank, [numBins, nMels])
    }
}
