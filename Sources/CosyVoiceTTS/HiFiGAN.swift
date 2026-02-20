import Foundation
import MLX
import MLXNN

// MARK: - Snake Activation

/// Snake activation: x + (1/alpha) * sin^2(alpha * x)
/// CosyVoice3 uses alpha_logscale=False: alpha is stored as raw values (NOT log-space).
/// Initialized to 1.0 in Python, then trained. No exp() applied.
public class SnakeActivation: Module {
    @ParameterInfo var alpha: MLXArray  // [channels]

    public init(channels: Int) {
        // alpha_logscale=False: initialize to 1.0 (raw, not log-space)
        self._alpha.wrappedValue = MLXArray.ones([channels])
        super.init()
    }

    /// Input/output: [B, C, T] (NCL / channels-first)
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Reshape alpha [C] -> [1, C, 1] for broadcasting with NCL input
        let a = alpha.reshaped([1, -1, 1])  // [1, C, 1], raw (no exp)
        let sinTerm = sin(a * x)
        return x + (1.0 / (a + 1e-9)) * (sinTerm * sinTerm)
    }
}

// MARK: - Causal Dilated Conv1d (NCL)

/// Conv1d wrapper operating in NCL (channels-first) format with causal padding.
/// Internally transposes to NLC for MLX's Conv1d, then back to NCL.
///
/// - `causalType = .left` (default): left-pad with zeros → standard causal (output at t depends on past)
/// - `causalType = .right`: right-pad with zeros → look-ahead (output at t depends on future)
public class CausalDilatedConv1d: Module {
    public enum CausalType { case left, right }

    @ModuleInfo var conv: Conv1d
    let padAmount: Int
    let causalType: CausalType

    public init(
        inputChannels: Int,
        outputChannels: Int,
        kernelSize: Int,
        stride: Int = 1,
        dilation: Int = 1,
        groups: Int = 1,
        bias: Bool = true,
        causalType: CausalType = .left
    ) {
        self.padAmount = (kernelSize - 1) * dilation
        self.causalType = causalType
        self._conv.wrappedValue = Conv1d(
            inputChannels: inputChannels,
            outputChannels: outputChannels,
            kernelSize: kernelSize,
            stride: stride,
            padding: 0,
            dilation: dilation,
            groups: groups,
            bias: bias)
        super.init()
    }

    /// Input: [B, C, T] (NCL) -> Output: [B, C_out, T_out] (NCL)
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        if padAmount > 0 {
            let zeros = MLXArray.zeros([x.dim(0), x.dim(1), padAmount], dtype: x.dtype)
            switch causalType {
            case .left:
                h = concatenated([zeros, h], axis: 2)   // left-pad (causal)
            case .right:
                h = concatenated([h, zeros], axis: 2)    // right-pad (look-ahead)
            }
        }
        // NCL -> NLC for MLX Conv1d
        h = h.transposed(0, 2, 1)
        h = conv(h)
        // NLC -> NCL
        return h.transposed(0, 2, 1)
    }
}

// MARK: - Causal Conv1d Upsample (NCL)

/// Causal upsampling: nearest-neighbor upsample + causal Conv1d.
/// CausalHiFTGenerator uses this instead of ConvTranspose1d.
/// The upsample is parameter-free (nearest-neighbor), Conv1d refines the result.
public class CausalConv1dUpsample: Module {
    @ModuleInfo var conv: CausalDilatedConv1d
    let stride: Int

    public init(
        inputChannels: Int,
        outputChannels: Int,
        kernelSize: Int,
        stride: Int
    ) {
        self.stride = stride
        self._conv.wrappedValue = CausalDilatedConv1d(
            inputChannels: inputChannels,
            outputChannels: outputChannels,
            kernelSize: kernelSize)
        super.init()
    }

    /// Input: [B, C, T] (NCL) -> Output: [B, C_out, T*stride] (NCL)
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Nearest-neighbor upsample along time axis (NCL: axis=2)
        // Repeat each time step `stride` times: [B, C, T] -> [B, C, T*stride]
        var h: MLXArray
        if stride > 1 {
            // [B, C, T] -> [B, C, T, 1] -> repeat -> [B, C, T, stride] -> reshape [B, C, T*stride]
            let expanded = x.expandedDimensions(axis: 3)
            let repeated = MLX.repeated(expanded, count: stride, axis: 3)
            h = repeated.reshaped(x.dim(0), x.dim(1), x.dim(2) * stride)
        } else {
            h = x
        }
        // Apply causal Conv1d
        return conv(h)
    }
}

// MARK: - Strided Conv1d (NCL)

/// Causal strided Conv1d for downsampling, operating in NCL format.
/// Uses left-only padding of (stride - 1) to maintain causal alignment,
/// matching Python CausalConv1dDownSample behavior.
public class CausalConv1dDownSample: Module {
    @ModuleInfo var conv: Conv1d
    let causalPadding: Int

    public init(
        inputChannels: Int,
        outputChannels: Int,
        kernelSize: Int,
        stride: Int = 1,
        bias: Bool = true
    ) {
        // Python: self.causal_padding = stride - 1
        self.causalPadding = stride > 1 ? stride - 1 : 0
        self._conv.wrappedValue = Conv1d(
            inputChannels: inputChannels,
            outputChannels: outputChannels,
            kernelSize: kernelSize,
            stride: stride,
            padding: 0,
            bias: bias)
        super.init()
    }

    /// Input: [B, C, T] (NCL) -> Output: [B, C_out, T_out] (NCL)
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        // Left-only causal padding (matches Python CausalConv1dDownSample)
        if causalPadding > 0 {
            let zeros = MLXArray.zeros([x.dim(0), x.dim(1), causalPadding], dtype: x.dtype)
            h = concatenated([zeros, h], axis: 2)
        }
        // NCL -> NLC for MLX Conv1d
        h = h.transposed(0, 2, 1)
        h = conv(h)
        // NLC -> NCL
        return h.transposed(0, 2, 1)
    }
}

// MARK: - ResBlock

/// Residual block with multiple dilated convolutions and Snake activations.
/// Each dilation stage: snake1 -> dilated_conv1 -> snake2 -> conv2(dilation=1) -> residual add.
/// All data in NCL format [B, C, T].
public class ResBlock: Module {
    @ModuleInfo var convs1: [CausalDilatedConv1d]
    @ModuleInfo var convs2: [CausalDilatedConv1d]
    @ModuleInfo var activations1: [SnakeActivation]
    @ModuleInfo var activations2: [SnakeActivation]

    public init(channels: Int, kernelSize: Int, dilations: [Int]) {
        var c1: [CausalDilatedConv1d] = []
        var c2: [CausalDilatedConv1d] = []
        var a1: [SnakeActivation] = []
        var a2: [SnakeActivation] = []

        for dilation in dilations {
            a1.append(SnakeActivation(channels: channels))
            c1.append(CausalDilatedConv1d(
                inputChannels: channels,
                outputChannels: channels,
                kernelSize: kernelSize,
                dilation: dilation))
            a2.append(SnakeActivation(channels: channels))
            c2.append(CausalDilatedConv1d(
                inputChannels: channels,
                outputChannels: channels,
                kernelSize: kernelSize,
                dilation: 1))
        }

        self._convs1 = ModuleInfo(wrappedValue: c1)
        self._convs2 = ModuleInfo(wrappedValue: c2)
        self._activations1 = ModuleInfo(wrappedValue: a1)
        self._activations2 = ModuleInfo(wrappedValue: a2)
        super.init()
    }

    /// Input/output: [B, C, T]
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for i in 0..<convs1.count {
            var xt = activations1[i](h)
            xt = convs1[i](xt)
            xt = activations2[i](xt)
            xt = convs2[i](xt)
            h = h + xt
        }
        return h
    }
}

// MARK: - Sine Generator

/// Generates harmonic sine waves from F0 for the neural source filter.
/// Produces sine_amp * sin(2*pi * cumsum(f0 * [1..numHarmonics+1] / sampleRate))
/// with voiced/unvoiced masking.
public class SineGenerator {
    let sampleRate: Int
    let harmonicNum: Int      // Number of additional harmonics (total = harmonicNum + 1)
    let sineAmp: Float
    let noiseStd: Float
    let voicedThreshold: Float

    public init(
        sampleRate: Int = 24000,
        harmonicNum: Int = 8,
        sineAmp: Float = 0.1,
        noiseStd: Float = 0.003,
        voicedThreshold: Float = 10.0
    ) {
        self.sampleRate = sampleRate
        self.harmonicNum = harmonicNum
        self.sineAmp = sineAmp
        self.noiseStd = noiseStd
        self.voicedThreshold = voicedThreshold
    }

    /// Generate sine waves and voiced/unvoiced mask from F0.
    /// - Parameter f0: [B, T, 1] fundamental frequency in Hz
    /// - Returns: (sineWaves: [B, T, numHarmonics+1], uvMask: [B, T, 1])
    public func callAsFunction(_ f0: MLXArray) -> (MLXArray, MLXArray) {
        let totalHarmonics = harmonicNum + 1  // fundamental + overtones

        // Create harmonic multipliers: [1, 2, 3, ..., totalHarmonics]
        let harmonics = MLXArray(Array(1...totalHarmonics).map { Float($0) })
            .reshaped([1, 1, totalHarmonics])  // [1, 1, H]

        // f0: [B, T, 1] -> frequencies: [B, T, H]
        let frequencies = (f0 * harmonics) / Float(sampleRate)

        // Voiced mask: f0 > threshold -> 1.0, else 0.0
        let uvMask = MLX.where(
            f0 .> MLXArray(voicedThreshold),
            MLXArray(Float(1.0)),
            MLXArray(Float(0.0)))

        // Zero out unvoiced regions before cumsum to prevent phase drift
        let maskedFreqs = frequencies * uvMask  // [B, T, H]

        // Phase: 2*pi * cumsum(freq) along time axis
        let phase = cumsum(maskedFreqs, axis: 1) * MLXArray(Float(2.0 * Float.pi))

        // Add random initial phase offset to avoid artifacts at boundaries
        let initPhase = MLXRandom.uniform(
            low: 0,
            high: Float(2.0 * Float.pi),
            [f0.dim(0), 1, totalHarmonics])
        let fullPhase = phase + initPhase

        // Generate sine waves
        var sineWaves = MLXArray(sineAmp) * sin(fullPhase)  // [B, T, H]

        // Apply voiced mask: voiced -> sine, unvoiced -> noise
        let noise = MLXRandom.normal(sineWaves.shape) * MLXArray(noiseStd)
        sineWaves = sineWaves * uvMask + noise * (1.0 - uvMask)

        return (sineWaves, uvMask)
    }
}

// MARK: - Source Module (NSF)

/// Neural Source Filter: generates excitation signal by merging harmonic sines.
/// sine_generator -> linear_merge -> tanh -> add noise
public class SourceModuleHnNSF: Module {
    let sineGen: SineGenerator
    @ModuleInfo var linearMerge: Linear
    let noiseStd: Float

    public init(
        sampleRate: Int = 24000,
        harmonicNum: Int = 8,
        sineAmp: Float = 0.1,
        noiseStd: Float = 0.003,
        voicedThreshold: Float = 10.0
    ) {
        self.sineGen = SineGenerator(
            sampleRate: sampleRate,
            harmonicNum: harmonicNum,
            sineAmp: sineAmp,
            noiseStd: noiseStd,
            voicedThreshold: voicedThreshold)
        self.noiseStd = noiseStd
        // Merge all harmonics (fundamental + overtones) into 1 channel
        self._linearMerge.wrappedValue = Linear(harmonicNum + 1, 1)
        super.init()
    }

    /// - Parameter f0: [B, T, 1] fundamental frequency in Hz
    /// - Returns: excitation signal [B, T, 1]
    public func callAsFunction(_ f0: MLXArray) -> MLXArray {
        let (sineWaves, _) = sineGen(f0)  // [B, T, H]
        let merged = tanh(linearMerge(sineWaves))  // [B, T, 1]
        let noise = MLXRandom.normal(merged.shape) * MLXArray(noiseStd)
        return merged + noise
    }
}

// MARK: - F0 Predictor

/// Predicts F0 (fundamental frequency) from mel spectrogram.
/// 6-layer Conv1d stack (80 -> 512 -> ... -> 512) with ELU + final Linear classifier.
/// Input: [B, 80, T] (NCL), output: [B, T] positive F0 values.
public class F0Predictor: Module {
    @ModuleInfo var condnet: [CausalDilatedConv1d]
    @ModuleInfo var classifier: Linear

    public init(inChannels: Int = 80, hiddenChannels: Int = 512, numLayers: Int = 5) {
        var layers: [CausalDilatedConv1d] = []
        for i in 0..<numLayers {
            let inC = (i == 0) ? inChannels : hiddenChannels
            let kernelSize = (i == 0) ? 4 : 3  // First layer kernel=4, rest kernel=3
            // First layer uses right-padding (look-ahead), rest use left-padding (causal)
            let causal: CausalDilatedConv1d.CausalType = (i == 0) ? .right : .left
            layers.append(CausalDilatedConv1d(
                inputChannels: inC,
                outputChannels: hiddenChannels,
                kernelSize: kernelSize,
                dilation: 1,
                causalType: causal))
        }
        self._condnet = ModuleInfo(wrappedValue: layers)
        self._classifier.wrappedValue = Linear(hiddenChannels, 1)
        super.init()
    }

    /// - Parameter mel: [B, 80, T] (NCL)
    /// - Returns: [B, T] positive F0 values
    public func callAsFunction(_ mel: MLXArray) -> MLXArray {
        var h = mel  // [B, 80, T]
        for layer in condnet {
            h = layer(h)
            // ELU activation: x if x > 0, else exp(x) - 1
            h = MLX.where(h .> 0, h, exp(h) - 1)
        }
        // NCL -> NLC for Linear
        h = h.transposed(0, 2, 1)  // [B, T, 512]
        h = classifier(h)          // [B, T, 1]
        h = h.squeezed(axis: 2)    // [B, T]
        return abs(h)
    }
}

// MARK: - F0 Interpolation

/// Upsample F0 by a given factor using nearest-neighbor (repeat) interpolation.
/// - Parameters:
///   - f0: [B, T] F0 values
///   - factor: integer upsample factor
/// - Returns: [B, T * factor] upsampled F0
private func interpolateF0(_ f0: MLXArray, factor: Int) -> MLXArray {
    let batch = f0.dim(0)
    let timeSteps = f0.dim(1)
    let outLen = timeSteps * factor

    if factor == 1 { return f0 }

    // Nearest-neighbor upsampling: repeat each sample `factor` times
    // f0: [B, T] -> [B, T, 1] -> repeat -> [B, T, factor] -> reshape [B, T*factor]
    let expanded = f0.expandedDimensions(axis: 2)  // [B, T, 1]
    let rep = repeated(expanded, count: factor, axis: 2)  // [B, T, factor]
    return rep.reshaped([batch, outLen])
}

// MARK: - STFT

/// Compute Short-Time Fourier Transform for small n_fft using DFT matrix multiplication.
///
/// For the HiFi-GAN vocoder, n_fft=16 is very small, so we use a direct DFT matrix approach
/// instead of an FFT algorithm. The DFT matrices are precomputed as Swift arrays and
/// converted to MLXArrays for batch matrix multiplication.
///
/// - Parameters:
///   - signal: [B, T] waveform
///   - nFFT: FFT size (16 for this vocoder)
///   - hopLen: hop length (4 for this vocoder)
/// - Returns: (real [B, nBins, nFrames], imag [B, nBins, nFrames]) where nBins = n_fft/2 + 1
private func stft(
    signal: MLXArray, nFFT: Int, hopLen: Int
) -> (MLXArray, MLXArray) {
    let batch = signal.dim(0)
    let sigLen = signal.dim(1)

    // Build Hann window: [n_fft]
    let hannCoeffs = (0..<nFFT).map { n in
        Float(0.5 * (1.0 - cos(2.0 * Double.pi * Double(n) / Double(nFFT))))
    }
    let window = MLXArray(hannCoeffs)  // [n_fft]

    // Build DFT matrix (real and imag parts)
    // W_real[k,n] = cos(2*pi*k*n/N), W_imag[k,n] = -sin(2*pi*k*n/N)
    let nBins = nFFT / 2 + 1
    var dftRealData = [Float](repeating: 0, count: nBins * nFFT)
    var dftImagData = [Float](repeating: 0, count: nBins * nFFT)
    for k in 0..<nBins {
        for n in 0..<nFFT {
            let angle = 2.0 * Double.pi * Double(k) * Double(n) / Double(nFFT)
            dftRealData[k * nFFT + n] = Float(cos(angle))
            dftImagData[k * nFFT + n] = Float(-sin(angle))
        }
    }
    let dftRealMat = MLXArray(dftRealData).reshaped([nBins, nFFT])
    let dftImagMat = MLXArray(dftImagData).reshaped([nBins, nFFT])

    // Center padding: pad n_fft//2 on each side with reflection (matches torch.stft center=True)
    var sig = signal
    let centerPad = nFFT / 2
    if centerPad > 0 {
        // Reflection padding: reverse indices [centerPad, centerPad-1, ..., 1]
        let leftIndices = MLXArray((1...centerPad).reversed().map { Int32($0) })
        let leftReflect = sig.take(leftIndices, axis: 1)
        // Reverse indices [sigLen-2, sigLen-3, ..., sigLen-1-centerPad]
        let rightIndices = MLXArray(((sigLen - 1 - centerPad)..<(sigLen - 1)).reversed().map { Int32($0) })
        let rightReflect = sig.take(rightIndices, axis: 1)
        sig = concatenated([leftReflect, sig, rightReflect], axis: 1)
    }
    if sig.dim(1) < nFFT {
        sig = concatenated([sig, MLXArray.zeros([batch, nFFT - sig.dim(1)])], axis: 1)
    }
    let paddedLen = sig.dim(1)

    // Frame the signal: extract overlapping windows
    let nFrames = Swift.max((paddedLen - nFFT) / hopLen + 1, 1)

    // Build framed signal using stacked slices: [B, nFrames, nFFT]
    var frames: [MLXArray] = []
    for f in 0..<nFrames {
        let start = f * hopLen
        let end = Swift.min(start + nFFT, paddedLen)
        if end - start == nFFT {
            frames.append(sig[0..., start..<end])
        } else {
            // Pad the last frame if needed
            let partial = sig[0..., start..<end]
            let padLen = nFFT - (end - start)
            let padArr = MLXArray.zeros([batch, padLen])
            frames.append(concatenated([partial, padArr], axis: 1))
        }
    }
    let framedSignal = stacked(frames, axis: 1)  // [B, nFrames, nFFT]

    // Apply window
    let windowed = framedSignal * window  // broadcast [nFFT] over [B, nFrames, nFFT]

    // DFT via matmul: [B, nFrames, nFFT] @ [nFFT, nBins] -> [B, nFrames, nBins]
    let dftRealT = dftRealMat.transposed()
    let dftImagT = dftImagMat.transposed()

    let real = matmul(windowed, dftRealT)  // [B, nFrames, nBins]
    let imag = matmul(windowed, dftImagT)  // [B, nFrames, nBins]

    // Transpose to NCL format: [B, nBins, nFrames]
    return (real.transposed(0, 2, 1), imag.transposed(0, 2, 1))
}

// MARK: - ISTFT

/// Compute Inverse STFT using DFT matrix for small n_fft with overlap-add.
///
/// The overlap-add is performed by reshaping windowed frames into segments of hopLen
/// and accumulating overlapping contributions. For n_fft=16 and hopLen=4, each frame
/// is split into 4 segments of 4 samples, and overlapping frames are summed.
///
/// - Parameters:
///   - magnitude: [B, nBins, nFrames] where nBins = n_fft/2 + 1
///   - phase: [B, nBins, nFrames]
///   - nFFT: FFT size
///   - hopLen: hop length
/// - Returns: [B, samples] reconstructed waveform
private func istft(
    magnitude: MLXArray, phase: MLXArray, nFFT: Int, hopLen: Int
) -> MLXArray {
    let batch = magnitude.dim(0)
    let nBins = nFFT / 2 + 1
    let nFrames = magnitude.dim(2)

    // Convert to NLC: [B, nFrames, nBins]
    let mag = magnitude.transposed(0, 2, 1)
    let ph = phase.transposed(0, 2, 1)

    // Reconstruct complex spectrum
    let real = mag * cos(ph)
    let imag = mag * sin(ph)

    // Mirror spectrum for bins nBins..nFFT-1 (conjugate symmetry)
    // Mirror indices: nBins-2, nBins-3, ..., 1
    let fullReal: MLXArray
    let fullImag: MLXArray
    if nBins < nFFT {
        // Build reversed index array for mirroring: [nBins-2, nBins-3, ..., 1]
        let mirrorIndices = MLXArray(
            (1...(nBins - 2)).reversed().map { Int32($0) })  // [7] for nBins=9

        // Gather mirrored bins: take along last axis
        let mirrorReal = real.take(mirrorIndices, axis: 2)
        let mirrorImag = -(imag.take(mirrorIndices, axis: 2))

        fullReal = concatenated([real, mirrorReal], axis: 2)  // [B, nFrames, nFFT]
        fullImag = concatenated([imag, mirrorImag], axis: 2)  // [B, nFrames, nFFT]
    } else {
        fullReal = real
        fullImag = imag
    }

    // Build IDFT matrix: W_inv[n,k] = exp(j*2*pi*n*k/N) / N
    // time[n] = (1/N) * sum_k (real[k]*cos(2*pi*n*k/N) - imag[k]*sin(2*pi*n*k/N))
    let invN = 1.0 / Double(nFFT)
    var idftCosData = [Float](repeating: 0, count: nFFT * nFFT)
    var idftSinData = [Float](repeating: 0, count: nFFT * nFFT)
    for n in 0..<nFFT {
        for k in 0..<nFFT {
            let angle = 2.0 * Double.pi * Double(n) * Double(k) / Double(nFFT)
            idftCosData[n * nFFT + k] = Float(cos(angle) * invN)
            idftSinData[n * nFFT + k] = Float(sin(angle) * invN)
        }
    }
    let idftCosMat = MLXArray(idftCosData).reshaped([nFFT, nFFT])
    let idftSinMat = MLXArray(idftSinData).reshaped([nFFT, nFFT])

    // IFFT: [B, nFrames, nFFT] @ [nFFT, nFFT]^T -> [B, nFrames, nFFT]
    let timeDomain = matmul(fullReal, idftCosMat.transposed())
                   - matmul(fullImag, idftSinMat.transposed())

    // Apply Hann window for overlap-add synthesis
    let hannCoeffs = (0..<nFFT).map { n in
        Float(0.5 * (1.0 - cos(2.0 * Double.pi * Double(n) / Double(nFFT))))
    }
    let window = MLXArray(hannCoeffs)
    let windowed = timeDomain * window  // [B, nFrames, nFFT]

    // Overlap-add via segment accumulation.
    // Split each frame into (nFFT / hopLen) segments of hopLen samples.
    // Then accumulate overlapping contributions at the correct output positions.
    let segmentsPerFrame = nFFT / hopLen  // 16 / 4 = 4
    let outHops = nFrames + segmentsPerFrame - 1  // Total output hops
    let outLen = outHops * hopLen

    // Reshape windowed to [B, nFrames, segmentsPerFrame, hopLen]
    let segments = windowed.reshaped([batch, nFrames, segmentsPerFrame, hopLen])

    // For each segment offset s (0..segmentsPerFrame-1), the contribution goes to
    // output hop position (frame_index + s).
    // Pad each segment channel so they align when summed:
    //   offset s: pad s hops on the left and (segmentsPerFrame - 1 - s) hops on the right
    var accumulated = MLXArray.zeros([batch, outLen])
    for s in 0..<segmentsPerFrame {
        // Extract segment s from all frames: [B, nFrames, hopLen]
        let seg = segments[0..., 0..., s, 0...]  // [B, nFrames, hopLen]
        // Flatten to [B, nFrames * hopLen]
        let segFlat = seg.reshaped([batch, nFrames * hopLen])
        // Pad: s*hopLen zeros on left, remaining on right
        let leftPad = s * hopLen
        let rightPad = outLen - leftPad - nFrames * hopLen
        if leftPad > 0 || rightPad > 0 {
            var parts: [MLXArray] = []
            if leftPad > 0 {
                parts.append(MLXArray.zeros([batch, leftPad]))
            }
            parts.append(segFlat)
            if rightPad > 0 {
                parts.append(MLXArray.zeros([batch, rightPad]))
            }
            accumulated = accumulated + concatenated(parts, axis: 1)
        } else {
            accumulated = accumulated + segFlat
        }
    }

    // Window normalization: compute sum of squared Hann window at each output position
    var windowSumData = [Float](repeating: 0, count: outLen)
    for f in 0..<nFrames {
        for n in 0..<nFFT {
            let outIdx = f * hopLen + n
            if outIdx < outLen {
                windowSumData[outIdx] += hannCoeffs[n] * hannCoeffs[n]
            }
        }
    }
    // Clamp to avoid division by zero
    for i in 0..<outLen {
        if windowSumData[i] < 1e-8 {
            windowSumData[i] = 1e-8
        }
    }
    let windowNorm = MLXArray(windowSumData).reshaped([1, outLen])

    return accumulated / windowNorm
}

// MARK: - HiFi-GAN Generator

/// HiFTGenerator vocoder with Neural Source Filter for CosyVoice3.
/// Converts 80-band mel spectrograms to 24kHz audio waveforms.
///
/// Architecture (HiFT = HiFi-GAN with Inverse STFT):
/// - F0 predictor: mel -> F0
/// - Source module: F0 -> harmonic excitation -> STFT
/// - Decoder: mel -> conv_pre -> 3x(Conv1d_reduce + source_inject + resblocks) -> conv_post -> ISTFT -> audio
///
/// Channel progression: 80 -> 512 -> 256 -> 128 -> 64 -> 18 (n_fft+2)
/// Note: "ups" are Conv1d channel-reduction layers (NOT ConvTranspose1d).
/// Spatial upsampling happens entirely through ISTFT at the end.
public class HiFiGANGenerator: Module {
    public let config: CosyVoiceHiFiGANConfig

    // Source filter
    @ModuleInfo var source: SourceModuleHnNSF
    @ModuleInfo var f0Predictor: F0Predictor

    // Main decoder
    @ModuleInfo var convPre: CausalDilatedConv1d
    @ModuleInfo var ups: [CausalConv1dUpsample]
    @ModuleInfo var sourceDowns: [Module]
    @ModuleInfo var sourceResblocks: [ResBlock]
    @ModuleInfo var resblocks: [[ResBlock]]
    @ModuleInfo var convPost: CausalDilatedConv1d

    /// Source downsample strides: cumulative reverse of upsample_rates
    /// For [8,5,3]: downsample_rates=[1,3,5], cumprod=[1,3,15], reversed=[15,3,1]
    static let sourceDownStrides = [15, 3, 1]
    /// Source downsample kernel sizes: stride*2 (or 1 for stride=1)
    static let sourceDownKernelSizes = [30, 6, 1]

    public init(config: CosyVoiceHiFiGANConfig = CosyVoiceHiFiGANConfig()) {
        self.config = config

        let numStages = config.upsampleRates.count  // 3

        // Source module
        self._source.wrappedValue = SourceModuleHnNSF(
            sampleRate: config.sampleRate,
            harmonicNum: config.nbHarmonics,
            sineAmp: config.nsfAlpha,
            noiseStd: config.nsfSigma,
            voicedThreshold: config.nsfVoicedThreshold)

        // F0 predictor
        self._f0Predictor.wrappedValue = F0Predictor(inChannels: config.inChannels)

        // conv_pre: 80 -> 512, kernel=5, right-padding (look-ahead)
        self._convPre.wrappedValue = CausalDilatedConv1d(
            inputChannels: config.inChannels,
            outputChannels: config.baseChannels,
            kernelSize: config.convPreLookRight + 1,  // 4 + 1 = 5
            causalType: .right)

        // Channel sizes: 512 -> 256 -> 128 -> 64
        var channels = [config.baseChannels]
        for _ in 0..<numStages {
            channels.append(channels.last! / 2)
        }

        // Causal upsample layers: nearest-neighbor upsample + Conv1d
        // CausalHiFTGenerator uses nn.Upsample(nearest) + Conv1d, not ConvTranspose1d
        var upLayers: [CausalConv1dUpsample] = []
        for i in 0..<numStages {
            upLayers.append(CausalConv1dUpsample(
                inputChannels: channels[i],
                outputChannels: channels[i + 1],
                kernelSize: config.upsampleKernelSizes[i],
                stride: config.upsampleRates[i]))
        }
        self._ups = ModuleInfo(wrappedValue: upLayers)

        // Source downsampling: each takes STFT source (18 channels) and reduces
        // to match the decoder channel count and time resolution at each stage.
        // Strides [15, 3, 1] downsample source from T*120 (STFT of T*480 waveform)
        // to match decoder resolution: T*8, T*40, T*120 after each ups stage.
        let stftChannels = config.istftNFFT + 2  // 18
        var srcDowns: [Module] = []
        var srcResblks: [ResBlock] = []
        for i in 0..<numStages {
            let stride = Self.sourceDownStrides[i]
            let kernelSize = Self.sourceDownKernelSizes[i]
            if stride > 1 {
                // CausalConv1dDownSample: strided conv with left-only causal padding
                srcDowns.append(CausalConv1dDownSample(
                    inputChannels: stftChannels,
                    outputChannels: channels[i + 1],
                    kernelSize: kernelSize,
                    stride: stride))
            } else {
                // CausalConv1d: kernel=1, no padding needed (left-pad, causal)
                srcDowns.append(CausalDilatedConv1d(
                    inputChannels: stftChannels,
                    outputChannels: channels[i + 1],
                    kernelSize: kernelSize))
            }
            srcResblks.append(ResBlock(
                channels: channels[i + 1],
                kernelSize: config.sourceResblockKernelSizes[i],
                dilations: [1, 3, 5]))
        }
        self._sourceDowns = ModuleInfo(wrappedValue: srcDowns)
        self._sourceResblocks = ModuleInfo(wrappedValue: srcResblks)

        // Multi-receptive-field fusion resblocks: numStages x numKernels each
        var allResblocks: [[ResBlock]] = []
        for i in 0..<numStages {
            var stageBlocks: [ResBlock] = []
            for j in 0..<config.resblockKernelSizes.count {
                stageBlocks.append(ResBlock(
                    channels: channels[i + 1],
                    kernelSize: config.resblockKernelSizes[j],
                    dilations: config.resblockDilationSizes[j]))
            }
            allResblocks.append(stageBlocks)
        }
        self._resblocks = ModuleInfo(wrappedValue: allResblocks)

        // conv_post: lastChannels -> (n_fft + 2)
        self._convPost.wrappedValue = CausalDilatedConv1d(
            inputChannels: channels[numStages],
            outputChannels: config.istftNFFT + 2,
            kernelSize: 7)

        super.init()
    }

    /// Convert mel spectrogram to audio waveform.
    /// - Parameter mel: [B, T, 80] or [B, 80, T] mel spectrogram
    /// - Returns: [B, samples] audio waveform clamped to [-audioLimit, audioLimit]
    public func callAsFunction(_ mel: MLXArray) -> MLXArray {
        // Ensure NCL format [B, 80, T]
        var melNCL: MLXArray
        if mel.dim(mel.ndim - 1) == config.inChannels {
            // [B, T, 80] -> [B, 80, T]
            melNCL = mel.transposed(0, 2, 1)
        } else {
            melNCL = mel
        }

        // 1. Predict F0 from mel: [B, 80, T] -> [B, T]
        let f0 = f0Predictor(melNCL)

        eval(f0)
        let f0Flat = f0.reshaped(-1)
        let f0Count = f0Flat.dim(0)
        var f0Sum: Float = 0
        var f0Max: Float = 0
        var f0NonZero = 0
        for i in 0..<f0Count {
            let v = f0Flat[i].item(Float.self)
            f0Sum += v
            if v > f0Max { f0Max = v }
            if v > 1.0 { f0NonZero += 1 }
        }
        print("  [HiFiGAN] F0: mean=\(f0Sum/Float(f0Count)), max=\(f0Max), nonzero(>1Hz)=\(f0NonZero)/\(f0Count)")

        // 2. Upsample F0 to waveform sample rate
        //    Total upsample: prod(upsampleRates) * istftHopLen = 120 * 4 = 480
        let totalUpsample = config.totalUpsampleFactor * config.istftHopLen
        let f0Up = interpolateF0(f0, factor: totalUpsample)  // [B, T*480]

        // 3. Generate source excitation signal
        let f0UpExpanded = f0Up.expandedDimensions(axis: 2)  // [B, T*480, 1]
        let sourceSignal = source(f0UpExpanded)  // [B, T*480, 1]

        eval(sourceSignal)
        let srcFlat2 = sourceSignal.reshaped(-1)
        let srcRMS = sqrt((srcFlat2 * srcFlat2).mean()).item(Float.self)
        print("  [HiFiGAN] Source signal: shape=\(sourceSignal.shape), RMS=\(srcRMS)")

        // 4. STFT of source -> real and imaginary parts in NCL format
        let sourceFlat = sourceSignal.squeezed(axis: 2)  // [B, T*480]
        let (sourceReal, sourceImag) = stft(
            signal: sourceFlat, nFFT: config.istftNFFT, hopLen: config.istftHopLen)
        // sourceReal, sourceImag: [B, nBins=9, T_stft]

        eval(sourceReal, sourceImag)
        print("  [HiFiGAN] Source STFT: real=\(sourceReal.shape), imag=\(sourceImag.shape)")

        // Concatenate real and imaginary as source STFT features: [B, 18, T_stft]
        let sourceSTFT = concatenated([sourceReal, sourceImag], axis: 1)

        // 5. Main decoder: conv_pre
        var x = convPre(melNCL)  // [B, 512, T]
        eval(x)
        let convPreRMS = sqrt((x * x).mean()).item(Float.self)
        print("  [HiFiGAN] After conv_pre: shape=\(x.shape), RMS=\(convPreRMS)")

        // 6. Channel-reduction stages with source injection and multi-receptive-field fusion
        let numKernels = Float(config.resblockKernelSizes.count)
        let lreluSlope = MLXArray(config.lreluSlope)
        for i in 0..<config.upsampleRates.count {
            // LeakyReLU activation before channel reduction
            x = maximum(x, lreluSlope * x)
            x = ups[i](x)  // Nearest-neighbor upsample + Conv1d channel reduction

            // Reflection pad at last upsample stage: pad 1 sample on the left
            if i == config.upsampleRates.count - 1 {
                // nn.ReflectionPad1d((1, 0)): for input [a, b, c, ...], output is [b, a, b, c, ...]
                // Reflects from position 1 (not copies position 0)
                let reflected = x[0..., 0..., 1..<2]  // Second time sample [B, C, 1]
                x = concatenated([reflected, x], axis: 2)
            }

            // Source injection: each sourceDowns independently projects the original
            // 18-channel STFT source to this stage's channel count, then adds via resblock.
            let debugSkipSource = false  // Set to true to debug without source
            if !debugSkipSource {
                let sDn: MLXArray
                if let downSample = sourceDowns[i] as? CausalConv1dDownSample {
                    sDn = downSample(sourceSTFT)
                } else if let causalConv = sourceDowns[i] as? CausalDilatedConv1d {
                    sDn = causalConv(sourceSTFT)
                } else {
                    fatalError("Unexpected sourceDowns type")
                }
                let sRes = sourceResblocks[i](sDn)

                // Match time dimensions (trim to minimum length)
                let xLen = x.dim(2)
                let sLen = sRes.dim(2)
                let minLen = Swift.min(xLen, sLen)
                if xLen > minLen {
                    x = x[0..., 0..., 0..<minLen]
                }
                let sResTrimmed = (sLen > minLen) ? sRes[0..., 0..., 0..<minLen] : sRes
                x = x + sResTrimmed
            }

            // Multi-receptive-field fusion: average outputs of resblocks with different kernel sizes
            var fused = resblocks[i][0](x)
            for j in 1..<resblocks[i].count {
                fused = fused + resblocks[i][j](x)
            }
            x = fused / MLXArray(numKernels)

            eval(x)
            let stageRMS = sqrt((x * x).mean()).item(Float.self)
            print("  [HiFiGAN] After stage \(i): shape=\(x.shape), RMS=\(stageRMS)")
        }

        // 7. Final layers (LeakyReLU + conv_post)
        // Python uses F.leaky_relu(x) with DEFAULT slope=0.01 (not config lrelu_slope=0.1)
        x = maximum(x, MLXArray(Float(0.01)) * x)
        x = convPost(x)  // [B, 18, T_final]

        eval(x)
        let postRMS = sqrt((x * x).mean()).item(Float.self)
        print("  [HiFiGAN] After conv_post: shape=\(x.shape), RMS=\(postRMS)")
        // Per-channel stats for magnitude (first 9) and phase (last 9)
        for ch in 0..<18 {
            let chData = x[0, ch]
            let chMean = chData.mean().item(Float.self)
            let chMin = chData.min().item(Float.self)
            let chMax = chData.max().item(Float.self)
            let label = ch < 9 ? "mag" : "phs"
            print("    ch[\(ch)] \(label): mean=\(chMean), range=[\(chMin), \(chMax)]")
        }

        // 8. Split into magnitude (exp) and phase (sin)
        let nBins = config.istftNFFT / 2 + 1  // 9
        let magPart = x[0..., 0..<nBins, 0...]             // [B, 9, T_final]
        let phasePart = x[0..., nBins..<(nBins * 2), 0...]  // [B, 9, T_final]

        let outputMag = exp(magPart)
        let outputPhase = sin(phasePart)

        eval(outputMag, outputPhase)
        let magMean = outputMag.mean().item(Float.self)
        let magMax = outputMag.max().item(Float.self)
        print("  [HiFiGAN] ISTFT mag: mean=\(magMean), max=\(magMax), phase_range=[\(phasePart.min().item(Float.self)), \(phasePart.max().item(Float.self))]")

        // 9. ISTFT to reconstruct audio
        let audio = istft(
            magnitude: outputMag, phase: outputPhase,
            nFFT: config.istftNFFT, hopLen: config.istftHopLen)  // [B, samples]

        // 10. Clamp to audio limit
        return clip(audio, min: -config.audioLimit, max: config.audioLimit)
    }

    /// Decode mel spectrogram to float audio samples.
    /// - Parameter mel: [B, T, 80] or [B, 80, T] mel spectrogram
    /// - Returns: Audio samples as [Float]
    public func decode(mel: MLXArray) -> [Float] {
        let waveform = callAsFunction(mel)
        let flat = waveform.squeezed()
        eval(flat)
        return flat.asArray(Float.self)
    }
}
