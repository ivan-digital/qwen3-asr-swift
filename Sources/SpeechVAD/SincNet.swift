import MLX
import MLXNN

/// SincNet feature extractor: 3 conv+pool+norm+activation layers.
///
/// The first conv layer uses pre-computed sinc bandpass filters (computed during
/// weight conversion). At runtime, all three layers are standard Conv1d.
///
/// All data flows in MLX channels-last format: `[batch, length, channels]`.
///
/// Architecture:
///   InstanceNorm(1) → Conv1d(1,80,k=251,s=10) → |·| → MaxPool(3,3) → InstanceNorm(80) → LeakyReLU
///   → Conv1d(80,60,k=5) → MaxPool(3,3) → InstanceNorm(60) → LeakyReLU
///   → Conv1d(60,60,k=5) → MaxPool(3,3) → InstanceNorm(60) → LeakyReLU
class SincNet: Module {
    /// Input waveform normalization
    @ModuleInfo(key: "wav_norm") var wavNorm: InstanceNorm

    /// Three conv layers (first is pre-computed sinc filterbank)
    let conv: [Conv1d]

    /// Instance norm after each conv+pool
    let norm: [InstanceNorm]

    init(config: SegmentationConfig) {
        let filters = config.sincnetFilters
        let kernels = config.sincnetKernelSizes
        let strides = config.sincnetStrides

        // Build conv layers: input channels are [1, 80, 60]
        let inputChannels = [1] + Array(filters.dropLast())
        self.conv = zip(zip(inputChannels, filters), zip(kernels, strides)).map { arg in
            let ((inC, outC), (k, s)) = arg
            return Conv1d(inputChannels: inC, outputChannels: outC, kernelSize: k, stride: s)
        }

        self.norm = filters.map { InstanceNorm(dimensions: $0) }

        self._wavNorm.wrappedValue = InstanceNorm(dimensions: 1)
    }

    /// Forward pass.
    /// - Parameter x: `[batch, 1, samples]` raw waveform (channels-first, transposed internally)
    /// - Returns: `[batch, channels, frames]` feature frames (channels-first for LSTM transpose)
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Convert from [B, C, T] to MLX channels-last [B, T, C]
        var out = x.transposed(0, 2, 1)

        // Normalize input waveform: [B, T, 1]
        out = wavNorm(out)

        for i in 0 ..< conv.count {
            // Conv1d: [B, T, Cin] → [B, T', Cout]
            out = conv[i](out)

            // First layer uses abs() (sinc filterbank — energy)
            if i == 0 {
                out = abs(out)
            }

            // MaxPool1d(3, stride=3) — pool over time (axis -2)
            out = maxPool1d(out, kernelSize: 3, stride: 3)

            // InstanceNorm + LeakyReLU
            out = norm[i](out)
            out = leakyRelu(out)
        }

        // Convert back to [B, C, T] for downstream compatibility
        return out.transposed(0, 2, 1)
    }
}

/// Instance normalization for 1D data (channels-last format).
///
/// Input shape: `[batch, length, channels]`
/// Normalizes over the length dimension per-channel, with learnable affine.
class InstanceNorm: Module {
    let dimensions: Int
    var weight: MLXArray
    var bias: MLXArray

    init(dimensions: Int) {
        self.dimensions = dimensions
        self.weight = MLXArray.ones([dimensions])
        self.bias = MLXArray.zeros([dimensions])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x: [B, L, C] — normalize over L dimension (axis 1)
        let mean = x.mean(axis: 1, keepDims: true)
        let variance = x.variance(axis: 1, keepDims: true)
        let eps: Float = 1e-5
        let normalized = (x - mean) * rsqrt(variance + eps)
        // Scale and shift: weight and bias are [C], broadcast naturally over [B, L, C]
        return normalized * weight + bias
    }
}

/// 1D max pooling for channels-last data.
///
/// Input: `[batch, length, channels]` — pools over the length dimension (axis -2).
///
/// - Parameters:
///   - x: `[batch, length, channels]`
///   - kernelSize: pooling window
///   - stride: pooling stride
/// - Returns: `[batch, floor((length - kernelSize) / stride) + 1, channels]`
func maxPool1d(_ x: MLXArray, kernelSize: Int, stride: Int) -> MLXArray {
    let length = x.dim(-2)  // time/length axis
    let outLen = (length - kernelSize) / stride + 1

    // Collect slices for each position in the kernel
    var slices = [MLXArray]()
    for k in 0 ..< kernelSize {
        // Take every stride-th element starting at offset k along time axis
        let slice = x[0..., .stride(from: k, to: k + outLen * stride, by: stride), 0...]
        slices.append(slice)
    }

    // Stack along new axis and take max: [B, outLen, C, kernelSize] → [B, outLen, C]
    let stacked = MLX.stacked(slices, axis: -1)
    return stacked.max(axis: -1)
}

/// LeakyReLU activation.
func leakyRelu(_ x: MLXArray, negativeSlope: Float = 0.01) -> MLXArray {
    maximum(x, x * negativeSlope)
}
