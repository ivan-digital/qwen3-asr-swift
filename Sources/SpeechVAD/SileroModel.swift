import Foundation
import MLX
import MLXNN

/// Silero VAD v5 neural network: STFT → Encoder → LSTM → Decoder.
///
/// Processes 512-sample audio chunks (32ms @ 16kHz) with 64 samples of context
/// prepended from the previous chunk. The STFT uses a pre-computed DFT basis
/// stored as Conv1d weights.
///
/// Architecture:
/// ```
/// Input: 576 samples (64 context + 512 new)
///   → ReflectionPad(right=64) → 640 samples
///   → STFT Conv1d(1→258, k=256, s=128) → 4 frames × 258
///   → Magnitude: √(real² + imag²) → 4 × 129
///   → Encoder: 4× Conv1d+ReLU → 1 × 128
///   → LSTM(128→128, 1 layer) → hidden state [B, 128]
///   → ReLU → Conv1d(128→1, k=1) → Sigmoid → probability
/// ```
///
/// Weight keys:
/// ```
/// stft.weight                 [258, 256, 1]
/// encoder.{0-3}.weight        Conv1d weights
/// encoder.{0-3}.bias          Conv1d biases
/// lstm.{Wx, Wh, bias}         LSTM parameters
/// decoder.{weight, bias}      Final Conv1d
/// ```
class SileroVADNetwork: Module {

    // STFT: pre-computed DFT basis as Conv1d (no bias)
    @ModuleInfo(key: "stft") var stft: Conv1d

    // Encoder: 4 Conv1d + ReLU
    let encoder: [Conv1d]

    // LSTM: 128→128, 1 layer (reuses LSTMLayer for parameter structure)
    @ModuleInfo(key: "lstm") var lstm: LSTMLayer

    // Decoder: Conv1d(128→1, k=1)
    @ModuleInfo(key: "decoder") var decoder: Conv1d

    override init() {
        // STFT: filter_length=256, hop_length=128, 1→258 channels (129 real + 129 imag)
        self._stft.wrappedValue = Conv1d(
            inputChannels: 1, outputChannels: 258, kernelSize: 256,
            stride: 128, bias: false)

        // Encoder
        self.encoder = [
            Conv1d(inputChannels: 129, outputChannels: 128, kernelSize: 3, stride: 1, padding: 1),
            Conv1d(inputChannels: 128, outputChannels: 64, kernelSize: 3, stride: 2, padding: 1),
            Conv1d(inputChannels: 64, outputChannels: 64, kernelSize: 3, stride: 2, padding: 1),
            Conv1d(inputChannels: 64, outputChannels: 128, kernelSize: 3, stride: 1, padding: 1),
        ]

        // LSTM: input_size=128, hidden_size=128
        self._lstm.wrappedValue = LSTMLayer(inputSize: 128, hiddenSize: 128)

        // Decoder: 128→1 with kernel=1
        self._decoder.wrappedValue = Conv1d(
            inputChannels: 128, outputChannels: 1, kernelSize: 1)
    }

    /// Forward pass for a single chunk.
    ///
    /// - Parameters:
    ///   - samples: `[B, T]` raw audio (576 samples: 64 context + 512 new)
    ///   - h: LSTM hidden state `[1, B, 128]` or nil for initial state
    ///   - c: LSTM cell state `[1, B, 128]` or nil for initial state
    /// - Returns: `(probability [B], new_h [1, B, 128], new_c [1, B, 128])`
    func forward(_ samples: MLXArray, h: MLXArray?, c: MLXArray?) -> (MLXArray, MLXArray, MLXArray) {
        // [B, T] → [B, T, 1] for channels-last Conv1d
        var x = samples.expandedDimensions(axis: -1)

        // Reflection padding: 64 samples on the RIGHT side only
        // (matching Silero's pad(input, [0, 64], "reflect"))
        x = reflectionPadRight(x, padding: 64)

        // STFT via Conv1d: [B, 640, 1] → [B, 4, 258]
        x = stft(x)

        // Split real/imaginary and compute magnitude
        let real = x[0..., 0..., ..<129]
        let imag = x[0..., 0..., 129...]
        x = sqrt(real * real + imag * imag)  // [B, 4, 129]

        // Encoder: 4× Conv1d + ReLU → [B, 1, 128]
        for conv in encoder {
            x = relu(conv(x))
        }

        // LSTM with explicit h/c state
        // Encoder output is [B, 1, 128] — single timestep
        let (newH, newC) = lstmForward(x, h: h, c: c)

        // Decoder: use LSTM hidden state h, not full sequence output
        // h: [1, B, 128] → [B, 128] → [B, 1, 128] for Conv1d
        let hForDecoder = newH.squeezed(axis: 0).expandedDimensions(axis: 1)

        // ReLU → Conv1d(128→1, k=1) → Sigmoid
        let prob = sigmoid(decoder(relu(hForDecoder)))  // [B, 1, 1]

        return (prob.squeezed(axes: [1, 2]), newH, newC)
    }

    /// Run LSTM with explicit hidden/cell state for streaming.
    ///
    /// Accesses LSTMLayer's parameters directly (Wx, Wh, bias) rather than
    /// calling its `callAsFunction`, which doesn't support stateful operation.
    ///
    /// Returns (new_h [1, B, H], new_c [1, B, H])
    private func lstmForward(
        _ x: MLXArray, h: MLXArray?, c: MLXArray?
    ) -> (MLXArray, MLXArray) {
        // Project all timesteps: [B, T, 4*H]
        let projected = addMM(lstm.bias, x, lstm.wx.T)
        let seqLen = x.dim(-2)

        // Squeeze state from [1, B, H] to [B, H]
        var hidden = h?.squeezed(axis: 0)
        var cell = c?.squeezed(axis: 0)

        for t in 0 ..< seqLen {
            var ifgo = projected[0..., t, 0...]
            if let h = hidden {
                ifgo = ifgo + matmul(h, lstm.wh.T)
            }

            let pieces = split(ifgo, parts: 4, axis: -1)
            let i = sigmoid(pieces[0])
            let f = sigmoid(pieces[1])
            let g = tanh(pieces[2])
            let o = sigmoid(pieces[3])

            if let c = cell {
                cell = f * c + i * g
            } else {
                cell = i * g
            }
            hidden = o * tanh(cell!)
        }

        let newH = hidden!.expandedDimensions(axis: 0)  // [1, B, H]
        let newC = cell!.expandedDimensions(axis: 0)  // [1, B, H]

        return (newH, newC)
    }
}

// MARK: - Reflection Padding

/// Right-only reflection padding for 1D data in channels-last format `[B, T, C]`.
///
/// Matches Silero's `F.pad(input, [0, 64], mode='reflect')`.
/// For input `[a, b, c, d, e]` with padding=2: `[a, b, c, d, e, d, c]`
func reflectionPadRight(_ x: MLXArray, padding: Int) -> MLXArray {
    let T = x.dim(1)
    guard padding > 0, T > padding else { return x }

    // Right: reflect indices [T-2, T-3, ..., T-1-padding]
    let rightIndices = MLXArray(Array(stride(from: T - 2, through: T - 1 - padding, by: -1)))
    let rightPad = x.take(rightIndices, axis: 1)

    return concatenated([x, rightPad], axis: 1)
}

/// Symmetric reflection padding for 1D data in channels-last format `[B, T, C]`.
///
/// Pads the time dimension by reflecting values at both boundaries.
/// For input `[a, b, c, d, e]` with padding=2: `[c, b, a, b, c, d, e, d, c]`
func reflectionPad1d(_ x: MLXArray, padding: Int) -> MLXArray {
    let T = x.dim(1)
    guard padding > 0, T > padding else { return x }

    // Left: reflect indices [padding, padding-1, ..., 1]
    let leftIndices = MLXArray(Array(stride(from: padding, through: 1, by: -1)))
    let leftPad = x.take(leftIndices, axis: 1)

    // Right: reflect indices [T-2, T-3, ..., T-1-padding]
    let rightIndices = MLXArray(Array(stride(from: T - 2, through: T - 1 - padding, by: -1)))
    let rightPad = x.take(rightIndices, axis: 1)

    return concatenated([leftPad, x, rightPad], axis: 1)
}
