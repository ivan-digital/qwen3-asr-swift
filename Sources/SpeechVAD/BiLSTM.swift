import Foundation
import MLX
import MLXNN

/// A single LSTM layer with updatable parameters.
///
/// Uses `var` properties so they can be loaded via `update(parameters:)`.
/// Parameter keys match MLX convention: `Wx` (input→hidden), `Wh` (hidden→hidden), `bias`.
class LSTMLayer: Module {
    let hiddenSize: Int

    @ParameterInfo(key: "Wx") var wx: MLXArray
    @ParameterInfo(key: "Wh") var wh: MLXArray
    var bias: MLXArray

    init(inputSize: Int, hiddenSize: Int) {
        self.hiddenSize = hiddenSize
        let scale = 1.0 / Foundation.sqrt(Float(hiddenSize))
        self._wx.wrappedValue = MLXRandom.uniform(low: -scale, high: scale, [4 * hiddenSize, inputSize])
        self._wh.wrappedValue = MLXRandom.uniform(low: -scale, high: scale, [4 * hiddenSize, hiddenSize])
        self.bias = MLXRandom.uniform(low: -scale, high: scale, [4 * hiddenSize])
    }

    /// Process a sequence.
    /// - Parameter x: `[batch, seq_len, input_size]`
    /// - Returns: `[batch, seq_len, hidden_size]`
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Project all timesteps at once: [B, L, 4H]
        let projected = addMM(bias, x, wx.T)

        let seqLen = x.dim(-2)
        var hidden: MLXArray? = nil
        var cell: MLXArray? = nil
        var allHidden = [MLXArray]()

        for t in 0 ..< seqLen {
            var ifgo = projected[.ellipsis, t, 0...]
            if let h = hidden {
                ifgo = ifgo + matmul(h, wh.T)
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

            allHidden.append(hidden!)
        }

        return stacked(allHidden, axis: -2)
    }
}

/// Container for LSTM layers (enables module parameter tree: `layers.0`, `layers.1`, etc.)
class LSTMStack: Module {
    let layers: [LSTMLayer]

    init(layers: [LSTMLayer]) {
        self.layers = layers
    }
}

/// Run bidirectional LSTM across multiple layers.
///
/// For each layer, runs forward LSTM on the sequence and backward LSTM on the
/// reversed sequence, then concatenates outputs along the feature dimension.
///
/// - Parameters:
///   - x: `[batch, seq_len, features]`
///   - fwd: forward LSTM stack
///   - bwd: backward LSTM stack
/// - Returns: `[batch, seq_len, 2 * hidden_size]`
func runBiLSTM(_ x: MLXArray, fwd: LSTMStack, bwd: LSTMStack) -> MLXArray {
    var input = x

    for i in 0 ..< fwd.layers.count {
        // Forward direction
        let fwdOut = fwd.layers[i](input)

        // Backward direction: reverse → process → reverse back
        let seqLen = input.dim(-2)
        let indices = MLXArray(Array((0 ..< seqLen).reversed()))
        let reversed = input.take(indices, axis: -2)
        let bwdOutRev = bwd.layers[i](reversed)
        let bwdOut = bwdOutRev.take(indices, axis: -2)

        // Concatenate along feature dimension
        input = concatenated([fwdOut, bwdOut], axis: -1)
    }

    return input
}
