import MLX
import MLXFast
import MLXNN
import Foundation

/// Open-Unmix music source separation model.
///
/// Separates a stereo mixture into 4 stems: vocals, drums, bass, other.
/// Each stem has its own model instance with independent weights.
///
/// Architecture per stem:
/// ```
/// STFT magnitude → normalize → FC1 → BN1 → tanh → BiLSTM (3-layer)
/// → skip connection → FC2 → BN2 → ReLU → FC3 → BN3
/// → denormalize → ReLU mask × input magnitude → iSTFT
/// ```
///
/// Reference: https://github.com/sigsep/open-unmix-pytorch
public final class OpenUnmixStemModel: Module {

    // MARK: - Configuration

    /// Number of FFT bins (n_fft/2 + 1 = 2049)
    let nbBins: Int = 2049
    /// Max frequency bin (~16kHz at 44.1kHz, 4096 FFT)
    let maxBin: Int = 1487
    /// Number of audio channels (stereo = 2)
    let nbChannels: Int = 2
    /// Hidden size for FC layers and LSTM
    let hiddenSize: Int

    // MARK: - Normalization parameters (learned, not trained)

    @ParameterInfo(key: "input_mean")
    var inputMean: MLXArray
    @ParameterInfo(key: "input_scale")
    var inputScale: MLXArray
    @ParameterInfo(key: "output_mean")
    var outputMean: MLXArray
    @ParameterInfo(key: "output_scale")
    var outputScale: MLXArray

    // MARK: - Layers

    let fc1: Linear
    let bn1: BatchNorm
    let lstm: BiLSTMStack
    let fc2: Linear
    let bn2: BatchNorm
    let fc3: Linear
    let bn3: BatchNorm

    public init(hiddenSize: Int = 512) {
        self.hiddenSize = hiddenSize
        let inputFeatures = nbChannels * maxBin  // 2 * 1487 = 2974

        // Normalization params (loaded from weights)
        self._inputMean.wrappedValue = MLXArray.zeros([maxBin])
        self._inputScale.wrappedValue = MLXArray.ones([maxBin])
        self._outputMean.wrappedValue = MLXArray.zeros([nbBins])
        self._outputScale.wrappedValue = MLXArray.ones([nbBins])

        // Encoder
        self.fc1 = Linear(inputFeatures, hiddenSize, bias: false)
        self.bn1 = BatchNorm(featureCount: hiddenSize)

        // BiLSTM (3 layers, hidden_size/2 per direction)
        self.lstm = BiLSTMStack(
            inputSize: hiddenSize,
            hiddenSize: hiddenSize / 2,
            numLayers: 3
        )

        // Decoder (skip connection doubles input)
        self.fc2 = Linear(hiddenSize * 2, hiddenSize, bias: false)
        self.bn2 = BatchNorm(featureCount: hiddenSize)
        self.fc3 = Linear(hiddenSize, nbChannels * nbBins, bias: false)
        self.bn3 = BatchNorm(featureCount: nbChannels * nbBins)
    }

    /// Forward pass: magnitude spectrogram → masked magnitude spectrogram.
    ///
    /// - Parameter x: Input magnitude spectrogram [T, channels, bins] = [T, 2, 2049]
    /// - Returns: Masked magnitude spectrogram [T, channels, bins]
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let T = x.shape[0]
        let mix = x  // Save for masking

        // Crop to max_bin and normalize
        var h = x[0..., 0..., ..<maxBin]  // [T, 2, 1487]
        h = h + inputMean
        h = h * inputScale

        // Reshape to [T, 2*1487]
        h = h.reshaped([T, nbChannels * maxBin])

        // Encoder
        h = fc1(h)           // [T, hidden]
        h = bn1(h)           // [T, hidden]
        h = tanh(h)          // [T, hidden]

        // BiLSTM with skip connection
        let skip = h
        h = lstm(h)          // [T, hidden]
        h = concatenated([skip, h], axis: -1)  // [T, hidden*2]

        // Decoder
        h = fc2(h)           // [T, hidden]
        h = relu(bn2(h))     // [T, hidden]
        h = fc3(h)           // [T, channels*bins]
        h = bn3(h)           // [T, channels*bins]

        // Reshape and denormalize
        h = h.reshaped([T, nbChannels, nbBins])  // [T, 2, 2049]
        h = h * outputScale
        h = h + outputMean
        h = relu(h)

        // Apply mask
        return h * mix  // [T, 2, 2049]
    }
}

// MARK: - BiLSTM Stack

/// 3-layer bidirectional LSTM for Open-Unmix.
final class BiLSTMStack: Module {
    let layers: [BiLSTMLayer]

    init(inputSize: Int, hiddenSize: Int, numLayers: Int) {
        self.layers = (0..<numLayers).map { i in
            BiLSTMLayer(
                inputSize: i == 0 ? inputSize : hiddenSize * 2,
                hiddenSize: hiddenSize
            )
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for layer in layers {
            h = layer(h)
        }
        return h
    }
}

/// Single bidirectional LSTM layer.
final class BiLSTMLayer: Module {
    let forward: LSTMCell
    let backward: LSTMCell
    let hiddenSize: Int

    init(inputSize: Int, hiddenSize: Int) {
        self.hiddenSize = hiddenSize
        self.forward = LSTMCell(inputSize: inputSize, hiddenSize: hiddenSize)
        self.backward = LSTMCell(inputSize: inputSize, hiddenSize: hiddenSize)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let T = x.shape[0]

        // Forward pass
        var fwdOutputs: [MLXArray] = []
        var fwdH = MLXArray.zeros([1, hiddenSize])
        var fwdC = MLXArray.zeros([1, hiddenSize])
        for t in 0..<T {
            let xt = x[t].reshaped([1, -1])
            (fwdH, fwdC) = forward.step(xt, h: fwdH, c: fwdC)
            fwdOutputs.append(fwdH)
        }

        // Backward pass
        var bwdOutputs: [MLXArray] = []
        var bwdH = MLXArray.zeros([1, hiddenSize])
        var bwdC = MLXArray.zeros([1, hiddenSize])
        for t in stride(from: T - 1, through: 0, by: -1) {
            let xt = x[t].reshaped([1, -1])
            (bwdH, bwdC) = backward.step(xt, h: bwdH, c: bwdC)
            bwdOutputs.append(bwdH)
        }
        bwdOutputs.reverse()

        // Concatenate forward + backward
        let fwd = stacked(fwdOutputs, axis: 0).squeezed(axis: 1)  // [T, hidden]
        let bwd = stacked(bwdOutputs, axis: 0).squeezed(axis: 1)  // [T, hidden]
        return concatenated([fwd, bwd], axis: -1)  // [T, hidden*2]
    }
}

/// Single LSTM cell (one step).
final class LSTMCell: Module {
    @ParameterInfo(key: "weight_ih")
    var weightIH: MLXArray  // [4*hidden, input]
    @ParameterInfo(key: "weight_hh")
    var weightHH: MLXArray  // [4*hidden, hidden]
    @ParameterInfo(key: "bias_ih")
    var biasIH: MLXArray    // [4*hidden]
    @ParameterInfo(key: "bias_hh")
    var biasHH: MLXArray    // [4*hidden]

    let hiddenSize: Int

    init(inputSize: Int, hiddenSize: Int) {
        self.hiddenSize = hiddenSize
        let gateSize = 4 * hiddenSize
        self._weightIH.wrappedValue = MLXArray.zeros([gateSize, inputSize])
        self._weightHH.wrappedValue = MLXArray.zeros([gateSize, hiddenSize])
        self._biasIH.wrappedValue = MLXArray.zeros([gateSize])
        self._biasHH.wrappedValue = MLXArray.zeros([gateSize])
    }

    func step(_ x: MLXArray, h: MLXArray, c: MLXArray) -> (MLXArray, MLXArray) {
        // gates = x @ W_ih^T + b_ih + h @ W_hh^T + b_hh
        let gates = matmul(x, weightIH.T) + biasIH + matmul(h, weightHH.T) + biasHH

        let hs = hiddenSize
        let i = sigmoid(gates[0..., 0..<hs])
        let f = sigmoid(gates[0..., hs..<(2*hs)])
        let g = tanh(gates[0..., (2*hs)..<(3*hs)])
        let o = sigmoid(gates[0..., (3*hs)..<(4*hs)])

        let newC = f * c + i * g
        let newH = o * tanh(newC)
        return (newH, newC)
    }
}
