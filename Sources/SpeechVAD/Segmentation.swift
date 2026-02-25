import MLX
import MLXNN

/// PyanNet segmentation model: SincNet → BiLSTM → Linear → Classifier.
///
/// Produces per-frame powerset class probabilities for up to 3 speakers.
/// Output classes (7): non-speech, spk1, spk2, spk3, spk1+2, spk1+3, spk2+3
///
/// Weight keys at model level:
/// ```
/// sincnet.{conv, norm, wav_norm}.*
/// lstm_fwd.layers.{i}.{Wx, Wh, bias}
/// lstm_bwd.layers.{i}.{Wx, Wh, bias}
/// linear.{0,1}.{weight, bias}
/// classifier.{weight, bias}
/// ```
class SegmentationModel: Module {
    let config: SegmentationConfig

    let sincnet: SincNet

    /// Forward and backward LSTM stacks (at top level for weight loading)
    @ModuleInfo(key: "lstm_fwd") var lstmFwd: LSTMStack
    @ModuleInfo(key: "lstm_bwd") var lstmBwd: LSTMStack

    let linear: [Linear]
    let classifier: Linear

    init(config: SegmentationConfig = .default) {
        self.config = config

        self.sincnet = SincNet(config: config)

        // Build LSTM stacks
        let sincnetOutputDim = config.sincnetFilters.last!  // 60
        var fwdLayers = [LSTMLayer]()
        var bwdLayers = [LSTMLayer]()
        for i in 0 ..< config.lstmNumLayers {
            let inSize = (i == 0) ? sincnetOutputDim : config.lstmHiddenSize * 2
            fwdLayers.append(LSTMLayer(inputSize: inSize, hiddenSize: config.lstmHiddenSize))
            bwdLayers.append(LSTMLayer(inputSize: inSize, hiddenSize: config.lstmHiddenSize))
        }

        // Linear layers with LeakyReLU
        let lstmOutputDim = config.lstmHiddenSize * 2  // 256 (bidirectional)
        var layers = [Linear]()
        for i in 0 ..< config.linearNumLayers {
            let inDim = (i == 0) ? lstmOutputDim : config.linearHiddenSize
            layers.append(Linear(inDim, config.linearHiddenSize))
        }
        self.linear = layers

        self.classifier = Linear(config.linearHiddenSize, config.numClasses)

        // Set @ModuleInfo properties after all stored properties
        self._lstmFwd.wrappedValue = LSTMStack(layers: fwdLayers)
        self._lstmBwd.wrappedValue = LSTMStack(layers: bwdLayers)
    }

    /// Run segmentation on audio.
    /// - Parameter waveform: `[batch, 1, samples]` (mono, 16kHz)
    /// - Returns: `[batch, num_frames, num_classes]` class probabilities
    func callAsFunction(_ waveform: MLXArray) -> MLXArray {
        // SincNet: [B, 1, T] → [B, 60, ~293]
        var x = sincnet(waveform)

        // Transpose for LSTM: [B, 60, L] → [B, L, 60]
        x = x.transposed(0, 2, 1)

        // BiLSTM: [B, L, 60] → [B, L, 256]
        x = runBiLSTM(x, fwd: lstmFwd, bwd: lstmBwd)

        // Linear layers with LeakyReLU
        for layer in linear {
            x = layer(x)
            x = leakyRelu(x)
        }

        // Classifier + softmax: [B, L, 7]
        x = classifier(x)
        x = softmax(x, axis: -1)

        return x
    }

    /// Extract speech probability from powerset output.
    ///
    /// Classes: [non-speech, spk1, spk2, spk3, spk1+2, spk1+3, spk2+3]
    /// Speech probability = 1 - P(non-speech).
    ///
    /// - Parameter posteriors: `[batch, frames, 7]`
    /// - Returns: `[batch, frames]` speech probability per frame
    static func speechProbability(from posteriors: MLXArray) -> MLXArray {
        let nonSpeech = posteriors[0..., 0..., 0]
        return 1.0 - nonSpeech
    }
}
