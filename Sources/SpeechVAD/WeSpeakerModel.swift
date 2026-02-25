import MLX
import MLXNN

/// ResNet BasicBlock with BN fused into Conv2d.
///
/// Each block has two 3×3 Conv2d layers with bias (fused BatchNorm).
/// Shortcut Conv2d(1×1) is added when stride≠1 or channels change.
class BasicBlock: Module {
    let conv1: Conv2d
    let conv2: Conv2d
    let shortcut: Conv2d?
    let stride: Int

    init(inChannels: Int, outChannels: Int, stride: Int = 1) {
        self.stride = stride

        self.conv1 = Conv2d(
            inputChannels: inChannels, outputChannels: outChannels,
            kernelSize: 3, stride: IntOrPair(arrayLiteral: stride, stride),
            padding: 1, bias: true
        )
        self.conv2 = Conv2d(
            inputChannels: outChannels, outputChannels: outChannels,
            kernelSize: 3, stride: 1, padding: 1, bias: true
        )

        if stride != 1 || inChannels != outChannels {
            self.shortcut = Conv2d(
                inputChannels: inChannels, outputChannels: outChannels,
                kernelSize: 1, stride: IntOrPair(arrayLiteral: stride, stride),
                padding: 0, bias: true
            )
        } else {
            self.shortcut = nil
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = relu(conv1(x))
        out = conv2(out)

        let residual: MLXArray
        if let shortcut {
            residual = shortcut(x)
        } else {
            residual = x
        }

        return relu(out + residual)
    }
}

/// WeSpeaker ResNet34 speaker embedding network (BN-fused).
///
/// Architecture:
/// ```
/// Input: [B, T, 80, 1] mel spectrogram
/// → Conv2d(1→32, k=3, p=1) + ReLU
/// → Layer1: 3× BasicBlock(32→32)
/// → Layer2: 4× BasicBlock(32→64, s=2)
/// → Layer3: 6× BasicBlock(64→128, s=2)
/// → Layer4: 3× BasicBlock(128→256, s=2)
/// → Statistics Pooling: mean + std → [B, 5120]
/// → Linear(5120→256) → L2 normalize
/// Output: [B, 256] speaker embedding
/// ```
class WeSpeakerNetwork: Module {
    let conv1: Conv2d
    let layer1: [BasicBlock]
    let layer2: [BasicBlock]
    let layer3: [BasicBlock]
    let layer4: [BasicBlock]
    let embedding: Linear

    override init() {
        self.conv1 = Conv2d(
            inputChannels: 1, outputChannels: 32,
            kernelSize: 3, stride: 1, padding: 1, bias: true
        )

        // Layer1: 3 blocks, 32→32
        var blocks1 = [BasicBlock]()
        for _ in 0..<3 {
            blocks1.append(BasicBlock(inChannels: 32, outChannels: 32))
        }
        self.layer1 = blocks1

        // Layer2: 4 blocks, 32→64, first stride=2
        var blocks2 = [BasicBlock]()
        for i in 0..<4 {
            blocks2.append(BasicBlock(
                inChannels: i == 0 ? 32 : 64,
                outChannels: 64,
                stride: i == 0 ? 2 : 1
            ))
        }
        self.layer2 = blocks2

        // Layer3: 6 blocks, 64→128, first stride=2
        var blocks3 = [BasicBlock]()
        for i in 0..<6 {
            blocks3.append(BasicBlock(
                inChannels: i == 0 ? 64 : 128,
                outChannels: 128,
                stride: i == 0 ? 2 : 1
            ))
        }
        self.layer3 = blocks3

        // Layer4: 3 blocks, 128→256, first stride=2
        var blocks4 = [BasicBlock]()
        for i in 0..<3 {
            blocks4.append(BasicBlock(
                inChannels: i == 0 ? 128 : 256,
                outChannels: 256,
                stride: i == 0 ? 2 : 1
            ))
        }
        self.layer4 = blocks4

        // Pooling output: T/8 * 10 * 256 → mean+std → 2 * 10 * 256 = 5120
        self.embedding = Linear(5120, 256)
    }

    /// Forward pass.
    /// - Parameter mel: `[B, T, 80, 1]` mel spectrogram (channels-last)
    /// - Returns: `[B, 256]` L2-normalized speaker embedding
    func callAsFunction(_ mel: MLXArray) -> MLXArray {
        // mel: [B, T, 80, 1]
        var x = relu(conv1(mel))

        // ResNet layers
        for block in layer1 { x = block(x) }
        for block in layer2 { x = block(x) }
        for block in layer3 { x = block(x) }
        for block in layer4 { x = block(x) }
        // x: [B, T/8, 10, 256]

        // Reshape for statistics pooling: [B, T/8, 10 * 256] = [B, T/8, 2560]
        let B = x.dim(0)
        let T = x.dim(1)
        x = x.reshaped(B, T, -1)

        // Statistics pooling: mean + std over time → [B, 5120]
        let mean = x.mean(axis: 1)  // [B, 2560]
        let variance = x.variance(axis: 1)  // [B, 2560]
        let std = sqrt(variance + 1e-10)
        let pooled = concatenated([mean, std], axis: -1)  // [B, 5120]

        // Embedding projection
        var emb = embedding(pooled)  // [B, 256]

        // L2 normalize
        let norm = sqrt((emb * emb).sum(axis: -1, keepDims: true) + 1e-10)
        emb = emb / norm

        return emb
    }
}
