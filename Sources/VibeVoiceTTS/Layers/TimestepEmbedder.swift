import Foundation
import MLX
import MLXNN

public class TimestepEmbedder: Module {
    public let mlp: MLP

    public let frequencyEmbeddingSize: Int

    public class MLP: Module {
        @ModuleInfo public var linear1: Linear
        @ModuleInfo public var linear2: Linear

        public init(frequencyEmbeddingSize: Int, hiddenSize: Int) {
            self._linear1.wrappedValue = Linear(frequencyEmbeddingSize, hiddenSize, bias: false)
            self._linear2.wrappedValue = Linear(hiddenSize, hiddenSize, bias: false)
            super.init()
        }
    }

    public init(hiddenSize: Int, frequencyEmbeddingSize: Int = 256) {
        self.frequencyEmbeddingSize = frequencyEmbeddingSize
        self.mlp = MLP(frequencyEmbeddingSize: frequencyEmbeddingSize, hiddenSize: hiddenSize)
        super.init()
    }

    public static func timestepEmbedding(_ t: MLXArray, dim: Int, maxPeriod: Float = 10000, dtype: DType = .float32) -> MLXArray {
        let half = dim / 2

        let logMaxPeriod = log(maxPeriod)
        let indices = MLXArray(0 ..< half).asType(dtype)
        let freqs = exp(-logMaxPeriod * indices / Float(half))

        let tExpanded = expandedDimensions(t.asType(dtype), axis: 1)
        let freqsExpanded = expandedDimensions(freqs, axis: 0)
        let args = tExpanded * freqsExpanded

        let cosArgs = cos(args)
        let sinArgs = sin(args)
        var embedding = concatenated([cosArgs, sinArgs], axis: -1)

        if dim % 2 != 0 {
            let padding = MLXArray.zeros([t.shape[0], 1], dtype: dtype)
            embedding = concatenated([embedding, padding], axis: -1)
        }

        return embedding
    }

    public func callAsFunction(_ t: MLXArray) -> MLXArray {
        let dtype = mlp.linear1.weight.dtype
        let tFreq = Self.timestepEmbedding(t, dim: frequencyEmbeddingSize, dtype: dtype)

        var x = mlp.linear1(tFreq)
        x = silu(x)
        x = mlp.linear2(x)

        return x
    }
}

public func modulate(_ x: MLXArray, shift: MLXArray, scale: MLXArray) -> MLXArray {
    return x * (1 + scale) + shift
}
