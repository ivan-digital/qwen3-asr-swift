import MLX
import MLXFast
import MLXNN

public class ConvRMSNorm: Module, UnaryLayer {
    public let weight: MLXArray
    public let eps: Float

    public init(dimensions: Int, eps: Float = 1e-5) {
        self.weight = MLXArray.ones([dimensions])
        self.eps = eps
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let transposed = x.transposed(0, 2, 1)
        let normalized = MLXFast.rmsNorm(transposed, weight: weight, eps: eps)
        return normalized.transposed(0, 2, 1)
    }

    public func forwardToNLC(_ x: MLXArray) -> MLXArray {
        let transposed = x.transposed(0, 2, 1)
        return MLXFast.rmsNorm(transposed, weight: weight, eps: eps)
    }
}

public class ConvLayerNorm: Module, UnaryLayer {
    public let weight: MLXArray
    public let bias: MLXArray
    public let eps: Float

    public init(dimensions: Int, eps: Float = 1e-5) {
        self.weight = MLXArray.ones([dimensions])
        self.bias = MLXArray.zeros([dimensions])
        self.eps = eps
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let transposed = x.transposed(0, 2, 1)
        let normalized = MLXFast.layerNorm(transposed, weight: weight, bias: bias, eps: eps)
        return normalized.transposed(0, 2, 1)
    }
}
