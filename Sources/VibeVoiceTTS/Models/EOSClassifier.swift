import Foundation
import MLX
import MLXNN

public class EOSClassifier: Module {
    @ModuleInfo(key: "fc1") public var fc1: Linear
    @ModuleInfo(key: "fc2") public var fc2: Linear

    public init(hiddenSize: Int) {
        _fc1.wrappedValue = Linear(hiddenSize, hiddenSize, bias: true)
        _fc2.wrappedValue = Linear(hiddenSize, 1, bias: true)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = relu(fc1(x))
        out = fc2(out)
        return out
    }

    public func probability(_ x: MLXArray) -> MLXArray {
        return sigmoid(callAsFunction(x))
    }
}
