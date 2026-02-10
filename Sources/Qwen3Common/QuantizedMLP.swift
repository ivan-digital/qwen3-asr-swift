import Foundation
import MLX
import MLXNN

/// SwiGLU MLP with quantized linear layers — shared by ASR text decoder, TTS Talker, and Code Predictor
public class QuantizedMLP: Module {
    @ModuleInfo public var gateProj: QuantizedLinear
    @ModuleInfo public var upProj: QuantizedLinear
    @ModuleInfo public var downProj: QuantizedLinear

    public init(hiddenSize: Int, intermediateSize: Int, groupSize: Int = 64, bits: Int = 4) {
        self._gateProj.wrappedValue = QuantizedLinear(
            hiddenSize, intermediateSize, bias: false,
            groupSize: groupSize, bits: bits)
        self._upProj.wrappedValue = QuantizedLinear(
            hiddenSize, intermediateSize, bias: false,
            groupSize: groupSize, bits: bits)
        self._downProj.wrappedValue = QuantizedLinear(
            intermediateSize, hiddenSize, bias: false,
            groupSize: groupSize, bits: bits)

        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // SwiGLU: down(silu(gate(x)) * up(x))
        let gate = silu(gateProj(x))
        let up = upProj(x)
        return downProj(gate * up)
    }
}
