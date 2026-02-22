import MLX

// MARK: - PadMode

public enum PadMode: Sendable {
    case constant
    case edge
}

// MARK: - NCL Padding

/// Pads an NCL tensor on the length (last) axis.
public func paddedNCL(_ xs: MLXArray, widths: [(Int, Int)], mode: PadMode) -> MLXArray {
    let (padL, padR) = widths[2]
    if padL == 0 && padR == 0 { return xs }

    var result = xs
    switch mode {
    case .constant:
        if padL > 0 {
            let leftZeros = MLXArray.zeros([xs.shape[0], xs.shape[1], padL], dtype: xs.dtype)
            result = concatenated([leftZeros, result], axis: 2)
        }
        if padR > 0 {
            let rightZeros = MLXArray.zeros([xs.shape[0], xs.shape[1], padR], dtype: xs.dtype)
            result = concatenated([result, rightZeros], axis: 2)
        }
    case .edge:
        if padL > 0 {
            let first = split(xs, indices: [1], axis: 2)[0]
            let leftPad = tiled(first, repetitions: [1, 1, padL])
            result = concatenated([leftPad, result], axis: 2)
        }
        if padR > 0 {
            let origLen = xs.shape[2]
            let last = split(xs, indices: [origLen - 1], axis: 2)[1]
            let rightPad = tiled(last, repetitions: [1, 1, padR])
            result = concatenated([result, rightPad], axis: 2)
        }
    }
    return result
}

// MARK: - ELU Activation

@inline(__always)
public func elu(_ x: MLXArray, alpha: Float = 1.0) -> MLXArray {
    MLX.where(x .> 0, x, MLXArray(alpha) * (exp(x) - 1))
}

// MARK: - IntOrPair Bridge

func padded(_ xs: MLXArray, widths: [IntOrPair], mode: PadMode) -> MLXArray {
    let tuples = widths.map { ($0.first, $0.second) }
    return paddedNCL(xs, widths: tuples, mode: mode)
}
