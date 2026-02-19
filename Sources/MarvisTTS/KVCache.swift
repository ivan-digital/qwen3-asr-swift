import MLX
import MLXFast
import MLXNN

// MARK: - KV Cache

public protocol KVCache: AnyObject {
    var offset: Int { get }
    func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray)
    func trim(_ count: Int)
}

public final class KVCacheSimple: KVCache {
    private var keys: MLXArray?
    private var values: MLXArray?

    public init() {}

    public var offset: Int { keys?.shape[2] ?? 0 }

    public func update(keys newK: MLXArray, values newV: MLXArray) -> (MLXArray, MLXArray) {
        if let k = keys, let v = values {
            keys = concatenated([k, newK], axis: 2)
            values = concatenated([v, newV], axis: 2)
        } else {
            keys = newK
            values = newV
        }
        return (keys!, values!)
    }

    public func trim(_ count: Int) {
        if count >= offset {
            keys = nil
            values = nil
        } else if count > 0 {
            let newLen = offset - count
            if let k = keys, let v = values {
                keys = split(k, indices: [newLen], axis: 2)[0]
                values = split(v, indices: [newLen], axis: 2)[0]
            }
        }
    }
}

// MARK: - Attention Helpers

public func createAttentionMask(h: MLXArray, cache: (any KVCache)?) -> MLXFast.ScaledDotProductAttentionMaskMode {
    let T = h.shape[1]
    if T <= 1, (cache?.offset ?? 0) > 0 {
        return .none
    }

    let offset = cache?.offset ?? 0
    let totalLen = offset + T
    let dtype = h.dtype

    // Causal mask: each position can attend to itself and all previous positions
    if offset > 0 {
        let causal = MLXArray.tri(T, m: totalLen, k: offset, type: Float.self) * 1e9 - 1e9
        return .array(causal.reshaped([1, 1, T, totalLen]).asType(dtype))
    }

    // Standard causal: lower triangular
    let causal = MLXArray.tri(T, m: T, k: 0, type: Float.self) * 1e9 - 1e9
    return .array(causal.reshaped([1, 1, T, T]).asType(dtype))
}

public func createAttentionMask(h: MLXArray, cache: [KVCacheSimple]?) -> MLXFast.ScaledDotProductAttentionMaskMode {
    createAttentionMask(h: h, cache: cache?.first)
}

// MARK: - Linear/QuantizedLinear Helper

@inline(__always)
func applyLinear(_ module: Module, _ x: MLXArray) -> MLXArray {
    switch module {
    case let l as Linear: return l(x)
    case let q as QuantizedLinear: return q(x)
    default: fatalError("Expected Linear or QuantizedLinear, got \(type(of: module))")
    }
}

func makeLinear(_ inputDims: Int, _ outputDims: Int, bias: Bool, groupSize: Int? = nil, bits: Int? = nil) -> Module {
    if let gs = groupSize, let b = bits {
        return QuantizedLinear(inputDims, outputDims, bias: bias, groupSize: gs, bits: b)
    }
    return Linear(inputDims, outputDims, bias: bias)
}

