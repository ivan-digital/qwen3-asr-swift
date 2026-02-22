import MLX
import MLXFast
import MLXNN

// MARK: - KV Cache Protocol

public protocol KVCache: AnyObject {
    var offset: Int { get }
    func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray)
    func trim(_ count: Int)
}

// MARK: - Simple KV Cache (concatenation-based)

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

// MARK: - Ring KV Cache (fixed-size circular buffer for bounded context)

public final class RingKVCache: KVCache {
    private var keys: MLXArray?
    private var values: MLXArray?
    private let capacity: Int
    private var writePos: Int = 0
    private var totalWritten: Int = 0

    public init(capacity: Int) {
        self.capacity = capacity
    }

    public var offset: Int { totalWritten }

    /// How many valid entries are in the buffer
    public var length: Int { min(totalWritten, capacity) }

    public func update(keys newK: MLXArray, values newV: MLXArray) -> (MLXArray, MLXArray) {
        let t = newK.shape[2]

        if keys == nil {
            // First call: allocate buffer
            let b = newK.shape[0]
            let h = newK.shape[1]
            let d = newK.shape[3]
            keys = MLXArray.zeros([b, h, capacity, d], dtype: newK.dtype)
            values = MLXArray.zeros([b, h, capacity, d], dtype: newV.dtype)
        }

        // Write new entries
        for i in 0..<t {
            let pos = (writePos + i) % capacity
            let kSlice = newK[0..., 0..., i..<(i + 1), 0...]
            let vSlice = newV[0..., 0..., i..<(i + 1), 0...]
            // Scatter into buffer position
            keys![0..., 0..., pos..<(pos + 1), 0...] = kSlice
            values![0..., 0..., pos..<(pos + 1), 0...] = vSlice
        }
        writePos = (writePos + t) % capacity
        totalWritten += t

        // Return valid portion in order
        let validLen = min(totalWritten, capacity)
        if totalWritten <= capacity {
            let k = keys![0..., 0..., 0..<validLen, 0...]
            let v = values![0..., 0..., 0..<validLen, 0...]
            return (k, v)
        } else {
            // Circular: reorder so oldest is first
            let readStart = writePos
            let part1K = keys![0..., 0..., readStart..<capacity, 0...]
            let part2K = keys![0..., 0..., 0..<readStart, 0...]
            let part1V = values![0..., 0..., readStart..<capacity, 0...]
            let part2V = values![0..., 0..., 0..<readStart, 0...]
            return (concatenated([part1K, part2K], axis: 2),
                    concatenated([part1V, part2V], axis: 2))
        }
    }

    public func trim(_ count: Int) {
        if count >= totalWritten {
            keys = nil
            values = nil
            writePos = 0
            totalWritten = 0
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

    if offset > 0 {
        let causal = MLXArray.tri(T, m: totalLen, k: offset, type: Float.self) * 1e9 - 1e9
        return .array(causal.reshaped([1, 1, T, totalLen]).asType(dtype))
    }

    let causal = MLXArray.tri(T, m: T, k: 0, type: Float.self) * 1e9 - 1e9
    return .array(causal.reshaped([1, 1, T, T]).asType(dtype))
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
