import Foundation
import MLX
import MLXFast

/// KV-cache protocol for the VibeVoice inference pipeline.
///
/// Kept public because several public `Qwen2*` forward methods take it as a
/// parameter. `PersonaPlex` ships a separate module-local `KVCache` protocol
/// with a different contract — at call sites that import both, disambiguate
/// as `VibeVoiceTTS.KVCache` / `PersonaPlex.KVCache`.
public protocol KVCache: AnyObject {
    var offset: Int { get }
    func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray)
}

public class KVCacheSimple: KVCache {
    internal var keys: MLXArray?
    internal var values: MLXArray?

    public private(set) var offset: Int = 0

    public var step: Int = 256

    public init(step: Int = 256) {
        self.step = step
    }

    public func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        let previous = self.offset
        let numNewTokens = keys.dim(2)

        let needsReset: Bool
        if let currentKeys = self.keys {
            needsReset = (previous + numNewTokens) > currentKeys.dim(2)
        } else {
            needsReset = true
        }

        if needsReset {
            let B = keys.dim(0)
            let kvHeads = keys.dim(1)
            let kHeadDim = keys.dim(3)
            let vHeadDim = values.dim(3)

            let nSteps = (step + numNewTokens - 1) / step
            let kShape = [B, kvHeads, nSteps * step, kHeadDim]
            let vShape = [B, kvHeads, nSteps * step, vHeadDim]
            let newK = MLXArray.zeros(kShape, dtype: keys.dtype)
            let newV = MLXArray.zeros(vShape, dtype: values.dtype)

            if var currentKeys = self.keys, var currentValues = self.values {
                if previous % step != 0 {
                    currentKeys = currentKeys[.ellipsis, ..<previous, 0...]
                    currentValues = currentValues[.ellipsis, ..<previous, 0...]
                }
                self.keys = concatenated([currentKeys, newK], axis: 2)
                self.values = concatenated([currentValues, newV], axis: 2)
            } else {
                self.keys = newK
                self.values = newV
            }
        }

        self.offset += numNewTokens

        self.keys?[.ellipsis, previous ..< self.offset, 0...] = keys
        self.values?[.ellipsis, previous ..< self.offset, 0...] = values

        guard let k = self.keys, let v = self.values else {
            return (keys, values)
        }

        return (k[.ellipsis, ..<self.offset, 0...], v[.ellipsis, ..<self.offset, 0...])
    }

    public func reset() {
        self.keys = nil
        self.values = nil
        self.offset = 0
    }

    public func initialize(keys: MLXArray, values: MLXArray) {
        self.keys = keys
        self.values = values
        self.offset = keys.dim(2)
    }

    public var sequenceLength: Int {
        keys?.dim(2) ?? 0
    }
}

public func createCausalMask(n: Int, offset: Int) -> MLXArray {
    var rinds = MLXArray(Int32(0) ..< Int32(offset + n))
    var linds = offset != 0 ? MLXArray(Int32(offset) ..< Int32(offset + n)) : rinds
    linds = linds[0..., .newAxis]
    rinds = rinds[.newAxis]
    return linds .>= rinds
}

public func createAttentionMask(h: MLXArray, cache: KVCache?) -> MLXFast.ScaledDotProductAttentionMaskMode {
    let n = h.dim(1)

    if n == 1 {
        return .none
    }

    let offset = cache?.offset ?? 0
    return .array(createCausalMask(n: n, offset: offset))
}

public func attentionWithCacheUpdate(
    queries: MLXArray,
    keys: MLXArray,
    values: MLXArray,
    cache: KVCache?,
    scale: Float,
    mask: MLXFast.ScaledDotProductAttentionMaskMode
) -> MLXArray {
    let (updatedKeys, updatedValues): (MLXArray, MLXArray)

    if let cache = cache {
        (updatedKeys, updatedValues) = cache.update(keys: keys, values: values)
    } else {
        (updatedKeys, updatedValues) = (keys, values)
    }

    return MLXFast.scaledDotProductAttention(
        queries: queries,
        keys: updatedKeys,
        values: updatedValues,
        scale: scale,
        mask: mask
    )
}
