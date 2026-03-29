#if canImport(CoreML)
import CoreML
import Foundation

/// CoreML-based autoregressive Talker for Qwen3-TTS (Conv2d / NCHW layout).
///
/// Optimized for Apple Neural Engine:
/// - Conv2d layers (kernel=1) instead of Linear
/// - NCHW data layout: (batch, channels, 1, 1) per token
/// - Pre-allocated fixed-size KV cache with scatter-write mask
/// - Precomputed RoPE cos/sin tables passed as inputs
/// - Stacked KV cache: all 28 layers concatenated along channel dim
final class TalkerGenerator {

    private let model: MLModel
    private let maxSeqLen: Int
    private let hiddenSize: Int
    private let headDim: Int
    private let kvDimPerLayer: Int  // numKVHeads * headDim = 1024
    private let totalKVDim: Int     // kvDimPerLayer * numLayers = 28672
    private let ropeTheta: Float

    // Pre-allocated KV cache: [1, totalKVDim, 1, maxSeqLen]
    private var keyCache: MLMultiArray!
    private var valueCache: MLMultiArray!
    private var currentPos: Int = 0

    // Precomputed RoPE tables
    private var ropeCosTables: [[Float16]] = []  // [maxSeqLen][headDim]
    private var ropeSinTables: [[Float16]] = []

    init(model: MLModel, maxSeqLen: Int = 256, hiddenSize: Int = 1024,
         numLayers: Int = 28, numKVHeads: Int = 8, headDim: Int = 128,
         ropeTheta: Float = 1_000_000) {
        self.model = model
        self.maxSeqLen = maxSeqLen
        self.hiddenSize = hiddenSize
        self.headDim = headDim
        self.kvDimPerLayer = numKVHeads * headDim
        self.totalKVDim = kvDimPerLayer * numLayers
        self.ropeTheta = ropeTheta
        precomputeRoPE()
    }

    // MARK: - Cache Management

    func resetCache() {
        keyCache = makeZeroArray(shape: [1, totalKVDim, 1, maxSeqLen])
        valueCache = makeZeroArray(shape: [1, totalKVDim, 1, maxSeqLen])
        currentPos = 0
    }

    // MARK: - Forward Pass

    /// Run a single-token decode step.
    ///
    /// - Parameter embed: Input embedding [hiddenSize] as Float16 (will be reshaped to NCHW)
    /// - Returns: (logits [codecVocabSize] as Float, hiddenState [hiddenSize] as Float16)
    func forward(embed: [Float16]) throws -> (logits: [Float], hidden: [Float16]) {
        guard currentPos < maxSeqLen else {
            throw TalkerError.cacheFull
        }

        // Input: [1, hiddenSize, 1, 1] — NCHW single token
        let inputEmbeds = try MLMultiArray(
            shape: [1, NSNumber(value: hiddenSize), 1, 1], dataType: .float16)
        let embPtr = inputEmbeds.dataPointer.assumingMemoryBound(to: Float16.self)
        for i in 0..<hiddenSize { embPtr[i] = embed[i] }

        // RoPE cos/sin for current position: [1, headDim, 1]
        let ropeCos = try makeRoPEArray(position: currentPos, isCos: true)
        let ropeSin = try makeRoPEArray(position: currentPos, isCos: false)

        // Key padding mask: [1, maxSeqLen] — 0 for valid, -1e4 for masked
        let keyPaddingMask = try MLMultiArray(
            shape: [1, NSNumber(value: maxSeqLen)], dataType: .float16)
        let maskPtr = keyPaddingMask.dataPointer.assumingMemoryBound(to: Float16.self)
        let maskVal = Float16(-1e4)
        for i in 0..<maxSeqLen {
            maskPtr[i] = i <= currentPos ? Float16(0) : maskVal
        }

        // KV cache update mask: [1, maxSeqLen] — one-hot at currentPos
        let updateMask = try MLMultiArray(
            shape: [1, NSNumber(value: maxSeqLen)], dataType: .float16)
        let updatePtr = updateMask.dataPointer.assumingMemoryBound(to: Float16.self)
        memset(updateMask.dataPointer, 0, maxSeqLen * 2)
        updatePtr[currentPos] = Float16(1.0)

        // Build inputs
        let inputs: [String: MLFeatureValue] = [
            "input_embeds": MLFeatureValue(multiArray: inputEmbeds),
            "rope_cos": MLFeatureValue(multiArray: ropeCos),
            "rope_sin": MLFeatureValue(multiArray: ropeSin),
            "key_padding_mask": MLFeatureValue(multiArray: keyPaddingMask),
            "kv_cache_update_mask": MLFeatureValue(multiArray: updateMask),
            "key_cache": MLFeatureValue(multiArray: keyCache),
            "value_cache": MLFeatureValue(multiArray: valueCache),
        ]

        let provider = try MLDictionaryFeatureProvider(dictionary: inputs)
        let result = try model.prediction(from: provider)

        // Extract outputs — all in NCHW format
        let logitsArray = result.featureValue(for: "logits")!.multiArrayValue!      // [1, 3072, 1, 1]
        let hiddenArray = result.featureValue(for: "hidden_states")!.multiArrayValue! // [1, 1024, 1, 1]
        let keyCacheUpdates = result.featureValue(for: "key_cache_updates")!.multiArrayValue!
        let valueCacheUpdates = result.featureValue(for: "value_cache_updates")!.multiArrayValue!

        // Scatter-write cache updates at currentPos
        updateCache(updates: keyCacheUpdates, into: &keyCache!, at: currentPos)
        updateCache(updates: valueCacheUpdates, into: &valueCache!, at: currentPos)

        currentPos += 1

        // Extract logits [3072] and hidden [1024]
        let logits = extractNCHW(logitsArray, channels: 3072)
        let hidden = extractNCHWFloat16(hiddenArray, channels: hiddenSize)

        return (logits, hidden)
    }

    /// Run prefill: process multiple tokens sequentially.
    /// Returns logits and hidden state from the LAST token only.
    func prefill(embeds: [[Float16]]) throws -> (logits: [Float], hidden: [Float16]) {
        var lastLogits = [Float]()
        var lastHidden = [Float16]()
        for embed in embeds {
            (lastLogits, lastHidden) = try forward(embed: embed)
        }
        return (lastLogits, lastHidden)
    }

    // MARK: - Helpers

    private func precomputeRoPE() {
        // Half-split RoPE (MLX traditional=false): stride half (x[:d/2], x[d/2:])
        let halfDim = headDim / 2
        for pos in 0..<maxSeqLen {
            var cosVals = [Float16](repeating: 0, count: headDim)
            var sinVals = [Float16](repeating: 0, count: headDim)
            for i in 0..<halfDim {
                let freq = 1.0 / pow(ropeTheta, Float(2 * i) / Float(headDim))
                let angle = Float(pos) * freq
                let c = Float16(cos(angle))
                let s = Float16(sin(angle))
                // Half-split: first half gets cos/sin for first half, second half same
                cosVals[i] = c
                cosVals[i + halfDim] = c
                sinVals[i] = s
                sinVals[i + halfDim] = s
            }
            ropeCosTables.append(cosVals)
            ropeSinTables.append(sinVals)
        }
    }

    private func makeRoPEArray(position: Int, isCos: Bool) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: [1, NSNumber(value: headDim), 1], dataType: .float16)
        let ptr = array.dataPointer.assumingMemoryBound(to: Float16.self)
        let table = isCos ? ropeCosTables[position] : ropeSinTables[position]
        for i in 0..<headDim { ptr[i] = table[i] }
        return array
    }

    /// Scatter-write: copy update values into cache at the given position.
    /// updates: [1, totalKVDim, 1, 1], cache: [1, totalKVDim, 1, maxSeqLen]
    private func updateCache(updates: MLMultiArray, into cache: inout MLMultiArray, at position: Int) {
        let srcPtr = updates.dataPointer.assumingMemoryBound(to: Float16.self)
        let dstPtr = cache.dataPointer.assumingMemoryBound(to: Float16.self)
        // cache layout: [1, totalKVDim, 1, maxSeqLen] → flat index = ch * maxSeqLen + pos
        for ch in 0..<totalKVDim {
            dstPtr[ch * maxSeqLen + position] = srcPtr[ch]
        }
    }

    /// Extract Float32 values from NCHW array [1, channels, 1, 1].
    private func extractNCHW(_ array: MLMultiArray, channels: Int) -> [Float] {
        var result = [Float](repeating: 0, count: channels)
        if array.dataType == .float16 {
            let ptr = array.dataPointer.assumingMemoryBound(to: Float16.self)
            for i in 0..<channels { result[i] = Float(ptr[i]) }
        } else {
            let ptr = array.dataPointer.assumingMemoryBound(to: Float.self)
            for i in 0..<channels { result[i] = ptr[i] }
        }
        return result
    }

    /// Extract Float16 values from NCHW array [1, channels, 1, 1].
    private func extractNCHWFloat16(_ array: MLMultiArray, channels: Int) -> [Float16] {
        var result = [Float16](repeating: 0, count: channels)
        let ptr = array.dataPointer.assumingMemoryBound(to: Float16.self)
        for i in 0..<channels { result[i] = ptr[i] }
        return result
    }

    private func makeZeroArray(shape: [Int]) -> MLMultiArray {
        let nsShape = shape.map { NSNumber(value: $0) }
        let array = try! MLMultiArray(shape: nsShape, dataType: .float16)
        let count = shape.reduce(1, *)
        memset(array.dataPointer, 0, count * 2)
        return array
    }

    enum TalkerError: Error {
        case cacheFull
    }
}
#endif
