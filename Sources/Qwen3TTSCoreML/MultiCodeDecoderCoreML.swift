#if canImport(CoreML)
import CoreML
import Foundation

/// CoreML MultiCodeDecoder: autoregressive CB1-15 prediction.
/// 5-layer transformer with 15 lm_heads, scatter-write KV cache.
final class MultiCodeDecoderCoreML {
    private let model: MLModel
    private let maxSeqLen = 16
    private let totalKVDim = 5120  // 5 layers * 8 KV heads * 128 head_dim
    private let numGroups = 15

    init(model: MLModel) { self.model = model }

    /// Predict 15 residual codebook tokens autoregressively.
    /// - Parameters:
    ///   - hiddenState: [1, 1024, 1, 1] from CodeDecoder
    ///   - cb0Token: First codebook token
    ///   - codeEmbedder: For embedding CB0
    ///   - multiCodeEmbedder: For embedding CB1-14 tokens
    func predict(
        hiddenState: MLMultiArray,
        cb0Token: Int32,
        codeEmbedder: CodeEmbedderModel,
        multiCodeEmbedder: MultiCodeEmbedderModel,
        temperature: Float = 0.9,
        topK: Int = 50
    ) throws -> [Int32] {
        // Init KV cache
        var keyCache = makeZeros(shape: [1, totalKVDim, 1, maxSeqLen])
        var valueCache = makeZeros(shape: [1, totalKVDim, 1, maxSeqLen])

        // Position 0: feed hidden_states from CodeDecoder
        var (_, kc, vc) = try step(
            embed: ensureNCHW(hiddenState, channels: 1024),
            position: 0, keyCache: keyCache, valueCache: valueCache)
        keyCache = kc; valueCache = vc

        // Position 1: feed CodeEmbedder(CB0) → read lm_head[0] → CB1
        let cb0Embed = ensureNCHW(try codeEmbedder.embed(Int(cb0Token)), channels: 1024)
        var (logits, kc2, vc2) = try step(embed: cb0Embed, position: 1, keyCache: keyCache, valueCache: valueCache)
        keyCache = kc2; valueCache = vc2

        // Sample CB1 from lm_head[0]
        let cb1Logits = extractGroupLogits(logits, group: 0)
        var prevToken = TTSSampler.sample(logits: cb1Logits, temperature: temperature, topK: topK)
        var tokens = [prevToken]

        // Positions 2-15: autoregressive CB2-CB15
        for cbStep in 1..<numGroups {
            let embed = ensureNCHW(
                try multiCodeEmbedder.embed(codebookIdx: cbStep - 1, tokenId: Int(prevToken)),
                channels: 1024)
            let pos = cbStep + 1
            (logits, kc, vc) = try step(embed: embed, position: pos, keyCache: keyCache, valueCache: valueCache)
            keyCache = kc; valueCache = vc

            let groupLogits = extractGroupLogits(logits, group: cbStep)
            prevToken = TTSSampler.sample(logits: groupLogits, temperature: temperature, topK: topK)
            tokens.append(prevToken)
        }

        return tokens
    }

    // MARK: - Helpers

    private func step(embed: MLMultiArray, position: Int, keyCache: MLMultiArray, valueCache: MLMultiArray)
        throws -> (logits: MLMultiArray, keyCache: MLMultiArray, valueCache: MLMultiArray) {
        let cacheLen = try MLMultiArray(shape: [1], dataType: .int32)
        cacheLen.dataPointer.assumingMemoryBound(to: Int32.self)[0] = Int32(position)

        let keyMask = try MLMultiArray(shape: [1, NSNumber(value: maxSeqLen)], dataType: .float16)
        let maskPtr = keyMask.dataPointer.assumingMemoryBound(to: Float16.self)
        for i in 0..<maxSeqLen { maskPtr[i] = i <= position ? Float16(0) : Float16(-1e4) }

        let updateMask = try MLMultiArray(shape: [1, NSNumber(value: maxSeqLen)], dataType: .float16)
        memset(updateMask.dataPointer, 0, maxSeqLen * 2)
        updateMask.dataPointer.assumingMemoryBound(to: Float16.self)[position] = Float16(1.0)

        let inputs: [String: MLFeatureValue] = [
            "input_embeds": MLFeatureValue(multiArray: embed),
            "cache_length": MLFeatureValue(multiArray: cacheLen),
            "key_cache": MLFeatureValue(multiArray: keyCache),
            "key_padding_mask": MLFeatureValue(multiArray: keyMask),
            "kv_cache_update_mask": MLFeatureValue(multiArray: updateMask),
            "value_cache": MLFeatureValue(multiArray: valueCache),
        ]

        let result = try model.prediction(from: try MLDictionaryFeatureProvider(dictionary: inputs))
        return (
            result.featureValue(for: "all_logits")!.multiArrayValue!,
            result.featureValue(for: "new_key_cache")!.multiArrayValue!,
            result.featureValue(for: "new_value_cache")!.multiArrayValue!
        )
    }

    /// Extract logits for a specific group from [1, 15, 2048] (stride-safe).
    private func extractGroupLogits(_ array: MLMultiArray, group: Int) -> [Float] {
        let vocabSize = 2048
        var result = [Float](repeating: 0, count: vocabSize)
        let ndim = array.shape.count
        for i in 0..<vocabSize {
            var idx = [NSNumber](repeating: 0, count: ndim)
            if ndim == 3 { idx[1] = group as NSNumber; idx[2] = i as NSNumber }
            else if ndim == 2 { idx[0] = group as NSNumber; idx[1] = i as NSNumber }
            result[i] = array[idx].floatValue
        }
        return result
    }

    private func makeZeros(shape: [Int]) -> MLMultiArray {
        let a = try! MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .float16)
        memset(a.dataPointer, 0, shape.reduce(1, *) * 2)
        return a
    }

    /// Fast contiguous copy for 4D FP16 [1, C, 1, S] arrays with non-unit strides.

}
#endif
