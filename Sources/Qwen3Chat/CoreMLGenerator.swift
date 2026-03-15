import CoreML
import Foundation

/// CoreML-based autoregressive text generator for Qwen3 chat models.
///
/// Design patterns:
/// - **Dual model**: Prefill (batch prompt) + Decode (single token, latency-optimized)
/// - **Prompt caching**: Snapshot KV state after system prompt, restore per turn
/// - **Adaptive measurement**: Track tokens/sec for downstream buffering decisions
/// - **`@unchecked Sendable`**: Safe via ownership isolation (single-task use)
final class CoreMLGenerator: @unchecked Sendable {

    // MARK: - Dual Model Architecture

    /// Prefill model: processes variable-length prompt in one pass.
    /// Optimized for throughput (batch tokens).
    private let prefillModel: MLModel

    /// Decode model: processes single token per step.
    /// Optimized for latency (fast per-step).
    /// Falls back to prefillModel if not provided.
    private let decodeModel: MLModel

    private let config: Qwen3ChatConfig

    // MARK: - KV Cache State

    /// KV cache: [layer][0=key, 1=value] = MLMultiArray
    private var kvCache: [[MLMultiArray]] = []
    private var currentSeqLen: Int = 0

    // MARK: - Prompt Cache

    /// Cached KV state after processing system prompt.
    /// Restored at the start of each conversation turn to avoid re-prefilling.
    private var promptCacheKV: [[MLMultiArray]]?
    private var promptCacheSeqLen: Int = 0

    // MARK: - Adaptive Speed Measurement

    /// Measured generation speed for adaptive buffering decisions.
    struct GenerationMetrics: Sendable {
        var prefillTimeMs: Double = 0
        var prefillTokens: Int = 0
        var decodeTimeMs: Double = 0
        var decodeTokens: Int = 0

        var tokensPerSecond: Double {
            guard decodeTimeMs > 0 else { return 0 }
            return Double(decodeTokens) / (decodeTimeMs / 1000.0)
        }

        var prefillTokensPerSecond: Double {
            guard prefillTimeMs > 0 else { return 0 }
            return Double(prefillTokens) / (prefillTimeMs / 1000.0)
        }

        var msPerToken: Double {
            guard decodeTokens > 0 else { return 0 }
            return decodeTimeMs / Double(decodeTokens)
        }
    }

    /// Latest generation metrics (read after generate completes).
    private(set) var metrics = GenerationMetrics()

    // MARK: - Init

    /// Initialize with separate prefill and decode models.
    ///
    /// If only one model is available, pass it as both parameters.
    /// Separate models allows CoreML to optimize each for its use case.
    init(prefillModel: MLModel, decodeModel: MLModel? = nil, config: Qwen3ChatConfig) {
        self.prefillModel = prefillModel
        self.decodeModel = decodeModel ?? prefillModel
        self.config = config
    }

    /// Convenience init with a single model for both prefill and decode.
    convenience init(model: MLModel, config: Qwen3ChatConfig) {
        self.init(prefillModel: model, config: config)
    }

    // MARK: - Cache Management

    /// Reset the KV cache for a new generation.
    func resetCache() {
        kvCache = makeEmptyKVCache()
        currentSeqLen = 0
    }

    /// Snapshot current KV state as prompt cache.
    ///
    /// Call after prefilling the system prompt. Subsequent turns restore
    /// from this snapshot instead of re-processing the system prompt (~300ms saved).
    func snapshotPromptCache() {
        promptCacheKV = kvCache.map { layer in
            layer.map { copyMultiArray($0) }
        }
        promptCacheSeqLen = currentSeqLen
    }

    /// Restore KV state from prompt cache.
    ///
    /// Returns true if cache was restored, false if no cache exists.
    @discardableResult
    func restorePromptCache() -> Bool {
        guard let cached = promptCacheKV else { return false }
        kvCache = cached.map { layer in
            layer.map { copyMultiArray($0) }
        }
        currentSeqLen = promptCacheSeqLen
        return true
    }

    /// Clear the prompt cache.
    func clearPromptCache() {
        promptCacheKV = nil
        promptCacheSeqLen = 0
    }

    // MARK: - Forward Pass

    /// Prefill: process multiple tokens at once (prompt processing).
    func prefill(tokenIds: [Int]) throws -> [Float] {
        let start = CFAbsoluteTimeGetCurrent()
        let logits = try forward(tokenIds: tokenIds, model: prefillModel)
        let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000
        metrics.prefillTimeMs += elapsed
        metrics.prefillTokens += tokenIds.count
        return logits
    }

    /// Decode: process a single token (autoregressive step).
    func decode(tokenId: Int) throws -> [Float] {
        let start = CFAbsoluteTimeGetCurrent()
        let logits = try forward(tokenIds: [tokenId], model: decodeModel)
        let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000
        metrics.decodeTimeMs += elapsed
        metrics.decodeTokens += 1
        return logits
    }

    /// Reset metrics before a new generation.
    func resetMetrics() {
        metrics = GenerationMetrics()
    }

    /// Core forward pass implementation.
    private func forward(tokenIds: [Int], model: MLModel) throws -> [Float] {
        let seqLen = tokenIds.count

        // Build input_ids [1, seq_len]
        let inputIds = try MLMultiArray(shape: [1, seqLen as NSNumber], dataType: .int32)
        for i in 0..<seqLen {
            inputIds[i] = NSNumber(value: Int32(tokenIds[i]))
        }

        // Build position_ids [1, seq_len]
        let positionIds = try MLMultiArray(shape: [1, seqLen as NSNumber], dataType: .int32)
        for i in 0..<seqLen {
            positionIds[i] = NSNumber(value: Int32(currentSeqLen + i))
        }

        // Build causal mask [1, 1, seq_len, kv_len + seq_len]
        // Use actual KV cache size (may differ from currentSeqLen due to
        // CoreML lower_bound=1 padding on first call)
        let kvLen = kvCache[0][0].shape[2].intValue
        let totalLen = kvLen + seqLen
        // Padding slots = kvLen - currentSeqLen (zero-filled, must be blocked)
        let paddingLen = kvLen - currentSeqLen
        let causalMask = try makeCausalMask(queryLen: seqLen, keyLen: totalLen, paddingLen: paddingLen)

        // Build feature provider with KV cache
        var inputDict: [String: MLFeatureValue] = [
            "input_ids": MLFeatureValue(multiArray: inputIds),
            "position_ids": MLFeatureValue(multiArray: positionIds),
            "causal_mask": MLFeatureValue(multiArray: causalMask),
        ]

        for i in 0..<config.numHiddenLayers {
            inputDict["layer_\(i)_key_cache"] = MLFeatureValue(multiArray: kvCache[i][0])
            inputDict["layer_\(i)_value_cache"] = MLFeatureValue(multiArray: kvCache[i][1])
        }

        let input = try MLDictionaryFeatureProvider(dictionary: inputDict)
        let output = try model.prediction(from: input)

        // Extract logits for last position
        guard let logitsArray = output.featureValue(for: "logits")?.multiArrayValue else {
            throw ChatModelError.inferenceFailed("Missing logits output")
        }

        // logits shape: [1, seq_len, vocab_size] — extract last position
        let vocabSize = config.vocabSize
        var logits = [Float](repeating: 0, count: vocabSize)
        let lastPos = seqLen - 1

        for v in 0..<vocabSize {
            let idx = lastPos * vocabSize + v
            logits[v] = Float(truncating: logitsArray[idx])
        }

        // Update KV cache from outputs
        for i in 0..<config.numHiddenLayers {
            if let newK = output.featureValue(for: "layer_\(i)_key_cache_out")?.multiArrayValue {
                kvCache[i][0] = newK
            }
            if let newV = output.featureValue(for: "layer_\(i)_value_cache_out")?.multiArrayValue {
                kvCache[i][1] = newV
            }
        }

        currentSeqLen += seqLen
        return logits
    }

    // MARK: - Sampling

    /// Sample a token from logits using temperature, top-k, top-p.
    func sample(logits: [Float], config: ChatSamplingConfig, previousTokens: [Int] = []) -> Int {
        ChatSampler.sample(logits: logits, config: config, previousTokens: previousTokens)
    }

    // MARK: - Helpers

    /// Build a 4D causal mask [1, 1, queryLen, keyLen].
    ///
    /// Values: 0 = attend, -inf = block.
    /// Blocks both future positions and zero-padded KV cache slots.
    /// `paddingLen` is the number of leading KV positions that are zero-filled
    /// padding (from CoreML's lower_bound=1 requirement) and must be blocked.
    private func makeCausalMask(queryLen: Int, keyLen: Int, paddingLen: Int = 0) throws -> MLMultiArray {
        let mask = try MLMultiArray(
            shape: [1, 1, queryLen as NSNumber, keyLen as NSNumber],
            dataType: .float16
        )
        let pastLen = keyLen - queryLen
        let ptr = mask.dataPointer.bindMemory(to: Float16.self, capacity: queryLen * keyLen)
        let minusInf = Float16(-65504.0)
        for q in 0..<queryLen {
            for k in 0..<keyLen {
                let idx = q * keyLen + k
                // Block: padding positions (k < paddingLen) and future positions (k > pastLen + q)
                if k < paddingLen || k > pastLen + q {
                    ptr[idx] = minusInf
                } else {
                    ptr[idx] = Float16(0.0)
                }
            }
        }
        return mask
    }

    private func makeEmptyKVCache() -> [[MLMultiArray]] {
        // CoreML model requires min seq_len=1 for KV cache dimensions.
        // Initialize with 1 zero-filled slot; currentSeqLen starts at 0
        // so position_ids remain correct.
        // IMPORTANT: MLMultiArray is NOT zero-initialized — must zero-fill explicitly.
        (0..<config.numHiddenLayers).map { _ in
            let shape = [1, config.numKeyValueHeads, 1, config.headDim] as [NSNumber]
            let count = config.numKeyValueHeads * 1 * config.headDim
            let emptyK = try! MLMultiArray(shape: shape, dataType: .float16)
            emptyK.dataPointer.bindMemory(to: UInt8.self, capacity: count * 2)
                .update(repeating: 0, count: count * 2)
            let emptyV = try! MLMultiArray(shape: shape, dataType: .float16)
            emptyV.dataPointer.bindMemory(to: UInt8.self, capacity: count * 2)
                .update(repeating: 0, count: count * 2)
            return [emptyK, emptyV]
        }
    }

    private func copyMultiArray(_ source: MLMultiArray) -> MLMultiArray {
        let copy = try! MLMultiArray(shape: source.shape, dataType: source.dataType)
        let count = source.count
        let srcPtr = source.dataPointer.bindMemory(to: UInt8.self, capacity: count * 2)
        let dstPtr = copy.dataPointer.bindMemory(to: UInt8.self, capacity: count * 2)
        dstPtr.update(from: srcPtr, count: count * 2)
        return copy
    }
}
