#if canImport(CoreML)
import CoreML
import Foundation

/// CoreML-based Code Predictor for Qwen3-TTS.
///
/// Predicts 15 residual codebooks autoregressively: runs 15 sequential forward passes
/// (prefill of 2 tokens + 14 single-token decode steps). The CoreML model outputs
/// normalized hidden states; per-group lm_head weights are applied in Swift.
final class CodePredictorCoreML {

    private let model: MLModel
    private let numLayers: Int = 5
    private let numKVHeads: Int = 8
    private let headDim: Int = 128
    private let hiddenSize: Int = 1024
    private let cpVocabSize: Int = 2048
    private let numGroups: Int = 15

    // Per-group lm_head weights [15][2048 * 1024] in row-major (FP16)
    private var lmHeadWeights: [[Float16]] = []
    // Per-group codec embedding tables [16][2048 * 1024] in row-major (FP16)
    private var codecEmbeddings: [[Float16]] = []

    init(model: MLModel) {
        self.model = model
    }

    /// Load weights that are applied in Swift (not baked into CoreML model).
    func loadWeights(lmHeads: [[Float16]], codecEmbeddings: [[Float16]]) {
        self.lmHeadWeights = lmHeads
        self.codecEmbeddings = codecEmbeddings
    }

    /// Predict 15 residual codebook tokens for one timestep (autoregressive).
    func predict(
        hiddenState: MLMultiArray,
        firstCodeEmbed: MLMultiArray,
        temperature: Float = 0.6,
        topK: Int = 50,
        repetitionPenalty: Float = 1.3
    ) throws -> [Int32] {
        // Step 0: Prefill with [hidden_state, code0_embed] — 2 tokens
        let prefillEmbeds = try concatenateEmbedsFloat32(hiddenState, firstCodeEmbed)

        let positionIds = try MLMultiArray(shape: [1, 2], dataType: .int32)
        let posPtr = positionIds.dataPointer.assumingMemoryBound(to: Int32.self)
        posPtr[0] = 0; posPtr[1] = 1

        // Causal mask: [1, 1, 2, 3] — kv_len=1 + seq_len=2
        let causalMask = try MLMultiArray(shape: [1, 1, 2, 3], dataType: .float32)
        memset(causalMask.dataPointer, 0, 6 * MemoryLayout<Float>.size)
        let maskPtr = causalMask.dataPointer.assumingMemoryBound(to: Float.self)
        maskPtr[2] = -1e4  // q=0 blocks k=2 (future)

        // Empty KV caches (FP32)
        var inputs: [String: MLFeatureValue] = [
            "input_embeds": MLFeatureValue(multiArray: prefillEmbeds),
            "position_ids": MLFeatureValue(multiArray: positionIds),
            "causal_mask": MLFeatureValue(multiArray: causalMask),
        ]
        for i in 0..<numLayers {
            let kvSize = numKVHeads * 1 * headDim
            let emptyK = try MLMultiArray(shape: [1, NSNumber(value: numKVHeads), 1, NSNumber(value: headDim)], dataType: .float32)
            let emptyV = try MLMultiArray(shape: [1, NSNumber(value: numKVHeads), 1, NSNumber(value: headDim)], dataType: .float32)
            memset(emptyK.dataPointer, 0, kvSize * MemoryLayout<Float>.size)
            memset(emptyV.dataPointer, 0, kvSize * MemoryLayout<Float>.size)
            inputs["layer_\(i)_key_cache"] = MLFeatureValue(multiArray: emptyK)
            inputs["layer_\(i)_value_cache"] = MLFeatureValue(multiArray: emptyV)
        }

        let provider = try MLDictionaryFeatureProvider(dictionary: inputs)
        let result = try model.prediction(from: provider)

        // Extract hidden_states [1, 1, 1024] and apply lm_head for group 0
        let hiddenOut = result.featureValue(for: "hidden_states")!.multiArrayValue!
        let hiddenVec = extractHiddenVector(hiddenOut)
        let logits0 = applyLMHead(hidden: hiddenVec, group: 0)
        let token0 = TTSSampler.sample(
            logits: logits0, temperature: temperature, topK: topK,
            repetitionPenalty: repetitionPenalty)

        var tokens = [token0]

        // Extract KV caches from prefill
        var kvCaches: [(MLMultiArray, MLMultiArray)] = []
        for i in 0..<numLayers {
            let kOut = result.featureValue(for: "layer_\(i)_key_cache_out")!.multiArrayValue!
            let vOut = result.featureValue(for: "layer_\(i)_value_cache_out")!.multiArrayValue!
            kvCaches.append((kOut, vOut))
        }

        // Steps 1..14: decode one group at a time
        for group in 1..<numGroups {
            let prevToken = tokens[group - 1]

            // Build input embedding: group_embedding[group] + codec_embedding[group-1][prevToken]
            let stepEmbed = try makeGroupStepEmbed(group: group, token: prevToken)

            // Position ID for this step: 2 (prefill was 0,1) + group - 1
            let stepPos = try MLMultiArray(shape: [1, 1], dataType: .int32)
            stepPos.dataPointer.assumingMemoryBound(to: Int32.self)[0] = Int32(1 + group)

            // Causal mask: [1, 1, 1, kvLen+1] — attend to all cached + self
            let kvLen = kvCaches[0].0.shape[2].intValue
            let totalLen = kvLen + 1
            let stepMask = try MLMultiArray(shape: [1, 1, 1, NSNumber(value: totalLen)], dataType: .float32)
            memset(stepMask.dataPointer, 0, totalLen * MemoryLayout<Float>.size)

            var stepInputs: [String: MLFeatureValue] = [
                "input_embeds": MLFeatureValue(multiArray: stepEmbed),
                "position_ids": MLFeatureValue(multiArray: stepPos),
                "causal_mask": MLFeatureValue(multiArray: stepMask),
            ]
            for i in 0..<numLayers {
                stepInputs["layer_\(i)_key_cache"] = MLFeatureValue(multiArray: kvCaches[i].0)
                stepInputs["layer_\(i)_value_cache"] = MLFeatureValue(multiArray: kvCaches[i].1)
            }

            let stepProvider = try MLDictionaryFeatureProvider(dictionary: stepInputs)
            let stepResult = try model.prediction(from: stepProvider)

            let stepHidden = stepResult.featureValue(for: "hidden_states")!.multiArrayValue!
            let stepVec = extractHiddenVector(stepHidden)
            let logits = applyLMHead(hidden: stepVec, group: group)
            let token = TTSSampler.sample(
                logits: logits, temperature: temperature, topK: topK,
                repetitionPenalty: repetitionPenalty)
            tokens.append(token)

            // Update KV caches
            for i in 0..<numLayers {
                kvCaches[i] = (
                    stepResult.featureValue(for: "layer_\(i)_key_cache_out")!.multiArrayValue!,
                    stepResult.featureValue(for: "layer_\(i)_value_cache_out")!.multiArrayValue!
                )
            }
        }

        return tokens
    }

    // MARK: - Helpers

    /// Concatenate two FP16 embeddings into a single FP32 array for the FP32 CP model.
    private func concatenateEmbedsFloat32(_ a: MLMultiArray, _ b: MLMultiArray) throws -> MLMultiArray {
        let result = try MLMultiArray(shape: [1, 2, NSNumber(value: hiddenSize)], dataType: .float32)
        let srcA = a.dataPointer.assumingMemoryBound(to: Float16.self)
        let srcB = b.dataPointer.assumingMemoryBound(to: Float16.self)
        let dst = result.dataPointer.assumingMemoryBound(to: Float.self)
        for i in 0..<hiddenSize { dst[i] = Float(srcA[i]) }
        for i in 0..<hiddenSize { dst[hiddenSize + i] = Float(srcB[i]) }
        return result
    }

    /// Build step embedding for autoregressive decode: group_embed + codec_embed (FP32).
    private func makeGroupStepEmbed(group: Int, token: Int32) throws -> MLMultiArray {
        let result = try MLMultiArray(shape: [1, 1, NSNumber(value: hiddenSize)], dataType: .float32)
        let dst = result.dataPointer.assumingMemoryBound(to: Float.self)

        // Group embedding (from codecEmbeddings, group index is 0-based for the CP groups)
        // CP group embedding table is loaded separately; for now use codec embedding for the group
        let tokenOffset = Int(token) * hiddenSize
        if group < codecEmbeddings.count && tokenOffset + hiddenSize <= codecEmbeddings[group].count {
            let src = codecEmbeddings[group]
            for i in 0..<hiddenSize {
                dst[i] = Float(src[tokenOffset + i])
            }
        } else {
            memset(dst, 0, hiddenSize * MemoryLayout<Float>.size)
        }

        return result
    }

    /// Extract [hiddenSize] float vector from hidden_states MLMultiArray.
    private func extractHiddenVector(_ array: MLMultiArray) -> [Float] {
        var result = [Float](repeating: 0, count: hiddenSize)
        if array.dataType == .float16 {
            let ptr = array.dataPointer.assumingMemoryBound(to: Float16.self)
            for i in 0..<hiddenSize { result[i] = Float(ptr[i]) }
        } else {
            let ptr = array.dataPointer.assumingMemoryBound(to: Float.self)
            for i in 0..<hiddenSize { result[i] = ptr[i] }
        }
        return result
    }

    /// Apply lm_head weight matrix for a specific group: logits = W @ hidden.
    /// W is [vocabSize, hiddenSize] row-major FP16.
    private func applyLMHead(hidden: [Float], group: Int) -> [Float] {
        guard group < lmHeadWeights.count else {
            return [Float](repeating: 0, count: cpVocabSize)
        }
        let w = lmHeadWeights[group]
        var logits = [Float](repeating: 0, count: cpVocabSize)
        for r in 0..<cpVocabSize {
            var sum: Float = 0
            let rowOffset = r * hiddenSize
            for c in 0..<hiddenSize {
                sum += Float(w[rowOffset + c]) * hidden[c]
            }
            logits[r] = sum
        }
        return logits
    }
}
#endif
