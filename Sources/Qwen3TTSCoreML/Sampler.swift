#if canImport(CoreML)
import CoreML
import Foundation

/// Pure-Swift token sampler for CoreML TTS inference (no MLX dependency).
///
/// Supports temperature, top-k, repetition penalty, and Gumbel-max sampling.
enum TTSSampler {

    /// Sample a single token from logits.
    ///
    /// - Parameters:
    ///   - logits: Float32 array of shape [vocabSize]
    ///   - temperature: Sampling temperature (0 = greedy)
    ///   - topK: Top-k filtering (0 = disabled)
    ///   - repetitionPenalty: Penalty for repeated tokens
    ///   - generatedTokens: Previously generated tokens (for rep penalty)
    ///   - suppressRange: Range of token IDs to suppress (set to -inf)
    ///   - eosTokenId: EOS token to preserve through filtering
    /// - Returns: Sampled token ID
    static func sample(
        logits: [Float],
        temperature: Float = 0.9,
        topK: Int = 50,
        repetitionPenalty: Float = 1.0,
        generatedTokens: [Int32] = [],
        suppressRange: (Int, Int)? = nil,
        eosTokenId: Int? = nil,
        eosLogitBias: Float = 0.0
    ) -> Int32 {
        var logits = logits
        let vocabSize = logits.count

        // 1. Token suppression
        if let (start, end) = suppressRange, start < end {
            for i in start..<min(end, vocabSize) {
                if let eos = eosTokenId, i == eos { continue }
                logits[i] = -1e9
            }
        }

        // 2. Repetition penalty
        if repetitionPenalty != 1.0, !generatedTokens.isEmpty {
            let unique = Set(generatedTokens)
            for token in unique {
                let idx = Int(token)
                guard idx >= 0, idx < vocabSize else { continue }
                if logits[idx] < 0 {
                    logits[idx] *= repetitionPenalty
                } else {
                    logits[idx] /= repetitionPenalty
                }
            }
        }

        // 3. Apply EOS bias BEFORE temperature (so it scales with temp)
        if let eos = eosTokenId, eos >= 0, eos < vocabSize, eosLogitBias != 0 {
            logits[eos] += eosLogitBias
        }

        // 4. Greedy
        if temperature <= 0 {
            return Int32(logits.enumerated().max(by: { $0.element < $1.element })!.offset)
        }

        // 5. Temperature
        for i in 0..<vocabSize { logits[i] /= temperature }

        // 6. Top-k
        if topK > 0, topK < vocabSize {
            let sorted = logits.sorted()
            let threshold = sorted[vocabSize - topK]
            for i in 0..<vocabSize {
                if logits[i] < threshold { logits[i] = -1e9 }
            }
        }

        // 8. Gumbel-max sampling: argmax(logits + Gumbel noise)
        var maxVal: Float = -Float.infinity
        var maxIdx = 0
        for i in 0..<vocabSize {
            // Gumbel noise: -log(-log(U)) where U ~ Uniform(0, 1)
            let u = Float.random(in: Float.leastNonzeroMagnitude..<1.0)
            let gumbel = -log(-log(u))
            let perturbed = logits[i] + gumbel
            if perturbed > maxVal {
                maxVal = perturbed
                maxIdx = i
            }
        }
        return Int32(maxIdx)
    }

    /// Extract Float32 logits from an MLMultiArray.
    /// Handles the last position of shape [1, seqLen, vocabSize].
    static func extractLogits(from array: MLMultiArray, vocabSize: Int) -> [Float] {
        let shape = (0..<array.shape.count).map { array.shape[$0].intValue }
        let seqLen = shape.count >= 2 ? shape[shape.count - 2] : 1
        let lastPosOffset = (seqLen - 1) * vocabSize

        var result = [Float](repeating: 0, count: vocabSize)
        if array.dataType == .float16 {
            let ptr = array.dataPointer.assumingMemoryBound(to: Float16.self)
            for i in 0..<vocabSize {
                result[i] = Float(ptr[lastPosOffset + i])
            }
        } else {
            let ptr = array.dataPointer.assumingMemoryBound(to: Float.self)
            for i in 0..<vocabSize {
                result[i] = ptr[lastPosOffset + i]
            }
        }
        return result
    }
}
#endif
