import Foundation
import MLX

// MARK: - Top-K Sampling

/// Sample from logits with temperature and top-k filtering.
/// - Parameters:
///   - logits: [B, vocabSize]
///   - temperature: sampling temperature (higher = more random)
///   - topK: number of top candidates to keep
/// - Returns: [B] sampled token indices
public func sampleTopK(
    logits: MLXArray,
    temperature: Float,
    topK: Int
) -> MLXArray {
    guard temperature > 0 else {
        return argMax(logits, axis: -1)
    }

    // Scale by temperature
    var scaled = logits / MLXArray(temperature)

    // Top-K filtering
    if topK > 0, topK < logits.shape[logits.ndim - 1] {
        // Get the k-th largest value as threshold
        let sorted = MLX.sorted(scaled, axis: -1)
        let vocabSize = scaled.shape[scaled.ndim - 1]
        let threshold = sorted[0..., (vocabSize - topK)..<(vocabSize - topK + 1)]
        // Mask out values below threshold
        scaled = MLX.where(scaled .>= threshold, scaled, MLXArray(Float(-1e9)))
    }

    // Sample from softmax distribution using Gumbel-max trick
    let gumbel = -log(-log(MLXRandom.uniform(low: 0.0, high: 1.0, scaled.shape)))
    return argMax(scaled + gumbel, axis: -1)
}

// MARK: - Top-K Sampling with Repetition Penalty

/// Sample from logits with temperature, top-k, and repetition penalty.
///
/// Applies per-token penalty to logits before temperature scaling:
/// - Positive logits for repeated tokens are divided by penalty
/// - Negative logits for repeated tokens are multiplied by penalty
///
/// This addresses the known Moshi repetition issue (kyutai-labs/delayed-streams-modeling#175).
///
/// - Parameters:
///   - logits: [B, vocabSize]
///   - temperature: sampling temperature
///   - topK: number of top candidates to keep
///   - pastTokens: recent token history for this codebook
///   - penalty: repetition penalty factor (1.0 = no penalty, >1.0 = penalize)
/// - Returns: [B] sampled token indices
public func sampleTopKWithPenalty(
    logits: MLXArray,
    temperature: Float,
    topK: Int,
    pastTokens: [Int32],
    penalty: Float
) -> MLXArray {
    guard penalty > 1.0, !pastTokens.isEmpty else {
        return sampleTopK(logits: logits, temperature: temperature, topK: topK)
    }

    // Build set of unique past tokens
    let uniquePast = Set(pastTokens)
    let vocabSize = logits.shape[logits.ndim - 1]

    // Create penalty mask: penalty for tokens in history, 1.0 for others
    var penaltyMask = [Float](repeating: 1.0, count: vocabSize)
    for tok in uniquePast {
        let idx = Int(tok)
        if idx >= 0, idx < vocabSize {
            penaltyMask[idx] = penalty
        }
    }
    let penaltyArr = MLXArray(penaltyMask).reshaped([1, vocabSize])

    // Apply penalty: divide positive logits, multiply negative logits
    let positive = logits .> MLXArray(Float(0))
    let penalized = MLX.where(positive, logits / penaltyArr, logits * penaltyArr)

    return sampleTopK(logits: penalized, temperature: temperature, topK: topK)
}
