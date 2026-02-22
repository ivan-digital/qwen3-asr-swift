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
