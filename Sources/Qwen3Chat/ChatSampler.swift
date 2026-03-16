import Foundation

/// Standalone token sampler for text generation.
///
/// Supports temperature scaling, top-K filtering, top-P (nucleus) filtering,
/// and repetition penalty. Used by CoreMLGenerator and available for testing.
enum ChatSampler {

    /// Sample a token from logits using the given sampling config.
    ///
    /// - Parameters:
    ///   - logits: Raw logits array of size vocab_size
    ///   - config: Sampling parameters (temperature, topK, topP, repetitionPenalty)
    ///   - previousTokens: Recently generated tokens for repetition penalty
    /// - Returns: Sampled token index
    static func sample(
        logits: [Float],
        config: ChatSamplingConfig,
        previousTokens: [Int] = []
    ) -> Int {
        var logits = logits

        // Repetition penalty
        if config.repetitionPenalty > 1.0 {
            let seen = Set(previousTokens.suffix(64))
            for tokenId in seen {
                if tokenId < logits.count {
                    if logits[tokenId] > 0 {
                        logits[tokenId] /= config.repetitionPenalty
                    } else {
                        logits[tokenId] *= config.repetitionPenalty
                    }
                }
            }
        }

        // Greedy (argmax) when temperature is 0
        if config.temperature <= 0 {
            var maxIdx = 0
            var maxVal = logits[0]
            for i in 1..<logits.count {
                if logits[i] > maxVal {
                    maxVal = logits[i]
                    maxIdx = i
                }
            }
            return maxIdx
        }

        // Temperature scaling
        if config.temperature != 1.0 {
            for i in 0..<logits.count {
                logits[i] /= config.temperature
            }
        }

        // Softmax
        let maxLogit = logits.max() ?? 0
        var probs = logits.map { exp($0 - maxLogit) }
        let sum = probs.reduce(0, +)
        probs = probs.map { $0 / sum }

        // Top-K filtering
        if config.topK > 0 && config.topK < probs.count {
            let indexed = probs.enumerated().sorted { $0.element > $1.element }
            let topK = Array(indexed.prefix(config.topK))
            var filtered = [Float](repeating: 0, count: probs.count)
            for (idx, prob) in topK {
                filtered[idx] = prob
            }
            let filteredSum = filtered.reduce(0, +)
            if filteredSum > 0 {
                probs = filtered.map { $0 / filteredSum }
            }
        }

        // Top-P (nucleus) filtering
        if config.topP < 1.0 {
            let indexed = probs.enumerated().sorted { $0.element > $1.element }
            var cumProb: Float = 0
            var mask = [Bool](repeating: false, count: probs.count)
            for (idx, prob) in indexed {
                cumProb += prob
                mask[idx] = true
                if cumProb >= config.topP { break }
            }
            for i in 0..<probs.count {
                if !mask[i] { probs[i] = 0 }
            }
            let filteredSum = probs.reduce(0, +)
            if filteredSum > 0 {
                probs = probs.map { $0 / filteredSum }
            }
        }

        // Sample from distribution
        let r = Float.random(in: 0..<1)
        var cumulative: Float = 0
        for (i, p) in probs.enumerated() {
            cumulative += p
            if cumulative >= r {
                return i
            }
        }
        return probs.count - 1
    }
}
