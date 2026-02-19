import MLX

// MARK: - TopP Sampler

public struct TopPSampler {
    public let temperature: Float
    public let topP: Float

    public init(temperature: Float = 0.9, topP: Float = 0.8) {
        self.temperature = temperature
        self.topP = topP
    }

    public func sample(_ logits: MLXArray) -> MLXArray {
        // Apply temperature
        var logits = logits
        if temperature > 0 && temperature != 1.0 {
            logits = logits / MLXArray(temperature)
        }

        // Apply softmax to get probabilities
        let probs = softmax(logits, axis: -1)

        // Sort probabilities in descending order
        let sortedIndices = argSort(probs * MLXArray(-1.0), axis: -1) // descending via negation
        let sortedProbs = takeAlong(probs, sortedIndices, axis: -1)

        // Compute cumulative probabilities
        let cumProbs = cumsum(sortedProbs, axis: -1)

        // Create mask: keep tokens where cumulative prob < topP (plus the first one that exceeds it)
        let mask = cumProbs - sortedProbs .< MLXArray(topP)

        // Zero out probabilities beyond the top-p threshold
        let filteredProbs = sortedProbs * mask.asType(sortedProbs.dtype)

        // Renormalize
        let sumProbs = filteredProbs.sum(axis: -1, keepDims: true)
        let normalizedProbs = filteredProbs / maximum(sumProbs, MLXArray(Float(1e-10)))

        // Sample from the filtered distribution
        let sampledIndex = categorical(normalizedProbs)

        // Map back to original token indices
        return takeAlong(sortedIndices, sampledIndex.expandedDimensions(axis: -1), axis: -1).squeezed(axis: -1)
    }
}
