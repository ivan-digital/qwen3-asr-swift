import Foundation
import MLX

/// Sampling configuration for TTS generation
public struct SamplingConfig: Sendable {
    public var temperature: Float = 0.9
    public var topK: Int = 50
    public var topP: Float = 1.0
    public var minP: Float = 0.0
    public var repetitionPenalty: Float = 1.05
    public var maxTokens: Int = 4096
    /// Additive bias to EOS logit (applied after temperature, before sampling).
    /// Positive values make the model more likely to stop. Useful for CustomVoice
    /// models that under-score EOS for certain languages.
    public var eosLogitBias: Float = 0.0

    public init() {}

    public init(temperature: Float, topK: Int = 50, topP: Float = 1.0, repetitionPenalty: Float = 1.05, maxTokens: Int = 4096, eosLogitBias: Float = 0.0) {
        self.temperature = temperature
        self.topK = topK
        self.topP = topP
        self.repetitionPenalty = repetitionPenalty
        self.maxTokens = maxTokens
        self.eosLogitBias = eosLogitBias
    }

    public static var `default`: SamplingConfig { SamplingConfig() }
    public static var greedy: SamplingConfig { SamplingConfig(temperature: 0, topK: 1) }
}

/// Sample a token from logits using temperature, top-k, top-p, and repetition penalty.
/// EOS protection: the EOS logit is saved before top-k/top-p filtering and restored after.
/// Uses top-k filtered multinomial sampling with explicit probability masking to avoid
/// MLX categorical sampling bugs with zero-probability tokens.
public func sampleToken(
    logits: MLXArray,
    config: SamplingConfig,
    generatedTokens: [Int32] = [],
    suppressRange: (Int, Int)? = nil,
    eosTokenId: Int? = nil
) -> Int32 {
    // logits: [1, 1, vocab] or [vocab] — work with last dim
    var logits = logits.squeezed().asType(.float32)  // [vocab]
    let vocabSize = logits.dim(0)

    // 1. Token suppression: set range to -inf (except EOS)
    if let (start, end) = suppressRange, start < end, start >= 0, end <= vocabSize {
        let indices = MLXArray(0..<Int32(vocabSize))
        let geStart = indices .>= MLXArray(Int32(start))
        let ltEnd = indices .< MLXArray(Int32(end))
        var suppressMask = logicalAnd(geStart, ltEnd)

        if let eos = eosTokenId, eos >= start, eos < end {
            let notEos = indices .!= MLXArray(Int32(eos))
            suppressMask = logicalAnd(suppressMask, notEos)
        }

        logits = MLX.where(suppressMask, MLXArray(Float(-1e9)), logits)
    }

    // 2. Repetition penalty
    if config.repetitionPenalty != 1.0 && !generatedTokens.isEmpty {
        let uniqueTokens = Array(Set(generatedTokens))
        let indices = MLXArray(0..<Int32(vocabSize))
        var penaltyMask = indices .== Int32(-1)  // all false
        for token in uniqueTokens {
            penaltyMask = logicalOr(penaltyMask, indices .== token)
        }

        let penalty = MLXArray(config.repetitionPenalty)
        let penalizedPos = logits / penalty
        let penalizedNeg = logits * penalty
        let penalized = MLX.where(logits .< MLXArray(Float(0)), penalizedNeg, penalizedPos)
        logits = MLX.where(penaltyMask, penalized, logits)
    }

    // 3. Greedy decoding
    if config.temperature <= 0 {
        return argMax(logits).item(Int32.self)
    }

    // 4. Apply temperature
    logits = logits / MLXArray(config.temperature)

    // 5. Save EOS logit before top-k/top-p (so it can't be filtered out)
    var savedEosLogit: MLXArray? = nil
    if let eos = eosTokenId, eos >= 0, eos < vocabSize {
        savedEosLogit = logits[eos]
    }

    // 6. Top-k filtering
    if config.topK > 0 && config.topK < vocabSize {
        let sorted = MLX.sorted(logits)
        let threshold = sorted[vocabSize - config.topK]
        logits = MLX.where(logits .< threshold, MLXArray(Float(-1e9)), logits)
    }

    // 7. Top-p (nucleus) filtering
    if config.topP < 1.0 {
        let sortedIndices = argSort(logits)
        let sortedLogits = logits[sortedIndices]
        let probs = softmax(sortedLogits)
        let cumProbs = cumsum(probs)

        let sortedMask = cumProbs - probs .> MLXArray(config.topP)
        let filteredLogits = MLX.where(sortedMask, MLXArray(Float(-1e9)), sortedLogits)

        let unsortIndices = argSort(sortedIndices)
        logits = filteredLogits[unsortIndices]
    }

    // 8. Restore EOS logit after top-k/top-p + apply EOS bias
    if let eos = eosTokenId, let eosLogit = savedEosLogit, eos >= 0, eos < vocabSize {
        let biasedEos = config.eosLogitBias != 0
            ? eosLogit + MLXArray(config.eosLogitBias)
            : eosLogit
        let indices = MLXArray(0..<Int32(vocabSize))
        let eosMask = indices .== MLXArray(Int32(eos))
        logits = MLX.where(eosMask, biasedEos, logits)
    }

    // 9. Multinomial sampling via Gumbel-max trick
    // argmax(logits + Gumbel) ~ Categorical(softmax(logits))
    // This avoids MLX categorical bugs where zero-probability tokens can be sampled.
    let gumbel = MLXRandom.gumbel(logits.shape)
    let perturbedLogits = logits + gumbel
    return argMax(perturbedLogits).item(Int32.self)
}

/// Lazy version of sampleToken for code predictor: returns MLXArray (no GPU sync).
///
/// Identical sampling logic (temperature, top-k, Gumbel-max) but returns the argmax result
/// as a lazy MLXArray scalar instead of calling `.item()`. This allows chaining 15 CP group
/// predictions into one lazy computation graph, evaluated with a single `eval()` at the end.
///
/// Omits repetition penalty, token suppression, and EOS protection (not needed for CP).
public func sampleTokenLazy(
    logits: MLXArray,
    config: SamplingConfig
) -> MLXArray {
    var logits = logits.squeezed().asType(.float32)  // [vocab]

    if config.temperature <= 0 {
        return argMax(logits).asType(.int32)
    }

    logits = logits / MLXArray(config.temperature)

    let vocabSize = logits.dim(0)
    if config.topK > 0 && config.topK < vocabSize {
        let sorted = MLX.sorted(logits)
        let threshold = sorted[vocabSize - config.topK]
        logits = MLX.where(logits .< threshold, MLXArray(Float(-1e9)), logits)
    }

    let gumbel = MLXRandom.gumbel(logits.shape)
    let perturbedLogits = logits + gumbel
    return argMax(perturbedLogits).asType(.int32)  // scalar MLXArray — NO .item()!
}

/// Batch-sample tokens from logits for B items simultaneously.
/// Supports temperature, top-k, top-p, token suppression, and EOS protection.
/// Finished items receive `padToken` instead of a sampled token.
/// Repetition penalty is skipped in batch mode (requires per-item history tracking).
///
/// - Parameters:
///   - logits: `[B, 1, vocab]` or `[B, vocab]`
///   - config: Sampling configuration (temperature, topK, topP)
///   - finishedMask: `[B]` bool — true = item already finished
///   - padToken: Token to feed back for finished items
///   - suppressRange: Optional range of token IDs to suppress (except EOS)
///   - eosTokenId: EOS token ID (protected from suppression/filtering)
/// - Returns: `[B]` Int32 sampled tokens
public func sampleTokensBatch(
    logits: MLXArray,
    config: SamplingConfig,
    finishedMask: MLXArray,
    padToken: Int32,
    suppressRange: (Int, Int)? = nil,
    eosTokenId: Int? = nil
) -> MLXArray {
    // Squeeze to [B, vocab]
    var logits2d: MLXArray
    if logits.ndim == 3 {
        logits2d = logits.squeezed(axis: 1).asType(.float32)
    } else {
        logits2d = logits.asType(.float32)
    }
    let vocabSize = logits2d.dim(1)

    // 1. Token suppression: broadcast [vocab] mask over [B, vocab]
    if let (start, end) = suppressRange, start < end, start >= 0, end <= vocabSize {
        let indices = MLXArray(0..<Int32(vocabSize))
        let geStart = indices .>= MLXArray(Int32(start))
        let ltEnd = indices .< MLXArray(Int32(end))
        var suppressMask = logicalAnd(geStart, ltEnd)  // [vocab]

        if let eos = eosTokenId, eos >= start, eos < end {
            let notEos = indices .!= MLXArray(Int32(eos))
            suppressMask = logicalAnd(suppressMask, notEos)
        }

        // Broadcast [vocab] over [B, vocab]
        logits2d = MLX.where(suppressMask, MLXArray(Float(-1e9)), logits2d)
    }

    // 2. Greedy decoding
    if config.temperature <= 0 {
        let tokens = argMax(logits2d, axis: 1).asType(.int32)  // [B]
        return MLX.where(finishedMask, MLXArray(padToken), tokens)
    }

    // 3. Apply temperature
    logits2d = logits2d / MLXArray(config.temperature)

    // 4. Save EOS logit per row before filtering
    var savedEosCol: MLXArray? = nil
    if let eos = eosTokenId, eos >= 0, eos < vocabSize {
        savedEosCol = logits2d[0..., eos..<(eos + 1)]  // [B, 1]
    }

    // 5. Top-k filtering (per row)
    if config.topK > 0 && config.topK < vocabSize {
        let sorted = MLX.sorted(logits2d, axis: 1)  // [B, vocab]
        let threshold = sorted[0..., (vocabSize - config.topK)..<(vocabSize - config.topK + 1)]  // [B, 1]
        logits2d = MLX.where(logits2d .< threshold, MLXArray(Float(-1e9)), logits2d)
    }

    // 6. Top-p filtering (per row)
    if config.topP < 1.0 {
        let sortedIndices = argSort(logits2d, axis: 1)  // [B, vocab]
        let sortedLogits = takeAlong(logits2d, sortedIndices, axis: 1)  // [B, vocab]
        let probs = softmax(sortedLogits, axis: 1)
        let cumProbs = cumsum(probs, axis: 1)

        let sortedMask = cumProbs - probs .> MLXArray(config.topP)
        let filteredSorted = MLX.where(sortedMask, MLXArray(Float(-1e9)), sortedLogits)

        let unsortIndices = argSort(sortedIndices, axis: 1)
        logits2d = takeAlong(filteredSorted, unsortIndices, axis: 1)
    }

    // 7. Restore EOS logit after filtering + apply EOS bias
    if let eos = eosTokenId, let eosCol = savedEosCol, eos >= 0, eos < vocabSize {
        let biasedEos = config.eosLogitBias != 0
            ? eosCol + MLXArray(config.eosLogitBias)
            : eosCol
        let indices = MLXArray(0..<Int32(vocabSize))
        let eosMask = indices .== MLXArray(Int32(eos))  // [vocab], broadcasts over [B, vocab]
        logits2d = MLX.where(eosMask, biasedEos, logits2d)
    }

    // 8. Gumbel-max sampling: argmax(logits + Gumbel) ~ Categorical(softmax(logits))
    let gumbel = MLXRandom.gumbel(logits2d.shape)
    let perturbed = logits2d + gumbel
    let sampledTokens = argMax(perturbed, axis: 1).asType(.int32)  // [B]

    // 9. Replace finished items with pad token
    return MLX.where(finishedMask, MLXArray(padToken), sampledTokens)
}
