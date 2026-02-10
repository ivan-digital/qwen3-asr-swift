import Foundation
import MLX
import MLXNN

/// Multimodal Rotary Position Embedding (MRoPE) for TTS Talker
/// Uses 3D position IDs with interleaved frequency assignment across sections [24, 20, 20]
///
/// Python reference (talker.py):
///   Global inv_freq = 1/(base^(arange(0,dim,2)/dim))  — single array of headDim/2
///   For each of 3 position dims: freqs = inv_freq @ pos → [B, headDim/2, S]
///   Interleave via masking: indices % 3 == 0 → temporal, == 1 → height, == 2 → width
///   Result: [B, S, headDim/2], then concat to [B, S, headDim]
public class TalkerRotaryEmbedding {
    let headDim: Int
    let sections: [Int]  // [24, 20, 20] — must sum to headDim/2
    let base: Float
    private var invFreqCache: MLXArray?
    private var maskCache: (hMask: MLXArray, wMask: MLXArray)?

    public init(headDim: Int, sections: [Int] = [24, 20, 20], base: Float = 1_000_000.0) {
        self.headDim = headDim
        self.sections = sections
        self.base = base
        assert(sections.reduce(0, +) == headDim / 2, "MRoPE sections must sum to headDim/2")
    }

    /// Global inverse frequencies: 1/(base^(arange(0, headDim, 2) / headDim))
    /// Shape: [headDim/2, 1]
    private func getInvFreq() -> MLXArray {
        if let cached = invFreqCache { return cached }

        let dim = Float(headDim)
        let halfDim = headDim / 2
        var freqs = [Float](repeating: 0, count: halfDim)
        for i in 0..<halfDim {
            freqs[i] = 1.0 / pow(base, Float(2 * i) / dim)
        }

        let result = MLXArray(freqs).reshaped([halfDim, 1])  // [halfDim, 1]
        invFreqCache = result
        return result
    }

    /// Interleave masks for height and width dimensions
    private func getMasks() -> (hMask: MLXArray, wMask: MLXArray) {
        if let cached = maskCache { return cached }

        let halfDim = headDim / 2
        // indices = [0, 1, 2, ..., halfDim-1]
        let indices = MLXArray(0..<Int32(halfDim))

        // h_mask: (index % 3 == 1) AND (index < sections[1] * 3)
        let mod3 = indices % 3
        let hMod = mod3 .== MLXArray(Int32(1))
        let hBound = indices .< MLXArray(Int32(sections[1] * 3))
        let hMask = logicalAnd(hMod, hBound)

        // w_mask: (index % 3 == 2) AND (index < sections[2] * 3)
        let wMod = mod3 .== MLXArray(Int32(2))
        let wBound = indices .< MLXArray(Int32(sections[2] * 3))
        let wMask = logicalAnd(wMod, wBound)

        let result = (hMask: hMask, wMask: wMask)
        maskCache = result
        return result
    }

    /// Compute cos/sin embeddings from position IDs
    /// - Parameter positionIds: [3, B, S] — temporal, height, width position indices
    /// - Returns: (cos, sin) each [B, 1, S, headDim] for broadcasting over heads
    public func forward(positionIds: MLXArray) -> (MLXArray, MLXArray) {
        let invFreq = getInvFreq()  // [halfDim, 1]
        let masks = getMasks()

        // positionIds: [3, B, S]
        // For each dim d, compute freqs_d = inv_freq @ pos_d
        // pos_d shape: [B, S] → expand to [B, 1, S] for matmul with [halfDim, 1]?
        // Actually: inv_freq is [halfDim, 1], pos is [B, S]
        // We want: freqs = inv_freq_expanded @ pos_expanded → [B, halfDim, S]
        // inv_freq: [halfDim, 1] → [1, halfDim, 1]
        // pos: [B, S] → [B, 1, S]
        // broadcast multiply: [B, halfDim, S]

        let invFreqExpanded = invFreq.reshaped([1, headDim / 2, 1])  // [1, halfDim, 1]

        // Temporal (section 0): positionIds[0] → [B, S]
        let posT = positionIds[0].asType(.float32)  // [B, S]
        let freqsT = invFreqExpanded * posT.expandedDimensions(axis: 1)  // [1, halfDim, 1] * [B, 1, S] → [B, halfDim, S]

        // Height (section 1)
        let posH = positionIds[1].asType(.float32)
        let freqsH = invFreqExpanded * posH.expandedDimensions(axis: 1)

        // Width (section 2)
        let posW = positionIds[2].asType(.float32)
        let freqsW = invFreqExpanded * posW.expandedDimensions(axis: 1)

        // Transpose to [B, S, halfDim]
        let freqsT_t = freqsT.transposed(0, 2, 1)  // [B, S, halfDim]
        let freqsH_t = freqsH.transposed(0, 2, 1)
        let freqsW_t = freqsW.transposed(0, 2, 1)

        // Interleave: default is temporal, then overlay height and width via masks
        var freqsCombined = MLX.where(masks.hMask, freqsH_t, freqsT_t)
        freqsCombined = MLX.where(masks.wMask, freqsW_t, freqsCombined)
        // freqsCombined: [B, S, halfDim]

        // Duplicate for rotate-half: [B, S, headDim]
        let emb = concatenated([freqsCombined, freqsCombined], axis: -1)

        // cos/sin with head dim: [B, 1, S, headDim]
        let cosEmbed = cos(emb).expandedDimensions(axis: 1)
        let sinEmbed = sin(emb).expandedDimensions(axis: 1)

        return (cosEmbed, sinEmbed)
    }
}

/// Apply rotary position embeddings using rotate-half approach
/// - Parameters:
///   - x: [B, N, S, D] — queries or keys
///   - cos: [B, 1, S, D] — cosine embeddings
///   - sin: [B, 1, S, D] — sine embeddings
/// - Returns: Rotated tensor [B, N, S, D]
public func applyRotaryPosEmb(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
    let halfDim = x.dim(-1) / 2
    let x1 = x[0..., 0..., 0..., 0..<halfDim]
    let x2 = x[0..., 0..., 0..., halfDim...]

    // Rotate half: [-x2, x1]
    let rotated = concatenated([-x2, x1], axis: -1)

    return (x * cos) + (rotated * sin)
}

/// Build position IDs for TTS generation
/// For basic TTS (no images): all 3 position dims are identical
/// - Parameters:
///   - seqLen: Total sequence length
///   - offset: Offset from KV cache
/// - Returns: [3, 1, seqLen] position IDs
public func buildTTSPositionIds(seqLen: Int, offset: Int = 0) -> MLXArray {
    let positions = MLXArray(Int32(offset)..<Int32(offset + seqLen)).expandedDimensions(axis: 0)  // [1, seqLen]
    // For basic TTS, T=H=W
    return stacked([positions, positions, positions], axis: 0)  // [3, 1, seqLen]
}
