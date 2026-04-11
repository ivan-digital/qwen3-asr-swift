import Foundation
import MLX
import MLXFast

/// Multi-head scaled dot-product attention helper used by every attention
/// module in the project. Takes already-projected per-token Q/K/V tensors of
/// shape `[B, T, numHeads * headDim]`, reshapes to `[B, H, T, headDim]`,
/// runs the optimised `MLXFast.scaledDotProductAttention` Metal kernel, then
/// merges the heads back to `[B, T, numHeads * headDim]` ready for the
/// output projection.
///
/// Each module still owns its own input projections (which vary in width,
/// quantisation, bias, etc.) and its own output projection — this helper
/// only collapses the boilerplate around the SDPA call itself.
public enum SDPA {

    /// Standard attention with optional bool/float mask.
    public static func multiHead(
        q: MLXArray, k: MLXArray, v: MLXArray,
        numHeads: Int, headDim: Int, scale: Float,
        mask: MLXArray? = nil
    ) -> MLXArray {
        let qLen = q.dim(1)
        let kLen = k.dim(1)

        // Q: [B, T_q, H*D] → [B, H, T_q, D]. Use -1 for the batch dim so the
        // helper composes with compile(shapeless:) graphs that vary batch.
        let qHeads = q.reshaped(-1, qLen, numHeads, headDim).transposed(0, 2, 1, 3)
        let kHeads = k.reshaped(-1, kLen, numHeads, headDim).transposed(0, 2, 1, 3)
        let vHeads = v.reshaped(-1, kLen, numHeads, headDim).transposed(0, 2, 1, 3)

        let attn = MLXFast.scaledDotProductAttention(
            queries: qHeads, keys: kHeads, values: vHeads,
            scale: scale, mask: mask)

        return attn.transposed(0, 2, 1, 3).reshaped(-1, qLen, numHeads * headDim)
    }

    /// GQA / MQA variant: query and key/value heads can have different
    /// counts. The kv tensors are repeated to match the query head count
    /// inside `MLXFast.scaledDotProductAttention`.
    public static func multiHead(
        q: MLXArray, k: MLXArray, v: MLXArray,
        numQueryHeads: Int, numKVHeads: Int, headDim: Int, scale: Float,
        mask: MLXArray? = nil
    ) -> MLXArray {
        let qLen = q.dim(1)
        let kLen = k.dim(1)

        let qHeads = q.reshaped(-1, qLen, numQueryHeads, headDim).transposed(0, 2, 1, 3)
        let kHeads = k.reshaped(-1, kLen, numKVHeads, headDim).transposed(0, 2, 1, 3)
        let vHeads = v.reshaped(-1, kLen, numKVHeads, headDim).transposed(0, 2, 1, 3)

        let attn = MLXFast.scaledDotProductAttention(
            queries: qHeads, keys: kHeads, values: vHeads,
            scale: scale, mask: mask)

        return attn.transposed(0, 2, 1, 3).reshaped(-1, qLen, numQueryHeads * headDim)
    }

    /// Run SDPA on tensors that are already shaped `[B, H, T, D]` (e.g. after
    /// RoPE / KV-cache concatenation in LLM-style attention) and merge the
    /// heads back to `[B, T, H * D]`. Saves the boilerplate transpose+reshape
    /// after the SDPA call without dictating where the projections live.
    public static func attendAndMerge(
        qHeads: MLXArray, kHeads: MLXArray, vHeads: MLXArray,
        scale: Float,
        mask: MLXArray? = nil
    ) -> MLXArray {
        let attn = MLXFast.scaledDotProductAttention(
            queries: qHeads, keys: kHeads, values: vHeads,
            scale: scale, mask: mask)
        return mergeHeads(attn)
    }

    /// `ScaledDotProductAttentionMaskMode` overload — used by modules that
    /// pass causal / additive masks via the newer mlx-swift API.
    public static func attendAndMerge(
        qHeads: MLXArray, kHeads: MLXArray, vHeads: MLXArray,
        scale: Float,
        mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> MLXArray {
        let attn = MLXFast.scaledDotProductAttention(
            queries: qHeads, keys: kHeads, values: vHeads,
            scale: scale, mask: mask)
        return mergeHeads(attn)
    }

    /// Merge heads back: `[B, H, T, D] → [B, T, H * D]`.
    ///
    /// Uses `-1` for the batch dimension so the result composes with
    /// `MLX.compile(shapeless: true)` graphs that vary the batch at runtime
    /// (e.g. Qwen3-TTS Talker autoregressive decode with different batch
    /// sizes per call).
    @inline(__always)
    public static func mergeHeads(_ attn: MLXArray) -> MLXArray {
        let H = attn.dim(1)
        let T = attn.dim(2)
        let D = attn.dim(3)
        return attn.transposed(0, 2, 1, 3).reshaped(-1, T, H * D)
    }
}
