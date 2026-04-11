import Foundation
import MLX
import MLXNN
import MLXCommon
import AudioCommon

/// Loads `model.safetensors` from a published Omnilingual MLX repo and applies
/// every tensor to the matching submodule of an `OmnilingualASRMLXModel`.
public enum OmnilingualMLXWeightLoader {

    public static func loadWeights(
        into model: OmnilingualASRMLXModel,
        from directory: URL
    ) throws {
        let candidate = directory.appendingPathComponent("model.safetensors")
        guard FileManager.default.fileExists(atPath: candidate.path) else {
            throw WeightLoadingError.noWeightsFound(directory)
        }
        let raw = try CommonWeightLoader.loadSafetensors(url: candidate)
        AudioLog.modelLoading.debug("Omnilingual MLX: loaded \(raw.count) tensors from \(candidate.lastPathComponent)")

        // Cast every weight to F32 — MLX-Swift's QuantizedLinear expects scales
        // and biases as F32 in many code paths even though the published file is
        // F16. Casting up front avoids repeated dtype mismatches at inference.
        var w: [String: MLXArray] = [:]
        w.reserveCapacity(raw.count)
        for (k, v) in raw {
            // Quantized weights are U32 packed bit-payloads — leave as-is.
            if v.dtype == .uint32 {
                w[k] = v
            } else {
                w[k] = v.asType(.float32)
            }
        }

        try applyFrontend(model.frontend, weights: w)
        try applyEncoder(model.encoder, weights: w)
        applyHead(model.ctcHead, weights: w)
    }

    // MARK: - Frontend

    private static func applyFrontend(
        _ frontend: Wav2Vec2Frontend, weights w: [String: MLXArray]
    ) throws {
        let p = "encoder_frontend"

        // Feature extractor: 7 conv layers, each Conv1d + LayerNorm
        for (i, layer) in frontend.featureExtractor.layers.enumerated() {
            let lp = "\(p).feature_extractor.layers.\(i)"
            CommonWeightLoader.applyConv1dWeights(
                to: layer.conv, prefix: "\(lp).conv", from: w, transpose: true)
            CommonWeightLoader.applyLayerNormWeights(
                to: layer.layerNorm, prefix: "\(lp).layer_norm", from: w)
        }

        CommonWeightLoader.applyLayerNormWeights(
            to: frontend.postExtractLayerNorm,
            prefix: "\(p).post_extract_layer_norm",
            from: w)
        CommonWeightLoader.applyLinearWeights(
            to: frontend.modelDimProj,
            prefix: "\(p).model_dim_proj",
            from: w)

        // Position encoder: weight_norm(dim=2) → fuse weight_g + weight_v into a
        // dense weight before handing it to MLX Conv1d.
        let posPrefix = "\(p).pos_encoder.conv"
        guard let g = w["\(posPrefix).weight_g"],
              let v = w["\(posPrefix).weight_v"]
        else {
            throw WeightLoadingError.missingRequiredWeight("\(posPrefix).weight_{g,v}")
        }
        let fused = fuseWeightNorm(g: g, v: v)
        // PyTorch [out, in/groups, K] → MLX [out, K, in/groups]
        let mlxWeight = fused.transposed(0, 2, 1)

        var posParams: [String: NestedItem<String, MLXArray>] = [
            "weight": .value(mlxWeight)
        ]
        if let bias = w["\(posPrefix).bias"] {
            posParams["bias"] = .value(bias)
        }
        frontend.posEncoder.conv.update(parameters: ModuleParameters(values: posParams))
    }

    /// Reconstruct PyTorch `weight_norm(conv, dim=2)`: per-kernel-position
    /// magnitude `g[1,1,K]` rescales each kernel slice of `v[O, I, K]` to that
    /// magnitude. `W[:,:,k] = g[0,0,k] * v[:,:,k] / ||v[:,:,k]||`.
    private static func fuseWeightNorm(g: MLXArray, v: MLXArray) -> MLXArray {
        let vF = v.asType(.float32)
        let gF = g.asType(.float32)
        // Norm over output and input channel dims (axes 0 and 1), keep K axis.
        let sq = (vF * vF).sum(axes: [0, 1], keepDims: true)
        let norm = sq.sqrt()
        let safe = MLX.maximum(norm, MLXArray(Float(1e-12)))
        return gF * vF / safe
    }

    // MARK: - Encoder

    private static func applyEncoder(
        _ encoder: Wav2Vec2Encoder, weights w: [String: MLXArray]
    ) throws {
        for (i, layer) in encoder.layers.enumerated() {
            let lp = "encoder.layers.\(i)"

            CommonWeightLoader.applyQuantizedLinearWeights(
                to: layer.selfAttn.qProj, prefix: "\(lp).self_attn.q_proj", from: w)
            CommonWeightLoader.applyQuantizedLinearWeights(
                to: layer.selfAttn.kProj, prefix: "\(lp).self_attn.k_proj", from: w)
            CommonWeightLoader.applyQuantizedLinearWeights(
                to: layer.selfAttn.vProj, prefix: "\(lp).self_attn.v_proj", from: w)
            CommonWeightLoader.applyQuantizedLinearWeights(
                to: layer.selfAttn.outputProj, prefix: "\(lp).self_attn.output_proj", from: w)
            CommonWeightLoader.applyLayerNormWeights(
                to: layer.selfAttnLayerNorm, prefix: "\(lp).self_attn_layer_norm", from: w)

            CommonWeightLoader.applyQuantizedLinearWeights(
                to: layer.ffn.innerProj, prefix: "\(lp).ffn.inner_proj", from: w)
            CommonWeightLoader.applyQuantizedLinearWeights(
                to: layer.ffn.outputProj, prefix: "\(lp).ffn.output_proj", from: w)
            CommonWeightLoader.applyLayerNormWeights(
                to: layer.ffnLayerNorm, prefix: "\(lp).ffn_layer_norm", from: w)
        }
        CommonWeightLoader.applyLayerNormWeights(
            to: encoder.layerNorm, prefix: "encoder.layer_norm", from: w)
    }

    // MARK: - Head

    private static func applyHead(_ head: CTCHead, weights w: [String: MLXArray]) {
        CommonWeightLoader.applyQuantizedLinearWeights(
            to: head.proj, prefix: "final_proj", from: w)
    }
}
