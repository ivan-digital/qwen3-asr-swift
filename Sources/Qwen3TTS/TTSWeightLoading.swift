import Foundation
import MLX
import MLXNN
import Qwen3Common

/// Weight loading for TTS components
public enum TTSWeightLoader {

    // MARK: - Combined Talker + Code Predictor (single file load)

    public static func loadTalkerAndCodePredictorWeights(
        talker: TalkerModel,
        codePredictor: CodePredictorModel,
        from directory: URL
    ) throws {
        let allWeights = try CommonWeightLoader.loadAllSafetensors(from: directory)
        print("Loaded \(allWeights.count) weights from safetensors")

        // Split into talker and code predictor weights
        var talkerWeights: [String: MLXArray] = [:]
        var cpWeights: [String: MLXArray] = [:]
        for (key, value) in allWeights {
            if key.hasPrefix("talker.code_predictor.") {
                let strippedKey = String(key.dropFirst("talker.code_predictor.".count))
                cpWeights[strippedKey] = value
            } else if key.hasPrefix("talker.") {
                let strippedKey = String(key.dropFirst("talker.".count))
                talkerWeights[strippedKey] = value
            }
        }

        applyTalkerWeights(to: talker, from: talkerWeights)
        applyCodePredictorWeights(to: codePredictor, from: cpWeights)
    }

    // MARK: - Talker

    private static func applyTalkerWeights(
        to talker: TalkerModel,
        from talkerWeights: [String: MLXArray]
    ) {
        print("Found \(talkerWeights.count) talker weights")

        // Codec embedding (float, not quantized)
        CommonWeightLoader.applyEmbeddingWeights(
            to: talker.codecEmbedding, prefix: "model.codec_embedding", from: talkerWeights)

        // Text embedding (float, not quantized)
        CommonWeightLoader.applyEmbeddingWeights(
            to: talker.textEmbedding, prefix: "model.text_embedding", from: talkerWeights)

        // Text projection MLP (safetensors keys use "linear_fc1"/"linear_fc2")
        CommonWeightLoader.applyQuantizedLinearWeights(
            to: talker.textProjection.fc1, prefix: "text_projection.linear_fc1", from: talkerWeights)
        if let bias1 = talkerWeights["text_projection.linear_fc1.bias"] {
            talker.textProjection.fc1.update(parameters: ModuleParameters(values: ["bias": .value(bias1)]))
        }
        CommonWeightLoader.applyQuantizedLinearWeights(
            to: talker.textProjection.fc2, prefix: "text_projection.linear_fc2", from: talkerWeights)
        if let bias2 = talkerWeights["text_projection.linear_fc2.bias"] {
            talker.textProjection.fc2.update(parameters: ModuleParameters(values: ["bias": .value(bias2)]))
        }

        // Codec head
        CommonWeightLoader.applyQuantizedLinearWeights(
            to: talker.codecHead, prefix: "codec_head", from: talkerWeights)

        // Final norm
        CommonWeightLoader.applyRMSNormWeights(
            to: talker.norm, prefix: "model.norm", from: talkerWeights)

        // Transformer layers
        for (i, layer) in talker.layers.enumerated() {
            let prefix = "model.layers.\(i)"
            applyTalkerLayerWeights(to: layer, prefix: prefix, from: talkerWeights)
        }

        print("Applied weights to Talker (\(talker.layers.count) layers)")
    }

    private static func applyTalkerLayerWeights(
        to layer: TalkerDecoderLayer,
        prefix: String,
        from weights: [String: MLXArray]
    ) {
        // Self attention
        CommonWeightLoader.applyQuantizedLinearWeights(
            to: layer.selfAttn.qProj, prefix: "\(prefix).self_attn.q_proj", from: weights)
        CommonWeightLoader.applyQuantizedLinearWeights(
            to: layer.selfAttn.kProj, prefix: "\(prefix).self_attn.k_proj", from: weights)
        CommonWeightLoader.applyQuantizedLinearWeights(
            to: layer.selfAttn.vProj, prefix: "\(prefix).self_attn.v_proj", from: weights)
        CommonWeightLoader.applyQuantizedLinearWeights(
            to: layer.selfAttn.oProj, prefix: "\(prefix).self_attn.o_proj", from: weights)

        // Q/K norms
        CommonWeightLoader.applyRMSNormWeights(
            to: layer.selfAttn.qNorm, prefix: "\(prefix).self_attn.q_norm", from: weights)
        CommonWeightLoader.applyRMSNormWeights(
            to: layer.selfAttn.kNorm, prefix: "\(prefix).self_attn.k_norm", from: weights)

        // Layer norms
        CommonWeightLoader.applyRMSNormWeights(
            to: layer.inputLayerNorm, prefix: "\(prefix).input_layernorm", from: weights)
        CommonWeightLoader.applyRMSNormWeights(
            to: layer.postAttentionLayerNorm, prefix: "\(prefix).post_attention_layernorm", from: weights)

        // MLP
        CommonWeightLoader.applyQuantizedMLPWeights(
            to: layer.mlp, prefix: "\(prefix).mlp", from: weights)
    }

    // MARK: - Code Predictor

    private static func applyCodePredictorWeights(
        to codePredictor: CodePredictorModel,
        from cpWeights: [String: MLXArray]
    ) {
        print("Found \(cpWeights.count) code predictor weights")

        // Codec embeddings (15 tables — safetensors uses singular "codec_embedding")
        for i in 0..<codePredictor.codecEmbeddings.count {
            CommonWeightLoader.applyEmbeddingWeights(
                to: codePredictor.codecEmbeddings[i],
                prefix: "model.codec_embedding.\(i)",
                from: cpWeights)
        }

        // Transformer layers
        for (i, layer) in codePredictor.layers.enumerated() {
            let prefix = "model.layers.\(i)"
            applyCodePredictorLayerWeights(to: layer, prefix: prefix, from: cpWeights)
        }

        // Norm
        CommonWeightLoader.applyRMSNormWeights(
            to: codePredictor.norm, prefix: "model.norm", from: cpWeights)

        // LM heads (15, quantized)
        for i in 0..<codePredictor.lmHeads.count {
            CommonWeightLoader.applyQuantizedLinearWeights(
                to: codePredictor.lmHeads[i],
                prefix: "lm_head.\(i)",
                from: cpWeights)
        }

        print("Applied weights to Code Predictor (\(codePredictor.layers.count) layers)")
    }

    private static func applyCodePredictorLayerWeights(
        to layer: CodePredictorDecoderLayer,
        prefix: String,
        from weights: [String: MLXArray]
    ) {
        CommonWeightLoader.applyQuantizedLinearWeights(
            to: layer.selfAttn.qProj, prefix: "\(prefix).self_attn.q_proj", from: weights)
        CommonWeightLoader.applyQuantizedLinearWeights(
            to: layer.selfAttn.kProj, prefix: "\(prefix).self_attn.k_proj", from: weights)
        CommonWeightLoader.applyQuantizedLinearWeights(
            to: layer.selfAttn.vProj, prefix: "\(prefix).self_attn.v_proj", from: weights)
        CommonWeightLoader.applyQuantizedLinearWeights(
            to: layer.selfAttn.oProj, prefix: "\(prefix).self_attn.o_proj", from: weights)

        CommonWeightLoader.applyRMSNormWeights(
            to: layer.selfAttn.qNorm, prefix: "\(prefix).self_attn.q_norm", from: weights)
        CommonWeightLoader.applyRMSNormWeights(
            to: layer.selfAttn.kNorm, prefix: "\(prefix).self_attn.k_norm", from: weights)

        CommonWeightLoader.applyRMSNormWeights(
            to: layer.inputLayerNorm, prefix: "\(prefix).input_layernorm", from: weights)
        CommonWeightLoader.applyRMSNormWeights(
            to: layer.postAttentionLayerNorm, prefix: "\(prefix).post_attention_layernorm", from: weights)

        CommonWeightLoader.applyQuantizedMLPWeights(
            to: layer.mlp, prefix: "\(prefix).mlp", from: weights)
    }

    // MARK: - Speech Tokenizer Decoder

    public static func loadSpeechTokenizerDecoderWeights(
        into decoder: SpeechTokenizerDecoder,
        from directory: URL
    ) throws {
        let allWeights = try CommonWeightLoader.loadAllSafetensors(from: directory)

        print("Found \(allWeights.count) speech tokenizer weights total")

        // Load RVQ codebook weights (decoder-side quantizer)
        loadRVQWeights(into: decoder.splitRVQ, from: allWeights)

        // Pre-conv: decoder.pre_conv.conv.{weight,bias}
        CommonWeightLoader.applyConv1dWeights(
            to: decoder.preConv.conv, prefix: "decoder.pre_conv.conv", from: allWeights, transpose: true)

        // Pre-transformer input/output projections
        CommonWeightLoader.applyLinearWeights(
            to: decoder.transformer.inputProj, prefix: "decoder.pre_transformer.input_proj", from: allWeights)
        CommonWeightLoader.applyLinearWeights(
            to: decoder.transformer.outputProj, prefix: "decoder.pre_transformer.output_proj", from: allWeights)

        // Pre-transformer layers
        for (i, layer) in decoder.transformer.layers.enumerated() {
            loadDecoderTransformerLayerWeights(to: layer, index: i, from: allWeights)
        }

        // Pre-transformer norm
        CommonWeightLoader.applyRMSNormWeights(
            to: decoder.transformer.norm, prefix: "decoder.pre_transformer.norm", from: allWeights)

        // Upsample stages: decoder.upsample.{0,1}
        // Stage 0: .0.conv (transposed conv) + .1 (ConvNeXt)
        CommonWeightLoader.applyConvTransposed1dWeights(
            to: decoder.preUpsample1.conv, prefix: "decoder.upsample.0.0.conv", from: allWeights, transpose: true)
        loadConvNeXtBlockWeights(to: decoder.preConvNeXt1, prefix: "decoder.upsample.0.1", from: allWeights)
        // Stage 1
        CommonWeightLoader.applyConvTransposed1dWeights(
            to: decoder.preUpsample2.conv, prefix: "decoder.upsample.1.0.conv", from: allWeights, transpose: true)
        loadConvNeXtBlockWeights(to: decoder.preConvNeXt2, prefix: "decoder.upsample.1.1", from: allWeights)

        // Decoder initial conv: decoder.decoder.0.conv
        CommonWeightLoader.applyConv1dWeights(
            to: decoder.inputConv.conv, prefix: "decoder.decoder.0.conv", from: allWeights, transpose: true)

        // Decoder blocks: decoder.decoder.{1,2,3,4}
        for (i, block) in decoder.decoderBlocks.enumerated() {
            loadDecoderBlockWeights(to: block, blockKey: "decoder.decoder.\(i + 1)", from: allWeights)
        }

        // Final snake: decoder.decoder.5
        loadSnakeBetaWeights(to: decoder.finalSnake, prefix: "decoder.decoder.5", from: allWeights)

        // Final conv: decoder.decoder.6.conv
        CommonWeightLoader.applyConv1dWeights(
            to: decoder.finalConv.conv, prefix: "decoder.decoder.6.conv", from: allWeights, transpose: true)

        print("Applied weights to Speech Tokenizer Decoder")
    }

    // MARK: - RVQ Weight Loading

    private static func loadRVQWeights(
        into splitRVQ: SplitResidualVectorQuantizer,
        from weights: [String: MLXArray]
    ) {
        // rvq_first: 1 semantic codebook
        loadQuantizerCodebook(
            into: splitRVQ.rvqFirst.quantizers[0].embedding,
            prefix: "decoder.quantizer.rvq_first.vq.layers.0._codebook",
            from: weights)
        // rvq_first output_proj (Conv1d 256->512, kernel=1)
        CommonWeightLoader.applyConv1dWeights(
            to: splitRVQ.rvqFirst.outputProj,
            prefix: "decoder.quantizer.rvq_first.output_proj",
            from: weights, transpose: true)

        // rvq_rest: 15 acoustic codebooks
        for i in 0..<splitRVQ.rvqRest.numQuantizers {
            loadQuantizerCodebook(
                into: splitRVQ.rvqRest.quantizers[i].embedding,
                prefix: "decoder.quantizer.rvq_rest.vq.layers.\(i)._codebook",
                from: weights)
        }
        // rvq_rest output_proj
        CommonWeightLoader.applyConv1dWeights(
            to: splitRVQ.rvqRest.outputProj,
            prefix: "decoder.quantizer.rvq_rest.output_proj",
            from: weights, transpose: true)
    }

    private static func loadQuantizerCodebook(
        into embedding: Embedding,
        prefix: String,
        from weights: [String: MLXArray]
    ) {
        // Check for pre-computed embeddings first
        if let embed = weights["\(prefix).embed"] {
            let params: [String: NestedItem<String, MLXArray>] = ["weight": .value(embed)]
            embedding.update(parameters: ModuleParameters(values: params))
            return
        }

        // Compute from cluster_usage + embedding_sum
        if let usage = weights["\(prefix).cluster_usage"],
           let embSum = weights["\(prefix).embedding_sum"] {
            let eps = MLXArray(Float(1e-7))
            let clampedUsage = maximum(usage, eps).expandedDimensions(axis: -1)
            let computed = embSum / clampedUsage
            let params: [String: NestedItem<String, MLXArray>] = ["weight": .value(computed)]
            embedding.update(parameters: ModuleParameters(values: params))
        }
    }

    // MARK: - Decoder Component Weight Loading

    private static func loadDecoderTransformerLayerWeights(
        to layer: DecoderTransformerLayer,
        index: Int,
        from weights: [String: MLXArray]
    ) {
        let prefix = "decoder.pre_transformer.layers.\(index)"

        // Attention projections
        CommonWeightLoader.applyLinearWeights(
            to: layer.selfAttn.qProj, prefix: "\(prefix).self_attn.q_proj", from: weights)
        CommonWeightLoader.applyLinearWeights(
            to: layer.selfAttn.kProj, prefix: "\(prefix).self_attn.k_proj", from: weights)
        CommonWeightLoader.applyLinearWeights(
            to: layer.selfAttn.vProj, prefix: "\(prefix).self_attn.v_proj", from: weights)
        CommonWeightLoader.applyLinearWeights(
            to: layer.selfAttn.oProj, prefix: "\(prefix).self_attn.o_proj", from: weights)

        // Layer norms
        CommonWeightLoader.applyRMSNormWeights(
            to: layer.norm1, prefix: "\(prefix).input_layernorm", from: weights)
        CommonWeightLoader.applyRMSNormWeights(
            to: layer.norm2, prefix: "\(prefix).post_attention_layernorm", from: weights)

        // SwiGLU MLP
        CommonWeightLoader.applyLinearWeights(
            to: layer.gateProj, prefix: "\(prefix).mlp.gate_proj", from: weights)
        CommonWeightLoader.applyLinearWeights(
            to: layer.upProj, prefix: "\(prefix).mlp.up_proj", from: weights)
        CommonWeightLoader.applyLinearWeights(
            to: layer.downProj, prefix: "\(prefix).mlp.down_proj", from: weights)

        // LayerScale for attention and MLP
        if let scale = weights["\(prefix).self_attn_layer_scale.scale"] {
            let params: [String: NestedItem<String, MLXArray>] = ["scale": .value(scale.reshaped([1, 1, -1]))]
            layer.attnLayerScale.update(parameters: ModuleParameters(values: params))
        }
        if let scale = weights["\(prefix).mlp_layer_scale.scale"] {
            let params: [String: NestedItem<String, MLXArray>] = ["scale": .value(scale.reshaped([1, 1, -1]))]
            layer.mlpLayerScale.update(parameters: ModuleParameters(values: params))
        }
    }

    private static func loadConvNeXtBlockWeights(
        to block: ConvNeXtBlock,
        prefix: String,
        from weights: [String: MLXArray]
    ) {
        CommonWeightLoader.applyConv1dWeights(
            to: block.dwConv.conv, prefix: "\(prefix).dwconv.conv", from: weights, transpose: true)
        CommonWeightLoader.applyLayerNormWeights(
            to: block.norm, prefix: "\(prefix).norm", from: weights)
        CommonWeightLoader.applyLinearWeights(
            to: block.pwConv1, prefix: "\(prefix).pwconv1", from: weights)
        CommonWeightLoader.applyLinearWeights(
            to: block.pwConv2, prefix: "\(prefix).pwconv2", from: weights)

        // LayerScale gamma
        if let scale = weights["\(prefix).gamma"] {
            let params: [String: NestedItem<String, MLXArray>] = ["scale": .value(scale.reshaped([1, 1, -1]))]
            block.layerScale.update(parameters: ModuleParameters(values: params))
        }
    }

    private static func loadSnakeBetaWeights(
        to snake: SnakeBeta,
        prefix: String,
        from weights: [String: MLXArray]
    ) {
        if let alpha = weights["\(prefix).alpha"] {
            let params: [String: NestedItem<String, MLXArray>] = ["alpha": .value(alpha.reshaped([1, 1, -1]))]
            snake.update(parameters: ModuleParameters(values: params))
        }
        if let beta = weights["\(prefix).beta"] {
            let params: [String: NestedItem<String, MLXArray>] = ["beta": .value(beta.reshaped([1, 1, -1]))]
            snake.update(parameters: ModuleParameters(values: params))
        }
    }

    /// Load decoder block weights (SEANet style)
    /// Block key structure: decoder.decoder.{N}.block.0 = Snake, .block.1 = TransposedConv,
    /// .block.{2,3,4} = ResBlocks (each with act1, conv1, act2, conv2)
    private static func loadDecoderBlockWeights(
        to block: DecoderBlock,
        blockKey: String,
        from weights: [String: MLXArray]
    ) {
        // block.0 = Snake activation
        loadSnakeBetaWeights(to: block.snake, prefix: "\(blockKey).block.0", from: weights)

        // block.1 = Transposed conv (upsample)
        CommonWeightLoader.applyConvTransposed1dWeights(
            to: block.upsample.conv, prefix: "\(blockKey).block.1.conv", from: weights, transpose: true)

        // block.{2,3,4} = 3 residual units
        for (j, unit) in block.residualUnits.enumerated() {
            let resPrefix = "\(blockKey).block.\(j + 2)"
            loadSnakeBetaWeights(to: unit.snake1, prefix: "\(resPrefix).act1", from: weights)
            CommonWeightLoader.applyConv1dWeights(
                to: unit.conv1.conv, prefix: "\(resPrefix).conv1.conv", from: weights, transpose: true)
            loadSnakeBetaWeights(to: unit.snake2, prefix: "\(resPrefix).act2", from: weights)
            CommonWeightLoader.applyConv1dWeights(
                to: unit.conv2.conv, prefix: "\(resPrefix).conv2.conv", from: weights, transpose: true)
        }
    }
}
