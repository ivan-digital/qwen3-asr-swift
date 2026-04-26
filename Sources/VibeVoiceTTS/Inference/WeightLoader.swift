import Foundation
import MLX
import MLXNN

public enum WeightLoadingError: Error {
    case fileNotFound(String)
    case configNotFound(String)
    case weightKeyMissing(String)
    case invalidWeightShape(key: String, expected: [Int], got: [Int])
}

public func loadVibeVoiceConfiguration(from directory: URL) throws -> VibeVoiceConfiguration {
    let configURL = directory.appendingPathComponent("config.json")
    let data = try Data(contentsOf: configURL)
    let decoder = JSONDecoder()
    return try decoder.decode(VibeVoiceConfiguration.self, from: data)
}

func loadWeights(from url: URL) throws -> [String: MLXArray] {
    try MLX.loadArrays(url: url)
}

func materializeWeights(_ weights: [String: MLXArray]) -> [String: MLXArray] {
    eval(Array(weights.values))
    return weights
}

func loadWeightsFromDirectory(_ directory: URL) throws -> [String: MLXArray] {
    let fileManager = FileManager.default
    let contents = try fileManager.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)

    var allWeights: [String: MLXArray] = [:]

    for file in contents {
        if file.pathExtension == "safetensors" {
            let weights = try loadWeights(from: file)
            for (key, value) in weights {
                allWeights[key] = value
            }
        }
    }

    return allWeights
}

private func mapCommonWeightKeys(_ key: String) -> String {
    var newKey = key
    newKey = newKey.replacingOccurrences(of: "adaLN_modulation.1.", with: "adaLN_modulation.linear.")
    newKey = newKey.replacingOccurrences(of: "t_embedder.mlp.0.", with: "t_embedder.mlp.linear1.")
    newKey = newKey.replacingOccurrences(of: "t_embedder.mlp.2.", with: "t_embedder.mlp.linear2.")
    return newKey
}

private let decoderStageOffsets: [Int] = [0, 8, 11, 14, 17, 20, 23]
/// For TokenizerEncoder, depths "3-3-3-3-3-3-8" → block index offsets:
/// stage 0: 0..3, stage 1: 3..6, ..., stage 6: 18..26.
private let encoderStageOffsets: [Int] = [0, 3, 6, 9, 12, 15, 18]

private func flattenDecoderStagesKey(_ key: String) -> String {
    let pattern = #"\.stages\.(\d+)\.(\d+)\."#
    guard let regex = try? NSRegularExpression(pattern: pattern),
          let match = regex.firstMatch(in: key, range: NSRange(key.startIndex..., in: key)),
          let stageRange = Range(match.range(at: 1), in: key),
          let blockRange = Range(match.range(at: 2), in: key),
          let stageIdx = Int(key[stageRange]),
          let blockIdx = Int(key[blockRange]),
          stageIdx < decoderStageOffsets.count else {
        return key
    }

    let flatIdx = decoderStageOffsets[stageIdx] + blockIdx
    let matchRange = Range(match.range, in: key)!
    return key.replacingCharacters(in: matchRange, with: ".stages.\(flatIdx).")
}

private func mapDecoderWeightKeys(_ key: String) -> String {
    var newKey = key
    newKey = newKey.replacingOccurrences(of: ".upsample_layers.", with: ".upsampleLayers.")
    if let range = newKey.range(of: #"\.upsampleLayers\.(\d+)\.0\."#, options: .regularExpression) {
        let match = newKey[range]
        if let indexMatch = match.range(of: #"\d+"#, options: .regularExpression) {
            let index = String(match[indexMatch])
            newKey = newKey.replacingOccurrences(of: ".upsampleLayers.\(index).0.", with: ".upsampleLayers.\(index).")
        }
    }
    newKey = newKey.replacingOccurrences(of: ".mixer.conv.conv.conv.", with: ".mixer.conv.")
    newKey = newKey.replacingOccurrences(of: ".conv.conv.weight", with: ".conv.weight")
    newKey = newKey.replacingOccurrences(of: ".conv.conv.bias", with: ".conv.bias")
    newKey = newKey.replacingOccurrences(of: ".convtr.convtr.weight", with: ".convtr.weight")
    newKey = newKey.replacingOccurrences(of: ".convtr.convtr.bias", with: ".convtr.bias")
    newKey = newKey.replacingOccurrences(of: ".ffn_gamma", with: ".ffnGamma")
    newKey = newKey.replacingOccurrences(of: ".ffn_norm.", with: ".ffnNorm.")
    newKey = flattenDecoderStagesKey(newKey)
    return newKey
}

/// Flatten 2D `(stage, block)` → 1D `flat` index, mirroring the decoder helper.
private func flattenEncoderStagesKey(_ key: String) -> String {
    let pattern = #"\.stages\.(\d+)\.(\d+)\."#
    guard let regex = try? NSRegularExpression(pattern: pattern),
          let match = regex.firstMatch(in: key, range: NSRange(key.startIndex..., in: key)),
          let stageRange = Range(match.range(at: 1), in: key),
          let blockRange = Range(match.range(at: 2), in: key),
          let stageIdx = Int(key[stageRange]),
          let blockIdx = Int(key[blockRange]),
          stageIdx < encoderStageOffsets.count else {
        return key
    }
    let flatIdx = encoderStageOffsets[stageIdx] + blockIdx
    let matchRange = Range(match.range, in: key)!
    return key.replacingCharacters(in: matchRange, with: ".stages.\(flatIdx).")
}

private func mapEncoderWeightKeys(_ key: String) -> String {
    var newKey = key
    newKey = newKey.replacingOccurrences(of: ".downsample_layers.", with: ".downsample_layers.")
    if let range = newKey.range(of: #"\.downsample_layers\.(\d+)\.0\."#, options: .regularExpression) {
        let match = newKey[range]
        if let indexMatch = match.range(of: #"\d+"#, options: .regularExpression) {
            let index = String(match[indexMatch])
            newKey = newKey.replacingOccurrences(of: ".downsample_layers.\(index).0.", with: ".downsample_layers.\(index).")
        }
    }
    newKey = newKey.replacingOccurrences(of: ".mixer.conv.conv.conv.", with: ".mixer.conv.")
    newKey = newKey.replacingOccurrences(of: ".conv.conv.weight", with: ".conv.weight")
    newKey = newKey.replacingOccurrences(of: ".conv.conv.bias", with: ".conv.bias")
    newKey = newKey.replacingOccurrences(of: ".ffn_gamma", with: ".ffnGamma")
    newKey = newKey.replacingOccurrences(of: ".ffn_norm.", with: ".ffnNorm.")
    newKey = flattenEncoderStagesKey(newKey)
    return newKey
}

func mapVibeVoiceWeightKeys(_ weights: [String: MLXArray]) -> [String: MLXArray] {
    var mapped: [String: MLXArray] = [:]

    let prefixMappings: [(from: String, to: String)] = [
        ("model.language_model.", "language_model."),
        ("model.tts_language_model.", "tts_language_model."),
        ("model.prediction_head.", "prediction_head."),
        ("model.acoustic_tokenizer.", "acoustic_tokenizer."),
        ("model.acoustic_connector.", "acoustic_connector."),
        ("model.semantic_tokenizer.", "semantic_tokenizer."),
        ("model.semantic_connector.", "semantic_connector."),
        ("model.tts_input_types.", "tts_input_types."),
        ("tts_eos_classifier.", "tts_eos_classifier."),
    ]

    for (key, value) in weights {
        var newKey = key

        for (from, to) in prefixMappings {
            if key.hasPrefix(from) {
                newKey = to + key.dropFirst(from.count)
                break
            }
        }

        if key == "model.speech_scaling_factor" || key == "model.speech_bias_factor" {
            continue
        }

        newKey = mapCommonWeightKeys(newKey)

        if newKey.contains("acoustic_tokenizer.decoder.") {
            newKey = mapDecoderWeightKeys(newKey)
        }
        if newKey.contains("acoustic_tokenizer.encoder.") {
            newKey = mapEncoderWeightKeys(newKey)
        }
        if newKey.contains("semantic_tokenizer.encoder.") {
            newKey = mapEncoderWeightKeys(newKey)
        }

        mapped[newKey] = value
    }

    return mapped
}

func transposeConv1dWeights(_ weights: [String: MLXArray]) -> [String: MLXArray] {
    var transposed: [String: MLXArray] = [:]

    for (key, value) in weights {
        if key.contains("acoustic_tokenizer.decoder.") &&
           key.hasSuffix(".weight") &&
           value.ndim == 3 {

            if key.contains(".convtr.") {
                let transposedWeight = value.transposed(1, 2, 0)
                transposed[key] = transposedWeight
            } else {
                let transposedWeight = value.transposed(0, 2, 1)
                transposed[key] = transposedWeight
            }
        }
        else if key.contains("acoustic_tokenizer.encoder.") &&
                key.hasSuffix(".weight") &&
                value.ndim == 3 {
            let transposedWeight = value.transposed(0, 2, 1)
            transposed[key] = transposedWeight
        }
        else if key.contains("semantic_tokenizer.encoder.") &&
                key.hasSuffix(".weight") &&
                value.ndim == 3 {
            let transposedWeight = value.transposed(0, 2, 1)
            transposed[key] = transposedWeight
        }
        else {
            transposed[key] = value
        }
    }

    return transposed
}

// Note: TokenizerEncoder.stages is now a flat [Block1D] (was [[Block1D]]).
// Standard model.update() handles it; the manual helpers below are kept for
// historical context but unused.
@available(*, deprecated, message: "stages is now flat — load via model.update")
internal func loadEncoderManually(_ encoder: TokenizerEncoder, weights: [String: MLXArray], prefix: String) {
    // 1) downsample_layers — list of SConv1d, each with `conv: Conv1d`.
    for (idx, sconv) in encoder.downsampleLayers.enumerated() {
        var w: [String: MLXArray] = [:]
        if let v = weights["\(prefix).downsample_layers.\(idx).conv.weight"] { w["weight"] = v }
        if let v = weights["\(prefix).downsample_layers.\(idx).conv.bias"] { w["bias"] = v }
        if let v = weights["\(prefix).downsample_layers.\(idx).conv.scales"] { w["scales"] = v }
        if let v = weights["\(prefix).downsample_layers.\(idx).conv.biases"] { w["biases"] = v }
        if !w.isEmpty {
            try? sconv.conv.update(parameters: ModuleParameters.unflattened(w), verify: .none)
        }
    }
    // 2) head — single SConv1d.
    var hw: [String: MLXArray] = [:]
    if let v = weights["\(prefix).head.conv.weight"] { hw["weight"] = v }
    if let v = weights["\(prefix).head.conv.bias"] { hw["bias"] = v }
    if let v = weights["\(prefix).head.conv.scales"] { hw["scales"] = v }
    if let v = weights["\(prefix).head.conv.biases"] { hw["biases"] = v }
    if !hw.isEmpty {
        try? encoder.head.conv.update(parameters: ModuleParameters.unflattened(hw), verify: .none)
    }
    // 3) normFinal — only if it's a ConvRMSNorm (not Identity).
    if let norm = encoder.normFinal as? ConvRMSNorm,
       let nw = weights["\(prefix).normFinal.weight"] {
        let params = ModuleParameters.unflattened(["weight": nw])
        try? norm.update(parameters: params, verify: .none)
    }
    // 4) stages — nested [[Block1D]].
    loadEncoderStagesWeights(encoder, weights: weights, prefix: prefix)
}

@available(*, deprecated)
internal func loadEncoderStagesWeights(_ encoder: TokenizerEncoder, weights: [String: MLXArray], prefix: String) {
    // No-op: stages is flat now and loads via model.update().
    _ = encoder; _ = weights; _ = prefix
}

private func loadStagesWeightsForDecoder(_ decoder: TokenizerDecoder, weights: [String: MLXArray]) {
    for (flatIdx, block) in decoder.blocks.enumerated() {
        let prefix = "acoustic_tokenizer.decoder.stages.\(flatIdx)"

        if let normWeight = weights["\(prefix).norm.weight"] {
            let params = ModuleParameters.unflattened(["weight": normWeight])
            try? block.norm.update(parameters: params, verify: .none)
        }

        if let ffnNormWeight = weights["\(prefix).ffnNorm.weight"] {
            let params = ModuleParameters.unflattened(["weight": ffnNormWeight])
            try? block.ffnNorm.update(parameters: params, verify: .none)
        }

        var mixerWeights: [String: MLXArray] = [:]
        if let w = weights["\(prefix).mixer.conv.weight"] { mixerWeights["weight"] = w }
        if let b = weights["\(prefix).mixer.conv.bias"] { mixerWeights["bias"] = b }
        if !mixerWeights.isEmpty {
            let params = ModuleParameters.unflattened(mixerWeights)
            try? block.mixer.conv.update(parameters: params, verify: .none)
        }

        var ffnLinear1Weights: [String: MLXArray] = [:]
        if let w = weights["\(prefix).ffn.linear1.weight"] { ffnLinear1Weights["weight"] = w }
        if let b = weights["\(prefix).ffn.linear1.bias"] { ffnLinear1Weights["bias"] = b }
        if let s = weights["\(prefix).ffn.linear1.scales"] { ffnLinear1Weights["scales"] = s }
        if let bs = weights["\(prefix).ffn.linear1.biases"] { ffnLinear1Weights["biases"] = bs }
        if !ffnLinear1Weights.isEmpty {
            let params = ModuleParameters.unflattened(ffnLinear1Weights)
            do {
                try block.ffn.linear1.update(parameters: params, verify: .none)
            } catch {
                print("[DEBUG] Error loading linear1 weights for \(prefix): \(error)")
            }
        }

        var ffnLinear2Weights: [String: MLXArray] = [:]
        if let w = weights["\(prefix).ffn.linear2.weight"] { ffnLinear2Weights["weight"] = w }
        if let b = weights["\(prefix).ffn.linear2.bias"] { ffnLinear2Weights["bias"] = b }
        if let s = weights["\(prefix).ffn.linear2.scales"] { ffnLinear2Weights["scales"] = s }
        if let bs = weights["\(prefix).ffn.linear2.biases"] { ffnLinear2Weights["biases"] = bs }
        if !ffnLinear2Weights.isEmpty {
            let params = ModuleParameters.unflattened(ffnLinear2Weights)
            do {
                try block.ffn.linear2.update(parameters: params, verify: .none)
            } catch {
                print("[DEBUG] Error loading linear2 weights for \(prefix): \(error)")
            }
        }

        if let gamma = weights["\(prefix).gamma"] {
            block.gamma = gamma
        }
        if let ffnGamma = weights["\(prefix).ffnGamma"] {
            block.ffnGamma = ffnGamma
        }

    }
}

private func loadHeadWeightsForDecoder(_ decoder: TokenizerDecoder, weights: [String: MLXArray]) {
    let prefix = "acoustic_tokenizer.decoder.head"
    var headWeights: [String: MLXArray] = [:]
    if let w = weights["\(prefix).conv.weight"] { headWeights["weight"] = w }
    if let b = weights["\(prefix).conv.bias"] { headWeights["bias"] = b }
    if !headWeights.isEmpty {
        let params = ModuleParameters.unflattened(headWeights)
        try? decoder.head.conv.update(parameters: params, verify: .none)
    }
}

private func loadUpsampleLayersWeightsForDecoder(_ decoder: TokenizerDecoder, weights: [String: MLXArray]) {
    for (idx, layer) in decoder.upsampleLayers.enumerated() {
        if let sconv = layer as? SConv1d {
            let weightKey = "acoustic_tokenizer.decoder.upsampleLayers.\(idx).conv.weight"
            let biasKey = "acoustic_tokenizer.decoder.upsampleLayers.\(idx).conv.bias"

            var convWeights: [String: MLXArray] = [:]
            if let weight = weights[weightKey] {
                convWeights["weight"] = weight
            }
            if let bias = weights[biasKey] {
                convWeights["bias"] = bias
            }
            if !convWeights.isEmpty {
                let params = ModuleParameters.unflattened(convWeights)
                sconv.conv.update(parameters: params)
            }
        } else if let sconvtr = layer as? SConvTranspose1d {
            let weightKey = "acoustic_tokenizer.decoder.upsampleLayers.\(idx).convtr.weight"
            let biasKey = "acoustic_tokenizer.decoder.upsampleLayers.\(idx).convtr.bias"

            var convWeights: [String: MLXArray] = [:]
            if let weight = weights[weightKey] {
                convWeights["weight"] = weight
            }
            if let bias = weights[biasKey] {
                convWeights["bias"] = bias
            }
            if !convWeights.isEmpty {
                let params = ModuleParameters.unflattened(convWeights)
                sconvtr.convtr.update(parameters: params)
            }
        }
    }

    var weightsToEval: [MLXArray] = []
    weightsToEval.reserveCapacity(decoder.upsampleLayers.count * 2)
    for layer in decoder.upsampleLayers {
        if let sconv = layer as? SConv1d {
            weightsToEval.append(sconv.conv.weight)
            if let b = sconv.conv.bias { weightsToEval.append(b) }
        } else if let sconvtr = layer as? SConvTranspose1d {
            weightsToEval.append(sconvtr.convtr.weight)
            if let b = sconvtr.convtr.bias { weightsToEval.append(b) }
        }
    }
    eval(weightsToEval)
}

/// Load weights into a 1.5B unified-LM model. Mirrors `loadVibeVoiceStreamModel`
/// but targets the `VibeVoice15BModel` topology — single Qwen2 stack, dual
/// encoders, no tts_language_model / no eos_classifier.
public func loadVibeVoice15BModel(from directory: URL) throws -> VibeVoice15BModel {
    let config = try loadVibeVoiceConfiguration(from: directory)
    let model = try VibeVoice15BModel(config)

    let isQuantized = VibeVoiceQuantizer.hasQuantization(at: directory)
    var quantManifest: VibeVoiceQuantizationManifest?
    if isQuantized {
        let manifestURL = directory.appendingPathComponent("quantization.json")
        quantManifest = try VibeVoiceQuantizationManifest.load(from: manifestURL)
    }

    var weights = try loadWeightsFromDirectory(directory)
    weights = materializeWeights(weights)

    let scalingFactor = weights["model.speech_scaling_factor"]
    let biasFactor = weights["model.speech_bias_factor"]

    weights = weights.filter { !$0.key.contains("rotary_emb.inv_freq") }
    var mappedWeights = mapVibeVoiceWeightKeys(weights)
    mappedWeights = transposeConv1dWeights(mappedWeights)
    let availableKeys = Set(mappedWeights.keys)

    if let sf = scalingFactor { model.speechScalingFactor = sf }
    if let bf = biasFactor { model.speechBiasFactor = bf }

    if let manifest = quantManifest {
        VibeVoiceQuantizer.applyQuantization(
            to: model,
            manifest: manifest,
            availableKeys: availableKeys
        )
    }

    let parameters = ModuleParameters.unflattened(mappedWeights)
    try model.update(parameters: parameters, verify: .none)

    loadUpsampleLayersWeightsForDecoder(model.acousticTokenizer.decoder, weights: mappedWeights)
    loadHeadWeightsForDecoder(model.acousticTokenizer.decoder, weights: mappedWeights)

    let allParams = model.parameters().flattened().map { $0.1 }
    eval(allParams)
    eval([
        model.noiseScheduler.betas,
        model.noiseScheduler.alphas,
        model.noiseScheduler.alphasCumprod,
        model.noiseScheduler.alphaT,
        model.noiseScheduler.sigmaT,
        model.noiseScheduler.lambdaT,
        model.noiseScheduler.sigmas
    ])
    return model
}

public func loadVibeVoiceStreamModel(from directory: URL) throws -> VibeVoiceStreamModel {
    let config = try loadVibeVoiceConfiguration(from: directory)
    let model = try VibeVoiceStreamModel(config)

    let isQuantized = VibeVoiceQuantizer.hasQuantization(at: directory)
    var quantManifest: VibeVoiceQuantizationManifest?
    if isQuantized {
        let manifestURL = directory.appendingPathComponent("quantization.json")
        quantManifest = try VibeVoiceQuantizationManifest.load(from: manifestURL)
    }

    var weights = try loadWeightsFromDirectory(directory)
    weights = materializeWeights(weights)

    let scalingFactor = weights["model.speech_scaling_factor"]
    let biasFactor = weights["model.speech_bias_factor"]

    weights = weights.filter { !$0.key.contains("rotary_emb.inv_freq") }

    var mappedWeights = mapVibeVoiceWeightKeys(weights)
    mappedWeights = transposeConv1dWeights(mappedWeights)

    let availableKeys = Set(mappedWeights.keys)

    if let sf = scalingFactor {
        model.speechScalingFactor = sf
    }
    if let bf = biasFactor {
        model.speechBiasFactor = bf
    }

    // 1.5B bundles do not ship `tts_eos_classifier.*` weights — the variant
    // doesn't have an EOS classifier. Track presence so checkEndOfSpeech can
    // skip the classifier path when the weights aren't there.
    let hasEos = mappedWeights.keys.contains { $0.hasPrefix("tts_eos_classifier.") }
    model.hasEosClassifier = hasEos

    if let manifest = quantManifest {
        VibeVoiceQuantizer.applyQuantization(
            to: model,
            manifest: manifest,
            availableKeys: availableKeys
        )
    }

    let parameters = ModuleParameters.unflattened(mappedWeights)
    try model.update(parameters: parameters, verify: .none)

    loadUpsampleLayersWeightsForDecoder(model.acousticTokenizer.decoder, weights: mappedWeights)
    loadHeadWeightsForDecoder(model.acousticTokenizer.decoder, weights: mappedWeights)

    let allParams = model.parameters().flattened().map { $0.1 }
    eval(allParams)

    eval([
        model.noiseScheduler.betas,
        model.noiseScheduler.alphas,
        model.noiseScheduler.alphasCumprod,
        model.noiseScheduler.alphaT,
        model.noiseScheduler.sigmaT,
        model.noiseScheduler.lambdaT,
        model.noiseScheduler.sigmas
    ])

    return model
}

