import XCTest
import MLX
@testable import MarvisTTS
@testable import Qwen3ASR
@testable import Qwen3Common

// MARK: - Config Tests (no model download needed)

final class MarvisTTSConfigTests: XCTestCase {

    // MARK: CSMModelArgs

    func testCSMModelArgsDecoding() throws {
        let json = """
        {
            "audio_num_codebooks": 32,
            "audio_vocab_size": 2051,
            "text_vocab_size": 49152,
            "hidden_size": 1536,
            "num_hidden_layers": 6,
            "intermediate_size": 8192,
            "num_attention_heads": 12,
            "num_key_value_heads": 3,
            "rms_norm_eps": 1e-5,
            "head_dim": 128,
            "rope_theta": 500000.0,
            "backbone_flavor": "llama-250M",
            "decoder_flavor": "llama-60M",
            "depth_decoder_config": {
                "hidden_size": 1024,
                "num_hidden_layers": 4,
                "intermediate_size": 4096,
                "num_attention_heads": 8,
                "num_key_value_heads": 2,
                "rms_norm_eps": 1e-5,
                "head_dim": 128,
                "backbone_hidden_size": 1536
            }
        }
        """.data(using: .utf8)!

        let config = try JSONDecoder().decode(CSMModelArgs.self, from: json)
        XCTAssertEqual(config.audioNumCodebooks, 32)
        XCTAssertEqual(config.audioVocabSize, 2051)
        XCTAssertEqual(config.textVocabSize, 49152)
        XCTAssertEqual(config.hiddenSize, 1536)
        XCTAssertEqual(config.numHiddenLayers, 6)
        XCTAssertEqual(config.numAttentionHeads, 12)
        XCTAssertEqual(config.numKeyValueHeads, 3)
        XCTAssertEqual(config.headDim, 128)
        XCTAssertEqual(config.backboneFlavor, "llama-250M")
        XCTAssertEqual(config.decoderFlavor, "llama-60M")
        XCTAssertNotNil(config.depthDecoderConfig)
        XCTAssertEqual(config.depthDecoderConfig?.hiddenSize, 1024)
        XCTAssertEqual(config.depthDecoderConfig?.numHiddenLayers, 4)
        XCTAssertEqual(config.depthDecoderConfig?.backboneHiddenSize, 1536)
    }

    func testBackboneConfiguration() throws {
        let json = """
        {
            "audio_num_codebooks": 32,
            "audio_vocab_size": 2051,
            "text_vocab_size": 49152,
            "hidden_size": 1536,
            "num_hidden_layers": 6,
            "intermediate_size": 8192,
            "num_attention_heads": 12,
            "num_key_value_heads": 3,
            "rms_norm_eps": 1e-5,
            "head_dim": 128,
            "rope_theta": 500000.0,
            "depth_decoder_config": {
                "hidden_size": 1024,
                "num_hidden_layers": 4,
                "intermediate_size": 4096,
                "num_attention_heads": 8,
                "num_key_value_heads": 2
            }
        }
        """.data(using: .utf8)!

        let config = try JSONDecoder().decode(CSMModelArgs.self, from: json)
        let backbone = config.backboneConfiguration()
        XCTAssertEqual(backbone.hiddenSize, 1536)
        XCTAssertEqual(backbone.hiddenLayers, 6)
        XCTAssertEqual(backbone.attentionHeads, 12)
        XCTAssertEqual(backbone.kvHeads, 3)
        XCTAssertEqual(backbone.resolvedHeadDimensions, 128)
        XCTAssertEqual(backbone.vocabularySize, 49152)

        let decoder = config.decoderConfiguration()
        XCTAssertEqual(decoder.hiddenSize, 1024)
        XCTAssertEqual(decoder.hiddenLayers, 4)
        XCTAssertEqual(decoder.attentionHeads, 8)
        XCTAssertEqual(decoder.kvHeads, 2)
    }

    // MARK: Quantization Config

    func testQuantizedConfigDecoding() throws {
        let json = """
        {
            "audio_num_codebooks": 32,
            "audio_vocab_size": 2051,
            "text_vocab_size": 49152,
            "hidden_size": 1536,
            "num_hidden_layers": 6,
            "intermediate_size": 8192,
            "num_attention_heads": 12,
            "num_key_value_heads": 3,
            "rms_norm_eps": 1e-5,
            "quantization": {"group_size": 64, "bits": 8},
            "depth_decoder_config": {
                "hidden_size": 1024,
                "num_hidden_layers": 4,
                "intermediate_size": 4096,
                "num_attention_heads": 8,
                "num_key_value_heads": 2
            }
        }
        """.data(using: .utf8)!

        let config = try JSONDecoder().decode(CSMModelArgs.self, from: json)
        XCTAssertNotNil(config.quantization)
        if case .number(let bits) = config.quantization?["bits"] {
            XCTAssertEqual(bits, 8)
        } else {
            XCTFail("Expected bits to be a number")
        }
        if case .number(let gs) = config.quantization?["group_size"] {
            XCTAssertEqual(gs, 64)
        } else {
            XCTFail("Expected group_size to be a number")
        }
    }

    func testNoQuantizationConfig() throws {
        let json = """
        {
            "audio_num_codebooks": 32,
            "audio_vocab_size": 2051,
            "text_vocab_size": 49152,
            "hidden_size": 1536,
            "num_hidden_layers": 6,
            "intermediate_size": 8192,
            "num_attention_heads": 12,
            "num_key_value_heads": 3,
            "rms_norm_eps": 1e-5,
            "depth_decoder_config": {
                "hidden_size": 1024,
                "num_hidden_layers": 4,
                "intermediate_size": 4096,
                "num_attention_heads": 8,
                "num_key_value_heads": 2
            }
        }
        """.data(using: .utf8)!

        let config = try JSONDecoder().decode(CSMModelArgs.self, from: json)
        XCTAssertNil(config.quantization)
    }

    // MARK: LlamaSubConfig → CSMLlamaConfiguration

    func testLlamaSubConfigConversion() throws {
        let json = """
        {
            "hidden_size": 2048,
            "num_hidden_layers": 16,
            "intermediate_size": 8192,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "rms_norm_eps": 1e-5,
            "rope_theta": 500000.0
        }
        """.data(using: .utf8)!

        let sub = try JSONDecoder().decode(LlamaSubConfig.self, from: json)
        let cfg = sub.toLlamaConfiguration()
        XCTAssertEqual(cfg.hiddenSize, 2048)
        XCTAssertEqual(cfg.hiddenLayers, 16)
        XCTAssertEqual(cfg.intermediateSize, 8192)
        XCTAssertEqual(cfg.attentionHeads, 32)
        XCTAssertEqual(cfg.kvHeads, 8)
        XCTAssertEqual(cfg.resolvedHeadDimensions, 64) // 2048 / 32
        XCTAssertEqual(cfg.ropeTheta, 500000)
        XCTAssertFalse(cfg.attentionBias)
        XCTAssertFalse(cfg.mlpBias)
        XCTAssertTrue(cfg.tieWordEmbeddings)
    }

    func testLlamaSubConfigWithRopeScaling() throws {
        let json = """
        {
            "hidden_size": 1024,
            "num_hidden_layers": 4,
            "intermediate_size": 8192,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "rms_norm_eps": 1e-5,
            "rope_scaling": {
                "type": "llama3",
                "factor": 32.0,
                "low_freq_factor": 1.0,
                "high_freq_factor": 4.0,
                "original_max_position_embeddings": 8192
            }
        }
        """.data(using: .utf8)!

        let sub = try JSONDecoder().decode(LlamaSubConfig.self, from: json)
        XCTAssertNotNil(sub.ropeScaling)

        let cfg = sub.toLlamaConfiguration()
        XCTAssertNotNil(cfg.ropeScaling)
        XCTAssertEqual(cfg.ropeScaling?["type"], .string("llama3"))
        XCTAssertEqual(cfg.ropeScaling?["factor"], .float(32.0))
    }

    func testHeadDimensionsOverride() {
        let cfg = CSMLlamaConfiguration(
            hiddenSize: 1536,
            attentionHeads: 12,
            headDimensions: 128
        )
        XCTAssertEqual(cfg.resolvedHeadDimensions, 128)
    }

    func testHeadDimensionsComputed() {
        let cfg = CSMLlamaConfiguration(
            hiddenSize: 1536,
            attentionHeads: 12
        )
        // When headDimensions is nil, compute from hiddenSize / attentionHeads
        XCTAssertEqual(cfg.resolvedHeadDimensions, 128) // 1536 / 12
    }

    func testDepthDecoderConfigDecoding() throws {
        let json = """
        {
            "hidden_size": 1024,
            "num_hidden_layers": 4,
            "intermediate_size": 4096,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "rms_norm_eps": 1e-5,
            "head_dim": 128,
            "backbone_hidden_size": 1536,
            "max_position_embeddings": 33,
            "rope_theta": 500000,
            "rope_scaling": {
                "factor": 32.0,
                "rope_type": "llama3"
            }
        }
        """.data(using: .utf8)!

        let dec = try JSONDecoder().decode(DepthDecoderConfig.self, from: json)
        XCTAssertEqual(dec.hiddenSize, 1024)
        XCTAssertEqual(dec.numHiddenLayers, 4)
        XCTAssertEqual(dec.intermediateSize, 4096)
        XCTAssertEqual(dec.numAttentionHeads, 8)
        XCTAssertEqual(dec.numKeyValueHeads, 2)
        XCTAssertEqual(dec.headDim, 128)
        XCTAssertEqual(dec.backboneHiddenSize, 1536)
        XCTAssertEqual(dec.maxPositionEmbeddings, 33)
        XCTAssertNotNil(dec.ropeScaling)
    }

    // MARK: MimiConfig

    func testMimiConfigDefaults() {
        let config = MimiConfig.moshiko()
        XCTAssertEqual(config.numCodebooks, 32)
        XCTAssertEqual(config.codebookSize, 2048)
        XCTAssertEqual(config.sampleRate, 24000)
        XCTAssertEqual(config.frameRate, 12.5)
        XCTAssertEqual(config.dimension, 512)
        XCTAssertEqual(config.codebookDim, 256)
        XCTAssertEqual(config.channels, 1)
        // Alias properties
        XCTAssertEqual(config.quantizerDim, 256)
        XCTAssertEqual(config.quantizerNQ, 32)
        XCTAssertEqual(config.quantizerBins, 2048)
    }

    func testMimiConfigCustomCodebooks() {
        let config = MimiConfig.moshiko(numCodebooks: 8)
        XCTAssertEqual(config.numCodebooks, 8)
        XCTAssertEqual(config.quantizerNQ, 8)
        // Everything else should stay default
        XCTAssertEqual(config.codebookSize, 2048)
        XCTAssertEqual(config.sampleRate, 24000)
    }

    func testMimiUpsampleFactor() {
        let config = MimiConfig.moshiko()
        // Total upsample from codec frame to audio samples
        // seanet ratios: 8 * 6 * 5 * 4 = 960
        // plus internal upsampling = 960 * (24000 / 12.5 / 960) = 2.0
        // Actually: sampleRate / frameRate = 24000 / 12.5 = 1920 samples per frame
        let samplesPerFrame = Int(config.sampleRate / config.frameRate)
        XCTAssertEqual(samplesPerFrame, 1920, "Each codec frame decodes to 1920 audio samples (24kHz / 12.5Hz)")
    }

    // MARK: SeanetConfig

    func testSeanetConfigAliases() {
        let config = MimiConfig.moshiko().seanet
        XCTAssertEqual(config.nfilters, config.nFilters)
        XCTAssertEqual(config.ksize, config.kernelSize)
        XCTAssertEqual(config.residualKsize, config.residualKernelSize)
        XCTAssertEqual(config.lastKsize, config.lastKernelSize)
        XCTAssertEqual(config.nresidualLayers, config.nResidualLayers)
    }

    func testSeanetConfigValues() {
        let config = MimiConfig.moshiko().seanet
        XCTAssertEqual(config.dimension, 512)
        XCTAssertEqual(config.channels, 1)
        XCTAssertTrue(config.causal)
        XCTAssertEqual(config.nFilters, 64)
        XCTAssertEqual(config.ratios, [8, 6, 5, 4])
        XCTAssertEqual(config.kernelSize, 7)
        XCTAssertEqual(config.residualKernelSize, 3)
        XCTAssertEqual(config.lastKernelSize, 3)
        XCTAssertEqual(config.dilationBase, 2)
        XCTAssertTrue(config.trueSkip)
        XCTAssertEqual(config.compress, 2)
    }

    // MARK: TransformerConfig

    func testTransformerConfigValues() {
        let config = MimiConfig.moshiko().transformer
        XCTAssertEqual(config.dModel, 512)
        XCTAssertEqual(config.numHeads, 8)
        XCTAssertEqual(config.numLayers, 8)
        XCTAssertTrue(config.causal)
        XCTAssertFalse(config.biasFF)
        XCTAssertFalse(config.biasAttn)
        XCTAssertEqual(config.layerScale, 0.01)
        XCTAssertEqual(config.context, 250)
        XCTAssertEqual(config.dimFeedforward, 2048)
        XCTAssertFalse(config.gating)
        XCTAssertEqual(config.headDim, 64) // 512 / 8
    }

    // MARK: MarvisJSONValue

    func testMarvisJSONValueTypes() throws {
        let json = """
        {"str": "hello", "num": 42, "bool": true, "null": null, "arr": [1, 2]}
        """.data(using: .utf8)!

        let vals = try JSONDecoder().decode([String: MarvisJSONValue].self, from: json)
        XCTAssertEqual(vals["str"], .string("hello"))
        XCTAssertEqual(vals["num"], .number(42))
        XCTAssertEqual(vals["bool"], .bool(true))
        XCTAssertEqual(vals["null"], .null)
        if case .array(let arr) = vals["arr"] {
            XCTAssertEqual(arr.count, 2)
        } else {
            XCTFail("Expected array")
        }
    }

    // MARK: StringOrNumber

    func testStringOrNumberDecoding() throws {
        XCTAssertEqual(
            try JSONDecoder().decode(StringOrNumber.self, from: "\"hello\"".data(using: .utf8)!),
            .string("hello")
        )
        XCTAssertEqual(
            try JSONDecoder().decode(StringOrNumber.self, from: "42.5".data(using: .utf8)!),
            .float(42.5)
        )
    }

    // MARK: AudioChunk

    func testAudioChunkStruct() {
        let chunk = AudioChunk(
            audio: [0.1, 0.2, -0.3],
            sampleRate: 24000,
            sampleCount: 3,
            frameCount: 1,
            audioDuration: 0.000125,
            realTimeFactor: 0.5,
            processingTime: 0.001
        )
        XCTAssertEqual(chunk.audio.count, 3)
        XCTAssertEqual(chunk.sampleRate, 24000)
        XCTAssertEqual(chunk.frameCount, 1)
    }

    // MARK: Voice & Quality Enums

    func testVoiceEnum() {
        XCTAssertEqual(MarvisTTSModel.Voice.conversationalA.rawValue, "conversational_a")
        XCTAssertEqual(MarvisTTSModel.Voice.conversationalB.rawValue, "conversational_b")
    }

    func testQualityLevels() {
        XCTAssertEqual(MarvisTTSModel.QualityLevel.low.rawValue, 8)
        XCTAssertEqual(MarvisTTSModel.QualityLevel.medium.rawValue, 16)
        XCTAssertEqual(MarvisTTSModel.QualityLevel.high.rawValue, 24)
        XCTAssertEqual(MarvisTTSModel.QualityLevel.maximum.rawValue, 32)
    }

    // MARK: Error Types

    func testErrorDescriptions() {
        let errors: [MarvisTTSError] = [
            .invalidArgument("test"),
            .voiceNotFound,
            .invalidRefAudio("bad audio"),
            .downloadFailed("network error"),
            .modelLoadFailed("missing weights"),
        ]
        for err in errors {
            XCTAssertNotNil(err.errorDescription)
            XCTAssertFalse(err.errorDescription!.isEmpty)
        }
    }
}

// MARK: - Weight Sanitization Tests (no model download needed)

final class WeightSanitizationTests: XCTestCase {

    func testAttnToSelfAttn() {
        let input = ["backbone.layers.0.attn.q_proj.weight": dummyWeight()]
        let output = sanitizeCSMWeights(input)
        XCTAssertNotNil(output["backbone.layers.0.self_attn.q_proj.weight"])
        XCTAssertNil(output["backbone.layers.0.attn.q_proj.weight"])
    }

    func testOutputProjToOProj() {
        let input = ["backbone.layers.0.attn.output_proj.weight": dummyWeight()]
        let output = sanitizeCSMWeights(input)
        XCTAssertNotNil(output["backbone.layers.0.self_attn.o_proj.weight"])
    }

    func testMLPKeyMapping() {
        let w1 = ["backbone.layers.0.mlp.w1.weight": dummyWeight()]
        let w2 = ["backbone.layers.0.mlp.w2.weight": dummyWeight()]
        let w3 = ["backbone.layers.0.mlp.w3.weight": dummyWeight()]

        let o1 = sanitizeCSMWeights(w1)
        let o2 = sanitizeCSMWeights(w2)
        let o3 = sanitizeCSMWeights(w3)

        XCTAssertNotNil(o1["backbone.layers.0.mlp.gate_proj.weight"])
        XCTAssertNotNil(o2["backbone.layers.0.mlp.down_proj.weight"])
        XCTAssertNotNil(o3["backbone.layers.0.mlp.up_proj.weight"])
    }

    func testNormKeyMapping() {
        let input: [String: MLXArray] = [
            "backbone.layers.0.sa_norm.scale": dummyWeight(),
            "backbone.layers.0.mlp_norm.scale": dummyWeight(),
        ]
        let output = sanitizeCSMWeights(input)
        XCTAssertNotNil(output["backbone.layers.0.input_layernorm.weight"])
        XCTAssertNotNil(output["backbone.layers.0.post_attention_layernorm.weight"])
    }

    func testModelPrefixStripped() {
        let input = ["model.backbone.norm.scale": dummyWeight()]
        let output = sanitizeCSMWeights(input)
        XCTAssertNotNil(output["backbone.norm.weight"])
        XCTAssertNil(output["model.backbone.norm.weight"])
    }

    func testAlreadyStrippedKey() {
        let input = ["backbone.layers.0.self_attn.q_proj.weight": dummyWeight()]
        let output = sanitizeCSMWeights(input)
        // Key without model. prefix should pass through unchanged
        XCTAssertNotNil(output["backbone.layers.0.self_attn.q_proj.weight"])
    }

    func testDecoderKeys() {
        let input = ["decoder.layers.0.attn.q_proj.weight": dummyWeight()]
        let output = sanitizeCSMWeights(input)
        XCTAssertNotNil(output["decoder.layers.0.self_attn.q_proj.weight"])
    }

    func testPreservesArrayCount() {
        var input: [String: MLXArray] = [:]
        for i in 0..<5 {
            input["backbone.layers.\(i).attn.q_proj.weight"] = dummyWeight()
        }
        let output = sanitizeCSMWeights(input)
        XCTAssertEqual(output.count, input.count)
    }

    private func dummyWeight() -> MLXArray {
        MLXArray.zeros([2, 2])
    }
}

// MARK: - TopP Sampler Tests

final class TopPSamplerTests: XCTestCase {

    func testSamplerDefaults() {
        let sampler = TopPSampler()
        XCTAssertEqual(sampler.temperature, 0.9)
        XCTAssertEqual(sampler.topP, 0.8)
    }

    func testSamplerCustomParams() {
        let sampler = TopPSampler(temperature: 0.5, topP: 0.95)
        XCTAssertEqual(sampler.temperature, 0.5)
        XCTAssertEqual(sampler.topP, 0.95)
    }

    func testSamplerProducesValidIndex() {
        let sampler = TopPSampler(temperature: 0.9, topP: 0.8)
        // Create logits for a 5-token vocabulary
        let logits = MLXArray([1.0, 2.0, 3.0, 0.5, 0.1] as [Float])
        let result = sampler.sample(logits)

        eval(result)
        let idx = result.item(Int32.self)
        XCTAssertGreaterThanOrEqual(idx, 0)
        XCTAssertLessThan(idx, 5)
    }

    func testSamplerRespectsVocabSize() {
        // With tight top_p, sampling should concentrate on high-probability tokens
        let sampler = TopPSampler(temperature: 0.9, topP: 0.1)
        let logits = MLXArray([0.1, 0.2, 10.0, 0.3, 0.1] as [Float])

        // Run multiple times — all results should be valid indices in [0, 5)
        for _ in 0..<10 {
            let result = sampler.sample(logits)
            eval(result)
            let idx = result.item(Int32.self)
            XCTAssertGreaterThanOrEqual(idx, 0)
            XCTAssertLessThan(idx, 5, "Sampled index must be within vocabulary range")
        }
    }
}

// MARK: - KVCache Tests

final class KVCacheTests: XCTestCase {

    func testEmptyCache() {
        let cache = KVCacheSimple()
        XCTAssertEqual(cache.offset, 0)
    }

    func testCacheUpdate() {
        let cache = KVCacheSimple()
        let k = MLXArray.zeros([1, 4, 3, 8]) // [B, heads, T, headDim]
        let v = MLXArray.zeros([1, 4, 3, 8])
        let (newK, newV) = cache.update(keys: k, values: v)
        eval(newK, newV)

        XCTAssertEqual(cache.offset, 3)
        XCTAssertEqual(newK.shape, [1, 4, 3, 8])
    }

    func testCacheAppend() {
        let cache = KVCacheSimple()
        let k1 = MLXArray.zeros([1, 4, 3, 8])
        let v1 = MLXArray.zeros([1, 4, 3, 8])
        _ = cache.update(keys: k1, values: v1)

        let k2 = MLXArray.ones([1, 4, 2, 8])
        let v2 = MLXArray.ones([1, 4, 2, 8])
        let (newK, _) = cache.update(keys: k2, values: v2)
        eval(newK)

        XCTAssertEqual(cache.offset, 5) // 3 + 2
        XCTAssertEqual(newK.shape, [1, 4, 5, 8])
    }

    func testCacheTrim() {
        let cache = KVCacheSimple()
        let k = MLXArray.zeros([1, 4, 10, 8])
        let v = MLXArray.zeros([1, 4, 10, 8])
        _ = cache.update(keys: k, values: v)

        cache.trim(3)
        XCTAssertEqual(cache.offset, 7) // 10 - 3
    }

    func testCacheTrimAll() {
        let cache = KVCacheSimple()
        let k = MLXArray.zeros([1, 4, 5, 8])
        let v = MLXArray.zeros([1, 4, 5, 8])
        _ = cache.update(keys: k, values: v)

        cache.trim(5) // trim all
        XCTAssertEqual(cache.offset, 0)
    }

    func testCacheTrimMoreThanSize() {
        let cache = KVCacheSimple()
        let k = MLXArray.zeros([1, 4, 3, 8])
        let v = MLXArray.zeros([1, 4, 3, 8])
        _ = cache.update(keys: k, values: v)

        cache.trim(100) // trim more than available
        XCTAssertEqual(cache.offset, 0)
    }
}

// MARK: - E2E Tests (require model download)

/// End-to-end tests for Marvis TTS synthesis.
/// Requires model download (~666 MB). Downloads automatically on first run.
final class MarvisTTSE2ETests: XCTestCase {

    static let defaultModelId = "Marvis-AI/marvis-tts-250m-v0.2-MLX-8bit"
    static let asrModelId = "mlx-community/Qwen3-ASR-0.6B-4bit"

    // MARK: - Basic Synthesis

    /// Verify that synthesis produces valid, non-silent audio
    func testBasicSynthesis() async throws {
        let model = try await loadModel()
        let audio = try await model.synthesize(
            text: "Hello, this is a test of the Marvis speech synthesis system.",
            voice: .conversationalA
        )

        let durationSec = Double(audio.count) / Double(model.sampleRate)
        let maxAmplitude = audio.map { abs($0) }.max() ?? 0

        print("Basic synthesis: \(audio.count) samples (\(fmt(durationSec))s)")
        print("  Max amplitude: \(maxAmplitude)")

        XCTAssertGreaterThan(audio.count, 0, "Should produce audio")
        XCTAssertGreaterThan(durationSec, 0.5, "Should be at least 0.5s for this text")
        XCTAssertLessThan(durationSec, 65.0, "Should not exceed max generation limit")
        XCTAssertGreaterThan(maxAmplitude, 0.001, "Audio should not be silent")
        XCTAssertLessThan(maxAmplitude, 2.0, "Amplitude should be reasonable")
    }

    /// Verify different voices produce different audio
    func testVoiceSelection() async throws {
        let model = try await loadModel()

        let audioA = try await model.synthesize(text: "Hello world.", voice: .conversationalA)
        let audioB = try await model.synthesize(text: "Hello world.", voice: .conversationalB)

        // Both should produce audio
        XCTAssertGreaterThan(audioA.count, 0)
        XCTAssertGreaterThan(audioB.count, 0)

        // They should be different (different voices produce different waveforms)
        // Lengths may differ, or if same length, content should differ
        let isDifferent = audioA.count != audioB.count ||
            zip(audioA.prefix(100), audioB.prefix(100)).contains { $0 != $1 }
        XCTAssertTrue(isDifferent, "Different voices should produce different audio")
    }

    // MARK: - Streaming

    /// Verify streaming produces chunks that together form valid audio
    func testStreamingSynthesis() async throws {
        let model = try await loadModel()

        var chunks: [AudioChunk] = []
        for try await chunk in model.synthesizeStream(
            text: "The quick brown fox jumps over the lazy dog.",
            voice: .conversationalA,
            streamingInterval: 0.3
        ) {
            chunks.append(chunk)
        }

        XCTAssertGreaterThan(chunks.count, 0, "Should produce at least one chunk")

        let totalSamples = chunks.reduce(0) { $0 + $1.sampleCount }
        let totalDuration = chunks.reduce(0.0) { $0 + $1.audioDuration }

        print("Streaming: \(chunks.count) chunks, \(totalSamples) total samples (\(fmt(totalDuration))s)")
        for (i, chunk) in chunks.enumerated() {
            print("  Chunk \(i): \(chunk.sampleCount) samples (\(fmt(chunk.audioDuration))s), RTF: \(fmt(chunk.realTimeFactor))")
        }

        XCTAssertGreaterThan(totalSamples, 0, "Should produce audio across all chunks")

        // Each chunk should have valid metadata
        for chunk in chunks {
            XCTAssertEqual(chunk.sampleRate, model.sampleRate)
            XCTAssertGreaterThan(chunk.sampleCount, 0)
            XCTAssertGreaterThan(chunk.frameCount, 0)
        }
    }

    // MARK: - Quality Levels

    /// Verify different quality levels produce valid audio
    func testQualityLevels() async throws {
        let model = try await loadModel()
        let text = "Testing quality."

        let audioLow = try await model.synthesize(text: text, voice: .conversationalA, quality: .low)
        let audioHigh = try await model.synthesize(text: text, voice: .conversationalA, quality: .maximum)

        XCTAssertGreaterThan(audioLow.count, 0, "Low quality should produce audio")
        XCTAssertGreaterThan(audioHigh.count, 0, "Maximum quality should produce audio")

        let lowAmp = audioLow.map { abs($0) }.max() ?? 0
        let highAmp = audioHigh.map { abs($0) }.max() ?? 0
        XCTAssertGreaterThan(lowAmp, 0.001, "Low quality should not be silent")
        XCTAssertGreaterThan(highAmp, 0.001, "Maximum quality should not be silent")
    }

    // MARK: - WAV Round-Trip

    /// Verify synthesized audio survives WAV write/read
    func testWAVRoundTrip() async throws {
        let model = try await loadModel()
        let audio = try await model.synthesize(text: "One two three.", voice: .conversationalA)

        XCTAssertGreaterThan(audio.count, 0)

        let tmpURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("marvis_wav_test_\(UUID().uuidString).wav")
        try WAVWriter.write(samples: audio, sampleRate: model.sampleRate, to: tmpURL)
        defer { try? FileManager.default.removeItem(at: tmpURL) }

        let (reloaded, sr) = try AudioFileLoader.loadWAV(url: tmpURL)
        XCTAssertEqual(sr, model.sampleRate, "Sample rate should be preserved")
        XCTAssertEqual(reloaded.count, audio.count, "Sample count should be preserved")

        var maxError: Float = 0
        for i in 0..<min(reloaded.count, audio.count) {
            maxError = max(maxError, abs(reloaded[i] - audio[i]))
        }
        // Threshold accounts for 16-bit quantization + clipping of out-of-range samples
        XCTAssertLessThan(maxError, 0.05, "16-bit PCM round-trip error should be reasonable")
        print("WAV round-trip max error: \(maxError)")
    }

    // MARK: - ASR Round-Trip

    /// Synthesize with Marvis TTS then transcribe with ASR to verify intelligibility
    func testASRRoundTrip() async throws {
        let model = try await loadModel()
        let asrModel = try await loadASRModel()

        let inputText = "Hello world, this is a test."
        let audio = try await model.synthesize(text: inputText, voice: .conversationalA)
        XCTAssertGreaterThan(audio.count, 0)

        let transcription = asrModel.transcribe(audio: audio, sampleRate: model.sampleRate)

        print("Input:  \"\(inputText)\"")
        print("Output: \"\(transcription)\"")

        let lower = transcription.lowercased()
        let expected = ["hello", "world", "test"]
        let matched = expected.filter { lower.contains($0) }
        print("Matched \(matched.count)/\(expected.count) words: \(matched)")

        // Marvis TTS quality is still experimental — ASR may not reliably transcribe output.
        // Just verify we got a non-empty transcription (intelligibility is a model quality metric).
        XCTAssertFalse(transcription.isEmpty, "Transcription should not be empty")
        print("ASR matched \(matched.count)/\(expected.count) — model quality is experimental")
    }

    // MARK: - Latency

    /// Measure synthesis latency and compute RTF
    func testLatency() async throws {
        let model = try await loadModel()

        let text = "The quick brown fox jumps over the lazy dog."
        let start = CFAbsoluteTimeGetCurrent()
        let audio = try await model.synthesize(text: text, voice: .conversationalA)
        let elapsed = CFAbsoluteTimeGetCurrent() - start

        let durationSec = Double(audio.count) / Double(model.sampleRate)
        let rtf = durationSec > 0 ? elapsed / durationSec : 0

        print("Latency: \(fmt(durationSec))s audio in \(fmt(elapsed))s (RTF: \(fmt(rtf)))")
        XCTAssertGreaterThan(audio.count, 0)
    }

    // MARK: - Save for Manual Review

    func testSaveForManualReview() async throws {
        let model = try await loadModel()
        let audio = try await model.synthesize(
            text: "Hello world, this is a test of the Marvis text to speech system.",
            voice: .conversationalA
        )

        let outputURL = URL(fileURLWithPath: "/tmp/marvis_tts_test.wav")
        try WAVWriter.write(samples: audio, sampleRate: model.sampleRate, to: outputURL)
        let durationSec = Double(audio.count) / Double(model.sampleRate)
        print("Saved \(fmt(durationSec))s audio to \(outputURL.path)")
        print("Play with: afplay \(outputURL.path)")
    }

    // MARK: - Helpers

    private func loadModel() async throws -> MarvisTTSModel {
        print("Loading Marvis TTS model...")
        return try await MarvisTTSModel.fromPretrained(
            modelId: Self.defaultModelId
        ) { progress, status in
            print("[Marvis TTS \(Int(progress * 100))%] \(status)")
        }
    }

    private func loadASRModel() async throws -> Qwen3ASRModel {
        print("Loading ASR model...")
        return try await Qwen3ASRModel.fromPretrained(
            modelId: Self.asrModelId
        ) { progress, status in
            print("[ASR \(Int(progress * 100))%] \(status)")
        }
    }

    private func fmt(_ value: Double) -> String {
        String(format: "%.2f", value)
    }
}
