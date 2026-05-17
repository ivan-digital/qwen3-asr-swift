import XCTest
import Foundation
import Hub
import MLX
import MLXNN
import Tokenizers
@testable import VoxCPM2TTS

final class StubVoxCPM2Tokenizer: Tokenizer {
    let idToToken: [Int: String]
    let tokenToId: [String: Int]
    let tokenizations: [String: [String]]

    init(_ mapping: [Int: String], tokenizations: [String: [String]] = [:]) {
        self.idToToken = mapping
        self.tokenToId = Dictionary(uniqueKeysWithValues: mapping.map { ($0.value, $0.key) })
        self.tokenizations = tokenizations
    }

    func tokenize(text: String) -> [String] { tokenizations[text] ?? text.map(String.init) }
    func encode(text: String) -> [Int] { encode(text: text, addSpecialTokens: true) }
    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        let ids = text.map { String($0) }.compactMap { tokenToId[$0] }
        guard addSpecialTokens else { return ids }
        return [bosTokenId ?? 1] + ids
    }
    func decode(tokens: [Int], skipSpecialTokens: Bool) -> String {
        tokens.compactMap { idToToken[$0] }.joined()
    }
    func convertTokenToId(_ token: String) -> Int? { tokenToId[token] }
    func convertIdToToken(_ id: Int) -> String? { idToToken[id] }
    var bosToken: String? { "<s>" }
    var bosTokenId: Int? { 1 }
    var eosToken: String? { "</s>" }
    var eosTokenId: Int? { 2 }
    var unknownToken: String? { "<unk>" }
    var unknownTokenId: Int? { 0 }
    var fuseUnknownTokens: Bool { false }
    func applyChatTemplate(messages: [Message]) throws -> [Int] { [] }
    func applyChatTemplate(messages: [Message], tools: [ToolSpec]?) throws -> [Int] { [] }
    func applyChatTemplate(messages: [Message], tools: [ToolSpec]?, additionalContext: [String : any Sendable]?) throws -> [Int] { [] }
    func applyChatTemplate(messages: [Message], chatTemplate: ChatTemplateArgument) throws -> [Int] { [] }
    func applyChatTemplate(messages: [Message], chatTemplate: String) throws -> [Int] { [] }
    func applyChatTemplate(messages: [Message], chatTemplate: ChatTemplateArgument?, addGenerationPrompt: Bool, truncation: Bool, maxLength: Int?, tools: [ToolSpec]?) throws -> [Int] { [] }
    func applyChatTemplate(messages: [Message], chatTemplate: ChatTemplateArgument?, addGenerationPrompt: Bool, truncation: Bool, maxLength: Int?, tools: [ToolSpec]?, additionalContext: [String : any Sendable]?) throws -> [Int] { [] }
}

final class VoxCPM2TTSConfigTests: XCTestCase {
    func testModelArgsRoundTripAndLoad() throws {
        var args = ModelArgs()
        args.lmConfig.hiddenSize = 1536
        args.lmConfig.numHiddenLayers = 12
        args.encoderConfig.numLayers = 3
        args.ditConfig.numLayers = 6
        args.audioVAEConfig.outSampleRate = 44_100
        args.scalarQuantizationLatentDim = 256
        args.maxLength = 4096

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let data = try encoder.encode(args)

        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let configURL = directory.appendingPathComponent("config.json")
        try data.write(to: configURL, options: .atomic)

        let loaded = try ModelArgs.load(from: directory)
        XCTAssertEqual(loaded.lmConfig.hiddenSize, 1536)
        XCTAssertEqual(loaded.lmConfig.numHiddenLayers, 12)
        XCTAssertEqual(loaded.lmConfig.kvChannels, 128)
        XCTAssertEqual(loaded.encoderConfig.numLayers, 3)
        XCTAssertEqual(loaded.encoderConfig.kvChannels, 128)
        XCTAssertEqual(loaded.ditConfig.numLayers, 6)
        XCTAssertEqual(loaded.ditConfig.kvChannels, 128)
        XCTAssertTrue(loaded.residualLMNoRope)
        XCTAssertEqual(loaded.audioVAEConfig.outSampleRate, 44_100)
        XCTAssertEqual(loaded.scalarQuantizationLatentDim, 256)
        XCTAssertEqual(loaded.maxLength, 4096)
    }

    func testModelArgsLoadsOfficialStyleConfigWithoutLegacyLmKeys() throws {
        let json = """
        {
          "lm_config": {
            "bos_token_id": 1,
            "eos_token_id": 2,
            "hidden_size": 2048,
            "intermediate_size": 6144,
            "max_position_embeddings": 32768,
            "num_attention_heads": 16,
            "num_hidden_layers": 28,
            "num_key_value_heads": 2,
            "rms_norm_eps": 1e-05,
            "rope_theta": 10000,
            "kv_channels": 128,
            "rope_scaling": {
              "type": "longrope",
              "long_factor": [1.0],
              "short_factor": [1.0]
            },
            "vocab_size": 73448,
            "use_mup": false,
            "scale_emb": 12,
            "dim_model_base": 256,
            "scale_depth": 1.4
          },
          "patch_size": 4,
          "feat_dim": 64,
          "scalar_quantization_latent_dim": 512,
          "scalar_quantization_scale": 9,
          "residual_lm_num_layers": 8,
          "residual_lm_no_rope": true,
          "encoder_config": {
            "hidden_dim": 1024,
            "ffn_dim": 4096,
            "num_heads": 16,
            "num_layers": 12,
            "kv_channels": 128
          },
          "dit_config": {
            "hidden_dim": 1024,
            "ffn_dim": 4096,
            "num_heads": 16,
            "num_layers": 12,
            "kv_channels": 128,
            "mean_mode": false,
            "cfm_config": {
              "sigma_min": 1e-06,
              "solver": "euler",
              "t_scheduler": "log-norm",
              "inference_cfg_rate": 2.0
            }
          },
          "audio_vae_config": {
            "encoder_dim": 128,
            "encoder_rates": [2, 5, 8, 8],
            "latent_dim": 64,
            "decoder_dim": 2048,
            "decoder_rates": [8, 6, 5, 2, 2, 2],
            "sr_bin_boundaries": [20000, 30000, 40000],
            "sample_rate": 16000,
            "out_sample_rate": 48000
          },
          "max_length": 8192,
          "model_type": "voxcpm2"
        }
        """.data(using: .utf8)!

        let decoded = try JSONDecoder().decode(ModelArgs.self, from: json)
        XCTAssertEqual(decoded.lmConfig.hiddenSize, 2048)
        XCTAssertEqual(decoded.lmConfig.kvChannels, 128)
        XCTAssertEqual(decoded.lmConfig.noRope, false)
        XCTAssertEqual(decoded.lmConfig.originalMaxPositionEmbeddings, 32768)
        XCTAssertEqual(decoded.residualLMNoRope, true)
        XCTAssertEqual(decoded.audioVAEConfig.outSampleRate, 48_000)
    }

    func testRopeScalingCodableSnakeCase() throws {
        let scaling = RopeScalingConfig()
        var copy = scaling
        copy.shortFactor = [1.0, 2.0]
        copy.longFactor = [3.0, 4.0]
        copy.originalMaxPositionEmbeddings = 8192

        let data = try JSONEncoder().encode(copy)
        let decoded = try JSONDecoder().decode(RopeScalingConfig.self, from: data)

        XCTAssertEqual(decoded.type, "longrope")
        XCTAssertEqual(decoded.shortFactor, [1.0, 2.0])
        XCTAssertEqual(decoded.longFactor, [3.0, 4.0])
        XCTAssertEqual(decoded.originalMaxPositionEmbeddings, 8192)
    }

    func testAudioVAEConfigDefaults() {
        let config = AudioVAEConfig()
        XCTAssertEqual(config.encoderDim, 128)
        XCTAssertEqual(config.encoderRates, [2, 5, 8, 8])
        XCTAssertEqual(config.latentDim, 64)
        XCTAssertEqual(config.decoderRates, [8, 6, 5, 2, 2, 2])
        XCTAssertEqual(config.sampleRate, 16_000)
        XCTAssertEqual(config.outSampleRate, 48_000)
        XCTAssertEqual(config.srBinBoundaries, [20_000, 30_000, 40_000])
    }

    func testAudioVAECastPromotesParametersToFloat32() throws {
        let vae = AudioVAE(AudioVAEConfig())

        // Seed the weight via the supported `update(parameters:)` path so MLX
        // rebuilds its items cache against the new array reference. Direct
        // property assignment bypasses the cache, and the subsequent
        // `apply(filter:map:)` traversal in castParametersToFloat32() can run
        // against a stale reference depending on Module.items() lazy-build
        // timing.
        let bf16Weight = MLXArray.ones(vae.decoder.conv_out.weight.shape, dtype: .bfloat16)
        try vae.decoder.conv_out.update(
            parameters: ModuleParameters.unflattened(["weight": bf16Weight]),
            verify: .shapeMismatch
        )
        XCTAssertEqual(vae.decoder.conv_out.weight.dtype, .bfloat16)

        vae.castParametersToFloat32()

        XCTAssertEqual(vae.decoder.conv_out.weight.dtype, .float32)
    }

    func testTokenizerConfigOverlayRewritesTokenizerClassToLlamaTokenizer() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let configURL = directory.appendingPathComponent("tokenizer_config.json")
        let tokenizerDataURL = directory.appendingPathComponent("tokenizer.json")

        let config = """
        {
          "tokenizer_class": "VoxCPM2Tokenizer",
          "auto_map": {
            "AutoTokenizer": ["tokenization_voxcpm2.VoxCPM2Tokenizer", null]
          }
        }
        """
        try config.data(using: .utf8)!.write(to: configURL, options: .atomic)
        try "{}".data(using: .utf8)!.write(to: tokenizerDataURL, options: .atomic)

        try VoxCPM2TTSModel.applyTokenizerConfigOverlay(in: directory)

        let patchedData = try Data(contentsOf: configURL)
        let patched = try JSONSerialization.jsonObject(with: patchedData) as? [String: Any]
        XCTAssertEqual(patched?["tokenizer_class"] as? String, "LlamaTokenizer")
        XCTAssertNil(patched?["auto_map"])
        XCTAssertFalse(try VoxCPM2TTSModel.tokenizerSnapshotNeedsRefresh(in: directory))
    }

    func testVoxCPM2TokenizerCompatibilityPathPromotesTokenizerClassToLlamaTokenizer() {
        let tokenizerConfig = Config([
            "tokenizer_class": "VoxCPM2Tokenizer",
            "legacy": true,
            "add_bos_token": true,
            "add_eos_token": false
        ])

        let compatible = VoxCPM2TTSModel.compatibleTokenizerConfig(for: tokenizerConfig)
        let compatibleDict = compatible.dictionary(or: [:])

        XCTAssertEqual(compatibleDict[Config.Key("tokenizer_class")]?.string(), "LlamaTokenizer")
        XCTAssertEqual(compatibleDict[Config.Key("legacy")]?.boolean(), true)
        XCTAssertEqual(compatibleDict[Config.Key("add_bos_token")]?.boolean(), true)
        XCTAssertEqual(compatibleDict[Config.Key("add_eos_token")]?.boolean(), false)
    }

    func testVoxCPM2ChineseTokenSplitMapExpandsMultiCharTokens() {
        let tokenizer = StubVoxCPM2Tokenizer([
            0: "<unk>",
            1: "你",
            2: "好",
            3: "你好",
            4: "▁你好",
            5: "Hello",
            6: "▁Hello"
        ])

        let splitMap = VoxCPM2TTSModel.buildVoxCPM2TokenizerSplitMap(tokenizer: tokenizer, vocabSize: 7)
        XCTAssertEqual(splitMap[3], [1, 2])
        XCTAssertEqual(splitMap[4], [1, 2])
        XCTAssertNil(splitMap[5])

        let expanded = VoxCPM2TTSModel.expandVoxCPM2TokenizerIds([4, 5, 3], using: splitMap)
        XCTAssertEqual(expanded, [1, 2, 5, 1, 2])
    }

    func testTokenizeMatchesWrappedTokenizerLikeUpstream() throws {
        let tokenizer = StubVoxCPM2Tokenizer([
            0: "<unk>",
            1: "<s>",
            2: "H",
            3: "i"
        ])

        let model = VoxCPM2TTSModel(args: ModelArgs())
        model.setTokenizer(tokenizer)

        let ids = try model.tokenize("Hi")
        XCTAssertEqual(ids, [2, 3])
    }

    func testTokenizePreservesCJKSplitMapWithLlamaStyleTokens() throws {
        let tokenizer = StubVoxCPM2Tokenizer(
            [
                0: "<unk>",
                1: "你",
                2: "好",
                3: "▁你好",
                4: "▁Hello",
            ],
            tokenizations: [
                "你好 Hello": ["▁你好", "▁Hello"]
            ]
        )

        let model = VoxCPM2TTSModel(args: ModelArgs())
        model.setTokenizer(tokenizer)

        let ids = try model.tokenize("你好 Hello")
        XCTAssertEqual(ids, [1, 2, 4])
    }

    func testUnifiedCFMTimeSpanMatchesUpstreamSwaySchedule() {
        let schedule = makeUnifiedCFMTimeSpan(timesteps: 10, scheduler: "log-norm", sigmaMin: 1e-6)
        let expected: [Float] = (0...10).map { step in
            let progress = Double(step) / 10.0
            let t = 1.0 - progress
            return Float(t + (cos(Double.pi / 2.0 * t) - 1.0 + t))
        }

        XCTAssertEqual(schedule.count, expected.count)
        for (actual, expectedValue) in zip(schedule, expected) {
            XCTAssertEqual(actual, expectedValue, accuracy: Float(1e-6))
        }
    }

    func testMiniCPMLongRoPEUsesLongFactorForOfficialVoxCPM2Config() {
        var config = LMConfig()
        config.hiddenSize = 4
        config.intermediateSize = 8
        config.numAttentionHeads = 2
        config.numKeyValueHeads = 1
        config.kvChannels = 2
        config.maxPositionEmbeddings = 32_768
        config.originalMaxPositionEmbeddings = 8_192
        config.ropeTheta = 10_000
        config.ropeScaling = RopeScalingConfig()
        config.ropeScaling?.originalMaxPositionEmbeddings = 8_192
        config.ropeScaling?.shortFactor = [1.0]
        config.ropeScaling?.longFactor = [2.0]

        let rope = MiniCPMLongRoPE(config: config)
        let positionIds = MLXArray([Int32(0), Int32(1), Int32(2)])
        let (cosEmb, _) = rope(positionIds)

        let scale = Float(sqrt(1.0 + log(32_768.0 / 8_192.0) / log(8_192.0)))
        let expectedLong = Float(Foundation.cos(0.5)) * scale
        XCTAssertEqual(cosEmb[1, 0].item(Float.self), expectedLong, accuracy: 1e-5)
    }
}

final class VoxCPM2TTSLayerTests: XCTestCase {
    func testScalarQuantizationLayerInitializes() throws {
        // Linear.shape is `(outputFeatures, inputFeatures)`, so:
        //   in_proj  : inDim=2 → latentDim=4  → shape = (4, 2)
        //   out_proj : latentDim=4 → outDim=3 → shape = (3, 4)
        let layer = ScalarQuantizationLayer(inDim: 2, outDim: 3, latentDim: 4, scale: 9)
        XCTAssertEqual(layer.scale, 9)
        XCTAssertEqual(layer.in_proj.shape.0, 4)
        XCTAssertEqual(layer.in_proj.shape.1, 2)
        XCTAssertEqual(layer.out_proj.shape.0, 3)
        XCTAssertEqual(layer.out_proj.shape.1, 4)
    }
}

final class VoxCPM2TTSQuantizationTests: XCTestCase {
    private final class PromotionProbe: Module {
        @ModuleInfo var dense: Linear
        @ModuleInfo var quantized: QuantizedLinear

        override init() {
            let denseWeight = MLXArray(Array(repeating: Float(1.0), count: 32 * 32), [32, 32]).asType(.bfloat16)
            let dense = Linear(weight: denseWeight, bias: nil)
            self._dense = ModuleInfo(wrappedValue: dense)
            self._quantized = ModuleInfo(wrappedValue: QuantizedLinear(dense, groupSize: 32, bits: 4))
            super.init()
        }
    }

    func testFloat32PromotionSkipsQuantizedModules() {
        let probe = PromotionProbe()
        XCTAssertEqual(probe.dense.weight.dtype, .bfloat16)
        XCTAssertEqual(probe.quantized.weight.dtype, .uint32)

        _ = probe.apply(filter: { module, _, _ in
            !(module is Quantized)
        }) { array in
            array.asType(.float32)
        }

        XCTAssertEqual(probe.dense.weight.dtype, .float32)
        XCTAssertEqual(probe.quantized.weight.dtype, .uint32)
    }
}

// MARK: - End-to-End Tests
//
// Each variant downloads ~2–5 GB on first run. Filtered out of CI via `--skip E2E`.
// Run locally with: swift test --filter E2EVoxCPM2TTSTests

import AudioCommon
import MLX
@testable import VoxCPM2TTS

final class E2EVoxCPM2TTSTests: XCTestCase {

    override func tearDown() {
        super.tearDown()
        // Free the previous variant's MLX buffers before the next test loads
        // its model — otherwise running bf16 + int8 + int4 in one process
        // blows past the GPU memory budget.
        Memory.clearCache()
    }

    private func runBasicSynthesis(modelId: String, file: StaticString = #filePath, line: UInt = #line) async throws {
        let model = try await VoxCPM2TTSModel.fromPretrained(modelId: modelId)
        defer { model.unload() }
        // Cap maxTokens to bound test runtime — the stop head doesn't always
        // fire quickly for short prompts on every variant. Keep
        // inferenceTimesteps + minTokens at defaults (10 / 2) so the Euler
        // CFM solver converges and produces non-NaN audio.
        let audio = try await model.generateVoxCPM2(text: "Hello.", maxTokens: 50)

        XCTAssertFalse(audio.isEmpty, "Should produce audio", file: file, line: line)
        XCTAssertEqual(model.sampleRate, 48_000, file: file, line: line)
        let duration = Double(audio.count) / Double(model.sampleRate)
        XCTAssertGreaterThan(duration, 0.05, "Should be at least 50ms", file: file, line: line)

        let maxAmp = audio.map { abs($0) }.max() ?? 0
        XCTAssertTrue(maxAmp.isFinite, "Audio must not contain NaN/Inf", file: file, line: line)
        XCTAssertGreaterThan(maxAmp, 0.001, "Should not be silent", file: file, line: line)
    }

    func testBasicSynthesisBF16() async throws {
        try await runBasicSynthesis(modelId: "aufklarer/VoxCPM2-MLX-bf16")
    }

    func testBasicSynthesisInt8() async throws {
        try await runBasicSynthesis(modelId: "aufklarer/VoxCPM2-MLX-int8")
    }

    func testBasicSynthesisInt4() async throws {
        try await runBasicSynthesis(modelId: "aufklarer/VoxCPM2-MLX-int4")
    }
}
