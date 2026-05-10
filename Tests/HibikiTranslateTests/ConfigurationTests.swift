import XCTest
import MLX
import MLXNN
import MLXRandom
import PersonaPlex   // KVCacheSimple
@testable import HibikiTranslate

final class ConfigurationTests: XCTestCase {
    func testZero3BTemporalDimensions() {
        let cfg = HibikiConfig.zero3B
        XCTAssertEqual(cfg.temporal.dim, 2048)
        XCTAssertEqual(cfg.temporal.numLayers, 28)
        XCTAssertEqual(cfg.temporal.numHeads, 16)
        XCTAssertEqual(cfg.temporal.headDim, 128)
        XCTAssertEqual(cfg.temporal.kvRepeat, 2)
        XCTAssertEqual(cfg.temporal.numKVHeads, 8)
        XCTAssertEqual(cfg.temporal.kvDim, 1024)
        XCTAssertEqual(cfg.temporal.intermediateSize, 8192)
        XCTAssertEqual(cfg.temporal.nQ, 16)
        XCTAssertEqual(cfg.temporal.numAudioEmbeddings, 32)
        XCTAssertEqual(cfg.temporal.numCodebooks, 33)
        XCTAssertEqual(cfg.temporal.textCard, 48000)
        XCTAssertEqual(cfg.temporal.maxPeriod, 20000)
        XCTAssertEqual(cfg.temporal.positionalEmbedding, .ropeConcat)
        XCTAssertEqual(cfg.temporal.positionalEmbedding.traditional, false)
    }

    func testZero3BDepformer() {
        let cfg = HibikiConfig.zero3B
        XCTAssertEqual(cfg.depformer.dim, 1024)
        XCTAssertEqual(cfg.depformer.numLayers, 6)
        XCTAssertEqual(cfg.depformer.numHeads, 16)
        XCTAssertEqual(cfg.depformer.headDim, 64)
        XCTAssertEqual(cfg.depformer.dimFeedforward, 4096)
        XCTAssertEqual(cfg.depformer.numSteps, 16)
        XCTAssertEqual(cfg.depformer.kvRepeat, 1)
        XCTAssertTrue(cfg.depformer.weightsPerStep)
        XCTAssertTrue(cfg.depformer.multiLinear)
    }

    func testZero3BSchedule() {
        let cfg = HibikiConfig.zero3B
        XCTAssertEqual(cfg.depformer.weightsPerStepSchedule,
                       [0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 8, 8, 8, 8, 8, 8])
        XCTAssertEqual(cfg.depformer.numUniqueSlices, 9)

        // Boundary mappings
        XCTAssertEqual(cfg.depformer.sliceIndex(forStep: 0), 0)
        XCTAssertEqual(cfg.depformer.sliceIndex(forStep: 7), 7)
        XCTAssertEqual(cfg.depformer.sliceIndex(forStep: 8), 8)
        XCTAssertEqual(cfg.depformer.sliceIndex(forStep: 15), 8)
    }

    func testStreamCounts() {
        let cfg = HibikiConfig.zero3B
        XCTAssertEqual(cfg.numStreams, 33)
        XCTAssertEqual(cfg.numSourceCodebooks, 16)
        XCTAssertEqual(cfg.numTargetCodebooks, 16)
        XCTAssertEqual(cfg.maxDelay, 2)
        XCTAssertEqual(cfg.delays.count, 33)
        XCTAssertEqual(cfg.delays[0], 0)             // text
        XCTAssertEqual(cfg.delays[1], 0)             // source cb 0 (semantic)
        XCTAssertEqual(cfg.delays[2], 2)             // source cb 1 (acoustic)
        XCTAssertEqual(cfg.delays[16], 2)            // last source acoustic
        XCTAssertEqual(cfg.delays[17], 0)            // target cb 0 (semantic)
        XCTAssertEqual(cfg.delays[18], 2)            // target cb 1 (acoustic)
        XCTAssertEqual(cfg.delays[32], 2)            // last target acoustic
    }

    func testLanguages() {
        let cfg = HibikiConfig.zero3B
        XCTAssertEqual(cfg.languages.source, [.fr, .es, .pt, .de])
        XCTAssertEqual(cfg.languages.target, .en)
    }

    func testMimi() {
        let cfg = HibikiConfig.zero3B
        XCTAssertEqual(cfg.mimi.numCodebooks, 16)
        XCTAssertEqual(cfg.mimi.sampleRate, 24000)
        XCTAssertEqual(cfg.mimi.frameRate, 12.5)
    }

    func testRoPELayoutMapping() {
        // Hibiki Zero-3B uses split-half RoPE -> mlx-swift's traditional=false
        XCTAssertFalse(HibikiPositionalEmbedding.ropeConcat.traditional)
        // Standard RoPE (Moshi/PersonaPlex/Hibiki 1B/2B) uses interleaved -> traditional=true
        XCTAssertTrue(HibikiPositionalEmbedding.rope.traditional)
    }

    /// Verifies HibikiTemporalAttention produces correctly shaped output and
    /// the GQA in_proj split is honored (Q has numHeads, K/V have numKVHeads).
    func testGQAAttentionShape() {
        let cfg = HibikiConfig.zero3B.temporal
        let attn = HibikiTemporalAttention(cfg: cfg)
        let b = 1, t = 4
        let xs = MLXRandom.normal([b, t, cfg.dim])
        eval(xs)
        let cache = KVCacheSimple()
        let out = attn(xs, cache: cache, offset: 0)
        eval(out)
        XCTAssertEqual(out.shape, [b, t, cfg.dim])
        // Cache populated with K/V at numKVHeads heads (GQA).
        if let k = cache.keysArray, let v = cache.valuesArray {
            XCTAssertEqual(k.shape, [b, cfg.numKVHeads, t, cfg.headDim])
            XCTAssertEqual(v.shape, [b, cfg.numKVHeads, t, cfg.headDim])
        } else {
            XCTFail("KV cache not populated")
        }
    }

    /// Verifies HibikiTemporalTransformer.forward end-to-end shape with random
    /// weights: text logits over textCard, hidden over dim, GQA caches populated.
    func testTemporalTransformerForward() {
        let cfg = HibikiConfig.zero3B.temporal
        let model = HibikiTemporalTransformer(cfg: cfg)
        let b = 1, t = 2
        let textTokens = MLXArray.zeros([b, t], dtype: .int32)
        let audioTokens = MLXArray.zeros([b, cfg.numAudioEmbeddings, t], dtype: .int32)
        let (hidden, textLogits) = model.forward(
            textTokens: textTokens, audioTokens: audioTokens, offset: 0)
        eval(hidden)
        eval(textLogits)
        XCTAssertEqual(hidden.shape, [b, t, cfg.dim])
        XCTAssertEqual(textLogits.shape, [b, t, cfg.textCard])
        // First-layer cache should have K/V at numKVHeads.
        XCTAssertEqual(model.cache[0].keysArray?.shape,
                       [b, cfg.numKVHeads, t, cfg.headDim])
    }

    /// Verifies that ScheduledMultiLinear stores `numUniqueSlices * outDim` rows
    /// and that step k sources weights from rows `[schedule[k]*outDim, ...)`.
    func testScheduledMultiLinearStorageShape() {
        let schedule = HibikiDepformerConfig.zero3BSchedule  // 9 unique over 16 steps
        let inDim = 64, outDim = 8
        let ml = ScheduledMultiLinear(schedule: schedule, inDim: inDim, outDim: outDim,
                                       bias: false, groupSize: 64, bits: 16)
        // Storage rows = numUniqueSlices * outDim = 9 * 8 = 72
        XCTAssertEqual(ml.weight.shape, [9 * outDim, inDim])
        XCTAssertNil(ml.scales)
        XCTAssertNil(ml.biases)
    }

    /// At steps 8..15 (all schedule=8), output should match the slice for index 8.
    func testScheduledMultiLinearSliceMapping() {
        let schedule = HibikiDepformerConfig.zero3BSchedule
        let inDim = 64, outDim = 4
        let ml = ScheduledMultiLinear(schedule: schedule, inDim: inDim, outDim: outDim,
                                       bias: false, groupSize: 64, bits: 16)
        // Zero everything, then set slice 0 to marker A and slice 8 to marker B.
        ml.weight = MLXArray.zeros([9 * outDim, inDim])
        let markerA: Float = 7.0
        let markerB: Float = 42.0
        ml.weight[0..<outDim, 0...] = MLXArray.ones([outDim, inDim]) * markerA
        ml.weight[(8 * outDim)..<(9 * outDim), 0...] = MLXArray.ones([outDim, inDim]) * markerB

        let xs = MLXArray.ones([1, 1, inDim])
        // Step 0 → slice 0 → markerA * inDim
        let out0 = ml(xs, step: 0)
        eval(out0)
        XCTAssertEqual(out0[0, 0, 0].item(Float.self), markerA * Float(inDim), accuracy: 1e-3)
        // Step 7 → slice 7 → 0 (untouched)
        let out7 = ml(xs, step: 7)
        eval(out7)
        XCTAssertEqual(out7[0, 0, 0].item(Float.self), 0, accuracy: 1e-3)
        // Steps 8..15 all map to slice 8 → markerB * inDim
        for step in 8...15 {
            let out = ml(xs, step: step)
            eval(out)
            XCTAssertEqual(out.shape, [1, 1, outDim])
            let firstVal = out[0, 0, 0].item(Float.self)
            XCTAssertEqual(firstVal, markerB * Float(inDim), accuracy: 1e-3,
                           "step \(step) should produce \(markerB * Float(inDim)), got \(firstVal)")
        }
    }

    /// HibikiDepformer.generate produces [B, numSteps] tokens.
    func testDepformerGenerateShape() {
        let cfg = HibikiConfig.zero3B
        let dep = HibikiDepformer(cfg: cfg.depformer, temporalDim: cfg.temporal.dim)

        let temporalHidden = MLXRandom.normal([1, 1, cfg.temporal.dim])
        eval(temporalHidden)
        let textToken = MLXArray([Int32(0)])

        // Trivial sampler: always emit token 0
        let tokens = dep.generate(
            temporalHidden: temporalHidden,
            textToken: textToken,
            sampleFn: { _, _ in MLXArray([Int32(0)]) }
        )
        eval(tokens)
        XCTAssertEqual(tokens.shape, [1, cfg.depformer.numSteps])
    }

    func testJSONLoading() throws {
        // Write a minimal config.json mirroring the converter output
        let json: [String: Any] = [
            "model_type": "hibiki",
            "version": "hibiki-zero-3b-v1",
            "temporal": [
                "dim": 2048, "num_layers": 28, "num_heads": 16,
                "hidden_scale": 6.0, "n_q": 32, "card": 2048,
                "text_card": 48000, "context": 3000, "max_period": 20000.0,
                "kv_repeat": 2, "positional_embedding": "rope_concat",
            ],
            "depformer": [
                "dim": 1024, "num_layers": 6, "num_heads": 16,
                "dim_feedforward": NSNull(), "num_steps": 16, "card": 2048,
                "text_card": 48000, "context": 16,
                "weights_per_step": true, "multi_linear": true, "kv_repeat": 1,
                "norm": "layer_norm",
                "weights_per_step_schedule": [0, 1, 2, 3, 4, 5, 6, 7, 8,
                                              8, 8, 8, 8, 8, 8, 8],
            ],
            "delays": HibikiConfig.zero3BDelays,
            "languages": ["source": ["fr", "es", "pt", "de"], "target": "en"],
            "quantization": ["bits": 4, "group_size": 64],
        ]
        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent("hibiki_test_\(UUID().uuidString).json")
        defer { try? FileManager.default.removeItem(at: tmp) }
        let data = try JSONSerialization.data(withJSONObject: json, options: [])
        try data.write(to: tmp)

        let loaded = try HibikiConfig.load(from: tmp)
        XCTAssertEqual(loaded.temporal.dim, 2048)
        XCTAssertEqual(loaded.temporal.numLayers, 28)
        XCTAssertEqual(loaded.temporal.kvRepeat, 2)
        XCTAssertEqual(loaded.temporal.positionalEmbedding, .ropeConcat)
        XCTAssertEqual(loaded.temporal.bits, 4)
        XCTAssertEqual(loaded.temporal.nQ, 16)
        XCTAssertEqual(loaded.depformer.dimFeedforward, 4096)  // computed when null
        XCTAssertEqual(loaded.depformer.numUniqueSlices, 9)
        XCTAssertEqual(loaded.depformer.sliceIndex(forStep: 15), 8)
        XCTAssertEqual(loaded.numStreams, 33)
        XCTAssertEqual(loaded.languages.source, [.fr, .es, .pt, .de])
        XCTAssertEqual(loaded.languages.target, .en)
        XCTAssertEqual(loaded.mimi.numCodebooks, 16)
    }
}
