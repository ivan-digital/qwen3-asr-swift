import XCTest
@testable import MADLADTranslation
import AudioCommon

// MARK: - Config

final class MADLADTranslationConfigTests: XCTestCase {

    func testDefault3BConfig() {
        let c = MADLADTranslationConfig.madlad3B
        XCTAssertEqual(c.dModel, 1024)
        XCTAssertEqual(c.dKv, 128)
        XCTAssertEqual(c.dFf, 8192)
        XCTAssertEqual(c.numLayers, 32)
        XCTAssertEqual(c.numDecoderLayers, 32)
        XCTAssertEqual(c.numHeads, 16)
        XCTAssertEqual(c.vocabSize, 256_000)
        XCTAssertEqual(c.relativeAttentionNumBuckets, 32)
        XCTAssertEqual(c.relativeAttentionMaxDistance, 128)
        XCTAssertEqual(c.decoderStartTokenId, 0)
        XCTAssertEqual(c.eosTokenId, 2)
        XCTAssertEqual(c.padTokenId, 1)
        XCTAssertFalse(c.tieWordEmbeddings)
    }

    func testJSONRoundTrip() throws {
        let original = MADLADTranslationConfig.madlad3B
        let data = try JSONEncoder().encode(original)
        let decoded = try JSONDecoder().decode(MADLADTranslationConfig.self, from: data)
        XCTAssertEqual(decoded.dModel, original.dModel)
        XCTAssertEqual(decoded.numLayers, original.numLayers)
        XCTAssertEqual(decoded.vocabSize, original.vocabSize)
        XCTAssertEqual(decoded.eosTokenId, original.eosTokenId)
    }

    func testDecodeFromHFStyleJSON() throws {
        // The conversion script writes config.json using these snake_case keys.
        let json = """
        {
            "d_model": 1024,
            "d_kv": 128,
            "d_ff": 8192,
            "num_layers": 32,
            "num_decoder_layers": 32,
            "num_heads": 16,
            "vocab_size": 256000,
            "relative_attention_num_buckets": 32,
            "relative_attention_max_distance": 128,
            "layer_norm_epsilon": 1e-6,
            "decoder_start_token_id": 0,
            "eos_token_id": 2,
            "pad_token_id": 1,
            "tie_word_embeddings": false,
            "quantization": "int4"
        }
        """.data(using: .utf8)!
        let cfg = try JSONDecoder().decode(MADLADTranslationConfig.self, from: json)
        XCTAssertEqual(cfg.quantization, "int4")
        XCTAssertEqual(cfg.dFf, 8192)
    }
}

// MARK: - Sampling config

final class TranslationSamplingConfigTests: XCTestCase {

    func testGreedyDefault() {
        let s = TranslationSamplingConfig.greedy
        XCTAssertEqual(s.temperature, 0.0)
        XCTAssertEqual(s.topK, 0)
        XCTAssertEqual(s.topP, 1.0)
        XCTAssertEqual(s.maxTokens, 256)
    }

    func testSamplingPreset() {
        let s = TranslationSamplingConfig.sampling
        XCTAssertGreaterThan(s.temperature, 0)
        XCTAssertGreaterThan(s.topK, 0)
        XCTAssertLessThan(s.topP, 1.0)
    }
}

// MARK: - Error type

final class MADLADTranslationErrorTests: XCTestCase {

    func testErrorDescriptions() {
        let e1 = MADLADTranslationError.unsupportedLanguage("xx")
        XCTAssertNotNil(e1.errorDescription)
        XCTAssertTrue(e1.errorDescription!.contains("xx"))

        let e2 = MADLADTranslationError.modelLoadFailed("missing")
        XCTAssertTrue(e2.errorDescription!.contains("missing"))

        let url = URL(fileURLWithPath: "/tmp/nope")
        let e3 = MADLADTranslationError.configNotFound(url)
        XCTAssertTrue(e3.errorDescription!.contains("/tmp/nope"))
    }
}
