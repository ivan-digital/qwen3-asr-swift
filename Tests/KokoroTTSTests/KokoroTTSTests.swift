import XCTest
@testable import KokoroTTS

final class KokoroTTSTests: XCTestCase {

    // MARK: - Configuration Tests

    func testDefaultConfig() {
        let config = KokoroConfig.default
        XCTAssertEqual(config.sampleRate, 24000)
        XCTAssertEqual(config.maxPhonemeLength, 510)
        XCTAssertEqual(config.styleDim, 256)
        XCTAssertEqual(config.samplesPerFrame, 600)
        XCTAssertEqual(config.languages.count, 8)
        XCTAssertTrue(config.languages.contains("en"))
        XCTAssertTrue(config.languages.contains("ja"))
    }

    func testConfigCodable() throws {
        let config = KokoroConfig.default
        let data = try JSONEncoder().encode(config)
        let decoded = try JSONDecoder().decode(KokoroConfig.self, from: data)
        XCTAssertEqual(decoded.sampleRate, config.sampleRate)
        XCTAssertEqual(decoded.maxPhonemeLength, config.maxPhonemeLength)
        XCTAssertEqual(decoded.styleDim, config.styleDim)
        XCTAssertEqual(decoded.samplesPerFrame, config.samplesPerFrame)
    }

    // MARK: - Model Bucket Tests

    func testBucketSelection() {
        // Short text → v24_10s (preferred, iOS 17+)
        let bucket = ModelBucket.select(forTokenCount: 50)
        XCTAssertEqual(bucket, .v24_10s)

        // Long text → v21_15s
        let longBucket = ModelBucket.select(forTokenCount: 245)
        XCTAssertEqual(longBucket, .v21_15s)

        // Too long → nil
        XCTAssertNil(ModelBucket.select(forTokenCount: 300))
    }

    func testBucketSelectionV21Fallback() {
        // When preferV24 is false, should use v21 buckets
        let bucket = ModelBucket.select(forTokenCount: 50, preferV24: false)
        XCTAssertEqual(bucket, .v21_5s)
    }

    func testBucketProperties() {
        XCTAssertEqual(ModelBucket.v21_5s.modelName, "kokoro_21_5s")
        XCTAssertEqual(ModelBucket.v24_10s.modelName, "kokoro_24_10s")
        XCTAssertEqual(ModelBucket.v21_5s.maxTokens, 124)
        XCTAssertEqual(ModelBucket.v24_10s.maxTokens, 242)
        XCTAssertEqual(ModelBucket.v21_5s.maxSamples, 175_800)
    }

    func testBucketDurations() {
        XCTAssertEqual(ModelBucket.v24_10s.maxDuration, 10.0, accuracy: 0.01)
        XCTAssertEqual(ModelBucket.v24_15s.maxDuration, 15.0, accuracy: 0.01)
    }

    // MARK: - Phonemizer Tests

    func testPhonemizerTokenize() {
        let vocab: [String: Int] = [
            "<pad>": 0, "<bos>": 1, "<eos>": 2,
            "h": 3, "e": 4, "l": 5, "o": 6, " ": 7,
        ]
        let phonemizer = KokoroPhonemizer(vocab: vocab)

        let ids = phonemizer.tokenize("hello")
        XCTAssertEqual(ids.first, 1) // BOS
        XCTAssertEqual(ids.last, 2)  // EOS
        XCTAssertTrue(ids.count >= 3)
    }

    func testPhonemizerPadding() {
        let vocab: [String: Int] = ["<pad>": 0, "<bos>": 1, "<eos>": 2, "a": 3]
        let phonemizer = KokoroPhonemizer(vocab: vocab)

        let ids = phonemizer.tokenize("a")
        let padded = phonemizer.pad(ids, to: 10)
        XCTAssertEqual(padded.count, 10)
        XCTAssertEqual(padded[0], 1) // BOS
        XCTAssertEqual(padded.last(where: { $0 != 0 }), 2) // EOS before padding
        XCTAssertEqual(padded[padded.count - 1], 0) // trailing pad
    }

    func testPhonemizerTruncation() {
        let vocab: [String: Int] = ["<pad>": 0, "<bos>": 1, "<eos>": 2, "a": 3]
        let phonemizer = KokoroPhonemizer(vocab: vocab)

        let longText = String(repeating: "a", count: 1000)
        let ids = phonemizer.tokenize(longText, maxLength: 20)
        XCTAssertEqual(ids.count, 20)
        XCTAssertEqual(ids.first, 1) // BOS preserved
        XCTAssertEqual(ids.last, 2)  // EOS preserved
    }

    func testPhonemizerUnknownChars() {
        let vocab: [String: Int] = ["<pad>": 0, "<bos>": 1, "<eos>": 2, "a": 3]
        let phonemizer = KokoroPhonemizer(vocab: vocab)

        let ids = phonemizer.tokenize("axyz")
        XCTAssertEqual(ids, [1, 3, 2]) // BOS + 'a' + EOS (xyz dropped)
    }

    func testPhonemizerLoadVocab() throws {
        let tempDir = FileManager.default.temporaryDirectory
        let vocabURL = tempDir.appendingPathComponent("test_vocab.json")
        let vocabData = try JSONEncoder().encode(["<pad>": 0, "<bos>": 1, "<eos>": 2, "t": 3, "s": 4])
        try vocabData.write(to: vocabURL)

        let phonemizer = try KokoroPhonemizer.loadVocab(from: vocabURL)
        let ids = phonemizer.tokenize("test")
        XCTAssertEqual(ids.first, 1)
        XCTAssertEqual(ids.last, 2)

        try? FileManager.default.removeItem(at: vocabURL)
    }
}
