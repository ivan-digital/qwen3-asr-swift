import XCTest
@testable import KokoroTTS

final class KokoroTTSTests: XCTestCase {

    func testDefaultModelId() {
        XCTAssertEqual(KokoroTTSModel.defaultModelId, "aufklarer/Kokoro-82M-CoreML")
    }

    func testDefaultConfig() {
        let config = KokoroConfig.default
        XCTAssertEqual(config.sampleRate, 24000)
        XCTAssertEqual(config.maxPhonemeLength, 128)
        XCTAssertEqual(config.styleDim, 256)
        XCTAssertEqual(config.languages.count, 8)
        XCTAssertTrue(config.languages.contains("en"))
    }

    func testConfigCodable() throws {
        let config = KokoroConfig.default
        let data = try JSONEncoder().encode(config)
        let decoded = try JSONDecoder().decode(KokoroConfig.self, from: data)
        XCTAssertEqual(decoded.sampleRate, config.sampleRate)
        XCTAssertEqual(decoded.maxPhonemeLength, config.maxPhonemeLength)
        XCTAssertEqual(decoded.styleDim, config.styleDim)
    }

    // MARK: - Phonemizer Tests

    func testPhonemizerTokenize() {
        let vocab: [String: Int] = [
            "<pad>": 0, "<bos>": 1, "<eos>": 2,
            "h": 3, "e": 4, "l": 5, "o": 6, " ": 7,
        ]
        let phonemizer = KokoroPhonemizer(vocab: vocab)
        let ids = phonemizer.tokenize("hello")
        XCTAssertEqual(ids.first, 1)
        XCTAssertEqual(ids.last, 2)
        XCTAssertTrue(ids.count >= 3)
    }

    func testPhonemizerPadding() {
        let vocab: [String: Int] = ["<pad>": 0, "<bos>": 1, "<eos>": 2, "a": 3]
        let phonemizer = KokoroPhonemizer(vocab: vocab)
        let ids = phonemizer.tokenize("a")
        let padded = phonemizer.pad(ids, to: 10)
        XCTAssertEqual(padded.count, 10)
        XCTAssertEqual(padded[0], 1)
        XCTAssertEqual(padded.last(where: { $0 != 0 }), 2)
    }

    func testPhonemizerTruncation() {
        let vocab: [String: Int] = ["<pad>": 0, "<bos>": 1, "<eos>": 2, "a": 3]
        let phonemizer = KokoroPhonemizer(vocab: vocab)
        let longText = String(repeating: "a", count: 1000)
        let ids = phonemizer.tokenize(longText, maxLength: 20)
        XCTAssertEqual(ids.count, 20)
        XCTAssertEqual(ids.first, 1)
        XCTAssertEqual(ids.last, 2)
    }

    func testPhonemizerUnknownChars() {
        let vocab: [String: Int] = ["<pad>": 0, "<bos>": 1, "<eos>": 2, "a": 3]
        let phonemizer = KokoroPhonemizer(vocab: vocab)
        let ids = phonemizer.tokenize("axyz")
        XCTAssertEqual(ids, [1, 3, 2])
    }
}
