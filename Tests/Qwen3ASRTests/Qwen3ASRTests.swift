import XCTest
import MLX
#if canImport(Metal)
import Metal
#endif
@testable import Qwen3ASR
@testable import AudioCommon

final class Qwen3ASRTests: XCTestCase {

    func testAudioEncoderConfig() {
        let config = Qwen3AudioEncoderConfig.default
        XCTAssertEqual(config.dModel, 896)
        XCTAssertEqual(config.encoderLayers, 18)
        XCTAssertEqual(config.encoderAttentionHeads, 14)
        XCTAssertEqual(config.numMelBins, 128)
        XCTAssertEqual(config.outputDim, 1024)
    }

    func testAudioEncoderLargeConfig() {
        let config = Qwen3AudioEncoderConfig.large
        XCTAssertEqual(config.dModel, 1024)
        XCTAssertEqual(config.encoderLayers, 24)
        XCTAssertEqual(config.encoderAttentionHeads, 16)
        XCTAssertEqual(config.numMelBins, 128)
        XCTAssertEqual(config.outputDim, 2048)
        XCTAssertEqual(config.downsampleHiddenSize, 480)
        XCTAssertEqual(config.convOutInputDim, 7680)
    }

    func testTextDecoderConfig() {
        let smallConfig = TextDecoderConfig.small
        XCTAssertEqual(smallConfig.hiddenSize, 1024)
        XCTAssertEqual(smallConfig.numLayers, 28)
        XCTAssertEqual(smallConfig.numHeads, 16)
        XCTAssertEqual(smallConfig.numKVHeads, 8)
        XCTAssertEqual(smallConfig.intermediateSize, 3072)
        XCTAssertEqual(smallConfig.bits, 4)

        let largeConfig = TextDecoderConfig.large
        XCTAssertEqual(largeConfig.hiddenSize, 2048)
        XCTAssertEqual(largeConfig.numLayers, 28)
        XCTAssertEqual(largeConfig.numHeads, 16)
        XCTAssertEqual(largeConfig.numKVHeads, 8)
        XCTAssertEqual(largeConfig.intermediateSize, 6144)
        XCTAssertEqual(largeConfig.bits, 8)
    }

    func testASRModelSizeDetection() {
        XCTAssertEqual(
            ASRModelSize.detect(from: "aufklarer/Qwen3-ASR-0.6B-MLX-4bit"),
            .small)
        XCTAssertEqual(
            ASRModelSize.detect(from: "aufklarer/Qwen3-ASR-1.7B-MLX-8bit"),
            .large)
        XCTAssertEqual(
            ASRModelSize.detect(from: "some-custom/model"),
            .small)  // defaults to small
    }

    func testASRModelSizeConfigs() {
        let small = ASRModelSize.small
        XCTAssertEqual(small.audioConfig.dModel, 896)
        XCTAssertEqual(small.textConfig.hiddenSize, 1024)

        let large = ASRModelSize.large
        XCTAssertEqual(large.audioConfig.dModel, 1024)
        XCTAssertEqual(large.textConfig.hiddenSize, 2048)
    }

    func testQwen3ASRConfig() {
        let config = Qwen3ASRConfig.small
        XCTAssertEqual(config.audioTokenIndex, 151646)
        XCTAssertEqual(config.eosTokenId, 151645)
        XCTAssertEqual(config.padTokenId, 151643)
    }

    func testFeatureExtractor() throws {
        let extractor = WhisperFeatureExtractor()
        XCTAssertEqual(extractor.sampleRate, 16000)
        XCTAssertEqual(extractor.nMels, 128)
        XCTAssertEqual(extractor.hopLength, 160)

        // Test with silent audio (1 second at 16kHz)
        let silentAudio = [Float](repeating: 0, count: 16000)
        let features = extractor.extractFeatures(silentAudio)

        // Check output shape
        XCTAssertEqual(features.dim(0), 128) // mel bins
        XCTAssertGreaterThan(features.dim(1), 0) // time frames
    }

    func testFeatureExtractorWithSineWave() throws {
        let extractor = WhisperFeatureExtractor()

        // Generate 1 second of 440Hz sine wave at 16kHz
        let sampleRate = 16000
        let frequency: Float = 440.0
        let duration = 1.0
        let numSamples = Int(Double(sampleRate) * duration)

        var audio = [Float](repeating: 0, count: numSamples)
        for i in 0..<numSamples {
            let t = Float(i) / Float(sampleRate)
            audio[i] = sin(2 * .pi * frequency * t) * 0.5
        }

        let features = extractor.extractFeatures(audio)

        // Verify features are computed
        XCTAssertEqual(features.dim(0), 128)
        XCTAssertGreaterThan(features.dim(1), 90) // Should have ~99 frames for 1s at 16kHz/160 hop

        // Features should not be all zeros
        let maxVal = features.max().item(Float.self)
        XCTAssertGreaterThan(maxVal, -100.0) // Log mel, so can be negative
    }

    func testAudioEncoderCreation() throws {
        #if canImport(Metal)
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal device unavailable")
        }
        #endif
        let config = Qwen3AudioEncoderConfig.default
        let encoder = Qwen3AudioEncoder(config: config)

        XCTAssertEqual(encoder.layers.count, 18)
    }

    func testModelCreation() throws {
        #if canImport(Metal)
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal device unavailable")
        }
        #endif
        let model = Qwen3ASRModel()

        XCTAssertNotNil(model.audioEncoder)
        XCTAssertNotNil(model.featureExtractor)
    }

    func testAudioFileLoaderWAV() throws {
        // Test loading a simple WAV file from bundle resources
        guard let wavURL = Bundle.module.url(forResource: "test_audio", withExtension: "wav") else {
            throw XCTSkip("Test WAV file not found in bundle resources")
        }

        let (samples, sampleRate) = try AudioFileLoader.loadWAV(url: wavURL)

        XCTAssertGreaterThan(samples.count, 0, "Should have audio samples")
        XCTAssertGreaterThan(sampleRate, 0, "Should have valid sample rate")
        print("Loaded \(samples.count) samples at \(sampleRate)Hz (\(Double(samples.count)/Double(sampleRate))s)")
    }

    // MARK: - Tokenizer Tests

    func testTokenizerLoadsVocab() throws {
        let cacheDir = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first!
            .appendingPathComponent("qwen3-asr")
            .appendingPathComponent("aufklarer_Qwen3-ASR-0.6B-MLX-4bit")

        let vocabPath = cacheDir.appendingPathComponent("vocab.json")

        guard FileManager.default.fileExists(atPath: vocabPath.path) else {
            throw XCTSkip("Tokenizer vocab.json not found - run model download first")
        }

        let tokenizer = Qwen3Tokenizer()
        try tokenizer.load(from: vocabPath)

        // Test basic token lookup
        XCTAssertEqual(tokenizer.getTokenId(for: "system"), 8948)
        XCTAssertEqual(tokenizer.getTokenId(for: "user"), 872)
        XCTAssertEqual(tokenizer.getTokenId(for: "assistant"), 77091)
    }

    func testTokenizerLoadsAddedTokens() throws {
        let cacheDir = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first!
            .appendingPathComponent("qwen3-asr")
            .appendingPathComponent("aufklarer_Qwen3-ASR-0.6B-MLX-4bit")

        let vocabPath = cacheDir.appendingPathComponent("vocab.json")

        guard FileManager.default.fileExists(atPath: vocabPath.path) else {
            throw XCTSkip("Tokenizer vocab.json not found - run model download first")
        }

        let tokenizer = Qwen3Tokenizer()
        try tokenizer.load(from: vocabPath)

        // Test that special tokens from tokenizer_config.json are loaded
        XCTAssertEqual(tokenizer.getTokenId(for: "<|im_start|>"), 151644, "Should have <|im_start|> token")
        XCTAssertEqual(tokenizer.getTokenId(for: "<|im_end|>"), 151645, "Should have <|im_end|> token")
        XCTAssertEqual(tokenizer.getTokenId(for: "<|audio_start|>"), 151669, "Should have <|audio_start|> token")
        XCTAssertEqual(tokenizer.getTokenId(for: "<|audio_end|>"), 151670, "Should have <|audio_end|> token")
        XCTAssertEqual(tokenizer.getTokenId(for: "<|audio_pad|>"), 151676, "Should have <|audio_pad|> token")
        XCTAssertEqual(tokenizer.getTokenId(for: "<asr_text>"), 151704, "Should have <asr_text> token")
        XCTAssertEqual(tokenizer.getTokenId(for: "<|endoftext|>"), 151643, "Should have <|endoftext|> token")
    }

    func testTokenizerDecodeWithASRMarker() throws {
        let cacheDir = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first!
            .appendingPathComponent("qwen3-asr")
            .appendingPathComponent("aufklarer_Qwen3-ASR-0.6B-MLX-4bit")

        let vocabPath = cacheDir.appendingPathComponent("vocab.json")

        guard FileManager.default.fileExists(atPath: vocabPath.path) else {
            throw XCTSkip("Tokenizer vocab.json not found - run model download first")
        }

        let tokenizer = Qwen3Tokenizer()
        try tokenizer.load(from: vocabPath)

        // Simulate model output: "language English<asr_text>Hello"
        // Token IDs: language(11528) + ĠEnglish(6364) + <asr_text>(151704) + Hello tokens
        let languageId = 11528
        let englishId = 6364
        let asrTextId = 151704

        // Get Hello token IDs (simplified - just use the word token if exists)
        let helloId = tokenizer.getTokenId(for: "Hello") ?? 0

        let tokens = [languageId, englishId, asrTextId, helloId]
        let decoded = tokenizer.decode(tokens: tokens)

        print("Decoded output: '\(decoded)'")

        // Should contain <asr_text> marker for parsing
        XCTAssertTrue(decoded.contains("<asr_text>"), "Decoded text should contain <asr_text> marker")
        XCTAssertTrue(decoded.contains("language"), "Decoded text should contain 'language'")
        XCTAssertTrue(decoded.contains("English"), "Decoded text should contain 'English'")
    }

    // MARK: - UTF-8 Decode Regression Tests (no model download)

    /// Synthetic token map for multi-byte UTF-8 decode tests.
    /// BPE token strings use the GPT-2 byte-to-unicode mapping:
    ///   bytes 33-126, 161-172, 174-255 → same Unicode scalar
    ///   remaining bytes → U+0100 onwards (e.g. space=Ġ U+0120, 0x86=Ĩ U+0128)
    private func makeUTF8TokenMap() -> [Int: String] {
        [
            // ASCII
            100: "Hello",
            101: "\u{0120}world",          // Ġworld → " world"
            // 來 (U+4F86) = E4 BE 86, split into 3 byte-tokens
            102: "\u{00E4}",               // byte 0xE4
            103: "\u{00BE}",               // byte 0xBE
            104: "\u{0128}",               // byte 0x86 → U+0128
            // 好 (U+597D) = E5 A5 BD, split into 3 byte-tokens
            105: "\u{00E5}",               // byte 0xE5
            106: "\u{00A5}",               // byte 0xA5
            107: "\u{00BD}",               // byte 0xBD
            // Ġ + 來 as a single merged token
            108: "\u{0120}\u{00E4}\u{00BE}\u{0128}",
            // Markers
            200: "<asr_text>",
            201: "<|im_start|>",
        ]
    }

    func testDecodeCJKSplitAcrossThreeTokens() {
        let tok = Qwen3Tokenizer(idToToken: makeUTF8TokenMap())
        // 來 = E4 BE 86, each byte is a separate BPE token
        let result = tok.decode(tokens: [102, 103, 104])
        XCTAssertEqual(result, "來")
    }

    func testDecodeMixedASCIIAndCJK() {
        let tok = Qwen3Tokenizer(idToToken: makeUTF8TokenMap())
        let result = tok.decode(tokens: [100, 102, 103, 104])
        XCTAssertEqual(result, "Hello來")
    }

    func testDecodeConsecutiveMultiByteCharacters() {
        let tok = Qwen3Tokenizer(idToToken: makeUTF8TokenMap())
        // 來好 — six byte-tokens back to back
        let result = tok.decode(tokens: [102, 103, 104, 105, 106, 107])
        XCTAssertEqual(result, "來好")
    }

    func testDecodeGPrefixBeforeMultiByteCharacter() {
        let tok = Qwen3Tokenizer(idToToken: makeUTF8TokenMap())
        // "Hello 來" — token 108 is Ġ + 來 bytes (space + CJK in one token)
        let result = tok.decode(tokens: [100, 108])
        XCTAssertEqual(result, "Hello 來")
    }

    func testDecodeASRTextMarkerBetweenMultiByteSequences() {
        let tok = Qwen3Tokenizer(idToToken: makeUTF8TokenMap())
        // 來<asr_text>好
        let result = tok.decode(tokens: [102, 103, 104, 200, 105, 106, 107])
        XCTAssertEqual(result, "來<asr_text>好")
    }

    // MARK: - Extended CJK / hieroglyph robustness tests

    /// GPT-2 byte-to-unicode: maps a raw byte to the BPE character used in vocab.
    private static func bpeChar(_ byte: UInt8) -> Character {
        if (33...126).contains(byte) || (161...172).contains(byte) || (174...255).contains(byte) {
            return Character(UnicodeScalar(byte))
        }
        var n = 0
        for b: UInt8 in 0...255 {
            if (33...126).contains(b) || (161...172).contains(b) || (174...255).contains(b) { continue }
            if b == byte { return Character(UnicodeScalar(0x100 + n)!) }
            n += 1
        }
        fatalError("unreachable")
    }

    /// Build a BPE token string from raw UTF-8 bytes.
    private static func bpeToken(_ bytes: [UInt8]) -> String {
        String(bytes.map { bpeChar($0) })
    }

    /// Build a token map from (id, UTF-8 bytes) pairs, plus optional literal entries.
    private func makeTokenMap(
        bytes: [(Int, [UInt8])],
        literals: [(Int, String)] = []
    ) -> [Int: String] {
        var map: [Int: String] = [:]
        for (id, b) in bytes { map[id] = Self.bpeToken(b) }
        for (id, s) in literals { map[id] = s }
        return map
    }

    func testDecodeTruncatedUTF8FallsBackToReplacementChar() {
        // Only 2 of 3 bytes for 來 (simulates max-token truncation mid-character)
        let tok = Qwen3Tokenizer(idToToken: makeTokenMap(
            bytes: [(1, [0xE4]), (2, [0xBE])]
        ))
        let result = tok.decode(tokens: [1, 2])
        // String(bytes:encoding:.utf8) returns nil → fallback uses U+FFFD replacement
        XCTAssertTrue(result.contains("\u{FFFD}"), "Truncated UTF-8 should produce replacement character")
    }

    func testDecodeSpecialTokenSkippedBetweenCJKBytes() {
        // <|im_end|> between bytes of 來 should be skipped, bytes land contiguously
        let tok = Qwen3Tokenizer(idToToken: makeTokenMap(
            bytes: [(1, [0xE4]), (2, [0xBE]), (3, [0x86])],
            literals: [(900, "<|im_end|>")]
        ))
        let result = tok.decode(tokens: [1, 2, 900, 3])
        XCTAssertEqual(result, "來")
    }

    func testDecode4ByteUTF8CJKExtensionB() {
        // 𠀀 (U+20000) = F0 A0 80 80 — rare CJK char needing 4 byte-tokens
        let tok = Qwen3Tokenizer(idToToken: makeTokenMap(
            bytes: [(1, [0xF0]), (2, [0xA0]), (3, [0x80]), (4, [0x80])]
        ))
        let result = tok.decode(tokens: [1, 2, 3, 4])
        XCTAssertEqual(result, "𠀀")
    }

    func testDecodeBPETokenSpanningUTF8Boundary() {
        // One merged BPE token contains last byte of 來 (0x86) + first byte of 好 (0xE5)
        // Buffer: [E4, BE, 86, E5, A5, BD] → "來好"
        let tok = Qwen3Tokenizer(idToToken: makeTokenMap(
            bytes: [
                (1, [0xE4]), (2, [0xBE]),
                (3, [0x86, 0xE5]),  // merged token spanning boundary
                (4, [0xA5]), (5, [0xBD]),
            ]
        ))
        let result = tok.decode(tokens: [1, 2, 3, 4, 5])
        XCTAssertEqual(result, "來好")
    }

    func testDecodeKoreanHangul() {
        // 한국 = ED 95 9C  EA B5 AD
        let tok = Qwen3Tokenizer(idToToken: makeTokenMap(
            bytes: [
                (1, [0xED]), (2, [0x95]), (3, [0x9C]),
                (4, [0xEA]), (5, [0xB5]), (6, [0xAD]),
            ]
        ))
        let result = tok.decode(tokens: [1, 2, 3, 4, 5, 6])
        XCTAssertEqual(result, "한국")
    }

    func testDecodeJapaneseMixedHiraganaKanji() {
        // こんにちは日本語 — 8 chars × 3 bytes each
        let tok = Qwen3Tokenizer(idToToken: makeTokenMap(
            bytes: [
                // こ E3 81 93
                (1, [0xE3]), (2, [0x81]), (3, [0x93]),
                // ん E3 82 93
                (4, [0x82]),
                // に E3 81 AB
                (5, [0xAB]),
                // ち E3 81 A1
                (6, [0xA1]),
                // は E3 81 AF
                (7, [0xAF]),
                // 日 E6 97 A5
                (8, [0xE6]), (9, [0x97]), (10, [0xA5]),
                // 本 E6 9C AC
                (11, [0x9C]), (12, [0xAC]),
                // 語 E8 AA 9E
                (13, [0xE8]), (14, [0xAA]), (15, [0x9E]),
            ]
        ))
        // こ      ん            に         ち         は         日         本            語
        let result = tok.decode(tokens: [1,2,3, 1,4,3, 1,2,5, 1,2,6, 1,2,7, 8,9,10, 8,11,12, 13,14,15])
        XCTAssertEqual(result, "こんにちは日本語")
    }

    func testDecodeUnknownTokenIdBetweenCJKBytes() {
        // Unknown token ID (999) between bytes of 來 — skipped, bytes still contiguous
        let tok = Qwen3Tokenizer(idToToken: makeTokenMap(
            bytes: [(1, [0xE4]), (2, [0xBE]), (3, [0x86])]
        ))
        let result = tok.decode(tokens: [1, 2, 999, 3])
        XCTAssertEqual(result, "來", "Unknown token IDs should be skipped without breaking byte sequence")
    }

    func testTokenizerSkipsSpecialTokensWithPipes() throws {
        let cacheDir = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first!
            .appendingPathComponent("qwen3-asr")
            .appendingPathComponent("aufklarer_Qwen3-ASR-0.6B-MLX-4bit")

        let vocabPath = cacheDir.appendingPathComponent("vocab.json")

        guard FileManager.default.fileExists(atPath: vocabPath.path) else {
            throw XCTSkip("Tokenizer vocab.json not found - run model download first")
        }

        let tokenizer = Qwen3Tokenizer()
        try tokenizer.load(from: vocabPath)

        // Test that <|im_start|>, <|im_end|>, <|endoftext|> are skipped in decode
        let imStartId = 151644
        let imEndId = 151645
        let eosId = 151643
        let helloId = tokenizer.getTokenId(for: "Hello") ?? 0

        let tokens = [imStartId, helloId, imEndId, eosId]
        let decoded = tokenizer.decode(tokens: tokens)

        print("Decoded (should skip special tokens): '\(decoded)'")

        // Should NOT contain <|...|> tokens
        XCTAssertFalse(decoded.contains("<|im_start|>"), "Should skip <|im_start|>")
        XCTAssertFalse(decoded.contains("<|im_end|>"), "Should skip <|im_end|>")
        XCTAssertFalse(decoded.contains("<|endoftext|>"), "Should skip <|endoftext|>")
    }
}
