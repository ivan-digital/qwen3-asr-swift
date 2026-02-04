import XCTest
import MLX
@testable import Qwen3ASR

final class Qwen3ASRTests: XCTestCase {

    func testAudioEncoderConfig() {
        let config = Qwen3AudioEncoderConfig.default
        XCTAssertEqual(config.dModel, 896)
        XCTAssertEqual(config.encoderLayers, 18)
        XCTAssertEqual(config.encoderAttentionHeads, 14)
        XCTAssertEqual(config.numMelBins, 128)
        XCTAssertEqual(config.outputDim, 1024)
    }

    func testTextDecoderConfig() {
        let smallConfig = TextDecoderConfig.small
        XCTAssertEqual(smallConfig.hiddenSize, 1024)
        XCTAssertEqual(smallConfig.numLayers, 28)

        let largeConfig = TextDecoderConfig.large
        XCTAssertEqual(largeConfig.hiddenSize, 1536)
        XCTAssertEqual(largeConfig.numLayers, 28)
    }

    func testQwen3ASRConfig() {
        let config = Qwen3ASRConfig.small
        XCTAssertEqual(config.audioTokenIndex, 151646)
        XCTAssertEqual(config.eosTokenId, 151645)
        XCTAssertEqual(config.padTokenId, 151643)
    }

    func testFeatureExtractor() {
        let extractor = WhisperFeatureExtractor()
        XCTAssertEqual(extractor.sampleRate, 24000)
        XCTAssertEqual(extractor.nMels, 128)
        XCTAssertEqual(extractor.hopLength, 192)

        // Test with silent audio
        let silentAudio = [Float](repeating: 0, count: 24000) // 1 second
        let features = extractor.extractFeatures(silentAudio)

        // Check output shape
        XCTAssertEqual(features.dim(0), 128) // mel bins
        XCTAssertGreaterThan(features.dim(1), 0) // time frames
    }

    func testFeatureExtractorWithSineWave() {
        let extractor = WhisperFeatureExtractor()

        // Generate 1 second of 440Hz sine wave
        let sampleRate = 24000
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
        XCTAssertGreaterThan(features.dim(1), 100) // Should have ~125 frames for 1s

        // Features should not be all zeros
        let maxVal = features.max().item(Float.self)
        XCTAssertGreaterThan(maxVal, -100.0) // Log mel, so can be negative
    }

    func testAudioEncoderCreation() {
        // Test that encoder can be created without crashing
        let config = Qwen3AudioEncoderConfig.default
        let encoder = Qwen3AudioEncoder(config: config)

        XCTAssertEqual(encoder.layers.count, 18)
    }

    func testModelCreation() {
        // Test that model can be created without crashing
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
            .appendingPathComponent("mlx-community_Qwen3-ASR-0.6B-4bit")

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
            .appendingPathComponent("mlx-community_Qwen3-ASR-0.6B-4bit")

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
            .appendingPathComponent("mlx-community_Qwen3-ASR-0.6B-4bit")

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

    func testTokenizerSkipsSpecialTokensWithPipes() throws {
        let cacheDir = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first!
            .appendingPathComponent("qwen3-asr")
            .appendingPathComponent("mlx-community_Qwen3-ASR-0.6B-4bit")

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
