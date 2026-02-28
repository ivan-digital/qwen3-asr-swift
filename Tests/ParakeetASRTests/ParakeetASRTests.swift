import XCTest
@testable import ParakeetASR
import AudioCommon

final class ParakeetASRTests: XCTestCase {

    // MARK: - Configuration Tests

    func testDefaultConfig() {
        let config = ParakeetConfig.default
        XCTAssertEqual(config.numMelBins, 128)
        XCTAssertEqual(config.sampleRate, 16000)
        XCTAssertEqual(config.nFFT, 512)
        XCTAssertEqual(config.hopLength, 160)
        XCTAssertEqual(config.winLength, 400)
        XCTAssertEqual(config.preEmphasis, 0.97)
        XCTAssertEqual(config.encoderHidden, 1024)
        XCTAssertEqual(config.encoderLayers, 24)
        XCTAssertEqual(config.subsamplingFactor, 8)
        XCTAssertEqual(config.decoderHidden, 640)
        XCTAssertEqual(config.decoderLayers, 2)
        XCTAssertEqual(config.vocabSize, 1024)
        XCTAssertEqual(config.blankTokenId, 1024)
        XCTAssertEqual(config.numDurationBins, 5)
        XCTAssertEqual(config.durationBins, [0, 1, 2, 3, 4])
    }

    func testConfigCodable() throws {
        let original = ParakeetConfig.default
        let data = try JSONEncoder().encode(original)
        let decoded = try JSONDecoder().decode(ParakeetConfig.self, from: data)

        XCTAssertEqual(decoded.numMelBins, original.numMelBins)
        XCTAssertEqual(decoded.sampleRate, original.sampleRate)
        XCTAssertEqual(decoded.nFFT, original.nFFT)
        XCTAssertEqual(decoded.hopLength, original.hopLength)
        XCTAssertEqual(decoded.winLength, original.winLength)
        XCTAssertEqual(decoded.preEmphasis, original.preEmphasis)
        XCTAssertEqual(decoded.encoderHidden, original.encoderHidden)
        XCTAssertEqual(decoded.encoderLayers, original.encoderLayers)
        XCTAssertEqual(decoded.subsamplingFactor, original.subsamplingFactor)
        XCTAssertEqual(decoded.decoderHidden, original.decoderHidden)
        XCTAssertEqual(decoded.decoderLayers, original.decoderLayers)
        XCTAssertEqual(decoded.vocabSize, original.vocabSize)
        XCTAssertEqual(decoded.blankTokenId, original.blankTokenId)
        XCTAssertEqual(decoded.numDurationBins, original.numDurationBins)
        XCTAssertEqual(decoded.durationBins, original.durationBins)
    }

    func testConfigSendable() async {
        let config = ParakeetConfig.default
        let result = await Task { config }.value
        XCTAssertEqual(result.numMelBins, config.numMelBins)
        XCTAssertEqual(result.encoderHidden, config.encoderHidden)
    }

    // MARK: - Vocabulary Tests

    func testVocabularyDecode() {
        let vocab = ParakeetVocabulary(idToToken: [
            0: "\u{2581}the",
            1: "\u{2581}cat",
            2: "\u{2581}sat",
        ])

        let text = vocab.decode([0, 1, 2])
        XCTAssertEqual(text, "the cat sat")
    }

    func testVocabularyDecodeSkipsUnknown() {
        let vocab = ParakeetVocabulary(idToToken: [
            0: "\u{2581}hello",
            1: "\u{2581}world",
        ])

        // Token ID 999 is not in vocab — should be skipped
        let text = vocab.decode([0, 999, 1])
        XCTAssertEqual(text, "hello world")
    }

    func testVocabularyDecodeSubword() {
        let vocab = ParakeetVocabulary(idToToken: [
            0: "\u{2581}un",
            1: "believ",
            2: "able",
        ])

        let text = vocab.decode([0, 1, 2])
        XCTAssertEqual(text, "unbelievable")
    }

    func testVocabularyEmpty() {
        let vocab = ParakeetVocabulary(idToToken: [:])
        let text = vocab.decode([0, 1, 2])
        XCTAssertEqual(text, "")
    }

    // MARK: - Integration Tests

    func testModelLoading() async throws {
        // Skip if model is not cached locally
        let modelId = ParakeetASRModel.defaultModelId
        let cacheDir: URL
        do {
            cacheDir = try HuggingFaceDownloader.getCacheDirectory(for: modelId)
        } catch {
            throw XCTSkip("Cannot resolve cache directory: \(error)")
        }

        let encoderPath = cacheDir.appendingPathComponent("encoder.mlmodelc")
        guard FileManager.default.fileExists(atPath: encoderPath.path) else {
            throw XCTSkip("Parakeet model not cached at \(cacheDir.path)")
        }

        let model = try await ParakeetASRModel.fromPretrained(modelId: modelId)
        XCTAssertEqual(model.config.sampleRate, 16000)
        XCTAssertEqual(model.config.encoderHidden, 1024)
    }

    func testTranscription() async throws {
        let modelId = ParakeetASRModel.defaultModelId
        let cacheDir: URL
        do {
            cacheDir = try HuggingFaceDownloader.getCacheDirectory(for: modelId)
        } catch {
            throw XCTSkip("Cannot resolve cache directory: \(error)")
        }

        let encoderPath = cacheDir.appendingPathComponent("encoder.mlmodelc")
        guard FileManager.default.fileExists(atPath: encoderPath.path) else {
            throw XCTSkip("Parakeet model not cached at \(cacheDir.path)")
        }

        let model = try await ParakeetASRModel.fromPretrained(modelId: modelId)

        // Load test audio
        guard let audioURL = Bundle.module.url(forResource: "test_audio", withExtension: "wav") else {
            throw XCTSkip("test_audio.wav not found in test resources")
        }

        let audio = try AudioFileLoader.load(url: audioURL, targetSampleRate: 16000)
        let result = try model.transcribeAudio(audio, sampleRate: 16000)

        XCTAssertFalse(result.isEmpty, "Transcription should not be empty")
        // The test audio says: "Can you guarantee that the replacement part will be shipped tomorrow?"
        let lower = result.lowercased()
        XCTAssertTrue(lower.contains("guarantee") || lower.contains("replacement") || lower.contains("shipped"),
                      "Transcription should contain expected words, got: \(result)")
    }
}
