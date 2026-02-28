import CoreML
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
        XCTAssertEqual(config.vocabSize, 8192)
        XCTAssertEqual(config.blankTokenId, 8192)
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
        XCTAssertTrue(lower.contains("guarantee"), "Should contain 'guarantee', got: \(result)")
        XCTAssertTrue(lower.contains("replacement"), "Should contain 'replacement', got: \(result)")
        XCTAssertTrue(lower.contains("shipped"), "Should contain 'shipped', got: \(result)")
    }

    func testWarmup() async throws {
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

        guard let audioURL = Bundle.module.url(forResource: "test_audio", withExtension: "wav") else {
            throw XCTSkip("test_audio.wav not found in test resources")
        }
        let audio = try AudioFileLoader.load(url: audioURL, targetSampleRate: 16000)
        let duration = Float(audio.count) / 16000.0

        // Cold inference (first call triggers CoreML graph compilation)
        let tCold0 = CFAbsoluteTimeGetCurrent()
        let coldResult = try model.transcribeAudio(audio, sampleRate: 16000)
        let coldElapsed = CFAbsoluteTimeGetCurrent() - tCold0

        // warmUp() should succeed (models already compiled from cold run above,
        // but exercises the warmUp API path with 1s dummy audio)
        try model.warmUp()

        // Warm inference — should be faster and produce identical correctness
        let tWarm0 = CFAbsoluteTimeGetCurrent()
        let warmResult = try model.transcribeAudio(audio, sampleRate: 16000)
        let warmElapsed = CFAbsoluteTimeGetCurrent() - tWarm0

        // Verify correctness: same keywords as testTranscription
        let lower = warmResult.lowercased()
        XCTAssertTrue(lower.contains("guarantee"), "Should contain 'guarantee', got: \(warmResult)")
        XCTAssertTrue(lower.contains("replacement"), "Should contain 'replacement', got: \(warmResult)")
        XCTAssertTrue(lower.contains("shipped"), "Should contain 'shipped', got: \(warmResult)")

        // Warm and cold should produce identical output
        XCTAssertEqual(warmResult, coldResult, "Warm and cold transcription should match")

        let coldRTF = coldElapsed / Double(duration)
        let warmRTF = warmElapsed / Double(duration)
        print("Parakeet cold=\(String(format: "%.0f", coldElapsed * 1000))ms (RTF \(String(format: "%.3f", coldRTF))), warm=\(String(format: "%.0f", warmElapsed * 1000))ms (RTF \(String(format: "%.3f", warmRTF)))")
    }

    // MARK: - Performance Tests

    func testTranscriptionLatency() async throws {
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

        guard let audioURL = Bundle.module.url(forResource: "test_audio", withExtension: "wav") else {
            throw XCTSkip("test_audio.wav not found in test resources")
        }

        let audio = try AudioFileLoader.load(url: audioURL, targetSampleRate: 16000)
        let duration = Float(audio.count) / 16000.0

        // Warmup
        _ = try model.transcribeAudio(audio, sampleRate: 16000)

        // Benchmark 3 runs
        var times = [Double]()
        for _ in 0..<3 {
            let t0 = CFAbsoluteTimeGetCurrent()
            let result = try model.transcribeAudio(audio, sampleRate: 16000)
            times.append(CFAbsoluteTimeGetCurrent() - t0)
            XCTAssertFalse(result.isEmpty)
        }
        let avg = times.reduce(0, +) / Double(times.count)
        let best = times.min()!
        let rtf = avg / Double(duration)
        print("Parakeet latency: avg=\(String(format: "%.0f", avg * 1000))ms, best=\(String(format: "%.0f", best * 1000))ms, RTF=\(String(format: "%.3f", rtf))")
    }

    // MARK: - Mel Preprocessing Tests

    func testMelNormalization() throws {
        let config = ParakeetConfig.default
        let preprocessor = MelPreprocessor(config: config)
        // 1s of 440Hz sine at 16kHz
        let audio = (0..<16000).map { Float(sin(2.0 * .pi * 440.0 * Float($0) / 16000.0)) * 0.5 }
        let (mel, melLength) = try preprocessor.extract(audio)

        XCTAssertEqual(mel.shape[0].intValue, 1)
        XCTAssertEqual(mel.shape[1].intValue, 128)
        XCTAssertGreaterThan(melLength, 90)  // ~99 frames for 1s

        // Verify normalization: mean should be near 0 for each bin
        let ptr = mel.dataPointer.assumingMemoryBound(to: Float16.self)
        let nFrames = mel.shape[2].intValue
        for bin in 0..<5 {  // spot-check first 5 bins
            var sum: Float = 0
            for t in 0..<melLength { sum += Float(ptr[bin * nFrames + t]) }
            let mean = sum / Float(melLength)
            XCTAssertEqual(mean, 0.0, accuracy: 0.1, "Bin \(bin) should be ~zero-mean after normalization")
        }
    }
}
