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

    func testModelVariantConstants() {
        XCTAssertEqual(ParakeetASRModel.defaultModelId, "aufklarer/Parakeet-TDT-v3-CoreML-INT8")
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

}

// MARK: - E2E Tests (require model download)

final class E2EParakeetASRTests: XCTestCase {

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

    func testWordConfidenceFromRealAudio() async throws {
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
        let result = model.transcribeWithLanguage(audio: audio, sampleRate: 16000, language: nil)

        // Verify overall confidence
        XCTAssertGreaterThan(result.confidence, 0.5, "Overall confidence should be high for clean English audio")
        XCTAssertLessThanOrEqual(result.confidence, 1.0, "Confidence must be <= 1.0")

        // Verify per-word confidences exist
        XCTAssertNotNil(result.words, "Should return per-word confidences")
        guard let words = result.words else { return }

        XCTAssertGreaterThan(words.count, 3, "Should have multiple words")
        print("Word confidences:")
        for w in words {
            print("  \(w.word): \(String(format: "%.3f", w.confidence))")
            // Each word confidence must be in [0, 1]
            XCTAssertGreaterThanOrEqual(w.confidence, 0.0)
            XCTAssertLessThanOrEqual(w.confidence, 1.0)
        }

        // Clean English audio should have mostly high-confidence words
        let highConfWords = words.filter { $0.confidence > 0.5 }
        XCTAssertGreaterThan(highConfWords.count, words.count / 2,
            "Most words should have confidence > 0.5 for clean audio")

        // Verify key words are present with reasonable confidence
        let guaranteeWord = words.first { $0.word.lowercased().contains("guarantee") }
        XCTAssertNotNil(guaranteeWord, "Should find 'guarantee' in word list")
        if let g = guaranteeWord {
            XCTAssertGreaterThan(g.confidence, 0.3, "'guarantee' should have decent confidence")
        }
    }

    func testGermanTranscription() async throws {
        let model = try await loadParakeetModel()

        guard let audioURL = Bundle.module.url(forResource: "test_audio_german", withExtension: "wav") else {
            throw XCTSkip("test_audio_german.wav not found in test resources")
        }

        let audio = try AudioFileLoader.load(url: audioURL, targetSampleRate: 16000)
        let result = try model.transcribeAudio(audio, sampleRate: 16000)

        XCTAssertFalse(result.isEmpty, "German transcription should not be empty")
        let lower = result.lowercased()
        // Parakeet v3 supports 25 European languages including German
        // TTS-generated audio — Parakeet should at least get "guten tag"
        XCTAssertTrue(lower.contains("guten tag"), "Should contain 'guten tag', got: \(result)")
        print("German transcription: \(result)")
    }

    // MARK: - Helpers

    private func loadParakeetModel() async throws -> ParakeetASRModel {
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
        return try await ParakeetASRModel.fromPretrained(modelId: modelId)
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

}

// MARK: - Unit Tests (continued, no model download)

final class ParakeetASRUnitTests: XCTestCase {

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

    // MARK: - Confidence Tests

    func testTranscriptionResultHasConfidence() {
        let result = TranscriptionResult(text: "hello", confidence: 0.85)
        XCTAssertEqual(Double(result.confidence), 0.85, accuracy: 0.01)
    }

    func testTranscriptionResultDefaultConfidence() {
        let result = TranscriptionResult(text: "hello")
        XCTAssertEqual(result.confidence, 0.0)
    }

    func testConfidenceExpRange() {
        // exp(mean log-prob) confidence — must be in [0, 1]
        for logProb: Float in [-10, -5, -1, -0.1, 0] {
            let confidence = min(1.0, exp(logProb))
            XCTAssertGreaterThanOrEqual(confidence, 0.0)
            XCTAssertLessThanOrEqual(confidence, 1.0)
        }
    }

    // MARK: - Per-Word Confidence Tests

    func testDecodeWordsBasic() {
        let vocab = ParakeetVocabulary(idToToken: [
            0: "\u{2581}the",
            1: "\u{2581}cat",
            2: "\u{2581}sat",
        ])
        let logProbs: [Float] = [-0.1, -0.5, -0.2]
        let words = vocab.decodeWords([0, 1, 2], logProbs: logProbs)

        XCTAssertEqual(words.count, 3)
        XCTAssertEqual(words[0].word, "the")
        XCTAssertEqual(words[1].word, "cat")
        XCTAssertEqual(words[2].word, "sat")

        // Each word's confidence should be exp(log_prob)
        XCTAssertEqual(Double(words[0].confidence), Double(exp(Float(-0.1))), accuracy: 0.01)
        XCTAssertEqual(Double(words[1].confidence), Double(exp(Float(-0.5))), accuracy: 0.01)
        XCTAssertEqual(Double(words[2].confidence), Double(exp(Float(-0.2))), accuracy: 0.01)
    }

    func testDecodeWordsSubword() {
        let vocab = ParakeetVocabulary(idToToken: [
            0: "\u{2581}un",
            1: "believ",
            2: "able",
            3: "\u{2581}word",
        ])
        let logProbs: [Float] = [-0.3, -0.5, -0.2, -0.1]
        let words = vocab.decodeWords([0, 1, 2, 3], logProbs: logProbs)

        XCTAssertEqual(words.count, 2)
        XCTAssertEqual(words[0].word, "unbelievable")
        XCTAssertEqual(words[1].word, "word")

        // "unbelievable" confidence = exp(mean(-0.3, -0.5, -0.2))
        let expectedConf: Float = exp((-0.3 + -0.5 + -0.2) / 3.0)
        XCTAssertEqual(Double(words[0].confidence), Double(expectedConf), accuracy: 0.01)
    }

    func testDecodeWordsEmpty() {
        let vocab = ParakeetVocabulary(idToToken: [:])
        let words = vocab.decodeWords([], logProbs: [])
        XCTAssertTrue(words.isEmpty)
    }

    func testDecodeWordsMismatchedLengths() {
        let vocab = ParakeetVocabulary(idToToken: [0: "\u{2581}hi"])
        // Mismatched token/logProb counts — should return single word with 0 confidence
        let words = vocab.decodeWords([0], logProbs: [])
        XCTAssertEqual(words.count, 1)
        XCTAssertEqual(words[0].word, "hi")
        XCTAssertEqual(words[0].confidence, 0)
    }

    func testTranscriptionResultWithWords() {
        let words = [
            WordConfidence(word: "hello", confidence: 0.95),
            WordConfidence(word: "world", confidence: 0.88),
        ]
        let result = TranscriptionResult(text: "hello world", confidence: 0.91, words: words)
        XCTAssertEqual(result.words?.count, 2)
        XCTAssertEqual(result.words?[0].word, "hello")
        XCTAssertEqual(Double(result.words?[1].confidence ?? 0), 0.88, accuracy: 0.01)
    }

    func testTranscriptionResultWithoutWords() {
        let result = TranscriptionResult(text: "hello")
        XCTAssertNil(result.words)
    }

    func testTranscriptionResultBackwardCompat() {
        // Existing callers that don't pass words should still work
        let result = TranscriptionResult(text: "test", language: "english", confidence: 0.9)
        XCTAssertNil(result.words)
        XCTAssertEqual(result.text, "test")
        XCTAssertEqual(Double(result.confidence), 0.9, accuracy: 0.01)
    }

    func testLogSoftmaxKnownValues() {
        // Verify log-softmax math: for logits [2, 1, 0],
        // log_softmax[0] = 2 - log(e^2 + e^1 + e^0) = 2 - log(7.389 + 2.718 + 1) ≈ 2 - 2.408 ≈ -0.408
        // This tests the same math used in TDTGreedyDecoder.logSoftmax
        let logits: [Float] = [2.0, 1.0, 0.0]
        let maxVal = logits.max()!
        let sumExp = logits.map { exp($0 - maxVal) }.reduce(0, +)
        let logSumExp = log(sumExp) + maxVal

        let logProb0 = logits[0] - logSumExp
        let logProb1 = logits[1] - logSumExp
        let logProb2 = logits[2] - logSumExp

        // Softmax probs should sum to ~1
        let probSum = exp(logProb0) + exp(logProb1) + exp(logProb2)
        XCTAssertEqual(Double(probSum), 1.0, accuracy: 0.001)

        // log_softmax values should be negative
        XCTAssertLessThan(logProb0, 0)
        XCTAssertLessThan(logProb1, 0)
        XCTAssertLessThan(logProb2, 0)

        // Highest logit should have highest log-prob
        XCTAssertGreaterThan(logProb0, logProb1)
        XCTAssertGreaterThan(logProb1, logProb2)

        // exp(log_softmax) ≈ known softmax values
        XCTAssertEqual(Double(exp(logProb0)), 0.665, accuracy: 0.01) // e^2 / sum
        XCTAssertEqual(Double(exp(logProb1)), 0.245, accuracy: 0.01) // e^1 / sum
        XCTAssertEqual(Double(exp(logProb2)), 0.090, accuracy: 0.01) // e^0 / sum
    }

    func testWordConfidenceRange() {
        // All word confidences from decodeWords must be in [0, 1]
        let vocab = ParakeetVocabulary(idToToken: [
            0: "\u{2581}high",
            1: "\u{2581}medium",
            2: "\u{2581}low",
        ])
        // Range of log-probs: near-0 (high conf) to very negative (low conf)
        let logProbs: [Float] = [-0.01, -1.0, -5.0]
        let words = vocab.decodeWords([0, 1, 2], logProbs: logProbs)

        for word in words {
            XCTAssertGreaterThanOrEqual(word.confidence, 0.0, "\(word.word) confidence < 0")
            XCTAssertLessThanOrEqual(word.confidence, 1.0, "\(word.word) confidence > 1")
        }
        // Higher log-prob → higher confidence
        XCTAssertGreaterThan(words[0].confidence, words[1].confidence)
        XCTAssertGreaterThan(words[1].confidence, words[2].confidence)
    }
}
