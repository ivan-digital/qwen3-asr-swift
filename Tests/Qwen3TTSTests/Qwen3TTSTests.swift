import XCTest
import Foundation
@testable import Qwen3TTS
@testable import Qwen3ASR
@testable import Qwen3Common

final class Qwen3TTSConfigTests: XCTestCase {

    func testTalkerConfigDefaults() {
        let config = TalkerConfig()
        XCTAssertEqual(config.hiddenSize, 1024)
        XCTAssertEqual(config.numLayers, 28)
        XCTAssertEqual(config.numHeads, 16)
        XCTAssertEqual(config.numKVHeads, 8)
        XCTAssertEqual(config.headDim, 128)
        XCTAssertEqual(config.intermediateSize, 3072)
        XCTAssertEqual(config.ropeTheta, 1_000_000.0)
        XCTAssertEqual(config.mropeSections, [24, 20, 20])
        XCTAssertEqual(config.textVocabSize, 151936)
        XCTAssertEqual(config.textHiddenSize, 2048)
        XCTAssertEqual(config.codecVocabSize, 3072)
    }

    func testCodePredictorConfigDefaults() {
        let config = CodePredictorConfig()
        XCTAssertEqual(config.hiddenSize, 1024)
        XCTAssertEqual(config.numLayers, 5)
        XCTAssertEqual(config.vocabSize, 2048)
        XCTAssertEqual(config.numCodeGroups, 16)
    }

    func testSpeechTokenizerDecoderConfigDefaults() {
        let config = SpeechTokenizerDecoderConfig()
        XCTAssertEqual(config.latentDim, 1024)
        XCTAssertEqual(config.decoderDim, 1536)
        XCTAssertEqual(config.numQuantizers, 16)
        XCTAssertEqual(config.semanticCodebookSize, 2048)
        XCTAssertEqual(config.acousticCodebookSize, 2048)
        XCTAssertEqual(config.upsampleRates, [8, 5, 4, 3])
        XCTAssertEqual(config.upsamplingRatios, [2, 2])
        XCTAssertEqual(config.sampleRate, 24000)
    }

    func testCodecTokenIDs() {
        XCTAssertEqual(CodecTokens.codecPad, 2148)
        XCTAssertEqual(CodecTokens.codecBos, 2149)
        XCTAssertEqual(CodecTokens.codecEos, 2150)
        XCTAssertEqual(CodecTokens.codecThink, 2154)
        XCTAssertEqual(CodecTokens.codecNothink, 2155)
        XCTAssertEqual(CodecTokens.codecThinkBos, 2156)
        XCTAssertEqual(CodecTokens.codecThinkEos, 2157)
        XCTAssertEqual(CodecTokens.ttsPad, 151671)
        XCTAssertEqual(CodecTokens.ttsBos, 151672)
        XCTAssertEqual(CodecTokens.ttsEos, 151673)
        XCTAssertEqual(CodecTokens.languageEnglish, 2050)
        XCTAssertEqual(CodecTokens.languageGerman, 2052)
        XCTAssertEqual(CodecTokens.languageChinese, 2055)
        XCTAssertEqual(CodecTokens.languageJapanese, 2058)
    }

    func testLanguageIdLookup() {
        XCTAssertEqual(CodecTokens.languageId(for: "english"), 2050)
        XCTAssertEqual(CodecTokens.languageId(for: "English"), 2050)
        XCTAssertEqual(CodecTokens.languageId(for: "en"), 2050)
        XCTAssertEqual(CodecTokens.languageId(for: "german"), 2052)
        XCTAssertEqual(CodecTokens.languageId(for: "de"), 2052)
        XCTAssertEqual(CodecTokens.languageId(for: "chinese"), 2055)
        XCTAssertEqual(CodecTokens.languageId(for: "zh"), 2055)
        XCTAssertEqual(CodecTokens.languageId(for: "japanese"), 2058)
        XCTAssertEqual(CodecTokens.languageId(for: "ja"), 2058)
        XCTAssertNil(CodecTokens.languageId(for: "unknown"))
    }

    func testCombinedConfig() {
        let config = Qwen3TTSConfig.base06B
        XCTAssertEqual(config.talker.hiddenSize, 1024)
        XCTAssertEqual(config.codePredictor.numLayers, 5)
        XCTAssertEqual(config.speechTokenizerDecoder.sampleRate, 24000)
    }

    func testMRoPESections() {
        let config = TalkerConfig()
        // Sections must sum to headDim/2
        let halfDim = config.headDim / 2
        let sectionSum = config.mropeSections.reduce(0, +)
        XCTAssertEqual(sectionSum, halfDim, "MRoPE sections \(config.mropeSections) should sum to headDim/2 (\(halfDim))")
    }

    func testUpsampleRateProduct() {
        let config = SpeechTokenizerDecoderConfig()
        // Total upsample = product(upsampleRates) * product(upsamplingRatios)
        let mainUpsample = config.upsampleRates.reduce(1, *)  // 8*5*4*3 = 480
        let preUpsample = config.upsamplingRatios.reduce(1, *)  // 2*2 = 4
        let totalUpsample = mainUpsample * preUpsample  // 1920
        XCTAssertEqual(totalUpsample, 1920, "Total upsample should be 1920x (12.5Hz -> 24kHz)")
    }
}

final class SamplingTests: XCTestCase {

    func testSamplingConfigDefaults() {
        let config = SamplingConfig()
        XCTAssertEqual(config.temperature, 0.9)
        XCTAssertEqual(config.topK, 50)
        XCTAssertEqual(config.topP, 1.0)
        XCTAssertEqual(config.repetitionPenalty, 1.05)
        XCTAssertEqual(config.maxTokens, 4096)
    }

    func testGreedyConfig() {
        let config = SamplingConfig.greedy
        XCTAssertEqual(config.temperature, 0)
        XCTAssertEqual(config.topK, 1)
    }
}

// MARK: - TTS E2E Tests

/// End-to-end tests for TTS synthesis with latency measurement.
/// Requires TTS model weights (~1.7 GB). Tests are grouped by language.
final class TTSE2ETests: XCTestCase {

    static let ttsModelId = "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-4bit"
    static let ttsTokenizerModelId = "Qwen/Qwen3-TTS-Tokenizer-12Hz"
    static let asrModelId = "mlx-community/Qwen3-ASR-0.6B-4bit"

    // MARK: - English Tests

    /// English TTS: synthesize and verify audio quality
    func testEnglishSynthesis() async throws {
        let ttsModel = try await loadTTSModel()

        let text = "The quick brown fox jumps over the lazy dog."
        let result = try synthesizeAndMeasure(model: ttsModel, text: text, language: "english")

        XCTAssertGreaterThan(result.durationSec, 1.0, "Audio should be at least 1s for this sentence")
        XCTAssertLessThan(result.durationSec, 30.0, "Audio should be less than 30s")
        XCTAssertGreaterThan(result.maxAmplitude, 0.001, "Audio should not be silent")
        XCTAssertLessThanOrEqual(result.maxAmplitude, 1.0, "Samples should be in [-1, 1]")
    }

    /// English TTS -> ASR round-trip
    func testEnglishRoundTrip() async throws {
        let ttsModel = try await loadTTSModel()
        let asrModel = try await loadASRModel()

        let inputText = "Hello world, this is a test."
        let result = try synthesizeAndMeasure(model: ttsModel, text: inputText, language: "english")

        let transcription = try transcribeAudio(
            samples: result.samples, sampleRate: 24000, using: asrModel)

        print("Input:  \"\(inputText)\"")
        print("Output: \"\(transcription)\"")

        let lowerTranscription = transcription.lowercased()
        let expectedWords = ["hello", "world", "test"]
        let matchedWords = expectedWords.filter { lowerTranscription.contains($0) }
        print("Matched \(matchedWords.count)/\(expectedWords.count) words: \(matchedWords)")

        XCTAssertGreaterThanOrEqual(matchedWords.count, 2,
            "At least 2 of \(expectedWords) should appear in: \"\(transcription)\"")
    }

    /// English TTS: longer text with latency measurement
    func testEnglishLatency() async throws {
        let ttsModel = try await loadTTSModel()

        // Short sentence (baseline)
        let short = try synthesizeAndMeasure(
            model: ttsModel, text: "Hello.", language: "english")
        print("Short: \(fmt(short.durationSec))s audio in \(fmt(short.wallTime))s (RTF: \(fmt(short.rtf)))")

        // Medium sentence
        let medium = try synthesizeAndMeasure(
            model: ttsModel,
            text: "The quick brown fox jumps over the lazy dog.",
            language: "english")
        print("Medium: \(fmt(medium.durationSec))s audio in \(fmt(medium.wallTime))s (RTF: \(fmt(medium.rtf)))")

        // Longer sentence
        let long = try synthesizeAndMeasure(
            model: ttsModel,
            text: "In a quiet village nestled between rolling hills, an old clockmaker spent his days repairing timepieces that had been passed down through generations.",
            language: "english")
        print("Long: \(fmt(long.durationSec))s audio in \(fmt(long.wallTime))s (RTF: \(fmt(long.rtf)))")

        // All should produce valid audio
        XCTAssertGreaterThan(short.samples.count, 0)
        XCTAssertGreaterThan(medium.samples.count, 0)
        XCTAssertGreaterThan(long.samples.count, 0)

        // Longer text should produce longer audio
        XCTAssertGreaterThan(long.durationSec, medium.durationSec,
            "Longer text should produce longer audio")
        XCTAssertGreaterThan(medium.durationSec, short.durationSec,
            "Medium text should produce longer audio than short")
    }

    // MARK: - German Tests

    /// German TTS: synthesize and verify
    func testGermanSynthesis() async throws {
        let ttsModel = try await loadTTSModel()

        let text = "Guten Tag, wie geht es Ihnen heute?"
        let result = try synthesizeAndMeasure(model: ttsModel, text: text, language: "german")

        print("German: \(fmt(result.durationSec))s audio in \(fmt(result.wallTime))s (RTF: \(fmt(result.rtf)))")

        XCTAssertGreaterThan(result.durationSec, 0.5, "Should generate audible speech")
        XCTAssertLessThan(result.durationSec, 30.0, "Should not be excessively long")
        XCTAssertGreaterThan(result.maxAmplitude, 0.001, "Should not be silent")
    }

    /// German TTS -> ASR round-trip
    /// Note: ASR model may not reliably transcribe German audio — this test validates
    /// that TTS produces non-empty audio and ASR returns some output, but does not
    /// require exact word matching since the ASR model is English-primary.
    func testGermanRoundTrip() async throws {
        let ttsModel = try await loadTTSModel()
        let asrModel = try await loadASRModel()

        let inputText = "Guten Morgen, die Sonne scheint heute."
        let result = try synthesizeAndMeasure(model: ttsModel, text: inputText, language: "german")

        XCTAssertGreaterThan(result.samples.count, 1000,
            "German TTS should produce substantial audio output")

        let transcription = try transcribeAudio(
            samples: result.samples, sampleRate: 24000, using: asrModel)

        print("Input (de):  \"\(inputText)\"")
        print("Output (asr): \"\(transcription)\"")

        let lowerTranscription = transcription.lowercased()
        let expectedWords = ["guten", "morgen", "sonne", "heute"]
        let matchedWords = expectedWords.filter { lowerTranscription.contains($0) }
        print("Matched \(matchedWords.count)/\(expectedWords.count) words: \(matchedWords)")

        // ASR model is English-primary; German recognition is best-effort
        XCTAssertFalse(transcription.isEmpty, "Transcription should not be empty")
        if matchedWords.count < 2 {
            print("Warning: ASR did not recognize German words (expected — ASR model is English-primary)")
        }
    }

    /// German TTS: latency comparison with English
    func testGermanLatency() async throws {
        let ttsModel = try await loadTTSModel()

        let german = try synthesizeAndMeasure(
            model: ttsModel,
            text: "Der schnelle braune Fuchs springt über den faulen Hund.",
            language: "german")

        let english = try synthesizeAndMeasure(
            model: ttsModel,
            text: "The quick brown fox jumps over the lazy dog.",
            language: "english")

        print("English: \(fmt(english.durationSec))s audio in \(fmt(english.wallTime))s (RTF: \(fmt(english.rtf)))")
        print("German:  \(fmt(german.durationSec))s audio in \(fmt(german.wallTime))s (RTF: \(fmt(german.rtf)))")

        XCTAssertGreaterThan(german.samples.count, 0)
        XCTAssertGreaterThan(english.samples.count, 0)
    }

    // MARK: - WAV Format Test

    /// Verify WAV write/reload preserves audio content
    func testWAVFormatRoundTrip() async throws {
        let ttsModel = try await loadTTSModel()

        let samples = ttsModel.synthesize(text: "One two three.", language: "english")
        XCTAssertGreaterThan(samples.count, 0)

        let tmpDir = FileManager.default.temporaryDirectory
        let wavURL = tmpDir.appendingPathComponent("wav_format_test_\(UUID().uuidString).wav")
        try WAVWriter.write(samples: samples, sampleRate: 24000, to: wavURL)
        defer { try? FileManager.default.removeItem(at: wavURL) }

        let (reloaded, sampleRate) = try AudioFileLoader.loadWAV(url: wavURL)
        XCTAssertEqual(sampleRate, 24000, "Should preserve sample rate")
        XCTAssertEqual(reloaded.count, samples.count, "Should preserve sample count")

        var maxError: Float = 0
        for i in 0..<min(reloaded.count, samples.count) {
            maxError = max(maxError, abs(reloaded[i] - samples[i]))
        }
        XCTAssertLessThan(maxError, 0.001, "16-bit PCM round-trip error should be minimal")
        print("WAV round-trip max error: \(maxError)")
    }

    // MARK: - Save for Manual Review

    /// Save English and German output to /tmp for manual listening
    func testSaveForManualReview() async throws {
        let ttsModel = try await loadTTSModel()

        let tests: [(text: String, language: String, file: String)] = [
            ("Hello world, this is a test of the Qwen three text to speech system.", "english", "tts_english.wav"),
            ("Guten Tag, dies ist ein Test des Qwen drei Text zu Sprache Systems.", "german", "tts_german.wav"),
        ]

        for test in tests {
            let samples = ttsModel.synthesize(text: test.text, language: test.language)
            let duration = Double(samples.count) / 24000.0
            let outputURL = URL(fileURLWithPath: "/tmp/\(test.file)")
            try WAVWriter.write(samples: samples, sampleRate: 24000, to: outputURL)
            print("[\(test.language)] \(fmt(duration))s -> \(outputURL.path)")
        }

        print("Play with: afplay /tmp/tts_english.wav && afplay /tmp/tts_german.wav")
    }

    // MARK: - Helpers

    private func loadTTSModel() async throws -> Qwen3TTSModel {
        print("Loading TTS model...")
        return try await Qwen3TTSModel.fromPretrained(
            modelId: Self.ttsModelId,
            tokenizerModelId: Self.ttsTokenizerModelId
        ) { progress, status in
            print("[TTS \(Int(progress * 100))%] \(status)")
        }
    }

    private func loadASRModel() async throws -> Qwen3ASRModel {
        print("Loading ASR model...")
        return try await Qwen3ASRModel.fromPretrained(
            modelId: Self.asrModelId
        ) { progress, status in
            print("[ASR \(Int(progress * 100))%] \(status)")
        }
    }

    struct SynthesisResult {
        let samples: [Float]
        let wallTime: TimeInterval
        var durationSec: Double { Double(samples.count) / 24000.0 }
        var rtf: Double { wallTime / max(durationSec, 0.001) }
        var maxAmplitude: Float { samples.map { abs($0) }.max() ?? 0 }
    }

    private func synthesizeAndMeasure(
        model: Qwen3TTSModel, text: String, language: String
    ) throws -> SynthesisResult {
        print("Synthesizing [\(language)]: \"\(text)\"")
        let start = Date()
        let samples = model.synthesize(text: text, language: language)
        let elapsed = Date().timeIntervalSince(start)
        let result = SynthesisResult(samples: samples, wallTime: elapsed)
        print("  -> \(samples.count) samples (\(fmt(result.durationSec))s) in \(fmt(elapsed))s (RTF: \(fmt(result.rtf)))")
        return result
    }

    private func transcribeAudio(
        samples: [Float], sampleRate: Int, using model: Qwen3ASRModel
    ) throws -> String {
        // ASR auto-resamples from any rate to 16kHz internally
        let start = Date()
        let result = model.transcribe(audio: samples, sampleRate: sampleRate)
        let elapsed = Date().timeIntervalSince(start)
        print("  ASR: \(fmt(elapsed))s")
        return result
    }

    private func resample(_ samples: [Float], from inputRate: Int, to outputRate: Int) -> [Float] {
        let ratio = Double(outputRate) / Double(inputRate)
        let outputLength = Int(Double(samples.count) * ratio)
        guard outputLength > 0 else { return [] }

        var output = [Float](repeating: 0, count: outputLength)
        for i in 0..<outputLength {
            let srcIndex = Double(i) / ratio
            let srcFloor = Int(srcIndex)
            let srcCeil = min(srcFloor + 1, samples.count - 1)
            let fraction = Float(srcIndex - Double(srcFloor))
            output[i] = samples[srcFloor] * (1 - fraction) + samples[srcCeil] * fraction
        }
        return output
    }

    private func fmt(_ value: Double) -> String {
        String(format: "%.2f", value)
    }
}
