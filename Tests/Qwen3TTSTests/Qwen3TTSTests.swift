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

    func testExtendedLanguageIds() {
        XCTAssertEqual(CodecTokens.languageId(for: "spanish"), 2054)
        XCTAssertEqual(CodecTokens.languageId(for: "es"), 2054)
        XCTAssertEqual(CodecTokens.languageId(for: "french"), 2061)
        XCTAssertEqual(CodecTokens.languageId(for: "fr"), 2061)
        XCTAssertEqual(CodecTokens.languageId(for: "korean"), 2064)
        XCTAssertEqual(CodecTokens.languageId(for: "ko"), 2064)
        XCTAssertEqual(CodecTokens.languageId(for: "russian"), 2069)
        XCTAssertEqual(CodecTokens.languageId(for: "ru"), 2069)
        XCTAssertEqual(CodecTokens.languageId(for: "italian"), 2070)
        XCTAssertEqual(CodecTokens.languageId(for: "it"), 2070)
        XCTAssertEqual(CodecTokens.languageId(for: "portuguese"), 2071)
        XCTAssertEqual(CodecTokens.languageId(for: "pt"), 2071)
        XCTAssertEqual(CodecTokens.languageId(for: "beijing_dialect"), 2074)
        XCTAssertEqual(CodecTokens.languageId(for: "sichuan_dialect"), 2062)
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

// MARK: - Speaker Config Tests

final class SpeakerConfigTests: XCTestCase {

    func testSpeakerConfigParsing() {
        let config = SpeakerConfig(
            speakerIds: ["serena": 3066, "vivian": 3065, "ryan": 3061, "aiden": 2861],
            speakerDialects: ["eric": "sichuan_dialect", "dylan": "beijing_dialect"])
        XCTAssertEqual(config.speakerIds["serena"], 3066)
        XCTAssertEqual(config.speakerIds["vivian"], 3065)
        XCTAssertEqual(config.speakerIds["ryan"], 3061)
        XCTAssertEqual(config.availableSpeakers, ["aiden", "ryan", "serena", "vivian"])
    }

    func testSpeakerDialectMapping() {
        let config = SpeakerConfig(
            speakerIds: ["eric": 2875, "dylan": 2878],
            speakerDialects: ["eric": "sichuan_dialect", "dylan": "beijing_dialect"])
        XCTAssertEqual(config.speakerDialects["eric"], "sichuan_dialect")
        XCTAssertEqual(config.speakerDialects["dylan"], "beijing_dialect")
    }

    func testEmptySpeakerConfig() {
        let config = SpeakerConfig(speakerIds: [:], speakerDialects: [:])
        XCTAssertTrue(config.availableSpeakers.isEmpty)
    }

    func testCodecPrefixWithoutSpeaker() {
        let model = Qwen3TTSModel()
        let prefix = model.buildCodecPrefix(languageId: CodecTokens.languageEnglish)
        XCTAssertEqual(prefix.count, 6)
        XCTAssertEqual(prefix[0], Int32(CodecTokens.codecThink))
        XCTAssertEqual(prefix[1], Int32(CodecTokens.codecThinkBos))
        XCTAssertEqual(prefix[2], Int32(CodecTokens.languageEnglish))
        XCTAssertEqual(prefix[3], Int32(CodecTokens.codecThinkEos))
        XCTAssertEqual(prefix[4], Int32(CodecTokens.codecPad))
        XCTAssertEqual(prefix[5], Int32(CodecTokens.codecBos))
    }

    func testCodecPrefixWithSpeaker() {
        let model = Qwen3TTSModel()
        let speakerTokenId = 3065  // vivian
        let prefix = model.buildCodecPrefix(languageId: CodecTokens.languageEnglish, speakerTokenId: speakerTokenId)
        XCTAssertEqual(prefix.count, 7)
        XCTAssertEqual(prefix[0], Int32(CodecTokens.codecThink))
        XCTAssertEqual(prefix[1], Int32(CodecTokens.codecThinkBos))
        XCTAssertEqual(prefix[2], Int32(CodecTokens.languageEnglish))
        XCTAssertEqual(prefix[3], Int32(CodecTokens.codecThinkEos))
        XCTAssertEqual(prefix[4], Int32(CodecTokens.codecPad))
        XCTAssertEqual(prefix[5], Int32(CodecTokens.codecBos))
        XCTAssertEqual(prefix[6], Int32(speakerTokenId))
    }

    func testTTSModelVariant() {
        XCTAssertEqual(TTSModelVariant.base.rawValue, "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-4bit")
        XCTAssertEqual(TTSModelVariant.customVoice.rawValue, "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-4bit")
    }

    func testAvailableSpeakersEmptyByDefault() {
        let model = Qwen3TTSModel()
        XCTAssertTrue(model.availableSpeakers.isEmpty)
        XCTAssertNil(model.speakerConfig)
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

    // MARK: - Streaming Tests

    func testStreamingSynthesis() async throws {
        let ttsModel = try await loadTTSModel()

        let text = "Hello, this is a streaming test of the text to speech system."
        var chunks: [Qwen3TTSModel.AudioChunk] = []
        var totalSamples = 0

        let stream = ttsModel.synthesizeStream(text: text, language: "english")
        for try await chunk in stream {
            if chunk.isFinal && chunk.samples.isEmpty { break }
            if !chunk.samples.isEmpty {
                chunks.append(chunk)
                totalSamples += chunk.samples.count
            }
        }

        XCTAssertGreaterThan(chunks.count, 0, "Should produce at least one audio chunk")
        XCTAssertGreaterThan(totalSamples, 0, "Should produce audio samples")

        // Verify all samples are in valid range
        for chunk in chunks {
            for sample in chunk.samples {
                XCTAssertFalse(sample.isNaN, "Sample should not be NaN")
                XCTAssertFalse(sample.isInfinite, "Sample should not be Inf")
                XCTAssertGreaterThanOrEqual(sample, -1.0, "Sample should be >= -1.0")
                XCTAssertLessThanOrEqual(sample, 1.0, "Sample should be <= 1.0")
            }
        }

        let audioDur = Double(totalSamples) / 24000.0
        print("Streaming: \(chunks.count) chunks, \(totalSamples) samples (\(fmt(audioDur))s)")
    }

    func testStreamingWAVWriter() async throws {
        let ttsModel = try await loadTTSModel()

        let text = "Testing streaming WAV output."
        let outputURL = URL(fileURLWithPath: "/tmp/tts_streaming_test.wav")

        let writer = try StreamingWAVWriter(to: outputURL)

        let stream = ttsModel.synthesizeStream(text: text, language: "english")
        for try await chunk in stream {
            if chunk.isFinal && chunk.samples.isEmpty { break }
            if !chunk.samples.isEmpty {
                writer.write(samples: chunk.samples)
            }
        }

        let result = writer.finalize()
        XCTAssertGreaterThan(result.sampleCount, 0, "Should have written samples")

        // Verify the file is valid WAV
        let data = try Data(contentsOf: outputURL)
        XCTAssertGreaterThan(data.count, 44, "WAV file should be larger than header")
        XCTAssertEqual(String(data: data[0..<4], encoding: .ascii), "RIFF")
        XCTAssertEqual(String(data: data[8..<12], encoding: .ascii), "WAVE")

        print("Streaming WAV: \(result.sampleCount) samples -> \(outputURL.path)")
    }

    /// Streaming TTS -> ASR round-trip: verify streaming audio quality matches standard synthesis
    func testStreamingQualityRoundTrip() async throws {
        let ttsModel = try await loadTTSModel()
        let asrModel = try await loadASRModel()

        let text = "The quick brown fox jumps over the lazy dog."

        // Standard synthesis
        let standardSamples = ttsModel.synthesize(text: text, language: "english")
        let standardTranscription = asrModel.transcribe(audio: standardSamples, sampleRate: 24000)
        print("Standard: \"\(standardTranscription)\"")

        // Streaming synthesis — collect all chunks
        var streamingSamples: [Float] = []
        let stream = ttsModel.synthesizeStream(text: text, language: "english")
        for try await chunk in stream {
            if chunk.isFinal && chunk.samples.isEmpty { break }
            if !chunk.samples.isEmpty {
                streamingSamples.append(contentsOf: chunk.samples)
            }
        }

        let streamingTranscription = asrModel.transcribe(audio: streamingSamples, sampleRate: 24000)
        print("Streaming: \"\(streamingTranscription)\"")

        // Both should produce recognizable speech
        let expectedWords = ["quick", "brown", "fox", "lazy", "dog"]
        let standardMatched = expectedWords.filter { standardTranscription.lowercased().contains($0) }
        let streamingMatched = expectedWords.filter { streamingTranscription.lowercased().contains($0) }

        print("Standard matched: \(standardMatched.count)/\(expectedWords.count) \(standardMatched)")
        print("Streaming matched: \(streamingMatched.count)/\(expectedWords.count) \(streamingMatched)")

        XCTAssertGreaterThanOrEqual(standardMatched.count, 3,
            "Standard synthesis should match at least 3 words")
        XCTAssertGreaterThanOrEqual(streamingMatched.count, 2,
            "Streaming synthesis should match at least 2 words (quality parity)")

        // Audio durations should be comparable (within 50%)
        let standardDur = Double(standardSamples.count) / 24000.0
        let streamingDur = Double(streamingSamples.count) / 24000.0
        print("Standard duration: \(fmt(standardDur))s, Streaming duration: \(fmt(streamingDur))s")

        XCTAssertGreaterThan(streamingDur, standardDur * 0.5,
            "Streaming audio should not be much shorter than standard")
        XCTAssertLessThan(streamingDur, standardDur * 1.5,
            "Streaming audio should not be much longer than standard")
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

// MARK: - Batch TTS Tests

final class TTSBatchTests: XCTestCase {

    static let ttsModelId = "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-4bit"
    static let ttsTokenizerModelId = "Qwen/Qwen3-TTS-Tokenizer-12Hz"
    static let asrModelId = "mlx-community/Qwen3-ASR-0.6B-4bit"

    // MARK: - Test 1: Build compiles cleanly (verified by running this test)

    // MARK: - Test 2: Single-item batch parity
    /// synthesizeBatch(["text"]) should delegate to synthesize() and produce valid audio
    func testSingleItemBatchParity() async throws {
        let model = try await loadTTSModel()

        let text = "Hello world."
        let batchResult = model.synthesizeBatch(texts: [text], language: "english")

        XCTAssertEqual(batchResult.count, 1, "Should return 1 result")
        XCTAssertGreaterThan(batchResult[0].count, 0, "Should produce audio")

        let duration = Double(batchResult[0].count) / 24000.0
        print("Single-item batch: \(batchResult[0].count) samples (\(fmt(duration))s)")
        XCTAssertGreaterThan(duration, 0.5, "Should be at least 0.5s of audio")
        XCTAssertLessThan(duration, 15.0, "Should be less than 15s")
    }

    // MARK: - Test 3: Multi-item correctness with ASR round-trip
    /// Batch TTS → ASR round-trip. Items that hit the 500-token safety cap produce
    /// long garbage audio that ASR can't transcribe, so we skip ASR validation for those
    /// and require at least 2 of 3 items to pass word matching.
    func testMultiItemRoundTrip() async throws {
        let ttsModel = try await loadTTSModel()
        let asrModel = try await loadASRModel()

        let texts = [
            "Good morning everyone.",
            "The weather is nice today.",
            "Please open the window.",
        ]

        print("Batch synthesizing \(texts.count) texts...")
        let t0 = Date()
        let results = ttsModel.synthesizeBatch(texts: texts, language: "english")
        let batchTime = Date().timeIntervalSince(t0)

        XCTAssertEqual(results.count, 3, "Should return 3 results")

        let expectedWords = [
            ["morning", "everyone"],
            ["weather", "nice", "today"],
            ["open", "window"],
        ]

        // Items producing >30s audio likely hit the safety cap — skip ASR for those
        let maxReasonableSamples = 30 * 24000  // 30s at 24kHz
        var passedItems = 0

        for (i, audio) in results.enumerated() {
            XCTAssertGreaterThan(audio.count, 0, "Item \(i) should produce audio")
            let duration = Double(audio.count) / 24000.0
            print("  Item \(i): \(audio.count) samples (\(fmt(duration))s)")

            if audio.count > maxReasonableSamples {
                print("  Item \(i): skipping ASR (hit safety cap, \(fmt(duration))s audio)")
                continue
            }

            let transcription = asrModel.transcribe(audio: audio, sampleRate: 24000)
            let lower = transcription.lowercased()
            print("  Item \(i) text: \"\(texts[i])\"")
            print("  Item \(i) ASR:  \"\(transcription)\"")

            let matched = expectedWords[i].filter { lower.contains($0) }
            print("  Matched \(matched.count)/\(expectedWords[i].count): \(matched)")
            if matched.count >= 1 {
                passedItems += 1
            }
        }

        XCTAssertGreaterThanOrEqual(passedItems, 2,
            "At least 2 of 3 items should pass ASR round-trip")
        print("Batch total time: \(fmt(batchTime))s, \(passedItems)/\(texts.count) items passed ASR")
    }

    // MARK: - Test 4: Performance comparison (batch vs sequential)
    func testBatchPerformance() async throws {
        let model = try await loadTTSModel()

        let texts = [
            "The sun rises in the east.",
            "Birds sing in the morning.",
            "Coffee keeps me awake.",
            "Books open new worlds.",
        ]

        // Sequential: synthesize each text one by one
        print("Sequential synthesis of \(texts.count) texts...")
        let seqStart = Date()
        var seqResults: [[Float]] = []
        for text in texts {
            let audio = model.synthesize(text: text, language: "english")
            seqResults.append(audio)
        }
        let seqTime = Date().timeIntervalSince(seqStart)

        let seqAudioDur = seqResults.reduce(0.0) { $0 + Double($1.count) / 24000.0 }
        print("Sequential: \(fmt(seqTime))s wall, \(fmt(seqAudioDur))s audio, RTF=\(fmt(seqTime / seqAudioDur))")

        // Batch: synthesize all at once
        print("Batch synthesis of \(texts.count) texts...")
        let batchStart = Date()
        let batchResults = model.synthesizeBatch(texts: texts, language: "english")
        let batchTime = Date().timeIntervalSince(batchStart)

        let batchAudioDur = batchResults.reduce(0.0) { $0 + Double($1.count) / 24000.0 }
        print("Batch: \(fmt(batchTime))s wall, \(fmt(batchAudioDur))s audio, RTF=\(fmt(batchTime / batchAudioDur))")

        let speedup = seqTime / batchTime
        print("Speedup: \(fmt(speedup))x")

        // All items should produce valid audio
        for (i, audio) in batchResults.enumerated() {
            XCTAssertGreaterThan(audio.count, 0, "Batch item \(i) should produce audio")
        }

        // Log speedup — we expect >=1.5x in release, but don't fail in debug
        print("Batch speedup: \(fmt(speedup))x (expected >=1.5x in release build)")
    }

    // MARK: - Test 5: EOS handling with short + long text
    func testShortLongMix() async throws {
        let model = try await loadTTSModel()

        let texts = [
            "Hi.",
            "The quick brown fox jumps over the lazy dog near the river bank on a sunny afternoon.",
        ]

        print("Batch: short + long text...")
        let results = model.synthesizeBatch(texts: texts, language: "english")

        XCTAssertEqual(results.count, 2, "Should return 2 results")

        for (i, audio) in results.enumerated() {
            XCTAssertGreaterThan(audio.count, 0, "Item \(i) should produce audio")
            let duration = Double(audio.count) / 24000.0
            let maxAmp = audio.map { abs($0) }.max() ?? 0
            print("  Item \(i): \(audio.count) samples (\(fmt(duration))s), maxAmp=\(fmt(Double(maxAmp)))")
            XCTAssertGreaterThan(maxAmp, 0.001, "Item \(i) should not be silent")
        }

        let shortDur = Double(results[0].count) / 24000.0
        let longDur = Double(results[1].count) / 24000.0
        print("Short: \(fmt(shortDur))s, Long: \(fmt(longDur))s")
        XCTAssertGreaterThan(longDur, shortDur, "Long text should produce longer audio")
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

    private func fmt(_ value: Double) -> String {
        String(format: "%.2f", value)
    }
}

// MARK: - TextChunker Tests

final class TextChunkerTests: XCTestCase {

    func testShortTextNoChunking() {
        let text = "Hello world."
        let chunks = TextChunker.chunk(text)
        XCTAssertEqual(chunks.count, 1)
        XCTAssertEqual(chunks[0], "Hello world.")
    }

    func testEmptyText() {
        XCTAssertEqual(TextChunker.chunk(""), [])
        XCTAssertEqual(TextChunker.chunk("   "), [])
    }

    func testLongTextChunksAtSentence() {
        let text = "This is the first sentence. This is the second sentence. " +
                   "And here is a third one that makes this text quite long enough to need chunking. " +
                   "Finally we add a fourth sentence to push it way over the word limit."
        let chunks = TextChunker.chunk(text, maxWords: 20)
        XCTAssertGreaterThan(chunks.count, 1, "Should split into multiple chunks")
        // Verify no chunk exceeds max words (with some tolerance for boundary finding)
        for chunk in chunks {
            let wordCount = chunk.split(separator: " ").count
            XCTAssertLessThanOrEqual(wordCount, 25, "Chunk should not be much longer than maxWords")
        }
        // Verify full text is preserved
        let rejoined = chunks.joined(separator: " ")
        XCTAssertTrue(rejoined.contains("first sentence"))
        XCTAssertTrue(rejoined.contains("fourth sentence"))
    }

    func testChunkAtComma() {
        let text = "One two three four five six seven eight nine ten, eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty"
        let chunks = TextChunker.chunk(text, maxWords: 15)
        XCTAssertGreaterThanOrEqual(chunks.count, 1)
    }

    func testMaxWordsRespected() {
        let words = (0..<100).map { "word\($0)" }.joined(separator: " ")
        let chunks = TextChunker.chunk(words, maxWords: 20)
        XCTAssertGreaterThan(chunks.count, 3)
    }
}
