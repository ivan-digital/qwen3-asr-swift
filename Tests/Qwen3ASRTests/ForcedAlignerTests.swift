import XCTest
import Foundation
import MLX
@testable import Qwen3ASR
@testable import AudioCommon

/// Tests for the Forced Aligner
final class ForcedAlignerTests: XCTestCase {

    // MARK: - Unit Tests (no model download)

    // MARK: - Default path (matches upstream `tokenize_space_lang`)

    func testTextPreprocessingEnglish() {
        let words = TextPreprocessor.splitIntoWords("Hello world test", language: "English")
        XCTAssertEqual(words, ["Hello", "world", "test"])
    }

    /// Pure Chinese has no whitespace → each Han ideograph becomes its own
    /// token via `split_segment_with_chinese`.
    func testTextPreprocessingChinese() {
        let words = TextPreprocessor.splitIntoWords("你好世界", language: "Chinese")
        XCTAssertEqual(words, ["你", "好", "世", "界"])
    }

    /// Han ideographs peel out of Latin-bordered text but Latin runs stay
    /// grouped. Matches upstream behavior exactly.
    func testTextPreprocessingMixedHanLatin() {
        let words = TextPreprocessor.splitIntoWords("Hello你好world", language: "Chinese")
        XCTAssertEqual(words, ["Hello", "你", "好", "world"])
    }

    /// Punctuation is stripped by `clean_token` (only Unicode L*/N* + `'` are
    /// kept). The full-width period `。` must NOT appear as a token.
    func testTextPreprocessingPunctuationStripped() {
        let words = TextPreprocessor.splitIntoWords("Hello, world!", language: "English")
        XCTAssertEqual(words, ["Hello", "world"])
    }

    func testTextPreprocessingApostropheKept() {
        let words = TextPreprocessor.splitIntoWords("don't stop", language: "English")
        XCTAssertEqual(words, ["don't", "stop"])
    }

    // MARK: - Japanese (NLTokenizer morpheme-level, matches upstream nagisa)

    /// Japanese must NOT split per kana. Upstream uses nagisa morphemes;
    /// we use Apple's NLTokenizer for equivalent granularity.
    /// "おはようございます。今日はいい天気ですね。" should produce a small
    /// number of morphemes (≈6–10), not 21 per-character tokens.
    func testTextPreprocessingJapaneseMorpheme() {
        let words = TextPreprocessor.splitIntoWords(
            "おはようございます。今日はいい天気ですね。",
            language: "japanese")
        XCTAssertGreaterThan(words.count, 1, "Should produce multiple morphemes")
        XCTAssertLessThan(words.count, 15,
            "Japanese should be morpheme-level (~6–10), not per-char (21). Got \(words.count): \(words)")
        // Punctuation must be stripped.
        XCTAssertFalse(words.contains("。"), "Full-width period should be cleaned")
    }

    /// Pure katakana word should be a single morpheme, not 6 per-char tokens.
    func testTextPreprocessingKatakanaSingleWord() {
        let words = TextPreprocessor.splitIntoWords("コンピュータ", language: "ja")
        XCTAssertEqual(words, ["コンピュータ"],
            "Katakana word should NOT split per character — got \(words)")
    }

    /// Pure hiragana greeting should not split per kana.
    func testTextPreprocessingHiraganaGreeting() {
        let words = TextPreprocessor.splitIntoWords("こんにちは", language: "ja")
        XCTAssertEqual(words.count, 1,
            "Hiragana greeting should be one morpheme, got \(words)")
    }

    /// Mixed Japanese + Latin: Latin words stay intact, JP segmented by
    /// NLTokenizer. Punctuation stripped.
    func testTextPreprocessingJapaneseWithLatin() {
        let words = TextPreprocessor.splitIntoWords(
            "iPhoneを使います。",
            language: "japanese")
        XCTAssertTrue(words.contains("iPhone"),
            "Latin run should stay grouped, got: \(words)")
        XCTAssertFalse(words.contains("。"))
        // Should be morpheme-level, not per-kana — at most ~5 tokens.
        XCTAssertLessThan(words.count, 7,
            "Japanese+Latin should be morpheme-level, got \(words.count): \(words)")
    }

    // MARK: - Korean (NLTokenizer, matches upstream soynlp)

    /// Korean must NOT split per Hangul syllable. Upstream uses soynlp
    /// LTokenizer; we use Apple's NLTokenizer for native word segmentation.
    func testTextPreprocessingKorean() {
        let words = TextPreprocessor.splitIntoWords("안녕하세요 반갑습니다", language: "korean")
        XCTAssertGreaterThan(words.count, 0)
        // Must NOT be 11 per-syllable tokens.
        XCTAssertLessThan(words.count, 6,
            "Korean should be word/morpheme level, not per-syllable. Got \(words.count): \(words)")
        XCTAssertFalse(words.contains(" "))
    }

    func testTimestampCorrectionAlreadyMonotonic() {
        let input = [1, 3, 5, 7, 9, 11]
        let corrected = TimestampCorrection.enforceMonotonicity(input)
        XCTAssertEqual(corrected, input)
    }

    func testTimestampCorrectionSingleOutOfOrder() {
        let input = [1, 3, 2, 7, 9, 11]
        let corrected = TimestampCorrection.enforceMonotonicity(input)

        // Verify monotonicity
        for i in 1..<corrected.count {
            XCTAssertGreaterThanOrEqual(corrected[i], corrected[i - 1],
                "Index \(i): \(corrected[i]) should be >= \(corrected[i-1])")
        }
    }

    func testTimestampCorrectionAllSame() {
        let input = [5, 5, 5, 5]
        let corrected = TimestampCorrection.enforceMonotonicity(input)
        // Should remain all 5 (monotonically non-decreasing)
        XCTAssertEqual(corrected, [5, 5, 5, 5])
    }

    func testTimestampCorrectionDescending() {
        let input = [10, 8, 6, 4, 2]
        let corrected = TimestampCorrection.enforceMonotonicity(input)

        // After LIS + correction, should be monotonically non-decreasing
        for i in 1..<corrected.count {
            XCTAssertGreaterThanOrEqual(corrected[i], corrected[i - 1])
        }
    }

    func testLISBasic() {
        let arr = [3, 1, 4, 1, 5, 9, 2, 6]
        let positions = TimestampCorrection.longestIncreasingSubsequencePositions(arr)

        // LIS should be length 4 or 5 (e.g., [1, 4, 5, 9] or [1, 4, 5, 6])
        XCTAssertGreaterThanOrEqual(positions.count, 4)

        // Values at LIS positions should be strictly increasing
        for i in 1..<positions.count {
            XCTAssertLessThan(arr[positions[i - 1]], arr[positions[i]])
        }
    }

    // MARK: - E2E Integration Test (requires model download)

    func testForcedAlignerE2E() async throws {
        guard let wavURL = Bundle.module.url(forResource: "test_audio", withExtension: "wav") else {
            throw XCTSkip("Test WAV file not found in bundle resources")
        }

        print("Loading Forced Aligner model...")
        let aligner = try await Qwen3ForcedAligner.fromPretrained(
            modelId: "aufklarer/Qwen3-ForcedAligner-0.6B-4bit"
        ) { progress, status in
            print("[\(Int(progress * 100))%] \(status)")
        }

        // Load audio
        let (samples, sampleRate) = try AudioFileLoader.loadWAV(url: wavURL)
        let targetSampleRate = 24000
        let audio: [Float]
        if sampleRate != targetSampleRate {
            audio = AudioFileLoader.resample(samples, from: sampleRate, to: targetSampleRate)
        } else {
            audio = samples
        }

        let knownText = "Can you guarantee that the replacement part will be shipped tomorrow?"

        print("Aligning...")
        let start = Date()
        let aligned = aligner.align(audio: audio, text: knownText, sampleRate: targetSampleRate)
        let elapsed = Date().timeIntervalSince(start)

        print("Alignment results:")
        for word in aligned {
            print("  [\(String(format: "%.2f", word.startTime))s - \(String(format: "%.2f", word.endTime))s] \(word.text)")
        }
        print("Alignment took \(String(format: "%.2f", elapsed))s")

        // Verify we got words
        XCTAssertFalse(aligned.isEmpty, "Should produce aligned words")

        // Expected word count (splitting "Can you guarantee that the replacement part will be shipped tomorrow?")
        let expectedWords = knownText.components(separatedBy: .whitespaces).filter { !$0.isEmpty }
        XCTAssertEqual(aligned.count, expectedWords.count, "Word count should match input")

        // Verify monotonicity: each word's start <= end, and next word starts >= previous end
        for (i, word) in aligned.enumerated() {
            XCTAssertGreaterThanOrEqual(word.endTime, word.startTime,
                "Word '\(word.text)' end should be >= start")

            if i > 0 {
                XCTAssertGreaterThanOrEqual(word.startTime, aligned[i - 1].startTime,
                    "Word '\(word.text)' start should be >= previous word start")
            }
        }

        // Verify total time is reasonable
        let audioDuration = Float(audio.count) / Float(targetSampleRate)
        if let lastWord = aligned.last {
            XCTAssertLessThanOrEqual(lastWord.endTime, audioDuration + 1.0,
                "Last word end should be within audio duration")
        }
        if let firstWord = aligned.first {
            XCTAssertGreaterThanOrEqual(firstWord.startTime, 0,
                "First word should start at >= 0")
        }
    }

    // MARK: - bf16 variant (exercises FloatTextDecoder / FloatTextAttention)

    /// Same alignment flow as `testForcedAlignerE2E` but using the
    /// non-quantised bf16 forced-aligner checkpoint. This is the only code
    /// path that loads `FloatTextDecoder` / `FloatTextAttention` — the 4-bit
    /// and 8-bit variants both go through `QuantizedTextAttention`. Without
    /// this test, the float attention refactor (`SDPA.attendAndMerge`) is
    /// only compile-checked.
    func testForcedAlignerE2EBf16Variant() async throws {
        guard let wavURL = Bundle.module.url(forResource: "test_audio", withExtension: "wav") else {
            throw XCTSkip("Test WAV file not found in bundle resources")
        }

        print("Loading Forced Aligner (bf16 variant)...")
        let aligner = try await Qwen3ForcedAligner.fromPretrained(
            modelId: "aufklarer/Qwen3-ForcedAligner-0.6B-bf16"
        ) { progress, status in
            print("[\(Int(progress * 100))%] \(status)")
        }

        // Sanity-check the text decoder is the float variant.
        XCTAssertTrue(aligner.textDecoder is FloatTextModel,
                      "bf16 model should load FloatTextModel (exercises FloatTextAttention)")

        let (samples, sampleRate) = try AudioFileLoader.loadWAV(url: wavURL)
        let targetSampleRate = 24000
        let audio: [Float]
        if sampleRate != targetSampleRate {
            audio = AudioFileLoader.resample(samples, from: sampleRate, to: targetSampleRate)
        } else {
            audio = samples
        }

        let knownText = "Can you guarantee that the replacement part will be shipped tomorrow?"
        let aligned = aligner.align(audio: audio, text: knownText, sampleRate: targetSampleRate)

        XCTAssertFalse(aligned.isEmpty, "bf16 aligner should produce aligned words")
        let expectedWords = knownText.components(separatedBy: .whitespaces).filter { !$0.isEmpty }
        XCTAssertEqual(aligned.count, expectedWords.count,
                       "bf16 word count should match input")

        for (i, word) in aligned.enumerated() {
            XCTAssertGreaterThanOrEqual(word.endTime, word.startTime)
            if i > 0 {
                XCTAssertGreaterThanOrEqual(word.startTime, aligned[i - 1].startTime)
            }
        }
    }

    // MARK: - Latency Benchmark

    func testForcedAlignerLatency() async throws {
        guard let wavURL = Bundle.module.url(forResource: "test_audio", withExtension: "wav") else {
            throw XCTSkip("Test WAV file not found in bundle resources")
        }

        let aligner = try await Qwen3ForcedAligner.fromPretrained(
            modelId: "aufklarer/Qwen3-ForcedAligner-0.6B-4bit"
        )

        let (samples, sampleRate) = try AudioFileLoader.loadWAV(url: wavURL)
        let targetSR = 24000
        let audio: [Float]
        if sampleRate != targetSR {
            audio = AudioFileLoader.resample(samples, from: sampleRate, to: targetSR)
        } else {
            audio = samples
        }
        let audioDuration = Double(audio.count) / Double(targetSR)
        let text = "Can you guarantee that the replacement part will be shipped tomorrow?"

        print("Audio: \(String(format: "%.1f", audioDuration))s (\(audio.count) samples at \(targetSR)Hz)")

        // Warmup run
        _ = aligner.align(audio: Array(audio.prefix(targetSR)), text: "warmup test", sampleRate: targetSR)

        // --- Stage 1: Mel + Audio Encoder ---
        let t1 = Date()
        let mel = aligner.featureExtractor.process(audio, sampleRate: targetSR)
        let batchedMel = mel.expandedDimensions(axis: 0)
        var audioEmbeds = aligner.audioEncoder(batchedMel)
        audioEmbeds = audioEmbeds.expandedDimensions(axis: 0)
        eval(audioEmbeds)
        let encoderMs = Date().timeIntervalSince(t1) * 1000

        print("Audio encoder: \(String(format: "%.0f", encoderMs))ms (mel + 24L transformer + projector)")
        print("  Audio tokens: \(audioEmbeds.dim(1))")

        // --- Stage 2: Full alignment (3 runs) ---
        var times: [Double] = []
        for run in 1...3 {
            let t = Date()
            let aligned = aligner.align(audio: audio, text: text, sampleRate: targetSR)
            eval(MLXArray(0))
            let ms = Date().timeIntervalSince(t) * 1000
            times.append(ms)
            print("Run \(run): \(String(format: "%.0f", ms))ms (\(aligned.count) words)")
        }

        let avgMs = times.reduce(0, +) / Double(times.count)
        let bestMs = times.min()!
        print("\nSummary (debug build):")
        print("  Encoder: \(String(format: "%.0f", encoderMs))ms")
        print("  Full align avg: \(String(format: "%.0f", avgMs))ms (best: \(String(format: "%.0f", bestMs))ms)")
        print("  Audio duration: \(String(format: "%.1f", audioDuration))s")
        print("  RTF: \(String(format: "%.3f", bestMs / 1000.0 / audioDuration))")
    }
}
