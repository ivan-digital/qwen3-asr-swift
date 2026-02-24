import XCTest
import Foundation
import MLX
@testable import Qwen3ASR
@testable import Qwen3Common

/// Tests for the Forced Aligner
final class ForcedAlignerTests: XCTestCase {

    // MARK: - Unit Tests (no model download)

    func testTextPreprocessingEnglish() {
        // Test the word splitting logic
        let words = TextPreprocessor.splitIntoWords("Hello world test", language: "English")
        XCTAssertEqual(words, ["Hello", "world", "test"])
    }

    func testTextPreprocessingCJK() {
        let words = TextPreprocessor.splitIntoWords("你好世界", language: "Chinese")
        XCTAssertEqual(words.count, 4)
        XCTAssertEqual(words[0], "你")
        XCTAssertEqual(words[1], "好")
        XCTAssertEqual(words[2], "世")
        XCTAssertEqual(words[3], "界")
    }

    func testTextPreprocessingMixedCJK() {
        let words = TextPreprocessor.splitIntoWords("Hello你好world", language: "Chinese")
        XCTAssertEqual(words.count, 4)
        XCTAssertEqual(words[0], "Hello")
        XCTAssertEqual(words[1], "你")
        XCTAssertEqual(words[2], "好")
        XCTAssertEqual(words[3], "world")
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
            audio = AudioFileLoader.resampleForTest(samples, from: sampleRate, to: targetSampleRate)
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
            audio = AudioFileLoader.resampleForTest(samples, from: sampleRate, to: targetSR)
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
