import XCTest
@testable import Qwen3ASR
@testable import SpeechVAD
@testable import AudioCommon

final class StreamingASRTests: XCTestCase {

    // MARK: - LongestCommonPrefix Tests

    func testLCPEmptyArrays() {
        XCTAssertEqual(longestCommonPrefix([], []), [])
        XCTAssertEqual(longestCommonPrefix(["hello"], []), [])
        XCTAssertEqual(longestCommonPrefix([], ["hello"]), [])
    }

    func testLCPFullMatch() {
        let a = ["can", "you", "guarantee"]
        let b = ["can", "you", "guarantee"]
        XCTAssertEqual(longestCommonPrefix(a, b), ["can", "you", "guarantee"])
    }

    func testLCPPartialMatch() {
        let a = ["can", "you", "guarantee"]
        let b = ["can", "you", "help"]
        XCTAssertEqual(longestCommonPrefix(a, b), ["can", "you"])
    }

    func testLCPNoMatch() {
        let a = ["hello"]
        let b = ["goodbye"]
        XCTAssertEqual(longestCommonPrefix(a, b), [])
    }

    func testLCPCaseInsensitive() {
        let a = ["Can", "You"]
        let b = ["can", "you", "help"]
        let result = longestCommonPrefix(a, b)
        XCTAssertEqual(result.count, 2)
        // Returns elements from b
        XCTAssertEqual(result, ["can", "you"])
    }

    func testLCPDifferentLengths() {
        let a = ["the", "quick", "brown", "fox"]
        let b = ["the", "quick"]
        XCTAssertEqual(longestCommonPrefix(a, b), ["the", "quick"])
    }

    // MARK: - StreamingASRConfig Tests

    func testDefaultConfig() {
        let config = StreamingASRConfig.default
        XCTAssertEqual(config.maxSegmentDuration, 10.0)
        XCTAssertEqual(config.maxTokens, 448)
        XCTAssertFalse(config.emitPartialResults)
        XCTAssertEqual(config.partialResultInterval, 1.0)
        XCTAssertNil(config.language)
    }

    func testCustomConfig() {
        let config = StreamingASRConfig(
            maxSegmentDuration: 15.0,
            language: "en",
            maxTokens: 256,
            emitPartialResults: true,
            partialResultInterval: 0.5
        )
        XCTAssertEqual(config.maxSegmentDuration, 15.0)
        XCTAssertEqual(config.language, "en")
        XCTAssertEqual(config.maxTokens, 256)
        XCTAssertTrue(config.emitPartialResults)
        XCTAssertEqual(config.partialResultInterval, 0.5)
    }

    // MARK: - TranscriptionSegment Tests

    func testTranscriptionSegment() {
        let segment = TranscriptionSegment(
            text: "hello world",
            startTime: 1.0,
            endTime: 2.5,
            isFinal: true,
            segmentIndex: 0
        )
        XCTAssertEqual(segment.text, "hello world")
        XCTAssertEqual(segment.startTime, 1.0)
        XCTAssertEqual(segment.endTime, 2.5)
        XCTAssertTrue(segment.isFinal)
        XCTAssertEqual(segment.segmentIndex, 0)
    }

    func testPartialSegment() {
        let segment = TranscriptionSegment(
            text: "hello",
            startTime: 1.0,
            endTime: 1.5,
            isFinal: false,
            segmentIndex: 0
        )
        XCTAssertFalse(segment.isFinal)
    }

    // MARK: - Integration Tests (require model downloads)

    func testStreamingTranscription() async throws {
        guard let wavURL = Bundle.module.url(forResource: "test_audio", withExtension: "wav") else {
            throw XCTSkip("Test WAV file not found in bundle resources")
        }

        let (samples, sampleRate) = try AudioFileLoader.loadWAV(url: wavURL)
        let audio = AudioFileLoader.resample(samples, from: sampleRate, to: 16000)
        print("Loaded audio: \(audio.count) samples (\(String(format: "%.2f", Float(audio.count) / 16000))s)")

        let streaming = try await StreamingASR.fromPretrained { progress, status in
            print("[\(Int(progress * 100))%] \(status)")
        }

        var segments: [TranscriptionSegment] = []
        let stream = streaming.transcribeStream(audio: audio, sampleRate: 16000)
        for try await segment in stream {
            let tag = segment.isFinal ? "FINAL" : "partial"
            print("[\(String(format: "%.2f", segment.startTime))s-\(String(format: "%.2f", segment.endTime))s] [\(tag)] \(segment.text)")
            segments.append(segment)
        }

        // Should produce at least one final segment
        let finalSegments = segments.filter { $0.isFinal }
        XCTAssertGreaterThan(finalSegments.count, 0, "Should have at least one final segment")

        // Verify transcription content
        let fullText = finalSegments.map { $0.text }.joined(separator: " ")
        print("Full transcription: \(fullText)")
        XCTAssertTrue(fullText.contains("guarantee"), "Should transcribe 'guarantee'")
        XCTAssertTrue(fullText.contains("replacement"), "Should transcribe 'replacement'")
        XCTAssertTrue(fullText.contains("shipped"), "Should transcribe 'shipped'")
        XCTAssertTrue(fullText.contains("tomorrow"), "Should transcribe 'tomorrow'")

        // Verify timestamps are reasonable
        for segment in finalSegments {
            XCTAssertGreaterThanOrEqual(segment.startTime, 0)
            XCTAssertGreaterThan(segment.endTime, segment.startTime)
        }
    }

    func testStreamingWithPartialResults() async throws {
        guard let wavURL = Bundle.module.url(forResource: "test_audio", withExtension: "wav") else {
            throw XCTSkip("Test WAV file not found in bundle resources")
        }

        let (samples, sampleRate) = try AudioFileLoader.loadWAV(url: wavURL)
        let audio = AudioFileLoader.resample(samples, from: sampleRate, to: 16000)

        let streaming = try await StreamingASR.fromPretrained { progress, status in
            print("[\(Int(progress * 100))%] \(status)")
        }

        let config = StreamingASRConfig(
            emitPartialResults: true,
            partialResultInterval: 0.5
        )

        var segments: [TranscriptionSegment] = []
        let stream = streaming.transcribeStream(audio: audio, sampleRate: 16000, config: config)
        for try await segment in stream {
            let tag = segment.isFinal ? "FINAL" : "partial"
            print("[\(String(format: "%.2f", segment.startTime))s-\(String(format: "%.2f", segment.endTime))s] [\(tag)] \(segment.text)")
            segments.append(segment)
        }

        let partials = segments.filter { !$0.isFinal }
        let finals = segments.filter { $0.isFinal }

        // With partial results, we should see some tentative segments before the final
        print("Partial segments: \(partials.count), Final segments: \(finals.count)")
        XCTAssertGreaterThan(finals.count, 0, "Should have at least one final segment")

        // Final segment should contain the full transcription
        let fullText = finals.map { $0.text }.joined(separator: " ")
        XCTAssertTrue(fullText.contains("guarantee"), "Final should contain 'guarantee'")
    }
}
