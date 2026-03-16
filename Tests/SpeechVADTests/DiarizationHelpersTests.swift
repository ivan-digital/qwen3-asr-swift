import XCTest
@testable import SpeechVAD
import AudioCommon

final class DiarizationHelpersTests: XCTestCase {

    // MARK: - mergeSegments

    func testMergeSegmentsEmpty() {
        let result = DiarizationHelpers.mergeSegments([], minSilence: 0.15)
        XCTAssertTrue(result.isEmpty)
    }

    func testMergeSegmentsSingleSegment() {
        let segs = [DiarizedSegment(startTime: 1.0, endTime: 2.0, speakerId: 0)]
        let result = DiarizationHelpers.mergeSegments(segs, minSilence: 0.15)
        XCTAssertEqual(result.count, 1)
        XCTAssertEqual(result[0].startTime, 1.0, accuracy: 0.001)
        XCTAssertEqual(result[0].endTime, 2.0, accuracy: 0.001)
    }

    func testMergeSegmentsAdjacentSameSpeaker() {
        // Two segments from same speaker with small gap → should merge
        let segs = [
            DiarizedSegment(startTime: 1.0, endTime: 2.0, speakerId: 0),
            DiarizedSegment(startTime: 2.1, endTime: 3.0, speakerId: 0),
        ]
        let result = DiarizationHelpers.mergeSegments(segs, minSilence: 0.15)
        XCTAssertEqual(result.count, 1)
        XCTAssertEqual(result[0].startTime, 1.0, accuracy: 0.001)
        XCTAssertEqual(result[0].endTime, 3.0, accuracy: 0.001)
    }

    func testMergeSegmentsLargeGapNotMerged() {
        // Two segments from same speaker with large gap → should NOT merge
        let segs = [
            DiarizedSegment(startTime: 1.0, endTime: 2.0, speakerId: 0),
            DiarizedSegment(startTime: 3.0, endTime: 4.0, speakerId: 0),
        ]
        let result = DiarizationHelpers.mergeSegments(segs, minSilence: 0.15)
        XCTAssertEqual(result.count, 2)
    }

    func testMergeSegmentsDifferentSpeakers() {
        // Adjacent segments from different speakers → should NOT merge
        let segs = [
            DiarizedSegment(startTime: 1.0, endTime: 2.0, speakerId: 0),
            DiarizedSegment(startTime: 2.05, endTime: 3.0, speakerId: 1),
        ]
        let result = DiarizationHelpers.mergeSegments(segs, minSilence: 0.15)
        XCTAssertEqual(result.count, 2)
    }

    func testMergeSegmentsMultipleSpeakersInterleaved() {
        // Interleaved speakers with small gaps within each speaker
        let segs = [
            DiarizedSegment(startTime: 0.0, endTime: 1.0, speakerId: 0),
            DiarizedSegment(startTime: 1.0, endTime: 2.0, speakerId: 1),
            DiarizedSegment(startTime: 2.0, endTime: 3.0, speakerId: 0),
            DiarizedSegment(startTime: 3.0, endTime: 4.0, speakerId: 1),
        ]
        let result = DiarizationHelpers.mergeSegments(segs, minSilence: 0.15)
        // Speaker 0 has gap of 1.0s between segments → not merged
        // Speaker 1 has gap of 1.0s between segments → not merged
        XCTAssertEqual(result.count, 4)
        // Should be sorted by start time
        for i in 1..<result.count {
            XCTAssertGreaterThanOrEqual(result[i].startTime, result[i-1].startTime)
        }
    }

    // MARK: - compactSpeakerIds

    func testCompactSpeakerIdsEmpty() {
        let result = DiarizationHelpers.compactSpeakerIds([])
        XCTAssertTrue(result.isEmpty)
    }

    func testCompactSpeakerIdsAlreadyContiguous() {
        let segs = [
            DiarizedSegment(startTime: 0, endTime: 1, speakerId: 0),
            DiarizedSegment(startTime: 1, endTime: 2, speakerId: 1),
        ]
        let result = DiarizationHelpers.compactSpeakerIds(segs)
        XCTAssertEqual(result[0].speakerId, 0)
        XCTAssertEqual(result[1].speakerId, 1)
    }

    func testCompactSpeakerIdsWithGaps() {
        // Speaker IDs 2, 5 → should become 0, 1
        let segs = [
            DiarizedSegment(startTime: 0, endTime: 1, speakerId: 2),
            DiarizedSegment(startTime: 1, endTime: 2, speakerId: 5),
            DiarizedSegment(startTime: 2, endTime: 3, speakerId: 2),
        ]
        let result = DiarizationHelpers.compactSpeakerIds(segs)
        XCTAssertEqual(result[0].speakerId, 0)
        XCTAssertEqual(result[1].speakerId, 1)
        XCTAssertEqual(result[2].speakerId, 0)
    }

    func testCompactSpeakerIdsPreservesTimestamps() {
        let segs = [
            DiarizedSegment(startTime: 1.5, endTime: 3.7, speakerId: 10),
        ]
        let result = DiarizationHelpers.compactSpeakerIds(segs)
        XCTAssertEqual(result[0].startTime, 1.5, accuracy: 0.001)
        XCTAssertEqual(result[0].endTime, 3.7, accuracy: 0.001)
        XCTAssertEqual(result[0].speakerId, 0)
    }

    // MARK: - resample

    func testResampleSameRate() {
        let audio: [Float] = [1.0, 2.0, 3.0, 4.0]
        let result = DiarizationHelpers.resample(audio, from: 16000, to: 16000)
        XCTAssertEqual(result, audio)
    }

    func testResampleUpsample() {
        // Use longer signal for AVAudioConverter (short signals may have different frame counts)
        let audio: [Float] = (0..<100).map { Float($0) / 100.0 }
        let result = DiarizationHelpers.resample(audio, from: 8000, to: 16000)
        // 100 samples at 8kHz → ~200 samples at 16kHz
        XCTAssertGreaterThan(result.count, 150)
        XCTAssertLessThan(result.count, 250)
    }

    func testResampleDownsample() {
        let audio: [Float] = (0..<200).map { Float($0) / 200.0 }
        let result = DiarizationHelpers.resample(audio, from: 16000, to: 8000)
        // 200 samples at 16kHz → ~100 samples at 8kHz
        XCTAssertGreaterThan(result.count, 75)
        XCTAssertLessThan(result.count, 125)
    }

    func testResampleEmpty() {
        let result = DiarizationHelpers.resample([], from: 16000, to: 8000)
        XCTAssertTrue(result.isEmpty)
    }

    // MARK: - DiarizationConfig

    func testDiarizationConfigDefaultValues() {
        let config = DiarizationConfig.default
        XCTAssertEqual(config.onset, 0.5, accuracy: 0.001)
        XCTAssertEqual(config.offset, 0.3, accuracy: 0.001)
        XCTAssertEqual(config.minSpeechDuration, 0.3, accuracy: 0.001)
        XCTAssertEqual(config.minSilenceDuration, 0.15, accuracy: 0.001)
    }

    func testDiarizationConfigCustomValues() {
        let config = DiarizationConfig(
            onset: 0.7, offset: 0.4,
            minSpeechDuration: 0.5, minSilenceDuration: 0.2
        )
        XCTAssertEqual(config.onset, 0.7, accuracy: 0.001)
        XCTAssertEqual(config.offset, 0.4, accuracy: 0.001)
        XCTAssertEqual(config.minSpeechDuration, 0.5, accuracy: 0.001)
        XCTAssertEqual(config.minSilenceDuration, 0.2, accuracy: 0.001)
    }

    // MARK: - DiarizationResult

    func testDiarizationResultEmpty() {
        let result = DiarizationResult(segments: [], numSpeakers: 0, speakerEmbeddings: [])
        XCTAssertTrue(result.segments.isEmpty)
        XCTAssertEqual(result.numSpeakers, 0)
        XCTAssertTrue(result.speakerEmbeddings.isEmpty)
    }

    // MARK: - Protocol Conformance

    func testPyannoteDiarizationPipelineTypealias() {
        // Verify the typealias compiles — DiarizationPipeline should be PyannoteDiarizationPipeline
        let _: DiarizationPipeline.Type = PyannoteDiarizationPipeline.self
    }

    func testSpeakerExtractionCapableConformance() {
        // PyannoteDiarizationPipeline conforms to SpeakerExtractionCapable
        XCTAssertTrue((PyannoteDiarizationPipeline.self as Any) is SpeakerExtractionCapable.Type)
    }

    func testSpeakerDiarizationModelConformance() {
        // PyannoteDiarizationPipeline conforms to SpeakerDiarizationModel
        XCTAssertTrue((PyannoteDiarizationPipeline.self as Any) is SpeakerDiarizationModel.Type)
    }

    #if canImport(CoreML)
    func testSortformerDiarizationModelConformance() {
        // SortformerDiarizer conforms to SpeakerDiarizationModel but NOT SpeakerExtractionCapable
        XCTAssertTrue((SortformerDiarizer.self as Any) is SpeakerDiarizationModel.Type)
        XCTAssertFalse((SortformerDiarizer.self as Any) is SpeakerExtractionCapable.Type)
    }
    #endif
}
