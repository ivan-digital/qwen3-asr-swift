import XCTest
@testable import SpeechVAD
import AudioCommon

final class DERScoringTests: XCTestCase {

    // MARK: - RTTM Format

    func testRTTMFormat() {
        let segments = [
            DiarizedSegment(startTime: 1.5, endTime: 3.2, speakerId: 0),
            DiarizedSegment(startTime: 4.0, endTime: 6.5, speakerId: 1),
        ]
        let rttm = toRTTM(segments: segments, filename: "test")

        XCTAssertEqual(rttm.count, 2)
        XCTAssertEqual(rttm[0].rttmLine, "SPEAKER test 1 1.500 1.700 <NA> <NA> speaker_0 <NA> <NA>")
        XCTAssertEqual(rttm[1].rttmLine, "SPEAKER test 1 4.000 2.500 <NA> <NA> speaker_1 <NA> <NA>")
    }

    func testRTTMRoundTrip() {
        let original = [
            DiarizedSegment(startTime: 0.5, endTime: 2.0, speakerId: 0),
            DiarizedSegment(startTime: 3.0, endTime: 5.0, speakerId: 1),
            DiarizedSegment(startTime: 5.5, endTime: 8.0, speakerId: 0),
        ]

        let rttmSegments = toRTTM(segments: original, filename: "roundtrip")
        let rttmText = formatRTTM(rttmSegments)
        let parsed = parseRTTM(rttmText)

        XCTAssertEqual(parsed.count, 3)
        XCTAssertEqual(parsed[0].startTime, 0.5, accuracy: 0.01)
        XCTAssertEqual(parsed[0].endTime, 2.0, accuracy: 0.01)
        XCTAssertEqual(parsed[1].speakerId, 1)
        XCTAssertEqual(parsed[2].endTime, 8.0, accuracy: 0.01)
    }

    func testParseRTTMWithExtraWhitespace() {
        let rttm = """
        SPEAKER  file1  1  1.000  2.000  <NA>  <NA>  spk_A  <NA>  <NA>
        SPEAKER  file1  1  4.000  1.500  <NA>  <NA>  spk_B  <NA>  <NA>
        """
        let segments = parseRTTM(rttm)

        XCTAssertEqual(segments.count, 2)
        XCTAssertEqual(segments[0].startTime, 1.0, accuracy: 0.01)
        XCTAssertEqual(segments[0].duration, 2.0, accuracy: 0.01)
        XCTAssertEqual(segments[1].speakerId, 1) // spk_B mapped to 1
    }

    // MARK: - DER: Perfect Match

    func testDERPerfectMatch() {
        let segments = [
            DiarizedSegment(startTime: 1.0, endTime: 3.0, speakerId: 0),
            DiarizedSegment(startTime: 4.0, endTime: 6.0, speakerId: 1),
        ]

        let result = computeDER(
            reference: segments, hypothesis: segments,
            collar: 0, resolution: 0.01
        )

        XCTAssertEqual(result.der, 0, accuracy: 0.01)
        XCTAssertEqual(result.totalSpeech, 4.0, accuracy: 0.1)
    }

    // MARK: - DER: Complete Miss

    func testDERCompleteMiss() {
        let ref = [DiarizedSegment(startTime: 1.0, endTime: 3.0, speakerId: 0)]
        let hyp: [DiarizedSegment] = []

        let result = computeDER(reference: ref, hypothesis: hyp, collar: 0, resolution: 0.01)

        XCTAssertEqual(result.missedSpeech, 2.0, accuracy: 0.1)
        XCTAssertEqual(result.falseAlarm, 0, accuracy: 0.01)
        XCTAssertEqual(result.der, 1.0, accuracy: 0.01) // 100% missed
    }

    // MARK: - DER: Complete False Alarm

    func testDERCompleteFalseAlarm() {
        let ref: [DiarizedSegment] = []
        let hyp = [DiarizedSegment(startTime: 1.0, endTime: 3.0, speakerId: 0)]

        let result = computeDER(reference: ref, hypothesis: hyp, collar: 0, resolution: 0.01)

        XCTAssertEqual(result.totalSpeech, 0, accuracy: 0.01)
        XCTAssertEqual(result.falseAlarm, 2.0, accuracy: 0.1)
    }

    // MARK: - DER: Collar Forgiveness

    func testDERCollarForgiveness() {
        // Hypothesis shifted by 0.2s from reference — within 0.25s collar
        let ref = [DiarizedSegment(startTime: 1.0, endTime: 3.0, speakerId: 0)]
        let hyp = [DiarizedSegment(startTime: 1.2, endTime: 3.2, speakerId: 0)]

        let withCollar = computeDER(reference: ref, hypothesis: hyp,
                                    collar: 0.25, resolution: 0.01)
        let withoutCollar = computeDER(reference: ref, hypothesis: hyp,
                                       collar: 0, resolution: 0.01)

        // With collar, boundary errors are forgiven
        XCTAssertLessThan(withCollar.der, withoutCollar.der)
    }

    // MARK: - DER: Speaker Confusion

    func testDERSpeakerConfusion() {
        // Same timing, wrong speaker ID
        let ref = [DiarizedSegment(startTime: 1.0, endTime: 3.0, speakerId: 0)]
        let hyp = [DiarizedSegment(startTime: 1.0, endTime: 3.0, speakerId: 1)]

        let result = computeDER(reference: ref, hypothesis: hyp, collar: 0, resolution: 0.01)

        // All frames are confusion (ref and hyp both have 1 speaker, but different IDs)
        XCTAssertEqual(result.confusion, 2.0, accuracy: 0.1)
        XCTAssertEqual(result.der, 1.0, accuracy: 0.01)
    }

    // MARK: - DER: Optimal Mapping

    func testDEROptimalMappingFixesSwappedIDs() {
        // Reference: speaker 0, then speaker 1
        // Hypothesis: same segments but IDs swapped (speaker 1, then speaker 0)
        let ref = [
            DiarizedSegment(startTime: 1.0, endTime: 3.0, speakerId: 0),
            DiarizedSegment(startTime: 4.0, endTime: 6.0, speakerId: 1),
        ]
        let hyp = [
            DiarizedSegment(startTime: 1.0, endTime: 3.0, speakerId: 1),
            DiarizedSegment(startTime: 4.0, endTime: 6.0, speakerId: 0),
        ]

        // Without optimal mapping: 100% confusion
        let naive = computeDER(reference: ref, hypothesis: hyp, collar: 0, resolution: 0.01)
        XCTAssertGreaterThan(naive.confusion, 3.0)

        // With optimal mapping: 0% DER
        let optimal = computeDERWithOptimalMapping(
            reference: ref, hypothesis: hyp, collar: 0, resolution: 0.01
        )
        XCTAssertEqual(optimal.der, 0, accuracy: 0.01)
    }

    // MARK: - DER: Partial Overlap

    func testDERPartialOverlap() {
        // Reference: [1.0, 5.0]
        // Hypothesis: [2.0, 6.0] — misses first 1s, adds 1s false alarm
        let ref = [DiarizedSegment(startTime: 1.0, endTime: 5.0, speakerId: 0)]
        let hyp = [DiarizedSegment(startTime: 2.0, endTime: 6.0, speakerId: 0)]

        let result = computeDER(reference: ref, hypothesis: hyp, collar: 0, resolution: 0.01)

        XCTAssertEqual(result.totalSpeech, 4.0, accuracy: 0.1)
        XCTAssertEqual(result.missedSpeech, 1.0, accuracy: 0.1) // [1.0-2.0]
        XCTAssertEqual(result.falseAlarm, 1.0, accuracy: 0.1)  // [5.0-6.0]
        XCTAssertEqual(result.confusion, 0, accuracy: 0.1)
    }
}
