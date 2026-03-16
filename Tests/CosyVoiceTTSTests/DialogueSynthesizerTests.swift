import XCTest
@testable import CosyVoiceTTS

final class DialogueSynthesizerTests: XCTestCase {

    // MARK: - Silence gap

    func testSilenceGapSampleCount() {
        let gap = DialogueSynthesizer.silenceGap(seconds: 0.2, sampleRate: 24000)
        XCTAssertEqual(gap.count, 4800) // 0.2 * 24000
        XCTAssertTrue(gap.allSatisfy { $0 == 0 })
    }

    func testSilenceGapZero() {
        let gap = DialogueSynthesizer.silenceGap(seconds: 0, sampleRate: 24000)
        XCTAssertEqual(gap.count, 0)
    }

    func testSilenceGapNegative() {
        let gap = DialogueSynthesizer.silenceGap(seconds: -1.0, sampleRate: 24000)
        XCTAssertEqual(gap.count, 0)
    }

    func testSilenceGapDifferentSampleRate() {
        let gap = DialogueSynthesizer.silenceGap(seconds: 1.0, sampleRate: 16000)
        XCTAssertEqual(gap.count, 16000)
    }

    // MARK: - Crossfade

    func testCrossfadeBasic() {
        // Left: [1, 1, 1, 1], Right: [0, 0, 0, 0], overlap=2
        let left: [Float] = [1, 1, 1, 1]
        let right: [Float] = [0, 0, 0, 0]
        let result = DialogueSynthesizer.crossfade(left: left, right: right, overlapSamples: 2)

        // Expected: [1, 1, blend(1→0), blend(1→0), 0, 0]
        // blend at i=0: 1*(1-0) + 0*0 = 1.0
        // blend at i=1: 1*(1-0.5) + 0*0.5 = 0.5
        XCTAssertEqual(result.count, 6)
        XCTAssertEqual(result[0], 1.0)
        XCTAssertEqual(result[1], 1.0)
        XCTAssertEqual(result[2], 1.0, accuracy: 0.001) // 1*(1-0/2) + 0*(0/2)
        XCTAssertEqual(result[3], 0.5, accuracy: 0.001) // 1*(1-1/2) + 0*(1/2)
        XCTAssertEqual(result[4], 0.0)
        XCTAssertEqual(result[5], 0.0)
    }

    func testCrossfadeZeroOverlap() {
        let left: [Float] = [1, 2, 3]
        let right: [Float] = [4, 5, 6]
        let result = DialogueSynthesizer.crossfade(left: left, right: right, overlapSamples: 0)

        // No crossfade — just concatenation
        XCTAssertEqual(result, [1, 2, 3, 4, 5, 6])
    }

    func testCrossfadeClampedToBufferLength() {
        let left: [Float] = [1, 2]
        let right: [Float] = [3, 4]
        // Overlap > both buffers → clamped to 2
        let result = DialogueSynthesizer.crossfade(left: left, right: right, overlapSamples: 100)

        // Full overlap: blend [1,2] with [3,4]
        XCTAssertEqual(result.count, 2)
        XCTAssertEqual(result[0], 1.0 * 1.0 + 3.0 * 0.0, accuracy: 0.001) // t=0/2
        XCTAssertEqual(result[1], 2.0 * 0.5 + 4.0 * 0.5, accuracy: 0.001) // t=1/2
    }

    func testCrossfadePreservesLength() {
        let left = [Float](repeating: 1.0, count: 100)
        let right = [Float](repeating: 0.0, count: 100)
        let result = DialogueSynthesizer.crossfade(left: left, right: right, overlapSamples: 20)

        // Total = 100 + 100 - 20 = 180
        XCTAssertEqual(result.count, 180)
    }

    func testCrossfadeSymmetry() {
        // At the midpoint of the crossfade, both signals should contribute equally
        let left: [Float] = [1, 1, 1, 1]
        let right: [Float] = [0, 0, 0, 0]
        let result = DialogueSynthesizer.crossfade(left: left, right: right, overlapSamples: 4)

        // Midpoint (i=2 of 4): t=0.5, blend = 1*0.5 + 0*0.5 = 0.5
        XCTAssertEqual(result[2], 0.5, accuracy: 0.001)
    }

    // MARK: - Config defaults

    func testConfigDefaults() {
        let config = DialogueSynthesisConfig()
        XCTAssertEqual(config.turnGapSeconds, 0.2)
        XCTAssertEqual(config.crossfadeSeconds, 0.0)
        XCTAssertEqual(config.defaultInstruction, "You are a helpful assistant.")
        XCTAssertEqual(config.maxTokensPerSegment, 500)
    }

    func testConfigCustomValues() {
        let config = DialogueSynthesisConfig(
            turnGapSeconds: 0.5,
            crossfadeSeconds: 0.1,
            defaultInstruction: "Speak cheerfully.",
            maxTokensPerSegment: 200
        )
        XCTAssertEqual(config.turnGapSeconds, 0.5)
        XCTAssertEqual(config.crossfadeSeconds, 0.1)
        XCTAssertEqual(config.defaultInstruction, "Speak cheerfully.")
        XCTAssertEqual(config.maxTokensPerSegment, 200)
    }
}
