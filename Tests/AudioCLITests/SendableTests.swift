import XCTest
import SpeechVAD
import Qwen3ASR
import ParakeetASR

/// Compile-time verification that config types conform to Sendable.
/// These tests pass if they compile — the Task boundary enforces Sendable.
final class SendableTests: XCTestCase {

    func testVADConfigSendable() async {
        let config = VADConfig.sileroDefault
        let result = await Task { config }.value
        XCTAssertEqual(result.onset, config.onset)
    }

    func testSegmentationConfigSendable() async {
        let config = SegmentationConfig.default
        let result = await Task { config }.value
        XCTAssertEqual(result.sampleRate, 16000)
    }

    func testDiarizationConfigSendable() async {
        let config = DiarizationConfig.default
        let result = await Task { config }.value
        XCTAssertEqual(result.clusteringThreshold, 0.5)
    }

    func testQwen3AudioEncoderConfigSendable() async {
        let config = Qwen3AudioEncoderConfig.default
        let result = await Task { config }.value
        XCTAssertEqual(result.dModel, 896)
    }

    func testParakeetConfigSendable() async {
        let config = ParakeetConfig.default
        let result = await Task { config }.value
        XCTAssertEqual(result.encoderHidden, 1024)
    }

    func testAllConfigsSendableInTaskGroup() async {
        await withTaskGroup(of: Bool.self) { group in
            let vadConfig = VADConfig.sileroDefault
            let segConfig = SegmentationConfig.default
            let diarConfig = DiarizationConfig.default

            group.addTask { vadConfig.onset > 0 }
            group.addTask { segConfig.sampleRate == 16000 }
            group.addTask { diarConfig.maxSpeakers == 0 }

            for await result in group {
                XCTAssertTrue(result)
            }
        }
    }
}
