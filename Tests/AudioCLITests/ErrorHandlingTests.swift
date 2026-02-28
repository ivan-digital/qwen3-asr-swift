import XCTest
@testable import AudioCommon

final class ErrorHandlingTests: XCTestCase {

    func testModelLoadFailedDescription() {
        let error = AudioModelError.modelLoadFailed(
            modelId: "test/model-1B", reason: "config.json missing")
        let desc = error.errorDescription!
        XCTAssertTrue(desc.contains("test/model-1B"), "Should include modelId")
        XCTAssertTrue(desc.contains("config.json missing"), "Should include reason")
    }

    func testModelLoadFailedWithUnderlying() {
        let underlying = NSError(domain: "test", code: 42, userInfo: [
            NSLocalizedDescriptionKey: "file not found"
        ])
        let error = AudioModelError.modelLoadFailed(
            modelId: "org/model", reason: "download failed", underlying: underlying)
        let desc = error.errorDescription!
        XCTAssertTrue(desc.contains("org/model"))
        XCTAssertTrue(desc.contains("file not found"), "Should include underlying error")
    }

    func testWeightLoadingFailedDescription() {
        let error = AudioModelError.weightLoadingFailed(path: "/tmp/weights.safetensors")
        let desc = error.errorDescription!
        XCTAssertTrue(desc.contains("/tmp/weights.safetensors"), "Should include path")
    }

    func testInferenceFailedDescription() {
        let error = AudioModelError.inferenceFailed(
            operation: "text decoding", reason: "max tokens exceeded")
        let desc = error.errorDescription!
        XCTAssertTrue(desc.contains("text decoding"), "Should include operation")
        XCTAssertTrue(desc.contains("max tokens exceeded"), "Should include reason")
    }

    func testInvalidConfigurationDescription() {
        let error = AudioModelError.invalidConfiguration(
            model: "Qwen3-ASR", reason: "encoder layers must be > 0")
        let desc = error.errorDescription!
        XCTAssertTrue(desc.contains("Qwen3-ASR"), "Should include model name")
        XCTAssertTrue(desc.contains("encoder layers must be > 0"), "Should include reason")
    }

    func testVoiceNotFoundDescription() {
        let error = AudioModelError.voiceNotFound(
            voice: "NATM0", searchPath: "/tmp/voices/NATM0.safetensors")
        let desc = error.errorDescription!
        XCTAssertTrue(desc.contains("NATM0"), "Should include voice name")
        XCTAssertTrue(desc.contains("/tmp/voices/NATM0.safetensors"), "Should include search path")
    }
}
