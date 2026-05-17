import XCTest
@testable import SpeechEnhancement

/// E2E tests for DeepFilterNet3 speech enhancement.
///
/// Runs on macOS (CoreML). The default model resolves to
/// ``SpeechEnhancer.defaultModelId`` and downloads only the pre-compiled
/// ``.mlmodelc`` bundle — ``MLModel.compileModel`` must never be reached at
/// runtime (the CoreML compile path drifts per runtime and caused garbage
/// audio on the iOS simulator, see the speech-models issue #4 history).
final class DeepFilterNet3E2ETests: XCTestCase {

    /// Verify the migration to `.mlmodelc/**` globs actually resolves at
    /// runtime against the published HuggingFace repo. CI filters this class
    /// out via the `--skip E2E` regex.
    func testHubFromPretrained() async throws {
        let enhancer = try await SpeechEnhancer.fromPretrained()

        // One-second 48 kHz silence -> enhance should return the same length.
        // We don't care about the output content here — this asserts that the
        // full pipeline (download, .mlmodelc load, ERB/DF signal processing,
        // auxiliary.npz parsing) works end-to-end.
        let sampleRate = 48000
        let samples = [Float](repeating: 0, count: sampleRate)
        let enhanced = try enhancer.enhance(audio: samples, sampleRate: sampleRate)

        XCTAssertEqual(enhanced.count, samples.count,
                       "Enhanced signal should match input length")

        // Silence in -> silence out (within FP16 conversion floor).
        let peak = enhanced.map(abs).max() ?? 0
        XCTAssertLessThan(peak, 0.01, "Enhanced silence should stay near 0")
    }
}
