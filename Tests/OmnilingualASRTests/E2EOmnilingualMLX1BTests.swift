import XCTest
import AudioCommon
@testable import OmnilingualASR

/// Sanity-check the 1B MLX variant loads and runs. This test downloads ~549 MB
/// and exercises the larger encoder dim / layer count code path. Skipped in CI.
@MainActor
final class E2EOmnilingualMLX1BTests: XCTestCase {

    func testLoadAndTranscribe1B() async throws {
        let model = try await OmnilingualASRMLXModel.fromPretrained(variant: .b1, bits: 4)
        XCTAssertEqual(model.config.modelDim, 1280)
        XCTAssertEqual(model.config.numLayers, 48)
        XCTAssertEqual(model.config.numHeads, 20)

        let audioURL = Bundle.module.url(forResource: "test_audio", withExtension: "wav")!
        let audio = try AudioFileLoader.load(url: audioURL, targetSampleRate: 16000)

        let text = try model.transcribeAudio(audio, sampleRate: 16000)
        print("Omnilingual MLX (1B-4bit) transcript: \(text)")
        XCTAssertFalse(text.isEmpty)
        let lower = text.lowercased()
        XCTAssertTrue(["guarantee", "shipped", "tomorrow", "replacement"].contains { lower.contains($0) },
                      "Expected 1B transcript to contain a content word, got: \(text)")
    }
}
