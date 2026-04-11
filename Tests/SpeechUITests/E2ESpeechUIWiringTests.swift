import XCTest
import AudioCommon
import ParakeetStreamingASR
@testable import SpeechUI

/// E2E test that validates the documented `SpeechUI` wiring against a real
/// streaming ASR backend. Skipped in CI (E2E prefix) — runs locally with
/// `make test`. Loads `ParakeetStreamingASRModel`, drains `transcribeStream`
/// over a real audio fixture through `TranscriptionStore.apply(text:isFinal:)`,
/// and asserts the store ends up with the expected transcript.
///
/// This is the test that catches:
/// - drift in `PartialTranscript.text` / `.isFinal` shape
/// - silent breakage of the README quick-start snippet
/// - regressions in the `apply(text:isFinal:)` adapter pattern
@MainActor
final class E2ESpeechUIWiringTests: XCTestCase {

    func testTranscriptionStoreDrainsStreamingASR() async throws {
        let model = try await ParakeetStreamingASRModel.fromPretrained()
        let audioURL = Bundle.module.url(forResource: "test_audio", withExtension: "wav")!
        let audio = try AudioFileLoader.load(url: audioURL, targetSampleRate: 16000)

        let store = TranscriptionStore()

        // The exact wiring from the README quick-start.
        for await partial in model.transcribeStream(audio: audio, sampleRate: 16000) {
            store.apply(text: partial.text, isFinal: partial.isFinal)
        }

        // After the stream finishes, the partial should be cleared and finals
        // should contain at least one committed line with the expected content.
        XCTAssertNil(store.currentPartial,
                     "Stream end should clear in-progress partial")
        XCTAssertFalse(store.finalLines.isEmpty,
                       "At least one final transcript should be committed")

        let joined = store.finalLines.joined(separator: " ")
        XCTAssertTrue(joined.contains("guarantee"),
                      "Combined transcript should include 'guarantee', got: \(joined)")
        XCTAssertTrue(joined.contains("shipped tomorrow"),
                      "Combined transcript should include 'shipped tomorrow', got: \(joined)")
    }

    func testResetClearsAfterRealStream() async throws {
        let model = try await ParakeetStreamingASRModel.fromPretrained()
        let audioURL = Bundle.module.url(forResource: "test_audio", withExtension: "wav")!
        let audio = try AudioFileLoader.load(url: audioURL, targetSampleRate: 16000)

        let store = TranscriptionStore()
        for await partial in model.transcribeStream(audio: audio, sampleRate: 16000) {
            store.apply(text: partial.text, isFinal: partial.isFinal)
        }

        XCTAssertFalse(store.finalLines.isEmpty)
        store.reset()
        XCTAssertTrue(store.finalLines.isEmpty)
        XCTAssertNil(store.currentPartial)
    }
}
