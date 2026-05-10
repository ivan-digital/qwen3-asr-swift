import XCTest
@testable import Qwen3ASR
@testable import AudioCommon

/// Regression test for the experimental `[B,1,H]` decoder path that ships
/// behind `QWEN3_ASR_EXPERIMENTAL_BATCH_DECODE=1`. The path is currently
/// known to truncate row 1 at batch size 2 on repeated identical chunks
/// (see PR #234), which is why the public `transcribeBatch` defaults to
/// the correctness-safe per-row decoder forwards. This test:
///
/// - Skips when the env var is unset (default CI run is unaffected).
/// - When the env var is set, runs `transcribeBatch` against `transcribe`
///   and `XCTExpectFailure`s the equality. As long as the bug exists, the
///   expected failure makes the test pass; once the path is fixed, the
///   `XCTExpectFailure` is unsatisfied and the test flips red — surfacing
///   the behavior change so we can drop the gate.
final class E2EQwen3ASRExperimentalBatchedDecodeTests: XCTestCase {

    static let modelId = "aufklarer/Qwen3-ASR-0.6B-MLX-4bit"
    static let targetSampleRate = 24000

    private func loadSpeechChunk(seconds: Int = 10) throws -> [Float] {
        guard let wavURL = Bundle.module.url(forResource: "test_audio", withExtension: "wav") else {
            throw XCTSkip("Test WAV file not found in bundle resources")
        }
        let (samples, sampleRate) = try AudioFileLoader.loadWAV(url: wavURL)
        let resampled = sampleRate == Self.targetSampleRate
            ? samples
            : AudioFileLoader.resample(samples, from: sampleRate, to: Self.targetSampleRate)
        return Array(resampled.prefix(Self.targetSampleRate * seconds))
    }

    func testExperimentalBatchTruncatesRow1ForRepeatedChunkAtBatchSize2() async throws {
        guard ProcessInfo.processInfo.environment["QWEN3_ASR_EXPERIMENTAL_BATCH_DECODE"] == "1" else {
            throw XCTSkip("Experimental batch decoder gate not set")
        }

        let model = try await Qwen3ASRModel.fromPretrained(modelId: Self.modelId)
        let chunk = try loadSpeechChunk()
        let audios = [chunk, chunk]

        let serial = audios.map {
            model.transcribe(audio: $0, sampleRate: Self.targetSampleRate, language: "en")
        }
        let batched = model.transcribeBatch(
            audios: audios,
            sampleRate: Self.targetSampleRate,
            language: "en"
        )

        XCTExpectFailure(
            "Experimental [B,1,H] decoder is known to truncate row 1 at batch=2 on repeated chunks. " +
            "When this path is fixed, this expected failure becomes an unexpected pass and this test " +
            "should be deleted (or repurposed as a strict equality assertion)."
        )
        XCTAssertEqual(batched, serial)
    }
}
