import XCTest
import Foundation
import MLX
@testable import Qwen3ASR
@testable import AudioCommon

/// Locks in the bit-exactness invariant that the asyncEval double-buffered
/// greedy decoder claims to preserve. The greedy fast path explicitly casts
/// argMax (uint32) -> int32 before quantized embedding lookup because MLX
/// dispatches differently across those dtypes on a small fraction of inputs.
/// If anyone removes that cast, or a future MLX upgrade changes argMax
/// dispatch, these tests fail with a clear diff instead of silently degrading
/// transcription.
final class E2EQwen3ASRGreedyDeterminismTests: XCTestCase {

    static let modelId = "aufklarer/Qwen3-ASR-0.6B-MLX-4bit"
    static let targetSampleRate = 24000

    private func loadAudio() throws -> [Float] {
        guard let wavURL = Bundle.module.url(forResource: "test_audio", withExtension: "wav") else {
            throw XCTSkip("Test WAV file not found in bundle resources")
        }
        let (samples, sampleRate) = try AudioFileLoader.loadWAV(url: wavURL)
        if sampleRate == Self.targetSampleRate { return samples }
        return AudioFileLoader.resample(samples, from: sampleRate, to: Self.targetSampleRate)
    }

    /// Greedy is fully deterministic: same audio + same model + default
    /// options must produce the same string every call. Catches accidental
    /// non-determinism (e.g. temperature sampling slipping into the path).
    func testGreedyDecodeIsDeterministic() async throws {
        let model = try await Qwen3ASRModel.fromPretrained(modelId: Self.modelId)
        let audio = try loadAudio()
        let first  = model.transcribe(audio: audio, sampleRate: Self.targetSampleRate)
        let second = model.transcribe(audio: audio, sampleRate: Self.targetSampleRate)
        XCTAssertEqual(first, second, "Greedy transcribe must be deterministic across calls")
        XCTAssertFalse(first.isEmpty, "Transcription should not be empty")
    }

    /// Snapshot test: exact-string match. The expected value is captured
    /// from the asyncEval greedy path on this exact model + audio combo,
    /// so any future change (intentional or not) that perturbs the token
    /// sequence surfaces here.
    ///
    /// First run: leave `expected` as `nil`. The test prints the live
    /// transcription and skips. Paste that string into `expected` below
    /// and re-run; the test then strict-asserts on every subsequent run.
    /// Same procedure to regenerate after an INTENTIONAL change.
    func testGreedyDecodeMatchesSnapshot() async throws {
        let expected: String? = "Can you guarantee that the replacement part will be shipped tomorrow?"

        let model = try await Qwen3ASRModel.fromPretrained(modelId: Self.modelId)
        let audio = try loadAudio()
        let result = model.transcribe(audio: audio, sampleRate: Self.targetSampleRate)
        print("Greedy snapshot transcription: \"\(result)\"")

        guard let expected else {
            throw XCTSkip("Snapshot not yet captured. Copy the printed transcription into `expected`.")
        }
        XCTAssertEqual(result, expected, "Greedy snapshot diverged — see diff")
    }
}
