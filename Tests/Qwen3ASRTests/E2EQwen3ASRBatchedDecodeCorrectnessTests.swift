import XCTest
@testable import Qwen3ASR
@testable import AudioCommon

/// Public batched transcription must preserve serial greedy correctness.
/// The experimental MLX batched decoder is intentionally environment-gated
/// until row-level bit-exactness is proven for batch sizes greater than one.
final class E2EQwen3ASRBatchedDecodeCorrectnessTests: XCTestCase {

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

    func testPublicBatchMatchesSerialForRepeatedSpeechChunk() async throws {
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

        XCTAssertEqual(batched, serial)
        XCTAssertEqual(Set(batched).count, 1, "Repeated chunks should produce identical transcriptions")
    }
}
