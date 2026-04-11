import XCTest
import AudioCommon
@testable import OmnilingualASR

/// E2E tests for the MLX backend of Omnilingual ASR. Loads the published
/// `aufklarer/Omnilingual-ASR-CTC-300M-MLX-4bit` repo from HuggingFace and
/// runs end-to-end on real audio. Skipped in CI (E2E prefix) — runs locally
/// with `make test`.
@MainActor
final class E2EOmnilingualMLXTests: XCTestCase {

    func testLoadsModel() async throws {
        let model = try await OmnilingualASRMLXModel.fromPretrained(variant: .m300, bits: 4)
        XCTAssertTrue(model.isLoaded)
        XCTAssertEqual(model.config.modelDim, 1024)
        XCTAssertEqual(model.config.numLayers, 24)
        XCTAssertEqual(model.config.numHeads, 16)
        XCTAssertEqual(model.config.ffnDim, 4096)
        XCTAssertEqual(model.config.vocabSize, 10288)
        XCTAssertEqual(model.config.bits, 4)
        XCTAssertGreaterThan(model.memoryFootprint, 0)
    }

    func testWarmup() async throws {
        let model = try await OmnilingualASRMLXModel.fromPretrained(variant: .m300, bits: 4)
        try model.warmUp()
    }

    func testTranscribeRealAudio() async throws {
        let model = try await OmnilingualASRMLXModel.fromPretrained(variant: .m300, bits: 4)

        let audioURL = Bundle.module.url(forResource: "test_audio", withExtension: "wav")!
        let audio = try AudioFileLoader.load(url: audioURL, targetSampleRate: 16000)
        XCTAssertGreaterThan(audio.count, 0)

        let text = try model.transcribeAudio(audio, sampleRate: 16000)
        print("Omnilingual MLX (300M-4bit) transcript: \(text)")

        // The clip says: "Can you guarantee that the replacement part will be shipped tomorrow?"
        XCTAssertFalse(text.isEmpty, "Transcription should not be empty")
        let lower = text.lowercased()
        let expectedAny = ["guarantee", "shipped", "replacement", "tomorrow"]
        let found = expectedAny.filter { lower.contains($0) }
        XCTAssertFalse(found.isEmpty,
                       "Expected at least one of \(expectedAny) in MLX transcript, got: \(text)")
    }

    func testRejectsAudioExceedingMaxSeconds() async throws {
        let model = try await OmnilingualASRMLXModel.fromPretrained(variant: .m300, bits: 4)
        let tooLong = [Float](repeating: 0, count: 41 * 16000)
        XCTAssertThrowsError(try model.transcribeAudio(tooLong, sampleRate: 16000)) { error in
            let message = "\(error)"
            XCTAssertTrue(message.contains("40") || message.lowercased().contains("cap"),
                          "Expected 40 s cap error, got: \(message)")
        }
    }

    func testTranscribeArabicFleurs() async throws {
        let model = try await OmnilingualASRMLXModel.fromPretrained(variant: .m300, bits: 4)
        let url = Bundle.module.url(forResource: "fleurs_ar", withExtension: "wav")!
        let audio = try AudioFileLoader.load(url: url, targetSampleRate: 16000)
        let text = try model.transcribeAudio(audio, sampleRate: 16000)
        print("Omnilingual MLX AR: \(text)")
        XCTAssertFalse(text.isEmpty)
        let containsArabic = text.unicodeScalars.contains { (0x0600...0x06FF).contains($0.value) }
        XCTAssertTrue(containsArabic, "Expected Arabic script, got: \(text)")
    }

    func testTranscribeHindiFleurs() async throws {
        let model = try await OmnilingualASRMLXModel.fromPretrained(variant: .m300, bits: 4)
        let url = Bundle.module.url(forResource: "fleurs_hi", withExtension: "wav")!
        let audio = try AudioFileLoader.load(url: url, targetSampleRate: 16000)
        let text = try model.transcribeAudio(audio, sampleRate: 16000)
        print("Omnilingual MLX HI: \(text)")
        XCTAssertFalse(text.isEmpty)
        let containsDevanagari = text.unicodeScalars.contains { (0x0900...0x097F).contains($0.value) }
        XCTAssertTrue(containsDevanagari, "Expected Devanagari script, got: \(text)")
    }

    func testTranscribeFrenchFleurs() async throws {
        let model = try await OmnilingualASRMLXModel.fromPretrained(variant: .m300, bits: 4)
        let url = Bundle.module.url(forResource: "fleurs_fr", withExtension: "wav")!
        let audio = try AudioFileLoader.load(url: url, targetSampleRate: 16000)
        let text = try model.transcribeAudio(audio, sampleRate: 16000)
        print("Omnilingual MLX FR: \(text)")
        XCTAssertFalse(text.isEmpty)
    }
}
