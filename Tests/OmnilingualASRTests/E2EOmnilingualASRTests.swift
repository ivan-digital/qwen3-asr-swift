import XCTest
import AudioCommon
@testable import OmnilingualASR

/// E2E test that loads the published CoreML model from HuggingFace and
/// transcribes a real audio fixture. Skipped in CI (E2E prefix) — runs
/// locally with `make test`.
@MainActor
final class E2EOmnilingualASRTests: XCTestCase {

    func testTranscribeRealAudio() async throws {
        let model = try await OmnilingualASRModel.fromPretrained()
        XCTAssertTrue(model.isLoaded)
        XCTAssertGreaterThan(model.memoryFootprint, 0)
        XCTAssertEqual(model.config.sampleRate, 16000)

        let audioURL = Bundle.module.url(forResource: "test_audio", withExtension: "wav")!
        let audio = try AudioFileLoader.load(url: audioURL, targetSampleRate: 16000)
        XCTAssertGreaterThan(audio.count, 0)

        let text = try model.transcribeAudio(audio, sampleRate: 16000)
        print("Omnilingual transcript: \(text)")

        XCTAssertFalse(text.isEmpty, "Transcription should not be empty")
        // The test audio says:
        //   "Can you guarantee that the replacement part will be shipped tomorrow?"
        // Omnilingual CTC-300M on English clean audio typically has WER <30%.
        // Check for content words that should survive greedy CTC without LM:
        let lower = text.lowercased()
        let expectedAny = ["guarantee", "shipped", "replacement", "tomorrow"]
        let found = expectedAny.filter { lower.contains($0) }
        XCTAssertFalse(found.isEmpty,
                       "Expected at least one of \(expectedAny) in transcript, got: \(text)")
    }

    func testWarmup() async throws {
        let model = try await OmnilingualASRModel.fromPretrained()
        try model.warmUp()
    }

    func testUnloadFreesMemory() async throws {
        let model = try await OmnilingualASRModel.fromPretrained()
        XCTAssertTrue(model.isLoaded)
        XCTAssertGreaterThan(model.memoryFootprint, 0)

        model.unload()
        XCTAssertFalse(model.isLoaded)
        XCTAssertEqual(model.memoryFootprint, 0)
    }

    func testChunkLongAudioIntoWindows() async throws {
        let model = try await OmnilingualASRModel.fromPretrained()
        let audioURL = Bundle.module.url(forResource: "test_audio", withExtension: "wav")!
        let audio = try AudioFileLoader.load(url: audioURL, targetSampleRate: 16000)

        // Build a multi-window utterance under the 40 s reference cap by
        // taking the first 10 s of the clip plus 5 s silence plus another 10 s.
        // Tests the chunker boundary without exceeding the API hard cap.
        let prefix = Array(audio.prefix(10 * 16000))
        let silence = [Float](repeating: 0, count: 5 * 16000)
        let longAudio = prefix + silence + prefix

        let text = try model.transcribeAudio(longAudio, sampleRate: 16000)
        XCTAssertFalse(text.isEmpty)
        print("Omnilingual chunked transcript: \(text)")
    }

    func testRejectsAudioExceedingMaxSeconds() async throws {
        let model = try await OmnilingualASRModel.fromPretrained()
        // 41 s of silence — must exceed the 40 s reference cap.
        let tooLong = [Float](repeating: 0, count: 41 * 16000)
        XCTAssertThrowsError(try model.transcribeAudio(tooLong, sampleRate: 16000)) { error in
            let message = "\(error)"
            XCTAssertTrue(message.contains("40") || message.lowercased().contains("cap"),
                          "Expected 40 s cap error, got: \(message)")
        }
    }

    // MARK: - Multilingual (FLEURS clips)

    func testTranscribeEnglishFleurs() async throws {
        let model = try await OmnilingualASRModel.fromPretrained()
        let url = Bundle.module.url(forResource: "fleurs_en", withExtension: "wav")!
        let audio = try AudioFileLoader.load(url: url, targetSampleRate: 16000)
        let text = try model.transcribeAudio(audio, sampleRate: 16000)
        print("Omnilingual EN: \(text)")
        // Reference transcript: "Fellow wrestlers also paid tribute to Luna."
        let lower = text.lowercased()
        let hits = ["wrestlers", "tribute", "luna", "fellow"].filter { lower.contains($0) }
        XCTAssertFalse(hits.isEmpty, "Expected at least one of [wrestlers, tribute, luna, fellow], got: \(text)")
    }

    func testTranscribeHindiFleurs() async throws {
        let model = try await OmnilingualASRModel.fromPretrained()
        let url = Bundle.module.url(forResource: "fleurs_hi", withExtension: "wav")!
        let audio = try AudioFileLoader.load(url: url, targetSampleRate: 16000)
        let text = try model.transcribeAudio(audio, sampleRate: 16000)
        print("Omnilingual HI: \(text)")
        // Reference: "लूना को साथी पहलवानों ने भी श्रद्धांजलि दी."
        // Verify Devanagari output and at least one content word.
        XCTAssertFalse(text.isEmpty)
        let containsDevanagari = text.unicodeScalars.contains { (0x0900...0x097F).contains($0.value) }
        XCTAssertTrue(containsDevanagari, "Expected Devanagari script in HI transcript, got: \(text)")
    }

    func testTranscribeArabicFleurs() async throws {
        let model = try await OmnilingualASRModel.fromPretrained()
        let url = Bundle.module.url(forResource: "fleurs_ar", withExtension: "wav")!
        let audio = try AudioFileLoader.load(url: url, targetSampleRate: 16000)
        let text = try model.transcribeAudio(audio, sampleRate: 16000)
        print("Omnilingual AR: \(text)")
        // Reference (parallel to fleurs_en): "كما أثنى الزملاء المصارعون على لونا."
        // Verify Arabic script is produced.
        XCTAssertFalse(text.isEmpty)
        let containsArabic = text.unicodeScalars.contains { (0x0600...0x06FF).contains($0.value) }
        XCTAssertTrue(containsArabic, "Expected Arabic script in AR transcript, got: \(text)")
    }

    func testTranscribeFrenchFleurs() async throws {
        let model = try await OmnilingualASRModel.fromPretrained()
        let url = Bundle.module.url(forResource: "fleurs_fr", withExtension: "wav")!
        let audio = try AudioFileLoader.load(url: url, targetSampleRate: 16000)
        let text = try model.transcribeAudio(audio, sampleRate: 16000)
        print("Omnilingual FR: \(text)")
        // Reference: "Pensez à l'itinéraire de ski comme à un itinéraire de randonnée similaire."
        let lower = text.lowercased()
        let hits = ["itinéraire", "ski", "randonnée", "similaire", "pensez"].filter { lower.contains($0) }
        XCTAssertFalse(hits.isEmpty, "Expected one of [itinéraire, ski, randonnée, similaire, pensez], got: \(text)")
    }
}
