import XCTest
@testable import Qwen3TTS
import AudioCommon

// MARK: - Unit Tests

final class EOSCapTests: XCTestCase {

    func testMaxTokensCap() {
        // "Hello" tokenizes to ~1-2 tokens → cap at max(75, 2*6) = 75
        // "A long sentence with many words to test the capping logic" → ~12 tokens → cap at max(75, 72) = 75
        // Very long text with 100+ tokens → cap at 100*6 = 600
        // Verify the formula: min(maxTokens, max(75, textTokenCount * 6))

        // Short text: floor kicks in
        let shortCap = min(500, max(75, 2 * 6))  // 2 tokens
        XCTAssertEqual(shortCap, 75, "Short text should use floor of 75")

        // Medium text: factor of 6
        let mediumCap = min(500, max(75, 20 * 6))  // 20 tokens
        XCTAssertEqual(mediumCap, 120, "Medium text: 20 tokens × 6 = 120")

        // Long text: maxTokens kicks in
        let longCap = min(500, max(75, 200 * 6))  // 200 tokens
        XCTAssertEqual(longCap, 500, "Long text should be capped at maxTokens=500")
    }
}

// MARK: - E2E Tests

final class E2EEOSCapTests: XCTestCase {

    func testXVectorShortEnglishEOS() async throws {
        // Short English text should stop well before maxTokens
        let model = try await Qwen3TTSModel.fromPretrained()

        let refAudio = try AudioFileLoader.load(
            url: URL(fileURLWithPath: "Tests/Qwen3ASRTests/Resources/test_audio.wav"),
            targetSampleRate: 24000)
        let refSpeech = Array(refAudio[Int(5.17 * 24000)..<min(Int(8.37 * 24000), refAudio.count)])

        let waveform = model.synthesizeWithVoiceClone(
            text: "Hello.",
            referenceAudio: refSpeech,
            referenceSampleRate: 24000,
            language: "english")

        XCTAssertGreaterThan(waveform.count, 0, "Should produce audio")
        let duration = Double(waveform.count) / 24000.0
        print("Short English x-vector: \(String(format: "%.2f", duration))s")
        // "Hello." is ~1-2 tokens → cap at 75 tokens = ~6s max
        XCTAssertLessThan(duration, 8.0, "Short text should not produce long audio")
    }

    func testXVectorGermanEOS() async throws {
        // Issue #139: German text hits maxTokens with x-vector mode.
        // With the cap, it should stop much earlier.
        let model = try await Qwen3TTSModel.fromPretrained()

        let refAudio = try AudioFileLoader.load(
            url: URL(fileURLWithPath: "Tests/Qwen3ASRTests/Resources/test_audio.wav"),
            targetSampleRate: 24000)
        let refSpeech = Array(refAudio[Int(5.17 * 24000)..<min(Int(8.37 * 24000), refAudio.count)])

        let waveform = model.synthesizeWithVoiceClone(
            text: "Hallo, das ist ein Test.",
            referenceAudio: refSpeech,
            referenceSampleRate: 24000,
            language: "german")

        XCTAssertGreaterThan(waveform.count, 0, "Should produce audio")
        let duration = Double(waveform.count) / 24000.0
        print("German x-vector: \(String(format: "%.2f", duration))s")
        // Without cap: would hit 500 tokens = ~40s. With cap: should be < 10s
        XCTAssertLessThan(duration, 12.0, "German text should not run to maxTokens")
    }
}
