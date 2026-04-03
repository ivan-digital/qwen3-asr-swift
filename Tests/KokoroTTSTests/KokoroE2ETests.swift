import XCTest
@testable import KokoroTTS
import CoreML

/// E2E tests using fromPretrained() — downloads models from HuggingFace.
/// Run with: swift test --filter E2EKokoroTests
final class E2EKokoroTests: XCTestCase {

    private static var _sharedModel: KokoroTTSModel?

    private func model() async throws -> KokoroTTSModel {
        if let m = Self._sharedModel { return m }
        let m = try await KokoroTTSModel.fromPretrained()
        Self._sharedModel = m
        return m
    }

    func testModelLoadsAndHasE2E() async throws {
        let m = try await model()
        XCTAssertTrue(m.availableVoices.contains("af_heart"))
    }

    func testSynthesizeEnglish() async throws {
        let m = try await model()
        let audio = try m.synthesize(text: "The quick brown fox jumps over the lazy dog.", voice: "af_heart")

        XCTAssertGreaterThan(audio.count, 1000, "Should produce meaningful audio")
        let duration = Double(audio.count) / 24000.0
        print("English: \(audio.count) samples (\(String(format: "%.2f", duration))s)")
        XCTAssertGreaterThan(duration, 0.3)
        XCTAssertLessThan(duration, 10.0)

        let rms = sqrt(audio.map { $0 * $0 }.reduce(0, +) / Float(audio.count))
        XCTAssertGreaterThan(rms, 0.001, "Audio should not be silence")
    }

    func testSynthesizeShortText() async throws {
        let m = try await model()
        let audio = try m.synthesize(text: "Hi", voice: "af_heart")
        XCTAssertGreaterThan(audio.count, 0, "Even short text should produce audio")
    }

    func testPhonemizerTokenizes() async throws {
        let m = try await model()
        // Verify phonemizer works via synthesis (if tokenization fails, synthesis throws)
        let audio = try m.synthesize(text: "Hello world", voice: "af_heart")
        XCTAssertGreaterThan(audio.count, 1000)
    }
}
