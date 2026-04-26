import XCTest
@testable import MADLADTranslation

/// End-to-end tests that download the MADLAD-400-3B-MT MLX model and run real
/// translations. Skipped automatically by `--skip E2E` in CI.
final class E2EMADLADTranslationTests: XCTestCase {

    func testEnglishToSpanish() async throws {
        let translator = try await loadTranslatorOrSkip()
        let result = try translator.translate("Hello, how are you?", to: "es")
        let lower = result.lowercased()
        // Accept any reasonable Spanish translation containing core lexicon.
        XCTAssertFalse(result.isEmpty, "Translation should be non-empty")
        XCTAssertTrue(
            lower.contains("hola") || lower.contains("cómo") || lower.contains("estás"),
            "Expected Spanish output, got: \(result)")
    }

    func testEnglishToFrench() async throws {
        let translator = try await loadTranslatorOrSkip()
        let result = try translator.translate("Good morning", to: "fr")
        XCTAssertFalse(result.isEmpty)
        let lower = result.lowercased()
        XCTAssertTrue(
            lower.contains("bonjour") || lower.contains("matin"),
            "Expected French output, got: \(result)")
    }

    func testEnglishToChinese() async throws {
        let translator = try await loadTranslatorOrSkip()
        let result = try translator.translate("Thank you", to: "zh")
        XCTAssertFalse(result.isEmpty)
        // Output should contain CJK characters
        let hasCJK = result.unicodeScalars.contains { scalar in
            (0x4E00...0x9FFF).contains(scalar.value)
        }
        XCTAssertTrue(hasCJK, "Expected Chinese characters, got: \(result)")
    }

    func testGreedyDeterministic() async throws {
        let translator = try await loadTranslatorOrSkip()
        let r1 = try translator.translate("Where is the library?", to: "es")
        let r2 = try translator.translate("Where is the library?", to: "es")
        XCTAssertEqual(r1, r2, "Greedy decode should be deterministic")
    }

    func testStreamingMatchesNonStreaming() async throws {
        let translator = try await loadTranslatorOrSkip()
        let direct = try translator.translate("Hello world", to: "es")

        var streamed = ""
        for try await piece in translator.translateStream("Hello world", to: "es") {
            streamed += piece
        }
        // Streaming output is per-token decoded; non-streaming joins at the
        // end. They should match closely (allow whitespace differences).
        XCTAssertEqual(
            streamed.trimmingCharacters(in: .whitespaces),
            direct.trimmingCharacters(in: .whitespaces))
    }

    func testUnsupportedLanguageThrows() async throws {
        let translator = try await loadTranslatorOrSkip()
        XCTAssertThrowsError(try translator.translate("Hello", to: "xxnotalang")) { err in
            guard case MADLADTranslationError.unsupportedLanguage = err else {
                return XCTFail("Expected unsupportedLanguage, got \(err)")
            }
        }
    }

    // MARK: - Helpers

    private func loadTranslatorOrSkip() async throws -> MADLADTranslator {
        try await MADLADTranslator.fromPretrained { progress, status in
            print("[\(Int(progress * 100))%] \(status)")
        }
    }
}
