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

    // MARK: - Long-form

    /// Multi-sentence paragraph (~70 source tokens) — verifies the encoder /
    /// relative position bias / cross-attn cache hold up beyond short phrases.
    func testParagraphTranslation() async throws {
        let translator = try await loadTranslatorOrSkip(maxTokens: 256)
        let source = """
        The library is located on the corner of Main Street and Oak Avenue. \
        It opens at nine in the morning and closes at six in the evening. \
        On weekends the hours are shorter — please check the website before you visit. \
        Anyone with a valid identification card can borrow up to ten books at a time.
        """
        let result = try translator.translate(source, to: "es")
        XCTAssertFalse(result.isEmpty)
        XCTAssertGreaterThan(result.count, 80,
            "Paragraph translation should be at least ~80 chars, got: \(result)")
        let lower = result.lowercased()
        // Spanish should contain core lexical anchors from each sentence.
        XCTAssertTrue(
            lower.contains("biblioteca") && (lower.contains("nueve") || lower.contains("9"))
              && lower.contains("identificación"),
            "Expected coherent paragraph translation, got: \(result)")
    }

    // MARK: - INT8 spot-check

    /// Same en→es path on the INT8 variant. INT8 weights are larger but
    /// produce closely-matching outputs (slight numerical drift from greater
    /// quantization fidelity is acceptable).
    func testInt8EnglishToSpanish() async throws {
        let translator = try await loadTranslatorOrSkip(quantization: .int8)
        let result = try translator.translate("Hello, how are you?", to: "es")
        let lower = result.lowercased()
        XCTAssertFalse(result.isEmpty)
        XCTAssertTrue(
            lower.contains("hola") || lower.contains("cómo") || lower.contains("estás"),
            "Expected Spanish output (INT8), got: \(result)")
    }


    // MARK: - Helpers

    private func loadTranslatorOrSkip(
        quantization: MADLADTranslator.Quantization = .int4,
        maxTokens: Int = 256
    ) async throws -> MADLADTranslator {
        try await MADLADTranslator.fromPretrained(quantization: quantization) { progress, status in
            print("[\(Int(progress * 100))%] \(status)")
        }
    }

}
