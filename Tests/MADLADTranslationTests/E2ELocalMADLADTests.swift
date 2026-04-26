import XCTest
@testable import MADLADTranslation

/// Local-directory smoke tests — validate weights produced by `convert_mlx.py`
/// before publishing to HuggingFace.
///
/// Set `MADLAD_LOCAL_DIR` to the converted variant directory (e.g.
/// `/tmp/MADLAD400-3B-MT-MLX/int4`) and run:
///
///     MADLAD_LOCAL_DIR=/tmp/MADLAD400-3B-MT-MLX/int4 \
///       swift test --filter E2ELocalMADLADTests
///
/// Each test SKIPS when the env var is absent, so this file is safe in CI.
final class E2ELocalMADLADTests: XCTestCase {

    func testLoadAndTranslateEnglishToSpanish() async throws {
        let translator = try await loadOrSkip()
        let result = try translator.translate("Hello, how are you?", to: "es")
        print("[en→es] \(result)")
        XCTAssertFalse(result.isEmpty)
        let lower = result.lowercased()
        XCTAssertTrue(
            lower.contains("hola") || lower.contains("cómo") || lower.contains("estás") || lower.contains("estas"),
            "Expected Spanish output, got: \(result)")
    }

    func testEnglishToFrench() async throws {
        let translator = try await loadOrSkip()
        let result = try translator.translate("Good morning", to: "fr")
        print("[en→fr] \(result)")
        XCTAssertFalse(result.isEmpty)
        let lower = result.lowercased()
        XCTAssertTrue(
            lower.contains("bonjour") || lower.contains("matin"),
            "Expected French output, got: \(result)")
    }

    func testEnglishToChinese() async throws {
        let translator = try await loadOrSkip()
        let result = try translator.translate("Thank you", to: "zh")
        print("[en→zh] \(result)")
        XCTAssertFalse(result.isEmpty)
        let hasCJK = result.unicodeScalars.contains { (0x4E00...0x9FFF).contains($0.value) }
        XCTAssertTrue(hasCJK, "Expected Chinese characters, got: \(result)")
    }

    func testGreedyDeterministic() async throws {
        let translator = try await loadOrSkip()
        let a = try translator.translate("Where is the library?", to: "es")
        let b = try translator.translate("Where is the library?", to: "es")
        print("[determinism] \(a)")
        XCTAssertEqual(a, b, "Greedy decode must be deterministic")
    }

    // MARK: - Helpers

    private func loadOrSkip() async throws -> MADLADTranslator {
        guard let path = ProcessInfo.processInfo.environment["MADLAD_LOCAL_DIR"], !path.isEmpty else {
            throw XCTSkip("Set MADLAD_LOCAL_DIR to the converted variant directory")
        }
        let url = URL(fileURLWithPath: path)
        let bits = path.contains("int8") ? 8 : 4
        return try await MADLADTranslator.fromLocal(directory: url, bits: bits) { progress, status in
            if Int(progress * 100) % 10 == 0 {
                print("[\(Int(progress * 100))%] \(status)")
            }
        }
    }
}
