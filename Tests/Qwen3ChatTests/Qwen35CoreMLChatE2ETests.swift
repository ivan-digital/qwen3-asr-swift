import XCTest
@testable import Qwen3Chat

/// E2E tests for Qwen3.5 CoreML chat model.
/// Runs on macOS (real GPU). Does NOT work on iOS simulator (CPU-only produces garbage).
final class E2EQwen35CoreMLChatTests: XCTestCase {

    func testLocalModelGeneration() async throws {
        let localDir = URL(fileURLWithPath: "/tmp/Qwen3.5-CoreML-INT8")
        // After the .mlpackage -> .mlmodelc migration, runtime loading only
        // accepts the compiled bundle. Skip if the local cache predates that.
        guard FileManager.default.fileExists(atPath: localDir.appendingPathComponent("decoder.mlmodelc").path) else {
            print("No compiled CoreML model at \(localDir.path)/decoder.mlmodelc, skipping")
            return
        }

        let model = try await Qwen35CoreMLChat.fromLocal(
            directory: localDir, computeUnits: .cpuAndGPU)

        let response = try model.generate(
            messages: [ChatMessage(role: .user, content: "What is 2+2? Reply with just the number.")],
            sampling: ChatSamplingConfig(temperature: 0.1, topK: 10, maxTokens: 20)
        )
        let trimmed = response.trimmingCharacters(in: .whitespacesAndNewlines)
        print("CoreML response: '\(trimmed)'")
        XCTAssertFalse(trimmed.isEmpty, "Should produce non-empty response")

        let m = model.lastMetrics
        print("CoreML performance: \(String(format: "%.1f", m.tokensPerSec)) tok/s, \(String(format: "%.0f", m.prefillMs))ms prefill")
    }

    func testNoMetaCommentary() async throws {
        let localDir = URL(fileURLWithPath: "/tmp/Qwen3.5-CoreML-INT8")
        guard FileManager.default.fileExists(atPath: localDir.appendingPathComponent("decoder.mlmodelc").path) else {
            print("No compiled CoreML model, skipping")
            return
        }

        let model = try await Qwen35CoreMLChat.fromLocal(
            directory: localDir, computeUnits: .cpuAndGPU)

        let response = try model.generate(
            messages: [
                ChatMessage(role: .system, content: "You are a helpful assistant."),
                ChatMessage(role: .user, content: "What is the capital of France?")
            ],
            sampling: ChatSamplingConfig(temperature: 0.3, topK: 20, maxTokens: 50)
        )
        let lower = response.lowercased()
        print("CoreML response: '\(response.trimmingCharacters(in: .whitespacesAndNewlines))'")
        XCTAssertFalse(lower.contains("the user"), "No meta-commentary")
        XCTAssertTrue(lower.contains("paris"), "Should mention Paris")
    }

    /// Downloads the pre-compiled `.mlmodelc` bundle from HuggingFace and
    /// runs one generation. Verifies the post-migration download glob
    /// (`*.mlmodelc/**` only) actually resolves at runtime on the default
    /// modelId. CI filters this class out via the `--skip E2E` regex.
    func testHubFromPretrained() async throws {
        let model = try await Qwen35CoreMLChat.fromPretrained(
            computeUnits: .cpuAndGPU
        )

        let response = try model.generate(
            messages: [ChatMessage(role: .user, content: "Reply with just the digit 7.")],
            sampling: ChatSamplingConfig(temperature: 0.1, topK: 5, maxTokens: 10)
        )
        let trimmed = response.trimmingCharacters(in: .whitespacesAndNewlines)
        print("Hub fromPretrained response: '\(trimmed)'")
        XCTAssertFalse(trimmed.isEmpty, "Should produce non-empty response")
    }
}
