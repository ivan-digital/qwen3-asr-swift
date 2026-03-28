import XCTest
@testable import Qwen3Chat

/// End-to-end tests for Qwen3.5-0.8B MLX chat model.
///
/// These tests download the model from HuggingFace on first run.
/// The model uses a hybrid DeltaNet + GatedAttention architecture.
final class E2EQwen35MLXChatTests: XCTestCase {

    func testModelLoading() async throws {
        let model = try await loadModelOrSkip()
        let metrics = model.lastMetrics
        XCTAssertEqual(metrics.tokensPerSec, 0)
    }

    func testBasicGeneration() async throws {
        let model = try await loadModelOrSkip()
        let response = try model.generate(
            messages: [
                ChatMessage(role: .user, content: "What is 2+2? Reply with just the number.")
            ],
            sampling: ChatSamplingConfig(temperature: 0.1, topK: 10, maxTokens: 20)
        )

        let trimmed = response.trimmingCharacters(in: .whitespacesAndNewlines)
        XCTAssertFalse(trimmed.isEmpty, "Should generate non-empty response")
        print("Qwen3.5 MLX generation: '\(trimmed)'")
        XCTAssertTrue(trimmed.contains("4"), "Should answer '4' but got: '\(trimmed)'")

        let metrics = model.lastMetrics
        XCTAssertTrue(metrics.tokensPerSec > 0, "Should measure decode speed")
        XCTAssertTrue(metrics.prefillMs > 0, "Should measure prefill time")
        print("Performance: \(String(format: "%.1f", metrics.tokensPerSec)) tok/s, "
            + "\(String(format: "%.0f", metrics.prefillMs))ms prefill")
    }

    func testNoMetaCommentary() async throws {
        let model = try await loadModelOrSkip()
        let response = try model.generate(
            messages: [
                ChatMessage(role: .system, content: "You are a helpful assistant."),
                ChatMessage(role: .user, content: "What is the capital of France?")
            ],
            sampling: ChatSamplingConfig(temperature: 0.3, topK: 20, maxTokens: 50)
        )

        let lower = response.lowercased()
        print("Response: '\(response.trimmingCharacters(in: .whitespacesAndNewlines))'")

        // Qwen3.5 should NOT produce meta-commentary like "Okay, the user is asking..."
        XCTAssertFalse(lower.contains("the user"), "Should not have meta-commentary")
        XCTAssertFalse(lower.hasPrefix("okay,"), "Should not start with 'Okay,'")
        XCTAssertTrue(lower.contains("paris"), "Should mention Paris")
    }

    func testStreamingGeneration() async throws {
        let model = try await loadModelOrSkip()
        var chunks: [String] = []

        let stream = model.generateStream(
            messages: [
                ChatMessage(role: .user, content: "Count to three.")
            ],
            sampling: ChatSamplingConfig(temperature: 0.3, topK: 20, maxTokens: 30)
        )

        for try await chunk in stream {
            chunks.append(chunk)
        }

        XCTAssertTrue(chunks.count > 0, "Should yield at least one chunk")
        let fullText = chunks.joined()
        XCTAssertFalse(fullText.isEmpty)
        print("Streaming: '\(fullText.trimmingCharacters(in: .whitespacesAndNewlines))'")
    }

    func testLocalModelGeneration() async throws {
        let localDir = URL(fileURLWithPath: "/tmp/Qwen3.5-0.8B-Chat-MLX-4bit")
        guard FileManager.default.fileExists(atPath: localDir.appendingPathComponent("model.safetensors").path) else {
            print("No local model at \(localDir.path), skipping")
            return
        }

        let model = try await Qwen35MLXChat.fromLocal(directory: localDir)

        // Debug: print the encoded prompt
        let messages = [ChatMessage(role: .user, content: "Say hello.")]
        let config = Qwen3ChatConfig.qwen35_08B
        let promptTokens = ChatTemplate.encode(
            messages: messages, tokenizer: model.tokenizer, config: config)
        print("Prompt tokens (\(promptTokens.count)): \(promptTokens)")
        print("Prompt decoded: '\(model.tokenizer.decode(promptTokens))'")

        let response = try model.generate(
            messages: messages,
            sampling: ChatSamplingConfig(temperature: 0.1, topK: 10, maxTokens: 20)
        )
        let trimmed = response.trimmingCharacters(in: .whitespacesAndNewlines)
        print("Local model response: '\(trimmed)'")
        XCTAssertFalse(trimmed.isEmpty, "Model should produce non-empty response")
    }

    // MARK: - Helpers

    private func loadModelOrSkip() async throws -> Qwen35MLXChat {
        return try await Qwen35MLXChat.fromPretrained { progress, status in
            print("[\(Int(progress * 100))%] \(status)")
        }
    }
}
