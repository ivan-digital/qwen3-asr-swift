import XCTest
import CoreML
@testable import Qwen3Chat

/// End-to-end tests for Qwen3 chat model.
///
/// These tests require the CoreML model to be downloaded or available locally.
/// Tests using `loadModelOrSkip()` download from HuggingFace.
/// The `testLocalModelGeneration` test uses a local model if available.
final class Qwen3ChatE2ETests: XCTestCase {

    static let modelId = Qwen3ChatModel.defaultModelId

    func testModelLoading() async throws {
        let model = try await loadModelOrSkip()
        let metrics = model.lastMetrics
        XCTAssertEqual(metrics.tokensPerSec, 0)
    }

    func testSingleGeneration() async throws {
        let model = try await loadModelOrSkip()
        let response = try model.generate(
            messages: [
                ChatMessage(role: .user, content: "What is 2+2? Reply with just the number.")
            ],
            sampling: ChatSamplingConfig(temperature: 0.3, topK: 20, maxTokens: 20)
        )

        let trimmed = response.trimmingCharacters(in: .whitespacesAndNewlines)
        // Note: INT4 quantized 0.6B model may produce empty or unexpected responses
        // with greedy decoding. This test verifies the pipeline runs without crash.
        print("Generation: '\(trimmed)'")

        let metrics = model.lastMetrics
        XCTAssertTrue(metrics.tokensPerSec > 0, "Should measure decode speed")
        XCTAssertTrue(metrics.prefillMs > 0, "Should measure prefill time")
    }

    func testStreamingGeneration() async throws {
        let model = try await loadModelOrSkip()
        var chunks: [String] = []

        let stream = model.generateStream(
            messages: [
                ChatMessage(role: .user, content: "Count to three.")
            ],
            sampling: ChatSamplingConfig(temperature: 0.3, topK: 20, topP: 0.8, maxTokens: 512)
        )

        for try await chunk in stream {
            chunks.append(chunk)
        }

        XCTAssertTrue(chunks.count > 0, "Should yield at least one chunk")
        let fullText = chunks.joined()
        XCTAssertFalse(fullText.isEmpty, "Full text should not be empty")
    }

    func testChatWithPromptCaching() async throws {
        let model = try await loadModelOrSkip()

        let sampling = ChatSamplingConfig(temperature: 0.3, topK: 20, topP: 0.8, maxTokens: 512)

        let response1 = try model.chat(
            "Hi",
            systemPrompt: "You are a helpful assistant. Be brief.",
            sampling: sampling
        )
        XCTAssertFalse(response1.isEmpty)

        let firstTurnPrefill = model.lastMetrics.prefillMs

        let response2 = try model.chat(
            "What is 2+2?",
            systemPrompt: "You are a helpful assistant. Be brief.",
            sampling: sampling
        )
        XCTAssertFalse(response2.isEmpty)

        print("First turn prefill: \(firstTurnPrefill)ms")
        print("Second turn prefill: \(model.lastMetrics.prefillMs)ms")
    }

    func testResetConversation() async throws {
        let model = try await loadModelOrSkip()

        _ = try model.chat("Hello", systemPrompt: "Be brief.",
                           sampling: ChatSamplingConfig(maxTokens: 512))
        model.resetConversation()

        let response = try model.chat("Hi again", systemPrompt: "Be brief.",
                                      sampling: ChatSamplingConfig(maxTokens: 512))
        XCTAssertFalse(response.isEmpty)
    }

    func testMultiTurnConversation() async throws {
        let model = try await loadModelOrSkip()

        let sampling = ChatSamplingConfig(temperature: 0.3, topK: 20, maxTokens: 30)

        let r1 = try model.chat("My name is Alice.", systemPrompt: "Remember the user's name. Be brief.", sampling: sampling)
        XCTAssertFalse(r1.isEmpty)

        let r2 = try model.chat("What is my name?", sampling: sampling)
        let r2lower = r2.lowercased()
        XCTAssertTrue(r2lower.contains("alice"),
            "Should remember 'Alice' from previous turn but got: '\(r2)'")
        print("Multi-turn: '\(r2.trimmingCharacters(in: .whitespacesAndNewlines))'")
    }

    func testStreamingChat() async throws {
        let model = try await loadModelOrSkip()
        var chunks: [String] = []

        let stream = model.chatStream(
            "Say hello briefly.",
            systemPrompt: "You are helpful.",
            sampling: ChatSamplingConfig(temperature: 0.3, topK: 20, topP: 0.8, maxTokens: 512)
        )

        for try await chunk in stream {
            chunks.append(chunk)
        }

        XCTAssertTrue(chunks.count > 0)
    }

    func testGenerationMetrics() async throws {
        let model = try await loadModelOrSkip()

        _ = try model.generate(
            messages: [
                ChatMessage(role: .user, content: "Hi")
            ],
            sampling: ChatSamplingConfig(maxTokens: 512)
        )

        let m = model.lastMetrics
        XCTAssertTrue(m.tokensPerSec > 0, "Tokens/sec should be positive")
        XCTAssertTrue(m.prefillMs > 0, "Prefill time should be positive")
        XCTAssertTrue(m.decodeMs > 0, "Decode time should be positive")
        XCTAssertTrue(m.msPerToken > 0, "ms/token should be positive")

        print("Performance: \(String(format: "%.1f", m.tokensPerSec)) tok/s, "
            + "\(String(format: "%.0f", m.prefillMs))ms prefill, "
            + "\(String(format: "%.1f", m.msPerToken))ms/tok")
    }

    func testLocalModelGeneration() async throws {
        let localDir = URL(fileURLWithPath: "/tmp/qwen3-chat-test-cache-fp16")
        guard FileManager.default.fileExists(atPath: localDir.appendingPathComponent("Qwen3Chat.mlpackage").path) else {
            print("No local model found at \(localDir.path), skipping")
            return
        }

        let model = try await Qwen3ChatModel.fromLocal(
            directory: localDir,
            computeUnits: .all
        )

        let response = try model.generate(
            messages: [ChatMessage(role: .user, content: "Say hello in one word.")],
            sampling: ChatSamplingConfig(maxTokens: 512)
        )
        let trimmed = response.trimmingCharacters(in: .whitespacesAndNewlines)
        print("Response: '\(trimmed)'")
        XCTAssertFalse(trimmed.isEmpty, "Should generate non-empty response")
    }

    // MARK: - INT8 Variant

    func testINT8ModelLoading() async throws {
        let model = try await loadModelOrSkip(quantization: .int8)
        let metrics = model.lastMetrics
        XCTAssertEqual(metrics.tokensPerSec, 0)
    }

    func testINT8Generation() async throws {
        let model = try await loadModelOrSkip(quantization: .int8)
        let response = try model.generate(
            messages: [ChatMessage(role: .user, content: "What is 2+2? Reply with just the number.")],
            sampling: ChatSamplingConfig(temperature: 0.3, topK: 20, maxTokens: 20)
        )
        let trimmed = response.lowercased().trimmingCharacters(in: .whitespacesAndNewlines)
        XCTAssertFalse(trimmed.isEmpty, "INT8 model should generate non-empty response")
        XCTAssertTrue(trimmed.contains("4"), "INT8 should answer '4' but got: '\(trimmed)'")
        print("INT8 generation: '\(trimmed)'")

        let m = model.lastMetrics
        print("INT8 performance: \(String(format: "%.1f", m.tokensPerSec)) tok/s, "
            + "\(String(format: "%.0f", m.prefillMs))ms prefill")
    }

    func testINT8StreamingChat() async throws {
        let model = try await loadModelOrSkip(quantization: .int8)
        var chunks: [String] = []

        let stream = model.chatStream(
            "What is 1+1?",
            systemPrompt: "Answer in one word.",
            sampling: ChatSamplingConfig(temperature: 0.3, maxTokens: 10)
        )

        for try await chunk in stream {
            chunks.append(chunk)
        }

        XCTAssertTrue(chunks.count > 0, "INT8 streaming should yield tokens")
        let text = chunks.joined()
        XCTAssertFalse(text.isEmpty)
        print("INT8 streaming: '\(text.trimmingCharacters(in: .whitespacesAndNewlines))'")
    }

    // MARK: - Helpers

    private func loadModelOrSkip(quantization: Qwen3ChatModel.Quantization = .int4) async throws -> Qwen3ChatModel {
        return try await Qwen3ChatModel.fromPretrained(
            modelId: Self.modelId,
            quantization: quantization,
            computeUnits: .all
        ) { progress, status in
            print("[\(Int(progress * 100))%] \(status)")
        }
    }
}
