import Foundation

/// Mock chat backend for iOS simulator testing.
///
/// CoreML INT4 produces garbage on CPU-only simulator (no GPU/ANE).
/// This mock returns canned responses so the full pipeline (ASR→LLM→TTS)
/// can be tested on simulator for UI/UX validation.
public final class MockChatBackend: Qwen35ChatBackend {
    public let config = Qwen3ChatConfig.qwen35_08B
    public let tokenizer = ChatTokenizer()

    public init() {}

    public func generateStream(
        messages: [ChatMessage],
        sampling: ChatSamplingConfig
    ) -> AsyncThrowingStream<String, Error> {
        let lastMessage = messages.last?.content.lowercased() ?? ""

        let response: String
        if lastMessage.contains("name") {
            response = "I'm Tama, your on-device companion!"
        } else if lastMessage.contains("hello") || lastMessage.contains("hi ") || lastMessage.hasPrefix("hi") {
            response = "Hello! How can I help you today?"
        } else if lastMessage.contains("how are") {
            response = "I'm great, thanks for asking!"
        } else if lastMessage.contains("old") || lastMessage.contains("age") {
            response = "I'm brand new! Just created to help you."
        } else if lastMessage.contains("weather") {
            response = "I can't check the weather, but I hope it's nice!"
        } else if lastMessage.contains("what") && lastMessage.contains("do") {
            response = "I can chat with you about anything you like!"
        } else if lastMessage.contains("thank") {
            response = "You're welcome! Happy to help."
        } else if lastMessage.contains("bye") || lastMessage.contains("goodbye") {
            response = "Goodbye! Have a wonderful day!"
        } else if lastMessage.contains("?") {
            response = "That's a great question! Let me think about it."
        } else {
            response = "That's interesting! Tell me more."
        }

        return AsyncThrowingStream { continuation in
            // Simulate token-by-token streaming
            Task {
                for word in response.split(separator: " ") {
                    try? await Task.sleep(for: .milliseconds(50))
                    continuation.yield(String(word) + " ")
                }
                continuation.finish()
            }
        }
    }

    public func resetState() {}
}
