import Foundation

/// Chat message for Qwen3.5 models.
public struct ChatMessage: Sendable {
    public enum Role: String, Sendable {
        case system
        case user
        case assistant
    }

    public let role: Role
    public let content: String

    public init(role: Role, content: String) {
        self.role = role
        self.content = content
    }
}

/// Formats messages into Qwen3.5 chat template tokens.
///
/// ```
/// <|im_start|>system
/// {system_message}<|im_end|>
/// <|im_start|>user
/// {user_message}<|im_end|>
/// <|im_start|>assistant
/// ```
enum ChatTemplate {
    // Qwen3.5 special token IDs (248K vocab)
    static let imStartId = 248045       // <|im_start|>
    static let imEndId = 248046         // <|im_end|>
    static let endOfTextId = 248044     // <|endoftext|>
    static let thinkStartId = 248068    // <think>
    static let thinkEndId = 248069      // </think>
    static let newlineId = 198          // \n

    /// Strip thinking block from generated tokens.
    ///
    /// Removes tokens from `<think>` through `</think>` (inclusive)
    /// and any trailing newlines, returning only the response content.
    static func stripThinking(from tokens: [Int]) -> [Int] {
        let thinkTokens: Set<Int> = [thinkStartId, thinkEndId]
        let newlines: Set<Int> = [newlineId, 271]  // 198 = \n, 271 = \n\n

        guard let startIdx = tokens.firstIndex(where: { thinkTokens.contains($0) && $0 == thinkStartId }) else {
            // No <think> — strip any leading </think> + newlines
            // (happens when non-thinking template causes model to echo end-think)
            var i = 0
            while i < tokens.count && (tokens[i] == thinkEndId || newlines.contains(tokens[i])) {
                i += 1
            }
            return i > 0 ? Array(tokens[i...]) : tokens
        }
        if let endIdx = tokens[startIdx...].firstIndex(of: thinkEndId) {
            var afterThink = endIdx + 1
            while afterThink < tokens.count && newlines.contains(tokens[afterThink]) {
                afterThink += 1
            }
            return Array(tokens[0..<startIdx]) + Array(tokens[afterThink...])
        }
        return Array(tokens[0..<startIdx])
    }

    /// Encode a conversation into token IDs using Qwen3.5 chat template.
    ///
    /// - Parameters:
    ///   - config: Model config (for future extensibility)
    ///   - enableThinking: If false, injects empty think block to skip reasoning
    static func encode(
        messages: [ChatMessage],
        tokenizer: ChatTokenizer,
        config: Qwen3ChatConfig? = nil,
        addGenerationPrompt: Bool = true,
        enableThinking: Bool = true
    ) -> [Int] {
        var tokens: [Int] = []

        for message in messages {
            tokens.append(imStartId)
            tokens.append(contentsOf: tokenizer.encode(message.role.rawValue))
            tokens.append(newlineId)
            tokens.append(contentsOf: tokenizer.encode(message.content))
            tokens.append(imEndId)
            tokens.append(newlineId)
        }

        if addGenerationPrompt {
            tokens.append(imStartId)
            tokens.append(contentsOf: tokenizer.encode("assistant"))
            tokens.append(newlineId)

            if !enableThinking {
                let doubleNewline = tokenizer.encode("\n\n")
                tokens.append(thinkStartId)
                tokens.append(contentsOf: doubleNewline)
                tokens.append(thinkEndId)
                tokens.append(contentsOf: doubleNewline)
            }
        }

        return tokens
    }
}
