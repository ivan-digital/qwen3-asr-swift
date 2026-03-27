import Foundation

/// Qwen3 chat message.
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

/// Formats messages into Qwen3/3.5 chat template tokens.
///
/// Both Qwen3 and Qwen3.5 use the same `<|im_start|>/<|im_end|>` format,
/// but with different token IDs due to different vocab sizes.
///
/// ```
/// <|im_start|>system
/// {system_message}<|im_end|>
/// <|im_start|>user
/// {user_message}<|im_end|>
/// <|im_start|>assistant
/// ```
enum ChatTemplate {
    // Token IDs differ between Qwen3 (152K vocab) and Qwen3.5 (248K vocab).
    // These are Qwen3 defaults — overridden by config.eosTokenId at runtime.
    static let endOfTextId = 151643   // <|endoftext|>
    static let imStartId = 151644
    static let imEndId = 151645
    static let newlineId = 198
    static let thinkStartId = 151667  // <think>
    static let thinkEndId = 151668    // </think>

    // Qwen3.5 special token IDs (248K vocab)
    static let qwen35ImStartId = 248042
    static let qwen35ImEndId = 248043
    static let qwen35EndOfTextId = 248044
    static let qwen35ThinkStartId = 248065  // <think>
    static let qwen35ThinkEndId = 248066    // </think>

    /// Get the correct special token IDs for a given config.
    static func tokenIds(for config: Qwen3ChatConfig) -> (
        imStart: Int, imEnd: Int, endOfText: Int,
        thinkStart: Int, thinkEnd: Int, newline: Int
    ) {
        if config.isQwen35 {
            return (qwen35ImStartId, qwen35ImEndId, qwen35EndOfTextId,
                    qwen35ThinkStartId, qwen35ThinkEndId, newlineId)
        }
        return (imStartId, imEndId, endOfTextId,
                thinkStartId, thinkEndId, newlineId)
    }

    /// Strip thinking block from generated tokens.
    ///
    /// Removes tokens from `<think>` through `</think>` (inclusive)
    /// and any trailing newline, returning only the response content.
    static func stripThinking(from tokens: [Int]) -> [Int] {
        guard let startIdx = tokens.firstIndex(of: thinkStartId) else {
            return tokens
        }
        if let endIdx = tokens.firstIndex(of: thinkEndId) {
            // Skip </think> and optional trailing newline
            var afterThink = endIdx + 1
            if afterThink < tokens.count && tokens[afterThink] == newlineId {
                afterThink += 1
            }
            return Array(tokens[0..<startIdx]) + Array(tokens[afterThink...])
        }
        // No </think> found — strip everything from <think> onwards
        return Array(tokens[0..<startIdx])
    }

    /// Encode a conversation into token IDs using Qwen3/3.5 chat template.
    ///
    /// - Parameters:
    ///   - config: Model config (determines token IDs for Qwen3 vs 3.5)
    ///   - enableThinking: If false, injects empty think block to skip reasoning
    static func encode(
        messages: [ChatMessage],
        tokenizer: ChatTokenizer,
        config: Qwen3ChatConfig? = nil,
        addGenerationPrompt: Bool = true,
        enableThinking: Bool = true
    ) -> [Int] {
        let ids = config.map { tokenIds(for: $0) } ??
            (imStart: imStartId, imEnd: imEndId, endOfText: endOfTextId,
             thinkStart: thinkStartId, thinkEnd: thinkEndId, newline: newlineId)

        var tokens: [Int] = []

        for message in messages {
            // <|im_start|>role\n
            tokens.append(ids.imStart)
            tokens.append(contentsOf: tokenizer.encode(message.role.rawValue))
            tokens.append(ids.newline)

            // content<|im_end|>\n
            tokens.append(contentsOf: tokenizer.encode(message.content))
            tokens.append(ids.imEnd)
            tokens.append(ids.newline)
        }

        // Add generation prompt for assistant response
        if addGenerationPrompt {
            tokens.append(ids.imStart)
            tokens.append(contentsOf: tokenizer.encode("assistant"))
            tokens.append(ids.newline)

            // Inject empty think block to skip reasoning
            if !enableThinking {
                tokens.append(ids.thinkStart)
                tokens.append(ids.newline)
                tokens.append(ids.newline)
                tokens.append(ids.thinkEnd)
                tokens.append(ids.newline)
                tokens.append(ids.newline)
            }
        }

        return tokens
    }
}
