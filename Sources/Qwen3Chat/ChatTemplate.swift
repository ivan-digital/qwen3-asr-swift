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
    static let qwen35ImStartId = 248045     // <|im_start|>
    static let qwen35ImEndId = 248046       // <|im_end|>
    static let qwen35EndOfTextId = 248044   // <|endoftext|>
    static let qwen35ThinkStartId = 248068  // <think>
    static let qwen35ThinkEndId = 248069    // </think>

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
    /// and any trailing newlines, returning only the response content.
    /// Handles both Qwen3 and Qwen3.5 token IDs.
    static func stripThinking(from tokens: [Int]) -> [Int] {
        let thinkStarts: Set<Int> = [thinkStartId, qwen35ThinkStartId]
        let thinkEnds: Set<Int> = [thinkEndId, qwen35ThinkEndId]
        let newlines: Set<Int> = [newlineId, 271]  // 198 = \n, 271 = \n\n

        guard let startIdx = tokens.firstIndex(where: { thinkStarts.contains($0) }) else {
            // No <think> — still strip any leading </think> + newlines
            // (happens when non-thinking template causes model to echo end-think)
            var i = 0
            while i < tokens.count && (thinkEnds.contains(tokens[i]) || newlines.contains(tokens[i])) {
                i += 1
            }
            return i > 0 ? Array(tokens[i...]) : tokens
        }
        if let endIdx = tokens.firstIndex(where: { thinkEnds.contains($0) }) {
            var afterThink = endIdx + 1
            while afterThink < tokens.count && newlines.contains(tokens[afterThink]) {
                afterThink += 1
            }
            return Array(tokens[0..<startIdx]) + Array(tokens[afterThink...])
        }
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

            // Inject empty think block to skip reasoning.
            // Must tokenize "\n\n" to get correct BPE token (271)
            // rather than two separate newlines (198, 198).
            if !enableThinking {
                let doubleNewline = tokenizer.encode("\n\n")
                tokens.append(ids.thinkStart)
                tokens.append(contentsOf: doubleNewline)
                tokens.append(ids.thinkEnd)
                tokens.append(contentsOf: doubleNewline)
            }
        }

        return tokens
    }
}
