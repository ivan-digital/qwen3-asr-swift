import Foundation

// MARK: - Errors

public enum TokenizerError: Error, LocalizedError {
    case invalidFormat(String)

    public var errorDescription: String? {
        switch self {
        case .invalidFormat(let reason):
            return "Invalid tokenizer format: \(reason)"
        }
    }
}

/// Simple tokenizer for Qwen3-ASR that loads from vocab.json
public class Qwen3Tokenizer {
    private var idToToken: [Int: String] = [:]
    private var tokenToId: [String: Int] = [:]

    public var eosTokenId: Int = 151643
    public var padTokenId: Int = 151643
    public var bosTokenId: Int = 151644

    public init() {}

    /// Load tokenizer from vocab.json file (direct token->id mapping)
    public func load(from url: URL) throws {
        let data = try Data(contentsOf: url)

        // vocab.json is a direct {token: id} mapping
        guard let vocab = try JSONSerialization.jsonObject(with: data) as? [String: Int] else {
            throw TokenizerError.invalidFormat("Expected {token: id} dictionary")
        }

        for (token, id) in vocab {
            idToToken[id] = token
            tokenToId[token] = id
        }

        // Also load added tokens from tokenizer_config.json if it exists
        let configUrl = url.deletingLastPathComponent().appendingPathComponent("tokenizer_config.json")
        if FileManager.default.fileExists(atPath: configUrl.path) {
            try loadAddedTokens(from: configUrl)
        }

        print("Loaded tokenizer with \(idToToken.count) tokens")
    }

    /// Load added tokens from tokenizer_config.json
    private func loadAddedTokens(from url: URL) throws {
        let data = try Data(contentsOf: url)

        guard let config = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return // Not a valid config, skip
        }

        // added_tokens_decoder is a dict with string keys (token IDs) and object values with "content" field
        if let addedTokens = config["added_tokens_decoder"] as? [String: [String: Any]] {
            var addedCount = 0
            for (idString, tokenInfo) in addedTokens {
                guard let id = Int(idString),
                      let content = tokenInfo["content"] as? String else {
                    continue
                }

                // Add to our mappings (overwrite if exists)
                idToToken[id] = content
                tokenToId[content] = id
                addedCount += 1
            }
            print("Loaded \(addedCount) added tokens from tokenizer_config.json")
        }
    }

    /// Decode token IDs to text
    public func decode(tokens: [Int]) -> String {
        var result = ""

        for tokenId in tokens {
            if let token = idToToken[tokenId] {
                // Handle special tokens - skip most but keep some for parsing
                if token.hasPrefix("<|") && token.hasSuffix("|>") {
                    // Skip special tokens like <|endoftext|>, <|im_start|>, etc.
                    continue
                }

                // Keep <asr_text> and similar markers that don't have |> suffix
                // These are needed for output parsing
                if token.hasPrefix("<") && token.hasSuffix(">") && !token.contains("|") {
                    result += token
                    continue
                }

                // Handle byte-level tokens (Ġ prefix means space)
                var decodedToken = token
                if decodedToken.hasPrefix("Ġ") {
                    decodedToken = " " + String(decodedToken.dropFirst())
                }

                // Handle other byte-level encodings
                decodedToken = decodeByteLevelToken(decodedToken)

                result += decodedToken
            }
        }

        return result.trimmingCharacters(in: .whitespaces)
    }

    /// Byte-to-unicode mapping table (GPT-2 style)
    /// Built lazily on first use
    private static var byteToUnicode: [UInt8: Character] = {
        var mapping: [UInt8: Character] = [:]
        var n = 0

        // Printable ASCII and some extended chars map directly
        let ranges: [(ClosedRange<UInt8>)] = [
            (UInt8(ascii: "!")...UInt8(ascii: "~")),  // 33-126
            (0xA1...0xAC),  // 161-172
            (0xAE...0xFF),  // 174-255
        ]

        for range in ranges {
            for b in range {
                mapping[b] = Character(UnicodeScalar(b))
            }
        }

        // Remaining bytes (0-32, 127-160, 173) map to U+0100 onwards
        for b: UInt8 in 0...255 {
            if mapping[b] == nil {
                mapping[b] = Character(UnicodeScalar(0x100 + n)!)
                n += 1
            }
        }

        return mapping
    }()

    /// Unicode-to-byte reverse mapping
    private static var unicodeToByte: [Character: UInt8] = {
        var reverse: [Character: UInt8] = [:]
        for (byte, char) in byteToUnicode {
            reverse[char] = byte
        }
        return reverse
    }()

    /// Decode byte-level BPE token to proper UTF-8 string
    private func decodeByteLevelToken(_ token: String) -> String {
        var bytes: [UInt8] = []

        for char in token {
            if let byte = Self.unicodeToByte[char] {
                bytes.append(byte)
            } else {
                // Character not in mapping - keep as UTF-8
                bytes.append(contentsOf: String(char).utf8)
            }
        }

        // Decode bytes as UTF-8
        if let decoded = String(bytes: bytes, encoding: .utf8) {
            return decoded
        } else {
            // Fallback to original if decoding fails
            return token
        }
    }

    /// Encode text to token IDs (for completeness)
    public func encode(_ text: String) -> [Int] {
        // Simple character-level encoding for now
        // Full BPE encoding would be more complex
        var tokens: [Int] = []

        for char in text {
            if let id = tokenToId[String(char)] {
                tokens.append(id)
            }
        }

        return tokens
    }

    /// Get token ID for a specific token string
    public func getTokenId(for token: String) -> Int? {
        return tokenToId[token]
    }

    /// Get token string for a specific ID
    public func getToken(for id: Int) -> String? {
        return idToToken[id]
    }

    /// Debug: print token mappings for common words
    public func debugTokenMappings() {
        let commonTokens = [
            "<|im_start|>", "<|im_end|>", "<|audio_start|>", "<|audio_end|>",
            "<|audio_pad|>", "<asr_text>", "<|endoftext|>",
            "system", "user", "assistant", "language", "English",
            "Ġsystem", "Ġuser", "Ġassistant", "Ġlanguage", "ĠEnglish",
            "\n", "Ċ"  // newline representations
        ]

        print("Token ID mappings:")
        for token in commonTokens {
            if let id = tokenToId[token] {
                print("  '\(token)' -> \(id)")
            } else {
                print("  '\(token)' -> NOT FOUND")
            }
        }
    }
}

/// Protocol for tokenizer to allow different implementations
public protocol TokenizerProtocol {
    func decode(tokens: [Int]) -> String
    func encode(_ text: String) -> [Int]
}

extension Qwen3Tokenizer: TokenizerProtocol {}
