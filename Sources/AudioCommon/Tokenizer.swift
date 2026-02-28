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

/// Simple tokenizer for Qwen3 that loads from vocab.json
/// Supports decoding (id->text) and basic BPE encoding (text->ids) via merges.txt
public class Qwen3Tokenizer {
    private var idToToken: [Int: String] = [:]
    private var tokenToId: [String: Int] = [:]
    private var bpeMerges: [(String, String)] = []
    private var bpeMergeRanks: [String: Int] = [:]

    public var eosTokenId: Int = 151643
    public var padTokenId: Int = 151643
    public var bosTokenId: Int = 151644

    public init() {}

    /// Test-only initializer with pre-built token mappings
    internal init(idToToken: [Int: String]) {
        self.idToToken = idToToken
        for (id, token) in idToToken { tokenToId[token] = id }
    }

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

        // Load BPE merges if available
        let mergesUrl = url.deletingLastPathComponent().appendingPathComponent("merges.txt")
        if FileManager.default.fileExists(atPath: mergesUrl.path) {
            try loadMerges(from: mergesUrl)
        }

        print("Loaded tokenizer with \(idToToken.count) tokens, \(bpeMerges.count) merges")
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

    /// Load BPE merge rules from merges.txt
    private func loadMerges(from url: URL) throws {
        let content = try String(contentsOf: url, encoding: .utf8)
        let lines = content.components(separatedBy: .newlines)

        for (index, line) in lines.enumerated() {
            // Skip header line and empty lines
            if line.hasPrefix("#") || line.isEmpty { continue }

            let parts = line.components(separatedBy: " ")
            guard parts.count == 2 else { continue }

            bpeMerges.append((parts[0], parts[1]))
            bpeMergeRanks["\(parts[0]) \(parts[1])"] = index
        }
    }

    /// Decode token IDs to text using a unified byte buffer.
    /// Collects all bytes before converting to UTF-8, so multi-byte characters
    /// split across BPE tokens (e.g. CJK) decode correctly.
    public func decode(tokens: [Int]) -> String {
        var buffer: [UInt8] = []

        for tokenId in tokens {
            guard let token = idToToken[tokenId] else { continue }

            // Skip <|...|> special tokens
            if token.hasPrefix("<|") && token.hasSuffix("|>") {
                continue
            }

            // Keep <asr_text> and similar markers — append their UTF-8 bytes
            if token.hasPrefix("<") && token.hasSuffix(">") && !token.contains("|") {
                buffer.append(contentsOf: Array(token.utf8))
                continue
            }

            // Convert each char via unicodeToByte (Ġ→0x20 space is handled
            // automatically since unicodeToByte maps Ġ (U+0120) → byte 32)
            for char in token {
                if let byte = Self.unicodeToByte[char] {
                    buffer.append(byte)
                } else {
                    buffer.append(contentsOf: String(char).utf8)
                }
            }
        }

        let text = String(bytes: buffer, encoding: .utf8)
            ?? String(decoding: buffer, as: UTF8.self)
        return text.trimmingCharacters(in: .whitespaces)
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

    /// Encode a byte-level BPE token string from raw text bytes
    private func encodeByteLevelToken(_ text: String) -> String {
        var result = ""
        for byte in text.utf8 {
            if let char = Self.byteToUnicode[byte] {
                result.append(char)
            }
        }
        return result
    }

    /// BPE encode text to token IDs
    public func encode(_ text: String) -> [Int] {
        guard !bpeMerges.isEmpty else {
            // Fallback: character-level encoding
            return characterEncode(text)
        }

        // Split text into words (whitespace-aware, GPT-2 style pre-tokenization)
        // Simple approach: split on word boundaries, preserving leading spaces as Ġ
        let words = preTokenize(text)

        var tokens: [Int] = []
        for word in words {
            // Convert word to byte-level BPE representation
            let bpeTokens = bpe(word)
            for bpeToken in bpeTokens {
                if let id = tokenToId[bpeToken] {
                    tokens.append(id)
                }
            }
        }

        return tokens
    }

    /// Pre-tokenize text into words (GPT-2 style)
    private func preTokenize(_ text: String) -> [String] {
        // Split on whitespace boundaries while preserving leading spaces as part of the next word
        var words: [String] = []
        var current = ""

        for char in text {
            if char == " " || char == "\n" || char == "\t" {
                if !current.isEmpty {
                    words.append(encodeByteLevelToken(current))
                    current = ""
                }
                current = String(char)
            } else {
                current.append(char)
            }
        }
        if !current.isEmpty {
            words.append(encodeByteLevelToken(current))
        }

        return words
    }

    /// Apply BPE merges to a word
    private func bpe(_ word: String) -> [String] {
        var pieces = word.map { String($0) }

        while pieces.count > 1 {
            // Find the pair with lowest merge rank
            var bestPair: (String, String)?
            var bestRank = Int.max

            for i in 0..<(pieces.count - 1) {
                let pair = "\(pieces[i]) \(pieces[i + 1])"
                if let rank = bpeMergeRanks[pair], rank < bestRank {
                    bestRank = rank
                    bestPair = (pieces[i], pieces[i + 1])
                }
            }

            guard let (first, second) = bestPair else { break }

            // Merge the pair
            var newPieces: [String] = []
            var i = 0
            while i < pieces.count {
                if i < pieces.count - 1 && pieces[i] == first && pieces[i + 1] == second {
                    newPieces.append(first + second)
                    i += 2
                } else {
                    newPieces.append(pieces[i])
                    i += 1
                }
            }
            pieces = newPieces
        }

        return pieces
    }

    /// Simple character-level encoding fallback
    private func characterEncode(_ text: String) -> [Int] {
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
