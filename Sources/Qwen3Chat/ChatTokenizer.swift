import Foundation

/// Tokenizer for Qwen3 chat models.
///
/// Loads vocabulary from HuggingFace tokenizer files and provides
/// encode/decode functionality for chat text.
public final class ChatTokenizer: @unchecked Sendable {
    private var idToToken: [Int: String] = [:]
    private var tokenToId: [String: Int] = [:]
    private var bpeMerges: [(String, String)] = []
    private var bpeMergeRanks: [String: Int] = [:]
    private var addedTokens: [String: Int] = [:]

    public var eosTokenId: Int = 151645
    public var vocabSize: Int { idToToken.count }

    public init() {}

    /// Load tokenizer from a directory.
    ///
    /// Supports two formats:
    /// 1. `tokenizer.json` (HuggingFace format, preferred) — contains vocab, merges, and added tokens
    /// 2. `vocab.json` + `merges.txt` (legacy) — separate files
    public func load(from directory: URL) throws {
        let tokenizerJsonURL = directory.appendingPathComponent("tokenizer.json")
        let vocabURL = directory.appendingPathComponent("vocab.json")

        if FileManager.default.fileExists(atPath: tokenizerJsonURL.path) {
            try loadFromTokenizerJson(from: tokenizerJsonURL)
        } else {
            try loadVocab(from: vocabURL)

            let mergesURL = directory.appendingPathComponent("merges.txt")
            if FileManager.default.fileExists(atPath: mergesURL.path) {
                try loadMerges(from: mergesURL)
            }
        }

        let configURL = directory.appendingPathComponent("tokenizer_config.json")
        if FileManager.default.fileExists(atPath: configURL.path) {
            try loadAddedTokens(from: configURL)
        }
    }

    /// Load from HuggingFace tokenizer.json (contains vocab + merges + added tokens).
    private func loadFromTokenizerJson(from url: URL) throws {
        let data = try Data(contentsOf: url)
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let model = root["model"] as? [String: Any],
              let vocab = model["vocab"] as? [String: Int] else {
            throw ChatModelError.tokenizerLoadFailed("Invalid tokenizer.json format")
        }

        tokenToId = vocab
        idToToken = Dictionary(uniqueKeysWithValues: vocab.map { ($1, $0) })

        // Load merges
        if let merges = model["merges"] as? [[String]] {
            for (i, pair) in merges.enumerated() {
                guard pair.count == 2 else { continue }
                bpeMerges.append((pair[0], pair[1]))
                bpeMergeRanks["\(pair[0]) \(pair[1])"] = i
            }
        } else if let merges = model["merges"] as? [String] {
            // Alternative format: each merge as "a b" string
            for (i, merge) in merges.enumerated() {
                let parts = merge.split(separator: " ", maxSplits: 1)
                guard parts.count == 2 else { continue }
                let pair = (String(parts[0]), String(parts[1]))
                bpeMerges.append(pair)
                bpeMergeRanks["\(pair.0) \(pair.1)"] = i
            }
        }

        // Load added tokens
        if let addedList = root["added_tokens"] as? [[String: Any]] {
            for entry in addedList {
                guard let content = entry["content"] as? String,
                      let id = entry["id"] as? Int else { continue }
                addedTokens[content] = id
                tokenToId[content] = id
                idToToken[id] = content
            }
        }
    }

    private func loadVocab(from url: URL) throws {
        let data = try Data(contentsOf: url)
        guard let vocab = try JSONSerialization.jsonObject(with: data) as? [String: Int] else {
            throw ChatModelError.tokenizerLoadFailed("Invalid vocab.json format")
        }
        tokenToId = vocab
        idToToken = Dictionary(uniqueKeysWithValues: vocab.map { ($1, $0) })
    }

    private func loadMerges(from url: URL) throws {
        let content = try String(contentsOf: url, encoding: .utf8)
        let lines = content.components(separatedBy: "\n")
        for (i, line) in lines.enumerated() {
            if line.hasPrefix("#") || line.isEmpty { continue }
            let parts = line.split(separator: " ", maxSplits: 1)
            if parts.count == 2 {
                let pair = (String(parts[0]), String(parts[1]))
                bpeMerges.append(pair)
                bpeMergeRanks["\(pair.0) \(pair.1)"] = i
            }
        }
    }

    private func loadAddedTokens(from url: URL) throws {
        let data = try Data(contentsOf: url)
        guard let config = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return
        }
        if let added = config["added_tokens_decoder"] as? [String: Any] {
            for (idStr, value) in added {
                guard let id = Int(idStr),
                      let info = value as? [String: Any],
                      let content = info["content"] as? String else { continue }
                addedTokens[content] = id
                tokenToId[content] = id
                idToToken[id] = content
            }
        }
    }

    // MARK: - Encode

    /// Encode text to token IDs using BPE.
    public func encode(_ text: String) -> [Int] {
        if text.isEmpty { return [] }

        // Check if it's a special/added token
        if let id = addedTokens[text] ?? tokenToId[text] {
            return [id]
        }

        // Simple BPE encoding
        var words = tokenizeToWords(text)
        var allTokens: [Int] = []

        for word in words {
            let wordTokens = bpeEncode(word)
            allTokens.append(contentsOf: wordTokens)
        }

        return allTokens
    }

    /// Split text into BPE-ready words (GPT-2/Qwen style).
    private func tokenizeToWords(_ text: String) -> [String] {
        // Simplified: split on spaces, prefix non-first words with Ġ (space marker)
        var words: [String] = []
        var isFirst = true
        for part in text.components(separatedBy: " ") {
            if part.isEmpty { continue }
            if isFirst {
                words.append(part)
                isFirst = false
            } else {
                words.append("Ġ" + part)
            }
        }
        return words
    }

    /// BPE encode a single word.
    private func bpeEncode(_ word: String) -> [Int] {
        if let id = tokenToId[word] {
            return [id]
        }

        var symbols = word.map { String($0) }
        if symbols.isEmpty { return [] }

        while symbols.count > 1 {
            // Find best merge
            var bestRank = Int.max
            var bestIdx = -1
            for i in 0..<(symbols.count - 1) {
                let pair = "\(symbols[i]) \(symbols[i + 1])"
                if let rank = bpeMergeRanks[pair], rank < bestRank {
                    bestRank = rank
                    bestIdx = i
                }
            }
            if bestIdx < 0 { break }

            // Apply merge
            let merged = symbols[bestIdx] + symbols[bestIdx + 1]
            symbols.replaceSubrange(bestIdx...bestIdx + 1, with: [merged])
        }

        // Look up token IDs
        return symbols.compactMap { tokenToId[$0] }
    }

    // MARK: - Decode

    /// Decode token IDs to text.
    ///
    /// Uses byte-level BPE decoding: token strings are mapped back to bytes
    /// via the GPT-2 byte-to-unicode table, then assembled into UTF-8 text.
    public func decode(_ tokenIds: [Int]) -> String {
        let pieces = tokenIds.compactMap { idToToken[$0] }
        let joined = pieces.joined()
        return decodeBPEString(joined)
    }

    /// Decode a single token ID.
    public func decodeToken(_ tokenId: Int) -> String? {
        guard let piece = idToToken[tokenId] else { return nil }
        return decodeBPEString(piece)
    }

    /// Convert a BPE token string to UTF-8 text.
    ///
    /// GPT-2/Qwen byte-level BPE represents each byte as a specific Unicode
    /// character. This reverses that mapping and decodes the bytes as UTF-8.
    private func decodeBPEString(_ bpeString: String) -> String {
        var bytes: [UInt8] = []
        for char in bpeString {
            if let byte = Self.unicodeToByte[char] {
                bytes.append(byte)
            }
        }
        return String(bytes: bytes, encoding: .utf8) ?? bpeString
    }

    /// GPT-2 byte-to-unicode mapping (reversed for decoding).
    ///
    /// Maps Unicode characters back to the byte values they represent in
    /// GPT-2/Qwen byte-level BPE vocabulary.
    private static let unicodeToByte: [Character: UInt8] = {
        // Build the standard GPT-2 bytes_to_unicode table
        var byteToUnicode: [UInt8: Character] = [:]
        var n = 0

        // Printable ASCII + Latin supplement ranges that map to themselves
        let ranges: [ClosedRange<UInt8>] = [
            33...126,   // ! through ~
            161...172,  // ¡ through ¬
            174...255,  // ® through ÿ
        ]
        for range in ranges {
            for b in range {
                byteToUnicode[b] = Character(Unicode.Scalar(UInt32(b))!)
            }
        }

        // Remaining bytes (0-32, 127-160, 173) map to 256+n
        for b: UInt16 in 0...255 {
            if byteToUnicode[UInt8(b)] == nil {
                byteToUnicode[UInt8(b)] = Character(Unicode.Scalar(256 + UInt32(n))!)
                n += 1
            }
        }

        // Reverse the mapping: unicode char → byte value
        var result: [Character: UInt8] = [:]
        for (byte, char) in byteToUnicode {
            result[char] = byte
        }
        return result
    }()

    /// Check if a token ID is a special token (should not appear in output).
    public func isSpecialToken(_ tokenId: Int) -> Bool {
        guard let token = idToToken[tokenId] else { return false }
        return token.hasPrefix("<|") && token.hasSuffix("|>")
    }
}
