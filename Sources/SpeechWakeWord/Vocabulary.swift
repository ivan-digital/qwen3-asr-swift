import Foundation

/// BPE vocabulary loaded from icefall's ``tokens.txt``.
///
/// The file is ``<symbol> <id>`` per line; the symbols are BPE pieces using
/// the SentencePiece ``\u{2581}`` prefix (``▁``) for word boundaries. id 0 is
/// always the blank. Shared with the SentencePiece BPE model for tokenizing
/// user-supplied keyword strings via :class:`BPETokenizer`.
public struct KWSVocabulary: Sendable {
    public let idToToken: [Int: String]
    public let tokenToId: [String: Int]
    public let blankId: Int
    public let unkId: Int?

    public init(idToToken: [Int: String], blankId: Int = 0, unkId: Int? = nil) {
        self.idToToken = idToToken
        var inverse = [String: Int]()
        inverse.reserveCapacity(idToToken.count)
        for (id, tok) in idToToken { inverse[tok] = id }
        self.tokenToId = inverse
        self.blankId = blankId
        self.unkId = unkId ?? inverse["<unk>"]
    }

    public var count: Int { idToToken.count }

    /// Parse an icefall ``tokens.txt`` file.
    public static func load(from url: URL, blankId: Int = 0) throws -> KWSVocabulary {
        let text = try String(contentsOf: url, encoding: .utf8)
        var map = [Int: String]()
        for line in text.split(whereSeparator: { $0 == "\n" || $0 == "\r" }) {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            guard !trimmed.isEmpty else { continue }
            let parts = trimmed.split(separator: " ", maxSplits: 1, omittingEmptySubsequences: true)
            guard parts.count == 2, let id = Int(parts[1]) else { continue }
            map[id] = String(parts[0])
        }
        return KWSVocabulary(idToToken: map, blankId: blankId)
    }

    /// Decode a token id list back to a human-readable phrase. Subword pieces
    /// that start with ``▁`` open a new word; others are glued to the previous
    /// piece. Unknown ids are dropped silently, matching ``ParakeetStreamingASR``.
    public func decode(_ ids: [Int]) -> String {
        var out = ""
        for id in ids {
            guard let piece = idToToken[id] else { continue }
            if piece.hasPrefix("\u{2581}") {
                if !out.isEmpty { out += " " }
                out += piece.dropFirst()
            } else {
                out += piece
            }
        }
        return out
    }
}
