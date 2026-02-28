import Foundation

/// SentencePiece-based vocabulary for Parakeet TDT.
///
/// Loads a `vocab.json` mapping from token ID strings to token strings,
/// and decodes sequences using SentencePiece conventions (`▁` → space).
public struct ParakeetVocabulary: Sendable {
    /// Mapping from token ID to token string.
    private let idToToken: [Int: String]

    /// Number of tokens in the vocabulary (excluding blank).
    public var count: Int { idToToken.count }

    /// Initialize from a pre-loaded dictionary.
    public init(idToToken: [Int: String]) {
        self.idToToken = idToToken
    }

    /// Load vocabulary from a `vocab.json` file.
    ///
    /// Expected format: `{"0": "▁the", "1": "▁a", ...}`
    public static func load(from url: URL) throws -> ParakeetVocabulary {
        let data = try Data(contentsOf: url)
        let raw = try JSONDecoder().decode([String: String].self, from: data)

        var mapping = [Int: String]()
        mapping.reserveCapacity(raw.count)
        for (key, value) in raw {
            guard let id = Int(key) else { continue }
            mapping[id] = value
        }

        return ParakeetVocabulary(idToToken: mapping)
    }

    /// Decode a sequence of token IDs into text.
    ///
    /// Applies SentencePiece conventions:
    /// - `▁` (U+2581) at the start of a token becomes a space
    /// - Leading space on the final result is trimmed
    public func decode(_ tokenIds: [Int]) -> String {
        var pieces = [String]()
        pieces.reserveCapacity(tokenIds.count)

        for id in tokenIds {
            guard let token = idToToken[id] else { continue }
            // Replace SentencePiece word-boundary marker with space
            let text = token.replacingOccurrences(of: "\u{2581}", with: " ")
            pieces.append(text)
        }

        let joined = pieces.joined()
        // Trim leading space that comes from the first token's ▁ prefix
        return joined.trimmingCharacters(in: .whitespaces)
    }
}
