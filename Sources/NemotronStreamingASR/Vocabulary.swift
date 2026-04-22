import Foundation
import AudioCommon

/// SentencePiece vocabulary for Nemotron Streaming (1024 tokens + blank).
///
/// Punctuation and capitalization tokens are part of the regular BPE vocab;
/// decode passes them through as-is.
public struct NemotronVocabulary: Sendable {
    private let idToToken: [Int: String]

    public var count: Int { idToToken.count }

    public init(idToToken: [Int: String]) {
        self.idToToken = idToToken
    }

    /// Load vocabulary from vocab.json (format: `{"0": "▁the", "1": "▁a", ...}`).
    public static func load(from url: URL) throws -> NemotronVocabulary {
        let data = try Data(contentsOf: url)
        let raw = try JSONDecoder().decode([String: String].self, from: data)
        var mapping: [Int: String] = [:]
        for (key, value) in raw {
            if let id = Int(key) {
                mapping[id] = value
            }
        }
        return NemotronVocabulary(idToToken: mapping)
    }

    /// Decode token IDs to text. Blank is filtered upstream.
    public func decode(_ tokenIds: [Int]) -> String {
        var text = ""
        for id in tokenIds {
            guard let token = idToToken[id] else { continue }
            text += token
        }
        return text.replacingOccurrences(of: "▁", with: " ").trimmingCharacters(in: .whitespaces)
    }

    /// Decode with per-word confidence scores.
    public func decodeWords(_ tokenIds: [Int], logProbs: [Float]) -> [WordConfidence] {
        guard tokenIds.count == logProbs.count else { return [] }

        var words: [WordConfidence] = []
        var currentWord = ""
        var currentLogProbs: [Float] = []

        for (i, id) in tokenIds.enumerated() {
            guard let token = idToToken[id] else { continue }
            let isWordStart = token.hasPrefix("▁") && !currentWord.isEmpty

            if isWordStart {
                let word = currentWord.replacingOccurrences(of: "▁", with: " ")
                    .trimmingCharacters(in: .whitespaces)
                if !word.isEmpty {
                    let mean = currentLogProbs.reduce(0, +) / Float(currentLogProbs.count)
                    words.append(WordConfidence(word: word, confidence: min(1.0, exp(mean))))
                }
                currentWord = token
                currentLogProbs = [logProbs[i]]
            } else {
                currentWord += token
                currentLogProbs.append(logProbs[i])
            }
        }

        if !currentWord.isEmpty {
            let word = currentWord.replacingOccurrences(of: "▁", with: " ")
                .trimmingCharacters(in: .whitespaces)
            if !word.isEmpty {
                let mean = currentLogProbs.reduce(0, +) / Float(currentLogProbs.count)
                words.append(WordConfidence(word: word, confidence: min(1.0, exp(mean))))
            }
        }

        return words
    }
}
