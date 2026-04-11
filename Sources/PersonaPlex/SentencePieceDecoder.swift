import Foundation
import AudioCommon

/// PersonaPlex SentencePiece encoder/decoder built on top of
/// `AudioCommon.SentencePieceModel`. Owns the PersonaPlex-specific control
/// token handling (text padding id 3, `<...>` control-piece skipping) plus
/// the greedy unigram encoder used to tokenise system prompts.
public struct SentencePieceDecoder: Sendable {
    private let model: SentencePieceModel
    private let pieceToId: [String: Int]

    public init(modelPath: String) throws {
        self.model = try SentencePieceModel(modelPath: modelPath)
        var p2i: [String: Int] = [:]
        p2i.reserveCapacity(model.count)
        for i in 0..<model.count {
            if let piece = model[i] {
                p2i[piece.text] = i
            }
        }
        self.pieceToId = p2i
    }

    /// Text padding token ID used by PersonaPlex (generated when model is
    /// producing audio but not text)
    private static let textPaddingId: Int32 = 3

    public func decode(_ tokens: [Int32]) -> String {
        var result = ""
        for token in tokens {
            if token == Self.textPaddingId { continue }
            guard let piece = model[Int(token)] else { continue }
            // Skip control tokens (e.g. <s>, </s>, <unk>)
            if piece.text.hasPrefix("<") && piece.text.hasSuffix(">") { continue }
            result += piece.text
        }
        return result
            .replacingOccurrences(of: "\u{2581}", with: " ")
            .trimmingCharacters(in: .whitespaces)
    }

    /// Encode a string into SentencePiece token IDs using greedy unigram
    /// segmentation. Prepends the SentencePiece word-boundary marker (U+2581)
    /// to the input and segments by choosing the longest matching piece at
    /// each position.
    public func encode(_ text: String) -> [Int32] {
        let normalized = "\u{2581}" + text.replacingOccurrences(of: " ", with: "\u{2581}")
        var tokens: [Int32] = []
        var i = normalized.startIndex

        while i < normalized.endIndex {
            var bestLen = 0
            var bestId: Int32 = 0  // <unk>
            for len in stride(from: min(32, normalized.distance(from: i, to: normalized.endIndex)), through: 1, by: -1) {
                let end = normalized.index(i, offsetBy: len)
                let candidate = String(normalized[i..<end])
                if let id = pieceToId[candidate] {
                    bestLen = len
                    bestId = Int32(id)
                    break
                }
            }
            if bestLen == 0 {
                tokens.append(0)
                i = normalized.index(after: i)
            } else {
                tokens.append(bestId)
                i = normalized.index(i, offsetBy: bestLen)
            }
        }
        return tokens
    }

    /// Encode a system prompt string, wrapping it with `<system>` tags as
    /// required by PersonaPlex.
    public func encodeSystemPrompt(_ text: String) -> [Int32] {
        return encode("<system> " + text + "<system>")
    }
}
