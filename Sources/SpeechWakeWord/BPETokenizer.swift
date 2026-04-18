import Foundation
import AudioCommon

/// Greedy longest-match BPE encoder over an icefall SentencePiece model.
///
/// Matches the behaviour of sentencepiece's unigram greedy encode closely
/// enough for short, well-formed keyword phrases — which is all the KWS
/// detector needs. For long utterances or byte-fallback unicode, use the
/// full sentencepiece decoder pipeline instead. Input is lowercased;
/// word-initial pieces are prefixed with the SentencePiece whitespace
/// marker ``▁`` (U+2581) to match the vocabulary in ``tokens.txt``.
public struct BPETokenizer: Sendable {
    public let pieceToId: [String: Int]
    public let idToPiece: [Int: String]
    public let unkId: Int

    public init(model: SentencePieceModel, unkId: Int = 1) {
        var p2i = [String: Int]()
        var i2p = [Int: String]()
        for (idx, piece) in model.pieces.enumerated() {
            p2i[piece.text] = idx
            i2p[idx] = piece.text
        }
        self.pieceToId = p2i
        self.idToPiece = i2p
        self.unkId = unkId
    }

    /// Encode a phrase like "hey soniqo" into BPE token ids.
    /// Tokens follow SentencePiece conventions: leading ``▁`` marks word-initial pieces.
    public func encode(_ phrase: String) -> [Int] {
        let normalized = phrase.lowercased()
        let words = normalized.split(whereSeparator: { $0.isWhitespace })
        var ids: [Int] = []
        for (wIdx, word) in words.enumerated() {
            // Prepend the whitespace marker to the first character. For every
            // subsequent word the marker also opens the run, since icefall's
            // tokeniser treats spaces as piece-initial.
            var chunk = "\u{2581}" + String(word)
            while !chunk.isEmpty {
                var matched = false
                for length in stride(from: chunk.count, through: 1, by: -1) {
                    let prefix = String(chunk.prefix(length))
                    if let id = pieceToId[prefix] {
                        ids.append(id)
                        chunk.removeFirst(length)
                        matched = true
                        break
                    }
                }
                if !matched {
                    ids.append(unkId)
                    chunk.removeFirst()
                }
            }
            _ = wIdx
        }
        return ids
    }
}
