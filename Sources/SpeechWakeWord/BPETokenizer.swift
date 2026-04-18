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
    /// Normalise input case before encoding. The icefall KWS vocab is
    /// uppercase — set ``.uppercase`` for that model, ``.none`` to preserve
    /// the caller's casing, ``.lowercase`` for all-lowercase vocabularies.
    public let caseHandling: CaseHandling

    public enum CaseHandling: Sendable, Equatable {
        case none
        case uppercase
        case lowercase
    }

    public init(
        model: SentencePieceModel,
        unkId: Int = 2,
        caseHandling: CaseHandling = .uppercase
    ) {
        var p2i = [String: Int]()
        var i2p = [Int: String]()
        for (idx, piece) in model.pieces.enumerated() {
            p2i[piece.text] = idx
            i2p[idx] = piece.text
        }
        self.pieceToId = p2i
        self.idToPiece = i2p
        self.unkId = unkId
        self.caseHandling = caseHandling
    }

    /// Encode a phrase like "HEY SONIQO" into BPE token ids.
    /// Tokens follow SentencePiece conventions: leading ``▁`` marks word-initial pieces.
    public func encode(_ phrase: String) -> [Int] {
        let normalized: String
        switch caseHandling {
        case .uppercase: normalized = phrase.uppercased()
        case .lowercase: normalized = phrase.lowercased()
        case .none: normalized = phrase
        }
        let words = normalized.split(whereSeparator: { $0.isWhitespace })
        var ids: [Int] = []
        for word in words {
            // Greedy longest-match over "▁<word>". When no prefix matches,
            // the lone word-start marker ``▁`` (its own vocab entry in the
            // icefall KWS model) is emitted and we continue with the bare
            // characters — this is what reproduces the reference
            // decompositions in ``sherpa-onnx/test_wavs/test_keywords.txt``.
            var chunk = Array("\u{2581}\(word)".unicodeScalars)
            while !chunk.isEmpty {
                var matched = false
                for length in stride(from: chunk.count, through: 1, by: -1) {
                    let prefix = String(String.UnicodeScalarView(chunk.prefix(length)))
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
        }
        return ids
    }
}
