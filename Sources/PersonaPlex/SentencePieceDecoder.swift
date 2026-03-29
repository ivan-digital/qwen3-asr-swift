import Foundation

/// Minimal SentencePiece .model file reader that extracts the vocabulary
/// for decoding token IDs back to text. Parses the protobuf wire format
/// directly without requiring protobuf dependencies.
///
/// SentencePiece model format (sentencepiece_model.proto):
///   ModelProto {
///     repeated SentencePiece pieces = 1;  // vocabulary
///     ...
///   }
///   SentencePiece {
///     string piece = 1;
///     float score = 2;
///     Type type = 3;  // NORMAL=1, UNKNOWN=2, CONTROL=3, ...
///   }
public struct SentencePieceDecoder: Sendable {
    private let vocab: [Int: String]
    private let pieceToId: [String: Int]
    private let scores: [Int: Float]

    public init(modelPath: String) throws {
        let data = try Data(contentsOf: URL(fileURLWithPath: modelPath))
        var pieces: [String] = []
        var pieceScores: [Float] = []
        var offset = 0

        while offset < data.count {
            let (fieldNumber, wireType, newOffset) = Self.readTag(data: data, offset: offset)
            offset = newOffset

            if fieldNumber == 1 && wireType == 2 {
                // Length-delimited: SentencePiece submessage
                let (length, dataOffset) = Self.readVarint(data: data, offset: offset)
                offset = dataOffset
                let end = offset + length

                // Parse submessage to extract piece string (field 1) and score (field 2)
                var piece: String?
                var score: Float = 0
                var subOffset = offset
                while subOffset < end {
                    let (subField, subWire, subNewOffset) = Self.readTag(data: data, offset: subOffset)
                    subOffset = subNewOffset

                    if subField == 1 && subWire == 2 {
                        // String field
                        let (strLen, strOffset) = Self.readVarint(data: data, offset: subOffset)
                        subOffset = strOffset
                        if let s = String(data: data[subOffset..<(subOffset + strLen)], encoding: .utf8) {
                            piece = s
                        }
                        subOffset += strLen
                    } else if subField == 2 && subWire == 5 {
                        // Float field (32-bit / wire type 5)
                        score = data[subOffset..<(subOffset + 4)].withUnsafeBytes { $0.load(as: Float.self) }
                        subOffset += 4
                    } else {
                        // Skip other fields
                        subOffset = Self.skipField(data: data, offset: subOffset, wireType: subWire)
                    }
                }

                pieces.append(piece ?? "")
                pieceScores.append(score)
                offset = end
            } else {
                // Skip non-piece fields
                offset = Self.skipField(data: data, offset: offset, wireType: wireType)
            }
        }

        var v: [Int: String] = [:]
        var p2i: [String: Int] = [:]
        var s: [Int: Float] = [:]
        for (i, piece) in pieces.enumerated() {
            v[i] = piece
            p2i[piece] = i
            s[i] = pieceScores[i]
        }
        self.vocab = v
        self.pieceToId = p2i
        self.scores = s
    }

    /// Text padding token ID used by PersonaPlex (generated when model is producing audio but not text)
    private static let textPaddingId: Int32 = 3

    public func decode(_ tokens: [Int32]) -> String {
        var result = ""
        for token in tokens {
            // Skip padding tokens (most steps are padding when model generates audio)
            if token == Self.textPaddingId { continue }
            guard let piece = vocab[Int(token)] else { continue }
            // Skip control tokens (e.g. <s>, </s>, <unk>)
            if piece.hasPrefix("<") && piece.hasSuffix(">") { continue }
            result += piece
        }
        // SentencePiece uses U+2581 (▁) as word boundary / space
        return result.replacingOccurrences(of: "\u{2581}", with: " ").trimmingCharacters(in: .whitespaces)
    }

    /// Encode a string into SentencePiece token IDs using greedy unigram segmentation.
    /// Prepends the SentencePiece word-boundary marker (U+2581) to the input and segments
    /// by choosing the longest matching piece at each position.
    public func encode(_ text: String) -> [Int32] {
        // SentencePiece prepends ▁ (U+2581) to represent leading space / word boundary
        let normalized = "\u{2581}" + text.replacingOccurrences(of: " ", with: "\u{2581}")
        var tokens: [Int32] = []
        var i = normalized.startIndex

        while i < normalized.endIndex {
            var bestLen = 0
            var bestId: Int32 = 0 // <unk>
            // Try longest match first
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
                // Unknown character — emit <unk> and advance one character
                tokens.append(0)
                i = normalized.index(after: i)
            } else {
                tokens.append(bestId)
                i = normalized.index(i, offsetBy: bestLen)
            }
        }
        return tokens
    }

    /// Encode a system prompt string, wrapping it with `<system>` tags as required by PersonaPlex.
    public func encodeSystemPrompt(_ text: String) -> [Int32] {
        return encode("<system> " + text + "<system>")
    }

    // MARK: - Protobuf Wire Format Helpers

    private static func readVarint(data: Data, offset: Int) -> (value: Int, newOffset: Int) {
        var result = 0
        var shift = 0
        var off = offset
        while off < data.count {
            let byte = Int(data[off])
            off += 1
            result |= (byte & 0x7F) << shift
            if byte & 0x80 == 0 { break }
            shift += 7
        }
        return (result, off)
    }

    private static func readTag(data: Data, offset: Int) -> (fieldNumber: Int, wireType: Int, newOffset: Int) {
        let (tag, newOffset) = readVarint(data: data, offset: offset)
        return (tag >> 3, tag & 0x07, newOffset)
    }

    private static func skipField(data: Data, offset: Int, wireType: Int) -> Int {
        switch wireType {
        case 0: // Varint
            let (_, newOffset) = readVarint(data: data, offset: offset)
            return newOffset
        case 1: // 64-bit
            return offset + 8
        case 2: // Length-delimited
            let (length, newOffset) = readVarint(data: data, offset: offset)
            return newOffset + length
        case 5: // 32-bit
            return offset + 4
        default:
            return data.count // Unknown wire type, skip to end
        }
    }
}
