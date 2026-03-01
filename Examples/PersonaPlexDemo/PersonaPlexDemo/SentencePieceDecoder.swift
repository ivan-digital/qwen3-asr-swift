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
struct SentencePieceDecoder {
    private let vocab: [Int: String]

    init(modelPath: String) throws {
        let data = try Data(contentsOf: URL(fileURLWithPath: modelPath))
        var pieces: [String] = []
        var offset = 0

        while offset < data.count {
            let (fieldNumber, wireType, newOffset) = Self.readTag(data: data, offset: offset)
            offset = newOffset

            if fieldNumber == 1 && wireType == 2 {
                // Length-delimited: SentencePiece submessage
                let (length, dataOffset) = Self.readVarint(data: data, offset: offset)
                offset = dataOffset
                let end = offset + length

                // Parse submessage to extract piece string (field 1)
                var piece: String?
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
                    } else {
                        // Skip other fields
                        subOffset = Self.skipField(data: data, offset: subOffset, wireType: subWire)
                    }
                }

                pieces.append(piece ?? "")
                offset = end
            } else {
                // Skip non-piece fields
                offset = Self.skipField(data: data, offset: offset, wireType: wireType)
            }
        }

        var v: [Int: String] = [:]
        for (i, p) in pieces.enumerated() {
            v[i] = p
        }
        self.vocab = v
    }

    func decode(_ tokens: [Int32]) -> String {
        var result = ""
        for token in tokens {
            guard let piece = vocab[Int(token)] else { continue }
            // Skip control tokens (e.g. <s>, </s>, <unk>)
            if piece.hasPrefix("<") && piece.hasSuffix(">") { continue }
            result += piece
        }
        // SentencePiece uses U+2581 (▁) as word boundary / space
        return result.replacingOccurrences(of: "\u{2581}", with: " ").trimmingCharacters(in: .whitespaces)
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
