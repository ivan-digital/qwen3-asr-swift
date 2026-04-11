import Foundation

/// Minimal SentencePiece `.model` (`sentencepiece_model.proto`) reader.
///
/// Extracts the vocabulary list — `(text, score, type)` for every piece —
/// without requiring a protobuf runtime dependency. Modules build their own
/// encode/decode logic on top: this struct only owns the wire-format parse
/// and the raw piece array.
///
/// `sentencepiece_model.proto` excerpt:
/// ```
/// message ModelProto {
///   repeated SentencePiece pieces = 1;  // field 1, length-delimited submsg
///   ...
/// }
/// message SentencePiece {
///   optional string piece = 1;  // field 1, length-delimited string
///   optional float  score = 2;  // field 2, fixed32 (wire type 5)
///   optional Type   type  = 3;  // field 3, varint (wire type 0)
/// }
/// ```
public struct SentencePieceModel: Sendable {

    /// Piece type constants from `sentencepiece_model.proto`. Values not in
    /// this enum are surfaced as `.unknown(rawValue)` so callers can apply
    /// their own special-token handling.
    public enum PieceType: Int32, Sendable {
        case normal      = 1
        case unknown     = 2
        case control     = 3
        case userDefined = 4
        case unused      = 5
        case byte        = 6
    }

    public struct Piece: Sendable, Equatable {
        public let text: String
        public let score: Float
        public let type: Int32

        public init(text: String, score: Float, type: Int32) {
            self.text = text
            self.score = score
            self.type = type
        }

        public var pieceType: PieceType? { PieceType(rawValue: type) }

        public var isControlOrUnknown: Bool {
            type == PieceType.control.rawValue ||
            type == PieceType.unknown.rawValue ||
            type == PieceType.unused.rawValue ||
            type == PieceType.byte.rawValue
        }
    }

    public let pieces: [Piece]

    public var count: Int { pieces.count }

    public subscript(_ id: Int) -> Piece? {
        guard id >= 0, id < pieces.count else { return nil }
        return pieces[id]
    }

    public init(contentsOf url: URL) throws {
        let data = try Data(contentsOf: url)
        try self.init(data: data)
    }

    public init(modelPath: String) throws {
        try self.init(contentsOf: URL(fileURLWithPath: modelPath))
    }

    public init(data: Data) throws {
        var parsed: [Piece] = []
        var offset = 0

        while offset < data.count {
            let (fieldNumber, wireType, afterTag) = Self.readTag(data: data, offset: offset)
            offset = afterTag

            // Top-level field 1 = repeated SentencePiece, length-delimited (wire 2)
            guard fieldNumber == 1, wireType == 2 else {
                offset = Self.skipField(data: data, offset: offset, wireType: wireType)
                continue
            }

            let (length, afterLen) = Self.readVarint(data: data, offset: afterTag)
            offset = afterLen
            let end = offset + length

            var piece = ""
            var score: Float = 0
            var type: Int32 = PieceType.normal.rawValue

            var sub = offset
            while sub < end {
                let (subField, subWire, afterSubTag) = Self.readTag(data: data, offset: sub)
                sub = afterSubTag
                switch (subField, subWire) {
                case (1, 2):  // piece string
                    let (strLen, afterStrLen) = Self.readVarint(data: data, offset: sub)
                    sub = afterStrLen
                    if let s = String(data: data[sub..<(sub + strLen)], encoding: .utf8) {
                        piece = s
                    }
                    sub += strLen
                case (2, 5):  // score (fixed32 / wire type 5)
                    score = data[sub..<(sub + 4)].withUnsafeBytes { $0.loadUnaligned(as: Float.self) }
                    sub += 4
                case (3, 0):  // type varint
                    let (typeValue, afterType) = Self.readVarint(data: data, offset: sub)
                    sub = afterType
                    type = Int32(typeValue)
                default:
                    sub = Self.skipField(data: data, offset: sub, wireType: subWire)
                }
            }

            parsed.append(Piece(text: piece, score: score, type: type))
            offset = end
        }

        guard !parsed.isEmpty else {
            throw SentencePieceModelError.emptyModel
        }
        self.pieces = parsed
    }

    // MARK: - Protobuf wire helpers

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
        case 0:
            let (_, newOffset) = readVarint(data: data, offset: offset)
            return newOffset
        case 1:
            return offset + 8
        case 2:
            let (length, newOffset) = readVarint(data: data, offset: offset)
            return newOffset + length
        case 5:
            return offset + 4
        default:
            return data.count
        }
    }
}

public enum SentencePieceModelError: Error, CustomStringConvertible {
    case emptyModel
    case invalidFile(URL)

    public var description: String {
        switch self {
        case .emptyModel:
            return "SentencePiece model contained no pieces"
        case .invalidFile(let url):
            return "Could not read SentencePiece model at \(url.path)"
        }
    }
}
