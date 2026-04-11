import XCTest
@testable import AudioCommon

/// Direct unit tests for the shared `SentencePieceModel` protobuf reader.
/// Builds synthetic `.model` byte streams (no real tokenizer files needed)
/// to verify field decoding for piece text, score, and type, plus the
/// helpers used by ASR / TTS modules.
final class SentencePieceModelTests: XCTestCase {

    // MARK: - Synthetic protobuf builder

    /// Build a minimal SentencePiece model byte stream from `(text, score, type)` tuples.
    private func buildSyntheticModel(pieces: [(String, Float, Int32)]) -> Data {
        var data = Data()
        for (text, score, type) in pieces {
            var sub = Data()

            // field 1 (piece string) — wire type 2
            let bytes = Array(text.utf8)
            sub.append(Self.encodeTag(field: 1, wire: 2))
            sub.append(Self.encodeVarint(bytes.count))
            sub.append(contentsOf: bytes)

            // field 2 (score float) — wire type 5 (32-bit)
            sub.append(Self.encodeTag(field: 2, wire: 5))
            withUnsafeBytes(of: score) { sub.append(contentsOf: $0) }

            // field 3 (type varint) — wire type 0
            sub.append(Self.encodeTag(field: 3, wire: 0))
            sub.append(Self.encodeVarint(Int(type)))

            // Outer: field 1 = pieces, length-delimited
            data.append(Self.encodeTag(field: 1, wire: 2))
            data.append(Self.encodeVarint(sub.count))
            data.append(sub)
        }
        return data
    }

    private static func encodeTag(field: Int, wire: Int) -> Data {
        return encodeVarint((field << 3) | wire)
    }

    private static func encodeVarint(_ value: Int) -> Data {
        var v = value
        var out = Data()
        while true {
            var byte = UInt8(v & 0x7F)
            v >>= 7
            if v != 0 {
                byte |= 0x80
                out.append(byte)
            } else {
                out.append(byte)
                break
            }
        }
        return out
    }

    private func writeTempModel(_ data: Data) throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("spm_test_\(UUID().uuidString).model")
        try data.write(to: url)
        addTeardownBlock { try? FileManager.default.removeItem(at: url) }
        return url
    }

    // MARK: - Tests

    func testParsesAllFields() throws {
        let pieces: [(String, Float, Int32)] = [
            ("<s>",   0.0, 3),  // CONTROL
            ("<pad>", 0.0, 3),
            ("</s>",  0.0, 3),
            ("<unk>", 0.0, 2),  // UNKNOWN
            ("\u{2581}hello", -1.5, 1),  // NORMAL
            ("\u{2581}world", -2.0, 1),
            ("!",     -3.0, 1),
        ]
        let url = try writeTempModel(buildSyntheticModel(pieces: pieces))
        let model = try SentencePieceModel(contentsOf: url)

        XCTAssertEqual(model.count, 7)
        XCTAssertEqual(model[0]?.text, "<s>")
        XCTAssertEqual(model[0]?.type, 3)
        XCTAssertEqual(model[4]?.text, "\u{2581}hello")
        XCTAssertEqual(model[4]?.score ?? 0, Float(-1.5), accuracy: Float(1e-6))
        XCTAssertEqual(model[4]?.type, 1)
        XCTAssertEqual(model[6]?.text, "!")
    }

    func testPieceTypeEnumDecode() throws {
        let pieces: [(String, Float, Int32)] = [
            ("a", 0, 1),  // normal
            ("b", 0, 2),  // unknown
            ("c", 0, 3),  // control
            ("d", 0, 4),  // userDefined
            ("e", 0, 5),  // unused
            ("f", 0, 6),  // byte
        ]
        let url = try writeTempModel(buildSyntheticModel(pieces: pieces))
        let model = try SentencePieceModel(contentsOf: url)

        XCTAssertEqual(model[0]?.pieceType, .normal)
        XCTAssertEqual(model[1]?.pieceType, .unknown)
        XCTAssertEqual(model[2]?.pieceType, .control)
        XCTAssertEqual(model[3]?.pieceType, .userDefined)
        XCTAssertEqual(model[4]?.pieceType, .unused)
        XCTAssertEqual(model[5]?.pieceType, .byte)
    }

    func testIsControlOrUnknown() throws {
        let pieces: [(String, Float, Int32)] = [
            ("a", 0, 1),  // normal — false
            ("b", 0, 2),  // unknown — true
            ("c", 0, 3),  // control — true
            ("d", 0, 4),  // userDefined — false
            ("e", 0, 5),  // unused — true
            ("f", 0, 6),  // byte — true
        ]
        let url = try writeTempModel(buildSyntheticModel(pieces: pieces))
        let model = try SentencePieceModel(contentsOf: url)

        XCTAssertFalse(model[0]!.isControlOrUnknown)
        XCTAssertTrue(model[1]!.isControlOrUnknown)
        XCTAssertTrue(model[2]!.isControlOrUnknown)
        XCTAssertFalse(model[3]!.isControlOrUnknown)
        XCTAssertTrue(model[4]!.isControlOrUnknown)
        XCTAssertTrue(model[5]!.isControlOrUnknown)
    }

    func testSubscriptOutOfRangeReturnsNil() throws {
        let pieces: [(String, Float, Int32)] = [("a", 0, 1), ("b", 0, 1)]
        let url = try writeTempModel(buildSyntheticModel(pieces: pieces))
        let model = try SentencePieceModel(contentsOf: url)
        XCTAssertNil(model[-1])
        XCTAssertNil(model[2])
        XCTAssertNil(model[10288])
        XCTAssertNotNil(model[0])
        XCTAssertNotNil(model[1])
    }

    func testEmptyFileThrows() throws {
        let url = try writeTempModel(Data())
        XCTAssertThrowsError(try SentencePieceModel(contentsOf: url)) { error in
            guard let spmError = error as? SentencePieceModelError else {
                XCTFail("Expected SentencePieceModelError, got \(error)")
                return
            }
            if case .emptyModel = spmError {
                // ok
            } else {
                XCTFail("Expected .emptyModel, got \(spmError)")
            }
        }
    }

    func testInitFromData() throws {
        let pieces: [(String, Float, Int32)] = [("hello", -1, 1), ("world", -1, 1)]
        let data = buildSyntheticModel(pieces: pieces)
        let model = try SentencePieceModel(data: data)
        XCTAssertEqual(model.count, 2)
        XCTAssertEqual(model[0]?.text, "hello")
        XCTAssertEqual(model[1]?.text, "world")
    }

    func testSkipsUnknownTopLevelFields() throws {
        // Build a model where field 1 (pieces) is interleaved with bogus
        // top-level fields that the parser must skip without crashing.
        var data = Data()
        // Bogus field 5 (varint) at the start
        data.append(Self.encodeTag(field: 5, wire: 0))
        data.append(Self.encodeVarint(42))

        // Real piece
        var sub = Data()
        let bytes = Array("foo".utf8)
        sub.append(Self.encodeTag(field: 1, wire: 2))
        sub.append(Self.encodeVarint(bytes.count))
        sub.append(contentsOf: bytes)
        sub.append(Self.encodeTag(field: 3, wire: 0))
        sub.append(Self.encodeVarint(1))
        data.append(Self.encodeTag(field: 1, wire: 2))
        data.append(Self.encodeVarint(sub.count))
        data.append(sub)

        // Bogus field 7 (length-delimited) at the end
        data.append(Self.encodeTag(field: 7, wire: 2))
        data.append(Self.encodeVarint(3))
        data.append(contentsOf: [0xAA, 0xBB, 0xCC])

        let model = try SentencePieceModel(data: data)
        XCTAssertEqual(model.count, 1)
        XCTAssertEqual(model[0]?.text, "foo")
    }
}
