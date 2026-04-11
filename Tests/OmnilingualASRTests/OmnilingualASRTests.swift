import XCTest
@testable import OmnilingualASR

// MARK: - Configuration

final class OmnilingualConfigTests: XCTestCase {

    func testDecodesPublishedConfig10s() throws {
        // Matches aufklarer/Omnilingual-ASR-CTC-300M-CoreML-INT8-10s/config.json
        let json = """
        {
          "model_type": "omnilingual_asr_ctc",
          "format": "coreml",
          "quantization": "palettize-int8",
          "sample_rate": 16000,
          "frame_rate": 50,
          "max_audio_seconds": 10.0,
          "input_samples": 160000,
          "encoder": {"num_layers": 24, "model_dim": 1024, "num_heads": 16},
          "ctc_head": {"vocab_size": 10288},
          "tokenizer": {
            "kind": "sentencepiece",
            "file": "tokenizer.model",
            "bos_idx": 0, "pad_idx": 1, "eos_idx": 2, "unk_idx": 3
          }
        }
        """.data(using: .utf8)!
        let config = try JSONDecoder().decode(OmnilingualConfig.self, from: json)

        XCTAssertEqual(config.sampleRate, 16000)
        XCTAssertEqual(config.frameRate, 50)
        XCTAssertEqual(config.maxAudioSeconds, 10.0)
        XCTAssertEqual(config.inputSamples, 160_000)
        XCTAssertEqual(config.encoder.numLayers, 24)
        XCTAssertEqual(config.encoder.modelDim, 1024)
        XCTAssertEqual(config.encoder.numHeads, 16)
        XCTAssertEqual(config.ctcHead.vocabSize, 10288)
        XCTAssertEqual(config.tokenizer.padIdx, 1)
    }

    func testDecodesPublishedConfig5s() throws {
        let json = """
        {
          "model_type": "omnilingual_asr_ctc",
          "format": "coreml",
          "sample_rate": 16000,
          "frame_rate": 50,
          "max_audio_seconds": 5.0,
          "input_samples": 80000,
          "encoder": {"num_layers": 24, "model_dim": 1024, "num_heads": 16},
          "ctc_head": {"vocab_size": 10288},
          "tokenizer": {
            "kind": "sentencepiece",
            "file": "tokenizer.model",
            "bos_idx": 0, "pad_idx": 1, "eos_idx": 2, "unk_idx": 3
          }
        }
        """.data(using: .utf8)!
        let config = try JSONDecoder().decode(OmnilingualConfig.self, from: json)
        XCTAssertEqual(config.maxAudioSeconds, 5.0)
        XCTAssertEqual(config.inputSamples, 80_000)
    }

    func testDefaultsMatchPublishedShapes() {
        XCTAssertEqual(OmnilingualConfig.default10s.inputSamples, 160_000)
        XCTAssertEqual(OmnilingualConfig.default10s.maxAudioSeconds, 10.0)
        XCTAssertEqual(OmnilingualConfig.default5s.inputSamples, 80_000)
        XCTAssertEqual(OmnilingualConfig.default5s.maxAudioSeconds, 5.0)
        XCTAssertEqual(OmnilingualConfig.default10s.ctcHead.vocabSize, 10288)
    }
}

// MARK: - Layer normalization

final class OmnilingualLayerNormTests: XCTestCase {

    func testZeroInputStaysFinite() {
        let input = [Float](repeating: 0, count: 1024)
        let output = OmnilingualASRModel.layerNormalize(input, eps: 1e-5)
        XCTAssertEqual(output.count, 1024)
        for value in output {
            XCTAssertTrue(value.isFinite, "Layer norm on silence should produce finite values")
            XCTAssertEqual(value, 0, accuracy: 1e-5)
        }
    }

    func testEmptyInputEchoes() {
        XCTAssertTrue(OmnilingualASRModel.layerNormalize([], eps: 1e-5).isEmpty)
    }

    func testMeanAndVarianceOfNormalizedBufferIsZeroAndOne() {
        var input = [Float](repeating: 0, count: 2048)
        for i in 0..<input.count {
            input[i] = sin(Float(i) * 0.1) * 0.3 + 0.1  // nonzero mean, nonzero variance
        }
        let output = OmnilingualASRModel.layerNormalize(input, eps: 1e-8)

        var sum: Float = 0
        var sumSq: Float = 0
        for v in output {
            sum += v
            sumSq += v * v
        }
        let mean = sum / Float(output.count)
        let variance = sumSq / Float(output.count) - mean * mean
        XCTAssertEqual(mean, 0, accuracy: 1e-4, "Normalized buffer should have zero mean")
        XCTAssertEqual(variance, 1, accuracy: 1e-3, "Normalized buffer should have unit variance")
    }

    func testScaleInvariance() {
        var input = [Float](repeating: 0, count: 512)
        for i in 0..<input.count {
            input[i] = Float(i % 16) - 8
        }
        let scaled = input.map { $0 * 10 }
        let a = OmnilingualASRModel.layerNormalize(input, eps: 1e-8)
        let b = OmnilingualASRModel.layerNormalize(scaled, eps: 1e-8)
        for i in 0..<input.count {
            XCTAssertEqual(a[i], b[i], accuracy: 1e-3,
                           "Layer norm is scale-invariant — both buffers should match after normalization")
        }
    }
}

// MARK: - CTC greedy decoder

final class CTCGreedyDecoderTests: XCTestCase {

    /// Build a `[T, V]` row-major logits tensor where frame `t` picks token
    /// `expected[t]` via a large positive value and all others at 0.
    private func makeLogits(frames expected: [Int], vocabSize V: Int) -> [Float] {
        var flat = [Float](repeating: 0, count: expected.count * V)
        for (t, id) in expected.enumerated() {
            flat[t * V + id] = 10.0
        }
        return flat
    }

    func testEmptyInput() {
        let ids = CTCGreedyDecoder.decode(logits: [], timeSteps: 0, vocabSize: 10)
        XCTAssertEqual(ids, [])
    }

    func testArgmaxMatchesExpectedFrames() {
        let expected = [5, 2, 7, 3]
        let logits = makeLogits(frames: expected, vocabSize: 16)
        let ids = CTCGreedyDecoder.decode(logits: logits, timeSteps: 4, vocabSize: 16)
        XCTAssertEqual(ids, [5, 2, 7, 3])
    }

    func testCollapsesConsecutiveDuplicates() {
        // frames: [5, 5, 5, 2, 2, 7, 7, 7, 3] → [5, 2, 7, 3]
        let frames = [5, 5, 5, 2, 2, 7, 7, 7, 3]
        let logits = makeLogits(frames: frames, vocabSize: 16)
        let ids = CTCGreedyDecoder.decode(logits: logits, timeSteps: frames.count, vocabSize: 16)
        XCTAssertEqual(ids, [5, 2, 7, 3])
    }

    func testDoesNotFilterBlankInternallyBlankFilteringIsVocabJob() {
        // Blank (pad_id=1) should be preserved at the decoder level —
        // special-token filtering happens in the vocabulary, not here.
        // This matches Meta's ASRInferencePipeline exactly.
        let frames = [1, 1, 5, 5, 1, 2, 2, 1]
        let logits = makeLogits(frames: frames, vocabSize: 16)
        let ids = CTCGreedyDecoder.decode(logits: logits, timeSteps: frames.count, vocabSize: 16)
        // Duplicates collapse → [1, 5, 1, 2, 1]
        XCTAssertEqual(ids, [1, 5, 1, 2, 1])
    }

    func testValidFramesClampsDecode() {
        // Full buffer is 8 frames; decoder should stop at validFrames=4.
        let frames = [5, 5, 3, 3, 8, 9, 10, 11]
        let logits = makeLogits(frames: frames, vocabSize: 16)
        let ids = CTCGreedyDecoder.decode(
            logits: logits, timeSteps: 8, vocabSize: 16, validFrames: 4)
        XCTAssertEqual(ids, [5, 3])
    }
}

// MARK: - Vocabulary (SentencePiece protobuf)

final class OmnilingualVocabularyTests: XCTestCase {

    /// Build a synthetic SentencePiece `.model` protobuf with N pieces and
    /// the specified `(type)` for each. Minimal: field 1 (piece) and field 3
    /// (type) per submessage; no scores.
    private func buildSyntheticModel(pieces: [(String, Int32)]) -> Data {
        var data = Data()
        for (piece, type) in pieces {
            var sub = Data()

            // field 1 (piece) — wire type 2, length-delimited string
            let bytes = Array(piece.utf8)
            sub.append(Self.encodeTag(field: 1, wire: 2))
            sub.append(Self.encodeVarint(bytes.count))
            sub.append(contentsOf: bytes)

            // field 3 (type) — wire type 0, varint
            sub.append(Self.encodeTag(field: 3, wire: 0))
            sub.append(Self.encodeVarint(Int(type)))

            // Outer: field 1 (pieces), wire type 2
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
            .appendingPathComponent("test_\(UUID().uuidString).model")
        try data.write(to: url)
        addTeardownBlock { try? FileManager.default.removeItem(at: url) }
        return url
    }

    private var defaultTokenizerConfig: OmnilingualConfig.Tokenizer {
        OmnilingualConfig.Tokenizer(
            kind: "sentencepiece", file: "tokenizer.model",
            bosIdx: 0, padIdx: 1, eosIdx: 2, unkIdx: 3)
    }

    func testLoadsPieces() throws {
        // ids 0..3 are standard specials; 4 onward are normal.
        let pieces: [(String, Int32)] = [
            ("<s>",     3), ("<pad>", 3), ("</s>", 3), ("<unk>", 2),
            ("\u{2581}hello", 1), ("\u{2581}world", 1), ("!", 1),
        ]
        let url = try writeTempModel(buildSyntheticModel(pieces: pieces))
        let vocab = try OmnilingualVocabulary.load(from: url, tokenizer: defaultTokenizerConfig)
        XCTAssertEqual(vocab.count, 7)
    }

    func testDecodeStripsSpecialsAndReplacesWordBoundary() throws {
        let pieces: [(String, Int32)] = [
            ("<s>",     3), ("<pad>", 3), ("</s>", 3), ("<unk>", 2),
            ("\u{2581}hello", 1), ("\u{2581}world", 1), ("!", 1),
        ]
        let url = try writeTempModel(buildSyntheticModel(pieces: pieces))
        let vocab = try OmnilingualVocabulary.load(from: url, tokenizer: defaultTokenizerConfig)

        // Simulate a CTC-decoded token stream including blank/pad and specials.
        let ids = [0, 1, 4, 1, 5, 1, 6, 2]  // <s> <pad> ▁hello <pad> ▁world <pad> ! </s>
        let text = vocab.decode(ids)
        XCTAssertEqual(text, "hello world!")
    }

    func testDecodeIgnoresOutOfRangeIds() throws {
        let pieces: [(String, Int32)] = [
            ("<s>", 3), ("<pad>", 3), ("</s>", 3), ("<unk>", 2),
            ("\u{2581}foo", 1),
        ]
        let url = try writeTempModel(buildSyntheticModel(pieces: pieces))
        let vocab = try OmnilingualVocabulary.load(from: url, tokenizer: defaultTokenizerConfig)

        let text = vocab.decode([999, 4, -1, 10288])
        XCTAssertEqual(text, "foo")
    }

    func testEmptyFileIsError() throws {
        let url = try writeTempModel(Data())
        XCTAssertThrowsError(try OmnilingualVocabulary.load(from: url, tokenizer: defaultTokenizerConfig))
    }

    func testSpecialClassificationHonorsPieceTypeAndConfigIds() throws {
        // Even a piece with NORMAL type should be treated as special if its
        // id matches one of bos/pad/eos/unk from the config.
        let pieces: [(String, Int32)] = [
            ("a", 1), ("b", 1), ("c", 1), ("d", 1),
            ("\u{2581}after", 1),
        ]
        let url = try writeTempModel(buildSyntheticModel(pieces: pieces))
        let config = OmnilingualConfig.Tokenizer(
            kind: "sentencepiece", file: "tokenizer.model",
            bosIdx: 0, padIdx: 1, eosIdx: 2, unkIdx: 3)
        let vocab = try OmnilingualVocabulary.load(from: url, tokenizer: config)

        // ids 0..3 should be treated as special via config, id 4 is normal.
        XCTAssertTrue(vocab.isSpecial(0))
        XCTAssertTrue(vocab.isSpecial(1))
        XCTAssertTrue(vocab.isSpecial(2))
        XCTAssertTrue(vocab.isSpecial(3))
        XCTAssertFalse(vocab.isSpecial(4))

        XCTAssertEqual(vocab.decode([0, 1, 2, 3, 4]), "after")
    }
}
