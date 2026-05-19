import XCTest
import MLX
@testable import Qwen3ASR

final class Qwen3ASRBatchedDecodeTests: XCTestCase {

    func testBatchOptionResolutionPreservesOptionsByDefault() {
        let options = Qwen3DecodingOptions(
            maxTokens: 64,
            language: "zh",
            context: "meeting vocabulary",
            repetitionPenalty: 1.15,
            noRepeatNgramSize: 3,
            temperature: 0.2
        )

        let resolved = Qwen3ASRModel.resolveBatchDecodingOptions(
            language: nil,
            maxTokens: nil,
            context: nil,
            options: options
        )

        XCTAssertEqual(resolved.maxTokens, 64)
        XCTAssertEqual(resolved.language, "zh")
        XCTAssertEqual(resolved.context, "meeting vocabulary")
        XCTAssertEqual(resolved.repetitionPenalty, 1.15)
        XCTAssertEqual(resolved.noRepeatNgramSize, 3)
        XCTAssertEqual(resolved.temperature, 0.2)
    }

    func testBatchConvenienceArgumentsOverrideMatchingOptions() {
        let options = Qwen3DecodingOptions(maxTokens: 64, language: "zh", context: "old")

        let resolved = Qwen3ASRModel.resolveBatchDecodingOptions(
            language: "en",
            maxTokens: 128,
            context: "new",
            options: options
        )

        XCTAssertEqual(resolved.maxTokens, 128)
        XCTAssertEqual(resolved.language, "en")
        XCTAssertEqual(resolved.context, "new")
    }

    func testSeqLenBucketsGroupEqualLengthsAndPreserveInputOrder() {
        let batches = Qwen3ASRModel.makeSeqLenBatches(
            seqLens: [10, 20, 10, 30, 20, 10],
            maxBatchSize: 8
        )

        XCTAssertEqual(batches, [[0, 2, 5], [1, 4], [3]])
    }

    func testSeqLenBucketsRespectMaxBatchSize() {
        let batches = Qwen3ASRModel.makeSeqLenBatches(
            seqLens: [10, 10, 10, 10, 20],
            maxBatchSize: 2
        )

        XCTAssertEqual(batches, [[0, 1], [2, 3], [4]])
    }

    func testPromptTokenIdsMatchDefaultTemplateBoundaries() {
        let (prefix, suffix) = Qwen3ASRModel.buildPromptTokenIds(
            language: nil,
            context: nil,
            tokenizer: nil
        )

        XCTAssertEqual(prefix, [
            151644, 8948, 198,
            151645, 198,
            151644, 872, 198, 151669
        ])
        XCTAssertEqual(suffix, [
            151670, 151645, 198,
            151644, 77091, 198,
            151704
        ])
    }

    func testBatchCausalMaskShapeAndValues() {
        let mask = Qwen3ASRModel.buildBatchedCausalMask(seqLen: 3, dtype: .float32)

        XCTAssertEqual(mask.shape, [1, 1, 3, 3])
        XCTAssertEqual(mask[0, 0, 0, 0].item(Float.self), 0)
        XCTAssertLessThan(mask[0, 0, 0, 1].item(Float.self), -1e8)
        XCTAssertLessThan(mask[0, 0, 0, 2].item(Float.self), -1e8)
        XCTAssertEqual(mask[0, 0, 1, 0].item(Float.self), 0)
        XCTAssertEqual(mask[0, 0, 1, 1].item(Float.self), 0)
        XCTAssertLessThan(mask[0, 0, 1, 2].item(Float.self), -1e8)
        XCTAssertEqual(mask[0, 0, 2, 0].item(Float.self), 0)
        XCTAssertEqual(mask[0, 0, 2, 1].item(Float.self), 0)
        XCTAssertEqual(mask[0, 0, 2, 2].item(Float.self), 0)
    }
}
