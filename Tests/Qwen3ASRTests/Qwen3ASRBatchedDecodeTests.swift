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

    // The greedy fast path is the only path that can run the batched decoder.
    // Anything that needs CPU-side logits (repetition penalty, n-gram mask,
    // temperature sampling) must drop back to serial per-chunk transcribe so
    // those checks stay correct. These assertions lock in the exact gate so
    // a future refactor cannot silently let non-greedy options reach the
    // batched path.

    func testIsGreedyFastPathTrueForDefaultOptions() {
        XCTAssertTrue(Qwen3ASRModel.isGreedyFastPath(Qwen3DecodingOptions()))
    }

    func testIsGreedyFastPathFalseForRepetitionPenalty() {
        XCTAssertFalse(Qwen3ASRModel.isGreedyFastPath(Qwen3DecodingOptions(repetitionPenalty: 1.1)))
    }

    func testIsGreedyFastPathFalseForNoRepeatNgram() {
        XCTAssertFalse(Qwen3ASRModel.isGreedyFastPath(Qwen3DecodingOptions(noRepeatNgramSize: 3)))
    }

    func testIsGreedyFastPathFalseForTemperature() {
        XCTAssertFalse(Qwen3ASRModel.isGreedyFastPath(Qwen3DecodingOptions(temperature: 0.5)))
    }

    func testIsGreedyFastPathFalseForCombinedNonGreedyOptions() {
        let opts = Qwen3DecodingOptions(
            repetitionPenalty: 1.15,
            noRepeatNgramSize: 3,
            temperature: 0.2
        )
        XCTAssertFalse(Qwen3ASRModel.isGreedyFastPath(opts))
    }

}
