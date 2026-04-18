import XCTest
import MLX
@testable import Qwen3ASR

/// Unit tests for ``Qwen3DecodingOptions`` and the pure-Swift
/// ``Qwen3ASRModel.pickNextToken(...)`` sampler. These are synthetic —
/// no model download, no GPU. Everything exercises only the logit
/// manipulation math added in the decoder-options feature.
final class Qwen3DecodingOptionsTests: XCTestCase {

    // MARK: - Qwen3DecodingOptions

    func testDecodingOptionsDefaults() {
        let opts = Qwen3DecodingOptions()
        XCTAssertEqual(opts.maxTokens, 448)
        XCTAssertNil(opts.language)
        XCTAssertNil(opts.context)
        XCTAssertEqual(opts.repetitionPenalty, 1.0)
        XCTAssertEqual(opts.noRepeatNgramSize, 0)
        XCTAssertEqual(opts.temperature, 0.0)
    }

    func testDecodingOptionsCustomInit() {
        let opts = Qwen3DecodingOptions(
            maxTokens: 128,
            language: "en",
            context: "hello",
            repetitionPenalty: 1.2,
            noRepeatNgramSize: 3,
            temperature: 0.7
        )
        XCTAssertEqual(opts.maxTokens, 128)
        XCTAssertEqual(opts.language, "en")
        XCTAssertEqual(opts.context, "hello")
        XCTAssertEqual(opts.repetitionPenalty, 1.2)
        XCTAssertEqual(opts.noRepeatNgramSize, 3)
        XCTAssertEqual(opts.temperature, 0.7)
    }

    func testDecodingOptionsSendable() async {
        let opts = Qwen3DecodingOptions(repetitionPenalty: 1.15)
        let mirrored = await Task { opts }.value
        XCTAssertEqual(mirrored.repetitionPenalty, 1.15)
    }

    // MARK: - pickNextToken — fast path

    /// With default options (no repetition, no n-gram mask, temperature=0)
    /// the sampler must be bit-identical to plain ``argMax``.
    func testFastPathMatchesArgMax() {
        var logits: [Float] = Array(repeating: -1.0, count: 64)
        logits[42] = 5.0
        logits[7] = 2.0
        let arr = MLXArray(logits, [1, 1, 64])
        let token = Qwen3ASRModel.pickNextToken(
            logits: arr,
            generatedSoFar: [],
            options: Qwen3DecodingOptions()
        )
        XCTAssertEqual(token, 42)
    }

    // MARK: - Repetition penalty

    func testRepetitionPenaltyDemotesRepeatedPositiveLogit() {
        // Without penalty the argmax is token 5 (highest positive logit).
        // With penalty 2.0 applied to already-generated [5], logit 5 drops
        // from 4.0 to 2.0 — below token 7 at 3.0, which now wins.
        var logits: [Float] = Array(repeating: -5.0, count: 32)
        logits[5] = 4.0
        logits[7] = 3.0
        let arr = MLXArray(logits, [1, 1, 32])

        var opts = Qwen3DecodingOptions()
        opts.repetitionPenalty = 2.0
        let token = Qwen3ASRModel.pickNextToken(
            logits: arr,
            generatedSoFar: [5],
            options: opts
        )
        XCTAssertEqual(token, 7, "repetition penalty must demote the already-generated token")
    }

    func testRepetitionPenaltyHandlesNegativeLogitSign() {
        // Negative logits must be MULTIPLIED by the penalty (not divided),
        // so that penalised negative logits become MORE negative. Without
        // this sign fix the "penalty" would actually BOOST repeated tokens
        // whose logit was negative.
        //
        // Baseline is a strongly negative "floor" so only tokens 3 and 11
        // are competitive candidates.
        var logits: [Float] = Array(repeating: -10.0, count: 16)
        logits[3] = -1.0      // already generated, negative but the current max
        logits[11] = -2.0     // competing candidate, slightly lower
        let arr = MLXArray(logits, [1, 1, 16])

        var opts = Qwen3DecodingOptions()
        opts.repetitionPenalty = 3.0
        let token = Qwen3ASRModel.pickNextToken(
            logits: arr,
            generatedSoFar: [3],
            options: opts
        )
        // With the HF-style sign-aware penalty, token 3's score becomes
        // -1.0 * 3.0 = -3.0, which is now *below* token 11's -2.0 so 11
        // wins. A naïve "always divide" implementation would map -1.0 to
        // -1.0/3.0 = -0.33 — making 3 an even stronger argmax. Catching
        // that regression is the whole point of this test.
        XCTAssertEqual(token, 11)
    }

    func testRepetitionPenaltyNoOpOnFirstToken() {
        // With no generated history, penalty has nothing to demote.
        var logits: [Float] = Array(repeating: -1.0, count: 16)
        logits[9] = 3.0
        let arr = MLXArray(logits, [1, 1, 16])

        var opts = Qwen3DecodingOptions()
        opts.repetitionPenalty = 2.5
        let token = Qwen3ASRModel.pickNextToken(
            logits: arr,
            generatedSoFar: [],
            options: opts
        )
        XCTAssertEqual(token, 9)
    }

    // MARK: - No-repeat n-gram

    func testNoRepeatNgramMasksRepeatedTrigram() {
        // generatedSoFar = [A, B, A, B]. With n=3, we've already seen the
        // 3-gram [A, B, A] once, so emitting A next (which would form a
        // second [A, B, A]) must be forbidden — even though A's raw logit
        // is the highest.
        let A: Int32 = 5, B: Int32 = 6, C: Int32 = 7
        var logits: [Float] = Array(repeating: -10.0, count: 32)
        logits[Int(A)] = 5.0
        logits[Int(B)] = 2.0
        logits[Int(C)] = 3.0
        let arr = MLXArray(logits, [1, 1, 32])

        var opts = Qwen3DecodingOptions()
        opts.noRepeatNgramSize = 3
        let token = Qwen3ASRModel.pickNextToken(
            logits: arr,
            generatedSoFar: [A, B, A, B],
            options: opts
        )
        XCTAssertNotEqual(token, A, "n-gram mask should forbid completing a repeated trigram")
        XCTAssertEqual(token, C, "C is the next-highest logit once A is masked")
    }

    func testNoRepeatNgramAllowsNovelFollowup() {
        // generatedSoFar = [A, B, C]. Last n-1=2 tokens are [B, C]; that
        // pair never appeared before in the sequence, so no token is
        // forbidden and argmax wins.
        let A: Int32 = 1, B: Int32 = 2, C: Int32 = 3, D: Int32 = 4
        var logits: [Float] = Array(repeating: -5.0, count: 16)
        logits[Int(D)] = 3.0
        let arr = MLXArray(logits, [1, 1, 16])

        var opts = Qwen3DecodingOptions()
        opts.noRepeatNgramSize = 3
        let token = Qwen3ASRModel.pickNextToken(
            logits: arr,
            generatedSoFar: [A, B, C],
            options: opts
        )
        XCTAssertEqual(token, D)
    }

    // MARK: - Temperature sampling

    func testTemperatureZeroRemainsDeterministic() {
        // Temperature=0 triggers the fast path; same input must give same
        // token every time.
        var logits: [Float] = Array(repeating: 0, count: 16)
        logits[4] = 2.0
        let arr = MLXArray(logits, [1, 1, 16])
        let opts = Qwen3DecodingOptions()  // temperature=0 by default
        let first = Qwen3ASRModel.pickNextToken(logits: arr, generatedSoFar: [], options: opts)
        for _ in 0..<5 {
            XCTAssertEqual(
                Qwen3ASRModel.pickNextToken(logits: arr, generatedSoFar: [], options: opts),
                first
            )
        }
    }

    func testTemperatureSamplingProducesVariety() {
        // With temperature > 0 the Gumbel-max trick should produce some
        // variety even on identical logits. Run the sampler 50 times over
        // a 16-wide uniform distribution and assert we see at least 3
        // distinct tokens — exceptionally unlikely to fail by chance.
        let logits: [Float] = Array(repeating: 0, count: 16)
        let arr = MLXArray(logits, [1, 1, 16])

        var opts = Qwen3DecodingOptions()
        opts.temperature = 1.0

        var seen: Set<Int32> = []
        for _ in 0..<50 {
            let t = Qwen3ASRModel.pickNextToken(logits: arr, generatedSoFar: [], options: opts)
            seen.insert(t)
        }
        XCTAssertGreaterThanOrEqual(seen.count, 3,
            "temperature=1.0 on uniform logits should sample ≥ 3 distinct tokens over 50 rolls")
    }

    func testLowTemperatureStaysMostlyAtPeak() {
        // With a peaked distribution and a very low temperature the
        // sampler should still pick the peak the vast majority of the
        // time — lightly validating that temperature is scaling, not
        // clobbering, the logits.
        var logits: [Float] = Array(repeating: 0, count: 16)
        logits[9] = 10.0
        let arr = MLXArray(logits, [1, 1, 16])

        var opts = Qwen3DecodingOptions()
        opts.temperature = 0.1

        var peakHits = 0
        let trials = 50
        for _ in 0..<trials {
            let t = Qwen3ASRModel.pickNextToken(logits: arr, generatedSoFar: [], options: opts)
            if t == 9 { peakHits += 1 }
        }
        XCTAssertGreaterThan(peakHits, trials / 2,
            "peaky distribution + low temperature should still pick the peak > 50% of trials")
    }
}
