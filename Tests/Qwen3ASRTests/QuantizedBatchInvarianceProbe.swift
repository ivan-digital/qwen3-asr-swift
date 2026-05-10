import XCTest
import MLX
import MLXNN
import MLXFast
import MLXRandom

/// Op-level probe for the MLX `B>1, T=1` row-asymmetric output bug we
/// hit in Qwen3-ASR's experimental batched decode path. Bisects which op
/// in `quantizedMatmul → RMSNorm → MLXFast.RoPE` is non-batch-invariant.
///
/// Filed upstream as ml-explore/mlx-swift#401. The bug isolates to
/// `MLXNN.RoPE`/`MLXFast.RoPE` at `[B>1, H, T=1, D]` — reproduces with
/// no quantization, no preceding ops, both `offset=0` and `offset>0`.
/// The same op at `T>1` or `B=1` is correct.
///
/// Tests for the broken shapes are wrapped in `XCTExpectFailure` so CI
/// stays green. When mlx-swift#401 is fixed, the equality assertions
/// succeed, the expected failures go unsatisfied, and these tests flip
/// red — that's the signal to delete the workaround in `transcribeBatch`.
///
/// These tests are intentionally NOT prefixed `E2E` (no model download).
final class QuantizedBatchInvarianceProbeTests: XCTestCase {

    static let inDim = 1024
    static let outDim = 1024
    static let groupSize = 64
    static let bits = 4
    static let headDim = 128
    static let kvHeads = 8

    /// Test 1: `quantizedMatmul` alone.
    ///
    /// Construct a row-symmetric input `[2, T, inDim]` (row 0 == row 1) and
    /// pass it through a quantized linear. If row 0 == row 1 of the output,
    /// the op is batch-invariant. If they differ, this op is the culprit.
    func testQuantizedMatmulAtT1() {
        MLXRandom.seed(42)

        let weight = MLXRandom.normal([Self.outDim, Self.inDim])
        let (qWeight, scales, biases) = quantized(
            weight, groupSize: Self.groupSize, bits: Self.bits)

        // Row-symmetric input
        let baseRow = MLXRandom.normal([1, 1, Self.inDim])
        let input = concatenated([baseRow, baseRow], axis: 0)  // [2, 1, inDim]

        // Sanity check: input rows are truly identical
        let i0 = input[0..<1, 0..., 0...].asArray(Float.self)
        let i1 = input[1..<2, 0..., 0...].asArray(Float.self)
        let inputDiff = zip(i0, i1).map { abs($0 - $1) }.max() ?? 0
        XCTAssertEqual(inputDiff, 0, "input rows must be byte-identical for this probe")

        // Forward
        let output = quantizedMatmul(
            input, qWeight, scales: scales, biases: biases,
            transpose: true, groupSize: Self.groupSize, bits: Self.bits)

        let o0 = output[0..<1, 0..., 0...].asArray(Float.self)
        let o1 = output[1..<2, 0..., 0...].asArray(Float.self)
        let outputDiff = zip(o0, o1).map { abs($0 - $1) }.max() ?? 0

        print("[PROBE] quantizedMatmul T=1: row0–row1 maxDiff = \(outputDiff)")
        XCTAssertEqual(outputDiff, 0, "quantizedMatmul on identical rows at T=1 must be row-symmetric")
    }

    /// Control: same op at `T=148` (prefill-like). Should be row-symmetric.
    func testQuantizedMatmulAtT148() {
        MLXRandom.seed(42)

        let weight = MLXRandom.normal([Self.outDim, Self.inDim])
        let (qWeight, scales, biases) = quantized(
            weight, groupSize: Self.groupSize, bits: Self.bits)

        let baseSeq = MLXRandom.normal([1, 148, Self.inDim])
        let input = concatenated([baseSeq, baseSeq], axis: 0)  // [2, 148, inDim]

        let output = quantizedMatmul(
            input, qWeight, scales: scales, biases: biases,
            transpose: true, groupSize: Self.groupSize, bits: Self.bits)

        let o0 = output[0..<1, 0..., 0...].asArray(Float.self)
        let o1 = output[1..<2, 0..., 0...].asArray(Float.self)
        let outputDiff = zip(o0, o1).map { abs($0 - $1) }.max() ?? 0

        print("[PROBE] quantizedMatmul T=148: row0–row1 maxDiff = \(outputDiff)")
        XCTAssertEqual(outputDiff, 0, "quantizedMatmul on identical rows at T=148 must be row-symmetric")
    }

    /// Test 2: `quantizedMatmul → reshape → RMSNorm → transpose` (the head
    /// projection chain in Qwen3 attention up to RoPE).
    func testQProjPlusKNormAtT1() {
        MLXRandom.seed(42)

        let weight = MLXRandom.normal([Self.kvHeads * Self.headDim, Self.inDim])
        let (qWeight, scales, biases) = quantized(
            weight, groupSize: Self.groupSize, bits: Self.bits)

        let kNorm = RMSNorm(dimensions: Self.headDim, eps: 1e-6)

        let baseRow = MLXRandom.normal([1, 1, Self.inDim])
        let input = concatenated([baseRow, baseRow], axis: 0)  // [2, 1, inDim]

        var keys = quantizedMatmul(
            input, qWeight, scales: scales, biases: biases,
            transpose: true, groupSize: Self.groupSize, bits: Self.bits)
        keys = keys.reshaped(2, 1, Self.kvHeads, Self.headDim)
        keys = kNorm(keys)
        keys = keys.transposed(0, 2, 1, 3)  // [2, kvHeads, 1, headDim]

        let r0 = keys[0..<1, 0..., 0..., 0...].asArray(Float.self)
        let r1 = keys[1..<2, 0..., 0..., 0...].asArray(Float.self)
        let diff = zip(r0, r1).map { abs($0 - $1) }.max() ?? 0

        print("[PROBE] qProj+RMSNorm T=1: row0–row1 maxDiff = \(diff)")
        XCTAssertEqual(diff, 0, "qProj + RMSNorm chain on identical rows at T=1 must be row-symmetric")
    }

    /// Test 3a: `MLXFast.RoPE` alone, T=1, B=2, offset=0.
    /// Currently fails — locked in via XCTExpectFailure. When mlx-swift#401
    /// is fixed, the assertion succeeds, the expectation goes unsatisfied,
    /// and this test flips red so we know to drop the workaround.
    func testRopeAtT1Offset0() {
        MLXRandom.seed(42)
        let rope = MLXNN.RoPE(dimensions: Self.headDim, traditional: false, base: 1_000_000)
        let baseRow = MLXRandom.normal([1, Self.kvHeads, 1, Self.headDim])
        let input = concatenated([baseRow, baseRow], axis: 0)
        let output = rope(input, offset: 0)
        let r0 = output[0..<1, 0..., 0..., 0...].asArray(Float.self)
        let r1 = output[1..<2, 0..., 0..., 0...].asArray(Float.self)
        let diff = zip(r0, r1).map { abs($0 - $1) }.max() ?? 0
        print("[PROBE] RoPE T=1 B=2 offset=0: row0–row1 maxDiff = \(diff)")
        XCTExpectFailure("ml-explore/mlx-swift#401 — drop this expectation when fixed.")
        XCTAssertEqual(diff, 0)
    }

    /// Test 3b: `MLXFast.RoPE` alone, T=1, B=2, offset=148.
    /// See `testRopeAtT1Offset0` for the XCTExpectFailure rationale.
    func testRopeAtT1Offset148() {
        MLXRandom.seed(42)
        let rope = MLXNN.RoPE(dimensions: Self.headDim, traditional: false, base: 1_000_000)
        let baseRow = MLXRandom.normal([1, Self.kvHeads, 1, Self.headDim])
        let input = concatenated([baseRow, baseRow], axis: 0)
        let output = rope(input, offset: 148)
        let r0 = output[0..<1, 0..., 0..., 0...].asArray(Float.self)
        let r1 = output[1..<2, 0..., 0..., 0...].asArray(Float.self)
        let diff = zip(r0, r1).map { abs($0 - $1) }.max() ?? 0
        print("[PROBE] RoPE T=1 B=2 offset=148: row0–row1 maxDiff = \(diff)")
        XCTExpectFailure("ml-explore/mlx-swift#401 — drop this expectation when fixed.")
        XCTAssertEqual(diff, 0)
    }

    /// Test 3c: `MLXFast.RoPE` alone, T=2, B=2, offset=148 (T>1 control).
    func testRopeAtT2Offset148() {
        MLXRandom.seed(42)
        let rope = MLXNN.RoPE(dimensions: Self.headDim, traditional: false, base: 1_000_000)
        let baseRow = MLXRandom.normal([1, Self.kvHeads, 2, Self.headDim])
        let input = concatenated([baseRow, baseRow], axis: 0)
        let output = rope(input, offset: 148)
        let r0 = output[0..<1, 0..., 0..., 0...].asArray(Float.self)
        let r1 = output[1..<2, 0..., 0..., 0...].asArray(Float.self)
        let diff = zip(r0, r1).map { abs($0 - $1) }.max() ?? 0
        print("[PROBE] RoPE T=2 B=2 offset=148: row0–row1 maxDiff = \(diff)")
        XCTAssertEqual(diff, 0)
    }

    /// Test 3d: `MLXFast.RoPE` alone, T=1, B=1, offset=148 (B=1 control).
    func testRopeAtT1B1Offset148() {
        MLXRandom.seed(42)
        let rope = MLXNN.RoPE(dimensions: Self.headDim, traditional: false, base: 1_000_000)
        let baseRow = MLXRandom.normal([1, Self.kvHeads, 1, Self.headDim])
        // Run twice and compare — control for kernel determinism
        let o1 = rope(baseRow, offset: 148).asArray(Float.self)
        let o2 = rope(baseRow, offset: 148).asArray(Float.self)
        let diff = zip(o1, o2).map { abs($0 - $1) }.max() ?? 0
        print("[PROBE] RoPE T=1 B=1 offset=148 (twice): maxDiff = \(diff)")
        XCTAssertEqual(diff, 0, "RoPE must be deterministic")
    }

    /// Test 4: full pipeline `qProj → reshape → RMSNorm → transpose → RoPE`.
    /// Mirrors what a Qwen3 attention head computes per decode step. The
    /// RoPE bug compounds through the chain so this also fails today.
    /// Locked in via XCTExpectFailure pending ml-explore/mlx-swift#401.
    func testQProjKNormRopeAtT1() {
        MLXRandom.seed(42)

        let weight = MLXRandom.normal([Self.kvHeads * Self.headDim, Self.inDim])
        let (qWeight, scales, biases) = quantized(
            weight, groupSize: Self.groupSize, bits: Self.bits)

        let kNorm = RMSNorm(dimensions: Self.headDim, eps: 1e-6)
        let rope = MLXNN.RoPE(dimensions: Self.headDim, traditional: false, base: 1_000_000)

        let baseRow = MLXRandom.normal([1, 1, Self.inDim])
        let input = concatenated([baseRow, baseRow], axis: 0)

        var keys = quantizedMatmul(
            input, qWeight, scales: scales, biases: biases,
            transpose: true, groupSize: Self.groupSize, bits: Self.bits)
        keys = keys.reshaped(2, 1, Self.kvHeads, Self.headDim)
        keys = kNorm(keys)
        keys = keys.transposed(0, 2, 1, 3)
        keys = rope(keys, offset: 148)

        let r0 = keys[0..<1, 0..., 0..., 0...].asArray(Float.self)
        let r1 = keys[1..<2, 0..., 0..., 0...].asArray(Float.self)
        let diff = zip(r0, r1).map { abs($0 - $1) }.max() ?? 0

        print("[PROBE] qProj+RMSNorm+RoPE T=1: row0–row1 maxDiff = \(diff)")
        XCTExpectFailure("ml-explore/mlx-swift#401 — drop this expectation when fixed.")
        XCTAssertEqual(diff, 0, "qProj + RMSNorm + RoPE chain on identical rows at T=1 must be row-symmetric")
    }
}
