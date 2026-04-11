import XCTest
@testable import OmnilingualASR

/// Unit tests for the MLX backend config and module shapes. These do not load
/// real weights — they verify the variant table and feature-extractor length
/// math against the published shapes.
final class OmnilingualMLXConfigTests: XCTestCase {

    func test300MVariant() {
        let c = OmnilingualMLXConfig.variant(.m300)
        XCTAssertEqual(c.modelDim, 1024)
        XCTAssertEqual(c.numLayers, 24)
        XCTAssertEqual(c.numHeads, 16)
        XCTAssertEqual(c.ffnDim, 4096)
        XCTAssertEqual(c.headDim, 64)
        XCTAssertEqual(c.vocabSize, 10288)
        XCTAssertEqual(c.bits, 4)
        XCTAssertEqual(c.groupSize, 64)
    }

    func test1BVariant() {
        let c = OmnilingualMLXConfig.variant(.b1)
        XCTAssertEqual(c.modelDim, 1280)
        XCTAssertEqual(c.numLayers, 48)
        XCTAssertEqual(c.numHeads, 20)
        XCTAssertEqual(c.ffnDim, 5120)
        XCTAssertEqual(c.headDim, 64)
    }

    func test3BVariant() {
        let c = OmnilingualMLXConfig.variant(.b3)
        XCTAssertEqual(c.modelDim, 2048)
        XCTAssertEqual(c.numLayers, 60)
        XCTAssertEqual(c.numHeads, 32)
        XCTAssertEqual(c.ffnDim, 8192)
        XCTAssertEqual(c.headDim, 64)
    }

    func test7BVariant() {
        let c = OmnilingualMLXConfig.variant(.b7)
        XCTAssertEqual(c.modelDim, 2048)
        XCTAssertEqual(c.numLayers, 128)
        XCTAssertEqual(c.numHeads, 32)
        XCTAssertEqual(c.ffnDim, 8192)
    }

    func testFrontendStrideIs320() {
        let c = OmnilingualMLXConfig.variant(.m300)
        XCTAssertEqual(c.encoderStride, 320)
        XCTAssertEqual(c.convStrides, [5, 2, 2, 2, 2, 2, 2])
        XCTAssertEqual(c.convKernels, [10, 3, 3, 3, 3, 2, 2])
    }

    func testDefaultModelIdResolution() {
        XCTAssertEqual(
            OmnilingualMLXConfig.defaultModelId(variant: .m300, bits: 4),
            "aufklarer/Omnilingual-ASR-CTC-300M-MLX-4bit")
        XCTAssertEqual(
            OmnilingualMLXConfig.defaultModelId(variant: .b1, bits: 8),
            "aufklarer/Omnilingual-ASR-CTC-1B-MLX-8bit")
        XCTAssertEqual(
            OmnilingualMLXConfig.defaultModelId(variant: .b7, bits: 4),
            "aufklarer/Omnilingual-ASR-CTC-7B-MLX-4bit")
    }
}

final class Wav2Vec2FrontendShapesTests: XCTestCase {

    func testFeatureExtractorOutputLength10s() {
        let c = OmnilingualMLXConfig.variant(.m300)
        let fe = Wav2Vec2FeatureExtractor(
            featureDim: c.featureDim, kernels: c.convKernels, strides: c.convStrides)
        // 10 s @ 16 kHz = 160_000 samples → 499 frames after the 7-conv stack
        XCTAssertEqual(fe.outputLength(for: 160_000), 499)
    }

    func testFeatureExtractorOutputLength5s() {
        let c = OmnilingualMLXConfig.variant(.m300)
        let fe = Wav2Vec2FeatureExtractor(
            featureDim: c.featureDim, kernels: c.convKernels, strides: c.convStrides)
        // 5 s @ 16 kHz = 80_000 samples → 249 frames
        XCTAssertEqual(fe.outputLength(for: 80_000), 249)
    }

    func testFeatureExtractorRejectsTooShortInput() {
        let c = OmnilingualMLXConfig.variant(.m300)
        let fe = Wav2Vec2FeatureExtractor(
            featureDim: c.featureDim, kernels: c.convKernels, strides: c.convStrides)
        XCTAssertEqual(fe.outputLength(for: 0), 0)
        XCTAssertEqual(fe.outputLength(for: 5), 0)
    }
}
