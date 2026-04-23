import XCTest
@testable import SourceSeparation
import MLX

final class OpenUnmixConfigTests: XCTestCase {

    func testUMXHQConfig() {
        let config = OpenUnmixConfig.umxhq
        XCTAssertEqual(config.hiddenSize, 512)
        XCTAssertEqual(config.nbBins, 2049)
        XCTAssertEqual(config.maxBin, 1487)
        XCTAssertEqual(config.nbChannels, 2)
        XCTAssertEqual(config.sampleRate, 44100)
        XCTAssertEqual(config.nFFT, 4096)
        XCTAssertEqual(config.nHop, 1024)
        XCTAssertEqual(config.targets.count, 4)
    }

    func testUMXLConfig() {
        let config = OpenUnmixConfig.umxl
        XCTAssertEqual(config.hiddenSize, 1024)
        XCTAssertEqual(config.model, "umxl")
    }

    func testConfigCodable() throws {
        let config = OpenUnmixConfig.umxhq
        let data = try JSONEncoder().encode(config)
        let decoded = try JSONDecoder().decode(OpenUnmixConfig.self, from: data)
        XCTAssertEqual(decoded.hiddenSize, config.hiddenSize)
        XCTAssertEqual(decoded.sampleRate, config.sampleRate)
    }

    func testAllTargets() {
        let targets = SeparationTarget.allCases
        XCTAssertEqual(targets.count, 4)
        XCTAssertTrue(targets.contains(.vocals))
        XCTAssertTrue(targets.contains(.drums))
        XCTAssertTrue(targets.contains(.bass))
        XCTAssertTrue(targets.contains(.other))
    }
}

final class OpenUnmixModelTests: XCTestCase {

    func testModelInit() {
        let model = OpenUnmixStemModel(hiddenSize: 512)
        XCTAssertEqual(model.hiddenSize, 512)
        XCTAssertEqual(model.nbBins, 2049)
        XCTAssertEqual(model.maxBin, 1487)
    }

    func testForwardShape() {
        let model = OpenUnmixStemModel(hiddenSize: 64)  // Small for test speed
        let T = 10
        let input = MLXArray.ones([T, 2, 2049])  // [T, channels, bins]
        let output = model(input)
        XCTAssertEqual(output.shape, [T, 2, 2049])
    }

    func testForwardProducesFiniteValues() {
        let model = OpenUnmixStemModel(hiddenSize: 32)
        let input = MLXRandom.normal([5, 2, 2049]).abs() + 0.01
        let output = model(input)
        eval(output)
        // Output should be non-negative (ReLU mask)
        let minVal = output.min().item(Float.self)
        XCTAssertGreaterThanOrEqual(minVal, 0.0)
    }
}

final class LSTMCellTests: XCTestCase {

    func testLSTMCellOutputShape() {
        let cell = LSTMCell(inputSize: 8, hiddenSize: 4)
        let x = MLXArray.ones([1, 8])
        let h = MLXArray.zeros([1, 4])
        let c = MLXArray.zeros([1, 4])
        let (newH, newC) = cell.step(x, h: h, c: c)
        eval(newH, newC)
        XCTAssertEqual(newH.shape, [1, 4])
        XCTAssertEqual(newC.shape, [1, 4])
    }

    func testBiLSTMLayerOutputShape() {
        let layer = BiLSTMLayer(inputSize: 8, hiddenSize: 4)
        let x = MLXArray.ones([5, 8])  // [T=5, features=8]
        let output = layer(x)
        eval(output)
        XCTAssertEqual(output.shape, [5, 8])  // [T, hidden*2]
    }

    func testBiLSTMStackOutputShape() {
        let stack = BiLSTMStack(inputSize: 16, hiddenSize: 8, numLayers: 3)
        let x = MLXArray.ones([10, 16])
        let output = stack(x)
        eval(output)
        XCTAssertEqual(output.shape, [10, 16])  // [T, hidden*2]
    }
}
