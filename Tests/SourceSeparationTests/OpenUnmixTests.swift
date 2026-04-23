import XCTest
@testable import SourceSeparation
@testable import AudioCommon
import MLX
import Foundation

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

// MARK: - E2E Tests (require model download)

final class E2EOpenUnmixTests: XCTestCase {

    private static var _separator: SourceSeparator?

    private var separator: SourceSeparator {
        get throws {
            guard let s = Self._separator else { throw XCTSkip("Model not loaded") }
            return s
        }
    }

    override func setUp() async throws {
        try await super.setUp()
        if Self._separator == nil {
            Self._separator = try await SourceSeparator.fromPretrained()
        }
    }

    /// Synthesize a 2-second stereo test signal at 44.1 kHz: a sine wave mixed
    /// with low-pass noise. Not a real song, but energetic and non-silent in
    /// the frequency bands the model cares about.
    private func makeStereoTestSignal(seconds: Double = 2.0, sampleRate: Int = 44100) -> [[Float]] {
        let n = Int(Double(sampleRate) * seconds)
        var left = [Float](repeating: 0, count: n)
        var right = [Float](repeating: 0, count: n)
        var rng = SystemRandomNumberGenerator()
        for i in 0..<n {
            let t = Float(i) / Float(sampleRate)
            // 220 Hz sine + 440 Hz sine + pink-ish noise
            let tone = 0.3 * sin(2 * .pi * 220 * t) + 0.2 * sin(2 * .pi * 440 * t)
            let noise = Float.random(in: -0.1...0.1, using: &rng)
            left[i] = tone + noise
            right[i] = tone * 0.9 + Float.random(in: -0.1...0.1, using: &rng)
        }
        return [left, right]
    }

    func testModelLoads() throws {
        let s = try separator
        XCTAssertEqual(s.config.sampleRate, 44100)
        XCTAssertEqual(s.config.nFFT, 4096)
        XCTAssertEqual(s.config.nHop, 1024)
    }

    func testSeparateProducesFourStems() throws {
        let s = try separator
        let audio = makeStereoTestSignal(seconds: 2.0)
        let stems = s.separate(audio: audio, sampleRate: 44100, wiener: false)
        XCTAssertEqual(stems.count, 4, "Should return all 4 stems")
        for target in SeparationTarget.allCases {
            guard let stereoStem = stems[target] else {
                XCTFail("Missing stem: \(target.rawValue)")
                continue
            }
            XCTAssertEqual(stereoStem.count, 2, "\(target.rawValue) should be stereo")
            XCTAssertEqual(stereoStem[0].count, audio[0].count, "\(target.rawValue) length mismatch L")
            XCTAssertEqual(stereoStem[1].count, audio[1].count, "\(target.rawValue) length mismatch R")
        }
    }

    func testSeparateStemsAreNotSilent() throws {
        let s = try separator
        let audio = makeStereoTestSignal(seconds: 2.0)
        let stems = s.separate(audio: audio, sampleRate: 44100, wiener: false)

        func rms(_ buf: [Float]) -> Float {
            guard !buf.isEmpty else { return 0 }
            var sumSq: Float = 0
            for x in buf { sumSq += x * x }
            return sqrt(sumSq / Float(buf.count))
        }
        for target in SeparationTarget.allCases {
            let stem = stems[target]!
            let leftRMS = rms(stem[0])
            let rightRMS = rms(stem[1])
            XCTAssertGreaterThan(leftRMS, 1e-5,
                "\(target.rawValue) L is silent (RMS=\(leftRMS))")
            XCTAssertGreaterThan(rightRMS, 1e-5,
                "\(target.rawValue) R is silent (RMS=\(rightRMS))")
        }
    }

    func testSeparateWithWienerProducesFiniteValues() throws {
        let s = try separator
        let audio = makeStereoTestSignal(seconds: 2.0)
        let stems = s.separate(audio: audio, sampleRate: 44100, wiener: true)
        for target in SeparationTarget.allCases {
            for channel in stems[target]! {
                for v in channel {
                    XCTAssertFalse(v.isNaN, "\(target.rawValue) contains NaN")
                    XCTAssertFalse(v.isInfinite, "\(target.rawValue) contains Inf")
                }
            }
        }
    }

    /// Reconstruction sanity: summed stems should be in the same rough energy
    /// range as the input mix. Open-Unmix is not exact but should be close.
    func testStemsApproximateInputEnergy() throws {
        let s = try separator
        let audio = makeStereoTestSignal(seconds: 2.0)
        let stems = s.separate(audio: audio, sampleRate: 44100, wiener: true)

        func totalEnergy(_ stereo: [[Float]]) -> Float {
            var e: Float = 0
            for ch in stereo { for v in ch { e += v * v } }
            return e
        }
        let inputEnergy = totalEnergy(audio)
        var summedStems = [[Float](repeating: 0, count: audio[0].count),
                           [Float](repeating: 0, count: audio[1].count)]
        for target in SeparationTarget.allCases {
            let stem = stems[target]!
            for i in 0..<audio[0].count { summedStems[0][i] += stem[0][i] }
            for i in 0..<audio[1].count { summedStems[1][i] += stem[1][i] }
        }
        let summedEnergy = totalEnergy(summedStems)
        let ratio = summedEnergy / max(inputEnergy, 1e-9)
        print("Input energy=\(inputEnergy), summed=\(summedEnergy), ratio=\(ratio)")
        // Sum should be within a small-to-few-x range of the input mix.
        XCTAssertGreaterThan(ratio, 0.05, "Summed stems suspiciously quiet vs input")
        XCTAssertLessThan(ratio, 10.0,    "Summed stems suspiciously louder than input")
    }
}
