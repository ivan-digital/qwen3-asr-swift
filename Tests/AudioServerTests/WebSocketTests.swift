import XCTest
@testable import AudioServer

final class PCMConversionTests: XCTestCase {

    func testPCM16LEToFloat_silence() {
        // 4 samples of silence
        let data = Data(repeating: 0, count: 8)
        let floats = pcm16LEToFloat(data)
        XCTAssertEqual(floats.count, 4)
        for s in floats {
            XCTAssertEqual(s, 0.0, accuracy: 1e-6)
        }
    }

    func testPCM16LEToFloat_maxPositive() {
        // Int16.max = 32767 -> should be ~1.0
        var data = Data(count: 2)
        data.withUnsafeMutableBytes { $0.bindMemory(to: Int16.self)[0] = Int16(32767).littleEndian }
        let floats = pcm16LEToFloat(data)
        XCTAssertEqual(floats.count, 1)
        XCTAssertEqual(floats[0], 32767.0 / 32768.0, accuracy: 1e-5)
    }

    func testPCM16LEToFloat_maxNegative() {
        // Int16.min = -32768 -> should be -1.0
        var data = Data(count: 2)
        data.withUnsafeMutableBytes { $0.bindMemory(to: Int16.self)[0] = Int16(-32768).littleEndian }
        let floats = pcm16LEToFloat(data)
        XCTAssertEqual(floats.count, 1)
        XCTAssertEqual(floats[0], -1.0, accuracy: 1e-5)
    }

    func testFloatToPCM16LE_silence() {
        let samples: [Float] = [0, 0, 0, 0]
        let data = floatToPCM16LE(samples)
        XCTAssertEqual(data.count, 8)
        let floats = pcm16LEToFloat(data)
        for s in floats {
            XCTAssertEqual(s, 0.0, accuracy: 1e-6)
        }
    }

    func testFloatToPCM16LE_clipping() {
        // Values beyond [-1, 1] should be clamped
        let samples: [Float] = [2.0, -2.0, 0.5]
        let data = floatToPCM16LE(samples)
        XCTAssertEqual(data.count, 6)
        let recovered = pcm16LEToFloat(data)
        XCTAssertEqual(recovered[0], 32767.0 / 32768.0, accuracy: 1e-4, "Clipped to +1")
        XCTAssertEqual(recovered[1], -1.0, accuracy: 1e-4, "Clipped to -1")
        XCTAssertEqual(recovered[2], 0.5, accuracy: 0.001)
    }

    func testRoundTrip() {
        // Float -> PCM16LE -> Float should be close to original
        let original: [Float] = [-1.0, -0.5, 0.0, 0.5, 0.999]
        let data = floatToPCM16LE(original)
        let recovered = pcm16LEToFloat(data)
        XCTAssertEqual(recovered.count, original.count)
        for i in 0..<original.count {
            XCTAssertEqual(recovered[i], original[i], accuracy: 0.001,
                "Sample \(i): expected \(original[i]), got \(recovered[i])")
        }
    }

    func testEmptyInput() {
        let emptyFloat = pcm16LEToFloat(Data())
        XCTAssertEqual(emptyFloat.count, 0)

        let emptyData = floatToPCM16LE([])
        XCTAssertEqual(emptyData.count, 0)
    }
}

final class ResampleTests: XCTestCase {

    func testSameRate() {
        let input: [Float] = [1.0, 2.0, 3.0, 4.0]
        let output = resample(input, from: 24000, to: 24000)
        XCTAssertEqual(output, input)
    }

    func testEmpty() {
        let output = resample([], from: 24000, to: 16000)
        XCTAssertEqual(output.count, 0)
    }

    func testDownsample24kTo16k() {
        // 24000 samples at 24kHz = 1 second → should produce ~16000 samples at 16kHz
        // AVAudioConverter (sinc interpolation) may produce slightly fewer samples
        let input = [Float](repeating: 0.5, count: 24000)
        let output = resample(input, from: 24000, to: 16000)
        let expectedCount = 16000
        XCTAssertTrue(abs(output.count - expectedCount) <= 20, "Expected ~\(expectedCount), got \(output.count)")
        // Sinc filter causes ramp-up/down at edges; check bulk (skip first/last 100 samples)
        let margin = min(100, output.count / 4)
        for i in margin..<(output.count - margin) {
            XCTAssertEqual(output[i], 0.5, accuracy: 0.01, "Sample \(i)")
        }
    }

    func testDownsamplePreservesShape() {
        // A longer ramp to avoid edge effects from sinc filter
        let input = (0..<2400).map { Float($0) / 2399.0 }
        let output = resample(input, from: 24000, to: 16000)
        let expectedCount = 1600
        XCTAssertTrue(abs(output.count - expectedCount) <= 20, "Expected ~\(expectedCount), got \(output.count)")
        // First sample should be ~0, last should be ~1
        XCTAssertEqual(output[0], 0.0, accuracy: 0.05)
        XCTAssertEqual(output[output.count - 1], 1.0, accuracy: 0.1)
    }
}

final class FormatJSONTests: XCTestCase {

    func testBasicJSON() {
        let json = formatJSON(["key": "value"])
        XCTAssertTrue(json.contains("\"key\""))
        XCTAssertTrue(json.contains("\"value\""))
    }

    func testBoolJSON() {
        let json = formatJSON(["done": true])
        XCTAssertTrue(json.contains("true"))
    }

    func testNumericJSON() {
        let json = formatJSON(["count": 42])
        XCTAssertTrue(json.contains("42"))
    }

    func testSortedKeys() {
        let json = formatJSON(["b": 2, "a": 1] as [String: Any])
        // "a" should appear before "b" in sorted output
        guard let aIdx = json.range(of: "\"a\""),
              let bIdx = json.range(of: "\"b\"") else {
            XCTFail("Keys not found"); return
        }
        XCTAssertLessThan(aIdx.lowerBound, bIdx.lowerBound)
    }
}
