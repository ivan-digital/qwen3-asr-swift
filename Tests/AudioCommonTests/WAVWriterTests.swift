import XCTest
@testable import AudioCommon

final class WAVWriterTests: XCTestCase {

    func testWriteAndReadBack() throws {
        let samples: [Float] = [0.0, 0.5, -0.5, 1.0, -1.0]
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("test_\(UUID().uuidString).wav")
        defer { try? FileManager.default.removeItem(at: url) }

        try WAVWriter.write(samples: samples, sampleRate: 16000, to: url)

        // File should exist and be non-empty
        let data = try Data(contentsOf: url)
        XCTAssertGreaterThan(data.count, 44, "WAV file should have header + data")

        // Verify RIFF header
        let header = String(data: data[0..<4], encoding: .ascii)
        XCTAssertEqual(header, "RIFF")

        let wave = String(data: data[8..<12], encoding: .ascii)
        XCTAssertEqual(wave, "WAVE")
    }

    func testWriteEmptySamples() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("test_empty_\(UUID().uuidString).wav")
        defer { try? FileManager.default.removeItem(at: url) }

        try WAVWriter.write(samples: [], sampleRate: 16000, to: url)

        let data = try Data(contentsOf: url)
        // Should still have a valid header (44 bytes)
        XCTAssertGreaterThanOrEqual(data.count, 44)
    }

    func testRoundTripPreservesLength() throws {
        let sampleCount = 1000
        let samples = (0..<sampleCount).map { Float(sin(Double($0) * 0.1)) }
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("test_rt_\(UUID().uuidString).wav")
        defer { try? FileManager.default.removeItem(at: url) }

        try WAVWriter.write(samples: samples, sampleRate: 24000, to: url)

        // Read back via AudioFileLoader
        let loaded = try AudioFileLoader.load(url: url, targetSampleRate: 24000)
        // Allow small difference from PCM16 quantization
        XCTAssertEqual(loaded.count, sampleCount, accuracy: 2,
            "Round-trip should preserve sample count")
    }

    func testDifferentSampleRates() throws {
        let samples: [Float] = [0.1, 0.2, 0.3]

        for rate in [8000, 16000, 22050, 24000, 44100, 48000] {
            let url = FileManager.default.temporaryDirectory
                .appendingPathComponent("test_\(rate)_\(UUID().uuidString).wav")
            defer { try? FileManager.default.removeItem(at: url) }

            try WAVWriter.write(samples: samples, sampleRate: rate, to: url)
            let data = try Data(contentsOf: url)
            XCTAssertGreaterThan(data.count, 44, "Should write valid WAV at \(rate)Hz")
        }
    }
}

// Helper for approximate equality
extension Int {
    func assertApproximatelyEqual(to other: Int, accuracy: Int, _ message: String = "") {
        XCTAssertTrue(abs(self - other) <= accuracy, "\(self) != \(other) ± \(accuracy). \(message)")
    }
}
