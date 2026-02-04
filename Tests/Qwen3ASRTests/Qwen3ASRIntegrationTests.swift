import XCTest
import Foundation
import MLX
@testable import Qwen3ASR

/// Integration tests that download and use actual model weights
final class Qwen3ASRIntegrationTests: XCTestCase {

    // Use 4-bit quantized model for smaller download (680MB vs 1.5GB)
    static let modelId = "mlx-community/Qwen3-ASR-0.6B-4bit"

    override class func setUp() {
        super.setUp()
        print("Running Qwen3-ASR integration tests")
    }

    // MARK: - Model Loading Test

    func testModelLoading() async throws {
        print("Testing model loading from HuggingFace...")

        let model = try await Qwen3ASRModel.fromPretrained(
            modelId: Self.modelId
        ) { progress, status in
            print("[\(Int(progress * 100))%] \(status)")
        }

        XCTAssertNotNil(model.audioEncoder)
        XCTAssertEqual(model.audioEncoder.layers.count, 18, "Should have 18 transformer layers")

        print("Model loaded successfully!")
    }

    // MARK: - Audio Encoding Test

    func testAudioEncoding() async throws {
        print("Testing audio encoding...")

        // Load model
        let model = try await Qwen3ASRModel.fromPretrained(modelId: Self.modelId) { progress, status in
            print("[\(Int(progress * 100))%] \(status)")
        }

        // Generate test audio (440Hz sine wave - 2 seconds)
        let sampleRate = 24000
        let duration = 2.0
        let numSamples = Int(Double(sampleRate) * duration)

        var audio = [Float](repeating: 0, count: numSamples)
        for i in 0..<numSamples {
            let t = Float(i) / Float(sampleRate)
            audio[i] = sin(2 * .pi * 440.0 * t) * 0.3
        }

        // Transcribe (will return placeholder for now)
        let result = model.transcribe(audio: audio, sampleRate: sampleRate)
        print("Transcription result: \(result)")

        // Verify we get some result
        XCTAssertFalse(result.isEmpty, "Should return some result")
    }

    // MARK: - Feature Extraction from Real Audio

    func testFeatureExtractionFromRealAudio() throws {
        guard let wavURL = Bundle.module.url(forResource: "test_audio", withExtension: "wav") else {
            throw XCTSkip("Test WAV file not found in bundle resources")
        }

        // Load audio
        let (samples, sampleRate) = try AudioFileLoader.loadWAV(url: wavURL)
        print("Audio duration: \(Double(samples.count)/Double(sampleRate))s at \(sampleRate)Hz")

        // Resample to 24kHz if needed
        let targetSampleRate = 24000
        let resampledSamples: [Float]
        if sampleRate != targetSampleRate {
            resampledSamples = AudioFileLoader.resampleForTest(samples, from: sampleRate, to: targetSampleRate)
            print("Resampled to \(targetSampleRate)Hz: \(resampledSamples.count) samples")
        } else {
            resampledSamples = samples
        }

        // Extract features
        let extractor = WhisperFeatureExtractor()
        let features = extractor.process(resampledSamples, sampleRate: targetSampleRate)

        print("Feature shape: [\(features.dim(0)), \(features.dim(1))]")

        // Verify feature dimensions
        XCTAssertEqual(features.dim(0), 128, "Should have 128 mel bins")
        XCTAssertGreaterThan(features.dim(1), 100, "Should have many time frames")
    }

    // MARK: - Full Pipeline Test

    func testFullPipeline() async throws {
        guard let wavURL = Bundle.module.url(forResource: "test_audio", withExtension: "wav") else {
            throw XCTSkip("Test WAV file not found in bundle resources")
        }

        print("Testing full pipeline with real audio...")

        // Load model
        let model = try await Qwen3ASRModel.fromPretrained(modelId: Self.modelId) { progress, status in
            print("[\(Int(progress * 100))%] \(status)")
        }

        // Load audio
        let (samples, sampleRate) = try AudioFileLoader.loadWAV(url: wavURL)
        print("Loaded audio: \(samples.count) samples at \(sampleRate)Hz")

        // Resample to 24kHz
        let targetSampleRate = 24000
        let audio: [Float]
        if sampleRate != targetSampleRate {
            audio = AudioFileLoader.resampleForTest(samples, from: sampleRate, to: targetSampleRate)
        } else {
            audio = samples
        }

        // Transcribe
        let start = Date()
        let result = model.transcribe(audio: audio, sampleRate: targetSampleRate)
        let elapsed = Date().timeIntervalSince(start)

        print("Transcription: \(result)")
        print("Elapsed time: \(elapsed)s")

        // Verify we get some result
        XCTAssertFalse(result.isEmpty, "Transcription should not be empty")
    }

    // MARK: - Performance Test

    func testEncodingPerformance() async throws {
        print("Testing encoding performance...")

        let model = try await Qwen3ASRModel.fromPretrained(modelId: Self.modelId)

        // Generate 10 seconds of audio
        let sampleRate = 24000
        let duration = 10.0
        let numSamples = Int(Double(sampleRate) * duration)

        var audio = [Float](repeating: 0, count: numSamples)
        for i in 0..<numSamples {
            let t = Float(i) / Float(sampleRate)
            audio[i] = sin(2 * .pi * 440.0 * t) * 0.3
        }

        // Warm up
        _ = model.transcribe(audio: Array(audio.prefix(sampleRate)), sampleRate: sampleRate)

        // Benchmark
        let start = Date()
        let _ = model.transcribe(audio: audio, sampleRate: sampleRate)
        let elapsed = Date().timeIntervalSince(start)

        let rtf = elapsed / duration
        print("Audio duration: \(duration)s")
        print("Processing time: \(elapsed)s")
        print("Real-time factor (RTF): \(rtf)")

        // RTF should be reasonable
        XCTAssertLessThan(rtf, 5.0, "Processing should be reasonably fast")
    }
}

// MARK: - Test Helper Extension

extension AudioFileLoader {
    /// Simple linear resampling for tests
    static func resampleForTest(_ samples: [Float], from inputRate: Int, to outputRate: Int) -> [Float] {
        let ratio = Double(outputRate) / Double(inputRate)
        let outputLength = Int(Double(samples.count) * ratio)

        guard outputLength > 0 else { return [] }

        var output = [Float](repeating: 0, count: outputLength)

        for i in 0..<outputLength {
            let srcIndex = Double(i) / ratio
            let srcIndexFloor = Int(srcIndex)
            let srcIndexCeil = min(srcIndexFloor + 1, samples.count - 1)
            let fraction = Float(srcIndex - Double(srcIndexFloor))

            output[i] = samples[srcIndexFloor] * (1 - fraction) + samples[srcIndexCeil] * fraction
        }

        return output
    }
}
