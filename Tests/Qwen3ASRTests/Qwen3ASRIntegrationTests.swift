import XCTest
import Foundation
import MLX
@testable import Qwen3ASR
@testable import AudioCommon

/// Integration tests that download and use actual model weights
final class Qwen3ASRIntegrationTests: XCTestCase {

    // Use 4-bit quantized model for smaller download (680MB vs 1.5GB)
    static let modelId = "aufklarer/Qwen3-ASR-0.6B-MLX-4bit"

    override class func setUp() {
        super.setUp()
        print("Running Qwen3-ASR integration tests")
    }

    // MARK: - Model Loading Test

    func testModelLoading() async throws {
        print("Testing 0.6B model loading from HuggingFace...")

        let model = try await Qwen3ASRModel.fromPretrained(
            modelId: Self.modelId
        ) { progress, status in
            print("[\(Int(progress * 100))%] \(status)")
        }

        XCTAssertNotNil(model.audioEncoder)
        XCTAssertEqual(model.audioEncoder.layers.count, 18, "0.6B should have 18 transformer layers")
        XCTAssertEqual(model.audioEncoder.config.dModel, 896)
        XCTAssertEqual(model.audioEncoder.config.outputDim, 1024)

        print("0.6B model loaded successfully!")
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

        // Transcribe (sine wave is not speech — model may return empty or gibberish)
        let result = model.transcribe(audio: audio, sampleRate: sampleRate)
        print("Transcription result: \(result)")

        // Sine wave is not speech, so we just verify the pipeline completes without crashing
        // The result may be empty (model outputs only language tag) or contain gibberish
        XCTAssertNotNil(result, "Pipeline should complete without crashing")
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

        // Verify correct transcription
        // Test audio contains: "Can you guarantee that the replacement part will be shipped tomorrow?"
        XCTAssertFalse(result.isEmpty, "Transcription should not be empty")
        XCTAssertTrue(result.contains("guarantee"), "Should transcribe 'guarantee'")
        XCTAssertTrue(result.contains("replacement"), "Should transcribe 'replacement'")
        XCTAssertTrue(result.contains("shipped"), "Should transcribe 'shipped'")
        XCTAssertTrue(result.contains("tomorrow"), "Should transcribe 'tomorrow'")
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

    // MARK: - 1.7B Model Tests

    func testLargeModelLoading() async throws {
        let largeModelId = ASRModelSize.large.defaultModelId
        print("Testing 1.7B model loading: \(largeModelId)")

        let model = try await Qwen3ASRModel.fromPretrained(
            modelId: largeModelId
        ) { progress, status in
            print("[\(Int(progress * 100))%] \(status)")
        }

        XCTAssertNotNil(model.audioEncoder)
        XCTAssertEqual(model.audioEncoder.layers.count, 24, "1.7B should have 24 transformer layers")
        XCTAssertEqual(model.audioEncoder.config.dModel, 1024)
        XCTAssertEqual(model.audioEncoder.config.outputDim, 2048)
        XCTAssertEqual(model.textConfig.hiddenSize, 2048)
        XCTAssertEqual(model.textConfig.intermediateSize, 6144)
        XCTAssertEqual(model.textConfig.bits, 8)

        print("1.7B model loaded successfully!")
    }

    func testLargeModelFullPipeline() async throws {
        guard let wavURL = Bundle.module.url(forResource: "test_audio", withExtension: "wav") else {
            throw XCTSkip("Test WAV file not found in bundle resources")
        }

        let largeModelId = ASRModelSize.large.defaultModelId
        print("Testing 1.7B full pipeline with real audio...")

        let model = try await Qwen3ASRModel.fromPretrained(
            modelId: largeModelId
        ) { progress, status in
            print("[\(Int(progress * 100))%] \(status)")
        }

        let (samples, sampleRate) = try AudioFileLoader.loadWAV(url: wavURL)
        let targetSampleRate = 24000
        let audio: [Float]
        if sampleRate != targetSampleRate {
            audio = AudioFileLoader.resampleForTest(samples, from: sampleRate, to: targetSampleRate)
        } else {
            audio = samples
        }

        let start = Date()
        let result = model.transcribe(audio: audio, sampleRate: targetSampleRate)
        let elapsed = Date().timeIntervalSince(start)

        print("1.7B Transcription: \(result)")
        print("Elapsed time: \(elapsed)s")

        // Verify correct transcription
        // Test audio contains: "Can you guarantee that the replacement part will be shipped tomorrow?"
        XCTAssertFalse(result.isEmpty, "Transcription should not be empty")
        XCTAssertTrue(result.contains("guarantee"), "Should transcribe 'guarantee'")
        XCTAssertTrue(result.contains("replacement"), "Should transcribe 'replacement'")
        XCTAssertTrue(result.contains("shipped"), "Should transcribe 'shipped'")
        XCTAssertTrue(result.contains("tomorrow"), "Should transcribe 'tomorrow'")
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
