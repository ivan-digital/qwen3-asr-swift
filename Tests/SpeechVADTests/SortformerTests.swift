import XCTest
@testable import SpeechVAD
import AudioCommon

final class SortformerTests: XCTestCase {

    // MARK: - Config Tests

    func testDefaultConfig() {
        let config = SortformerConfig.default
        XCTAssertEqual(config.nMels, 128)
        XCTAssertEqual(config.nFFT, 400)
        XCTAssertEqual(config.hopLength, 160)
        XCTAssertEqual(config.sampleRate, 16000)
        XCTAssertEqual(config.spkcacheLen, 188)
        XCTAssertEqual(config.fifoLen, 40)
        XCTAssertEqual(config.fcDModel, 512)
        XCTAssertEqual(config.maxSpeakers, 4)
        XCTAssertEqual(config.subsamplingFactor, 8)
        XCTAssertEqual(config.onset, 0.5, accuracy: 0.001)
        XCTAssertEqual(config.offset, 0.3, accuracy: 0.001)
    }

    func testCustomConfig() {
        let config = SortformerConfig(
            nMels: 80,
            onset: 0.6,
            offset: 0.4,
            minSpeechDuration: 0.5
        )
        XCTAssertEqual(config.nMels, 80)
        XCTAssertEqual(config.onset, 0.6, accuracy: 0.001)
        XCTAssertEqual(config.offset, 0.4, accuracy: 0.001)
        XCTAssertEqual(config.minSpeechDuration, 0.5, accuracy: 0.001)
        // Other fields should have defaults
        XCTAssertEqual(config.hopLength, 160)
        XCTAssertEqual(config.sampleRate, 16000)
    }

    // MARK: - Mel Extractor Tests

    func testMelExtractorOutputShape() {
        let config = SortformerConfig.default
        let extractor = SortformerMelExtractor(config: config)

        // 1 second of audio at 16kHz
        let audio = [Float](repeating: 0.1, count: 16000)
        let (melSpec, nFrames) = extractor.extract(audio)

        // Expected frames: (16000 + 200 + 200 - 400) / 160 + 1 = 101
        // With reflect padding: (16000) / 160 + 1 = 101
        XCTAssertGreaterThan(nFrames, 90, "Should produce ~100 frames for 1s audio")
        XCTAssertLessThan(nFrames, 110, "Should produce ~100 frames for 1s audio")
        XCTAssertEqual(melSpec.count, nFrames * 128, "Flat array should be nFrames * 128")
    }

    func testMelExtractorEmptyAudio() {
        let extractor = SortformerMelExtractor()
        let (melSpec, nFrames) = extractor.extract([])

        // Empty audio should produce zero-length output (padded frame at most)
        XCTAssertEqual(melSpec.count, nFrames * 128)
    }

    func testMelExtractorShortAudio() {
        let extractor = SortformerMelExtractor()

        // Very short: 400 samples (25ms at 16kHz)
        let audio = [Float](repeating: 0.5, count: 400)
        let (melSpec, nFrames) = extractor.extract(audio)

        XCTAssertGreaterThan(nFrames, 0, "Should produce at least 1 frame")
        XCTAssertEqual(melSpec.count, nFrames * 128)

        // Values should be finite
        for val in melSpec {
            XCTAssertFalse(val.isNaN, "Mel values should not be NaN")
            XCTAssertFalse(val.isInfinite, "Mel values should not be infinite")
        }
    }

    func testMelExtractorValuesFinite() {
        let extractor = SortformerMelExtractor()

        // Sine wave at 440Hz
        let audio = (0..<16000).map { i in
            sinf(2.0 * Float.pi * 440.0 * Float(i) / 16000.0) * 0.5
        }
        let (melSpec, nFrames) = extractor.extract(audio)

        XCTAssertGreaterThan(nFrames, 0)
        for val in melSpec {
            XCTAssertFalse(val.isNaN)
            XCTAssertFalse(val.isInfinite)
        }

        // 440Hz should produce a peak in a low-mid mel bin, not at the very top
        // Find the bin with max energy in the first frame
        var maxBin = 0
        var maxEnergy: Float = -Float.infinity
        for b in 0..<128 {
            if melSpec[b] > maxEnergy {
                maxEnergy = melSpec[b]
                maxBin = b
            }
        }
        // 440Hz maps to mel bin ~20-30 (out of 128), should be in lower half
        XCTAssertLessThan(maxBin, 64,
                          "440Hz peak should be in the lower half of mel bins (got bin \(maxBin))")
    }

    // MARK: - Binarization Tests (reuses PowersetDecoder.binarize)

    func testBinarizationSingleSpeaker() {
        // Simulate a single speaker active from frames 10-50
        let numFrames = 100
        let frameDuration: Float = 0.01  // 10ms per frame
        var probs = [Float](repeating: 0.0, count: numFrames)
        for i in 10..<50 {
            probs[i] = 0.9
        }

        let segments = PowersetDecoder.binarize(
            probs: probs, onset: 0.5, offset: 0.3, frameDuration: frameDuration)

        XCTAssertEqual(segments.count, 1, "Should detect exactly 1 segment")
        if let seg = segments.first {
            XCTAssertEqual(seg.startTime, 0.1, accuracy: 0.02)
            XCTAssertEqual(seg.endTime, 0.5, accuracy: 0.02)
        }
    }

    func testBinarizationHysteresis() {
        // Test that offset < onset creates hysteresis
        let numFrames = 100
        let frameDuration: Float = 0.01
        var probs = [Float](repeating: 0.0, count: numFrames)

        // Rising above onset at frame 10
        for i in 10..<20 { probs[i] = 0.8 }
        // Dip between onset and offset (should stay active)
        for i in 20..<30 { probs[i] = 0.4 }
        // Back up
        for i in 30..<40 { probs[i] = 0.8 }
        // Drop below offset at frame 40

        let segments = PowersetDecoder.binarize(
            probs: probs, onset: 0.5, offset: 0.3, frameDuration: frameDuration)

        // The dip to 0.4 is above offset (0.3), so it should be one continuous segment
        XCTAssertEqual(segments.count, 1,
                       "Hysteresis should merge segments when prob stays above offset")
    }

    func testBinarizationNoSpeech() {
        let probs = [Float](repeating: 0.1, count: 100)
        let segments = PowersetDecoder.binarize(
            probs: probs, onset: 0.5, offset: 0.3, frameDuration: 0.01)
        XCTAssertTrue(segments.isEmpty, "Should produce no segments for low probs")
    }

    // MARK: - State Buffer Tests

    func testStateBufferDimensions() {
        let config = SortformerConfig.default

        // Verify state buffer sizes
        let spkcacheSize = config.spkcacheLen * config.fcDModel
        XCTAssertEqual(spkcacheSize, 188 * 512, "Speaker cache should be 188 * 512")

        let fifoSize = config.fifoLen * config.fcDModel
        XCTAssertEqual(fifoSize, 40 * 512, "FIFO should be 40 * 512")
    }

    #if canImport(CoreML)
    // MARK: - E2E Integration Test (requires model download)

    func testE2EWithRealModel() async throws {
        let diarizer: SortformerDiarizer
        do {
            diarizer = try await SortformerDiarizer.fromPretrained()
        } catch {
            throw XCTSkip("Sortformer model not cached: \(error)")
        }

        let audioURL = URL(fileURLWithPath: "Tests/Qwen3ASRTests/Resources/test_audio.wav")
        guard FileManager.default.fileExists(atPath: audioURL.path) else {
            throw XCTSkip("Test audio file not found")
        }

        let audio = try AudioFileLoader.load(
            url: audioURL, targetSampleRate: 16000)

        let result = diarizer.diarize(
            audio: audio, sampleRate: 16000, config: .default)

        XCTAssertGreaterThanOrEqual(result.segments.count, 1,
                                     "Should detect at least 1 segment")

        for seg in result.segments {
            XCTAssertGreaterThanOrEqual(seg.startTime, 0)
            XCTAssertGreaterThan(seg.endTime, seg.startTime)
            XCTAssertGreaterThanOrEqual(seg.speakerId, 0)
            XCTAssertLessThan(seg.speakerId, 4)
        }

        // Speaker embeddings should be empty for Sortformer (end-to-end)
        XCTAssertTrue(result.speakerEmbeddings.isEmpty)
    }
    #endif
}
