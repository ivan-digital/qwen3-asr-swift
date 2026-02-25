import XCTest
import MLX
@testable import SpeechVAD
import AudioCommon

final class SpeechVADTests: XCTestCase {

    // MARK: - Configuration Tests

    func testDefaultSegmentationConfig() {
        let config = SegmentationConfig.default
        XCTAssertEqual(config.sampleRate, 16000)
        XCTAssertEqual(config.sincnetFilters, [80, 60, 60])
        XCTAssertEqual(config.sincnetKernelSizes, [251, 5, 5])
        XCTAssertEqual(config.sincnetStrides, [10, 1, 1])
        XCTAssertEqual(config.sincnetPoolSizes, [3, 3, 3])
        XCTAssertEqual(config.lstmHiddenSize, 128)
        XCTAssertEqual(config.lstmNumLayers, 4)
        XCTAssertEqual(config.linearHiddenSize, 128)
        XCTAssertEqual(config.linearNumLayers, 2)
        XCTAssertEqual(config.numClasses, 7)
    }

    func testDefaultVADConfig() {
        let config = VADConfig.default
        XCTAssertEqual(config.onset, 0.767, accuracy: 0.001)
        XCTAssertEqual(config.offset, 0.377, accuracy: 0.001)
        XCTAssertEqual(config.minSpeechDuration, 0.136, accuracy: 0.001)
        XCTAssertEqual(config.minSilenceDuration, 0.067, accuracy: 0.001)
        XCTAssertEqual(config.windowDuration, 10.0, accuracy: 0.001)
        XCTAssertEqual(config.stepRatio, 0.1, accuracy: 0.001)
    }

    // MARK: - SpeechSegment Tests

    func testSpeechSegmentDuration() {
        let seg = SpeechSegment(startTime: 1.5, endTime: 3.7)
        XCTAssertEqual(seg.duration, 2.2, accuracy: 0.001)
    }

    // MARK: - Pipeline Windowing Tests

    func testWindowPositionsSingleWindow() {
        let pipeline = VADPipeline(sampleRate: 16000)
        // 5 seconds < 10 second window
        let positions = pipeline.windowPositions(numSamples: 80000)
        XCTAssertEqual(positions.count, 1)
        XCTAssertEqual(positions[0].start, 0)
        XCTAssertEqual(positions[0].end, 80000)
    }

    func testWindowPositionsMultipleWindows() {
        let pipeline = VADPipeline(sampleRate: 16000)
        // 30 seconds of audio
        let numSamples = 480000
        let positions = pipeline.windowPositions(numSamples: numSamples)

        // Window = 160000 samples (10s), step = 16000 samples (1s)
        // Should have multiple overlapping windows
        XCTAssertGreaterThan(positions.count, 1)

        // First window starts at 0
        XCTAssertEqual(positions[0].start, 0)

        // All windows should be within bounds
        for pos in positions {
            XCTAssertGreaterThanOrEqual(pos.start, 0)
            XCTAssertLessThanOrEqual(pos.end, numSamples)
        }

        // Last window should cover the end
        XCTAssertEqual(positions.last!.end, numSamples)
    }

    func testWindowPositionsEmpty() {
        let pipeline = VADPipeline(sampleRate: 16000)
        let positions = pipeline.windowPositions(numSamples: 0)
        XCTAssertTrue(positions.isEmpty)
    }

    // MARK: - Binarization Tests

    func testBinarizeAllSilence() {
        let pipeline = VADPipeline()
        let probs = [Float](repeating: 0.1, count: 293)
        let segments = pipeline.binarize(probs: probs)
        XCTAssertTrue(segments.isEmpty)
    }

    func testBinarizeAllSpeech() {
        let pipeline = VADPipeline()
        let probs = [Float](repeating: 0.9, count: 293)
        let segments = pipeline.binarize(probs: probs)
        // Should produce one long segment
        XCTAssertEqual(segments.count, 1)
        XCTAssertEqual(segments[0].startTime, 0.0, accuracy: 0.05)
    }

    func testBinarizeSpeechInMiddle() {
        let pipeline = VADPipeline()
        var probs = [Float](repeating: 0.1, count: 293)
        // Speech from frame 50 to 200
        for i in 50 ..< 200 {
            probs[i] = 0.9
        }
        let segments = pipeline.binarize(probs: probs)
        XCTAssertEqual(segments.count, 1)
        // Verify the segment covers roughly the right region
        XCTAssertGreaterThan(segments[0].startTime, 0.5)
        XCTAssertLessThan(segments[0].endTime, 8.0)
    }

    func testBinarizeHysteresis() {
        // Test that brief dips below onset but above offset don't split segments
        let pipeline = VADPipeline()
        var probs = [Float](repeating: 0.5, count: 293) // above offset, below onset
        // Speech onset
        for i in 20 ..< 50 {
            probs[i] = 0.9
        }
        // Brief dip (still above offset)
        for i in 50 ..< 55 {
            probs[i] = 0.5  // above offset (0.377), below onset (0.767)
        }
        // Speech continues
        for i in 55 ..< 100 {
            probs[i] = 0.9
        }
        // End: drop below offset
        for i in 100 ..< 120 {
            probs[i] = 0.1
        }

        let segments = pipeline.binarize(probs: probs)
        // Should be one continuous segment (the dip doesn't cross offset)
        XCTAssertEqual(segments.count, 1)
    }

    func testMinDurationFiltering() {
        let config = VADConfig(
            onset: 0.5,
            offset: 0.3,
            minSpeechDuration: 1.0,  // 1 second minimum
            minSilenceDuration: 0.5,
            windowDuration: 10.0,
            stepRatio: 0.1
        )
        let pipeline = VADPipeline(config: config)

        // Very short speech burst (only 3 frames = ~0.1s)
        var probs = [Float](repeating: 0.1, count: 293)
        for i in 50 ..< 53 {
            probs[i] = 0.9
        }
        let segments = pipeline.binarize(probs: probs)
        // Should be filtered out (too short)
        XCTAssertTrue(segments.isEmpty)
    }

    func testMinSilenceMerging() {
        let config = VADConfig(
            onset: 0.5,
            offset: 0.3,
            minSpeechDuration: 0.0,  // no minimum
            minSilenceDuration: 2.0,  // 2 second minimum silence
            windowDuration: 10.0,
            stepRatio: 0.1
        )
        let pipeline = VADPipeline(config: config)

        var probs = [Float](repeating: 0.1, count: 293)
        // Two speech segments separated by a short gap
        for i in 20 ..< 80 {
            probs[i] = 0.9
        }
        for i in 80 ..< 85 {  // ~0.17s gap — shorter than 2s min silence
            probs[i] = 0.1
        }
        for i in 85 ..< 150 {
            probs[i] = 0.9
        }
        let segments = pipeline.binarize(probs: probs)
        // Should be merged into one segment
        XCTAssertEqual(segments.count, 1)
    }

    // MARK: - Model Shape Tests (random weights, no download)

    func testSegmentationModelOutputShape() {
        // 10s of audio at 16kHz = 160000 samples
        let config = SegmentationConfig.default
        let model = SegmentationModel(config: config)

        let input = MLXRandom.normal([1, 1, 160000])
        let output = model(input)
        eval(output)

        // Should produce [1, 589, 7] for 10s @ 16kHz with padding=0
        XCTAssertEqual(output.shape[0], 1)     // batch
        XCTAssertEqual(output.shape[2], 7)     // classes
        // 160000 → SincConv(k=251,s=10): 15975 → Pool(3,3): 5325
        // → Conv(k=5): 5321 → Pool(3,3): 1773 → Conv(k=5): 1769 → Pool(3,3): 589
        XCTAssertEqual(output.shape[1], 589)

        // Softmax: each frame's class probs should sum to ~1
        let frameSum = output[0, 0].sum().item(Float.self)
        XCTAssertEqual(frameSum, 1.0, accuracy: 0.01)
    }

    func testSpeechProbabilityExtraction() {
        // Create fake posteriors: [1, 10, 7]
        let posteriors = MLXRandom.uniform(low: 0.0, high: 1.0, [1, 10, 7])
        let normalized = posteriors / posteriors.sum(axis: -1, keepDims: true)
        let speechProb = SegmentationModel.speechProbability(from: normalized)
        eval(speechProb)

        XCTAssertEqual(speechProb.shape, [1, 10])
        // Speech prob should be 1 - P(non-speech), so in [0, 1]
        let values = speechProb[0].asArray(Float.self)
        for v in values {
            XCTAssertGreaterThanOrEqual(v, 0.0)
            XCTAssertLessThanOrEqual(v, 1.0)
        }
    }

    func testSincNetOutputShape() {
        let config = SegmentationConfig.default
        let sincnet = SincNet(config: config)

        let input = MLXRandom.normal([1, 1, 160000])
        let output = sincnet(input)
        eval(output)

        // SincNet: [1, 1, 160000] → [1, 60, ~293]
        XCTAssertEqual(output.shape[0], 1)
        XCTAssertEqual(output.shape[1], 60)
        XCTAssertGreaterThan(output.shape[2], 200)
    }

    func testBiLSTMOutputShape() {
        let fwdLayers = [LSTMLayer(inputSize: 60, hiddenSize: 128)]
        let bwdLayers = [LSTMLayer(inputSize: 60, hiddenSize: 128)]
        let fwd = LSTMStack(layers: fwdLayers)
        let bwd = LSTMStack(layers: bwdLayers)

        let input = MLXRandom.normal([1, 50, 60])
        let output = runBiLSTM(input, fwd: fwd, bwd: bwd)
        eval(output)

        // BiLSTM: [1, 50, 60] → [1, 50, 256]
        XCTAssertEqual(output.shape, [1, 50, 256])
    }

    func testDetectSpeechWithRandomWeights() {
        // Full pipeline test with random weights (won't detect real speech,
        // but verifies the pipeline runs end-to-end without crashes)
        let segConfig = SegmentationConfig.default
        let vadConfig = VADConfig.default
        let model = SegmentationModel(config: segConfig)

        let vadModel = PyannoteVADModel(
            model: model, segConfig: segConfig, vadConfig: vadConfig)

        // 5s of random audio at 16kHz
        let audio = [Float](repeating: 0, count: 80000)
        let segments = vadModel.detectSpeech(audio: audio, sampleRate: 16000)

        // With random weights, might or might not detect "speech" — just verify no crash
        // and segments are valid
        for seg in segments {
            XCTAssertGreaterThanOrEqual(seg.startTime, 0)
            XCTAssertGreaterThan(seg.endTime, seg.startTime)
        }
    }

    // MARK: - E2E Integration Test (requires real weights)

    func testE2EWithRealWeights() async throws {
        // Load the real pre-trained model
        let vadModel = try await PyannoteVADModel.fromPretrained()

        // Load the test audio file
        let audioURL = URL(fileURLWithPath: "Tests/Qwen3ASRTests/Resources/test_audio.wav")
        guard FileManager.default.fileExists(atPath: audioURL.path) else {
            throw XCTSkip("Test audio file not found")
        }

        let (samples, sampleRate) = try AudioFileLoader.loadWAV(url: audioURL)
        XCTAssertGreaterThan(samples.count, 0)

        // Run VAD
        let segments = vadModel.detectSpeech(audio: samples, sampleRate: sampleRate)

        // Should detect exactly one speech segment (matches Python pyannote output)
        XCTAssertEqual(segments.count, 1, "Expected 1 speech segment, got \(segments.count)")

        if let seg = segments.first {
            // Python pyannote: [5.16s - 8.40s]
            // Swift:           [5.16s - 8.44s] (within ~0.05s tolerance)
            XCTAssertEqual(seg.startTime, 5.16, accuracy: 0.1,
                           "Start time should be ~5.16s (got \(seg.startTime))")
            XCTAssertEqual(seg.endTime, 8.42, accuracy: 0.1,
                           "End time should be ~8.40-8.44s (got \(seg.endTime))")
            XCTAssertGreaterThan(seg.duration, 3.0)
            XCTAssertLessThan(seg.duration, 4.0)
        }
    }

    // MARK: - Frame Aggregation Tests

    func testAggregateFramesSingleWindow() {
        let pipeline = VADPipeline(sampleRate: 16000)
        let probs: [Float] = (0 ..< 293).map { Float($0) / 293.0 }
        let positions = [(start: 0, end: 160000)]

        let aggregated = pipeline.aggregateFrames(
            windowProbs: [probs],
            positions: positions,
            numSamples: 160000
        )

        XCTAssertGreaterThan(aggregated.count, 0)
        // First frame should be near 0, last near 1
        XCTAssertLessThan(aggregated[0], 0.1)
    }
}
