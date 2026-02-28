import XCTest
import MLX
@testable import SpeechVAD
import AudioCommon

final class SileroVADTests: XCTestCase {

    // MARK: - Configuration Tests

    func testSileroDefaultVADConfig() {
        let config = VADConfig.sileroDefault
        XCTAssertEqual(config.onset, 0.5, accuracy: 0.001)
        XCTAssertEqual(config.offset, 0.35, accuracy: 0.001)
        XCTAssertEqual(config.minSpeechDuration, 0.25, accuracy: 0.001)
        XCTAssertEqual(config.minSilenceDuration, 0.1, accuracy: 0.001)
    }

    // MARK: - Model Shape Tests (random weights, no download)

    func testSileroNetworkForwardShape() {
        let network = SileroVADNetwork()

        // 576 samples = 64 context + 512 new
        let input = MLXRandom.normal([1, 576])
        let (prob, h, c) = network.forward(input, h: nil, c: nil)
        eval(prob, h, c)

        // Probability should be a scalar per batch element
        XCTAssertEqual(prob.shape, [1])

        // LSTM state: [1, batch, 128]
        XCTAssertEqual(h.shape, [1, 1, 128])
        XCTAssertEqual(c.shape, [1, 1, 128])

        // Probability should be in [0, 1] (sigmoid output)
        let p = prob.item(Float.self)
        XCTAssertGreaterThanOrEqual(p, 0.0)
        XCTAssertLessThanOrEqual(p, 1.0)
    }

    func testSileroNetworkStatefulLSTM() {
        let network = SileroVADNetwork()

        let input1 = MLXRandom.normal([1, 576])
        let input2 = MLXRandom.normal([1, 576])

        // First forward: no state
        let (prob1, h1, c1) = network.forward(input1, h: nil, c: nil)
        eval(prob1, h1, c1)

        // Second forward: with state from first (different input)
        let (prob2, h2, c2) = network.forward(input2, h: h1, c: c1)
        eval(prob2, h2, c2)

        // State should be different between calls (LSTM updates)
        let h1val = h1[0, 0, 0].item(Float.self)
        let h2val = h2[0, 0, 0].item(Float.self)
        // With random weights and different inputs, states will differ
        XCTAssertNotEqual(h1val, h2val, accuracy: 1e-6,
                          "LSTM state should change between calls")
    }

    func testReflectionPad1d() {
        // Test reflection padding: [1, 2, 3, 4, 5] with padding=2
        // Expected: [3, 2, 1, 2, 3, 4, 5, 4, 3]
        let x = MLXArray([Float](arrayLiteral: 1, 2, 3, 4, 5)).reshaped(1, 5, 1)
        let padded = reflectionPad1d(x, padding: 2)
        eval(padded)

        XCTAssertEqual(padded.shape, [1, 9, 1])

        let values = padded.squeezed().asArray(Float.self)
        XCTAssertEqual(values, [3, 2, 1, 2, 3, 4, 5, 4, 3])
    }

    // MARK: - SileroVADModel Tests (random weights)

    func testProcessChunk() {
        let network = SileroVADNetwork()
        let model = SileroVADModel(network: network)

        let samples = [Float](repeating: 0, count: SileroVADModel.chunkSize)
        let prob = model.processChunk(samples)

        XCTAssertGreaterThanOrEqual(prob, 0.0)
        XCTAssertLessThanOrEqual(prob, 1.0)
    }

    func testProcessChunkPrecondition() {
        let network = SileroVADNetwork()
        let model = SileroVADModel(network: network)

        // Wrong chunk size should trigger precondition
        // (Can't test precondition failure in XCTest without crashing)
        // Just verify correct size works
        let samples = [Float](repeating: 0, count: 512)
        let prob = model.processChunk(samples)
        XCTAssertGreaterThanOrEqual(prob, 0.0)
    }

    func testResetState() {
        let network = SileroVADNetwork()
        let model = SileroVADModel(network: network)

        // Process a chunk to populate state
        let samples = [Float](repeating: 0.1, count: 512)
        _ = model.processChunk(samples)

        // Reset
        model.resetState()

        // Process same chunk again — should produce same result as fresh model
        let prob1 = model.processChunk(samples)

        let model2 = SileroVADModel(network: network)
        let prob2 = model2.processChunk(samples)

        // Both should be equal since state was reset
        XCTAssertEqual(prob1, prob2, accuracy: 1e-5)
    }

    func testDetectSpeechWithRandomWeights() {
        let network = SileroVADNetwork()
        let model = SileroVADModel(network: network)

        // 5 seconds of silence at 16kHz
        let audio = [Float](repeating: 0, count: 80000)
        let segments = model.detectSpeech(audio: audio, sampleRate: 16000)

        // With random weights, verify no crash and valid segments
        for seg in segments {
            XCTAssertGreaterThanOrEqual(seg.startTime, 0)
            XCTAssertGreaterThan(seg.endTime, seg.startTime)
        }
    }

    // MARK: - StreamingVADProcessor Tests

    func testStreamingProcessorNoSpeech() {
        let network = SileroVADNetwork()
        let model = SileroVADModel(network: network)

        // Use very high onset to ensure no false positives
        let config = VADConfig(
            onset: 0.99, offset: 0.98,
            minSpeechDuration: 0.25, minSilenceDuration: 0.1,
            windowDuration: 0.032, stepRatio: 1.0
        )
        let processor = StreamingVADProcessor(model: model, config: config)

        let silence = [Float](repeating: 0, count: 16000)  // 1 second
        let events = processor.process(samples: silence)
        let flushEvents = processor.flush()

        // With onset=0.99 and silence input, should get no speech events
        let speechStarts = (events + flushEvents).filter {
            if case .speechStarted = $0 { return true }
            return false
        }
        XCTAssertTrue(speechStarts.isEmpty, "Should not detect speech in silence with high threshold")
    }

    func testStreamingProcessorBuffering() {
        let network = SileroVADNetwork()
        let model = SileroVADModel(network: network)
        let config = VADConfig.sileroDefault
        let processor = StreamingVADProcessor(model: model, config: config)

        // Feed less than one chunk
        let shortSamples = [Float](repeating: 0, count: 100)
        let events = processor.process(samples: shortSamples)

        // Should buffer, no events yet (not enough for a chunk)
        XCTAssertTrue(events.isEmpty, "Should buffer samples until chunk is complete")
    }

    func testStreamingProcessorReset() {
        let network = SileroVADNetwork()
        let model = SileroVADModel(network: network)
        let processor = StreamingVADProcessor(model: model, config: .sileroDefault)

        // Process some audio
        let samples = [Float](repeating: 0, count: 1024)
        _ = processor.process(samples: samples)

        // Reset
        processor.reset()

        // Current time should be 0
        XCTAssertEqual(processor.currentTime, 0.0)
    }

    func testStreamingProcessorFlushEmptyState() {
        let network = SileroVADNetwork()
        let model = SileroVADModel(network: network)
        let processor = StreamingVADProcessor(model: model, config: .sileroDefault)

        // Flush with no data
        let events = processor.flush()
        XCTAssertTrue(events.isEmpty)
    }

    func testStreamingProcessorChunkTiming() {
        let network = SileroVADNetwork()
        let model = SileroVADModel(network: network)
        let processor = StreamingVADProcessor(model: model, config: .sileroDefault)

        // Process exactly 2 chunks (1024 samples = 64ms)
        let samples = [Float](repeating: 0, count: 1024)
        _ = processor.process(samples: samples)

        // After 2 chunks, current time should be ~0.064s
        XCTAssertEqual(processor.currentTime, 0.064, accuracy: 0.001)
    }

    // MARK: - VADEvent Tests

    func testVADEventSendable() {
        // Verify VADEvent conforms to Sendable (compile-time check)
        let event: VADEvent = .speechStarted(time: 1.0)
        let _: any Sendable = event
        XCTAssertTrue(true)  // If this compiles, Sendable conformance works
    }

    // MARK: - E2E Integration Test (requires real weights)

    func testE2EWithRealWeights() async throws {
        let model = try await SileroVADModel.fromPretrained()

        let audioURL = URL(fileURLWithPath: "Tests/Qwen3ASRTests/Resources/test_audio.wav")
        guard FileManager.default.fileExists(atPath: audioURL.path) else {
            throw XCTSkip("Test audio file not found")
        }

        let (samples, sampleRate) = try AudioFileLoader.loadWAV(url: audioURL)
        XCTAssertGreaterThan(samples.count, 0)

        // Batch mode
        let segments = model.detectSpeech(audio: samples, sampleRate: sampleRate)

        // Should detect speech in the test audio (speech is around 5-8.5s)
        XCTAssertGreaterThanOrEqual(segments.count, 1,
                                     "Should detect at least 1 speech segment")

        if let seg = segments.first {
            // Speech region is approximately 5.16s - 8.44s
            XCTAssertGreaterThan(seg.startTime, 3.0,
                                 "Speech should start after 3s (got \(seg.startTime))")
            XCTAssertLessThan(seg.startTime, 7.0,
                              "Speech should start before 7s (got \(seg.startTime))")
            XCTAssertGreaterThan(seg.endTime, 7.0,
                                 "Speech should end after 7s (got \(seg.endTime))")
            XCTAssertLessThan(seg.endTime, 10.0,
                              "Speech should end before 10s (got \(seg.endTime))")
        }
    }

    func testE2EStreamingWithRealWeights() async throws {
        let model = try await SileroVADModel.fromPretrained()

        let audioURL = URL(fileURLWithPath: "Tests/Qwen3ASRTests/Resources/test_audio.wav")
        guard FileManager.default.fileExists(atPath: audioURL.path) else {
            throw XCTSkip("Test audio file not found")
        }

        let (samples, sampleRate) = try AudioFileLoader.loadWAV(url: audioURL)

        // Resample to 16kHz if needed
        let audio: [Float]
        if sampleRate != 16000 {
            // Simple linear resample
            let ratio = 16000.0 / Double(sampleRate)
            let outLen = Int(Double(samples.count) * ratio)
            audio = (0 ..< outLen).map { i in
                let src = Double(i) / ratio
                let idx = Int(src)
                let frac = Float(src - Double(idx))
                if idx + 1 < samples.count {
                    return samples[idx] * (1 - frac) + samples[idx + 1] * frac
                }
                return idx < samples.count ? samples[idx] : 0
            }
        } else {
            audio = samples
        }

        // Streaming mode
        let processor = StreamingVADProcessor(model: model)
        var allEvents = [VADEvent]()

        // Feed in 512-sample chunks
        var offset = 0
        while offset + 512 <= audio.count {
            let chunk = Array(audio[offset ..< offset + 512])
            allEvents.append(contentsOf: processor.process(samples: chunk))
            offset += 512
        }
        if offset < audio.count {
            allEvents.append(contentsOf: processor.process(samples: Array(audio[offset...])))
        }
        allEvents.append(contentsOf: processor.flush())

        // Should detect at least one speech segment
        let segments = allEvents.compactMap { event -> SpeechSegment? in
            if case .speechEnded(let seg) = event { return seg }
            return nil
        }
        XCTAssertGreaterThanOrEqual(segments.count, 1,
                                     "Streaming should detect at least 1 speech segment")

        // Verify segment is in the right ballpark
        if let seg = segments.first {
            XCTAssertGreaterThan(seg.duration, 1.0,
                                 "Speech segment should be at least 1s")
        }
    }

    // MARK: - CoreML E2E Tests (requires CoreML weights)

    func testE2ECoreMLWithRealWeights() async throws {
        let model: SileroVADModel
        do {
            model = try await SileroVADModel.fromPretrained(engine: .coreml)
        } catch {
            throw XCTSkip("CoreML model not cached: \(error)")
        }

        let audioURL = URL(fileURLWithPath: "Tests/Qwen3ASRTests/Resources/test_audio.wav")
        guard FileManager.default.fileExists(atPath: audioURL.path) else {
            throw XCTSkip("Test audio file not found")
        }

        let (samples, sampleRate) = try AudioFileLoader.loadWAV(url: audioURL)
        XCTAssertGreaterThan(samples.count, 0)

        // Batch mode
        let segments = model.detectSpeech(audio: samples, sampleRate: sampleRate)

        XCTAssertGreaterThanOrEqual(segments.count, 1,
                                     "CoreML should detect at least 1 speech segment")

        if let seg = segments.first {
            XCTAssertGreaterThan(seg.startTime, 3.0,
                                 "Speech should start after 3s (got \(seg.startTime))")
            XCTAssertLessThan(seg.startTime, 7.0,
                              "Speech should start before 7s (got \(seg.startTime))")
            XCTAssertGreaterThan(seg.endTime, 7.0,
                                 "Speech should end after 7s (got \(seg.endTime))")
            XCTAssertLessThan(seg.endTime, 10.0,
                              "Speech should end before 10s (got \(seg.endTime))")
        }
    }

    func testE2ECoreMLStreamingWithRealWeights() async throws {
        let model: SileroVADModel
        do {
            model = try await SileroVADModel.fromPretrained(engine: .coreml)
        } catch {
            throw XCTSkip("CoreML model not cached: \(error)")
        }

        let audioURL = URL(fileURLWithPath: "Tests/Qwen3ASRTests/Resources/test_audio.wav")
        guard FileManager.default.fileExists(atPath: audioURL.path) else {
            throw XCTSkip("Test audio file not found")
        }

        let (samples, sampleRate) = try AudioFileLoader.loadWAV(url: audioURL)

        // Resample to 16kHz if needed
        let audio: [Float]
        if sampleRate != 16000 {
            audio = AudioFileLoader.resample(samples, from: sampleRate, to: 16000)
        } else {
            audio = samples
        }

        // Streaming mode
        let processor = StreamingVADProcessor(model: model)
        var allEvents = [VADEvent]()

        var offset = 0
        while offset + 512 <= audio.count {
            let chunk = Array(audio[offset ..< offset + 512])
            allEvents.append(contentsOf: processor.process(samples: chunk))
            offset += 512
        }
        if offset < audio.count {
            allEvents.append(contentsOf: processor.process(samples: Array(audio[offset...])))
        }
        allEvents.append(contentsOf: processor.flush())

        let segments = allEvents.compactMap { event -> SpeechSegment? in
            if case .speechEnded(let seg) = event { return seg }
            return nil
        }
        XCTAssertGreaterThanOrEqual(segments.count, 1,
                                     "CoreML streaming should detect at least 1 speech segment")

        if let seg = segments.first {
            XCTAssertGreaterThan(seg.duration, 1.0,
                                 "Speech segment should be at least 1s")
        }
    }

    // MARK: - Per-Chunk Probability Tests

    func testE2EPerChunkProbabilities() async throws {
        let model = try await SileroVADModel.fromPretrained()

        let audioURL = URL(fileURLWithPath: "Tests/Qwen3ASRTests/Resources/test_audio.wav")
        guard FileManager.default.fileExists(atPath: audioURL.path) else {
            throw XCTSkip("Test audio file not found")
        }

        let (samples, sampleRate) = try AudioFileLoader.loadWAV(url: audioURL)

        // Resample to 16kHz
        let audio: [Float]
        if sampleRate != 16000 {
            let ratio = 16000.0 / Double(sampleRate)
            let outLen = Int(Double(samples.count) * ratio)
            audio = (0 ..< outLen).map { i in
                let src = Double(i) / ratio
                let idx = Int(src)
                let frac = Float(src - Double(idx))
                if idx + 1 < samples.count {
                    return samples[idx] * (1 - frac) + samples[idx + 1] * frac
                }
                return idx < samples.count ? samples[idx] : 0
            }
        } else {
            audio = samples
        }

        // Collect per-chunk probabilities
        model.resetState()
        var probs = [Float]()
        var offset = 0
        while offset + 512 <= audio.count {
            let chunk = Array(audio[offset ..< offset + 512])
            probs.append(model.processChunk(chunk))
            offset += 512
        }

        // Verify probabilities are in valid range
        for (i, p) in probs.enumerated() {
            XCTAssertGreaterThanOrEqual(p, 0.0, "Prob at chunk \(i) should be >= 0")
            XCTAssertLessThanOrEqual(p, 1.0, "Prob at chunk \(i) should be <= 1")
        }

        // The speech region (~5.16-8.44s) corresponds to chunks ~161-263 (at 32ms each)
        // There should be high probabilities in that range
        let speechStart = 5.0 / 0.032  // ~156
        let speechEnd = 8.5 / 0.032    // ~265

        if probs.count > Int(speechEnd) {
            let speechProbs = probs[Int(speechStart) ..< min(Int(speechEnd), probs.count)]
            let maxProb = speechProbs.max() ?? 0
            XCTAssertGreaterThan(maxProb, 0.3,
                                 "Should have high probability during speech region")
        }

        print("Per-chunk probabilities: \(probs.count) chunks")
        print("Max prob: \(probs.max() ?? 0)")
    }
}
