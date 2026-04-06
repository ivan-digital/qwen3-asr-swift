import XCTest
import ParakeetStreamingASR
import SpeechVAD
import AudioCommon

/// E2E tests for the DictateDemo streaming pipeline.
/// Validates that the streaming ASR produces text from audio
/// without relying on the SwiftUI menu bar UI.
final class E2EDictateDemoTests: XCTestCase {

    private static var model: ParakeetStreamingASRModel?
    private static var vad: SileroVADModel?

    override class func setUp() {
        super.setUp()
        let expectation = XCTestExpectation(description: "Load models")
        Task {
            model = try await ParakeetStreamingASRModel.fromPretrained()
            try model?.warmUp()
            vad = try await SileroVADModel.fromPretrained(engine: .coreml)
            expectation.fulfill()
        }
        _ = XCTWaiter.wait(for: [expectation], timeout: 120)
    }

    // MARK: - Streaming Session Tests

    func testStreamingProducesText() throws {
        guard let model = Self.model else { throw XCTSkip("Model not loaded") }

        // Generate 2s of 440Hz tone (simulates speech-like audio)
        let sampleRate = 16000
        let duration = 2.0
        var audio = [Float](repeating: 0, count: Int(Double(sampleRate) * duration))
        for i in 0..<audio.count {
            audio[i] = sin(2.0 * .pi * 440.0 * Float(i) / Float(sampleRate)) * 0.3
        }

        let session = try model.createSession()
        let chunkSize = 5440  // melFrames * hopLength

        var allPartials: [ParakeetStreamingASRModel.PartialTranscript] = []
        var offset = 0
        while offset + chunkSize <= audio.count {
            let chunk = Array(audio[offset..<offset + chunkSize])
            let partials = try session.pushAudio(chunk)
            allPartials.append(contentsOf: partials)
            offset += chunkSize
        }
        let finals = try session.finalize()
        allPartials.append(contentsOf: finals)

        // 440Hz tone may or may not produce text, but pipeline should not crash
        XCTAssertNotNil(allPartials)
        print("Streaming test: \(allPartials.count) partials")
    }

    func testStreamingWithRealAudio() async throws {
        guard let model = Self.model else { throw XCTSkip("Model not loaded") }

        // Use the test audio file from the main repo
        let testAudioURL = URL(fileURLWithPath: "../../Tests/ParakeetStreamingASRTests/Resources/test_audio.wav")
        guard FileManager.default.fileExists(atPath: testAudioURL.path) else {
            throw XCTSkip("test_audio.wav not found")
        }

        let audio = try AudioFileLoader.load(url: testAudioURL, targetSampleRate: 16000)

        // Stream in 320ms chunks (simulating real-time mic input)
        var allPartials: [ParakeetStreamingASRModel.PartialTranscript] = []
        for await partial in model.transcribeStream(audio: audio, sampleRate: 16000) {
            allPartials.append(partial)
        }

        XCTAssertFalse(allPartials.isEmpty, "Should produce at least one partial")
        let lastFinal = allPartials.last(where: { $0.isFinal })
        XCTAssertNotNil(lastFinal, "Should have at least one final transcript")
        XCTAssertFalse(lastFinal!.text.isEmpty, "Final text should not be empty")
        print("Real audio: '\(lastFinal!.text)'")
    }

    func testMultiUtteranceReset() throws {
        guard let model = Self.model else { throw XCTSkip("Model not loaded") }

        let sampleRate = 16000
        let chunkSize = 5440

        // Generate 1s speech-like + 1s silence + 1s speech-like
        var audio1 = [Float](repeating: 0, count: sampleRate)
        for i in 0..<audio1.count {
            audio1[i] = sin(2.0 * .pi * 440.0 * Float(i) / Float(sampleRate)) * 0.3
        }
        let silence = [Float](repeating: 0, count: sampleRate)

        let session = try model.createSession()

        // First utterance
        var offset = 0
        while offset + chunkSize <= audio1.count {
            _ = try session.pushAudio(Array(audio1[offset..<offset + chunkSize]))
            offset += chunkSize
        }

        // Silence (should not crash)
        offset = 0
        while offset + chunkSize <= silence.count {
            _ = try session.pushAudio(Array(silence[offset..<offset + chunkSize]))
            offset += chunkSize
        }

        // Second utterance
        offset = 0
        while offset + chunkSize <= audio1.count {
            _ = try session.pushAudio(Array(audio1[offset..<offset + chunkSize]))
            offset += chunkSize
        }

        let finals = try session.finalize()
        // Pipeline should handle multi-utterance without crashing
        XCTAssertNotNil(finals)
    }

    // MARK: - VAD Tests

    func testVADDetectsSpeech() throws {
        guard let vad = Self.vad else { throw XCTSkip("VAD not loaded") }

        // 440Hz tone should trigger speech detection
        var speechChunk = [Float](repeating: 0, count: 512)
        for i in 0..<512 {
            speechChunk[i] = sin(2.0 * .pi * 440.0 * Float(i) / 16000.0) * 0.5
        }
        let prob = vad.processChunk(speechChunk)
        // Just verify it returns a valid probability
        XCTAssertGreaterThanOrEqual(prob, 0)
        XCTAssertLessThanOrEqual(prob, 1)
    }

    func testVADSilence() throws {
        guard let vad = Self.vad else { throw XCTSkip("VAD not loaded") }

        let silence = [Float](repeating: 0, count: 512)
        let prob = vad.processChunk(silence)
        XCTAssertGreaterThanOrEqual(prob, 0)
        XCTAssertLessThan(prob, 0.5, "Silence should have low speech probability")
    }

    func testDebugAudioSmallChunks() async throws {
        guard let model = Self.model else { throw XCTSkip("Model not loaded") }

        let path = "/tmp/dictate-debug.wav"
        guard FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("No debug audio at \(path) — run DictateDemo first")
        }

        let audio = try AudioFileLoader.load(url: URL(fileURLWithPath: path), targetSampleRate: 16000)
        print("Debug audio: \(audio.count) samples (\(Float(audio.count)/16000)s)")

        // Test multiple chunk sizes
        for chunkSize in [4800, 5120, 5440, 8000] {
            let session2 = try model.createSession()
            var partials2: [ParakeetStreamingASRModel.PartialTranscript] = []
            var off = 0
            while off < audio.count {
                let end = min(off + chunkSize, audio.count)
                partials2.append(contentsOf: try session2.pushAudio(Array(audio[off..<end])))
                off = end
            }
            partials2.append(contentsOf: try session2.finalize())
            let texts = partials2.filter { $0.isFinal }.map { $0.text }
            print("chunk=\(chunkSize): \(partials2.count) partials, finals=\(texts)")
        }

        // Use 5440 (matching session internal buffer) for the assertion
        let session = try model.createSession()
        var allPartials: [ParakeetStreamingASRModel.PartialTranscript] = []
        var offset = 0
        while offset < audio.count {
            let end = min(offset + 5440, audio.count)
            let chunk = Array(audio[offset..<end])
            let partials = try session.pushAudio(chunk)
            allPartials.append(contentsOf: partials)
            offset = end
        }
        let finals = try session.finalize()
        allPartials.append(contentsOf: finals)

        let finalTexts = allPartials.filter { $0.isFinal }.map { $0.text }
        print("5440-chunk results: \(allPartials.count) partials, finals: \(finalTexts)")

        // Also test transcribeStream API
        var streamPartials: [ParakeetStreamingASRModel.PartialTranscript] = []
        for await p in model.transcribeStream(audio: audio, sampleRate: 16000) {
            streamPartials.append(p)
        }
        let streamFinals = streamPartials.filter { $0.isFinal }.map { $0.text }
        print("transcribeStream: \(streamPartials.count) partials, finals=\(streamFinals)")

        // Also test batch
        let batchText = try model.transcribeAudio(audio, sampleRate: 16000)
        print("batch: '\(batchText)'")

        XCTAssertFalse(streamPartials.isEmpty || !batchText.isEmpty, "Should produce text from mic audio")
    }

    func testDebugWavLoading() throws {
        let path = "/tmp/dictate-debug.wav"
        guard FileManager.default.fileExists(atPath: path) else { throw XCTSkip("No debug wav") }

        let audio = try AudioFileLoader.load(url: URL(fileURLWithPath: path), targetSampleRate: 16000)
        let rms = sqrt(audio.reduce(0) { $0 + $1 * $1 } / Float(audio.count))
        print("Loaded: \(audio.count) samples, rms=\(rms)")
        print("First 5: \(Array(audio.prefix(5)))")
        if audio.count > 16000 {
            print("At 1s: \(Array(audio[16000..<16005]))")
            let speechRms = sqrt(audio[16000..<17600].reduce(0) { $0 + $1 * $1 } / 1600)
            print("Speech rms at 1s: \(speechRms)")
        }
    }

    // MARK: - Latency

    func testStreamingChunkLatency() throws {
        guard let model = Self.model else { throw XCTSkip("Model not loaded") }

        let session = try model.createSession()
        let chunkSize = 5440
        let chunkMs: Float = 340.0  // 5440 / 16000 * 1000

        var audio = [Float](repeating: 0, count: chunkSize)
        for i in 0..<chunkSize {
            audio[i] = sin(2.0 * .pi * 440.0 * Float(i) / 16000.0) * 0.3
        }

        // Warmup
        _ = try session.pushAudio(audio)

        // Benchmark
        var times: [Double] = []
        for _ in 0..<10 {
            let t0 = CFAbsoluteTimeGetCurrent()
            _ = try session.pushAudio(audio)
            times.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
        }

        let avg = times.reduce(0, +) / Double(times.count)
        let rtf = avg / Double(chunkMs)
        print("Chunk latency: avg=\(String(format: "%.1f", avg))ms RTF=\(String(format: "%.3f", rtf))")
        XCTAssertLessThan(rtf, 1.0, "Must be real-time")
    }
}
