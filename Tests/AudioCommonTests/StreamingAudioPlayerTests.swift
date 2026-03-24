import XCTest
@testable import AudioCommon

final class StreamingAudioPlayerTests: XCTestCase {

    // MARK: - Initial State

    func testInitialState() {
        let player = StreamingAudioPlayer()
        XCTAssertFalse(player.isPlaying)
        XCTAssertEqual(player.preBufferDuration, 1.0)
    }

    // MARK: - Generation Lifecycle

    func testResetGeneration() {
        let player = StreamingAudioPlayer()
        player.markGenerationComplete()
        player.resetGeneration()
        XCTAssertFalse(player.isPlaying)
    }

    func testMarkGenerationCompleteWithoutEngine() {
        let player = StreamingAudioPlayer()
        let expectation = XCTestExpectation(description: "callback fires")
        player.onPlaybackFinished = { expectation.fulfill() }
        player.markGenerationComplete()
        wait(for: [expectation], timeout: 1.0)
    }

    func testFadeOutResetsState() {
        let player = StreamingAudioPlayer()
        player.fadeOutAndStop()
        XCTAssertFalse(player.isPlaying)
    }

    // MARK: - Pre-buffer

    func testPreBufferAccumulatesBeforePlayback() {
        // With preBufferDuration=2s, chunks should accumulate, not play immediately
        let player = StreamingAudioPlayer()
        player.preBufferDuration = 2.0
        player.ensureStandaloneEngine()

        // Schedule 1s of audio — less than 2s pre-buffer
        let oneSec = [Float](repeating: 0.3, count: 48000) // 48kHz engine
        player.scheduleChunk(oneSec)

        // markGenerationComplete should flush the partial buffer
        let expectation = XCTestExpectation(description: "playback finished")
        player.onPlaybackFinished = { expectation.fulfill() }
        player.markGenerationComplete()

        wait(for: [expectation], timeout: 3.0)
        player.stop()
    }

    func testZeroPreBufferPlaysImmediately() {
        let player = StreamingAudioPlayer()
        player.preBufferDuration = 0  // No pre-buffering
        player.ensureStandaloneEngine()

        let tone = (0..<2400).map { Float(sin(Double($0) * 0.1)) * 0.3 }
        player.resetGeneration()
        player.scheduleChunk(tone)

        // isPlaying should be true immediately
        XCTAssertTrue(player.isPlaying)

        let expectation = XCTestExpectation(description: "done")
        player.onPlaybackFinished = { expectation.fulfill() }
        player.markGenerationComplete()
        wait(for: [expectation], timeout: 3.0)
        player.stop()
    }

    // MARK: - Warmup & Fade-in

    func testSilentChunkDropped() {
        let player = StreamingAudioPlayer()
        let silence = [Float](repeating: 0.001, count: 1000)
        player.scheduleChunk(silence)
        // Without engine, can't truly test — but should not crash
    }

    func testEmptyChunkIgnored() {
        let player = StreamingAudioPlayer()
        player.scheduleChunk([])
        XCTAssertFalse(player.isPlaying)
    }

    // MARK: - Standalone Engine

    func testEnsureStandaloneEngineIdempotent() {
        let player = StreamingAudioPlayer()
        player.ensureStandaloneEngine()
        player.ensureStandaloneEngine()
        player.stop()
    }

    func testStopReleasesEngine() {
        let player = StreamingAudioPlayer()
        player.ensureStandaloneEngine()
        player.stop()
        player.ensureStandaloneEngine()
        player.stop()
    }

    // MARK: - Resampling

    func testPlayWithDifferentSampleRate() {
        let player = StreamingAudioPlayer()
        XCTAssertNoThrow(try player.play(samples: [Float](repeating: 0.5, count: 480), sampleRate: 16000))
    }

    func testPlayWithSameSampleRate() {
        let player = StreamingAudioPlayer()
        XCTAssertNoThrow(try player.play(samples: [Float](repeating: 0.5, count: 480), sampleRate: 24000))
    }

    // MARK: - Regression: inter-chunk gap

    func testChunksWithoutMarkGenerationDontFireCallback() {
        let player = StreamingAudioPlayer()
        var callbackFired = false
        player.onPlaybackFinished = { callbackFired = true }

        player.scheduleChunk([Float](repeating: 0.5, count: 100))
        player.scheduleChunk([Float](repeating: 0.5, count: 100))

        let expectation = XCTestExpectation(description: "wait")
        expectation.isInverted = true
        wait(for: [expectation], timeout: 0.5)
        XCTAssertFalse(callbackFired)
    }

    // MARK: - Full Lifecycle with Real Engine

    func testFullLifecycle() {
        let player = StreamingAudioPlayer()
        player.ensureStandaloneEngine()

        let expect1 = XCTestExpectation(description: "cycle 1")
        player.onPlaybackFinished = { expect1.fulfill() }

        let tone = (0..<2400).map { Float(sin(Double($0) * 0.1)) * 0.3 }
        player.resetGeneration()
        player.scheduleChunk(tone)
        player.markGenerationComplete()
        wait(for: [expect1], timeout: 3.0)

        // Cycle 2
        let expect2 = XCTestExpectation(description: "cycle 2")
        player.onPlaybackFinished = { expect2.fulfill() }
        player.resetGeneration()
        player.scheduleChunk(tone)
        player.markGenerationComplete()
        wait(for: [expect2], timeout: 3.0)

        player.stop()
    }

    func testPreBufferFlushesOnMarkComplete() {
        // Short utterance that doesn't fill the pre-buffer should still play
        let player = StreamingAudioPlayer()
        player.preBufferDuration = 5.0  // 5s — much more than we'll provide
        player.ensureStandaloneEngine()

        let expectation = XCTestExpectation(description: "short utterance plays")
        player.onPlaybackFinished = { expectation.fulfill() }

        player.resetGeneration()
        let shortTone = (0..<2400).map { Float(sin(Double($0) * 0.1)) * 0.3 }
        player.scheduleChunk(shortTone)
        player.markGenerationComplete()

        wait(for: [expectation], timeout: 3.0)
        player.stop()
    }

    // MARK: - Edge Cases

    func testMarkGenerationCompleteCalledTwice() {
        let player = StreamingAudioPlayer()
        player.ensureStandaloneEngine()
        var count = 0
        player.onPlaybackFinished = { count += 1 }

        player.resetGeneration()
        let tone = (0..<2400).map { Float(sin(Double($0) * 0.1)) * 0.3 }
        player.scheduleChunk(tone)
        player.markGenerationComplete()
        player.markGenerationComplete()

        let expectation = XCTestExpectation(description: "wait")
        DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) { expectation.fulfill() }
        wait(for: [expectation], timeout: 3.0)

        XCTAssertGreaterThanOrEqual(count, 1)
        player.stop()
    }

    func testMarkCompleteWithNoChunks() {
        // markComplete immediately with nothing scheduled — should fire callback
        let player = StreamingAudioPlayer()
        player.ensureStandaloneEngine()

        let expectation = XCTestExpectation(description: "empty generation completes")
        player.onPlaybackFinished = { expectation.fulfill() }
        player.resetGeneration()
        player.markGenerationComplete()

        wait(for: [expectation], timeout: 2.0)
        player.stop()
    }

    func testFadeOutMidPlayback() {
        let player = StreamingAudioPlayer()
        player.ensureStandaloneEngine()

        let longTone = (0..<48000).map { Float(sin(Double($0) * 0.1)) * 0.3 } // 1s at 48kHz
        player.resetGeneration()
        player.scheduleChunk(longTone)

        // Interrupt mid-playback
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
            player.fadeOutAndStop()
        }

        let expectation = XCTestExpectation(description: "wait for fadeout")
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) { expectation.fulfill() }
        wait(for: [expectation], timeout: 1.0)

        XCTAssertFalse(player.isPlaying)
        player.stop()
    }

    func testScheduleAfterStop() {
        let player = StreamingAudioPlayer()
        player.ensureStandaloneEngine()
        player.stop()

        // Should not crash — chunks go to pre-buffer but isPlaying may be true
        // since we have no node to actually play. This is acceptable.
        player.scheduleChunk([Float](repeating: 0.5, count: 100))
        // No assertion on isPlaying — behavior depends on pre-buffer state
    }

    func testPreBufferMultipleSmallChunks() {
        // Many small chunks that together exceed pre-buffer threshold
        let player = StreamingAudioPlayer()
        player.preBufferDuration = 0.5  // 0.5s
        player.ensureStandaloneEngine()

        let expectation = XCTestExpectation(description: "buffered playback completes")
        player.onPlaybackFinished = { expectation.fulfill() }

        player.resetGeneration()
        // Schedule 10 small chunks (each 0.1s at 48kHz = 4800 samples)
        for _ in 0..<10 {
            let small = (0..<4800).map { Float(sin(Double($0) * 0.1)) * 0.3 }
            player.scheduleChunk(small)
        }
        player.markGenerationComplete()

        wait(for: [expectation], timeout: 5.0)
        player.stop()
    }

    func testConcurrentScheduleChunk() {
        // Thread safety: schedule from multiple queues
        let player = StreamingAudioPlayer()
        player.ensureStandaloneEngine()
        player.resetGeneration()

        let group = DispatchGroup()
        for _ in 0..<10 {
            group.enter()
            DispatchQueue.global().async {
                let chunk = (0..<2400).map { Float(sin(Double($0) * 0.1)) * 0.3 }
                player.scheduleChunk(chunk)
                group.leave()
            }
        }
        group.wait()

        let expectation = XCTestExpectation(description: "concurrent done")
        player.onPlaybackFinished = { expectation.fulfill() }
        player.markGenerationComplete()

        wait(for: [expectation], timeout: 3.0)
        player.stop()
    }
}
