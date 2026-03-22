#if os(macOS)
import AVFoundation
import XCTest
@testable import SpeechDemo

final class AudioPlayerTests: XCTestCase {

    // MARK: - State machine tests (no audio hardware needed)

    /// markGenerationComplete with zero pending buffers fires callback (via main queue).
    func testMarkGenerationCompleteFiresWhenNoPendingBuffers() {
        let player = AudioPlayer()

        let exp = expectation(description: "playback finished")
        player.onPlaybackFinished = { exp.fulfill() }

        player.markGenerationComplete()
        wait(for: [exp], timeout: 1.0)
    }

    /// Without markGenerationComplete, callback never fires (even with no buffers).
    func testNoCallbackWithoutMarkGenerationComplete() {
        let player = AudioPlayer()

        var finished = false
        player.onPlaybackFinished = { finished = true }

        // Don't call markGenerationComplete
        RunLoop.main.run(until: Date().addingTimeInterval(0.1))
        XCTAssertFalse(finished, "Should not fire without markGenerationComplete")
    }

    /// resetGeneration prevents stale generationComplete from firing.
    func testResetGenerationClearsFlag() {
        let player = AudioPlayer()

        var finishCount = 0
        let exp = expectation(description: "two finishes")
        exp.expectedFulfillmentCount = 2
        player.onPlaybackFinished = {
            finishCount += 1
            exp.fulfill()
        }

        // First cycle: mark complete
        player.markGenerationComplete()

        // Reset for new cycle
        player.resetGeneration()

        // markGenerationComplete again — should fire (new cycle)
        player.markGenerationComplete()

        wait(for: [exp], timeout: 1.0)
        XCTAssertEqual(finishCount, 2)
    }

    /// stop() resets generationComplete, allowing clean next cycle.
    func testStopResetsGenerationComplete() {
        let player = AudioPlayer()

        var finishCount = 0
        let exp = expectation(description: "two finishes")
        exp.expectedFulfillmentCount = 2
        player.onPlaybackFinished = {
            finishCount += 1
            exp.fulfill()
        }

        player.markGenerationComplete()
        // Let first callback fire
        RunLoop.main.run(until: Date().addingTimeInterval(0.1))

        player.stop()

        // New cycle after stop
        player.markGenerationComplete()

        wait(for: [exp], timeout: 1.0)
        XCTAssertEqual(finishCount, 2)
    }

    /// Without markGenerationComplete, play() alone never triggers callback.
    func testRaceConditionPrevented() throws {
        let player = AudioPlayer()

        var callbackFired = false
        player.onPlaybackFinished = { callbackFired = true }

        // Simulate: chunks arrive but no engine → play() is a no-op
        let samples = [Float](repeating: 0.1, count: 2400)
        try player.play(samples: samples, sampleRate: 24000)

        // Without engine, playerNode is nil → pendingBuffers stays 0
        // But callback still shouldn't fire without markGenerationComplete
        RunLoop.main.run(until: Date().addingTimeInterval(0.1))
        XCTAssertFalse(callbackFired, "Callback must not fire without markGenerationComplete")

        // Signal done
        let exp = expectation(description: "callback fires")
        player.onPlaybackFinished = { exp.fulfill() }
        player.markGenerationComplete()
        wait(for: [exp], timeout: 1.0)
    }

    /// Two full cycles back-to-back (simulates two Echo responses).
    func testTwoCyclesBackToBack() {
        let player = AudioPlayer()

        var finishCount = 0
        player.onPlaybackFinished = { finishCount += 1 }

        // Cycle 1: responseCreated → chunks → responseDone
        player.resetGeneration()
        player.markGenerationComplete()
        RunLoop.main.run(until: Date().addingTimeInterval(0.1))
        XCTAssertEqual(finishCount, 1)

        // Cycle 2: responseCreated → chunks → responseDone
        player.resetGeneration()
        // Verify no premature fire
        RunLoop.main.run(until: Date().addingTimeInterval(0.1))
        XCTAssertEqual(finishCount, 1, "Must not fire after reset")

        player.markGenerationComplete()
        RunLoop.main.run(until: Date().addingTimeInterval(0.1))
        XCTAssertEqual(finishCount, 2)
    }

    /// Interrupt during playback: stop() mid-cycle, then new cycle works.
    func testInterruptThenNewCycle() {
        let player = AudioPlayer()

        var finishCount = 0
        player.onPlaybackFinished = { finishCount += 1 }

        // Start cycle but interrupt before markGenerationComplete
        player.resetGeneration()
        player.stop()  // User interrupted

        RunLoop.main.run(until: Date().addingTimeInterval(0.1))
        // No callback from interrupted cycle
        XCTAssertEqual(finishCount, 0)

        // New cycle after interrupt
        player.resetGeneration()
        player.markGenerationComplete()
        RunLoop.main.run(until: Date().addingTimeInterval(0.1))
        XCTAssertEqual(finishCount, 1)
    }
}

#endif
