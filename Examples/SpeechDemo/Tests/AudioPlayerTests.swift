#if os(macOS)
import AVFoundation
import XCTest
@testable import SpeechDemo

final class AudioPlayerTests: XCTestCase {

    // MARK: - State machine tests (no audio hardware needed)

    /// markGenerationComplete with zero pending buffers fires callback immediately.
    func testMarkGenerationCompleteFiresWhenNoPendingBuffers() {
        let player = AudioPlayer()

        var finished = false
        player.onPlaybackFinished = { finished = true }

        player.markGenerationComplete()
        XCTAssertTrue(finished, "Should fire immediately when pendingBuffers == 0")
    }

    /// Without markGenerationComplete, callback never fires (even with no buffers).
    func testNoCallbackWithoutMarkGenerationComplete() {
        let player = AudioPlayer()

        var finished = false
        player.onPlaybackFinished = { finished = true }

        // Don't call markGenerationComplete
        RunLoop.main.run(until: Date().addingTimeInterval(0.05))
        XCTAssertFalse(finished, "Should not fire without markGenerationComplete")
    }

    /// resetGeneration prevents stale generationComplete from firing.
    func testResetGenerationClearsFlag() {
        let player = AudioPlayer()

        var finishCount = 0
        player.onPlaybackFinished = { finishCount += 1 }

        // First cycle: mark complete
        player.markGenerationComplete()
        XCTAssertEqual(finishCount, 1)

        // Reset for new cycle
        player.resetGeneration()

        // markGenerationComplete again — should fire (new cycle)
        player.markGenerationComplete()
        XCTAssertEqual(finishCount, 2)
    }

    /// stop() resets generationComplete, allowing clean next cycle.
    func testStopResetsGenerationComplete() {
        let player = AudioPlayer()

        var finishCount = 0
        player.onPlaybackFinished = { finishCount += 1 }

        player.markGenerationComplete()
        XCTAssertEqual(finishCount, 1)

        player.stop()

        // New cycle after stop
        player.markGenerationComplete()
        XCTAssertEqual(finishCount, 2)
    }

    /// Simulates the real timeline: chunks arrive → responseDone → last buffer completes.
    /// Without the fix, pendingBuffers hitting 0 between chunks would fire early.
    func testRaceConditionPrevented() throws {
        // This test verifies the fix conceptually:
        // 1. Player has no engine (play() returns early, so pendingBuffers stays 0)
        // 2. markGenerationComplete fires callback since pendingBuffers == 0
        // 3. The key invariant: without markGenerationComplete, callback NEVER fires
        //
        // The full race condition requires real audio scheduling (buffer completion handlers).
        // That requires audio hardware — tested manually via the Echo tab.

        let player = AudioPlayer()

        var callbackFired = false
        player.onPlaybackFinished = { callbackFired = true }

        // Simulate: chunks arrive but no engine → play() is a no-op
        let samples = [Float](repeating: 0.1, count: 2400)
        try player.play(samples: samples, sampleRate: 24000)

        // Without engine, playerNode is nil → pendingBuffers stays 0
        // But callback still shouldn't fire without markGenerationComplete
        RunLoop.main.run(until: Date().addingTimeInterval(0.05))
        XCTAssertFalse(callbackFired, "Callback must not fire without markGenerationComplete")

        // Signal done
        player.markGenerationComplete()
        XCTAssertTrue(callbackFired, "Callback fires after markGenerationComplete")
    }

    /// Two full cycles back-to-back (simulates two Echo responses).
    func testTwoCyclesBackToBack() {
        let player = AudioPlayer()

        var finishCount = 0
        player.onPlaybackFinished = { finishCount += 1 }

        // Cycle 1: responseCreated → chunks → responseDone
        player.resetGeneration()
        player.markGenerationComplete()
        XCTAssertEqual(finishCount, 1)

        // Cycle 2: responseCreated → chunks → responseDone
        player.resetGeneration()
        // Verify no premature fire
        RunLoop.main.run(until: Date().addingTimeInterval(0.05))
        XCTAssertEqual(finishCount, 1, "Must not fire after reset")

        player.markGenerationComplete()
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

        // No callback from interrupted cycle
        XCTAssertEqual(finishCount, 0)

        // New cycle after interrupt
        player.resetGeneration()
        player.markGenerationComplete()
        XCTAssertEqual(finishCount, 1)
    }
}

#endif
