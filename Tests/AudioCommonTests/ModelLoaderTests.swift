import XCTest
@testable import AudioCommon

final class ModelLoaderTests: XCTestCase {

    // MARK: - Mock Models

    final class MockVAD: StreamingVADProvider, @unchecked Sendable {
        var inputSampleRate: Int { 16000 }
        var chunkSize: Int { 512 }
        func processChunk(_ samples: [Float]) -> Float { 0.0 }
        func resetState() {}
    }

    final class MockSTT: SpeechRecognitionModel, @unchecked Sendable {
        var inputSampleRate: Int { 16000 }
        func transcribe(audio: [Float], sampleRate: Int, language: String?) -> String { "hello" }
        func transcribeWithLanguage(audio: [Float], sampleRate: Int, language: String?) -> TranscriptionResult {
            TranscriptionResult(text: "hello")
        }
    }

    final class MockTTS: SpeechGenerationModel, @unchecked Sendable {
        var sampleRate: Int { 24000 }
        func generate(text: String, language: String?) async throws -> [Float] { [0.1, 0.2] }
        func generateStream(text: String, language: String?) -> AsyncThrowingStream<AudioChunk, Error> {
            AsyncThrowingStream { $0.finish() }
        }
    }

    // MARK: - Tests

    func testLoadEmpty() async throws {
        let models = try await ModelLoader.load([])
        XCTAssertNil(models.vad)
        XCTAssertNil(models.stt)
        XCTAssertNil(models.tts)
    }

    func testLoadAllModels() async throws {
        var progressUpdates: [(Double, String)] = []

        let models = try await ModelLoader.load([
            .vad { _ in MockVAD() },
            .stt { _ in MockSTT() },
            .tts { _ in MockTTS() },
        ], onProgress: { progress, stage in
            progressUpdates.append((progress, stage))
        })

        XCTAssertNotNil(models.vad)
        XCTAssertNotNil(models.stt)
        XCTAssertNotNil(models.tts)

        // Progress should reach 1.0
        XCTAssertEqual(progressUpdates.last?.0, 1.0)
        XCTAssertEqual(progressUpdates.last?.1, "Ready")
    }

    func testLoadSubset() async throws {
        let models = try await ModelLoader.load([
            .vad { _ in MockVAD() },
            .stt { _ in MockSTT() },
        ])

        XCTAssertNotNil(models.vad)
        XCTAssertNotNil(models.stt)
        XCTAssertNil(models.tts)
    }

    func testProgressIsMonotonic() async throws {
        var progressValues: [Double] = []

        _ = try await ModelLoader.load([
            .vad { p in p(0.5, "downloading"); p(1.0, "done"); return MockVAD() },
            .stt { p in p(0.5, "downloading"); p(1.0, "done"); return MockSTT() },
            .tts { p in p(0.5, "downloading"); p(1.0, "done"); return MockTTS() },
        ], onProgress: { progress, _ in
            progressValues.append(progress)
        })

        // Final value should be 1.0
        XCTAssertEqual(progressValues.last, 1.0)
        // All values should be in [0, 1]
        for v in progressValues {
            XCTAssertGreaterThanOrEqual(v, 0.0)
            XCTAssertLessThanOrEqual(v, 1.0)
        }
    }

    func testErrorIdentifiesFailedModel() async throws {
        do {
            _ = try await ModelLoader.load([
                .vad { _ in MockVAD() },
                .stt { _ in throw AudioModelError.modelLoadFailed(modelId: "test", reason: "test error") },
            ])
            XCTFail("Should have thrown")
        } catch {
            let desc = error.localizedDescription
            XCTAssertTrue(desc.contains("test error"), "Error should contain reason: \(desc)")
        }
    }

    func testStageNamesIncludeModelName() async throws {
        var stages: [String] = []

        _ = try await ModelLoader.load([
            .tts { p in p(0.5, "downloading"); return MockTTS() },
        ], onProgress: { _, stage in
            stages.append(stage)
        })

        let hasTTSStage = stages.contains { $0.contains("TTS") }
        XCTAssertTrue(hasTTSStage, "Stage names should mention TTS: \(stages)")
    }

    // MARK: - Parallel Loading

    /// Group 0 models (VAD + STT) should load concurrently.
    func testGroup0LoadsInParallel() async throws {
        let vadStarted = expectation(description: "VAD started")
        let sttStarted = expectation(description: "STT started")
        let bothStarted = expectation(description: "both running")
        bothStarted.expectedFulfillmentCount = 2

        let vadRunning = Atomic(false)
        let sttRunning = Atomic(false)

        _ = try await ModelLoader.load([
            .vad { _ in
                vadStarted.fulfill()
                vadRunning.set(true)
                bothStarted.fulfill()
                // Brief delay so both can be running
                try await Task.sleep(nanoseconds: 50_000_000)
                vadRunning.set(false)
                return MockVAD()
            },
            .stt { _ in
                sttStarted.fulfill()
                sttRunning.set(true)
                bothStarted.fulfill()
                try await Task.sleep(nanoseconds: 50_000_000)
                sttRunning.set(false)
                return MockSTT()
            },
        ])

        await fulfillment(of: [vadStarted, sttStarted, bothStarted], timeout: 2)
    }

    /// Group 1 (TTS) should load AFTER group 0 completes.
    func testTTSLoadsAfterGroup0() async throws {
        var order: [String] = []
        let lock = NSLock()

        func record(_ name: String) {
            lock.lock(); order.append(name); lock.unlock()
        }

        _ = try await ModelLoader.load([
            .vad { _ in record("vad"); return MockVAD() },
            .stt { _ in record("stt"); return MockSTT() },
            .tts { _ in record("tts"); return MockTTS() },
        ])

        // TTS should be last
        lock.lock()
        let finalOrder = order
        lock.unlock()

        XCTAssertEqual(finalOrder.last, "tts",
            "TTS should load after VAD+STT but got order: \(finalOrder)")
    }

    // MARK: - Error Handling

    /// Error in one group 0 model should cancel the other.
    func testGroup0ErrorCancelsParallel() async throws {
        do {
            _ = try await ModelLoader.load([
                .vad { _ in
                    try await Task.sleep(nanoseconds: 100_000_000)
                    return MockVAD()
                },
                .stt { _ in
                    throw AudioModelError.modelLoadFailed(modelId: "parakeet", reason: "download failed")
                },
            ])
            XCTFail("Should have thrown")
        } catch {
            // Error should propagate from the failed STT
            XCTAssertTrue(error.localizedDescription.contains("download failed"))
        }
    }

    /// Error in group 0 should prevent group 1 from loading.
    func testGroup0ErrorSkipsGroup1() async throws {
        var ttsLoaded = false

        do {
            _ = try await ModelLoader.load([
                .stt { _ in
                    throw AudioModelError.modelLoadFailed(modelId: "test", reason: "broken")
                },
                .tts { _ in
                    ttsLoaded = true
                    return MockTTS()
                },
            ])
            XCTFail("Should have thrown")
        } catch {
            XCTAssertFalse(ttsLoaded, "TTS should not load if group 0 failed")
        }
    }

    // MARK: - Progress Details

    /// Progress from individual model callbacks should be reflected in overall progress.
    func testProgressForwardsSubProgress() async throws {
        var progressValues: [Double] = []

        _ = try await ModelLoader.load([
            .tts { p in
                p(0.0, "starting")
                p(0.25, "downloading")
                p(0.5, "halfway")
                p(0.75, "loading")
                p(1.0, "done")
                return MockTTS()
            },
        ], onProgress: { progress, _ in
            progressValues.append(progress)
        })

        // Should have intermediate progress values between 0 and 1
        XCTAssertTrue(progressValues.count >= 5,
            "Expected at least 5 progress updates, got \(progressValues.count)")
    }

    /// Stage strings should forward model-specific status.
    func testProgressForwardsStatusText() async throws {
        var stages: [String] = []

        _ = try await ModelLoader.load([
            .stt { p in
                p(0.5, "Downloading 85 MB...")
                return MockSTT()
            },
        ], onProgress: { _, stage in
            stages.append(stage)
        })

        let hasDownloadStage = stages.contains { $0.contains("85 MB") }
        XCTAssertTrue(hasDownloadStage,
            "Should forward download size in stage: \(stages)")
    }

    // MARK: - Single Model

    func testLoadOnlyVAD() async throws {
        let models = try await ModelLoader.load([.vad { _ in MockVAD() }])
        XCTAssertNotNil(models.vad)
        XCTAssertNil(models.stt)
        XCTAssertNil(models.tts)
    }

    func testLoadOnlyTTS() async throws {
        let models = try await ModelLoader.load([.tts { _ in MockTTS() }])
        XCTAssertNil(models.vad)
        XCTAssertNil(models.stt)
        XCTAssertNotNil(models.tts)
    }

    // MARK: - Helpers

    final class Atomic<T>: @unchecked Sendable {
        private var value: T
        private let lock = NSLock()
        init(_ value: T) { self.value = value }
        func set(_ newValue: T) { lock.lock(); value = newValue; lock.unlock() }
        func get() -> T { lock.lock(); defer { lock.unlock() }; return value }
    }
}
