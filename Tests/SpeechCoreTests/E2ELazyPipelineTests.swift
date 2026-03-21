#if canImport(CoreML)
import XCTest
@testable import SpeechCore
import AudioCommon
import SpeechVAD
import KokoroTTS
import ParakeetASR

/// E2E test for lazy pipeline loading and memory management.
/// Requires model downloads — skipped if models not cached.
final class E2ELazyPipelineTests: XCTestCase {

    /// Measure resident memory in MB.
    private func residentMB() -> Int {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
            }
        }
        guard result == KERN_SUCCESS else { return 0 }
        return Int(info.resident_size / (1024 * 1024))
    }

    /// Test that lazy pipeline creates without loading models.
    func testLazyPipelineDoesNotLoadModelsOnInit() async throws {
        let vad: SileroVADModel
        do {
            vad = try await SileroVADModel.fromPretrained(engine: .coreml)
        } catch {
            throw XCTSkip("Silero VAD not cached: \(error)")
        }

        let memBefore = residentMB()

        var config = PipelineConfig()
        config.mode = .echo
        config.autoUnloadModels = true

        let expectation = XCTestExpectation(description: "session created")

        let pipeline = VoicePipeline(
            sttFactory: {
                // This should NOT be called during init
                XCTFail("STT factory should not be called during pipeline init")
                throw NSError(domain: "test", code: 1)
            },
            ttsFactory: {
                XCTFail("TTS factory should not be called during pipeline init")
                throw NSError(domain: "test", code: 1)
            },
            vad: vad,
            config: config,
            onEvent: { event in
                if case .sessionCreated = event {
                    expectation.fulfill()
                }
            }
        )

        let memAfter = residentMB()
        let delta = memAfter - memBefore

        // Lazy pipeline init should add minimal memory (VAD only ~1MB, no STT/TTS)
        XCTAssertLessThan(delta, 50,
                          "Lazy pipeline init should not load heavy models (delta: \(delta)MB)")

        // Clean up — don't start the pipeline, just verify init was lazy
        _ = pipeline
    }

    /// Test that Kokoro TTS loads and generates audio correctly.
    func testKokoroTTSGeneratesAudio() async throws {
        let tts: KokoroTTSModel
        do {
            tts = try await KokoroTTSModel.fromPretrained { _, _ in }
        } catch {
            throw XCTSkip("Kokoro TTS not cached: \(error)")
        }

        let audio = try await tts.generate(text: "Hello world", language: "en")
        XCTAssertGreaterThan(audio.count, 1000,
                             "Should generate substantial audio for 'Hello world'")

        // Verify audio is valid float samples
        let hasSignal = audio.contains { abs($0) > 0.01 }
        XCTAssertTrue(hasSignal, "Audio should contain non-silent samples")
    }

    /// Test that Parakeet ASR loads and transcribes.
    func testParakeetASRTranscribes() async throws {
        let asr: ParakeetASRModel
        do {
            asr = try await ParakeetASRModel.fromPretrained { _, _ in }
        } catch {
            throw XCTSkip("Parakeet ASR not cached: \(error)")
        }

        let audioURL = URL(fileURLWithPath: "Tests/Qwen3ASRTests/Resources/test_audio.wav")
        guard FileManager.default.fileExists(atPath: audioURL.path) else {
            throw XCTSkip("Test audio file not found")
        }

        let audio = try AudioFileLoader.load(url: audioURL, targetSampleRate: 16000)
        let result = asr.transcribeWithLanguage(audio: audio, sampleRate: 16000, language: nil)

        XCTAssertFalse(result.text.isEmpty, "Should produce non-empty transcription")
    }

    /// Measure memory delta for loading and unloading a CoreML model.
    /// This test documents whether CoreML actually frees memory on nil.
    func testCoreMLMemoryReclamation() async throws {
        let memBaseline = residentMB()

        // Load Kokoro TTS (CoreML model)
        let tts: KokoroTTSModel
        do {
            tts = try await KokoroTTSModel.fromPretrained { _, _ in }
        } catch {
            throw XCTSkip("Kokoro TTS not cached: \(error)")
        }

        let memLoaded = residentMB()
        let loadDelta = memLoaded - memBaseline

        // Force deallocation
        withExtendedLifetime(tts) {}
        // tts goes out of scope here

        // Give CoreML time to clean up
        try await Task.sleep(nanoseconds: 2_000_000_000) // 2s

        let memAfterUnload = residentMB()
        let unloadDelta = memLoaded - memAfterUnload

        // Document the actual behavior (may or may not reclaim)
        print("""
        CoreML Memory Reclamation Test:
          Baseline: \(memBaseline) MB
          After load: \(memLoaded) MB (delta: +\(loadDelta) MB)
          After unload: \(memAfterUnload) MB (reclaimed: \(unloadDelta) MB)
          Reclamation ratio: \(loadDelta > 0 ? Double(unloadDelta) / Double(loadDelta) * 100 : 0)%
        """)

        // We don't assert reclamation (CoreML may not free) — this test documents behavior
        XCTAssertGreaterThan(loadDelta, 0,
                             "Loading a CoreML model should increase memory")
    }
}
#endif
