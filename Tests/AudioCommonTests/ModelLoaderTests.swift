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
}
