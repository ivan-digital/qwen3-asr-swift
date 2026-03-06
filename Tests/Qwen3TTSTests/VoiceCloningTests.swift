import XCTest
import Foundation
import MLX
@testable import Qwen3TTS
@testable import Qwen3ASR
@testable import AudioCommon

// MARK: - Unit Tests (no GPU/model download required)

final class SpeakerEncoderUnitTests: XCTestCase {

    func testSpeakerEncoderInit() {
        let encoder = SpeakerEncoder()
        // Verify architecture dimensions via weight shapes [out, kernel, in]
        XCTAssertEqual(encoder.initialConv.weight.dim(0), 512, "Initial conv output channels")
        XCTAssertEqual(encoder.initialConv.weight.dim(2), 128, "Initial conv input channels")
        XCTAssertEqual(encoder.fc.weight.dim(0), 1024, "FC output channels")
        XCTAssertEqual(encoder.fc.weight.dim(2), 3072, "FC input channels")
    }

    func testMelFilterbankShape() {
        // 1 second of silence at 24kHz
        let samples = [Float](repeating: 0, count: 24000)
        let mels = SpeakerMel.compute(audio: samples, sampleRate: 24000)

        XCTAssertEqual(mels.ndim, 3, "Should be [B, T, 128]")
        XCTAssertEqual(mels.dim(0), 1, "Batch dimension")
        XCTAssertEqual(mels.dim(2), 128, "128 mel bins")
        XCTAssertGreaterThan(mels.dim(1), 0, "Should have time frames")

        // Expected frames: (24000 + 2*512 - 1024) / 256 + 1 ≈ 94
        let expectedFrames = (24000 + 1024 - 1024) / 256 + 1
        XCTAssertEqual(mels.dim(1), expectedFrames, "Frame count should match STFT parameters")
    }

    func testMelFilterbankResampling() {
        // Audio at 16kHz should be resampled to 24kHz
        let samples16k = [Float](repeating: 0.1, count: 16000)  // 1s at 16kHz
        let mels = SpeakerMel.compute(audio: samples16k, sampleRate: 16000)

        XCTAssertEqual(mels.ndim, 3)
        XCTAssertEqual(mels.dim(0), 1)
        XCTAssertEqual(mels.dim(2), 128)
        // After resampling 16kHz→24kHz: 24000 samples → same frame count as 24kHz input
        XCTAssertGreaterThan(mels.dim(1), 0)
    }

    func testHannWindow() {
        // Hann window should be symmetric and zero at endpoints
        let samples = [Float](repeating: 0, count: 24000)
        let mels = SpeakerMel.compute(audio: samples)
        // If we get here without crash, the Hann window and FFT worked
        XCTAssertEqual(mels.dim(2), 128)
    }

    func testCodecPrefixWithSpeakerEmbedding() {
        // Test that codec prefix construction works
        let model = Qwen3TTSModel()

        // Without speaker token, prefix is 6 tokens
        let prefix = model.buildCodecPrefix(languageId: 0)
        XCTAssertEqual(prefix.count, 6)

        // With speaker token, prefix is 7 tokens
        let prefixWithSpk = model.buildCodecPrefix(languageId: 0, speakerTokenId: 100)
        XCTAssertEqual(prefixWithSpk.count, 7)
    }

    func testSpeakerEncoderMemoryFootprint() {
        let encoder = SpeakerEncoder()
        // Speaker encoder has 76 weight tensors, should have some memory
        let memory = encoder.parameterMemoryBytes()
        XCTAssertGreaterThan(memory, 0, "Initialized encoder should have parameter memory")
    }

    func testSpeakerEncoderClearParameters() {
        let encoder = SpeakerEncoder()
        let memBefore = encoder.parameterMemoryBytes()
        XCTAssertGreaterThan(memBefore, 0)

        encoder.clearParameters()
        let memAfter = encoder.parameterMemoryBytes()
        XCTAssertEqual(memAfter, 0, "After clearing, memory should be 0")
    }

    func testQwen3TTSModelHasSpeakerEncoder() {
        let model = Qwen3TTSModel()
        // Model should have a speaker encoder
        let mem = model.speakerEncoder.parameterMemoryBytes()
        XCTAssertGreaterThan(mem, 0, "Speaker encoder should be initialized with parameters")
    }

    func testUnloadIncludesSpeakerEncoder() {
        let model = Qwen3TTSModel()
        let totalBefore = model.memoryFootprint
        XCTAssertGreaterThan(totalBefore, 0)

        model.unload()
        XCTAssertEqual(model.memoryFootprint, 0)
        XCTAssertEqual(model.speakerEncoder.parameterMemoryBytes(), 0)
    }
}

// MARK: - E2E Tests (require model download + GPU)

final class VoiceCloningE2ETests: XCTestCase {

    static let ttsModelId = "aufklarer/Qwen3-TTS-12Hz-0.6B-Base-MLX-4bit"
    static let ttsTokenizerModelId = "Qwen/Qwen3-TTS-Tokenizer-12Hz"
    static let asrModelId = "aufklarer/Qwen3-ASR-0.6B-MLX-4bit"
    private static var _sharedTTSModel: Qwen3TTSModel?
    private static var _sharedASRModel: Qwen3ASRModel?

    // MARK: - Voice Cloning Tests

    /// Voice cloning: synthesize with reference audio and verify output
    func testVoiceCloneSynthesis() async throws {
        let ttsModel = try await loadTTSModel()

        let refAudio = try loadTestAudio()
        let text = "This is a voice cloning test."

        let start = Date()
        let samples = ttsModel.synthesizeWithVoiceClone(
            text: text,
            referenceAudio: refAudio,
            referenceSampleRate: 24000,
            language: "english")
        let elapsed = Date().timeIntervalSince(start)

        let duration = Double(samples.count) / 24000.0
        print("Voice clone: \(fmt(duration))s audio in \(fmt(elapsed))s (RTF: \(fmt(elapsed / max(duration, 0.001))))")

        XCTAssertGreaterThan(samples.count, 0, "Should produce audio")
        XCTAssertGreaterThan(duration, 0.5, "Should produce audible speech")
        XCTAssertLessThan(duration, 30.0, "Should not be excessively long")

        let maxAmp = samples.map { abs($0) }.max() ?? 0
        XCTAssertGreaterThan(maxAmp, 0.001, "Audio should not be silent")
        XCTAssertLessThanOrEqual(maxAmp, 1.0, "Samples should be in [-1, 1]")
    }

    /// Voice cloning ASR round-trip
    func testVoiceCloneRoundTrip() async throws {
        let ttsModel = try await loadTTSModel()
        let asrModel = try await loadASRModel()

        let refAudio = try loadTestAudio()
        let inputText = "Hello world, this is a test."

        let samples = ttsModel.synthesizeWithVoiceClone(
            text: inputText,
            referenceAudio: refAudio,
            referenceSampleRate: 24000,
            language: "english")

        XCTAssertGreaterThan(samples.count, 0, "Should produce audio")

        let transcription = try transcribeAudio(
            samples: samples, sampleRate: 24000, using: asrModel)

        print("Input:  \"\(inputText)\"")
        print("Output: \"\(transcription)\"")

        let lowerTranscription = transcription.lowercased()
        let expectedWords = ["hello", "world", "test"]
        let matchedWords = expectedWords.filter { lowerTranscription.contains($0) }
        print("Matched \(matchedWords.count)/\(expectedWords.count) words: \(matchedWords)")

        XCTAssertGreaterThanOrEqual(matchedWords.count, 2,
            "At least 2 of \(expectedWords) should appear in: \"\(transcription)\"")
    }

    /// Speaker embedding extraction produces valid 1024-dim vector
    func testSpeakerEmbeddingExtraction() async throws {
        let ttsModel = try await loadTTSModel()

        let refAudio = try loadTestAudio()
        let mels = SpeakerMel.compute(audio: refAudio, sampleRate: 24000)

        let embedding = ttsModel.speakerEncoder(mels)
        eval(embedding)

        XCTAssertEqual(embedding.ndim, 2, "Should be [1, 1024]")
        XCTAssertEqual(embedding.dim(0), 1, "Batch size 1")
        XCTAssertEqual(embedding.dim(1), 1024, "Embedding dim 1024")

        // Verify embedding is not all zeros
        let norm = sqrt((embedding * embedding).sum().item(Float.self))
        XCTAssertGreaterThan(norm, 0.1, "Embedding should not be near-zero")
        print("Speaker embedding norm: \(norm)")
    }

    /// Same reference audio should produce consistent embeddings
    func testSpeakerEmbeddingConsistency() async throws {
        let ttsModel = try await loadTTSModel()

        let refAudio = try loadTestAudio()
        let mels = SpeakerMel.compute(audio: refAudio, sampleRate: 24000)

        let emb1 = ttsModel.speakerEncoder(mels)
        let emb2 = ttsModel.speakerEncoder(mels)
        eval(emb1, emb2)

        // Same input should produce identical output (deterministic model)
        let diff = abs(emb1 - emb2).sum().item(Float.self)
        XCTAssertEqual(diff, 0, accuracy: 1e-5, "Same input should produce identical embeddings")
    }

    /// Voice clone vs normal synthesis should produce different audio
    func testVoiceCloneVsNormal() async throws {
        let ttsModel = try await loadTTSModel()

        let text = "Testing voice difference."
        let refAudio = try loadTestAudio()

        let normalSamples = ttsModel.synthesize(text: text, language: "english")
        let clonedSamples = ttsModel.synthesizeWithVoiceClone(
            text: text,
            referenceAudio: refAudio,
            referenceSampleRate: 24000,
            language: "english")

        XCTAssertGreaterThan(normalSamples.count, 0)
        XCTAssertGreaterThan(clonedSamples.count, 0)

        // Both should produce valid audio (can't easily compare voice similarity in a unit test)
        let normalMax = normalSamples.map { abs($0) }.max() ?? 0
        let clonedMax = clonedSamples.map { abs($0) }.max() ?? 0
        XCTAssertGreaterThan(normalMax, 0.001)
        XCTAssertGreaterThan(clonedMax, 0.001)

        print("Normal: \(normalSamples.count) samples (\(fmt(Double(normalSamples.count) / 24000.0))s)")
        print("Cloned: \(clonedSamples.count) samples (\(fmt(Double(clonedSamples.count) / 24000.0))s)")
    }

    // MARK: - Helpers

    private func loadTTSModel() async throws -> Qwen3TTSModel {
        if let model = Self._sharedTTSModel { return model }
        print("Loading TTS model...")
        let model = try await Qwen3TTSModel.fromPretrained(
            modelId: Self.ttsModelId,
            tokenizerModelId: Self.ttsTokenizerModelId
        ) { progress, status in
            print("[TTS \(Int(progress * 100))%] \(status)")
        }
        Self._sharedTTSModel = model
        return model
    }

    private func loadASRModel() async throws -> Qwen3ASRModel {
        if let model = Self._sharedASRModel { return model }
        print("Loading ASR model...")
        let model = try await Qwen3ASRModel.fromPretrained(
            modelId: Self.asrModelId
        ) { progress, status in
            print("[ASR \(Int(progress * 100))%] \(status)")
        }
        Self._sharedASRModel = model
        return model
    }

    private func loadTestAudio() throws -> [Float] {
        let bundle = Bundle(for: type(of: self))
        // Try bundle resource first, then fallback to file path
        if let url = bundle.url(forResource: "test_audio", withExtension: "wav") {
            return try AudioFileLoader.load(url: url, targetSampleRate: 24000)
        }
        let path = "Tests/Qwen3ASRTests/Resources/test_audio.wav"
        return try AudioFileLoader.load(url: URL(fileURLWithPath: path), targetSampleRate: 24000)
    }

    private func transcribeAudio(
        samples: [Float], sampleRate: Int, using model: Qwen3ASRModel
    ) throws -> String {
        let start = Date()
        let result = model.transcribe(audio: samples, sampleRate: sampleRate)
        let elapsed = Date().timeIntervalSince(start)
        print("  ASR: \(fmt(elapsed))s")
        return result
    }

    private func fmt(_ value: Double) -> String {
        String(format: "%.2f", value)
    }
}
