import XCTest
@testable import Qwen3TTSCoreML
@testable import Qwen3ASR
import AudioCommon

#if canImport(CoreML)
/// Unit tests for Qwen3-TTS CoreML module (no model download required)
final class Qwen3TTSCoreMLTests: XCTestCase {

    func testSamplerGreedy() {
        let logits: [Float] = [0.1, 0.5, 0.9, 0.2, 0.3]
        let token = TTSSampler.sample(logits: logits, temperature: 0, topK: 0)
        XCTAssertEqual(token, 2)
    }

    func testSamplerSuppressRange() {
        var logits = [Float](repeating: -10, count: 10)
        logits[5] = 100; logits[7] = 50; logits[2] = 1
        let token = TTSSampler.sample(logits: logits, temperature: 0, suppressRange: (4, 9))
        XCTAssertEqual(token, 2)
    }

    func testSamplerEOSPreserved() {
        var logits = [Float](repeating: -10, count: 10)
        logits[5] = 100
        let token = TTSSampler.sample(logits: logits, temperature: 0, suppressRange: (4, 9), eosTokenId: 5)
        XCTAssertEqual(token, 5)
    }

    func testSamplerEOSBias() {
        var logits = [Float](repeating: 0, count: 10)
        logits[3] = 5.0  // top token without bias
        logits[7] = 1.0  // EOS
        let token = TTSSampler.sample(logits: logits, temperature: 0, eosTokenId: 7, eosLogitBias: 10.0)
        XCTAssertEqual(token, 7)  // EOS wins with +10 bias (1+10=11 > 5)
    }

    func testPromptBuilderLanguageIds() {
        XCTAssertEqual(PromptBuilder.languageIds["english"], 2050)
        XCTAssertEqual(PromptBuilder.languageIds["chinese"], 2055)
        XCTAssertEqual(PromptBuilder.languageIds["japanese"], 2058)
        XCTAssertEqual(PromptBuilder.languageIds["german"], 2053)
        XCTAssertEqual(PromptBuilder.languageIds["french"], 2061)
    }

    func testSamplerRepetitionPenalty() {
        var logits = [Float](repeating: 0, count: 10)
        logits[3] = 10.0  // top token, previously generated
        logits[5] = 9.0   // second best, not generated
        // With penalty 2.0, token 3 becomes 10/2=5, token 5 stays 9 → 5 wins
        let token = TTSSampler.sample(
            logits: logits, temperature: 0,
            repetitionPenalty: 2.0, generatedTokens: [3])
        XCTAssertEqual(token, 5)
    }
}

/// E2E tests requiring model download
final class E2EQwen3TTSCoreMLTests: XCTestCase {

    func testSynthesizeProducesAudio() async throws {
        let model: Qwen3TTSCoreMLModel
        do {
            model = try await Qwen3TTSCoreMLModel.fromPretrained()
        } catch {
            throw XCTSkip("Model not available: \(error)")
        }
        defer { model.unload() }

        let audio = try model.synthesize(
            text: "Hello world",
            language: "english",
            temperature: 0.9,
            maxTokens: 50)

        XCTAssertGreaterThan(audio.count, 0, "Should produce audio samples")

        let duration = Double(audio.count) / 24000.0
        XCTAssertGreaterThan(duration, 0.1, "Audio should be at least 0.1s")
        XCTAssertLessThan(duration, 30.0, "Audio should be less than 30s")
        print("Generated \(audio.count) samples (\(String(format: "%.2f", duration))s)")
    }

    func testSynthesizeMultipleLanguages() async throws {
        let model: Qwen3TTSCoreMLModel
        do {
            model = try await Qwen3TTSCoreMLModel.fromPretrained()
        } catch {
            throw XCTSkip("Model not available: \(error)")
        }
        defer { model.unload() }

        for (lang, text) in [("english", "Hello"), ("chinese", "你好"), ("german", "Hallo")] {
            let audio = try model.synthesize(text: text, language: lang, maxTokens: 30)
            XCTAssertGreaterThan(audio.count, 0, "\(lang) should produce audio")
            print("\(lang) (\(text)): \(audio.count) samples (\(String(format: "%.2f", Double(audio.count)/24000))s)")
        }
    }

    func testProtocolConformance() async throws {
        let model: Qwen3TTSCoreMLModel
        do {
            model = try await Qwen3TTSCoreMLModel.fromPretrained()
        } catch {
            throw XCTSkip("Model not available: \(error)")
        }
        defer { model.unload() }

        // Test via protocol
        let audio = try await model.generate(text: "Test", language: "english")
        XCTAssertGreaterThan(audio.count, 0)
    }

    /// Round-trip: synthesize → transcribe → verify keywords recognized.
    func testRoundTripSynthesizeTranscribe() async throws {
        let ttsModel: Qwen3TTSCoreMLModel
        do {
            ttsModel = try await Qwen3TTSCoreMLModel.fromPretrained()
        } catch {
            throw XCTSkip("TTS model not available: \(error)")
        }
        defer { ttsModel.unload() }

        let asrModel: Qwen3ASRModel
        do {
            asrModel = try await Qwen3ASRModel.fromPretrained()
        } catch {
            throw XCTSkip("ASR model not available: \(error)")
        }

        let text = "The quick brown fox jumps over the lazy dog"
        let audio = try ttsModel.synthesize(text: text, language: "english", maxTokens: 125)
        XCTAssertGreaterThan(audio.count, 0, "Should produce audio")

        let duration = Double(audio.count) / 24000.0
        print("Synthesized \(audio.count) samples (\(String(format: "%.2f", duration))s)")

        let transcription = asrModel.transcribe(audio: audio, sampleRate: 24000)
        print("Input:  \"\(text)\"")
        print("Output: \"\(transcription)\"")

        // Check that ASR recognizes at least 3 keywords
        let keywords = ["quick", "brown", "fox", "jumps", "lazy", "dog"]
        let matched = keywords.filter { transcription.lowercased().contains($0) }
        print("Matched \(matched.count)/\(keywords.count) keywords: \(matched)")

        XCTAssertGreaterThanOrEqual(matched.count, 3,
            "ASR should recognize at least 3 keywords from CoreML TTS output. Got: \"\(transcription)\"")
    }
}
#endif
