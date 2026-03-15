import XCTest
@testable import KokoroTTS
import CoreML
import ParakeetASR

/// E2E tests that require downloaded CoreML models.
/// Run with: swift test --filter KokoroE2ETests
final class KokoroE2ETests: XCTestCase {

    static let testModelDir = "/tmp/kokoro-coreml-test"

    /// Test loading vocab_index.json from aufklarer/Kokoro-82M-CoreML.
    func testLoadVocabIndex() throws {
        let url = URL(fileURLWithPath: Self.testModelDir + "/vocab_index.json")
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw XCTSkip("Models not downloaded — run download script first")
        }
        let phonemizer = try KokoroPhonemizer.loadVocab(from: url)

        // Tokenize a simple IPA string
        let ids = phonemizer.tokenize("hello")
        XCTAssertEqual(ids.first, 1) // BOS
        XCTAssertEqual(ids.last, 2)  // EOS
        XCTAssertTrue(ids.count >= 3)
    }

    /// Test loading pronunciation dictionaries.
    func testLoadDictionaries() throws {
        let dir = URL(fileURLWithPath: Self.testModelDir)
        guard FileManager.default.fileExists(atPath: dir.appendingPathComponent("us_gold.json").path) else {
            throw XCTSkip("Models not downloaded")
        }
        let vocab = URL(fileURLWithPath: Self.testModelDir + "/vocab_index.json")
        let phonemizer = try KokoroPhonemizer.loadVocab(from: vocab)
        try phonemizer.loadDictionaries(from: dir)

        // "hello" should be in the dictionary → produce IPA
        let ids = phonemizer.tokenize("hello")
        XCTAssertTrue(ids.count > 3, "Expected more than BOS+EOS for 'hello'")
    }

    /// Test loading G2P encoder + decoder.
    func testLoadG2PModels() throws {
        let dir = URL(fileURLWithPath: Self.testModelDir)
        let encoderURL = dir.appendingPathComponent("G2PEncoder.mlmodelc")
        let decoderURL = dir.appendingPathComponent("G2PDecoder.mlmodelc")
        let vocabURL = dir.appendingPathComponent("g2p_vocab.json")
        guard FileManager.default.fileExists(atPath: encoderURL.path) else {
            throw XCTSkip("Models not downloaded")
        }

        let mainVocab = URL(fileURLWithPath: Self.testModelDir + "/vocab_index.json")
        let phonemizer = try KokoroPhonemizer.loadVocab(from: mainVocab)
        try phonemizer.loadG2PModels(encoderURL: encoderURL, decoderURL: decoderURL, vocabURL: vocabURL)

        // Try phonemizing an OOV word through the neural G2P
        let ids = phonemizer.tokenize("supercalifragilistic")
        XCTAssertTrue(ids.count > 3, "G2P should produce tokens for OOV word")
    }

    /// Test loading voice embedding JSON.
    func testLoadVoiceEmbedding() throws {
        let voiceURL = URL(fileURLWithPath: Self.testModelDir + "/voices/af_heart.json")
        guard FileManager.default.fileExists(atPath: voiceURL.path) else {
            throw XCTSkip("Models not downloaded")
        }

        let data = try Data(contentsOf: voiceURL)
        let json = try JSONSerialization.jsonObject(with: data) as! [String: Any]
        let embedding = json["embedding"] as! [Double]

        // Voice embedding is 256-dim (matches ref_s input)
        XCTAssertEqual(embedding.count, 256)

        let refS = embedding.map { Float($0) }
        XCTAssertEqual(refS.count, 256)
        XCTAssertFalse(refS.allSatisfy { $0 == 0 }, "Embedding shouldn't be all zeros")
    }

    /// Test loading the CoreML Kokoro model.
    func testLoadKokoroModel() throws {
        let dir = URL(fileURLWithPath: Self.testModelDir)
        let modelURL = dir.appendingPathComponent("kokoro_24_10s.mlmodelc")
        guard FileManager.default.fileExists(atPath: modelURL.path) else {
            throw XCTSkip("Models not downloaded")
        }

        let network = try KokoroNetwork(directory: dir)
        XCTAssertTrue(network.availableBuckets.contains(.v24_10s))
    }

    /// Full E2E: text → phonemes → CoreML inference → audio.
    func testEndToEndSynthesis() throws {
        let dir = URL(fileURLWithPath: Self.testModelDir)
        guard FileManager.default.fileExists(atPath: dir.appendingPathComponent("kokoro_24_10s.mlmodelc").path) else {
            throw XCTSkip("Models not downloaded")
        }

        // Load phonemizer
        let vocab = dir.appendingPathComponent("vocab_index.json")
        let phonemizer = try KokoroPhonemizer.loadVocab(from: vocab)
        try phonemizer.loadDictionaries(from: dir)

        let encoderURL = dir.appendingPathComponent("G2PEncoder.mlmodelc")
        let decoderURL = dir.appendingPathComponent("G2PDecoder.mlmodelc")
        let vocabURL = dir.appendingPathComponent("g2p_vocab.json")
        if FileManager.default.fileExists(atPath: encoderURL.path) {
            try phonemizer.loadG2PModels(encoderURL: encoderURL, decoderURL: decoderURL, vocabURL: vocabURL)
        }

        // Load voice embedding (256-dim)
        let voiceData = try Data(contentsOf: dir.appendingPathComponent("voices/af_heart.json"))
        let voiceJson = try JSONSerialization.jsonObject(with: voiceData) as! [String: Any]
        let embedding = voiceJson["embedding"] as! [Double]
        let styleVector = embedding.map { Float($0) }

        // Load network
        let network = try KokoroNetwork(directory: dir)

        // Create model
        let config = KokoroConfig.default
        let model = KokoroTTSModel(
            config: config,
            network: network,
            phonemizer: phonemizer,
            voiceEmbeddings: ["af_heart": styleVector]
        )

        // Synthesize
        let audio = try model.synthesize(text: "Hello world", voice: "af_heart")

        XCTAssertTrue(audio.count > 0, "Should produce audio samples")
        XCTAssertTrue(audio.count > 1000, "Should produce meaningful audio (got \(audio.count) samples)")

        let duration = Double(audio.count) / 24000.0
        print("E2E synthesis: \(audio.count) samples, \(String(format: "%.2f", duration))s")

        // Audio should have non-zero energy
        let rms = sqrt(audio.map { $0 * $0 }.reduce(0, +) / Float(audio.count))
        XCTAssertGreaterThan(rms, 0.001, "Audio should have non-zero energy")
    }

    // MARK: - CPU-only tests

    /// Test loading the CoreML Kokoro model with CPU-only compute units.
    func testLoadKokoroModelCPUOnly() throws {
        let dir = URL(fileURLWithPath: Self.testModelDir)
        let modelURL = dir.appendingPathComponent("kokoro_24_10s.mlmodelc")
        guard FileManager.default.fileExists(atPath: modelURL.path) else {
            throw XCTSkip("Models not downloaded")
        }

        let network = try KokoroNetwork(directory: dir, computeUnits: .cpuOnly)
        XCTAssertTrue(network.availableBuckets.contains(.v24_10s))
    }

    /// Full E2E synthesis using CPU-only compute units (for iOS Simulator / no ANE).
    func testEndToEndSynthesisCPUOnly() throws {
        let dir = URL(fileURLWithPath: Self.testModelDir)
        guard FileManager.default.fileExists(atPath: dir.appendingPathComponent("kokoro_24_10s.mlmodelc").path) else {
            throw XCTSkip("Models not downloaded")
        }

        // Load phonemizer
        let vocab = dir.appendingPathComponent("vocab_index.json")
        let phonemizer = try KokoroPhonemizer.loadVocab(from: vocab)
        try phonemizer.loadDictionaries(from: dir)

        let encoderURL = dir.appendingPathComponent("G2PEncoder.mlmodelc")
        let decoderURL = dir.appendingPathComponent("G2PDecoder.mlmodelc")
        let vocabURL = dir.appendingPathComponent("g2p_vocab.json")
        if FileManager.default.fileExists(atPath: encoderURL.path) {
            try phonemizer.loadG2PModels(encoderURL: encoderURL, decoderURL: decoderURL, vocabURL: vocabURL)
        }

        // Load voice embedding (256-dim)
        let voiceData = try Data(contentsOf: dir.appendingPathComponent("voices/af_heart.json"))
        let voiceJson = try JSONSerialization.jsonObject(with: voiceData) as! [String: Any]
        let embedding = voiceJson["embedding"] as! [Double]
        let styleVector = embedding.map { Float($0) }

        // Load network with CPU-only compute units
        let network = try KokoroNetwork(directory: dir, computeUnits: .cpuOnly)

        // Create model
        let config = KokoroConfig.default
        let model = KokoroTTSModel(
            config: config,
            network: network,
            phonemizer: phonemizer,
            voiceEmbeddings: ["af_heart": styleVector]
        )

        // Synthesize
        let audio = try model.synthesize(text: "Hello world", voice: "af_heart")

        XCTAssertTrue(audio.count > 0, "Should produce audio samples")
        XCTAssertTrue(audio.count > 1000, "Should produce meaningful audio (got \(audio.count) samples)")

        let duration = Double(audio.count) / 24000.0
        print("E2E CPU-only synthesis: \(audio.count) samples, \(String(format: "%.2f", duration))s")

        // Audio should have non-zero energy
        let rms = sqrt(audio.map { $0 * $0 }.reduce(0, +) / Float(audio.count))
        XCTAssertGreaterThan(rms, 0.001, "Audio should have non-zero energy")

        // All samples should be in valid [-1, 1] range (not garbage values)
        let minSample = audio.min() ?? 0
        let maxSample = audio.max() ?? 0
        XCTAssertGreaterThanOrEqual(minSample, -1.0, "Audio samples should be >= -1.0 (got \(minSample))")
        XCTAssertLessThanOrEqual(maxSample, 1.0, "Audio samples should be <= 1.0 (got \(maxSample))")

        // Audio should NOT be clipped garbage (>30% clipped = broken model)
        let clippedCount = audio.filter { abs($0) > 0.99 }.count
        let clippedPct = Float(clippedCount) / Float(audio.count) * 100
        print("E2E CPU-only clipping: \(String(format: "%.1f", clippedPct))%")
        XCTAssertLessThan(clippedPct, 30, "Audio is \(String(format: "%.1f", clippedPct))% clipped — model output is garbage")

        // RMS should be speech-like (0.01-0.5), not saturated noise (>0.7)
        XCTAssertLessThan(rms, 0.5, "RMS \(String(format: "%.3f", rms)) is too high — model output is saturated")

        // Duration should be reasonable for "Hello world" (0.1s to 5s)
        XCTAssertGreaterThan(duration, 0.1, "Audio duration should be at least 0.1s (got \(String(format: "%.2f", duration))s)")
        XCTAssertLessThan(duration, 5.0, "Audio duration should be under 5s (got \(String(format: "%.2f", duration))s)")
    }

    // MARK: - Round-trip test (TTS → ASR)

    /// Synthesize speech then transcribe it back — verifies audio quality end-to-end.
    /// If TTS produces garbage, ASR won't recognize the original phrase.
    func testRoundTripTTStoASR() async throws {
        let dir = URL(fileURLWithPath: Self.testModelDir)
        guard FileManager.default.fileExists(atPath: dir.appendingPathComponent("kokoro_24_10s.mlmodelc").path) else {
            throw XCTSkip("Kokoro models not downloaded")
        }

        // Load TTS
        let vocab = dir.appendingPathComponent("vocab_index.json")
        let phonemizer = try KokoroPhonemizer.loadVocab(from: vocab)
        try phonemizer.loadDictionaries(from: dir)

        let voiceData = try Data(contentsOf: dir.appendingPathComponent("voices/af_heart.json"))
        let voiceJson = try JSONSerialization.jsonObject(with: voiceData) as! [String: Any]
        let embedding = voiceJson["embedding"] as! [Double]
        let styleVector = embedding.map { Float($0) }

        let network = try KokoroNetwork(directory: dir, computeUnits: .cpuOnly)
        let tts = KokoroTTSModel(
            config: .default, network: network, phonemizer: phonemizer,
            voiceEmbeddings: ["af_heart": styleVector])

        // Load ASR
        let asr = try await ParakeetASRModel.fromPretrained()

        // Synthesize
        let inputText = "Hello world"
        let audio = try tts.synthesize(text: inputText, voice: "af_heart")
        print("TTS: \(audio.count) samples, \(String(format: "%.2f", Double(audio.count) / 24000.0))s")

        // Resample 24kHz → 16kHz for ASR
        let resampledCount = audio.count * 16000 / 24000
        var resampled = [Float](repeating: 0, count: resampledCount)
        for i in 0..<resampledCount {
            let srcIdx = Float(i) * 24000.0 / 16000.0
            let idx0 = Int(srcIdx)
            let frac = srcIdx - Float(idx0)
            let s0 = idx0 < audio.count ? audio[idx0] : 0
            let s1 = (idx0 + 1) < audio.count ? audio[idx0 + 1] : s0
            resampled[i] = s0 + frac * (s1 - s0)
        }

        // Transcribe
        let transcript = try asr.transcribeAudio(resampled, sampleRate: 16000)
            .lowercased().trimmingCharacters(in: .whitespacesAndNewlines)
        print("ASR transcript: '\(transcript)' (expected: '\(inputText.lowercased())')")

        // The transcript should contain the key words
        XCTAssertTrue(
            transcript.contains("hello") && transcript.contains("world"),
            "Round-trip failed: TTS→ASR produced '\(transcript)' instead of '\(inputText)'"
        )
    }

    /// Round-trip on CPU only — catches iOS Simulator CoreML issues.
    func testRoundTripTTStoASR_CPUOnly() async throws {
        let dir = URL(fileURLWithPath: Self.testModelDir)
        guard FileManager.default.fileExists(atPath: dir.appendingPathComponent("kokoro_24_10s.mlmodelc").path) else {
            throw XCTSkip("Kokoro models not downloaded")
        }

        // Load TTS with CPU only
        let vocab = dir.appendingPathComponent("vocab_index.json")
        let phonemizer = try KokoroPhonemizer.loadVocab(from: vocab)
        try phonemizer.loadDictionaries(from: dir)

        let voiceData = try Data(contentsOf: dir.appendingPathComponent("voices/af_heart.json"))
        let voiceJson = try JSONSerialization.jsonObject(with: voiceData) as! [String: Any]
        let embedding = voiceJson["embedding"] as! [Double]
        let styleVector = embedding.map { Float($0) }

        let network = try KokoroNetwork(directory: dir, computeUnits: .cpuOnly)
        let tts = KokoroTTSModel(
            config: .default, network: network, phonemizer: phonemizer,
            voiceEmbeddings: ["af_heart": styleVector])

        // Load ASR (also CPU only to simulate device constraints)
        let asr = try await ParakeetASRModel.fromPretrained()

        // Synthesize
        let inputText = "Hello world"
        let audio = try tts.synthesize(text: inputText, voice: "af_heart")

        // Resample 24kHz → 16kHz
        let resampledCount = audio.count * 16000 / 24000
        var resampled = [Float](repeating: 0, count: resampledCount)
        for i in 0..<resampledCount {
            let srcIdx = Float(i) * 24000.0 / 16000.0
            let idx0 = Int(srcIdx)
            let frac = srcIdx - Float(idx0)
            let s0 = idx0 < audio.count ? audio[idx0] : 0
            let s1 = (idx0 + 1) < audio.count ? audio[idx0 + 1] : s0
            resampled[i] = s0 + frac * (s1 - s0)
        }

        // Transcribe
        let transcript = try asr.transcribeAudio(resampled, sampleRate: 16000)
            .lowercased().trimmingCharacters(in: .whitespacesAndNewlines)
        print("CPU-only round-trip: TTS→ASR = '\(transcript)'")

        XCTAssertTrue(
            transcript.contains("hello") && transcript.contains("world"),
            "CPU round-trip failed: TTS→ASR produced '\(transcript)' instead of '\(inputText)'"
        )
    }
}
