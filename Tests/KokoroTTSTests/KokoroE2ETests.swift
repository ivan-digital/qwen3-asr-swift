import XCTest
@testable import KokoroTTS
import CoreML

/// E2E tests that require downloaded CoreML models.
/// Run with: swift test --filter KokoroE2ETests
final class E2EKokoroTests: XCTestCase {

    /// v2 3-stage models (duration + prosody + decoder)
    static let v2ModelDir = "/tmp/kokoro-v2"

    /// Legacy test directory (old end-to-end models)
    static let legacyModelDir = "/tmp/kokoro-coreml-test"

    /// Shared resource directory (vocab, voices, G2P — same for both versions)
    static var resourceDir: String {
        if FileManager.default.fileExists(atPath: v2ModelDir + "/vocab_index.json") {
            return v2ModelDir
        }
        return legacyModelDir
    }

    /// Test loading vocab_index.json.
    func testLoadVocabIndex() throws {
        let url = URL(fileURLWithPath: Self.resourceDir + "/vocab_index.json")
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw XCTSkip("Models not downloaded")
        }
        let phonemizer = try KokoroPhonemizer.loadVocab(from: url)
        let ids = phonemizer.tokenize("hello")
        XCTAssertEqual(ids.first, 1) // BOS
        XCTAssertEqual(ids.last, 2)  // EOS
        XCTAssertTrue(ids.count >= 3)
    }

    /// Test loading pronunciation dictionaries.
    func testLoadDictionaries() throws {
        let dir = URL(fileURLWithPath: Self.resourceDir)
        guard FileManager.default.fileExists(atPath: dir.appendingPathComponent("us_gold.json").path) else {
            throw XCTSkip("Models not downloaded")
        }
        let vocab = URL(fileURLWithPath: Self.resourceDir + "/vocab_index.json")
        let phonemizer = try KokoroPhonemizer.loadVocab(from: vocab)
        try phonemizer.loadDictionaries(from: dir)
        let ids = phonemizer.tokenize("hello")
        XCTAssertTrue(ids.count > 3, "Expected more than BOS+EOS for 'hello'")
    }

    /// Test loading G2P encoder + decoder.
    func testLoadG2PModels() throws {
        let dir = URL(fileURLWithPath: Self.resourceDir)
        let encoderURL = dir.appendingPathComponent("G2PEncoder.mlmodelc")
        let decoderURL = dir.appendingPathComponent("G2PDecoder.mlmodelc")
        let vocabURL = dir.appendingPathComponent("g2p_vocab.json")
        guard FileManager.default.fileExists(atPath: encoderURL.path) else {
            throw XCTSkip("Models not downloaded")
        }

        let mainVocab = URL(fileURLWithPath: Self.resourceDir + "/vocab_index.json")
        let phonemizer = try KokoroPhonemizer.loadVocab(from: mainVocab)
        try phonemizer.loadG2PModels(encoderURL: encoderURL, decoderURL: decoderURL, vocabURL: vocabURL)
        let ids = phonemizer.tokenize("supercalifragilistic")
        XCTAssertTrue(ids.count > 3, "G2P should produce tokens for OOV word")
    }

    /// Test loading voice embedding JSON.
    func testLoadVoiceEmbedding() throws {
        let voiceURL = URL(fileURLWithPath: Self.resourceDir + "/voices/af_heart.json")
        guard FileManager.default.fileExists(atPath: voiceURL.path) else {
            throw XCTSkip("Models not downloaded")
        }

        let data = try Data(contentsOf: voiceURL)
        let json = try JSONSerialization.jsonObject(with: data) as! [String: Any]
        let embedding = json["embedding"] as! [Double]

        XCTAssertEqual(embedding.count, 256)
        let refS = embedding.map { Float($0) }
        XCTAssertFalse(refS.allSatisfy { $0 == 0 }, "Embedding shouldn't be all zeros")
    }

    /// Test loading the 3-stage CoreML pipeline.
    func testLoadV2Pipeline() throws {
        let dir = URL(fileURLWithPath: Self.v2ModelDir)
        guard FileManager.default.fileExists(atPath: dir.appendingPathComponent("duration.mlmodelc").path) else {
            throw XCTSkip("V2 models not converted — run convert_kokoro_v2.py first")
        }

        let network = try KokoroNetwork(directory: dir)
        XCTAssertTrue(network.hasThreeStagePipeline)
        XCTAssertFalse(network.availableDecoderBuckets.isEmpty)
    }

    /// Full E2E: text → phonemes → 3-stage CoreML → audio.
    func testEndToEndSynthesisV2() throws {
        let dir = URL(fileURLWithPath: Self.v2ModelDir)
        guard FileManager.default.fileExists(atPath: dir.appendingPathComponent("duration.mlmodelc").path) else {
            throw XCTSkip("V2 models not converted")
        }

        // Load phonemizer
        let vocabURL = dir.appendingPathComponent("vocab_index.json")
        guard FileManager.default.fileExists(atPath: vocabURL.path) else {
            throw XCTSkip("Vocab not found")
        }
        let phonemizer = try KokoroPhonemizer.loadVocab(from: vocabURL)
        try phonemizer.loadDictionaries(from: dir)

        let encoderURL = dir.appendingPathComponent("G2PEncoder.mlmodelc")
        let decoderURL = dir.appendingPathComponent("G2PDecoder.mlmodelc")
        let g2pVocabURL = dir.appendingPathComponent("g2p_vocab.json")
        if FileManager.default.fileExists(atPath: encoderURL.path) {
            try phonemizer.loadG2PModels(encoderURL: encoderURL, decoderURL: decoderURL, vocabURL: g2pVocabURL)
        }

        // Load voice
        let voiceData = try Data(contentsOf: dir.appendingPathComponent("voices/af_heart.json"))
        let voiceJson = try JSONSerialization.jsonObject(with: voiceData) as! [String: Any]
        let embedding = voiceJson["embedding"] as! [Double]
        let styleVector = embedding.map { Float($0) }

        // Load 3-stage network
        let network = try KokoroNetwork(directory: dir)
        XCTAssertTrue(network.hasThreeStagePipeline)

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
        print("E2E v2 synthesis: \(audio.count) samples, \(String(format: "%.2f", duration))s")
        XCTAssertGreaterThan(duration, 0.3, "Audio should be at least 0.3s")
        XCTAssertLessThan(duration, 5.0, "Audio should be less than 5s for 'Hello world'")

        // Audio should have non-zero energy
        let rms = sqrt(audio.map { $0 * $0 }.reduce(0, +) / Float(audio.count))
        XCTAssertGreaterThan(rms, 0.001, "Audio should have non-zero energy")
    }
}
