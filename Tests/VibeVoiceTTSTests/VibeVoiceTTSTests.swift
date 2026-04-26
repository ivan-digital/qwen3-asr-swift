import XCTest
@testable import VibeVoiceTTS
import NemotronStreamingASR
import AudioCommon
import MLX
import MLXNN
import Foundation

// MARK: - Unit tests (no network, fast)

final class VibeVoiceConfigurationTests: XCTestCase {

    func testQwen2DefaultConfigIsRealtime05B() {
        let cfg = Qwen2Configuration()
        XCTAssertEqual(cfg.hiddenSize, 896)
        XCTAssertEqual(cfg.hiddenLayers, 24)
        XCTAssertEqual(cfg.attentionHeads, 14)
        XCTAssertEqual(cfg.kvHeads, 2)
        XCTAssertEqual(cfg.vocabularySize, 151936)
        XCTAssertEqual(cfg.intermediateSize, 4864)
    }

    func testQwen2HeadDim() {
        let cfg = Qwen2Configuration()
        XCTAssertEqual(cfg.headDim, 896 / 14)
    }

    func testQwen2DecodesFromHFJSON() throws {
        let json = """
        {
          "hidden_size": 1536,
          "num_hidden_layers": 28,
          "intermediate_size": 8960,
          "num_attention_heads": 12,
          "num_key_value_heads": 2,
          "rms_norm_eps": 1e-6,
          "vocab_size": 151936,
          "rope_theta": 1000000.0,
          "max_position_embeddings": 32768
        }
        """.data(using: .utf8)!
        let cfg = try JSONDecoder().decode(Qwen2Configuration.self, from: json)
        XCTAssertEqual(cfg.hiddenSize, 1536)
        XCTAssertEqual(cfg.hiddenLayers, 28)
        XCTAssertEqual(cfg.attentionHeads, 12)
    }

    func testVibeVoiceConfigurationDecodesFromHFJSON() throws {
        let json = """
        {
          "decoder_config": {
            "hidden_size": 896,
            "num_hidden_layers": 24,
            "intermediate_size": 4864,
            "num_attention_heads": 14,
            "num_key_value_heads": 2,
            "rms_norm_eps": 1e-6,
            "vocab_size": 151936,
            "rope_theta": 1000000.0,
            "max_position_embeddings": 8192
          },
          "acoustic_tokenizer_config": {},
          "diffusion_head_config": {},
          "tts_backbone_num_hidden_layers": 20,
          "acoustic_vae_dim": 64
        }
        """.data(using: .utf8)!
        let cfg = try JSONDecoder().decode(VibeVoiceConfiguration.self, from: json)
        XCTAssertEqual(cfg.decoderConfig.hiddenSize, 896)
        XCTAssertEqual(cfg.ttsBackboneNumHiddenLayers, 20)
        XCTAssertEqual(cfg.acousticVaeDim, 64)
    }

    func testVibeVoiceConfigurationInfersTTSLayersWhenAbsent() throws {
        let json = """
        {
          "decoder_config": {
            "hidden_size": 896,
            "num_hidden_layers": 24,
            "intermediate_size": 4864,
            "num_attention_heads": 14,
            "num_key_value_heads": 2,
            "rms_norm_eps": 1e-6,
            "vocab_size": 151936,
            "rope_theta": 1000000.0,
            "max_position_embeddings": 8192
          },
          "acoustic_tokenizer_config": {},
          "diffusion_head_config": {}
        }
        """.data(using: .utf8)!
        let cfg = try JSONDecoder().decode(VibeVoiceConfiguration.self, from: json)
        // Default rule: max(totalLayers - 4, totalLayers * 3 / 4) = max(20, 18) = 20
        XCTAssertEqual(cfg.ttsBackboneNumHiddenLayers, 20)
    }
}

final class VibeVoiceQuantizationManifestTests: XCTestCase {

    func testDecodesInt4AffineManifest() throws {
        let json = """
        {
          "model_id": "microsoft/VibeVoice-Realtime-0.5B",
          "revision": "main",
          "group_size": 32,
          "bits": 4,
          "mode": "affine",
          "layers": [
            {
              "name": "language_model.layers.0.self_attn.q_proj",
              "shape": [896, 896],
              "in_dim": 896,
              "out_dim": 896,
              "file": "model.safetensors",
              "quant_file": "model.safetensors",
              "group_size": 32,
              "bits": 4,
              "mode": "affine"
            }
          ]
        }
        """.data(using: .utf8)!
        let manifest = try JSONDecoder().decode(VibeVoiceQuantizationManifest.self, from: json)
        XCTAssertEqual(manifest.bits, 4)
        XCTAssertEqual(manifest.groupSize, 32)
        XCTAssertEqual(manifest.mode, "affine")
        XCTAssertEqual(manifest.layers.count, 1)
        XCTAssertEqual(manifest.layers[0].inDim, 896)
        XCTAssertEqual(manifest.layers[0].outDim, 896)
    }

    func testQuantizationSpecDefaults() {
        let spec = VibeVoiceQuantizationSpec()
        XCTAssertEqual(spec.groupSize, 32)
        XCTAssertEqual(spec.bits, 8)
        XCTAssertEqual(spec.mode, .affine)
    }

    func testSupportedGroupSizesAndBits() {
        XCTAssertTrue(VibeVoiceQuantizer.supportedGroupSizes.contains(32))
        XCTAssertTrue(VibeVoiceQuantizer.supportedGroupSizes.contains(64))
        XCTAssertTrue(VibeVoiceQuantizer.supportedGroupSizes.contains(128))
        XCTAssertTrue(VibeVoiceQuantizer.supportedBits.contains(4))
        XCTAssertTrue(VibeVoiceQuantizer.supportedBits.contains(8))
    }
}

final class VibeVoiceKVCacheTests: XCTestCase {

    func testInitializeAndRetrieve() {
        let cache = KVCacheSimple()
        let k = MLXArray.ones([1, 2, 4, 8])
        let v = MLXArray.ones([1, 2, 4, 8]) * 2.0
        cache.initialize(keys: k, values: v)
        // KVCacheSimple must retain the stored values; we can't assert internal
        // layout but re-initialization shouldn't crash and shapes must be valid.
        XCTAssertNoThrow(cache.initialize(keys: k, values: v))
    }
}

final class VibeVoiceConstantsTests: XCTestCase {

    func testAudioConstantsIs24kHz() {
        XCTAssertEqual(AudioConstants.sampleRate, 24000)
    }

    func testTTSWindowConstants() {
        XCTAssertEqual(TTSConstants.textWindowSize, 5)
        XCTAssertEqual(TTSConstants.speechWindowSize, 6)
    }

    func testTokenConstants() {
        XCTAssertEqual(TokenConstants.eosTokenId, 151643)
        XCTAssertEqual(TokenConstants.negativeTextId, 151655)
    }
}

final class VibeVoiceDiffusionHeadTests: XCTestCase {

    func testDiffusionHeadForwardShape() {
        let headCfg = DiffusionHeadConfiguration()
        let head = VibeVoiceDiffusionHead(headCfg)

        let batch = 2
        let latentDim = headCfg.latentSize
        let condDim = headCfg.hiddenSize

        let noisy = MLXRandom.normal([batch, latentDim])
        let t = MLXArray([Float(1.0), Float(1.0)])
        let cond = MLXRandom.normal([batch, condDim])

        let out = head(noisyImages: noisy, timesteps: t, condition: cond)
        eval(out)
        XCTAssertEqual(out.shape, [batch, latentDim])
    }
}

final class VibeVoiceEOSClassifierTests: XCTestCase {

    func testEOSClassifierOutputShape() {
        let classifier = EOSClassifier(hiddenSize: 896)
        let x = MLXRandom.normal([4, 896])
        let out = classifier(x)
        eval(out)
        XCTAssertEqual(out.shape, [4, 1])
    }
}

// MARK: - E2E tests (model download required)

/// Real-model round-trip tests using `VibeVoiceTTSModel.fromPretrained`.
///
/// Prefixed `E2E` so CI `--skip E2E` filter excludes them. Downloads
/// `microsoft/VibeVoice-Realtime-0.5B` + `Qwen/Qwen2.5-0.5B` on first run via
/// our `HuggingFaceDownloader`.
///
/// Override model via `VIBEVOICE_MODEL`; provide a voice cache via
/// `VIBEVOICE_VOICE_CACHE=/path/to/voice.safetensors` for the generation test.
final class E2EVibeVoiceTests: XCTestCase {

    private static var _model: VibeVoiceTTSModel?
    private static var _modelLoaded = false

    private var model: VibeVoiceTTSModel {
        get throws {
            guard let m = Self._model else {
                throw XCTSkip("VibeVoice-Realtime-0.5B not loaded (download failed or skipped)")
            }
            return m
        }
    }

    override func setUp() async throws {
        try await super.setUp()
        guard !Self._modelLoaded else { return }
        Self._modelLoaded = true

        var config = VibeVoiceTTSModel.Configuration()
        if let override = ProcessInfo.processInfo.environment["VIBEVOICE_MODEL"] {
            config.modelId = override
        }
        config.numInferenceSteps = 10  // speed up tests

        do {
            Self._model = try await VibeVoiceTTSModel.fromPretrained(configuration: config)
        } catch {
            print("[E2EVibeVoice] Model load failed: \(error)")
            Self._model = nil
        }
    }

    func testModelLoadsWithExpectedConfig() throws {
        let m = try model
        // Realtime 0.5B: 24 total layers, 20 tts backbone layers, 24 kHz.
        XCTAssertEqual(m.inference.model.config.decoderConfig.hiddenLayers, 24)
        XCTAssertEqual(m.inference.model.config.ttsBackboneNumHiddenLayers, 20)
        XCTAssertEqual(m.inference.model.config.acousticVaeDim, 64)
        XCTAssertEqual(m.sampleRate, 24000)
    }

    func testGeneratesFromVoiceCacheProducesNonSilentAudio() async throws {
        let m = try model

        // Voice cache is not bundled. Point the test at a cache file via
        // VIBEVOICE_VOICE_CACHE=/path/to/name.safetensors.
        guard let cachePath = ProcessInfo.processInfo.environment["VIBEVOICE_VOICE_CACHE"],
              FileManager.default.fileExists(atPath: cachePath) else {
            throw XCTSkip("set VIBEVOICE_VOICE_CACHE=/path/to/voice.safetensors to run this test")
        }
        try m.loadVoice(from: cachePath)

        let pcm = try await m.generate(text: "Hello world.")
        XCTAssertGreaterThan(pcm.count, 0, "generated audio must have samples")

        // Non-silence: RMS > 1e-5
        var sumSq: Float = 0
        for s in pcm { sumSq += s * s }
        let rms = sqrt(sumSq / Float(pcm.count))
        XCTAssertGreaterThan(rms, 1e-5, "generated audio is silent (RMS=\(rms))")
    }

    func testSampleRateIs24kHz() throws {
        let m = try model
        XCTAssertEqual(m.sampleRate, 24000)
    }
}

// MARK: - 1.5B (long-form) E2E

/// 1.5B exercises the dual-encoder voice prefill path: acoustic + semantic
/// tokenizers summed at audio positions, no eos_classifier. Each test mints
/// a fresh voice cache from a tiny synthesized PCM clip (no external file
/// fixture), then generates text and asserts non-silent output.
///
/// Skips automatically if `aufklarer/VibeVoice-1.5B-MLX-INT4` isn't downloadable
/// or already cached. Set `VIBEVOICE_SKIP_1_5B=1` to force-skip.
final class E2EVibeVoice1_5BTests: XCTestCase {

    private static var _model: VibeVoiceTTSModel?
    private static var _modelLoaded = false

    private var model: VibeVoiceTTSModel {
        get throws {
            guard let m = Self._model else {
                throw XCTSkip("VibeVoice-1.5B-MLX-INT4 not loaded (download failed or skipped)")
            }
            return m
        }
    }

    /// Synthesize a 2 s mono PCM at 24 kHz — sine + low-pass noise. Not real
    /// speech, but enough non-silent energy across the bands the encoder
    /// processes to produce a deterministic voice cache.
    private func makeMonoTestSignal(seconds: Double = 2.0) -> [Float] {
        let n = Int(24000.0 * seconds)
        var out = [Float](repeating: 0, count: n)
        var rng = SystemRandomNumberGenerator()
        for i in 0..<n {
            let t = Float(i) / 24000.0
            let tone = 0.3 * sin(2 * .pi * 220 * t) + 0.2 * sin(2 * .pi * 440 * t)
            let noise = Float.random(in: -0.1...0.1, using: &rng)
            out[i] = tone + noise
        }
        return out
    }

    override func setUp() async throws {
        try await super.setUp()
        if ProcessInfo.processInfo.environment["VIBEVOICE_SKIP_1_5B"] == "1" {
            return
        }
        guard !Self._modelLoaded else { return }
        Self._modelLoaded = true

        var config = VibeVoiceTTSModel.Configuration.longForm1_5B
        // The 1.5B variant has no EOS classifier, so generation runs until
        // maxSpeechTokens. Default longForm1_5B is 4000 tokens (~9 min audio @ 7.5
        // Hz, ~7 min wallclock per test on M2 Max). Cap to 50 tokens (~7 s audio,
        // ~5 s wall) for CI throughput — we're validating the dual-encoder path,
        // not audio length.
        config.maxSpeechTokens = 50
        config.numInferenceSteps = 5
        do {
            Self._model = try await VibeVoiceTTSModel.fromPretrained(configuration: config)
        } catch {
            print("[E2EVibeVoice1.5B] Model load failed: \(error)")
            Self._model = nil
        }
    }

    func testModelHasSemanticTokenizer() throws {
        let m = try model
        // 1.5B config carries a semantic tokenizer.
        XCTAssertTrue(m.inference.model.config.hasSemanticTokenizer,
                      "1.5B config must declare a semantic_tokenizer_config")
        XCTAssertNotNil(m.inference.model.semanticTokenizer)
        XCTAssertNotNil(m.inference.model.semanticConnector)
        // 1.5B does not ship eos_classifier weights.
        XCTAssertFalse(m.inference.model.hasEosClassifier,
                       "1.5B bundles do not ship tts_eos_classifier weights")
    }

    /// Smoke round-trip: encode voice from synthetic sine + noise, generate
    /// a short English line, assert non-silent. Doesn't ASR-verify intent
    /// because the synthetic reference doesn't carry real speaker identity.
    /// The ASR-verified test is `testEnglishRoundTripASR` below.
    func testEnglishRoundTripSmoke() async throws {
        let m = try model
        let pcm = makeMonoTestSignal()
        let cacheURL = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("vibevoice_1_5b_en_\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: cacheURL) }

        try m.encodeAndSaveVoice(
            referenceAudio: pcm,
            sampleRate: 24000,
            transcript: "Test voice for English long-form generation.",
            to: cacheURL
        )

        let audio = try await m.generate(text: "Hello world.")
        XCTAssertGreaterThan(audio.count, 0, "must produce samples")

        var sumSq: Float = 0
        for s in audio { sumSq += s * s }
        let rms = sqrt(sumSq / Float(audio.count))
        XCTAssertGreaterThan(rms, 1e-5, "1.5B EN output is silent (RMS=\(rms))")
    }

    func testChineseRoundTrip() async throws {
        let m = try model
        let pcm = makeMonoTestSignal()
        let cacheURL = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("vibevoice_1_5b_zh_\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: cacheURL) }

        try m.encodeAndSaveVoice(
            referenceAudio: pcm,
            sampleRate: 24000,
            transcript: "用于中文长篇语音合成的测试音色。",
            to: cacheURL
        )

        let audio = try await m.generate(text: "你好，世界。")
        XCTAssertGreaterThan(audio.count, 0, "must produce samples")

        var sumSq: Float = 0
        for s in audio { sumSq += s * s }
        let rms = sqrt(sumSq / Float(audio.count))
        XCTAssertGreaterThan(rms, 1e-5, "1.5B ZH output is silent (RMS=\(rms))")
    }

    /// ASR-verified round-trip: requires VIBEVOICE_REFERENCE_AUDIO env to point
    /// at a real English speech WAV. Uses 1.5B + dual-encoder prefill via
    /// `VibeVoice15BTTSModel`, generates a known phrase, runs Nemotron
    /// Streaming ASR over the result, asserts the transcript contains key
    /// content words. This is the real correctness check.
    func testEnglishRoundTripASR() async throws {
        let m = try model
        guard let refPath = ProcessInfo.processInfo.environment["VIBEVOICE_REFERENCE_AUDIO"],
              FileManager.default.fileExists(atPath: refPath) else {
            throw XCTSkip("set VIBEVOICE_REFERENCE_AUDIO=/path/to/english_speech.wav to run ASR-verified test")
        }
        let refSamples = try AudioFileLoader.load(url: URL(fileURLWithPath: refPath), targetSampleRate: 24000)

        // Use the unified-LM 1.5B model — the 0.5B-style API can't drive 1.5B.
        var cfg = VibeVoice15BTTSModel.Configuration()
        cfg.numInferenceSteps = 20
        cfg.maxSpeechTokens = 500
        let tts15 = try await VibeVoice15BTTSModel.fromPretrained(configuration: cfg)
        // Prefer the cached config — but since we already loaded `m`, reload only if needed.
        _ = m

        let prompt = "Hello world. This is the one point five billion VibeVoice variant of the Microsoft text to speech model."
        let pcm = try await tts15.generate(
            text: prompt,
            referenceAudio: refSamples,
            referenceTranscript: "",
            sampleRate: 24000
        )
        XCTAssertGreaterThan(pcm.count, 24000, "expect more than 1 s of audio")

        // Save to a temp WAV for ASR.
        let wavURL = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("vibevoice_15b_asr_\(UUID().uuidString).wav")
        defer { try? FileManager.default.removeItem(at: wavURL) }
        try WAVWriter.write(samples: pcm, sampleRate: 24000, to: wavURL)

        // Transcribe with Nemotron (English, handles >30 s).
        let asr = try await NemotronStreamingASRModel.fromPretrained()
        let asrText = (try asr.transcribeAudio(pcm, sampleRate: 24000)).lowercased()
        print("[E2EVibeVoice1.5B ASR] generated transcript: \(asrText)")

        // Generation is stochastic (diffusion sampling + LM token sampling),
        // so we can't assert a specific transcript. Instead we assert two
        // properties:
        //   1) The audio is long enough that the model didn't bail
        //      immediately (>= 4 s). 4 s of speech ≈ ~10 words.
        //   2) The ASR transcript contains at least 2 of the 4 acoustic-
        //      robust anchor words from the prompt — proves the model is
        //      tracking the input text, not babbling.
        let durationSec = Double(pcm.count) / 24000.0
        XCTAssertGreaterThanOrEqual(durationSec, 4.0,
            "1.5B produced only \(durationSec) s — likely terminated early")

        let anchors = ["billion", "microsoft", "model", "speech"]
        let hits = anchors.filter { asrText.contains($0) }
        XCTAssertGreaterThanOrEqual(hits.count, 2,
            "ASR transcript matched only \(hits.count)/\(anchors.count) anchors → not tracking input. Got: \(asrText)")
    }
}
