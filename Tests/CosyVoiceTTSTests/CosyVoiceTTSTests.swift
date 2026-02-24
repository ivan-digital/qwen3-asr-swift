import XCTest
import MLX
@testable import CosyVoiceTTS
import Qwen3Common

final class CosyVoiceTTSConfigTests: XCTestCase {

    func testDefaultConfig() {
        let config = CosyVoiceConfig.default

        // LLM
        XCTAssertEqual(config.llm.hiddenSize, 896)
        XCTAssertEqual(config.llm.numLayers, 24)
        XCTAssertEqual(config.llm.numHeads, 14)
        XCTAssertEqual(config.llm.numKVHeads, 2)
        XCTAssertEqual(config.llm.headDim, 64)
        XCTAssertEqual(config.llm.intermediateSize, 4864)
        XCTAssertEqual(config.llm.textVocabSize, 151936)
        XCTAssertEqual(config.llm.speechTokenSize, 6561)
        XCTAssertEqual(config.llm.totalSpeechVocabSize, 6761)

        // Special tokens
        XCTAssertEqual(config.llm.sosToken, 6561)
        XCTAssertEqual(config.llm.eosToken, 6562)
        XCTAssertEqual(config.llm.taskIdToken, 6563)
        XCTAssertEqual(config.llm.fillToken, 6564)

        // DiT
        XCTAssertEqual(config.flow.dit.dim, 1024)
        XCTAssertEqual(config.flow.dit.depth, 22)
        XCTAssertEqual(config.flow.dit.heads, 16)
        XCTAssertEqual(config.flow.dit.dimHead, 64)
        XCTAssertEqual(config.flow.dit.ffMult, 2)
        XCTAssertEqual(config.flow.dit.ffDim, 2048)
        XCTAssertEqual(config.flow.dit.melDim, 80)

        // Flow
        XCTAssertEqual(config.flow.inputSize, 512)
        XCTAssertEqual(config.flow.vocabSize, 6561)
        XCTAssertEqual(config.flow.spkEmbedDim, 192)
        XCTAssertEqual(config.flow.tokenMelRatio, 2)
        XCTAssertEqual(config.flow.nTimesteps, 10)
        XCTAssertEqual(config.flow.cfgRate, 0.7, accuracy: 0.001)

        // HiFi-GAN
        XCTAssertEqual(config.hifigan.baseChannels, 512)
        XCTAssertEqual(config.hifigan.upsampleRates, [8, 5, 3])
        XCTAssertEqual(config.hifigan.totalUpsampleFactor, 120)
        XCTAssertEqual(config.hifigan.istftNFFT, 16)
        XCTAssertEqual(config.hifigan.istftHopLen, 4)
        XCTAssertEqual(config.hifigan.sampleRate, 24000)

        // Mel
        XCTAssertEqual(config.mel.nFFT, 1920)
        XCTAssertEqual(config.mel.numMels, 80)
        XCTAssertEqual(config.mel.hopSize, 480)

        // Sampling
        XCTAssertEqual(config.sampling.topK, 25)
        XCTAssertEqual(config.sampling.topP, 0.8, accuracy: 0.001)

        // Top-level
        XCTAssertEqual(config.sampleRate, 24000)
        XCTAssertEqual(config.chunkSize, 25)
    }

    func testConfigCodable() throws {
        let config = CosyVoiceConfig.default
        let data = try JSONEncoder().encode(config)
        let decoded = try JSONDecoder().decode(CosyVoiceConfig.self, from: data)

        XCTAssertEqual(decoded.llm.hiddenSize, config.llm.hiddenSize)
        XCTAssertEqual(decoded.flow.dit.depth, config.flow.dit.depth)
        XCTAssertEqual(decoded.hifigan.upsampleRates, config.hifigan.upsampleRates)
        XCTAssertEqual(decoded.sampleRate, config.sampleRate)
    }

    func testLLMConfigDimensions() {
        let config = CosyVoiceLLMConfig()

        // Verify head dimensions are consistent
        XCTAssertEqual(config.numHeads * config.headDim, config.hiddenSize)  // 14 * 64 = 896
        XCTAssertEqual(config.numKVHeads * config.headDim, 128)  // 2 * 64 = 128

        // Verify special token indices are sequential
        XCTAssertEqual(config.sosToken, config.speechTokenSize)
        XCTAssertEqual(config.eosToken, config.speechTokenSize + 1)
        XCTAssertEqual(config.taskIdToken, config.speechTokenSize + 2)
        XCTAssertEqual(config.fillToken, config.speechTokenSize + 3)
    }

    func testHiFiGANUpsampleMath() {
        let config = CosyVoiceHiFiGANConfig()

        // Total audio upsample: conv upsample * ISTFT hop = mel frames to audio samples
        let totalAudioUpsample = config.totalUpsampleFactor * config.istftHopLen
        XCTAssertEqual(totalAudioUpsample, 480)

        // At 24kHz with hop_size=480: mel frame rate = 24000/480 = 50 Hz
        XCTAssertEqual(config.sampleRate / totalAudioUpsample, 50)

        // Channel progression through upsampling
        var channels = config.baseChannels  // 512
        for _ in config.upsampleRates {
            channels /= 2
        }
        XCTAssertEqual(channels, 64)  // 512 → 256 → 128 → 64
    }

    func testErrorDescriptions() {
        let errors: [CosyVoiceTTSError] = [
            .modelLoadFailed("test"),
            .downloadFailed("test"),
            .invalidInput("test"),
            .generationFailed("test")
        ]

        for error in errors {
            XCTAssertNotNil(error.errorDescription)
            XCTAssertFalse(error.errorDescription!.isEmpty)
        }
    }
}

// MARK: - Weight Loading Tests (require converted safetensors)

final class CosyVoiceWeightLoadingTests: XCTestCase {

    // Run with: COSYVOICE_WEIGHTS=/path/to/cosyvoice3-mlx-4bit swift test --filter CosyVoiceWeightLoadingTests
    static let weightsDir: String? = ProcessInfo.processInfo.environment["COSYVOICE_WEIGHTS"]

    override func setUpWithError() throws {
        try XCTSkipUnless(Self.weightsDir != nil, "Set COSYVOICE_WEIGHTS=/path/to/cosyvoice3-mlx-4bit")
    }

    func testLoadHiFiGAN() throws {
        let dir = Self.weightsDir!
        let url = URL(fileURLWithPath: dir).appendingPathComponent("hifigan.safetensors")
        let config = CosyVoiceHiFiGANConfig()
        let hifigan = HiFiGANGenerator(config: config)

        // Load weights — should not crash
        try CosyVoiceWeightLoader.loadHiFiGAN(hifigan, from: url)
        print("HiFi-GAN weights loaded successfully")
    }

    func testLoadFlow() throws {
        let dir = Self.weightsDir!
        let url = URL(fileURLWithPath: dir).appendingPathComponent("flow.safetensors")
        let flow = CosyVoiceFlowModel(config: CosyVoiceFlowConfig())

        try CosyVoiceWeightLoader.loadFlow(flow, from: url)
        print("Flow weights loaded successfully")
    }

    func testLoadLLM() throws {
        let dir = Self.weightsDir!
        let url = URL(fileURLWithPath: dir).appendingPathComponent("llm.safetensors")
        let llm = CosyVoiceLLM(config: CosyVoiceLLMConfig())

        try CosyVoiceWeightLoader.loadLLM(llm, from: url)
        print("LLM weights loaded successfully")
    }
}

// MARK: - Tokenizer Tests (require tokenizer files in weights dir)

final class CosyVoiceTokenizerTests: XCTestCase {

    static let weightsDir: String? = ProcessInfo.processInfo.environment["COSYVOICE_WEIGHTS"]

    override func setUpWithError() throws {
        try XCTSkipUnless(Self.weightsDir != nil, "Set COSYVOICE_WEIGHTS=/path/to/cosyvoice3-mlx-4bit")
        let vocabPath = URL(fileURLWithPath: Self.weightsDir!).appendingPathComponent("vocab.json").path
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: vocabPath),
            "vocab.json not found in COSYVOICE_WEIGHTS dir")
    }

    func testTokenizerLoads() throws {
        let tokenizer = Qwen3Tokenizer()
        let vocabURL = URL(fileURLWithPath: Self.weightsDir!).appendingPathComponent("vocab.json")
        try tokenizer.load(from: vocabURL)

        // Basic sanity: should have loaded a large vocabulary
        let helloId = tokenizer.encode("Hello")
        XCTAssertFalse(helloId.isEmpty, "Should encode 'Hello' to non-empty tokens")
        print("'Hello' -> \(helloId)")
    }

    func testTokenizerEncodesEnglish() throws {
        let tokenizer = Qwen3Tokenizer()
        let vocabURL = URL(fileURLWithPath: Self.weightsDir!).appendingPathComponent("vocab.json")
        try tokenizer.load(from: vocabURL)

        let tokens = tokenizer.encode("Hello, how are you?")
        XCTAssertFalse(tokens.isEmpty)
        print("'Hello, how are you?' -> \(tokens) (\(tokens.count) tokens)")

        // Decode back should be close to original
        let decoded = tokenizer.decode(tokens: tokens)
        XCTAssertTrue(decoded.contains("Hello"), "Decoded should contain 'Hello', got: \(decoded)")
        XCTAssertTrue(decoded.contains("you"), "Decoded should contain 'you', got: \(decoded)")
    }

    func testTokenizerEncodesChinese() throws {
        let tokenizer = Qwen3Tokenizer()
        let vocabURL = URL(fileURLWithPath: Self.weightsDir!).appendingPathComponent("vocab.json")
        try tokenizer.load(from: vocabURL)

        let tokens = tokenizer.encode("你好世界")
        XCTAssertFalse(tokens.isEmpty)
        print("'你好世界' -> \(tokens) (\(tokens.count) tokens)")
    }

    func testTokenizerEncodesGerman() throws {
        let tokenizer = Qwen3Tokenizer()
        let vocabURL = URL(fileURLWithPath: Self.weightsDir!).appendingPathComponent("vocab.json")
        try tokenizer.load(from: vocabURL)

        let tokens = tokenizer.encode("Guten Tag, wie geht es Ihnen?")
        XCTAssertFalse(tokens.isEmpty)
        print("'Guten Tag, wie geht es Ihnen?' -> \(tokens) (\(tokens.count) tokens)")

        let decoded = tokenizer.decode(tokens: tokens)
        XCTAssertTrue(decoded.contains("Guten"), "Decoded should contain 'Guten', got: \(decoded)")
    }

    func testTokenizerRoundTrip() throws {
        let tokenizer = Qwen3Tokenizer()
        let vocabURL = URL(fileURLWithPath: Self.weightsDir!).appendingPathComponent("vocab.json")
        try tokenizer.load(from: vocabURL)

        let texts = [
            "Hello world",
            "The quick brown fox jumps over the lazy dog.",
            "你好世界",
            "Guten Tag!",
            "こんにちは世界",
        ]

        for text in texts {
            let tokens = tokenizer.encode(text)
            let decoded = tokenizer.decode(tokens: tokens)
            print("'\(text)' -> \(tokens.count) tokens -> '\(decoded)'")
            XCTAssertFalse(tokens.isEmpty, "Should encode '\(text)' to non-empty tokens")
        }
    }
}

// MARK: - Forward Pass Tests (require converted safetensors)

final class CosyVoiceForwardPassTests: XCTestCase {

    // Run with: COSYVOICE_WEIGHTS=./cosyvoice3-mlx-4bit swift test --filter CosyVoiceForwardPassTests
    static let weightsDir: String? = ProcessInfo.processInfo.environment["COSYVOICE_WEIGHTS"]

    override func setUpWithError() throws {
        try XCTSkipUnless(Self.weightsDir != nil, "Set COSYVOICE_WEIGHTS=/path/to/cosyvoice3-mlx-4bit")
    }

    func testHiFiGANForward() throws {
        let dir = Self.weightsDir!
        let config = CosyVoiceHiFiGANConfig()
        let hifigan = HiFiGANGenerator(config: config)
        try CosyVoiceWeightLoader.loadHiFiGAN(
            hifigan, from: URL(fileURLWithPath: dir).appendingPathComponent("hifigan.safetensors"))

        // Dummy mel: [1, 80, 20] (20 mel frames = ~0.4s at 50 Hz)
        let mel = MLXRandom.normal([1, 80, 20])
        let audio = hifigan(mel)
        eval(audio)

        // Expected: 20 mel frames * 120 (total upsample) * 4 (ISTFT hop) = 9600 samples
        let samples = audio.dim(audio.ndim - 1)
        print("HiFi-GAN output: \(audio.shape) (\(samples) samples, \(Double(samples)/24000.0)s)")
        XCTAssertGreaterThan(samples, 0, "Should produce audio samples")
    }

    func testFlowForward() throws {
        let dir = Self.weightsDir!
        let flow = CosyVoiceFlowModel(config: CosyVoiceFlowConfig())
        try CosyVoiceWeightLoader.loadFlow(
            flow, from: URL(fileURLWithPath: dir).appendingPathComponent("flow.safetensors"))

        // Dummy speech tokens: [1, 10] (10 tokens = 0.4s at 25 Hz)
        let tokens = MLXArray([0, 100, 200, 300, 400, 500, 1000, 2000, 3000, 4000]).expandedDimensions(axis: 0)
        let mel = flow(tokens: tokens)
        eval(mel)

        // Expected: [1, 80, 20] (10 tokens * tokenMelRatio=2 = 20 mel frames)
        print("Flow output: \(mel.shape)")
        let flat = mel.reshaped(-1)
        print("Flow mel range: [\(MLX.min(flat).item(Float.self)), \(MLX.max(flat).item(Float.self))], mean: \(MLX.mean(flat).item(Float.self))")
        XCTAssertEqual(mel.dim(0), 1, "Batch size should be 1")
        XCTAssertEqual(mel.dim(1), 80, "Should have 80 mel bins")
        XCTAssertEqual(mel.dim(2), 20, "Should have 20 mel frames (10 tokens * 2)")
    }

    func testLLMPrefill() throws {
        let dir = Self.weightsDir!
        let llmConfig = CosyVoiceLLMConfig()
        let llm = CosyVoiceLLM(config: llmConfig)
        try CosyVoiceWeightLoader.loadLLM(
            llm, from: URL(fileURLWithPath: dir).appendingPathComponent("llm.safetensors"))

        // Build prefix: [sos, text_tokens..., task_id]
        let textTokens: [Int32] = [9707, 1917]  // "Hello world" approx token IDs
        let prefix = llm.buildInputSequence(textTokens: textTokens)
        eval(prefix)

        // Expected: [1, 4, 896] (sos + 2 text tokens + task_id)
        print("LLM prefix: \(prefix.shape)")
        XCTAssertEqual(prefix.dim(0), 1, "Batch size should be 1")
        XCTAssertEqual(prefix.dim(1), 4, "Prefix should be 4 tokens (sos + 2 + task_id)")
        XCTAssertEqual(prefix.dim(2), llmConfig.hiddenSize, "Hidden dim should be \(llmConfig.hiddenSize)")

        // Prefill forward pass
        let offset = MLXArray(Int32(0))
        let (logits, cache) = llm.forwardStep(prefix, offset: offset, cache: nil)
        eval(logits, cache)

        print("LLM logits: \(logits.shape)")
        XCTAssertEqual(logits.dim(0), 1, "Batch size should be 1")
        XCTAssertEqual(logits.dim(1), 4, "Sequence length should be 4")
        XCTAssertEqual(logits.dim(2), llmConfig.totalSpeechVocabSize, "Vocab size should be \(llmConfig.totalSpeechVocabSize)")

        // Check cache was created for all layers
        XCTAssertEqual(cache.count, llmConfig.numLayers, "Should have cache for all \(llmConfig.numLayers) layers")
    }

    func testLLMGenerate5Tokens() throws {
        let dir = Self.weightsDir!
        let llm = CosyVoiceLLM(config: CosyVoiceLLMConfig())
        try CosyVoiceWeightLoader.loadLLM(
            llm, from: URL(fileURLWithPath: dir).appendingPathComponent("llm.safetensors"))

        // Generate just 5 tokens to verify the loop works
        let textTokens: [Int32] = [9707, 1917]  // approximate "Hello world"
        let tokens = llm.generate(textTokens: textTokens, maxTokens: 5)

        print("LLM generated \(tokens.count) tokens: \(tokens)")
        XCTAssertGreaterThan(tokens.count, 0, "Should generate at least 1 token")
        XCTAssertLessThanOrEqual(tokens.count, 5, "Should not exceed maxTokens")

        // All tokens should be valid speech tokens (0-6560) or EOS wasn't hit
        for token in tokens {
            XCTAssertGreaterThanOrEqual(token, 0, "Token should be >= 0")
            XCTAssertLessThan(token, Int32(6561), "Token should be < 6561 (speech token range)")
        }
    }
    func testFullPipelineE2E() throws {
        let dir = Self.weightsDir!
        let config = CosyVoiceConfig.default

        // 1. Load all three models
        let llm = CosyVoiceLLM(config: config.llm)
        try CosyVoiceWeightLoader.loadLLM(
            llm, from: URL(fileURLWithPath: dir).appendingPathComponent("llm.safetensors"))

        let flow = CosyVoiceFlowModel(config: config.flow)
        try CosyVoiceWeightLoader.loadFlow(
            flow, from: URL(fileURLWithPath: dir).appendingPathComponent("flow.safetensors"))

        let hifigan = HiFiGANGenerator(config: config.hifigan)
        try CosyVoiceWeightLoader.loadHiFiGAN(
            hifigan, from: URL(fileURLWithPath: dir).appendingPathComponent("hifigan.safetensors"))

        print("All models loaded")

        // 2. Tokenize text with real BPE tokenizer
        let tokenizer = Qwen3Tokenizer()
        let vocabURL = URL(fileURLWithPath: dir).appendingPathComponent("vocab.json")
        try tokenizer.load(from: vocabURL)
        let textTokens = tokenizer.encode("Hello world").map { Int32($0) }
        print("Tokenized 'Hello world' -> \(textTokens)")

        let start = CFAbsoluteTimeGetCurrent()
        let speechTokens = llm.generate(textTokens: textTokens, maxTokens: 50)
        let llmTime = CFAbsoluteTimeGetCurrent() - start

        print("LLM generated \(speechTokens.count) speech tokens in \(String(format: "%.2f", llmTime))s: \(speechTokens.prefix(10))...")
        XCTAssertGreaterThan(speechTokens.count, 0, "LLM should generate at least 1 speech token")

        // 3. Convert speech tokens to mel via flow matching
        let tokenArray = MLXArray(speechTokens).expandedDimensions(axis: 0)  // [1, T]
        let flowStart = CFAbsoluteTimeGetCurrent()
        let mel = flow(tokens: tokenArray)  // [1, 80, T_mel]
        eval(mel)
        let flowTime = CFAbsoluteTimeGetCurrent() - flowStart

        let melFrames = mel.dim(2)
        print("Flow: \(mel.shape) (\(melFrames) mel frames) in \(String(format: "%.2f", flowTime))s")
        XCTAssertEqual(mel.dim(1), 80, "Should have 80 mel bins")
        XCTAssertEqual(melFrames, speechTokens.count * 2, "Mel frames = tokens * 2")

        // 4. Convert mel to audio via HiFi-GAN
        let vocoderStart = CFAbsoluteTimeGetCurrent()
        let audio = hifigan(mel)
        eval(audio)
        let vocoderTime = CFAbsoluteTimeGetCurrent() - vocoderStart

        let audioSamples = audio.dim(audio.ndim - 1)
        let duration = Double(audioSamples) / 24000.0
        let totalTime = llmTime + flowTime + vocoderTime
        print("HiFi-GAN: \(audio.shape) (\(audioSamples) samples, \(String(format: "%.2f", duration))s audio)")
        print("Total pipeline: \(String(format: "%.2f", totalTime))s (LLM: \(String(format: "%.2f", llmTime))s, Flow: \(String(format: "%.2f", flowTime))s, Vocoder: \(String(format: "%.2f", vocoderTime))s)")

        XCTAssertGreaterThan(audioSamples, 0, "Should produce audio samples")

        // 5. Verify audio is in valid range [-1, 1]
        let flatAudio = audio.reshaped(-1)
        let maxVal = MLX.max(flatAudio).item(Float.self)
        let minVal = MLX.min(flatAudio).item(Float.self)
        print("Audio range: [\(String(format: "%.4f", minVal)), \(String(format: "%.4f", maxVal))]")
        XCTAssertGreaterThanOrEqual(minVal, -1.0, "Audio min should be >= -1.0")
        XCTAssertLessThanOrEqual(maxVal, 1.0, "Audio max should be <= 1.0")

        // 6. Check audio is not silent
        let maxAmp = Swift.max(abs(minVal), abs(maxVal))
        XCTAssertGreaterThan(maxAmp, 0.001, "Audio should not be silent")
    }
}

// MARK: - E2E Tests (require model download)

// These tests download the model (~2 GB) on first run and cache it.
// Run with: swift test --filter CosyVoiceTTSE2ETests

final class CosyVoiceTTSE2ETests: XCTestCase {

    func testBasicSynthesis() async throws {
        let model = try await CosyVoiceTTSModel.fromPretrained()
        let samples = model.synthesize(text: "Hello world")

        XCTAssertFalse(samples.isEmpty, "Should produce audio")
        let duration = Double(samples.count) / 24000.0
        XCTAssertGreaterThan(duration, 0.5, "Should be at least 0.5s")
        XCTAssertLessThan(duration, 10.0, "Should be less than 10s")

        // Check not silent
        let maxAmp = samples.map { abs($0) }.max() ?? 0
        XCTAssertGreaterThan(maxAmp, 0.01, "Should not be silent")
    }

    func testGermanSynthesis() async throws {
        let model = try await CosyVoiceTTSModel.fromPretrained()
        let samples = model.synthesize(text: "Guten Tag, wie geht es Ihnen?", language: "german")

        XCTAssertFalse(samples.isEmpty, "Should produce German audio")
        let maxAmp = samples.map { abs($0) }.max() ?? 0
        XCTAssertGreaterThan(maxAmp, 0.01, "Should not be silent")
    }

    func testChineseSynthesis() async throws {
        let model = try await CosyVoiceTTSModel.fromPretrained()
        let samples = model.synthesize(text: "你好世界", language: "chinese")

        XCTAssertFalse(samples.isEmpty, "Should produce Chinese audio")
    }

    func testStreamingSynthesis() async throws {
        let model = try await CosyVoiceTTSModel.fromPretrained()

        var chunks: [AudioChunk] = []
        for try await chunk in model.synthesizeStream(text: "Hello world") {
            chunks.append(chunk)
        }

        XCTAssertFalse(chunks.isEmpty, "Should produce at least one chunk")
        XCTAssertTrue(chunks.last!.isFinal, "Last chunk should be final")
    }

    func testEmptyTextHandling() async throws {
        let model = try await CosyVoiceTTSModel.fromPretrained()
        let samples = model.synthesize(text: "")

        // Empty text should produce empty or very short audio
        XCTAssertTrue(samples.isEmpty || samples.count < 24000,
                      "Empty text should not produce long audio")
    }

    func testMaxLengthSafety() async throws {
        let model = try await CosyVoiceTTSModel.fromPretrained()
        let longText = String(repeating: "This is a long test sentence. ", count: 50)
        let samples = model.synthesize(text: longText)

        let duration = Double(samples.count) / 24000.0
        // Max 500 tokens at 25 Hz = 20s, plus some buffer
        XCTAssertLessThan(duration, 25.0, "Should not exceed safety limit")
    }
}
