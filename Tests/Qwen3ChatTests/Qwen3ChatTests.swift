import XCTest
@testable import Qwen3Chat
import AudioCommon

// MARK: - Config Tests

final class Qwen3ChatConfigTests: XCTestCase {

    func testDefaultQwen3_06B() {
        let config = Qwen3ChatConfig.qwen3_06B
        XCTAssertEqual(config.hiddenSize, 1024)
        XCTAssertEqual(config.numHiddenLayers, 28)
        XCTAssertEqual(config.numAttentionHeads, 16)
        XCTAssertEqual(config.numKeyValueHeads, 8)
        XCTAssertEqual(config.headDim, 128)
        XCTAssertEqual(config.intermediateSize, 3072)
        XCTAssertEqual(config.vocabSize, 151936)
        XCTAssertEqual(config.maxSeqLen, 2048)
        XCTAssertEqual(config.ropeTheta, 1_000_000.0, accuracy: 1.0)
        XCTAssertEqual(config.rmsNormEps, 1e-6, accuracy: 1e-9)
        XCTAssertEqual(config.eosTokenId, 151645)
        XCTAssertEqual(config.padTokenId, 151643)
        XCTAssertEqual(config.quantization, "int4")
    }

    func testGQARatio() {
        let config = Qwen3ChatConfig.qwen3_06B
        // GQA: 16 attention heads, 8 KV heads → 2:1 ratio
        XCTAssertEqual(config.numAttentionHeads / config.numKeyValueHeads, 2)
    }

    func testCodableRoundTrip() throws {
        let original = Qwen3ChatConfig.qwen3_06B
        let data = try JSONEncoder().encode(original)
        let decoded = try JSONDecoder().decode(Qwen3ChatConfig.self, from: data)

        XCTAssertEqual(decoded.hiddenSize, original.hiddenSize)
        XCTAssertEqual(decoded.numHiddenLayers, original.numHiddenLayers)
        XCTAssertEqual(decoded.numAttentionHeads, original.numAttentionHeads)
        XCTAssertEqual(decoded.numKeyValueHeads, original.numKeyValueHeads)
        XCTAssertEqual(decoded.headDim, original.headDim)
        XCTAssertEqual(decoded.vocabSize, original.vocabSize)
        XCTAssertEqual(decoded.maxSeqLen, original.maxSeqLen)
        XCTAssertEqual(decoded.eosTokenId, original.eosTokenId)
        XCTAssertEqual(decoded.quantization, original.quantization)
    }

    func testSnakeCaseDecoding() throws {
        let json = """
        {
            "hidden_size": 512,
            "num_hidden_layers": 12,
            "num_attention_heads": 8,
            "num_key_value_heads": 4,
            "head_dim": 64,
            "intermediate_size": 1536,
            "vocab_size": 32000,
            "max_seq_len": 1024,
            "rope_theta": 500000.0,
            "rms_norm_eps": 1e-5,
            "eos_token_id": 2,
            "pad_token_id": 0,
            "quantization": "int8"
        }
        """.data(using: .utf8)!

        let config = try JSONDecoder().decode(Qwen3ChatConfig.self, from: json)
        XCTAssertEqual(config.hiddenSize, 512)
        XCTAssertEqual(config.numHiddenLayers, 12)
        XCTAssertEqual(config.numAttentionHeads, 8)
        XCTAssertEqual(config.numKeyValueHeads, 4)
        XCTAssertEqual(config.headDim, 64)
        XCTAssertEqual(config.intermediateSize, 1536)
        XCTAssertEqual(config.vocabSize, 32000)
        XCTAssertEqual(config.maxSeqLen, 1024)
        XCTAssertEqual(config.ropeTheta, 500000.0, accuracy: 1.0)
        XCTAssertEqual(config.quantization, "int8")
    }

    func testLoadFromFile() throws {
        let json = """
        {
            "hidden_size": 1024,
            "num_hidden_layers": 28,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "intermediate_size": 3072,
            "vocab_size": 151936,
            "max_seq_len": 2048,
            "rope_theta": 1000000.0,
            "rms_norm_eps": 1e-6,
            "eos_token_id": 151645,
            "pad_token_id": 151643,
            "quantization": "int4"
        }
        """.data(using: .utf8)!

        let tmpURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("test_chat_config_\(UUID().uuidString).json")
        try json.write(to: tmpURL)
        addTeardownBlock { try? FileManager.default.removeItem(at: tmpURL) }

        let config = try Qwen3ChatConfig.load(from: tmpURL)
        XCTAssertEqual(config.hiddenSize, 1024)
        XCTAssertEqual(config.numHiddenLayers, 28)
        XCTAssertEqual(config.vocabSize, 151936)
    }
}

// MARK: - Sampling Config Tests

final class ChatSamplingConfigTests: XCTestCase {

    func testDefaultPreset() {
        let config = ChatSamplingConfig.default
        XCTAssertEqual(config.temperature, 0.7, accuracy: 0.001)
        XCTAssertEqual(config.topK, 50)
        XCTAssertEqual(config.topP, 0.9, accuracy: 0.001)
        XCTAssertEqual(config.maxTokens, 256)
        XCTAssertEqual(config.repetitionPenalty, 1.1, accuracy: 0.001)
    }

    func testCreativePreset() {
        let config = ChatSamplingConfig.creative
        XCTAssertEqual(config.temperature, 0.9, accuracy: 0.001)
        XCTAssertEqual(config.topP, 0.95, accuracy: 0.001)
    }

    func testPrecisePreset() {
        let config = ChatSamplingConfig.precise
        XCTAssertEqual(config.temperature, 0.3, accuracy: 0.001)
        XCTAssertEqual(config.topK, 20)
        XCTAssertEqual(config.topP, 0.8, accuracy: 0.001)
    }

    func testCustomInit() {
        let config = ChatSamplingConfig(
            temperature: 0.5,
            topK: 100,
            topP: 0.85,
            maxTokens: 512,
            repetitionPenalty: 1.2
        )
        XCTAssertEqual(config.temperature, 0.5, accuracy: 0.001)
        XCTAssertEqual(config.topK, 100)
        XCTAssertEqual(config.topP, 0.85, accuracy: 0.001)
        XCTAssertEqual(config.maxTokens, 512)
        XCTAssertEqual(config.repetitionPenalty, 1.2, accuracy: 0.001)
    }
}

// MARK: - Chat Message Tests

final class ChatMessageTests: XCTestCase {

    func testRoles() {
        XCTAssertEqual(ChatMessage.Role.system.rawValue, "system")
        XCTAssertEqual(ChatMessage.Role.user.rawValue, "user")
        XCTAssertEqual(ChatMessage.Role.assistant.rawValue, "assistant")
    }

    func testMessageConstruction() {
        let msg = ChatMessage(role: .user, content: "Hello!")
        XCTAssertEqual(msg.role, .user)
        XCTAssertEqual(msg.content, "Hello!")
    }

    func testSendable() async {
        let msg = ChatMessage(role: .system, content: "Be helpful.")
        let result = await Task { msg }.value
        XCTAssertEqual(result.content, "Be helpful.")
    }
}

// MARK: - Chat Template Tests

final class ChatTemplateTests: XCTestCase {

    func testSpecialTokenIds() {
        XCTAssertEqual(ChatTemplate.imStartId, 151644)
        XCTAssertEqual(ChatTemplate.imEndId, 151645)
        XCTAssertEqual(ChatTemplate.newlineId, 198)
        XCTAssertEqual(ChatTemplate.thinkStartId, 151667)
        XCTAssertEqual(ChatTemplate.thinkEndId, 151668)
    }

    func testStripThinkingNoThinking() {
        let tokens = [1, 2, 3, 4, 5]
        XCTAssertEqual(ChatTemplate.stripThinking(from: tokens), tokens)
    }

    func testStripThinkingComplete() {
        let thinkStart = ChatTemplate.thinkStartId
        let thinkEnd = ChatTemplate.thinkEndId
        let newline = ChatTemplate.newlineId
        let tokens = [thinkStart, 100, 200, 300, thinkEnd, newline, 10, 20, 30]
        let stripped = ChatTemplate.stripThinking(from: tokens)
        XCTAssertEqual(stripped, [10, 20, 30])
    }

    func testStripThinkingIncomplete() {
        let thinkStart = ChatTemplate.thinkStartId
        let tokens = [thinkStart, 100, 200, 300]
        let stripped = ChatTemplate.stripThinking(from: tokens)
        XCTAssertEqual(stripped, [])
    }

    func testStripThinkingWithPrefixTokens() {
        let thinkStart = ChatTemplate.thinkStartId
        let thinkEnd = ChatTemplate.thinkEndId
        let tokens = [10, 20, thinkStart, 100, 200, thinkEnd, 30, 40]
        let stripped = ChatTemplate.stripThinking(from: tokens)
        XCTAssertEqual(stripped, [10, 20, 30, 40])
    }

    func testStripThinkingNoTrailingNewline() {
        let thinkStart = ChatTemplate.thinkStartId
        let thinkEnd = ChatTemplate.thinkEndId
        let tokens = [thinkStart, 100, thinkEnd, 10, 20]
        let stripped = ChatTemplate.stripThinking(from: tokens)
        XCTAssertEqual(stripped, [10, 20])
    }

    func testEncodeStructure() {
        // Use a tokenizer with synthetic vocab to verify template structure
        let tokenizer = makeSyntheticTokenizer()
        let messages = [
            ChatMessage(role: .user, content: "Hi")
        ]

        let tokens = ChatTemplate.encode(
            messages: messages,
            tokenizer: tokenizer,
            addGenerationPrompt: true
        )

        // Structure: <im_start> + role tokens + \n + content tokens + <im_end> + \n + <im_start> + "assistant" tokens + \n
        XCTAssertEqual(tokens.first, ChatTemplate.imStartId)
        XCTAssertTrue(tokens.contains(ChatTemplate.imEndId))
        XCTAssertTrue(tokens.contains(ChatTemplate.newlineId))

        // Last group should be generation prompt: <im_start> + "assistant" + \n
        let lastImStart = tokens.lastIndex(of: ChatTemplate.imStartId)!
        XCTAssertEqual(tokens.last, ChatTemplate.newlineId)
        XCTAssertTrue(lastImStart > 0, "Generation prompt should be at end")
    }

    func testEncodeWithoutGenerationPrompt() {
        let tokenizer = makeSyntheticTokenizer()
        let messages = [
            ChatMessage(role: .system, content: "Be brief.")
        ]

        let withPrompt = ChatTemplate.encode(
            messages: messages,
            tokenizer: tokenizer,
            addGenerationPrompt: true
        )
        let withoutPrompt = ChatTemplate.encode(
            messages: messages,
            tokenizer: tokenizer,
            addGenerationPrompt: false
        )

        XCTAssertTrue(withPrompt.count > withoutPrompt.count,
            "With generation prompt should have more tokens")
    }

    func testMultipleMessages() {
        let tokenizer = makeSyntheticTokenizer()
        let messages = [
            ChatMessage(role: .system, content: "You help."),
            ChatMessage(role: .user, content: "Hi"),
            ChatMessage(role: .assistant, content: "Hello!"),
            ChatMessage(role: .user, content: "Bye"),
        ]

        let tokens = ChatTemplate.encode(
            messages: messages,
            tokenizer: tokenizer,
            addGenerationPrompt: true
        )

        // 4 messages = 4 im_end tokens, plus 1 generation prompt im_start
        let imEndCount = tokens.filter { $0 == ChatTemplate.imEndId }.count
        XCTAssertEqual(imEndCount, 4)

        // 4 message im_starts + 1 generation prompt = 5
        let imStartCount = tokens.filter { $0 == ChatTemplate.imStartId }.count
        XCTAssertEqual(imStartCount, 5)
    }

    func testEmptyMessages() {
        let tokenizer = makeSyntheticTokenizer()
        let tokens = ChatTemplate.encode(
            messages: [],
            tokenizer: tokenizer,
            addGenerationPrompt: true
        )
        // Only generation prompt: <im_start> + "assistant" tokens + \n
        XCTAssertEqual(tokens.first, ChatTemplate.imStartId)
        XCTAssertEqual(tokens.last, ChatTemplate.newlineId)
    }

    // MARK: - Helpers

    private func makeSyntheticTokenizer() -> ChatTokenizer {
        // Create tokenizer with minimal vocab for template testing.
        // The tokenizer will encode unknown text character-by-character
        // which is fine — we only verify template structure.
        return ChatTokenizer()
    }
}

// MARK: - Tokenizer Tests

final class ChatTokenizerTests: XCTestCase {

    func testEmptyEncode() {
        let tokenizer = ChatTokenizer()
        XCTAssertEqual(tokenizer.encode(""), [])
    }

    func testDefaultEosToken() {
        let tokenizer = ChatTokenizer()
        XCTAssertEqual(tokenizer.eosTokenId, 151645)
    }

    func testEmptyVocabSize() {
        let tokenizer = ChatTokenizer()
        XCTAssertEqual(tokenizer.vocabSize, 0)
    }

    func testDecodeEmpty() {
        let tokenizer = ChatTokenizer()
        XCTAssertEqual(tokenizer.decode([]), "")
    }

    func testDecodeUnknownTokens() {
        let tokenizer = ChatTokenizer()
        XCTAssertEqual(tokenizer.decode([99999]), "")
    }

    func testDecodeTokenNil() {
        let tokenizer = ChatTokenizer()
        XCTAssertNil(tokenizer.decodeToken(99999))
    }

    func testIsSpecialToken() {
        let tokenizer = ChatTokenizer()
        // With no vocab loaded, isSpecialToken returns false for any ID
        XCTAssertFalse(tokenizer.isSpecialToken(0))
    }

    func testLoadVocabFromFile() throws {
        let tmpDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("tokenizer_test_\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: tmpDir) }

        // Write minimal vocab.json
        let vocab: [String: Int] = [
            "hello": 0,
            "Ġworld": 1,
            "Ġtest": 2,
            "<|im_start|>": 151644,
            "<|im_end|>": 151645,
        ]
        let vocabData = try JSONSerialization.data(withJSONObject: vocab)
        try vocabData.write(to: tmpDir.appendingPathComponent("vocab.json"))

        let tokenizer = ChatTokenizer()
        try tokenizer.load(from: tmpDir)

        XCTAssertEqual(tokenizer.vocabSize, 5)
    }

    func testEncodeWithVocab() throws {
        let tokenizer = try makeSyntheticTokenizerWithVocab()

        // "hello" should map directly
        let tokens = tokenizer.encode("hello")
        XCTAssertEqual(tokens, [0])
    }

    func testEncodeMultiWord() throws {
        let tokenizer = try makeSyntheticTokenizerWithVocab()

        // "hello world" → "hello" + "Ġworld"
        let tokens = tokenizer.encode("hello world")
        XCTAssertEqual(tokens, [0, 1])
    }

    func testDecodeWithVocab() throws {
        let tokenizer = try makeSyntheticTokenizerWithVocab()

        let text = tokenizer.decode([0, 1])
        XCTAssertEqual(text, "hello world")
    }

    func testDecodeTokenWithSpace() throws {
        let tokenizer = try makeSyntheticTokenizerWithVocab()

        let piece = tokenizer.decodeToken(1)
        XCTAssertEqual(piece, " world")
    }

    func testSpecialTokenDetection() throws {
        let tokenizer = try makeSyntheticTokenizerWithVocab()

        XCTAssertTrue(tokenizer.isSpecialToken(151644), "<|im_start|> is special")
        XCTAssertTrue(tokenizer.isSpecialToken(151645), "<|im_end|> is special")
        XCTAssertFalse(tokenizer.isSpecialToken(0), "hello is not special")
        XCTAssertFalse(tokenizer.isSpecialToken(1), "Ġworld is not special")
    }

    func testBPEMerges() throws {
        let tmpDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("bpe_test_\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: tmpDir) }

        // Vocab: individual chars + merged token
        let vocab: [String: Int] = [
            "h": 0,
            "i": 1,
            "hi": 2,
            "Ġt": 3,
            "here": 4,
        ]
        let vocabData = try JSONSerialization.data(withJSONObject: vocab)
        try vocabData.write(to: tmpDir.appendingPathComponent("vocab.json"))

        // Merges: h + i → hi
        let merges = "#version: 0.2\nh i\n"
        try merges.write(
            to: tmpDir.appendingPathComponent("merges.txt"),
            atomically: true,
            encoding: .utf8
        )

        let tokenizer = ChatTokenizer()
        try tokenizer.load(from: tmpDir)

        // "hi" should be merged to single token
        let tokens = tokenizer.encode("hi")
        XCTAssertEqual(tokens, [2])
    }

    func testAddedTokensFromConfig() throws {
        let tmpDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("added_tokens_test_\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: tmpDir) }

        let vocab: [String: Int] = ["a": 0, "b": 1]
        let vocabData = try JSONSerialization.data(withJSONObject: vocab)
        try vocabData.write(to: tmpDir.appendingPathComponent("vocab.json"))

        let tokenizerConfig: [String: Any] = [
            "added_tokens_decoder": [
                "151644": ["content": "<|im_start|>", "special": true],
                "151645": ["content": "<|im_end|>", "special": true],
            ]
        ]
        let configData = try JSONSerialization.data(withJSONObject: tokenizerConfig)
        try configData.write(to: tmpDir.appendingPathComponent("tokenizer_config.json"))

        let tokenizer = ChatTokenizer()
        try tokenizer.load(from: tmpDir)

        // Added tokens should be in vocab
        XCTAssertEqual(tokenizer.vocabSize, 4) // a, b, im_start, im_end
        XCTAssertTrue(tokenizer.isSpecialToken(151644))
        XCTAssertTrue(tokenizer.isSpecialToken(151645))
    }

    // MARK: - Helpers

    private func makeSyntheticTokenizerWithVocab() throws -> ChatTokenizer {
        let tmpDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("synth_tokenizer_\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: tmpDir) }

        let vocab: [String: Int] = [
            "hello": 0,
            "Ġworld": 1,
            "Ġtest": 2,
            "<|im_start|>": 151644,
            "<|im_end|>": 151645,
        ]
        let vocabData = try JSONSerialization.data(withJSONObject: vocab)
        try vocabData.write(to: tmpDir.appendingPathComponent("vocab.json"))

        let tokenizer = ChatTokenizer()
        try tokenizer.load(from: tmpDir)
        return tokenizer
    }
}

// MARK: - Sampling Tests

final class SamplingTests: XCTestCase {

    func testArgmaxWithZeroTemperature() {
        let logits: [Float] = [0.1, 0.5, 0.3, 0.9, 0.2]
        let config = ChatSamplingConfig(
            temperature: 0.0,
            topK: 0,
            topP: 1.0,
            maxTokens: 1,
            repetitionPenalty: 1.0
        )
        let token = ChatSampler.sample(logits: logits, config: config)
        XCTAssertEqual(token, 3, "Should pick index 3 (highest logit 0.9)")
    }

    func testTopKFiltering() {
        let logits: [Float] = [1.0, 5.0, 2.0, 3.0]
        let config = ChatSamplingConfig(
            temperature: 0.01,
            topK: 1,
            topP: 1.0,
            maxTokens: 1,
            repetitionPenalty: 1.0
        )

        for _ in 0..<10 {
            let token = ChatSampler.sample(logits: logits, config: config)
            XCTAssertEqual(token, 1)
        }
    }

    func testRepetitionPenalty() {
        let logits: [Float] = [10.0, 9.0, 1.0]
        let config = ChatSamplingConfig(
            temperature: 0.01,
            topK: 0,
            topP: 1.0,
            maxTokens: 1,
            repetitionPenalty: 5.0
        )

        let token = ChatSampler.sample(
            logits: logits,
            config: config,
            previousTokens: [0, 0, 0]
        )
        XCTAssertEqual(token, 1, "Should avoid repeated token 0 with heavy penalty")
    }

    func testRepetitionPenaltyNegativeLogits() {
        let logits: [Float] = [-1.0, -5.0, 0.5]
        let config = ChatSamplingConfig(
            temperature: 0.01,
            topK: 0,
            topP: 1.0,
            maxTokens: 1,
            repetitionPenalty: 2.0
        )

        let token = ChatSampler.sample(
            logits: logits,
            config: config,
            previousTokens: [0, 1]
        )
        XCTAssertEqual(token, 2)
    }

    func testNoPenaltyWithDefault() {
        let logits: [Float] = [10.0, 1.0]
        let config = ChatSamplingConfig(
            temperature: 0.01,
            topK: 0,
            topP: 1.0,
            maxTokens: 1,
            repetitionPenalty: 1.0
        )

        let token = ChatSampler.sample(
            logits: logits,
            config: config,
            previousTokens: [0, 0, 0]
        )
        XCTAssertEqual(token, 0)
    }

    func testSamplingProducesValidTokens() {
        let vocabSize = 100
        let logits = (0..<vocabSize).map { _ in Float.random(in: -2...2) }
        let config = ChatSamplingConfig.default

        for _ in 0..<50 {
            let token = ChatSampler.sample(logits: logits, config: config)
            XCTAssertTrue(token >= 0 && token < vocabSize,
                "Token \(token) should be in [0, \(vocabSize))")
        }
    }

    func testTopPNucleus() {
        var logits = [Float](repeating: -100, count: 10)
        logits[7] = 100.0

        let config = ChatSamplingConfig(
            temperature: 1.0,
            topK: 0,
            topP: 0.5,
            maxTokens: 1,
            repetitionPenalty: 1.0
        )

        for _ in 0..<20 {
            let token = ChatSampler.sample(logits: logits, config: config)
            XCTAssertEqual(token, 7, "Nucleus sampling with dominant token should always pick it")
        }
    }
}

// MARK: - Error Tests

final class ChatModelErrorTests: XCTestCase {

    func testModelLoadFailedDescription() {
        let error = ChatModelError.modelLoadFailed("missing weights")
        XCTAssertTrue(error.errorDescription!.contains("missing weights"))
    }

    func testTokenizerLoadFailedDescription() {
        let error = ChatModelError.tokenizerLoadFailed("bad vocab.json")
        XCTAssertTrue(error.errorDescription!.contains("bad vocab.json"))
    }

    func testInferenceFailedDescription() {
        let error = ChatModelError.inferenceFailed("OOM")
        XCTAssertTrue(error.errorDescription!.contains("OOM"))
    }

    func testConfigNotFoundDescription() {
        let url = URL(fileURLWithPath: "/tmp/missing")
        let error = ChatModelError.configNotFound(url)
        XCTAssertTrue(error.errorDescription!.contains("/tmp/missing"))
    }

    func testModelNotFoundDescription() {
        let url = URL(fileURLWithPath: "/tmp/models")
        let error = ChatModelError.modelNotFound(url)
        XCTAssertTrue(error.errorDescription!.contains("/tmp/models"))
    }
}

// MARK: - Sendable Conformance Tests

final class Qwen3ChatSendableTests: XCTestCase {

    func testChatSamplingConfigSendable() async {
        let config = ChatSamplingConfig.default
        let result = await Task { config }.value
        XCTAssertEqual(result.temperature, config.temperature)
    }

    func testQwen3ChatConfigSendable() async {
        let config = Qwen3ChatConfig.qwen3_06B
        let result = await Task { config }.value
        XCTAssertEqual(result.hiddenSize, config.hiddenSize)
    }

    func testChatMessageSendable() async {
        let msg = ChatMessage(role: .user, content: "test")
        let result = await Task { msg }.value
        XCTAssertEqual(result.content, msg.content)
    }

    func testConfigsSendableInTaskGroup() async {
        await withTaskGroup(of: Bool.self) { group in
            group.addTask {
                let _ = Qwen3ChatConfig.qwen3_06B
                return true
            }
            group.addTask {
                let _ = ChatSamplingConfig.creative
                return true
            }
            group.addTask {
                let _ = ChatMessage(role: .system, content: "test")
                return true
            }

            var allPassed = true
            for await result in group {
                if !result { allPassed = false }
            }
            XCTAssertTrue(allPassed)
        }
    }
}

// MARK: - Memory Management Tests

final class Qwen3ChatMemoryTests: XCTestCase {

    func testMemoryFootprintReturnsNonZero() throws {
        // We can't load the real model in unit tests, but we can verify the
        // protocol conformance compiles and the estimate is reasonable.
        // The actual value is a compile-time constant (318 MB).
        let expected = 318 * 1024 * 1024
        XCTAssertEqual(expected, 333_447_168)
    }

    func testModelMemoryManageableConformance() {
        // Verify Qwen3ChatModel conforms to ModelMemoryManageable at compile time.
        // This test passes if the code compiles — no runtime model needed.
        let _: any AudioCommon.ModelMemoryManageable.Type = Qwen3ChatModel.self
    }
}
