import CoreML
import Foundation
import AudioCommon

/// On-device chat model using Qwen3-0.6B on CoreML.
///
/// Efficient on-device inference patterns:
/// - **Dual model**: Separate Prefill (batch) + Decode (single token) CoreML models
/// - **Prompt caching**: Snapshot system prompt KV state, restore per turn (~300ms saved)
/// - **Adaptive metrics**: Track tokens/sec for downstream buffering decisions
/// - **`@unchecked Sendable`**: Safe via ownership isolation (single-task use)
///
/// ```swift
/// let chat = try await Qwen3ChatModel.fromPretrained()
/// let response = try chat.generate(messages: [
///     ChatMessage(role: .system, content: "You are a friendly companion."),
///     ChatMessage(role: .user, content: "Hello!"),
/// ])
/// print(response)       // "Hi there! How can I help?"
/// print(chat.lastMetrics) // tokens/sec, prefill time, etc.
/// ```
///
/// - Important: Not safe for concurrent access. Use from a single task/thread.
///   The `@unchecked Sendable` conformance allows passing across isolation
///   boundaries but does not provide internal synchronization.
public final class Qwen3ChatModel: @unchecked Sendable {
    public static let defaultModelId = "aufklarer/Qwen3-0.6B-Chat-CoreML"
    public static let qwen35ModelId = "aufklarer/Qwen3.5-0.8B-Chat-CoreML"

    let config: Qwen3ChatConfig
    let tokenizer: ChatTokenizer
    let generator: CoreMLGenerator
    var conversationHistory: [ChatMessage] = []
    var systemPromptCached = false
    var _isLoaded = true

    /// Quantization variant to load.
    public enum Quantization: String, Sendable {
        case int4 = "Qwen3Chat-INT4"
        case int8 = "Qwen3Chat-INT8"
    }

    /// Metrics from the last generation (tokens/sec, prefill time, etc.).
    public var lastMetrics: (tokensPerSec: Double, prefillMs: Double, decodeMs: Double, msPerToken: Double) {
        let m = generator.metrics
        return (m.tokensPerSecond, m.prefillTimeMs, m.decodeTimeMs, m.msPerToken)
    }

    private init(config: Qwen3ChatConfig, tokenizer: ChatTokenizer, generator: CoreMLGenerator) {
        self.config = config
        self.tokenizer = tokenizer
        self.generator = generator
        self.generator.resetCache()
    }

    // MARK: - Factory

    /// Load a pre-trained Qwen3 chat model from HuggingFace.
    ///
    /// Downloads the CoreML model and tokenizer on first use (~300MB for INT4).
    /// If separate Prefill and Decode models exist, loads both for optimal performance.
    ///
    /// - Parameters:
    ///   - modelId: HuggingFace model ID
    ///   - computeUnits: CoreML compute units (default: .all for Neural Engine + CPU + GPU)
    ///   - progressHandler: Optional callback for download progress
    public static func fromPretrained(
        modelId: String = defaultModelId,
        quantization: Quantization = .int4,
        computeUnits: MLComputeUnits = .all,
        progressHandler: ((Double, String) -> Void)? = nil
    ) async throws -> Qwen3ChatModel {
        let cacheDir = try HuggingFaceDownloader.getCacheDirectory(for: modelId)

        // Download model files
        progressHandler?(0.05, "Downloading model...")
        try await HuggingFaceDownloader.downloadWeights(
            modelId: modelId,
            to: cacheDir,
            additionalFiles: [
                "chat_config.json",
                "vocab.json",
                "merges.txt",
                "tokenizer_config.json",
                "*.mlmodelc/**",
                "*.mlpackage/**",
            ],
            progressHandler: { progress in
                progressHandler?(progress * 0.7, "Downloading...")
            }
        )

        // Load config
        progressHandler?(0.7, "Loading config...")
        let configURL = cacheDir.appendingPathComponent("chat_config.json")
        let config: Qwen3ChatConfig
        if FileManager.default.fileExists(atPath: configURL.path) {
            config = try Qwen3ChatConfig.load(from: configURL)
        } else {
            config = .qwen3_06B
        }

        // Load tokenizer
        progressHandler?(0.75, "Loading tokenizer...")
        let tokenizer = ChatTokenizer()
        try tokenizer.load(from: cacheDir)

        // Load CoreML models (try separate prefill + decode first)
        progressHandler?(0.8, "Loading CoreML model...")
        let mlConfig = MLModelConfiguration()
        mlConfig.computeUnits = computeUnits

        let generator: CoreMLGenerator

        let prefillURL = findModel(named: "Qwen3ChatPrefill", in: cacheDir)
        let decodeURL = findModel(named: "Qwen3ChatDecode", in: cacheDir)

        if let prefillURL, let decodeURL {
            // Dual model: separate prefill + decode (optimal)
            let prefillModel = try MLModel(contentsOf: prefillURL, configuration: mlConfig)
            let decodeModel = try MLModel(contentsOf: decodeURL, configuration: mlConfig)
            generator = CoreMLGenerator(
                prefillModel: prefillModel,
                decodeModel: decodeModel,
                config: config
            )
        } else if let singleURL = findModel(named: quantization.rawValue, in: cacheDir)
                    ?? findModel(named: "Qwen3Chat", in: cacheDir)
                    ?? findAnyModel(in: cacheDir) {
            // Single model fallback
            let model = try MLModel(contentsOf: singleURL, configuration: mlConfig)
            generator = CoreMLGenerator(model: model, config: config)
        } else {
            throw ChatModelError.modelNotFound(cacheDir)
        }

        progressHandler?(1.0, "Ready")
        return Qwen3ChatModel(config: config, tokenizer: tokenizer, generator: generator)
    }

    /// Load a chat model from a local directory (no HuggingFace download).
    ///
    /// The directory must contain: `chat_config.json`, `vocab.json`, `merges.txt`,
    /// `tokenizer_config.json`, and a `.mlpackage` or `.mlmodelc` model.
    public static func fromLocal(
        directory: URL,
        computeUnits: MLComputeUnits = .all
    ) async throws -> Qwen3ChatModel {
        let configURL = directory.appendingPathComponent("chat_config.json")
        let config: Qwen3ChatConfig
        if FileManager.default.fileExists(atPath: configURL.path) {
            config = try Qwen3ChatConfig.load(from: configURL)
        } else {
            config = .qwen3_06B
        }

        let tokenizer = ChatTokenizer()
        try tokenizer.load(from: directory)

        let mlConfig = MLModelConfiguration()
        mlConfig.computeUnits = computeUnits

        let generator: CoreMLGenerator
        if let singleURL = findModel(named: "Qwen3Chat", in: directory)
                    ?? findAnyModel(in: directory) {
            let model = try MLModel(contentsOf: singleURL, configuration: mlConfig)
            generator = CoreMLGenerator(model: model, config: config)
        } else {
            throw ChatModelError.modelNotFound(directory)
        }

        return Qwen3ChatModel(config: config, tokenizer: tokenizer, generator: generator)
    }

    private static func findModel(named name: String, in directory: URL) -> URL? {
        let fm = FileManager.default
        let compiledURL = directory.appendingPathComponent("\(name).mlmodelc")
        let packageURL = directory.appendingPathComponent("\(name).mlpackage")

        // If compiled model exists, check if it's still up to date
        if fm.fileExists(atPath: compiledURL.path) {
            if fm.fileExists(atPath: packageURL.path),
               isNewer(packageURL, than: compiledURL) {
                // Package was updated (e.g., HuggingFace re-download) — recompile
                try? fm.removeItem(at: compiledURL)
                return compileIfNeeded(packageURL, compiledAs: compiledURL)
            }
            return compiledURL
        }

        // Compile .mlpackage on first use
        if fm.fileExists(atPath: packageURL.path) {
            return compileIfNeeded(packageURL, compiledAs: compiledURL)
        }

        return nil
    }

    private static func isNewer(_ a: URL, than b: URL) -> Bool {
        let fm = FileManager.default
        guard let aDate = try? fm.attributesOfItem(atPath: a.path)[.modificationDate] as? Date,
              let bDate = try? fm.attributesOfItem(atPath: b.path)[.modificationDate] as? Date else {
            return false
        }
        return aDate > bDate
    }

    private static func compileIfNeeded(_ packageURL: URL, compiledAs targetURL: URL) -> URL? {
        // Compile .mlpackage → .mlmodelc
        guard let compiledURL = try? MLModel.compileModel(at: packageURL) else { return nil }

        // Move compiled model next to the package for caching
        try? FileManager.default.moveItem(at: compiledURL, to: targetURL)
        if FileManager.default.fileExists(atPath: targetURL.path) {
            return targetURL
        }
        // Fallback: use the temp compiled location
        return compiledURL
    }

    private static func findAnyModel(in directory: URL) -> URL? {
        guard let contents = try? FileManager.default.contentsOfDirectory(
            at: directory, includingPropertiesForKeys: nil
        ) else { return nil }

        // Prefer pre-compiled
        if let compiled = contents.first(where: { $0.pathExtension == "mlmodelc" }) {
            return compiled
        }
        // Compile any .mlpackage found
        if let pkg = contents.first(where: { $0.pathExtension == "mlpackage" }) {
            let target = pkg.deletingPathExtension().appendingPathExtension("mlmodelc")
            return compileIfNeeded(pkg, compiledAs: target)
        }
        return nil
    }

    // MARK: - Generation

    /// Generate a response given a list of messages.
    ///
    /// Uses prefill for prompt tokens, then decode for autoregressive generation.
    /// Metrics are available via `lastMetrics` after completion.
    public func generate(
        messages: [ChatMessage],
        sampling: ChatSamplingConfig = .default
    ) throws -> String {
        generator.resetCache()
        generator.resetMetrics()

        let promptTokens = ChatTemplate.encode(
            messages: messages,
            tokenizer: tokenizer
        )

        // Prefill: process all prompt tokens at once
        var logits = try generator.prefill(tokenIds: promptTokens)

        // Autoregressive decode
        var generatedTokens: [Int] = []
        var responseTokenCount = 0
        var inThinking = false
        let thinkBudget = 100  // extra tokens for <think>...</think>

        for _ in 0..<(sampling.maxTokens + thinkBudget) {
            let nextToken = generator.sample(
                logits: logits,
                config: sampling,
                previousTokens: promptTokens + generatedTokens
            )

            if nextToken == config.eosTokenId { break }
            if nextToken == ChatTemplate.imEndId { break }

            generatedTokens.append(nextToken)

            // Track thinking vs response tokens
            if nextToken == ChatTemplate.thinkStartId { inThinking = true }
            else if nextToken == ChatTemplate.thinkEndId { inThinking = false }
            else if !inThinking { responseTokenCount += 1 }

            // Cap thinking: if thinking exceeds budget, force </think>
            if inThinking && generatedTokens.count > thinkBudget {
                generatedTokens.append(ChatTemplate.thinkEndId)
                logits = try generator.decode(tokenId: ChatTemplate.thinkEndId)
                inThinking = false
                continue
            }

            if responseTokenCount >= sampling.maxTokens { break }

            logits = try generator.decode(tokenId: nextToken)
        }

        let responseTokens = ChatTemplate.stripThinking(from: generatedTokens)
        return tokenizer.decode(responseTokens)
    }

    /// Generate a streaming response given a list of messages.
    ///
    /// Yields partial text as tokens are generated. Metrics updated live.
    public func generateStream(
        messages: [ChatMessage],
        sampling: ChatSamplingConfig = .default
    ) -> AsyncThrowingStream<String, Error> {
        AsyncThrowingStream { continuation in
            Task {
                do {
                    self.generator.resetCache()
                    self.generator.resetMetrics()

                    let promptTokens = ChatTemplate.encode(
                        messages: messages,
                        tokenizer: self.tokenizer
                    )

                    // Prefill
                    var logits = try self.generator.prefill(tokenIds: promptTokens)
                    var generatedTokens: [Int] = []

                    // Decode loop — skip thinking block tokens
                    var inThinking = false
                    for _ in 0..<sampling.maxTokens {
                        let nextToken = self.generator.sample(
                            logits: logits,
                            config: sampling,
                            previousTokens: promptTokens + generatedTokens
                        )

                        if nextToken == self.config.eosTokenId { break }
                        if nextToken == ChatTemplate.imEndId { break }

                        generatedTokens.append(nextToken)

                        if nextToken == ChatTemplate.thinkStartId {
                            inThinking = true
                        } else if nextToken == ChatTemplate.thinkEndId {
                            inThinking = false
                        } else if !inThinking,
                                  let text = self.tokenizer.decodeToken(nextToken),
                                  !self.tokenizer.isSpecialToken(nextToken) {
                            continuation.yield(text)
                        }

                        logits = try self.generator.decode(tokenId: nextToken)
                    }

                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    // MARK: - Conversation with Prompt Caching

    /// Chat with prompt caching.
    ///
    /// On the first call with a system prompt, prefills and caches the KV state.
    /// Subsequent calls restore from cache instead of re-prefilling (~300ms saved).
    public func chat(
        _ userMessage: String,
        systemPrompt: String? = nil,
        sampling: ChatSamplingConfig = .default
    ) throws -> String {
        // Cache system prompt on first turn
        if let system = systemPrompt, !systemPromptCached {
            generator.resetCache()
            generator.resetMetrics()

            let systemTokens = ChatTemplate.encode(
                messages: [ChatMessage(role: .system, content: system)],
                tokenizer: tokenizer,
                addGenerationPrompt: false
            )
            _ = try generator.prefill(tokenIds: systemTokens)
            generator.snapshotPromptCache()
            systemPromptCached = true
        } else if systemPromptCached {
            // Restore cached system prompt KV state
            generator.restorePromptCache()
            generator.resetMetrics()
        } else {
            generator.resetCache()
            generator.resetMetrics()
        }

        // Build turn tokens: history + new user message + generation prompt
        var turnMessages = conversationHistory
        turnMessages.append(ChatMessage(role: .user, content: userMessage))

        let turnTokens = ChatTemplate.encode(
            messages: turnMessages,
            tokenizer: tokenizer
        )

        // Prefill turn tokens
        var logits = try generator.prefill(tokenIds: turnTokens)

        // Decode response
        var generatedTokens: [Int] = []
        for _ in 0..<sampling.maxTokens {
            let nextToken = generator.sample(
                logits: logits,
                config: sampling,
                previousTokens: generatedTokens
            )

            if nextToken == config.eosTokenId { break }
            if nextToken == ChatTemplate.imEndId { break }

            generatedTokens.append(nextToken)
            logits = try generator.decode(tokenId: nextToken)
        }

        let responseTokens = ChatTemplate.stripThinking(from: generatedTokens)
        let response = tokenizer.decode(responseTokens)

        // Update history
        conversationHistory.append(ChatMessage(role: .user, content: userMessage))
        conversationHistory.append(ChatMessage(role: .assistant, content: response))

        return response
    }

    /// Stream a chat response with prompt caching.
    public func chatStream(
        _ userMessage: String,
        systemPrompt: String? = nil,
        sampling: ChatSamplingConfig = .default
    ) -> AsyncThrowingStream<String, Error> {
        AsyncThrowingStream { continuation in
            Task {
                var fullResponse = ""
                do {
                    // Cache system prompt on first turn
                    if let system = systemPrompt, !self.systemPromptCached {
                        self.generator.resetCache()
                        self.generator.resetMetrics()
                        let systemTokens = ChatTemplate.encode(
                            messages: [ChatMessage(role: .system, content: system)],
                            tokenizer: self.tokenizer,
                            addGenerationPrompt: false
                        )
                        _ = try self.generator.prefill(tokenIds: systemTokens)
                        self.generator.snapshotPromptCache()
                        self.systemPromptCached = true
                    } else if self.systemPromptCached {
                        self.generator.restorePromptCache()
                        self.generator.resetMetrics()
                    } else {
                        self.generator.resetCache()
                        self.generator.resetMetrics()
                    }

                    var turnMessages = self.conversationHistory
                    turnMessages.append(ChatMessage(role: .user, content: userMessage))

                    let turnTokens = ChatTemplate.encode(
                        messages: turnMessages,
                        tokenizer: self.tokenizer
                    )

                    var logits = try self.generator.prefill(tokenIds: turnTokens)
                    var generatedTokens: [Int] = []

                    var inThinking = false
                    for _ in 0..<sampling.maxTokens {
                        let nextToken = self.generator.sample(
                            logits: logits,
                            config: sampling,
                            previousTokens: generatedTokens
                        )

                        if nextToken == self.config.eosTokenId { break }
                        if nextToken == ChatTemplate.imEndId { break }

                        generatedTokens.append(nextToken)

                        if nextToken == ChatTemplate.thinkStartId {
                            inThinking = true
                        } else if nextToken == ChatTemplate.thinkEndId {
                            inThinking = false
                        } else if !inThinking,
                                  let text = self.tokenizer.decodeToken(nextToken),
                                  !self.tokenizer.isSpecialToken(nextToken) {
                            fullResponse += text
                            continuation.yield(text)
                        }

                        logits = try self.generator.decode(tokenId: nextToken)
                    }

                    self.conversationHistory.append(
                        ChatMessage(role: .user, content: userMessage)
                    )
                    self.conversationHistory.append(
                        ChatMessage(role: .assistant, content: fullResponse)
                    )
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    /// Clear conversation history and prompt cache.
    public func resetConversation() {
        conversationHistory = []
        systemPromptCached = false
        generator.clearPromptCache()
    }
}
