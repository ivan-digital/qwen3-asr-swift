import CoreML
import Foundation
import os
import AudioCommon

private let log = Logger(subsystem: "com.qwen3speech", category: "Chat")

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
public final class Qwen3ChatModel: @unchecked Sendable {
    public static let defaultModelId = "aufklarer/Qwen3-0.6B-Chat-CoreML"

    private let config: Qwen3ChatConfig
    private let tokenizer: ChatTokenizer
    private let generator: CoreMLGenerator
    private var conversationHistory: [ChatMessage] = []
    private var systemPromptCached = false

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
            localCheckFiles: [
                "config.json",
                "chat_config.json",
                "vocab.json",
            ],
            progressHandler: { progress in
                progressHandler?(progress * 0.7, "")
            },
            statusHandler: { status in
                progressHandler?(0.05, status)
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
        } else if let singleURL = findModel(named: "Qwen3Chat", in: cacheDir)
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
        for _ in 0..<sampling.maxTokens {
            let nextToken = generator.sample(
                logits: logits,
                config: sampling,
                previousTokens: promptTokens + generatedTokens
            )

            if nextToken == config.eosTokenId { break }
            if nextToken == ChatTemplate.imEndId { break }

            generatedTokens.append(nextToken)
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
            let task = Task {
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
                        if Task.isCancelled { break }
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
            continuation.onTermination = { @Sendable _ in
                task.cancel()
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

        // Prefill turn tokens (optionally with no-think prefix)
        var logits: [Float]
        if sampling.disableThinking {
            let noThinkTokens = [
                ChatTemplate.thinkStartId,
                ChatTemplate.newlineId,
                ChatTemplate.thinkEndId,
                ChatTemplate.newlineId,
            ]
            logits = try generator.prefill(tokenIds: turnTokens + noThinkTokens)
        } else {
            logits = try generator.prefill(tokenIds: turnTokens)
        }

        // Decode response
        var generatedTokens: [Int] = []
        var inThinking = false
        var thinkingTokenCount = 0
        for _ in 0..<sampling.maxTokens {
            let nextToken = generator.sample(
                logits: logits,
                config: sampling,
                previousTokens: generatedTokens
            )

            if nextToken == config.eosTokenId { break }
            if nextToken == ChatTemplate.imEndId { break }

            generatedTokens.append(nextToken)

            if nextToken == ChatTemplate.thinkStartId {
                inThinking = true
                thinkingTokenCount = 0
            } else if nextToken == ChatTemplate.thinkEndId {
                inThinking = false
            } else if inThinking {
                thinkingTokenCount += 1
                if sampling.maxThinkingTokens > 0 &&
                   thinkingTokenCount >= sampling.maxThinkingTokens {
                    generatedTokens.append(ChatTemplate.thinkEndId)
                    logits = try generator.decode(tokenId: ChatTemplate.thinkEndId)
                    generatedTokens.append(ChatTemplate.newlineId)
                    logits = try generator.decode(tokenId: ChatTemplate.newlineId)
                    inThinking = false
                    continue
                }
            }

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
            let task = Task {
                var fullResponse = ""
                do {
                    let t0 = CFAbsoluteTimeGetCurrent()

                    // Cache system prompt on first turn
                    if let system = systemPrompt, !self.systemPromptCached {
                        self.generator.resetCache()
                        self.generator.resetMetrics()
                        let systemTokens = ChatTemplate.encode(
                            messages: [ChatMessage(role: .system, content: system)],
                            tokenizer: self.tokenizer,
                            addGenerationPrompt: false
                        )
                        log.warning("chatStream: prefilling system prompt (\(systemTokens.count) tokens)")
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

                    var logits: [Float]

                    if sampling.disableThinking {
                        let noThinkTokens = [
                            ChatTemplate.thinkStartId,
                            ChatTemplate.newlineId,
                            ChatTemplate.thinkEndId,
                            ChatTemplate.newlineId,
                        ]
                        log.warning("chatStream: prefilling \(turnTokens.count) turn tokens + no-think prefix")
                        logits = try self.generator.prefill(
                            tokenIds: turnTokens + noThinkTokens)
                    } else {
                        log.warning("chatStream: prefilling \(turnTokens.count) turn tokens (maxThink=\(sampling.maxThinkingTokens))")
                        logits = try self.generator.prefill(tokenIds: turnTokens)
                    }

                    let tPrefill = CFAbsoluteTimeGetCurrent()
                    log.warning("chatStream: prefill done in \(String(format: "%.0f", (tPrefill - t0) * 1000))ms")

                    var generatedTokens: [Int] = []
                    var totalThinkingTokens = 0
                    var totalResponseTokens = 0
                    var thinkingCapHit = false

                    var inThinking = false
                    var thinkingTokenCount = 0
                    var skipNextThinkEnd = false  // After cap injection, skip echo </think>
                    var emptyResponseTokens = 0   // Track consecutive non-response tokens after thinking
                    for i in 0..<sampling.maxTokens {
                        // Check cancellation so we stop when consumer disconnects
                        if Task.isCancelled {
                            log.warning("chatStream: cancelled at token \(i)")
                            break
                        }
                        let nextToken = self.generator.sample(
                            logits: logits,
                            config: sampling,
                            previousTokens: generatedTokens
                        )

                        let tokenText = self.tokenizer.decodeToken(nextToken) ?? "?"
                        if i < 5 || nextToken == self.config.eosTokenId || nextToken == ChatTemplate.imEndId ||
                           nextToken == ChatTemplate.thinkStartId || nextToken == ChatTemplate.thinkEndId {
                            log.warning("chatStream: token[\(i)] id=\(nextToken) think=\(inThinking) text='\(tokenText)'")
                        }

                        if nextToken == self.config.eosTokenId {
                            log.warning("chatStream: EOS at token \(i)")
                            break
                        }
                        if nextToken == ChatTemplate.imEndId {
                            log.warning("chatStream: imEnd at token \(i)")
                            break
                        }

                        generatedTokens.append(nextToken)

                        if nextToken == ChatTemplate.thinkStartId {
                            inThinking = true
                            thinkingTokenCount = 0
                            log.warning("chatStream: <think> started")
                        } else if nextToken == ChatTemplate.thinkEndId {
                            if skipNextThinkEnd {
                                // Model echoed </think> after our injection — skip it
                                skipNextThinkEnd = false
                                log.warning("chatStream: skipped echo </think>")
                            } else {
                                inThinking = false
                                log.warning("chatStream: </think> ended (natural, \(thinkingTokenCount) tokens)")
                            }
                        } else if inThinking {
                            thinkingTokenCount += 1
                            totalThinkingTokens += 1
                            // Force-end thinking if cap reached
                            if sampling.maxThinkingTokens > 0 &&
                               thinkingTokenCount >= sampling.maxThinkingTokens {
                                log.warning("chatStream: THINKING CAP HIT at \(thinkingTokenCount) tokens, injecting </think>")
                                thinkingCapHit = true
                                skipNextThinkEnd = true
                                generatedTokens.append(ChatTemplate.thinkEndId)
                                logits = try self.generator.decode(
                                    tokenId: ChatTemplate.thinkEndId)
                                generatedTokens.append(ChatTemplate.newlineId)
                                logits = try self.generator.decode(
                                    tokenId: ChatTemplate.newlineId)
                                inThinking = false
                                continue
                            }
                        } else if !inThinking,
                                  let text = self.tokenizer.decodeToken(nextToken),
                                  !self.tokenizer.isSpecialToken(nextToken) {
                            totalResponseTokens += 1
                            emptyResponseTokens = 0
                            fullResponse += text
                            continuation.yield(text)
                        } else if !inThinking {
                            // Non-yielded token after thinking ended (garbage/special)
                            emptyResponseTokens += 1
                            if emptyResponseTokens >= 10 {
                                log.warning("chatStream: aborting — \(emptyResponseTokens) non-response tokens after thinking")
                                break
                            }
                        }

                        logits = try self.generator.decode(tokenId: nextToken)
                    }

                    let tDone = CFAbsoluteTimeGetCurrent()
                    let totalMs = (tDone - t0) * 1000
                    let decodeMs = (tDone - tPrefill) * 1000
                    let totalTokens = generatedTokens.count
                    let tokPerSec = totalTokens > 0 ? Double(totalTokens) / (decodeMs / 1000) : 0
                    log.warning("chatStream: DONE total=\(String(format: "%.0f", totalMs))ms decode=\(String(format: "%.0f", decodeMs))ms tokens=\(totalTokens) (think=\(totalThinkingTokens) resp=\(totalResponseTokens)) \(String(format: "%.1f", tokPerSec)) tok/s capHit=\(thinkingCapHit) response='\(fullResponse)'")

                    // Only save to conversation history if we got a real response.
                    // Empty responses (e.g. from garbage thinking) poison the context.
                    let trimmedResponse = fullResponse.trimmingCharacters(in: .whitespacesAndNewlines)
                    if !trimmedResponse.isEmpty {
                        self.conversationHistory.append(
                            ChatMessage(role: .user, content: userMessage)
                        )
                        self.conversationHistory.append(
                            ChatMessage(role: .assistant, content: trimmedResponse)
                        )
                    } else {
                        log.warning("chatStream: skipping empty response in conversation history")
                    }
                    continuation.finish()
                } catch {
                    log.error("chatStream: ERROR \(error)")
                    continuation.finish(throwing: error)
                }
            }
            // When consumer stops iterating (e.g. cancellation), cancel the producer
            // so it stops calling decode() on the generator immediately.
            continuation.onTermination = { @Sendable _ in
                task.cancel()
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
