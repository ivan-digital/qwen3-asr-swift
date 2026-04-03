import Foundation
import MLXCommon
import MLX
import MLXNN
import AudioCommon
import os.log

private let log = OSLog(subsystem: "com.soniqo.qwen3chat", category: "MLX")

// MARK: - MLX Generator for Qwen3.5 Chat

/// MLX-based text generator for Qwen3.5-0.8B hybrid model.
///
/// Uses MLX for GPU inference
/// on Apple Silicon GPUs. The hybrid DeltaNet + GatedAttention architecture
/// requires managing two types of state:
///   1. DeltaNet recurrent states (carried across all tokens, O(1) per layer)
///   2. GatedAttention KV caches (grow with sequence length, only 6 layers)
///
/// Usage:
/// ```swift
/// let model = try await Qwen35MLXChat.fromPretrained()
/// let response = try model.generate(messages: [
///     ChatMessage(role: .user, content: "Hello!")
/// ])
/// ```
public final class Qwen35MLXChat: @unchecked Sendable {
    public static let defaultModelId = "aufklarer/Qwen3.5-0.8B-Chat-MLX"

    public let config: Qwen3ChatConfig
    public let tokenizer: ChatTokenizer
    let model: Qwen35MLXModel
    var state: Qwen35MLXModel.InferenceState
    var _isLoaded = true

    // MARK: - Metrics

    /// Generation metrics for performance tracking.
    public struct Metrics {
        public var prefillTimeMs: Double = 0
        public var prefillTokens: Int = 0
        public var decodeTimeMs: Double = 0
        public var decodeTokens: Int = 0

        public var tokensPerSecond: Double {
            guard decodeTimeMs > 0 else { return 0 }
            return Double(decodeTokens) / (decodeTimeMs / 1000.0)
        }

        public var msPerToken: Double {
            guard decodeTokens > 0 else { return 0 }
            return decodeTimeMs / Double(decodeTokens)
        }

        public var prefillTokensPerSecond: Double {
            guard prefillTimeMs > 0 else { return 0 }
            return Double(prefillTokens) / (prefillTimeMs / 1000.0)
        }
    }

    private(set) var metrics = Metrics()

    /// Latest generation metrics.
    public var lastMetrics: (tokensPerSec: Double, prefillMs: Double, decodeMs: Double, msPerToken: Double) {
        (metrics.tokensPerSecond, metrics.prefillTimeMs, metrics.decodeTimeMs, metrics.msPerToken)
    }

    // MARK: - Init

    private init(config: Qwen3ChatConfig, tokenizer: ChatTokenizer, model: Qwen35MLXModel) {
        self.config = config
        self.tokenizer = tokenizer
        self.model = model
        self.state = .initial(config: config)
    }

    // MARK: - Factory

    /// Quantization variant.
    public enum Quantization: String {
        case int4
        case int8
    }

    /// Load a pre-trained Qwen3.5 chat model from HuggingFace.
    ///
    /// Downloads quantized safetensors and tokenizer on first use.
    /// Model is loaded into MLX for GPU inference on Apple Silicon.
    ///
    /// - Parameters:
    ///   - modelId: HuggingFace model ID (repo with int4/ and int8/ subdirs)
    ///   - quantization: INT4 (404 MB) or INT8 (763 MB)
    ///   - progressHandler: Optional callback for download/load progress
    public static func fromPretrained(
        modelId: String = defaultModelId,
        quantization: Quantization = .int4,
        progressHandler: ((Double, String) -> Void)? = nil
    ) async throws -> Qwen35MLXChat {
        let cacheDir = try HuggingFaceDownloader.getCacheDirectory(for: modelId)
        let variant = quantization.rawValue

        // Download model files from variant subdirectory (int4/ or int8/)
        progressHandler?(0.05, "Downloading \(variant) model...")
        try await HuggingFaceDownloader.downloadWeights(
            modelId: modelId,
            to: cacheDir,
            additionalFiles: [
                "\(variant)/model.safetensors",
                "\(variant)/config.json",
                "\(variant)/tokenizer.json",
                "\(variant)/tokenizer_config.json",
            ],
            progressHandler: { progress in
                progressHandler?(progress * 0.5, "Downloading...")
            }
        )

        // Variant files are in a subdirectory
        let variantDir = cacheDir.appendingPathComponent(variant)

        // Load config
        progressHandler?(0.5, "Loading config...")
        let config: Qwen3ChatConfig
        let configURL = variantDir.appendingPathComponent("config.json")
        if FileManager.default.fileExists(atPath: configURL.path) {
            config = try Qwen3ChatConfig.load(from: configURL)
        } else {
            config = .qwen35_08B
        }

        // Load tokenizer
        progressHandler?(0.55, "Loading tokenizer...")
        let tokenizer = ChatTokenizer()
        try tokenizer.load(from: variantDir)

        // Create model
        progressHandler?(0.6, "Creating model...")
        let model = Qwen35MLXModel(config: config)

        // Load weights
        progressHandler?(0.65, "Loading weights...")
        try Qwen35WeightLoader.loadWeights(
            into: model, from: variantDir,
            progressHandler: { pct, msg in
                progressHandler?(0.65 + pct * 0.3, msg)
            })

        progressHandler?(1.0, "Ready")
        return Qwen35MLXChat(config: config, tokenizer: tokenizer, model: model)
    }

    /// Load from a local directory (no HuggingFace download).
    public static func fromLocal(
        directory: URL,
        progressHandler: ((Double, String) -> Void)? = nil
    ) async throws -> Qwen35MLXChat {
        let config: Qwen3ChatConfig
        let configURL = directory.appendingPathComponent("config.json")
        if FileManager.default.fileExists(atPath: configURL.path) {
            config = try Qwen3ChatConfig.load(from: configURL)
        } else {
            config = .qwen35_08B
        }

        let tokenizer = ChatTokenizer()
        try tokenizer.load(from: directory)

        let model = Qwen35MLXModel(config: config)
        try Qwen35WeightLoader.loadWeights(
            into: model, from: directory,
            progressHandler: progressHandler)

        return Qwen35MLXChat(config: config, tokenizer: tokenizer, model: model)
    }

    // MARK: - State Management

    /// Reset all inference state for a new conversation.
    public func resetState() {
        state = .initial(config: config)
        metrics = Metrics()
    }

    // MARK: - Generation

    /// Generate a response from chat messages.
    ///
    /// Encodes the messages using the chat template, prefills the prompt,
    /// then generates tokens autoregressively until EOS or max tokens.
    public func generate(
        messages: [ChatMessage],
        sampling: ChatSamplingConfig = .default
    ) throws -> String {
        resetState()

        let promptTokens = ChatTemplate.encode(
            messages: messages,
            tokenizer: tokenizer,
            config: config,
            enableThinking: false)

        // Prefill
        let prefillStart = CFAbsoluteTimeGetCurrent()
        let promptArray = MLXArray(promptTokens.map { Int32($0) })
            .expandedDimensions(axis: 0)
        let (prefillLogits, prefillState) = model.forward(inputIds: promptArray, state: state)
        eval(prefillLogits)
        state = prefillState

        let prefillMs = (CFAbsoluteTimeGetCurrent() - prefillStart) * 1000
        metrics.prefillTimeMs = prefillMs
        metrics.prefillTokens = promptTokens.count

        // Extract last-position logits and sample first token
        var logits = extractLastPositionLogits(prefillLogits)
        var generatedTokens: [Int] = []
        var inThinking = false
        let thinkBudget = 100

        // Decode loop
        let decodeStart = CFAbsoluteTimeGetCurrent()

        for _ in 0..<(sampling.maxTokens + thinkBudget) {
            let nextToken = ChatSampler.sample(
                logits: logits,
                config: sampling,
                previousTokens: promptTokens + generatedTokens)

            if nextToken == config.eosTokenId { break }
            if nextToken == ChatTemplate.imEndId { break }

            generatedTokens.append(nextToken)

            // Thinking token tracking (handle both Qwen3 and Qwen3.5 token IDs)
            if nextToken == ChatTemplate.thinkStartId {
                inThinking = true
            } else if nextToken == ChatTemplate.thinkEndId {
                inThinking = false
            }

            if inThinking && generatedTokens.count > thinkBudget {
                let thinkEnd = ChatTemplate.thinkEndId
                generatedTokens.append(thinkEnd)
                let tokenArr = MLXArray([Int32(thinkEnd)])
                    .expandedDimensions(axis: 0)
                let (stepLogits, newState) = model.forward(inputIds: tokenArr, state: state)
                eval(stepLogits)
                state = newState
                logits = extractLastPositionLogits(stepLogits)
                inThinking = false
                continue
            }

            let thinkTokens: Set<Int> = [
                ChatTemplate.thinkStartId, ChatTemplate.thinkEndId
            ]
            let responseCount = generatedTokens.filter { !thinkTokens.contains($0) }.count
            if !inThinking && responseCount >= sampling.maxTokens { break }

            // Decode one step
            let tokenArr = MLXArray([Int32(nextToken)]).expandedDimensions(axis: 0)
            let (stepLogits, newState) = model.forward(inputIds: tokenArr, state: state)
            eval(stepLogits)
            state = newState
            logits = extractLastPositionLogits(stepLogits)
        }

        let decodeMs = (CFAbsoluteTimeGetCurrent() - decodeStart) * 1000
        metrics.decodeTimeMs = decodeMs
        metrics.decodeTokens = generatedTokens.count

        var memInfo = mach_task_basic_info()
        var memCount = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4
        withUnsafeMutablePointer(to: &memInfo) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(memCount)) {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &memCount)
            }
        }
        let memMB = Double(memInfo.resident_size) / 1024 / 1024
        let tps = decodeMs > 0 ? Double(generatedTokens.count) / (decodeMs / 1000.0) : 0
        os_log(.info, log: log,
               "Generate done: prefill=%.0fms (%d tokens), decode=%.0fms (%d tokens, %.1f tok/s), memory=%.0f MB",
               metrics.prefillTimeMs, promptTokens.count, decodeMs, generatedTokens.count, tps, memMB)

        let responseTokens = ChatTemplate.stripThinking(from: generatedTokens)
        return tokenizer.decode(responseTokens)
    }

    /// Generate a streaming response.
    public func generateStream(
        messages: [ChatMessage],
        sampling: ChatSamplingConfig = .default
    ) -> AsyncThrowingStream<String, Error> {
        AsyncThrowingStream { continuation in
            Task {
                do {
                    self.resetState()

                    let promptTokens = ChatTemplate.encode(
                        messages: messages,
                        tokenizer: self.tokenizer,
                        config: self.config,
                        enableThinking: false)

                    let promptArray = MLXArray(promptTokens.map { Int32($0) })
                        .expandedDimensions(axis: 0)
                    let (prefillLogits, prefillState) = self.model.forward(
                        inputIds: promptArray, state: self.state)
                    eval(prefillLogits)
                    self.state = prefillState

                    var logits = self.extractLastPositionLogits(prefillLogits)
                    var generatedTokens: [Int] = []
                    var inThinking = false

                    for _ in 0..<sampling.maxTokens {
                        let nextToken = ChatSampler.sample(
                            logits: logits,
                            config: sampling,
                            previousTokens: promptTokens + generatedTokens)

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

                        let tokenArr = MLXArray([Int32(nextToken)])
                            .expandedDimensions(axis: 0)
                        let (stepLogits, newState) = self.model.forward(
                            inputIds: tokenArr, state: self.state)
                        eval(stepLogits)
                        self.state = newState
                        logits = self.extractLastPositionLogits(stepLogits)
                    }

                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    // MARK: - Helpers

    /// Extract logits for the last sequence position as a Float array.
    private func extractLastPositionLogits(_ logits: MLXArray) -> [Float] {
        let t = logits.dim(1)
        let lastPos = logits[0, t - 1].asType(.float32)  // [vocabSize]
        eval(lastPos)
        // Bulk extract all floats at once — do NOT use per-element .item() (248K syncs)
        let all: [Float] = lastPos.asArray(Float.self)
        return Array(all.prefix(config.vocabSize))
    }
}

// MARK: - Memory Management

extension Qwen35MLXChat: ModelMemoryManageable {
    public var isLoaded: Bool { _isLoaded }

    public func unload() {
        guard _isLoaded else { return }
        model.clearParameters()
        state = .initial(config: config)
        _isLoaded = false
    }

    public var memoryFootprint: Int {
        guard _isLoaded else { return 0 }
        return model.parameterMemoryBytes()
    }
}
