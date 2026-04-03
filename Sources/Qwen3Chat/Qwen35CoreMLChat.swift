import CoreML
import Foundation
import AudioCommon
import os.log

private let log = OSLog(subsystem: "com.soniqo.qwen3chat", category: "CoreML")

/// CoreML-based Qwen3.5-0.8B chat for iOS Neural Engine.
///
/// Uses two CoreML models:
/// - `embedding.mlmodelc` — token ID → embedding vector
/// - `decoder.mlmodelc` — autoregressive transformer with MLState
///
/// All DeltaNet recurrent states and GatedAttention KV caches are managed
/// by CoreML's MLState API — no manual cache tracking needed.
public final class Qwen35CoreMLChat: @unchecked Sendable {
    public static let defaultModelId = "aufklarer/Qwen3.5-0.8B-Chat-CoreML"

    private let embeddingModel: MLModel
    private let decoderModel: MLModel
    private var decoderState: MLState
    public let config: Qwen3ChatConfig
    public let tokenizer: ChatTokenizer
    private var position: Int = 0
    private let maxSeqLen: Int

    /// Quantization variant. Only INT8 available (INT4 removed — CoreML dequantization issues).
    public enum Quantization: String { case int8 }

    // MARK: - Metrics

    private var _prefillMs: Double = 0
    private var _decodeMs: Double = 0
    private var _decodeTokens: Int = 0

    public var lastMetrics: (tokensPerSec: Double, prefillMs: Double, decodeMs: Double, msPerToken: Double) {
        let tps = _decodeMs > 0 ? Double(_decodeTokens) / (_decodeMs / 1000.0) : 0
        let mpt = _decodeTokens > 0 ? _decodeMs / Double(_decodeTokens) : 0
        return (tps, _prefillMs, _decodeMs, mpt)
    }

    /// Current process memory in MB.
    private static var memoryMB: Double {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
            }
        }
        guard result == KERN_SUCCESS else { return 0 }
        return Double(info.resident_size) / 1024 / 1024
    }

    // MARK: - Init

    private init(embedding: MLModel, decoder: MLModel, state: MLState,
                 config: Qwen3ChatConfig, tokenizer: ChatTokenizer, maxSeqLen: Int) {
        self.embeddingModel = embedding
        self.decoderModel = decoder
        self.decoderState = state
        self.config = config
        self.tokenizer = tokenizer
        self.maxSeqLen = maxSeqLen
    }

    // MARK: - Factory

    /// Load from HuggingFace.
    public static func fromPretrained(
        modelId: String = defaultModelId,
        quantization: Quantization = .int8,
        computeUnits: MLComputeUnits = .cpuAndNeuralEngine,
        progressHandler: ((Double, String) -> Void)? = nil
    ) async throws -> Qwen35CoreMLChat {
        let cacheDir = try HuggingFaceDownloader.getCacheDirectory(for: modelId)
        let variant = quantization.rawValue

        progressHandler?(0.05, "Downloading \(variant) model...")
        try await HuggingFaceDownloader.downloadWeights(
            modelId: modelId,
            to: cacheDir,
            additionalFiles: [
                "\(variant)/*.json",
                "\(variant)/**/*.mlmodel",
                "\(variant)/**/*.bin",
                "\(variant)/**/Manifest.json",
            ],
            progressHandler: { p in progressHandler?(p * 0.5, "Downloading...") }
        )

        let variantDir = cacheDir.appendingPathComponent(variant)
        return try await fromLocal(directory: variantDir, computeUnits: computeUnits,
                                   progressHandler: progressHandler)
    }

    /// Load from a local directory.
    public static func fromLocal(
        directory: URL,
        computeUnits: MLComputeUnits = .cpuAndNeuralEngine,
        progressHandler: ((Double, String) -> Void)? = nil
    ) async throws -> Qwen35CoreMLChat {
        progressHandler?(0.5, "Loading config...")

        // Debug: list directory contents to diagnose missing file issues
        os_log(.info, log: log, "Loading from directory: %{public}@", directory.path)
        if let contents = try? FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil) {
            for item in contents {
                os_log(.info, log: log, "  %{public}@", item.lastPathComponent)
            }
        } else {
            os_log(.error, log: log, "Cannot list directory: %{public}@", directory.path)
        }

        // Use built-in config — chat_config.json from CoreML conversion has a different schema
        let config = Qwen3ChatConfig.qwen35_08B
        os_log(.info, log: log, "Using built-in Qwen3.5-0.8B config")

        progressHandler?(0.55, "Loading tokenizer...")
        os_log(.info, log: log, "Loading tokenizer from: %{public}@", directory.resolvingSymlinksInPath().path)
        let tokenizer = ChatTokenizer()
        try tokenizer.load(from: directory.resolvingSymlinksInPath())

        let memBefore = memoryMB
        os_log(.info, log: log, "Loading CoreML models, memory before: %.0f MB", memBefore)

        progressHandler?(0.6, "Compiling embedding...")
        let embModel: MLModel
        do {
            embModel = try await loadModel(named: "embedding", from: directory, computeUnits: computeUnits)
            os_log(.info, log: log, "Embedding loaded, memory: %.0f MB", memoryMB)
        } catch {
            os_log(.error, log: log, "Embedding load FAILED: %{public}@", error.localizedDescription)
            throw error
        }

        progressHandler?(0.75, "Compiling decoder...")
        let decModel: MLModel
        do {
            decModel = try await loadModel(named: "decoder", from: directory, computeUnits: computeUnits)
            os_log(.info, log: log, "Decoder loaded, memory: %.0f MB", memoryMB)
        } catch {
            os_log(.error, log: log, "Decoder load FAILED: %{public}@", error.localizedDescription)
            throw error
        }

        let state = decModel.makeState()
        let maxSeq = config.maxSeqLen

        os_log(.info, log: log, "Model ready, total memory: %.0f MB (delta: +%.0f MB)",
               memoryMB, memoryMB - memBefore)
        progressHandler?(1.0, "Ready")
        return Qwen35CoreMLChat(
            embedding: embModel, decoder: decModel, state: state,
            config: config, tokenizer: tokenizer, maxSeqLen: maxSeq)
    }

    // MARK: - Model Loading Helpers

    private static func loadModel(
        named name: String, from dir: URL, computeUnits: MLComputeUnits
    ) async throws -> MLModel {
        let compiledURL = dir.appendingPathComponent("\(name).mlmodelc")
        let packageURL = dir.appendingPathComponent("\(name).mlpackage")

        let modelURL: URL
        if FileManager.default.fileExists(atPath: compiledURL.path) {
            modelURL = compiledURL
        } else if FileManager.default.fileExists(atPath: packageURL.path) {
            // Resolve symlinks before compiling — MLModel.compileModel
            // can fail on Hub-style symlinked .mlpackage directories
            let resolved = packageURL.resolvingSymlinksInPath()
            os_log(.info, log: log, "Compiling %{public}@.mlpackage from %{public}@", name, resolved.path)
            modelURL = try await MLModel.compileModel(at: resolved)
        } else {
            os_log(.error, log: log, "Model not found: %{public}@.mlpackage in %{public}@", name, dir.path)
            throw ChatModelError.modelNotFound(dir)
        }

        let mlConfig = MLModelConfiguration()
        mlConfig.computeUnits = computeUnits
        return try await MLModel.load(contentsOf: modelURL, configuration: mlConfig)
    }

    // MARK: - Generation

    /// Reset state for a new conversation.
    public func resetState() {
        decoderState = decoderModel.makeState()
        position = 0
        _prefillMs = 0; _decodeMs = 0; _decodeTokens = 0
    }

    /// Generate a response from chat messages.
    public func generate(
        messages: [ChatMessage],
        sampling: ChatSamplingConfig = .default
    ) throws -> String {
        resetState()

        let promptTokens = ChatTemplate.encode(
            messages: messages, tokenizer: tokenizer,
            config: config, enableThinking: false)

        // Prefill: feed all prompt tokens one at a time
        let prefillStart = CFAbsoluteTimeGetCurrent()
        var lastLogits: [Float] = []

        for token in promptTokens {
            lastLogits = try forwardStep(tokenId: token)
        }
        _prefillMs = (CFAbsoluteTimeGetCurrent() - prefillStart) * 1000

        // Decode loop
        let decodeStart = CFAbsoluteTimeGetCurrent()
        var generatedTokens: [Int] = []

        for _ in 0..<sampling.maxTokens {
            let nextToken = ChatSampler.sample(
                logits: lastLogits, config: sampling,
                previousTokens: promptTokens + generatedTokens)

            if nextToken == config.eosTokenId { break }
            if nextToken == ChatTemplate.imEndId { break }

            generatedTokens.append(nextToken)
            if generatedTokens.count <= 10 {
                print("[CoreML] token[\(generatedTokens.count-1)]: \(nextToken) '\(tokenizer.decodeToken(nextToken) ?? "?")'")
            }
            lastLogits = try forwardStep(tokenId: nextToken)
        }

        _decodeMs = (CFAbsoluteTimeGetCurrent() - decodeStart) * 1000
        _decodeTokens = generatedTokens.count

        let tps = _decodeMs > 0 ? Double(_decodeTokens) / (_decodeMs / 1000.0) : 0
        os_log(.info, log: log,
               "Generate done: prefill=%.0fms (%d tokens), decode=%.0fms (%d tokens, %.1f tok/s), memory=%.0f MB",
               _prefillMs, promptTokens.count, _decodeMs, _decodeTokens, tps, Self.memoryMB)

        let responseTokens = ChatTemplate.stripThinking(from: generatedTokens)
        let responseText = tokenizer.decode(responseTokens)
        os_log(.info, log: log, "Response (%d tokens → %d after strip): '%{public}@'",
               generatedTokens.count, responseTokens.count, responseText)
        return responseText
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
                        messages: messages, tokenizer: self.tokenizer,
                        config: self.config, enableThinking: false)

                    var lastLogits: [Float] = []
                    for token in promptTokens {
                        lastLogits = try self.forwardStep(tokenId: token)
                    }

                    var generatedTokens: [Int] = []
                    var inThinking = false
                    let thinkBudget = 100
                    let thinkTokens: Set<Int> = [
                        ChatTemplate.thinkStartId, ChatTemplate.thinkEndId
                    ]

                    for _ in 0..<(sampling.maxTokens + thinkBudget) {
                        let nextToken = ChatSampler.sample(
                            logits: lastLogits, config: sampling,
                            previousTokens: promptTokens + generatedTokens)

                        if nextToken == self.config.eosTokenId { break }
                        if nextToken == ChatTemplate.imEndId { break }

                        generatedTokens.append(nextToken)

                        if nextToken == ChatTemplate.thinkStartId { inThinking = true }
                        else if nextToken == ChatTemplate.thinkEndId { inThinking = false }
                        else if !inThinking,
                                let text = self.tokenizer.decodeToken(nextToken),
                                !self.tokenizer.isSpecialToken(nextToken) {
                            continuation.yield(text)
                        }

                        // Force-end thinking if budget exceeded
                        if inThinking && generatedTokens.count > thinkBudget {
                            generatedTokens.append(ChatTemplate.thinkEndId)
                            lastLogits = try self.forwardStep(tokenId: ChatTemplate.thinkEndId)
                            inThinking = false
                            continue
                        }

                        // Only count non-thinking tokens against maxTokens
                        let responseCount = generatedTokens.filter { !thinkTokens.contains($0) }.count
                        if !inThinking && responseCount >= sampling.maxTokens { break }

                        lastLogits = try self.forwardStep(tokenId: nextToken)
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    // MARK: - Single Step

    /// Run one decoder step: token ID → logits.
    private func forwardStep(tokenId: Int) throws -> [Float] {
        // Embedding lookup
        let tokenInput = try MLMultiArray(shape: [1, 1], dataType: .int32)
        tokenInput[0] = NSNumber(value: Int32(tokenId))

        let embFeatures = try MLDictionaryFeatureProvider(dictionary: [
            "token_id": MLFeatureValue(multiArray: tokenInput)
        ])
        let embResult = try embeddingModel.prediction(from: embFeatures)
        guard let embedding = embResult.featureValue(for: "embedding")?.multiArrayValue else {
            throw ChatModelError.inferenceFailed("Embedding output missing")
        }

        // Build attention mask: 0 for positions ≤ current, -FLT_MAX for future
        let mask = try MLMultiArray(shape: [1, 1, 1, maxSeqLen as NSNumber], dataType: .float32)
        let maskPtr = mask.dataPointer.bindMemory(to: Float.self, capacity: maxSeqLen)
        for i in 0..<maxSeqLen {
            maskPtr[i] = i <= position ? 0 : -Float.greatestFiniteMagnitude
        }

        // Position
        let posArray = try MLMultiArray(shape: [1], dataType: .int32)
        posArray[0] = NSNumber(value: Int32(position))

        // Decoder forward
        let decoderInput = try MLDictionaryFeatureProvider(dictionary: [
            "input_embeds": MLFeatureValue(multiArray: embedding),
            "position": MLFeatureValue(multiArray: posArray),
            "attention_mask": MLFeatureValue(multiArray: mask),
        ])

        let result = try decoderModel.prediction(
            from: decoderInput, using: decoderState)

        position += 1

        // Extract logits
        guard let logitsArray = result.featureValue(for: "logits")?.multiArrayValue else {
            throw ChatModelError.inferenceFailed("Decoder output missing")
        }

        let vocabSize = config.vocabSize

        // Debug: log dtype and shape on first few calls
        if position <= 3 {
            print("[CoreML] pos=\(position) logits shape=\(logitsArray.shape) dtype=\(logitsArray.dataType.rawValue) count=\(logitsArray.count)")
        }

        // Handle both Float16 and Float32 output
        let logits: [Float]
        if logitsArray.dataType == .float32 {
            let ptr = logitsArray.dataPointer.bindMemory(to: Float.self, capacity: vocabSize)
            logits = Array(UnsafeBufferPointer(start: ptr, count: vocabSize))
        } else {
            let ptr = logitsArray.dataPointer.bindMemory(to: Float16.self, capacity: vocabSize)
            logits = (0..<vocabSize).map { Float(ptr[$0]) }
        }
        return logits
    }
}

// MARK: - Memory Management

extension Qwen35CoreMLChat: ModelMemoryManageable {
    public var isLoaded: Bool { true }
    public func unload() { /* CoreML manages its own memory */ }
    public var memoryFootprint: Int { 500 * 1024 * 1024 } // ~500 MB estimate
}
