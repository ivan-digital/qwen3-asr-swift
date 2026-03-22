import Foundation
import AudioCommon
import os

private let log = Logger(subsystem: "com.qwen3speech", category: "PipelineLLM")

private let llmLogger = Logger(subsystem: "audio.soniqo.speech", category: "LLM")

private func llmDebug(_ msg: String) {
    llmLogger.warning("[LLM] \(msg, privacy: .public)")
}

/// Adapter bridging Qwen3ChatModel to the PipelineLLM protocol for VoicePipeline.
///
/// Handles async-to-blocking bridge, cancellation, token cleanup, and
/// pending phrase accumulation for interrupted turns.
///
/// ```swift
/// let chat = try await Qwen3ChatModel.fromPretrained()
/// let llm = Qwen3PipelineLLM(model: chat, systemPrompt: "Be brief.")
/// let pipeline = VoicePipeline(stt: asr, tts: tts, vad: vad, llm: llm, ...)
/// ```
public final class Qwen3PipelineLLM: PipelineLLM {
    private let model: Qwen3ChatModel
    private let systemPrompt: String?
    private let sampling: ChatSamplingConfig
    private var cancelled = false
    private var consumeTask: Task<Void, Never>?

    /// Unanswered user phrases from cancelled turns — prepended to the next LLM call.
    private var pendingPhrases: [String] = []

    /// Optional callback to forward tokens to the UI (called on background thread).
    public var onToken: ((String) -> Void)?

    /// Maximum response length in characters. Prevents TTS OOM from long outputs.
    public var maxResponseChars: Int = 200

    public init(
        model: Qwen3ChatModel,
        systemPrompt: String? = nil,
        sampling: ChatSamplingConfig = ChatSamplingConfig(
            temperature: 0.6, topK: 40, maxTokens: 30)
    ) {
        self.model = model
        self.systemPrompt = systemPrompt
        self.sampling = sampling
    }

    public func chat(
        messages: [(role: MessageRole, content: String)],
        onToken: @escaping (String, Bool) -> Void
    ) {
        cancelled = false

        // Extract the last user message
        guard let lastUser = messages.last(where: { $0.role == .user }) else {
            onToken("", true)
            return
        }

        // Combine any unanswered phrases from cancelled turns
        let combinedInput: String
        if pendingPhrases.isEmpty {
            combinedInput = lastUser.content
        } else {
            pendingPhrases.append(lastUser.content)
            combinedInput = pendingPhrases.joined(separator: ". ")
            log.info("Combined \(self.pendingPhrases.count) phrases: '\(combinedInput)'")
            pendingPhrases.removeAll()
        }

        log.info("Input: '\(combinedInput)'")

        let stream = model.chatStream(
            combinedInput,
            systemPrompt: systemPrompt,
            sampling: sampling
        )

        // Block until stream completes (pipeline calls from background thread)
        let sem = DispatchSemaphore(value: 0)
        cancelSem = sem
        var fullResponse = ""
        llmDebug("start: '\(combinedInput)' maxTokens=\(sampling.maxTokens)")
        let task = Task {
            defer {
                llmDebug("done: \(fullResponse.count) chars response='\(fullResponse)'")
                sem.signal()
            }
            do {
                for try await token in stream {
                    llmDebug("token: '\(token)'")
                    guard !self.cancelled else {
                        log.info("Cancelled after \(fullResponse.count) chars")
                        break
                    }
                    // Clean token: strip thinking tags, markdown, newlines, broken unicode.
                    // Keep leading whitespace — tokenizer uses it to mark word boundaries.
                    var clean = token
                        .replacingOccurrences(of: "</think>", with: "")
                        .replacingOccurrences(of: "<think>", with: "")
                        .replacingOccurrences(of: "/think", with: "")
                        .replacingOccurrences(of: "**", with: "")
                        .replacingOccurrences(of: "* ", with: "")
                        .replacingOccurrences(of: "#", with: "")
                        .replacingOccurrences(of: "```", with: "")
                        .replacingOccurrences(of: "\n", with: " ")
                        .replacingOccurrences(of: "  ", with: " ")
                    // Strip non-ASCII / broken unicode from INT4 — keep only Latin text
                    clean = clean.filter {
                        $0.isASCII && ($0.isLetter || $0.isNumber || $0.isPunctuation ||
                        $0.isWhitespace || $0 == "'" || $0 == "-")
                    }
                    // Only trim trailing whitespace — leading space is word boundary marker
                    while clean.hasSuffix(" ") { clean = String(clean.dropLast()) }
                    guard !clean.isEmpty else { continue }

                    // Cap response length to prevent TTS OOM
                    guard fullResponse.count < self.maxResponseChars else {
                        log.info("Response capped at \(self.maxResponseChars) chars")
                        break
                    }

                    fullResponse += clean
                    self.onToken?(clean)
                    onToken(clean, false)
                }
                log.info("Output (\(fullResponse.count) chars): '\(fullResponse)'")
                // If model produced nothing, emit a minimal response so pipeline doesn't stall
                if fullResponse.isEmpty {
                    llmDebug("empty response — emitting fallback")
                    onToken("...", false)
                }
                onToken("", true)
            } catch {
                log.error("Error: \(error)")
                onToken("", true)
            }
        }
        consumeTask = task
        sem.wait()
        consumeTask = nil
        cancelSem = nil

        // If cancelled, save phrase for next call
        if cancelled {
            pendingPhrases.append(lastUser.content)
            log.info("Queued unanswered phrase (pending: \(self.pendingPhrases.count))")
        }
    }

    public func cancel() {
        cancelled = true
        consumeTask?.cancel()
        model.cancelGeneration()
        // Signal semaphore to unblock the pipeline worker thread immediately.
        // The Task's defer block will signal again but DispatchSemaphore handles that.
        cancelSem?.signal()
    }

    private var cancelSem: DispatchSemaphore?
}
