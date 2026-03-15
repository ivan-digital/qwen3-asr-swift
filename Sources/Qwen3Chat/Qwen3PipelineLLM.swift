import Foundation
import AudioCommon
import os

private let log = Logger(subsystem: "com.qwen3speech", category: "PipelineLLM")

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
        var fullResponse = ""
        let task = Task {
            defer { sem.signal() }
            do {
                for try await token in stream {
                    guard !self.cancelled else {
                        log.info("Cancelled after \(fullResponse.count) chars")
                        break
                    }
                    // Skip garbage tokens (broken unicode from INT4 quantization)
                    let clean = token.filter {
                        $0.isASCII || $0.isLetter || $0.isNumber ||
                        $0.isPunctuation || $0.isWhitespace
                    }
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
                onToken("", true)
            } catch {
                log.error("Error: \(error)")
                onToken("", true)
            }
        }
        consumeTask = task
        sem.wait()
        consumeTask = nil

        // If cancelled, save phrase for next call
        if cancelled {
            pendingPhrases.append(lastUser.content)
            log.info("Queued unanswered phrase (pending: \(self.pendingPhrases.count))")
        }
    }

    public func cancel() {
        cancelled = true
        consumeTask?.cancel()
    }
}
