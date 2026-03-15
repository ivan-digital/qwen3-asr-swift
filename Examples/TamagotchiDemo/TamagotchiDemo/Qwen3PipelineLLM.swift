import Foundation
import os
import Qwen3Chat
import SpeechCore

private let log = Logger(subsystem: "audio.soniqo.TamagotchiDemo", category: "LLM")

/// Adapter bridging Qwen3ChatModel to the PipelineLLM protocol for VoicePipeline.
final class Qwen3PipelineLLM: PipelineLLM {
    private let model: Qwen3ChatModel
    private let systemPrompt: String
    private let sampling: ChatSamplingConfig
    private var cancelled = false
    private var consumeTask: Task<Void, Never>?

    /// Unanswered user phrases from cancelled turns — prepended to the next LLM call.
    private var pendingPhrases: [String] = []

    /// Optional callback to forward tokens to the UI (called on background thread).
    var onUIToken: ((String) -> Void)?

    init(
        model: Qwen3ChatModel,
        systemPrompt: String,
        sampling: ChatSamplingConfig = ChatSamplingConfig(temperature: 0.6, maxTokens: 128)
    ) {
        self.model = model
        self.systemPrompt = systemPrompt
        self.sampling = sampling
    }

    func chat(
        messages: [(role: MessageRole, content: String)],
        onToken: @escaping (String, Bool) -> Void
    ) {
        cancelled = false

        // Extract the last user message
        guard let lastUser = messages.last(where: { $0.role == .user }) else {
            onToken("", true)
            return
        }

        // Combine any unanswered phrases from cancelled turns with the current one.
        let combinedInput: String
        if pendingPhrases.isEmpty {
            combinedInput = lastUser.content
        } else {
            pendingPhrases.append(lastUser.content)
            combinedInput = pendingPhrases.joined(separator: ". ")
            log.warning("LLM combined \(self.pendingPhrases.count) phrases: '\(combinedInput)'")
            pendingPhrases.removeAll()
        }

        log.warning("LLM input: '\(combinedInput)' temp=\(self.sampling.temperature) maxTok=\(self.sampling.maxTokens) maxThink=\(self.sampling.maxThinkingTokens) disableThink=\(self.sampling.disableThinking)")

        let stream = model.chatStream(
            combinedInput,
            systemPrompt: systemPrompt,
            sampling: sampling
        )

        // Block until stream completes (C pipeline calls from background thread)
        let sem = DispatchSemaphore(value: 0)
        var fullResponse = ""
        let task = Task {
            defer { sem.signal() }
            do {
                for try await token in stream {
                    guard !self.cancelled else {
                        log.warning("LLM cancelled after \(fullResponse.count) chars")
                        break
                    }
                    // Skip garbage tokens (broken unicode from INT4 quantization)
                    let clean = token.filter { $0.isASCII || $0.isLetter || $0.isNumber || $0.isPunctuation || $0.isWhitespace }
                    guard !clean.isEmpty else { continue }
                    // Cap total response length to prevent TTS OOM (Kokoro allocates
                    // memory proportional to token count; >50 chars risks jetsam on iPhone)
                    guard fullResponse.count < 80 else {
                        log.warning("LLM response capped at 80 chars to prevent TTS OOM")
                        break
                    }
                    fullResponse += clean
                    self.onUIToken?(clean)
                    onToken(clean, false)
                }
                log.warning("LLM output (\(fullResponse.count) chars): '\(fullResponse)'")
                // Trim leading/trailing whitespace for TTS
                let trimmed = fullResponse.trimmingCharacters(in: .whitespacesAndNewlines)
                if trimmed.isEmpty {
                    log.warning("LLM produced empty response after cleanup")
                }
                onToken("", true)
            } catch {
                log.error("LLM error: \(error)")
                onToken("", true)
            }
        }
        consumeTask = task
        sem.wait()
        consumeTask = nil

        // If cancelled (interrupted), save this phrase so it's included in the next call.
        if cancelled {
            pendingPhrases.append(lastUser.content)
            log.warning("LLM queued unanswered phrase: '\(lastUser.content)' (pending: \(self.pendingPhrases.count))")
        }
    }

    func cancel() {
        cancelled = true
        consumeTask?.cancel()
    }
}
