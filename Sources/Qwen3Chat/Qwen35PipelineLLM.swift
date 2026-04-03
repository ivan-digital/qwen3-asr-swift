import Foundation
import AudioCommon

/// Common interface for Qwen3.5 chat backends (MLX or CoreML).
public protocol Qwen35ChatBackend: AnyObject {
    var tokenizer: ChatTokenizer { get }
    var config: Qwen3ChatConfig { get }
    func generateStream(messages: [ChatMessage], sampling: ChatSamplingConfig)
        -> AsyncThrowingStream<String, Error>
    func resetState()
}

extension Qwen35MLXChat: Qwen35ChatBackend {}
extension Qwen35CoreMLChat: Qwen35ChatBackend {}

/// Bridges any Qwen3.5 backend to VoicePipeline's PipelineLLM protocol.
public final class Qwen35PipelineLLM: PipelineLLM {
    private let model: any Qwen35ChatBackend
    private let systemPrompt: String
    private let sampling: ChatSamplingConfig
    private var cancelled = false

    public var onToken: ((String) -> Void)?

    public init(
        model: any Qwen35ChatBackend,
        systemPrompt: String = "Your name is Tama. Give short direct answers. Do not explain your reasoning.",
        sampling: ChatSamplingConfig = .default
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

        let chatMessages = messages.compactMap { msg -> ChatMessage? in
            switch msg.role {
            case .system:  return ChatMessage(role: .system, content: msg.content)
            case .user:    return ChatMessage(role: .user, content: msg.content)
            case .assistant: return ChatMessage(role: .assistant, content: msg.content)
            default: return nil
            }
        }

        var fullMessages = [ChatMessage(role: .system, content: systemPrompt)]
        fullMessages.append(contentsOf: chatMessages)

        let stream = model.generateStream(messages: fullMessages, sampling: sampling)
        let semaphore = DispatchSemaphore(value: 0)
        var fullResponse = ""

        Task {
            do {
                for try await chunk in stream {
                    guard !self.cancelled else { break }
                    fullResponse += chunk
                    self.onToken?(chunk)
                    onToken(chunk, false)
                }
            } catch { }

            if !fullResponse.isEmpty {
                onToken("", true)
            }
            semaphore.signal()
        }

        semaphore.wait()
    }

    public func cancel() {
        cancelled = true
    }
}
