// MARK: - LLM Protocol

/// Protocol for language model integration with voice pipelines.
///
/// Conforming types bridge an LLM (local or remote) to the VoicePipeline's
/// ASR → LLM → TTS flow. The pipeline calls `chat()` on a background thread
/// and expects blocking behavior (return when generation is complete).
public protocol PipelineLLM: AnyObject {
    /// Generate a response given conversation messages.
    ///
    /// Called on the pipeline's worker thread (blocking). Emit tokens via
    /// `onToken(text, isFinal)` — the pipeline forwards them to TTS.
    func chat(messages: [(role: MessageRole, content: String)],
              onToken: @escaping (String, Bool) -> Void)

    /// Cancel in-progress generation. Thread-safe.
    func cancel()
}

/// Message roles for LLM conversation.
public enum MessageRole: Int, Sendable {
    case system = 0
    case user = 1
    case assistant = 2
    case tool = 3
}

// MARK: - Tool Calling

/// A tool that can be invoked by the LLM during voice pipeline execution.
public struct PipelineTool {
    public let name: String
    public let description: String
    public let handler: (String) -> String
    public let cooldown: Int

    /// - Parameters:
    ///   - name: Tool name (used by LLM to invoke)
    ///   - description: What the tool does (included in LLM system prompt)
    ///   - cooldown: Minimum seconds between invocations (0 = no limit)
    ///   - handler: Synchronous handler `(arguments) -> result`. Called on pipeline worker thread.
    public init(
        name: String,
        description: String,
        cooldown: Int = 0,
        handler: @escaping (String) -> String
    ) {
        self.name = name
        self.description = description
        self.cooldown = cooldown
        self.handler = handler
    }
}
