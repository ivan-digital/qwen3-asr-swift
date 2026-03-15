import Foundation

/// Errors for Qwen3 chat model operations.
public enum ChatModelError: LocalizedError {
    case modelLoadFailed(String)
    case tokenizerLoadFailed(String)
    case inferenceFailed(String)
    case configNotFound(URL)
    case modelNotFound(URL)

    public var errorDescription: String? {
        switch self {
        case .modelLoadFailed(let reason):
            "Failed to load chat model: \(reason)"
        case .tokenizerLoadFailed(let reason):
            "Failed to load tokenizer: \(reason)"
        case .inferenceFailed(let reason):
            "Inference failed: \(reason)"
        case .configNotFound(let url):
            "Config not found at \(url.path)"
        case .modelNotFound(let url):
            "CoreML model not found at \(url.path)"
        }
    }
}
