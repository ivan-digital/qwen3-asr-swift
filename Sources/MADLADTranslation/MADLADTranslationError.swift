import Foundation

/// Errors for MADLAD translation model operations.
public enum MADLADTranslationError: LocalizedError {
    case modelLoadFailed(String)
    case tokenizerLoadFailed(String)
    case inferenceFailed(String)
    case configNotFound(URL)
    case modelNotFound(URL)
    case unsupportedLanguage(String)

    public var errorDescription: String? {
        switch self {
        case .modelLoadFailed(let reason):
            "Failed to load MADLAD model: \(reason)"
        case .tokenizerLoadFailed(let reason):
            "Failed to load tokenizer: \(reason)"
        case .inferenceFailed(let reason):
            "Translation failed: \(reason)"
        case .configNotFound(let url):
            "Config not found at \(url.path)"
        case .modelNotFound(let url):
            "Model not found at \(url.path)"
        case .unsupportedLanguage(let code):
            "Language token <2\(code)> not in vocabulary"
        }
    }
}
