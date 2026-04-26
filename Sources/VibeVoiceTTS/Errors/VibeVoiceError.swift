import Foundation

public enum VibeVoiceError: Error, LocalizedError {
    case voiceCacheNotLoaded
    case modelNotInitialized(component: String)
    case weightsMissing(key: String)
    case unsupportedSchedulerConfig(String)
    case shapeMismatch(expected: [Int], actual: [Int], context: String? = nil)
    case invalidConfiguration(String)
    case fileNotFound(path: String)
    case audioProcessingError(String)
    case cacheError(String)
    case validationFailed(name: String, maxDiff: Float, tolerance: Float)
    case invalidInput(parameter: String, reason: String)

    public var errorDescription: String? {
        switch self {
        case .voiceCacheNotLoaded:
            return "Voice cache not loaded. Call loadVoiceCache() first."
        case .modelNotInitialized(let component):
            return "Model component '\(component)' not initialized"
        case .weightsMissing(let key):
            return "Required weight key missing: \(key)"
        case .unsupportedSchedulerConfig(let msg):
            return "Unsupported scheduler configuration: \(msg)"
        case .shapeMismatch(let expected, let actual, let context):
            let base = "Shape mismatch: expected \(expected), got \(actual)"
            if let ctx = context {
                return "\(base) (\(ctx))"
            }
            return base
        case .invalidConfiguration(let msg):
            return "Invalid configuration: \(msg)"
        case .fileNotFound(let path):
            return "File not found: \(path)"
        case .audioProcessingError(let msg):
            return "Audio processing error: \(msg)"
        case .cacheError(let msg):
            return "Cache error: \(msg)"
        case .validationFailed(let name, let maxDiff, let tolerance):
            return "Validation failed for '\(name)': max diff \(maxDiff) exceeds tolerance \(tolerance)"
        case .invalidInput(let parameter, let reason):
            return "Invalid input for '\(parameter)': \(reason)"
        }
    }
}
