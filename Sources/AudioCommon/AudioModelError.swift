import Foundation

/// Unified error type for audio model operations.
public enum AudioModelError: Error, LocalizedError {
    /// Model failed to load from disk or network.
    case modelLoadFailed(modelId: String, reason: String, underlying: Error? = nil)
    /// Weight file could not be read or parsed.
    case weightLoadingFailed(path: String, underlying: Error? = nil)
    /// Inference or generation step failed.
    case inferenceFailed(operation: String, reason: String)
    /// Model configuration is invalid or incompatible.
    case invalidConfiguration(model: String, reason: String)
    /// Voice preset file not found.
    case voiceNotFound(voice: String, searchPath: String)

    public var errorDescription: String? {
        switch self {
        case .modelLoadFailed(let modelId, let reason, let underlying):
            var msg = "Failed to load model '\(modelId)': \(reason)"
            if let underlying { msg += " (\(underlying.localizedDescription))" }
            return msg
        case .weightLoadingFailed(let path, let underlying):
            var msg = "Failed to load weights from '\(path)'"
            if let underlying { msg += ": \(underlying.localizedDescription)" }
            return msg
        case .inferenceFailed(let operation, let reason):
            return "Inference failed during \(operation): \(reason)"
        case .invalidConfiguration(let model, let reason):
            return "Invalid configuration for '\(model)': \(reason)"
        case .voiceNotFound(let voice, let searchPath):
            return "Voice preset '\(voice)' not found at '\(searchPath)'"
        }
    }
}
