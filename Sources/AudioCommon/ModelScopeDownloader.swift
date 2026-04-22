import Foundation
import os

/// ModelScope download errors
public enum ModelScopeDownloadError: Error, LocalizedError {
    case failedToDownload(String)
    case invalidModelId(String)
    case networkError(Error)
    case fileNotFound(String)

    public var errorDescription: String? {
        switch self {
        case .failedToDownload(let file):
            return "Failed to download from ModelScope: \(file)"
        case .invalidModelId(let modelId):
            return "Invalid ModelScope model ID: \(modelId)"
        case .networkError(let error):
            return "Network error: \(error.localizedDescription)"
        case .fileNotFound(let file):
            return "File not found: \(file)"
        }
    }
}

/// ModelScope model downloader - alternative to HuggingFace
///
/// Downloads models from modelscope.cn API
public enum ModelScopeDownloader {

    // MARK: - Configuration

    /// ModelScope API base URL
    private static let baseURL = "https://www.modelscope.cn/api/v1/models"

    /// Map HuggingFace model IDs to ModelScope equivalents
    private static let modelMapping: [String: String] = [
        "aufklarer/Qwen3-ASR-0.6B-MLX-4bit": "Qwen/Qwen3-ASR-0.6B-MLX-4bit",
        "aufklarer/Qwen3-ASR-1.7B-MLX-8bit": "Qwen/Qwen3-ASR-1.7B-MLX-8bit",
        "aufklarer/Qwen3-TTS-0.6B": "Qwen/Qwen3-TTS-0.6B",
        "pyannote/segmentation-3.0": "pyannote/segmentation-3.0",
        "pyannote/wespeaker-resnet34-LM": "pyannote/wespeaker-resnet34-LM"
    ]

    // MARK: - Model ID Mapping

    /// Convert HuggingFace model ID to ModelScope equivalent
    public static func mapToModelScopeId(_ huggingFaceId: String) -> String {
        // First check direct mapping
        if let mapped = modelMapping[huggingFaceId] {
            return mapped
        }

        // For Qwen models, try to map aufklarer -> Qwen
        if huggingFaceId.hasPrefix("aufklarer/Qwen") {
            return huggingFaceId.replacingOccurrences(of: "aufklarer/Qwen", with: "Qwen/Qwen")
        }

        // For other models, try to use as-is (many are cross-platform)
        return huggingFaceId
    }

    // MARK: - Cache Directory

    /// Get cache directory for a model (same as HuggingFace for compatibility)
    public static func getCacheDirectory(for modelId: String, basePath: URL? = nil, cacheDirName: String = "qwen3-speech") throws -> URL {
        // Reuse HuggingFace downloader's cache logic for consistency
        try HuggingFaceDownloader.getCacheDirectory(for: modelId, basePath: basePath, cacheDirName: cacheDirName)
    }

    // MARK: - Download

    /// Download model files from ModelScope
    public static func downloadWeights(
        modelId: String,
        to directory: URL,
        additionalFiles: [String] = [],
        offlineMode: Bool = false,
        progressHandler: ((Double) -> Void)? = nil
    ) async throws {
        // Skip network requests when weights are already cached
        if offlineMode && weightsExist(in: directory) {
            progressHandler?(1.0)
            return
        }

        let modelScopeId = mapToModelScopeId(modelId)

        // Build list of files to download
        var filesToDownload = ["config.json"]

        // Add safetensors files if not explicitly listed
        let hasExplicitWeights = additionalFiles.contains { $0.hasSuffix(".safetensors") }
        if !hasExplicitWeights {
            // Try to download safetensors files
            filesToDownload.append("*.safetensors")
            filesToDownload.append("model.safetensors.index.json")
        }

        // Add additional files
        for file in additionalFiles where !filesToDownload.contains(file) {
            filesToDownload.append(file)
        }

        // Download each file
        let totalFiles = filesToDownload.count
        var downloadedFiles = 0

        for filePattern in filesToDownload {
            do {
                if filePattern.contains("*") {
                    // Handle glob patterns by listing files first
                    let files = try await listFiles(modelId: modelScopeId, pattern: filePattern)
                    for file in files {
                        try await downloadFile(modelId: modelScopeId, fileName: file, to: directory)
                        downloadedFiles += 1
                        progressHandler?(Double(downloadedFiles) / Double(totalFiles))
                    }
                } else {
                    // Download specific file
                    try await downloadFile(modelId: modelScopeId, fileName: filePattern, to: directory)
                    downloadedFiles += 1
                    progressHandler?(Double(downloadedFiles) / Double(totalFiles))
                }
            } catch {
                AudioLog.download.debug("Failed to download \(filePattern) from ModelScope: \(error)")
                // Continue with other files - some might be optional
            }
        }

        // Verify we got at least some files
        if !weightsExist(in: directory) {
            throw ModelScopeDownloadError.failedToDownload("No model weights found for \(modelId)")
        }
    }

    // MARK: - Private Helpers

    /// Check if safetensors weights exist in a directory
    private static func weightsExist(in directory: URL) -> Bool {
        HuggingFaceDownloader.weightsExist(in: directory)
    }

    /// List files in ModelScope repository matching a pattern
    private static func listFiles(modelId: String, pattern: String) async throws -> [String] {
        let url = URL(string: "\(baseURL)/\(modelId)/tree")!

        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        request.setValue("application/json", forHTTPHeaderField: "Accept")

        let (data, response) = try await URLSession.shared.data(for: request)

        guard let httpResponse = response as? HTTPURLResponse,
              httpResponse.statusCode == 200 else {
            throw ModelScopeDownloadError.networkError(URLError(.badServerResponse))
        }

        // Parse the response to get file list
        // This is a simplified implementation - actual ModelScope API might differ
        if let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let files = json["files"] as? [[String: Any]] {

            let fileNames = files.compactMap { $0["name"] as? String }

            // Simple pattern matching for *.safetensors
            if pattern == "*.safetensors" {
                return fileNames.filter { $0.hasSuffix(".safetensors") }
            }

            return fileNames
        }

        // Fallback - try common safetensors files
        if pattern == "*.safetensors" {
            return ["model.safetensors"]
        }

        return []
    }

    /// Download a single file from ModelScope
    private static func downloadFile(modelId: String, fileName: String, to directory: URL) async throws {
        let encodedFileName = fileName.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? fileName
        let url = URL(string: "\(baseURL)/\(modelId)/resolve/master/\(encodedFileName)")!

        let fileURL = directory.appendingPathComponent(fileName)

        // Skip if file already exists
        if FileManager.default.fileExists(atPath: fileURL.path) {
            AudioLog.download.debug("File already exists: \(fileName)")
            return
        }

        var request = URLRequest(url: url)
        request.httpMethod = "GET"

        let (data, response) = try await URLSession.shared.data(for: request)

        guard let httpResponse = response as? HTTPURLResponse else {
            throw ModelScopeDownloadError.networkError(URLError(.badServerResponse))
        }

        guard httpResponse.statusCode == 200 else {
            throw ModelScopeDownloadError.fileNotFound(fileName)
        }

        // Write file to disk
        try data.write(to: fileURL)
        AudioLog.download.debug("Downloaded \(fileName) to \(fileURL.path)")
    }
}

// MARK: - Model Source Selection

/// Enum for selecting model download source
public enum ModelSource {
    case huggingFace
    case modelScope
}

/// Get the configured model source from environment
public func getModelSource() -> ModelSource {
    if ProcessInfo.processInfo.environment["QWEN3_MODEL_SOURCE"] == "modelscope" {
        return .modelScope
    }
    return .huggingFace
}