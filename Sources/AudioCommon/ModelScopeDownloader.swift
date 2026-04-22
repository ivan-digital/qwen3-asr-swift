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

    // MARK: - Model ID Mapping

    /// ModelScope uses the same model IDs as HuggingFace - no mapping needed
    public static func mapToModelScopeId(_ huggingFaceId: String) -> String {
        // Use the original model ID as-is
        // ModelScope supports HuggingFace-style model paths like:
        // aufklarer/Qwen3-TTS-12Hz-0.6B-Base-MLX-4bit
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
        // ModelScope uses a different API structure than HuggingFace
        // Try to list files using the model's file browser endpoint
        let url = URL(string: "https://www.modelscope.cn/models/\(modelId)/files")!

        AudioLog.download.debug("ModelScope: Listing files at URL: \(url.absoluteString), pattern: \(pattern)")

        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        request.setValue("application/json", forHTTPHeaderField: "Accept")

        let (data, response) = try await URLSession.shared.data(for: request)

        AudioLog.download.debug("ModelScope: Response status: \((response as? HTTPURLResponse)?.statusCode ?? -1)")
        AudioLog.download.debug("ModelScope: Response data size: \(data.count) bytes")

        // For now, use a simpler approach - try to download common files directly
        // This avoids the complexity of parsing ModelScope's web interface
        if pattern == "*.safetensors" {
            // Try common safetensors file names
            return ["model.safetensors"]
        } else if pattern == "config.json" {
            return ["config.json"]
        }

        // For other patterns, return empty array for now
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