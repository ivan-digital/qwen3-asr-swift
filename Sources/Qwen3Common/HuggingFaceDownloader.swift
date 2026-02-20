import Foundation

/// Download errors
public enum DownloadError: Error, LocalizedError {
    case failedToDownload(String)
    case invalidRemoteFileName(String)

    public var errorDescription: String? {
        switch self {
        case .failedToDownload(let file):
            return "Failed to download: \(file)"
        case .invalidRemoteFileName(let file):
            return "Refusing to write unsafe remote file name: \(file)"
        }
    }
}

/// HuggingFace model downloader — shared between ASR and TTS
public enum HuggingFaceDownloader {

    /// Max retries per file download
    private static let maxRetries = 3

    /// Get cache directory for a model
    public static func getCacheDirectory(for modelId: String, cacheDirName: String = "qwen3-speech") throws -> URL {
        let cacheKey = sanitizedCacheKey(for: modelId)
        let fm = FileManager.default

        let baseCacheDir: URL
        if let override = ProcessInfo.processInfo.environment["QWEN3_CACHE_DIR"],
           !override.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            baseCacheDir = URL(fileURLWithPath: override, isDirectory: true)
        } else if let override = ProcessInfo.processInfo.environment["QWEN3_ASR_CACHE_DIR"],
                  !override.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            // Legacy env var support
            baseCacheDir = URL(fileURLWithPath: override, isDirectory: true)
        } else {
            baseCacheDir = fm.urls(for: .cachesDirectory, in: .userDomainMask).first!
        }

        let cacheDir = baseCacheDir
            .appendingPathComponent(cacheDirName, isDirectory: true)
            .appendingPathComponent(cacheKey, isDirectory: true)

        try fm.createDirectory(at: cacheDir, withIntermediateDirectories: true)
        return cacheDir
    }

    /// Convert an arbitrary modelId into a single, safe path component for on-disk caching.
    public static func sanitizedCacheKey(for modelId: String) -> String {
        let replaced = modelId.replacingOccurrences(of: "/", with: "_")

        let allowed = CharacterSet(charactersIn: "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
        var scalars: [UnicodeScalar] = []
        scalars.reserveCapacity(replaced.unicodeScalars.count)
        for s in replaced.unicodeScalars {
            scalars.append(allowed.contains(s) ? s : "_")
        }

        var cleaned = String(String.UnicodeScalarView(scalars))
        cleaned = cleaned.trimmingCharacters(in: CharacterSet(charactersIn: "._"))

        if cleaned.isEmpty || cleaned == "." || cleaned == ".." {
            cleaned = "model"
        }

        return cleaned
    }

    /// Check if safetensors weights exist in a directory
    public static func weightsExist(in directory: URL) -> Bool {
        let fm = FileManager.default
        let contents = (try? fm.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)) ?? []
        return contents.contains { $0.pathExtension == "safetensors" }
    }

    /// Validate that a remote file name is safe
    public static func validatedRemoteFileName(_ file: String) throws -> String {
        let base = URL(fileURLWithPath: file).lastPathComponent
        guard base == file else {
            throw DownloadError.invalidRemoteFileName(file)
        }
        guard !base.isEmpty, !base.hasPrefix("."), !base.contains("..") else {
            throw DownloadError.invalidRemoteFileName(file)
        }
        guard base.range(of: #"^[A-Za-z0-9._-]+$"#, options: .regularExpression) != nil else {
            throw DownloadError.invalidRemoteFileName(file)
        }
        return base
    }

    /// Validate that a local path stays within the expected directory
    public static func validatedLocalPath(directory: URL, fileName: String) throws -> URL {
        let local = directory.appendingPathComponent(fileName, isDirectory: false)
        let dirPath = directory.standardizedFileURL.path
        let localPath = local.standardizedFileURL.path
        let prefix = dirPath.hasSuffix("/") ? dirPath : (dirPath + "/")
        guard localPath.hasPrefix(prefix) else {
            throw DownloadError.invalidRemoteFileName(fileName)
        }
        return local
    }

    /// Create a URLSession configured for large model downloads
    private static func makeSession() -> URLSession {
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 30      // 30s to start receiving data
        config.timeoutIntervalForResource = 600    // 10 min max per file
        config.waitsForConnectivity = true
        return URLSession(configuration: config)
    }

    /// Download a single file with retry logic, streaming to disk
    private static func downloadFile(
        url: URL,
        to localPath: URL,
        session: URLSession,
        fileName: String
    ) async throws {
        var lastError: Error?

        for attempt in 1...maxRetries {
            do {
                let (tempURL, response) = try await session.download(from: url)

                guard let httpResponse = response as? HTTPURLResponse,
                      httpResponse.statusCode == 200 else {
                    let status = (response as? HTTPURLResponse)?.statusCode ?? -1
                    throw DownloadError.failedToDownload("\(fileName) (HTTP \(status))")
                }

                // Move downloaded file to final location
                let fm = FileManager.default
                if fm.fileExists(atPath: localPath.path) {
                    try fm.removeItem(at: localPath)
                }
                try fm.moveItem(at: tempURL, to: localPath)
                return  // Success
            } catch {
                lastError = error
                if attempt < maxRetries {
                    // Exponential backoff: 1s, 2s, 4s
                    let delay = UInt64(pow(2.0, Double(attempt - 1))) * 1_000_000_000
                    print("[Download] Retry \(attempt)/\(maxRetries) for \(fileName): \(error.localizedDescription)")
                    try? await Task.sleep(nanoseconds: delay)
                }
            }
        }

        throw lastError ?? DownloadError.failedToDownload(fileName)
    }

    /// Download model files from HuggingFace
    public static func downloadWeights(
        modelId: String,
        to directory: URL,
        additionalFiles: [String] = [],
        progressHandler: ((Double) -> Void)? = nil
    ) async throws {
        let baseURL = "https://huggingface.co/\(modelId)/resolve/main"
        let session = makeSession()
        defer { session.finishTasksAndInvalidate() }

        // Files to download (config and tokenizer)
        var filesToDownload = [
            "config.json"
        ]
        filesToDownload.append(contentsOf: additionalFiles)

        // Discover model files only if additionalFiles doesn't already include safetensors
        let hasExplicitWeights = additionalFiles.contains { $0.hasSuffix(".safetensors") }
        if !hasExplicitWeights {
            let indexPath = directory.appendingPathComponent("model.safetensors.index.json")

            if !FileManager.default.fileExists(atPath: indexPath.path) {
                let indexURL = URL(string: "\(baseURL)/model.safetensors.index.json")!
                if let (tempURL, indexResponse) = try? await session.download(from: indexURL),
                   let httpResponse = indexResponse as? HTTPURLResponse,
                   httpResponse.statusCode == 200 {
                    try? FileManager.default.moveItem(at: tempURL, to: indexPath)
                }
            }

            var modelFiles: [String] = []
            if FileManager.default.fileExists(atPath: indexPath.path),
               let indexData = try? Data(contentsOf: indexPath),
               let index = try? JSONSerialization.jsonObject(with: indexData) as? [String: Any],
               let weightMap = index["weight_map"] as? [String: String],
               !weightMap.isEmpty {
                let uniqueFiles = Set(weightMap.values)
                modelFiles = Array(uniqueFiles).sorted()
            } else {
                try? FileManager.default.removeItem(at: indexPath)
                modelFiles = ["model.safetensors"]
            }

            filesToDownload.append(contentsOf: modelFiles)
        }

        for (index, file) in filesToDownload.enumerated() {
            let safeFile = try validatedRemoteFileName(file)
            let localPath = try validatedLocalPath(directory: directory, fileName: safeFile)

            if FileManager.default.fileExists(atPath: localPath.path) {
                progressHandler?(Double(index + 1) / Double(filesToDownload.count))
                continue
            }

            let url = URL(string: "\(baseURL)/\(safeFile)")!
            try await downloadFile(url: url, to: localPath, session: session, fileName: safeFile)

            progressHandler?(Double(index + 1) / Double(filesToDownload.count))
        }
    }
}
