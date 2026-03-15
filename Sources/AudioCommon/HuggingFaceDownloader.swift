import Foundation
import Hub
import os

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

/// HuggingFace model downloader — shared between ASR, TTS, VAD, etc.
///
/// Uses `HubApi` from the swift-transformers `Hub` module for downloads,
/// which provides HF token auth, metadata tracking, and resume support.
public enum HuggingFaceDownloader {

    // MARK: - Cache Directory

    /// Get cache directory for a model.
    ///
    /// Returns the old flat cache path if it already contains model files (preserving
    /// ~10 GB of existing cached models), otherwise returns the new Hub-style path.
    public static func getCacheDirectory(for modelId: String, cacheDirName: String = "qwen3-speech") throws -> URL {
        let base = resolveBaseCacheDir(cacheDirName: cacheDirName)
        let fm = FileManager.default

        // Check old (flat) cache path for backward compat:
        //   ~/Library/Caches/qwen3-speech/aufklarer_Qwen3-ASR-0.6B-MLX-4bit/
        let oldDir = base.appendingPathComponent(sanitizedCacheKey(for: modelId), isDirectory: true)
        if weightsExist(in: oldDir) {
            return oldDir
        }

        // New Hub-style path:
        //   ~/Library/Caches/qwen3-speech/models/aufklarer/Qwen3-ASR-0.6B-MLX-4bit/
        let hub = HubApi(downloadBase: base)
        let repo = Hub.Repo(id: modelId)
        let dir = hub.localRepoLocation(repo)
        try fm.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    // MARK: - Weight Existence Check

    /// Check if safetensors weights exist in a directory.
    public static func weightsExist(in directory: URL) -> Bool {
        let fm = FileManager.default
        let contents: [URL]
        do {
            contents = try fm.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
        } catch {
            AudioLog.download.debug("Could not list directory \(directory.path): \(error)")
            contents = []
        }
        return contents.contains { $0.pathExtension == "safetensors" }
    }

    // MARK: - Download

    /// Download model files from HuggingFace using `HubApi.snapshot()`.
    ///
    /// Builds glob patterns from the file list:
    /// - Always includes `config.json`
    /// - If `additionalFiles` doesn't contain `.safetensors` files, adds `*.safetensors`
    ///   and `model.safetensors.index.json` to discover sharded weights automatically
    /// - All entries in `additionalFiles` are added as-is (they work as glob patterns)
    ///
    /// - Parameter localCheckFiles: Concrete file/directory paths (relative to `directory`)
    ///   to check before contacting the Hub. If **all** exist locally, the download is
    ///   skipped entirely — avoiding sequential HEAD-request verification for every cached
    ///   file. Pass an empty array (the default) to always verify against the Hub.
    public static func downloadWeights(
        modelId: String,
        to directory: URL,
        additionalFiles: [String] = [],
        localCheckFiles: [String] = [],
        progressHandler: ((Double) -> Void)? = nil,
        statusHandler: ((String) -> Void)? = nil
    ) async throws {
        // Fast path: skip network verification when key files already exist locally.
        if !localCheckFiles.isEmpty {
            let fm = FileManager.default
            let allPresent = localCheckFiles.allSatisfy { path in
                fm.fileExists(atPath: directory.appendingPathComponent(path).path)
            }
            if allPresent {
                AudioLog.download.info("All local check files present for \(modelId), skipping download")
                statusHandler?("Cached")
                progressHandler?(1.0)
                return
            }
        }

        var globs: [String] = ["config.json"]

        let hasExplicitWeights = additionalFiles.contains { $0.hasSuffix(".safetensors") }
        if !hasExplicitWeights {
            globs.append("*.safetensors")
            globs.append("model.safetensors.index.json")
        }
        for file in additionalFiles where !globs.contains(file) {
            globs.append(file)
        }

        let hub = makeHubApi(for: modelId, repoDir: directory)
        let repo = Hub.Repo(id: modelId)

        // Report total download size
        if let totalBytes = try? await getDownloadSize(hub: hub, repo: repo, globs: globs) {
            statusHandler?("Downloading \(formatBytes(totalBytes))...")
        } else {
            statusHandler?("Downloading...")
        }

        do {
            try await hub.snapshot(from: repo, matching: globs) { progress in
                progressHandler?(progress.fractionCompleted)
            }
        } catch {
            throw DownloadError.failedToDownload("\(modelId): \(error.localizedDescription)")
        }
    }

    /// Download model files using concurrent snapshot groups for faster first-time downloads.
    ///
    /// Each group of glob patterns runs its own `HubApi.snapshot()` call concurrently,
    /// parallelizing HEAD-request verification and file downloads across groups.
    /// This is significantly faster than a single sequential snapshot for models with
    /// many files (e.g. CoreML bundles + voices).
    ///
    /// - Parameter globGroups: Array of glob-pattern arrays, each group downloads concurrently.
    /// - Parameter localCheckFiles: Same as `downloadWeights` — skip everything if all present.
    public static func downloadWeightsParallel(
        modelId: String,
        to directory: URL,
        globGroups: [[String]],
        localCheckFiles: [String] = [],
        progressHandler: ((Double) -> Void)? = nil,
        statusHandler: ((String) -> Void)? = nil
    ) async throws {
        // Fast path: skip network when cached
        if !localCheckFiles.isEmpty {
            let fm = FileManager.default
            let allPresent = localCheckFiles.allSatisfy { path in
                fm.fileExists(atPath: directory.appendingPathComponent(path).path)
            }
            if allPresent {
                AudioLog.download.info("All local check files present for \(modelId), skipping download")
                statusHandler?("Cached")
                progressHandler?(1.0)
                return
            }
        }

        let hub = makeHubApi(for: modelId, repoDir: directory)
        let repo = Hub.Repo(id: modelId)

        // Query total download size before starting
        let allGlobs = globGroups.flatMap { $0 }
        if let totalBytes = try? await getDownloadSize(hub: hub, repo: repo, globs: allGlobs) {
            statusHandler?("Downloading \(formatBytes(totalBytes))...")
        } else {
            statusHandler?("Downloading...")
        }

        let totalGroups = Double(globGroups.count)
        let completedGroups = OSAllocatedUnfairLock(initialState: 0.0)

        try await withThrowingTaskGroup(of: Void.self) { group in
            for globs in globGroups {
                group.addTask {
                    try await hub.snapshot(from: repo, matching: globs) { progress in
                        let groupFraction = progress.fractionCompleted / totalGroups
                        let base = completedGroups.withLock { $0 }
                        progressHandler?(base + groupFraction)
                    }
                }
            }
            for try await _ in group {
                completedGroups.withLock { $0 += 1.0 / totalGroups }
            }
        }

        progressHandler?(1.0)
    }

    /// Query total download size for given glob patterns (via HF API metadata).
    private static func getDownloadSize(hub: HubApi, repo: Hub.Repo, globs: [String]) async throws -> Int {
        let metadata = try await hub.getFileMetadata(from: repo, matching: globs)
        return metadata.compactMap(\.size).reduce(0, +)
    }

    /// Format byte count as human-readable string.
    private static func formatBytes(_ bytes: Int) -> String {
        if bytes >= 1_000_000_000 {
            return String(format: "%.1f GB", Double(bytes) / 1_000_000_000)
        } else if bytes >= 1_000_000 {
            return String(format: "%.0f MB", Double(bytes) / 1_000_000)
        } else if bytes >= 1_000 {
            return String(format: "%.0f KB", Double(bytes) / 1_000)
        }
        return "\(bytes) B"
    }

    // MARK: - Security Helpers (kept for backward compat + security tests)

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

    /// Validate that a remote file name is safe.
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

    /// Validate that a local path stays within the expected directory.
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

    // MARK: - Private Helpers

    /// Resolve the base cache directory from env vars or system default.
    private static func resolveBaseCacheDir(cacheDirName: String) -> URL {
        let fm = FileManager.default
        let root: URL
        if let override = ProcessInfo.processInfo.environment["QWEN3_CACHE_DIR"],
           !override.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            root = URL(fileURLWithPath: override, isDirectory: true)
        } else if let override = ProcessInfo.processInfo.environment["QWEN3_ASR_CACHE_DIR"],
                  !override.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            // Legacy env var support
            root = URL(fileURLWithPath: override, isDirectory: true)
        } else {
            root = fm.urls(for: .cachesDirectory, in: .userDomainMask).first!
        }
        return root.appendingPathComponent(cacheDirName, isDirectory: true)
    }

    /// Create a `HubApi` whose `downloadBase` is derived from the repo directory that
    /// `getCacheDirectory` returned (strips the `models/<org>/<model>` suffix).
    private static func makeHubApi(for modelId: String, repoDir: URL) -> HubApi {
        // repoDir is  base/models/org/model
        // We need     base
        let repo = Hub.Repo(id: modelId)
        let suffix = "/\(repo.type.rawValue)/\(repo.id)"
        let repoDirPath = repoDir.path
        let downloadBase: URL
        if repoDirPath.hasSuffix(suffix) {
            let basePath = String(repoDirPath.dropLast(suffix.count))
            downloadBase = URL(fileURLWithPath: basePath, isDirectory: true)
        } else {
            // Fallback: old-style flat dir — use its parent as downloadBase.
            // Hub won't match this path, so we derive base from env/defaults.
            downloadBase = resolveBaseCacheDir(cacheDirName: repoDir.deletingLastPathComponent().lastPathComponent)
        }
        return HubApi(downloadBase: downloadBase)
    }
}
