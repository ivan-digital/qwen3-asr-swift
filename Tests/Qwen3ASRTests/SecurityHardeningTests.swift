import XCTest
import Foundation
@testable import Qwen3ASR
@testable import AudioCommon

// MARK: - WAV Parsing Hardening Tests

final class WAVParsingSecurityTests: XCTestCase {

    /// Build a minimal valid 16-bit mono PCM WAV with the given samples.
    private func buildWAV(
        sampleRate: UInt32 = 16000,
        numChannels: UInt16 = 1,
        bitsPerSample: UInt16 = 16,
        audioFormat: UInt16 = 1,
        samples: [Int16] = [0, 100, -100, 200, -200, 0, 0, 0],
        overrideDataChunkSize: UInt32? = nil,
        appendExtraChunk: Data? = nil,
        corruptDataChunk: Bool = false
    ) -> Data {
        var d = Data()

        let blockAlign = numChannels * (bitsPerSample / 8)
        let byteRate = sampleRate * UInt32(blockAlign)
        let sampleBytes = samples.count * 2

        let dataChunkSize = overrideDataChunkSize ?? UInt32(sampleBytes)

        // "fmt " sub-chunk: 16 bytes
        var fmtChunk = Data()
        fmtChunk.append(contentsOf: withUnsafeBytes(of: audioFormat.littleEndian) { Array($0) })
        fmtChunk.append(contentsOf: withUnsafeBytes(of: numChannels.littleEndian) { Array($0) })
        fmtChunk.append(contentsOf: withUnsafeBytes(of: sampleRate.littleEndian) { Array($0) })
        fmtChunk.append(contentsOf: withUnsafeBytes(of: byteRate.littleEndian) { Array($0) })
        fmtChunk.append(contentsOf: withUnsafeBytes(of: blockAlign.littleEndian) { Array($0) })
        fmtChunk.append(contentsOf: withUnsafeBytes(of: bitsPerSample.littleEndian) { Array($0) })

        // Optional extra chunk before data (e.g. "LIST" or "INFO")
        let extraChunkData = appendExtraChunk ?? Data()

        // data sub-chunk
        var dataChunk = Data()
        dataChunk.append("data".data(using: .ascii)!)
        dataChunk.append(contentsOf: withUnsafeBytes(of: dataChunkSize.littleEndian) { Array($0) })
        if !corruptDataChunk {
            for s in samples {
                dataChunk.append(contentsOf: withUnsafeBytes(of: s.littleEndian) { Array($0) })
            }
        }

        let totalSize = UInt32(4 + 8 + fmtChunk.count + extraChunkData.count + dataChunk.count)

        // RIFF header
        d.append("RIFF".data(using: .ascii)!)
        d.append(contentsOf: withUnsafeBytes(of: totalSize.littleEndian) { Array($0) })
        d.append("WAVE".data(using: .ascii)!)

        // fmt  chunk
        d.append("fmt ".data(using: .ascii)!)
        let fmtSize = UInt32(fmtChunk.count)
        d.append(contentsOf: withUnsafeBytes(of: fmtSize.littleEndian) { Array($0) })
        d.append(fmtChunk)

        // Extra chunk
        d.append(extraChunkData)

        // data chunk
        d.append(dataChunk)

        return d
    }

    private func writeTempWAV(_ data: Data) throws -> URL {
        let tmpDir = FileManager.default.temporaryDirectory
        let url = tmpDir.appendingPathComponent("test_\(UUID().uuidString).wav")
        try data.write(to: url)
        addTeardownBlock { try? FileManager.default.removeItem(at: url) }
        return url
    }

    // MARK: - Valid WAV

    func testValidMonoWAV() throws {
        let wavData = buildWAV()
        let url = try writeTempWAV(wavData)
        let (samples, rate) = try AudioFileLoader.loadWAV(url: url)
        XCTAssertEqual(rate, 16000)
        XCTAssertEqual(samples.count, 8)
    }

    func testValidStereoWAV() throws {
        // 4 stereo frames = 8 Int16 samples
        let wavData = buildWAV(numChannels: 2, samples: [100, -100, 200, -200, 300, -300, 400, -400])
        let url = try writeTempWAV(wavData)
        let (samples, rate) = try AudioFileLoader.loadWAV(url: url)
        XCTAssertEqual(rate, 16000)
        XCTAssertEqual(samples.count, 4, "Should have 4 mono frames from 4 stereo frames")
        // First channel values
        XCTAssertEqual(samples[0], Float(100) / 32768.0, accuracy: 0.0001)
        XCTAssertEqual(samples[1], Float(200) / 32768.0, accuracy: 0.0001)
    }

    // MARK: - Truncated / Too Small

    func testTooSmallFile() throws {
        let url = try writeTempWAV(Data(repeating: 0, count: 20))
        XCTAssertThrowsError(try AudioFileLoader.loadWAV(url: url)) { error in
            XCTAssertTrue(error is AudioLoadError)
        }
    }

    func testMissingRIFFHeader() throws {
        var data = buildWAV()
        data.replaceSubrange(0..<4, with: "NOPE".data(using: .ascii)!)
        let url = try writeTempWAV(data)
        XCTAssertThrowsError(try AudioFileLoader.loadWAV(url: url))
    }

    func testMissingWAVEFormat() throws {
        var data = buildWAV()
        data.replaceSubrange(8..<12, with: "NOPE".data(using: .ascii)!)
        let url = try writeTempWAV(data)
        XCTAssertThrowsError(try AudioFileLoader.loadWAV(url: url))
    }

    // MARK: - Zero Channels

    func testZeroChannelsRejected() throws {
        let wavData = buildWAV(numChannels: 0)
        let url = try writeTempWAV(wavData)
        XCTAssertThrowsError(try AudioFileLoader.loadWAV(url: url)) { error in
            XCTAssertTrue(error is AudioLoadError)
        }
    }

    // MARK: - Oversized Data Chunk

    func testOversizedDataChunkSize() throws {
        // Data chunk claims 99999 bytes but file only has 16 bytes of samples
        let wavData = buildWAV(overrideDataChunkSize: 99999)
        let url = try writeTempWAV(wavData)
        XCTAssertThrowsError(try AudioFileLoader.loadWAV(url: url)) { error in
            XCTAssertTrue(error is AudioLoadError, "Should reject oversized data chunk")
        }
    }

    // MARK: - Missing Data Chunk

    func testNoDataChunk() throws {
        // Build a WAV but replace the "data" marker with something else
        var wavData = buildWAV()
        // Find "data" and replace with "xxxx"
        if let range = wavData.range(of: "data".data(using: .ascii)!, in: 36..<wavData.count) {
            wavData.replaceSubrange(range, with: "xxxx".data(using: .ascii)!)
        }
        let url = try writeTempWAV(wavData)
        XCTAssertThrowsError(try AudioFileLoader.loadWAV(url: url)) { error in
            XCTAssertTrue(error is AudioLoadError, "Should reject WAV without data chunk")
        }
    }

    // MARK: - Chunk with oversized size field (overflow protection)

    func testExtraChunkWithHugeSize() throws {
        // Build an extra chunk between fmt and data that claims a huge size
        var extraChunk = Data()
        extraChunk.append("LIST".data(using: .ascii)!)
        let hugeSize: UInt32 = 0xFFFFFFFF
        extraChunk.append(contentsOf: withUnsafeBytes(of: hugeSize.littleEndian) { Array($0) })
        // Only 4 bytes of actual content (way less than claimed)
        extraChunk.append(contentsOf: [0, 0, 0, 0])

        let wavData = buildWAV(appendExtraChunk: extraChunk)
        let url = try writeTempWAV(wavData)
        XCTAssertThrowsError(try AudioFileLoader.loadWAV(url: url)) { error in
            XCTAssertTrue(error is AudioLoadError, "Should reject chunk with oversized size")
        }
    }

    // MARK: - Samples constrained to dataChunkSize

    func testSamplesConstrainedToChunkSize() throws {
        // Claim only 4 bytes in data chunk (2 samples for mono) but actually have 16 bytes
        let wavData = buildWAV(overrideDataChunkSize: 4)
        let url = try writeTempWAV(wavData)
        // Should not crash; chunkSize says 4 bytes = 2 samples
        let (samples, _) = try AudioFileLoader.loadWAV(url: url)
        XCTAssertEqual(samples.count, 2, "Should only read samples up to claimed chunk size")
    }
}

// MARK: - Download Path Security Tests

final class DownloadSecurityTests: XCTestCase {

    // MARK: - sanitizedCacheKey

    func testSanitizedCacheKeyNormal() {
        let key = Qwen3ASRModel.sanitizedCacheKey(for: "aufklarer/Qwen3-ASR-0.6B-MLX-4bit")
        XCTAssertEqual(key, "aufklarer_Qwen3-ASR-0.6B-MLX-4bit")
    }

    func testSanitizedCacheKeyMatchesExistingCache() throws {
        // Verify the new sanitizedCacheKey produces a key compatible with the
        // existing cached model directory (previously used plain replacingOccurrences)
        let modelId = "aufklarer/Qwen3-ASR-0.6B-MLX-4bit"
        let key = Qwen3ASRModel.sanitizedCacheKey(for: modelId)
        let legacyKey = modelId.replacingOccurrences(of: "/", with: "_")
        XCTAssertEqual(key, legacyKey, "New sanitized key should match legacy format for standard model IDs")

        // Check the cached directory actually exists if model was downloaded
        let cacheDir = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first!
            .appendingPathComponent("qwen3-asr")
            .appendingPathComponent(key)
        if FileManager.default.fileExists(atPath: cacheDir.path) {
            // Verify expected files are present
            let vocabPath = cacheDir.appendingPathComponent("vocab.json")
            XCTAssertTrue(FileManager.default.fileExists(atPath: vocabPath.path), "vocab.json should exist in cache")

            // Verify tokenizer loads from the cached path
            let tokenizer = Qwen3Tokenizer()
            try tokenizer.load(from: vocabPath)
            XCTAssertEqual(tokenizer.getTokenId(for: "<|im_start|>"), 151644)

            // Verify file validation passes on real cached files
            let contents = try FileManager.default.contentsOfDirectory(at: cacheDir, includingPropertiesForKeys: nil)
            for file in contents {
                let name = file.lastPathComponent
                XCTAssertNoThrow(try Qwen3ASRModel.validatedRemoteFileName(name),
                    "Cached file '\(name)' should pass validation")
                XCTAssertNoThrow(try Qwen3ASRModel.validatedLocalPath(directory: cacheDir, fileName: name),
                    "Cached file '\(name)' should pass local path validation")
            }
        }
    }

    func testSanitizedCacheKeyPathTraversal() {
        let key = Qwen3ASRModel.sanitizedCacheKey(for: "../../etc/passwd")
        XCTAssertFalse(key.contains(".."), "Should strip path traversal")
        XCTAssertFalse(key.contains("/"), "Should not contain path separators")
    }

    func testSanitizedCacheKeySlashesRemoved() {
        let key = Qwen3ASRModel.sanitizedCacheKey(for: "org/model/variant")
        XCTAssertFalse(key.contains("/"))
        XCTAssertTrue(key.contains("org_model_variant"))
    }

    func testSanitizedCacheKeySpecialChars() {
        let key = Qwen3ASRModel.sanitizedCacheKey(for: "model with spaces & $pecial!")
        // Should only contain allowed chars
        let allowed = CharacterSet(charactersIn: "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
        for scalar in key.unicodeScalars {
            XCTAssertTrue(allowed.contains(scalar), "Unexpected char: \(scalar)")
        }
    }

    func testSanitizedCacheKeyEmpty() {
        let key = Qwen3ASRModel.sanitizedCacheKey(for: "")
        XCTAssertEqual(key, "model", "Empty input should fall back to 'model'")
    }

    func testSanitizedCacheKeyDotDot() {
        let key = Qwen3ASRModel.sanitizedCacheKey(for: "..")
        XCTAssertEqual(key, "model", "'..' should fall back to 'model'")
    }

    func testSanitizedCacheKeyLeadingDots() {
        let key = Qwen3ASRModel.sanitizedCacheKey(for: "...hidden")
        XCTAssertFalse(key.hasPrefix("."), "Should trim leading dots")
    }

    // MARK: - validatedRemoteFileName

    func testValidFileNameAccepted() throws {
        let result = try Qwen3ASRModel.validatedRemoteFileName("model.safetensors")
        XCTAssertEqual(result, "model.safetensors")
    }

    func testValidFileNameWithDashes() throws {
        let result = try Qwen3ASRModel.validatedRemoteFileName("model-00001-of-00002.safetensors")
        XCTAssertEqual(result, "model-00001-of-00002.safetensors")
    }

    func testFileNameWithPathSeparatorRejected() {
        XCTAssertThrowsError(try Qwen3ASRModel.validatedRemoteFileName("../etc/passwd")) { error in
            XCTAssertTrue(error is DownloadError)
        }
    }

    func testFileNameWithSlashRejected() {
        XCTAssertThrowsError(try Qwen3ASRModel.validatedRemoteFileName("sub/file.bin")) { error in
            XCTAssertTrue(error is DownloadError)
        }
    }

    func testHiddenFileRejected() {
        XCTAssertThrowsError(try Qwen3ASRModel.validatedRemoteFileName(".hidden")) { error in
            XCTAssertTrue(error is DownloadError)
        }
    }

    func testEmptyFileNameRejected() {
        XCTAssertThrowsError(try Qwen3ASRModel.validatedRemoteFileName("")) { error in
            XCTAssertTrue(error is DownloadError)
        }
    }

    func testFileNameWithSpacesRejected() {
        XCTAssertThrowsError(try Qwen3ASRModel.validatedRemoteFileName("model weights.bin")) { error in
            XCTAssertTrue(error is DownloadError)
        }
    }

    func testFileNameDotDotRejected() {
        XCTAssertThrowsError(try Qwen3ASRModel.validatedRemoteFileName("..")) { error in
            XCTAssertTrue(error is DownloadError)
        }
    }

    // MARK: - validatedLocalPath

    func testValidLocalPath() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("test-cache")
        let result = try Qwen3ASRModel.validatedLocalPath(directory: dir, fileName: "model.safetensors")
        XCTAssertTrue(result.path.hasSuffix("model.safetensors"))
        XCTAssertTrue(result.path.hasPrefix(dir.path))
    }

    func testLocalPathTraversalRejected() {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("test-cache")
        // This shouldn't happen if validatedRemoteFileName is called first,
        // but validatedLocalPath is a defense-in-depth check
        XCTAssertThrowsError(try Qwen3ASRModel.validatedLocalPath(directory: dir, fileName: "../escape.txt")) { error in
            XCTAssertTrue(error is DownloadError)
        }
    }
}

// MARK: - Metallib Build Script Tests

final class MetallibScriptTests: XCTestCase {

    func testScriptExists() {
        let scriptPath = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()  // Tests/Qwen3ASRTests/
            .deletingLastPathComponent()  // Tests/
            .deletingLastPathComponent()  // repo root
            .appendingPathComponent("scripts/build_mlx_metallib.sh")
        XCTAssertTrue(
            FileManager.default.fileExists(atPath: scriptPath.path),
            "build_mlx_metallib.sh should exist at \(scriptPath.path)"
        )
    }

    func testScriptIsExecutable() {
        let scriptPath = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("scripts/build_mlx_metallib.sh")
        let attrs = try? FileManager.default.attributesOfItem(atPath: scriptPath.path)
        let perms = (attrs?[.posixPermissions] as? NSNumber)?.intValue ?? 0
        XCTAssertTrue(perms & 0o111 != 0, "Script should be executable")
    }

    func testScriptRejectsInvalidConfig() throws {
        let scriptPath = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("scripts/build_mlx_metallib.sh")
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/bash")
        process.arguments = [scriptPath.path, "invalid_config"]
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe
        try process.run()
        process.waitUntilExit()
        XCTAssertEqual(process.terminationStatus, 2, "Should exit 2 for invalid config argument")
    }

    func testScriptHasProperShebang() throws {
        let scriptPath = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("scripts/build_mlx_metallib.sh")
        let content = try String(contentsOfFile: scriptPath.path, encoding: .utf8)
        XCTAssertTrue(content.hasPrefix("#!/usr/bin/env bash"), "Script should have bash shebang")
        XCTAssertTrue(content.contains("set -euo pipefail"), "Script should use strict mode")
    }
}
