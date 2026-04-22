import XCTest
@testable import AudioCommon

final class ModelSourceTests: XCTestCase {

    func testQwen3CacheDirPath() throws {
        // Test that QWEN3_CACHE_DIR doesn't create nested directories
        let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent("test-cache-\(UUID().uuidString)")

        // Set environment variable
        setenv("QWEN3_CACHE_DIR", tempDir.path, 1)

        // Get cache directory for a model
        let cacheDir = try HuggingFaceDownloader.getCacheDirectory(for: "test/model")

        // Should be tempDir/models/test/model, NOT tempDir/qwen3-speech/models/test/model
        XCTAssertTrue(cacheDir.path.hasPrefix(tempDir.path))
        XCTAssertFalse(cacheDir.path.contains("qwen3-speech/qwen3-speech"))

        // Clean up
        unsetenv("QWEN3_CACHE_DIR")
        try? FileManager.default.removeItem(at: tempDir)
    }

    func testModelScopeMapping() {
        // Test model ID mapping from HuggingFace to ModelScope
        XCTAssertEqual(
            ModelScopeDownloader.mapToModelScopeId("aufklarer/Qwen3-ASR-0.6B-MLX-4bit"),
            "Qwen/Qwen3-ASR-0.6B-MLX-4bit"
        )

        XCTAssertEqual(
            ModelScopeDownloader.mapToModelScopeId("aufklarer/Qwen3-ASR-1.7B-MLX-8bit"),
            "Qwen/Qwen3-ASR-1.7B-MLX-8bit"
        )

        // Test non-Qwen model
        XCTAssertEqual(
            ModelScopeDownloader.mapToModelScopeId("pyannote/segmentation-3.0"),
            "pyannote/segmentation-3.0"
        )
    }

    func testModelSourceSelection() {
        // Default should be HuggingFace
        XCTAssertEqual(getModelSource(), .huggingFace)

        // Test ModelScope selection
        setenv("QWEN3_MODEL_SOURCE", "modelscope", 1)
        XCTAssertEqual(getModelSource(), .modelScope)

        // Clean up
        unsetenv("QWEN3_MODEL_SOURCE")
    }

    func testUnifiedDownloadFunctionExists() {
        // Verify the unified download function exists and can be called
        // (We won't actually download, just verify the method exists)
        XCTAssertNotNil(HuggingFaceDownloader.downloadWeightsWithSourceSelection)
    }
}