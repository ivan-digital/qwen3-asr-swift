import XCTest
import MLX
@testable import Qwen3TTS
@testable import AudioCommon

final class MemoryManagementTests: XCTestCase {

    func testQwen3TTSUnload() {
        let model = Qwen3TTSModel()
        XCTAssertTrue(model.isLoaded)
        XCTAssertGreaterThan(model.memoryFootprint, 0)

        model.unload()
        XCTAssertFalse(model.isLoaded)
        XCTAssertEqual(model.memoryFootprint, 0)
    }

    func testUnloadIdempotent() {
        let model = Qwen3TTSModel()
        model.unload()
        model.unload()  // second call should not crash
        XCTAssertFalse(model.isLoaded)
    }
}
