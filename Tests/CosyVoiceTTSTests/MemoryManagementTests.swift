import XCTest
import MLX
@testable import CosyVoiceTTS
@testable import AudioCommon

final class CosyVoiceMemoryTests: XCTestCase {

    func testCosyVoiceUnload() {
        let model = CosyVoiceTTSModel()
        XCTAssertTrue(model.isLoaded)
        XCTAssertGreaterThan(model.memoryFootprint, 0)

        model.unload()
        XCTAssertFalse(model.isLoaded)
        XCTAssertEqual(model.memoryFootprint, 0)
    }

    func testUnloadIdempotent() {
        let model = CosyVoiceTTSModel()
        model.unload()
        model.unload()
        XCTAssertFalse(model.isLoaded)
    }
}
