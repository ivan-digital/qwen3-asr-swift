import XCTest
@testable import SpeechUI

@MainActor
final class TranscriptionStoreTests: XCTestCase {

    func testInitiallyEmpty() {
        let store = TranscriptionStore()
        XCTAssertTrue(store.finalLines.isEmpty)
        XCTAssertNil(store.currentPartial)
    }

    func testPartialUpdatesPartialOnly() {
        let store = TranscriptionStore()
        store.apply(text: "hello", isFinal: false)
        XCTAssertEqual(store.currentPartial, "hello")
        XCTAssertTrue(store.finalLines.isEmpty)

        store.apply(text: "hello world", isFinal: false)
        XCTAssertEqual(store.currentPartial, "hello world")
        XCTAssertTrue(store.finalLines.isEmpty)
    }

    func testFinalCommitsAndClearsPartial() {
        let store = TranscriptionStore()
        store.apply(text: "hello world", isFinal: false)
        store.apply(text: "hello world", isFinal: true)

        XCTAssertEqual(store.finalLines, ["hello world"])
        XCTAssertNil(store.currentPartial)
    }

    func testMultipleFinalsAccumulate() {
        let store = TranscriptionStore()
        store.apply(text: "first sentence", isFinal: true)
        store.apply(text: "second sentence", isFinal: true)
        store.apply(text: "third sentence", isFinal: true)

        XCTAssertEqual(store.finalLines, ["first sentence", "second sentence", "third sentence"])
        XCTAssertNil(store.currentPartial)
    }

    func testEmptyFinalIsIgnored() {
        let store = TranscriptionStore()
        store.apply(text: "   ", isFinal: true)
        XCTAssertTrue(store.finalLines.isEmpty)
        XCTAssertNil(store.currentPartial)
    }

    func testEmptyPartialClearsPartialState() {
        let store = TranscriptionStore()
        store.apply(text: "hello", isFinal: false)
        store.apply(text: "", isFinal: false)
        XCTAssertNil(store.currentPartial)
    }

    func testWhitespaceTrimming() {
        let store = TranscriptionStore()
        store.apply(text: "  hello  ", isFinal: true)
        XCTAssertEqual(store.finalLines, ["hello"])
    }

    func testReset() {
        let store = TranscriptionStore()
        store.apply(text: "first", isFinal: true)
        store.apply(text: "in progress", isFinal: false)

        store.reset()

        XCTAssertTrue(store.finalLines.isEmpty)
        XCTAssertNil(store.currentPartial)
    }

    func testInterleavedPartialsAndFinals() {
        let store = TranscriptionStore()
        store.apply(text: "first", isFinal: false)
        store.apply(text: "first sentence", isFinal: true)
        store.apply(text: "sec", isFinal: false)
        store.apply(text: "second", isFinal: false)
        store.apply(text: "second sentence", isFinal: true)

        XCTAssertEqual(store.finalLines, ["first sentence", "second sentence"])
        XCTAssertNil(store.currentPartial)
    }
}
