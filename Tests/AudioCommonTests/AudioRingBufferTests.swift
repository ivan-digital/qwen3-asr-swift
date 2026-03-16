import XCTest
@testable import AudioCommon

final class AudioRingBufferTests: XCTestCase {

    // MARK: Basic write / read

    func testWriteAndRead() {
        let buf = AudioRingBuffer(capacity: 8)
        buf.write([1, 2, 3, 4])
        XCTAssertEqual(buf.available, 4)
        let result = buf.read(4)
        XCTAssertEqual(result, [1, 2, 3, 4])
        XCTAssertEqual(buf.available, 0)
    }

    func testPartialRead() {
        let buf = AudioRingBuffer(capacity: 8)
        buf.write([10, 20, 30, 40])
        let first = buf.read(2)
        XCTAssertEqual(first, [10, 20])
        XCTAssertEqual(buf.available, 2)
        let second = buf.read(2)
        XCTAssertEqual(second, [30, 40])
        XCTAssertEqual(buf.available, 0)
    }

    // MARK: Underrun

    func testReadUnderrunReturnsPaddedZeros() {
        let buf = AudioRingBuffer(capacity: 8)
        buf.write([5, 6])
        let result = buf.read(4)
        XCTAssertEqual(result, [5, 6, 0, 0])
        XCTAssertEqual(buf.available, 0)
    }

    func testReadFromEmptyBufferReturnsZeros() {
        let buf = AudioRingBuffer(capacity: 8)
        let result = buf.read(4)
        XCTAssertEqual(result, [0, 0, 0, 0])
    }

    // MARK: Overwrite

    func testOverwriteDropsOldest() {
        let buf = AudioRingBuffer(capacity: 4)
        buf.write([1, 2, 3, 4])
        buf.write([5])
        XCTAssertEqual(buf.available, 4)
        let result = buf.read(4)
        XCTAssertEqual(result, [2, 3, 4, 5])
    }

    func testOverwriteByLargeBlockDropsCorrectly() {
        let buf = AudioRingBuffer(capacity: 4)
        buf.write([1, 2, 3, 4])
        buf.write([5, 6, 7, 8])
        XCTAssertEqual(buf.available, 4)
        let result = buf.read(4)
        XCTAssertEqual(result, [5, 6, 7, 8])
    }

    // MARK: Wrap-around

    func testWriteAndReadAcrossBufferBoundary() {
        let buf = AudioRingBuffer(capacity: 4)
        buf.write([1, 2, 3])
        _ = buf.read(2)
        buf.write([4, 5])
        XCTAssertEqual(buf.available, 3)
        let result = buf.read(3)
        XCTAssertEqual(result, [3, 4, 5])
    }

    func testMultipleWriteReadCycles() {
        let buf = AudioRingBuffer(capacity: 4)
        for i in 0..<10 {
            let sample = Float(i)
            buf.write([sample])
            let result = buf.read(1)
            XCTAssertEqual(result, [sample], "cycle \(i)")
        }
        XCTAssertEqual(buf.available, 0)
    }

    // MARK: Thread safety

    func testConcurrentWriteAndRead() {
        let buf = AudioRingBuffer(capacity: 1024)
        let iterations = 10_000
        let expectation = XCTestExpectation(description: "concurrent")
        expectation.expectedFulfillmentCount = 2

        DispatchQueue.global().async {
            for i in 0..<iterations { buf.write([Float(i % 128)]) }
            expectation.fulfill()
        }
        DispatchQueue.global().async {
            for _ in 0..<iterations { _ = buf.read(1) }
            expectation.fulfill()
        }

        wait(for: [expectation], timeout: 10)
        XCTAssertGreaterThanOrEqual(buf.available, 0)
        XCTAssertLessThanOrEqual(buf.available, 1024)
    }

    func testConcurrentMultipleWriters() {
        let buf = AudioRingBuffer(capacity: 512)
        let group = DispatchGroup()

        for _ in 0..<4 {
            group.enter()
            DispatchQueue.global().async {
                for i in 0..<1000 { buf.write([Float(i)]) }
                group.leave()
            }
        }
        group.wait()
        XCTAssertGreaterThanOrEqual(buf.available, 0)
        XCTAssertLessThanOrEqual(buf.available, 512)
    }
}
