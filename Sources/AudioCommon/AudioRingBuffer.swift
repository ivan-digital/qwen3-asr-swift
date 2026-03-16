import Foundation

/// Thread-safe ring buffer for passing audio between the audio capture thread and the MLX
/// inference thread. Writes drop oldest data when full; reads return zeros on underrun.
public final class AudioRingBuffer: @unchecked Sendable {
    private var buffer: [Float]
    private var readPos = 0
    private var writePos = 0
    private var count = 0
    private let lock = NSLock()
    private let capacity: Int

    public init(capacity: Int) {
        self.capacity = capacity
        self.buffer = [Float](repeating: 0, count: capacity)
    }

    /// Called from audio capture thread — non-blocking; drops oldest data if full.
    public func write(_ samples: [Float]) {
        lock.lock()
        defer { lock.unlock() }
        for sample in samples {
            if count == capacity {
                // Drop oldest sample
                readPos = (readPos + 1) % capacity
                count -= 1
            }
            buffer[writePos] = sample
            writePos = (writePos + 1) % capacity
            count += 1
        }
    }

    /// Called from MLX inference thread — returns zeros on underrun; never blocks.
    public func read(_ n: Int) -> [Float] {
        lock.lock()
        defer { lock.unlock() }
        var result = [Float](repeating: 0, count: n)
        let available = min(n, count)
        for i in 0..<available {
            result[i] = buffer[(readPos + i) % capacity]
        }
        readPos = (readPos + available) % capacity
        count -= available
        // Remaining positions stay as zero (underrun padding)
        return result
    }

    /// Number of samples currently available to read.
    public var available: Int {
        lock.lock()
        defer { lock.unlock() }
        return count
    }
}
