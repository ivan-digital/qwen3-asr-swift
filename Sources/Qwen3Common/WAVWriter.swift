import Foundation

/// Write float audio samples to WAV file
public enum WAVWriter {

    /// Write mono float samples to a 16-bit PCM WAV file
    /// - Parameters:
    ///   - samples: Float audio samples in [-1.0, 1.0] range
    ///   - sampleRate: Sample rate in Hz (default 24000)
    ///   - url: Output file URL
    public static func write(samples: [Float], sampleRate: Int = 24000, to url: URL) throws {
        let numChannels: UInt16 = 1
        let bitsPerSample: UInt16 = 16
        let bytesPerSample = Int(bitsPerSample) / 8
        let dataSize = samples.count * bytesPerSample
        let fileSize = 36 + dataSize

        var data = Data(capacity: fileSize + 8)

        // RIFF header
        data.append(contentsOf: "RIFF".utf8)
        appendUInt32(&data, UInt32(fileSize))
        data.append(contentsOf: "WAVE".utf8)

        // fmt chunk
        data.append(contentsOf: "fmt ".utf8)
        appendUInt32(&data, 16)                         // chunk size
        appendUInt16(&data, 1)                          // PCM format
        appendUInt16(&data, numChannels)
        appendUInt32(&data, UInt32(sampleRate))
        appendUInt32(&data, UInt32(sampleRate * Int(numChannels) * bytesPerSample))  // byte rate
        appendUInt16(&data, numChannels * UInt16(bytesPerSample))  // block align
        appendUInt16(&data, bitsPerSample)

        // data chunk
        data.append(contentsOf: "data".utf8)
        appendUInt32(&data, UInt32(dataSize))

        // Convert float samples to 16-bit PCM
        for sample in samples {
            let clamped = max(-1.0, min(1.0, sample))
            let int16Value = Int16(clamped * 32767.0)
            appendInt16(&data, int16Value)
        }

        try data.write(to: url)
    }

    private static func appendUInt32(_ data: inout Data, _ value: UInt32) {
        var v = value.littleEndian
        data.append(Data(bytes: &v, count: 4))
    }

    private static func appendUInt16(_ data: inout Data, _ value: UInt16) {
        var v = value.littleEndian
        data.append(Data(bytes: &v, count: 2))
    }

    private static func appendInt16(_ data: inout Data, _ value: Int16) {
        var v = value.littleEndian
        data.append(Data(bytes: &v, count: 2))
    }
}

// MARK: - Streaming WAV Writer

/// Incremental WAV writer that appends samples to disk as they arrive.
/// Writes a placeholder header on init, appends PCM data via write(), then
/// seeks back to update the header with final sizes on finalize().
public final class StreamingWAVWriter {
    public struct Result {
        public let sampleCount: Int
    }

    private let url: URL
    private let sampleRate: Int
    private let fileHandle: FileHandle
    public private(set) var sampleCount: Int = 0

    public init(to url: URL, sampleRate: Int = 24000) throws {
        self.url = url
        self.sampleRate = sampleRate

        // Write placeholder 44-byte WAV header
        let header = Data(count: 44)
        try header.write(to: url)

        self.fileHandle = try FileHandle(forWritingTo: url)
        fileHandle.seekToEndOfFile()
    }

    /// Append audio samples to the file.
    public func write(samples: [Float]) {
        var data = Data(capacity: samples.count * 2)
        for sample in samples {
            let clamped = max(-1.0, min(1.0, sample))
            let int16 = Int16(clamped * 32767.0)
            var v = int16.littleEndian
            withUnsafeBytes(of: &v) { data.append(contentsOf: $0) }
        }
        fileHandle.write(data)
        sampleCount += samples.count
    }

    /// Finalize the file by writing the correct WAV header with actual data sizes.
    @discardableResult
    public func finalize() -> Result {
        let numChannels: UInt16 = 1
        let bitsPerSample: UInt16 = 16
        let bytesPerSample = Int(bitsPerSample) / 8
        let dataSize = UInt32(sampleCount * bytesPerSample)
        let fileSize = UInt32(36) + dataSize

        var header = Data(capacity: 44)

        // RIFF header
        header.append(contentsOf: "RIFF".utf8)
        var fs = fileSize.littleEndian
        withUnsafeBytes(of: &fs) { header.append(contentsOf: $0) }
        header.append(contentsOf: "WAVE".utf8)

        // fmt chunk
        header.append(contentsOf: "fmt ".utf8)
        var chunkSize = UInt32(16).littleEndian
        withUnsafeBytes(of: &chunkSize) { header.append(contentsOf: $0) }
        var pcmFormat = UInt16(1).littleEndian
        withUnsafeBytes(of: &pcmFormat) { header.append(contentsOf: $0) }
        var nc = numChannels.littleEndian
        withUnsafeBytes(of: &nc) { header.append(contentsOf: $0) }
        var sr = UInt32(sampleRate).littleEndian
        withUnsafeBytes(of: &sr) { header.append(contentsOf: $0) }
        var byteRate = UInt32(sampleRate * Int(numChannels) * bytesPerSample).littleEndian
        withUnsafeBytes(of: &byteRate) { header.append(contentsOf: $0) }
        var blockAlign = (numChannels * UInt16(bytesPerSample)).littleEndian
        withUnsafeBytes(of: &blockAlign) { header.append(contentsOf: $0) }
        var bps = bitsPerSample.littleEndian
        withUnsafeBytes(of: &bps) { header.append(contentsOf: $0) }

        // data chunk
        header.append(contentsOf: "data".utf8)
        var ds = dataSize.littleEndian
        withUnsafeBytes(of: &ds) { header.append(contentsOf: $0) }

        fileHandle.seek(toFileOffset: 0)
        fileHandle.write(header)
        fileHandle.closeFile()

        return Result(sampleCount: sampleCount)
    }
}
