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
