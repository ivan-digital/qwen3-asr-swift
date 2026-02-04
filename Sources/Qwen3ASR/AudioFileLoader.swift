import Foundation
import AVFoundation

/// Loads audio files and converts to float samples
public enum AudioFileLoader {
    /// Load audio file and return samples at target sample rate
    public static func load(url: URL, targetSampleRate: Int = 24000) throws -> [Float] {
        let audioFile = try AVAudioFile(forReading: url)
        let format = audioFile.processingFormat
        let frameCount = AVAudioFrameCount(audioFile.length)

        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else {
            throw AudioLoadError.bufferCreationFailed
        }

        try audioFile.read(into: buffer)

        guard let floatData = buffer.floatChannelData else {
            throw AudioLoadError.noFloatData
        }

        // Get mono samples (use first channel)
        let samples = Array(UnsafeBufferPointer(start: floatData[0], count: Int(buffer.frameLength)))

        // Resample if needed
        let inputSampleRate = Int(format.sampleRate)
        if inputSampleRate != targetSampleRate {
            return resample(samples, from: inputSampleRate, to: targetSampleRate)
        }

        return samples
    }

    /// Load WAV file directly (for 16-bit PCM)
    public static func loadWAV(url: URL) throws -> (samples: [Float], sampleRate: Int) {
        let data = try Data(contentsOf: url)

        // Parse WAV header
        guard data.count > 44 else {
            throw AudioLoadError.invalidWAVFile
        }

        // Check RIFF header
        let riff = String(data: data[0..<4], encoding: .ascii)
        guard riff == "RIFF" else {
            throw AudioLoadError.invalidWAVFile
        }

        // Check WAVE format
        let wave = String(data: data[8..<12], encoding: .ascii)
        guard wave == "WAVE" else {
            throw AudioLoadError.invalidWAVFile
        }

        // Parse format chunk (handle unaligned reads)
        let audioFormat = data[20..<22].withUnsafeBytes { $0.loadUnaligned(as: UInt16.self) }
        let numChannels = data[22..<24].withUnsafeBytes { $0.loadUnaligned(as: UInt16.self) }
        let sampleRate = data[24..<28].withUnsafeBytes { $0.loadUnaligned(as: UInt32.self) }
        let bitsPerSample = data[34..<36].withUnsafeBytes { $0.loadUnaligned(as: UInt16.self) }

        guard audioFormat == 1 else { // PCM
            throw AudioLoadError.unsupportedFormat("Not PCM format")
        }

        guard bitsPerSample == 16 else {
            throw AudioLoadError.unsupportedFormat("Not 16-bit")
        }

        // Find data chunk
        var dataOffset = 36
        while dataOffset < data.count - 8 {
            let chunkId = String(data: data[dataOffset..<(dataOffset+4)], encoding: .ascii)
            let chunkSize = data[(dataOffset+4)..<(dataOffset+8)].withUnsafeBytes { $0.loadUnaligned(as: UInt32.self) }

            if chunkId == "data" {
                dataOffset += 8
                break
            }
            dataOffset += 8 + Int(chunkSize)
        }

        // Read samples
        let sampleData = data[dataOffset...]
        let sampleCount = sampleData.count / 2 / Int(numChannels)

        var samples = [Float](repeating: 0, count: sampleCount)
        sampleData.withUnsafeBytes { ptr in
            let int16Ptr = ptr.bindMemory(to: Int16.self)
            for i in 0..<sampleCount {
                // Take first channel only
                let sampleIndex = i * Int(numChannels)
                if sampleIndex < int16Ptr.count {
                    samples[i] = Float(int16Ptr[sampleIndex]) / 32768.0
                }
            }
        }

        return (samples, Int(sampleRate))
    }

    /// Simple linear resampling
    private static func resample(_ samples: [Float], from inputRate: Int, to outputRate: Int) -> [Float] {
        let ratio = Double(outputRate) / Double(inputRate)
        let outputLength = Int(Double(samples.count) * ratio)

        guard outputLength > 0 else { return [] }

        var output = [Float](repeating: 0, count: outputLength)

        for i in 0..<outputLength {
            let srcIndex = Double(i) / ratio
            let srcIndexFloor = Int(srcIndex)
            let srcIndexCeil = min(srcIndexFloor + 1, samples.count - 1)
            let fraction = Float(srcIndex - Double(srcIndexFloor))

            output[i] = samples[srcIndexFloor] * (1 - fraction) + samples[srcIndexCeil] * fraction
        }

        return output
    }
}

public enum AudioLoadError: Error, LocalizedError {
    case bufferCreationFailed
    case noFloatData
    case invalidWAVFile
    case unsupportedFormat(String)

    public var errorDescription: String? {
        switch self {
        case .bufferCreationFailed:
            return "Failed to create audio buffer"
        case .noFloatData:
            return "No float channel data available"
        case .invalidWAVFile:
            return "Invalid WAV file format"
        case .unsupportedFormat(let reason):
            return "Unsupported audio format: \(reason)"
        }
    }
}
