import Foundation
import MLXCommon
import AudioCommon

/// Weight loading for DeepFilterNet3 Core ML model.
///
/// Loads the .mlpackage model and auxiliary signal processing data (ERB filterbank,
/// Vorbis window, normalization states) from an npz file.
enum DeepFilterNet3WeightLoader {

    /// Load Core ML model and auxiliary data.
    ///
    /// - Parameters:
    ///   - directory: Directory containing DeepFilterNet3.mlpackage and auxiliary.npz
    /// - Returns: `(network, auxiliaryData)`
    static func load(from directory: URL) throws -> (DeepFilterNet3Network, AuxiliaryData) {
        let modelURL = directory.appendingPathComponent("DeepFilterNet3.mlpackage")
        let auxURL = directory.appendingPathComponent("auxiliary.npz")

        guard FileManager.default.fileExists(atPath: modelURL.path) else {
            throw WeightLoadingError.noWeightsFound(directory)
        }
        guard FileManager.default.fileExists(atPath: auxURL.path) else {
            throw WeightLoadingError.missingRequiredWeight(
                "auxiliary.npz not found in \(directory.path)")
        }

        let network = try DeepFilterNet3Network(modelURL: modelURL)
        let auxData = try loadAuxiliaryData(from: auxURL)

        return (network, auxData)
    }

    /// Parse the auxiliary.npz file containing signal processing constants.
    private static func loadAuxiliaryData(from url: URL) throws -> AuxiliaryData {
        let arrays = try NpzReader.read(url: url)

        guard let erbFb = arrays["erb_fb"],
              let erbInvFb = arrays["erb_inv_fb"],
              let window = arrays["window"],
              let meanNormState = arrays["mean_norm_state"],
              let unitNormState = arrays["unit_norm_state"] else {
            throw WeightLoadingError.missingRequiredWeight(
                "Missing keys in auxiliary.npz")
        }

        return AuxiliaryData(
            erbFb: erbFb,
            erbInvFb: erbInvFb,
            window: window,
            meanNormState: meanNormState,
            unitNormState: unitNormState
        )
    }

    /// Auxiliary data loaded alongside Core ML model.
    struct AuxiliaryData {
        /// Forward ERB filterbank [freqBins, erbBands]
        let erbFb: [Float]
        /// Inverse ERB filterbank [erbBands, freqBins]
        let erbInvFb: [Float]
        /// Vorbis analysis/synthesis window [fftSize]
        let window: [Float]
        /// Initial mean normalization state [erbBands]
        let meanNormState: [Float]
        /// Initial unit normalization state [dfBins]
        let unitNormState: [Float]
    }
}

// MARK: - NPZ Reader

/// Minimal reader for NumPy .npz files (uncompressed, float32 only).
enum NpzReader {

    static func read(url: URL) throws -> [String: [Float]] {
        let data = try Data(contentsOf: url)
        var result = [String: [Float]]()

        // NPZ is a ZIP archive of .npy files
        var offset = 0
        while offset + 30 <= data.count {
            // ZIP local file header signature
            let b0 = data[offset], b1 = data[offset+1], b2 = data[offset+2], b3 = data[offset+3]
            guard b0 == 0x50 && b1 == 0x4b && b2 == 0x03 && b3 == 0x04 else { break }

            var compressedSize = Int(readUInt32(data, at: offset + 18))
            var uncompressedSize = Int(readUInt32(data, at: offset + 22))
            let nameLen = Int(readUInt16(data, at: offset + 26))
            let extraLen = Int(readUInt16(data, at: offset + 28))

            let nameStart = offset + 30
            guard nameStart + nameLen <= data.count else { break }
            let nameData = data.subdata(in: nameStart..<nameStart + nameLen)
            var name = String(data: nameData, encoding: .utf8) ?? ""

            // Strip .npy extension
            if name.hasSuffix(".npy") {
                name = String(name.dropLast(4))
            }

            // Handle ZIP64 extra field (sizes stored as UInt64 when 0xFFFFFFFF)
            if compressedSize == 0xFFFFFFFF || uncompressedSize == 0xFFFFFFFF {
                let extraStart = nameStart + nameLen
                if extraLen >= 4 {
                    let tag = readUInt16(data, at: extraStart)
                    if tag == 0x0001 {  // ZIP64 extended information
                        var extraOffset = extraStart + 4
                        if uncompressedSize == 0xFFFFFFFF {
                            uncompressedSize = Int(readUInt64(data, at: extraOffset))
                            extraOffset += 8
                        }
                        if compressedSize == 0xFFFFFFFF {
                            compressedSize = Int(readUInt64(data, at: extraOffset))
                        }
                    }
                }
            }

            let dataStart = nameStart + nameLen + extraLen
            let dataSize = max(compressedSize, uncompressedSize)
            guard dataStart + dataSize <= data.count else { break }

            if let floats = parseNpy(data, npyOffset: dataStart, npySize: uncompressedSize) {
                result[name] = floats
            }

            offset = dataStart + dataSize
        }

        return result
    }

    /// Parse a .npy record at the given offset in the data buffer.
    private static func parseNpy(_ data: Data, npyOffset: Int, npySize: Int) -> [Float]? {
        // Magic: \x93NUMPY
        guard npySize >= 10,
              data[npyOffset] == 0x93,
              data[npyOffset + 1] == 0x4E else { return nil }

        let majorVersion = data[npyOffset + 6]

        let headerLen: Int
        if majorVersion == 1 {
            headerLen = Int(readUInt16(data, at: npyOffset + 8))
        } else {
            headerLen = Int(readUInt32(data, at: npyOffset + 8))
        }

        let headerSize = (majorVersion == 1) ? 10 : 12
        let floatStart = npyOffset + headerSize + headerLen
        let numBytes = npySize - headerSize - headerLen

        guard numBytes > 0 else { return nil }
        let numFloats = numBytes / 4

        var result = [Float](repeating: 0, count: numFloats)
        _ = result.withUnsafeMutableBytes { dst in
            data.copyBytes(to: dst, from: floatStart..<floatStart + numFloats * 4)
        }
        return result
    }

    private static func readUInt16(_ data: Data, at offset: Int) -> UInt16 {
        data.withUnsafeBytes { raw in
            raw.loadUnaligned(fromByteOffset: offset - data.startIndex, as: UInt16.self)
        }
    }

    private static func readUInt32(_ data: Data, at offset: Int) -> UInt32 {
        data.withUnsafeBytes { raw in
            raw.loadUnaligned(fromByteOffset: offset - data.startIndex, as: UInt32.self)
        }
    }

    private static func readUInt64(_ data: Data, at offset: Int) -> UInt64 {
        data.withUnsafeBytes { raw in
            raw.loadUnaligned(fromByteOffset: offset - data.startIndex, as: UInt64.self)
        }
    }
}
