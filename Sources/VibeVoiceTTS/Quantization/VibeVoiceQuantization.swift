import Foundation
import MLX
import MLXNN

public enum VibeVoiceQuantizationMode: String, Codable, Sendable {
    case affine
    case mxfp4

    public var mlxMode: QuantizationMode {
        switch self {
        case .affine: return .affine
        case .mxfp4: return .mxfp4
        }
    }
}

public struct VibeVoiceQuantizationSpec: Codable, Sendable {
    public var groupSize: Int
    public var bits: Int
    public var mode: VibeVoiceQuantizationMode

    public init(groupSize: Int = 32, bits: Int = 8, mode: VibeVoiceQuantizationMode = .affine) {
        self.groupSize = groupSize
        self.bits = bits
        self.mode = mode
    }

    enum CodingKeys: String, CodingKey {
        case groupSize = "group_size"
        case bits
        case mode
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.groupSize = try container.decodeIfPresent(Int.self, forKey: .groupSize) ?? 32
        self.bits = try container.decodeIfPresent(Int.self, forKey: .bits) ?? 8
        let modeStr = try container.decodeIfPresent(String.self, forKey: .mode) ?? "affine"
        self.mode = VibeVoiceQuantizationMode(rawValue: modeStr) ?? .affine
    }
}

public struct VibeVoiceQuantizationManifest: Codable {
    public var modelId: String?
    public var revision: String?
    public var groupSize: Int
    public var bits: Int
    public var mode: String
    public var layers: [QuantizedLayerInfo]

    public struct QuantizedLayerInfo: Codable {
        public var name: String
        public var shape: [Int]
        public var inDim: Int
        public var outDim: Int
        public var file: String
        public var quantFile: String?
        public var groupSize: Int?
        public var bits: Int?
        public var mode: String?

        enum CodingKeys: String, CodingKey {
            case name
            case shape
            case inDim = "in_dim"
            case outDim = "out_dim"
            case file
            case quantFile = "quant_file"
            case groupSize = "group_size"
            case bits
            case mode
        }
    }

    enum CodingKeys: String, CodingKey {
        case modelId = "model_id"
        case revision
        case groupSize = "group_size"
        case bits
        case mode
        case layers
    }

    public static func load(from url: URL) throws -> VibeVoiceQuantizationManifest {
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(VibeVoiceQuantizationManifest.self, from: data)
    }
}

public enum VibeVoiceQuantizationError: Error, LocalizedError {
    case noSafetensorsFound(URL)
    case invalidGroupSize(Int)
    case invalidBits(Int)
    case quantizationFailed(String)
    case outputDirectoryCreationFailed(URL)

    public var errorDescription: String? {
        switch self {
        case .noSafetensorsFound(let url):
            return "No safetensors files found in \(url.path)"
        case .invalidGroupSize(let size):
            return "Invalid group size: \(size). Supported sizes: 32, 64, 128"
        case .invalidBits(let bits):
            return "Invalid bits: \(bits). Supported values: 4, 8"
        case .quantizationFailed(let reason):
            return "Quantization failed: \(reason)"
        case .outputDirectoryCreationFailed(let url):
            return "Failed to create output directory: \(url.path)"
        }
    }
}

public struct VibeVoiceQuantizer {
    public static let supportedGroupSizes: Set<Int> = [32, 64, 128]
    public static let supportedBits: Set<Int> = [4, 8]

    public static func quantizeAndSave(
        from sourceURL: URL,
        to outputURL: URL,
        spec: VibeVoiceQuantizationSpec,
        modelId: String? = nil,
        revision: String? = nil,
        verbose: Bool = false
    ) throws {
        guard supportedGroupSizes.contains(spec.groupSize) else {
            throw VibeVoiceQuantizationError.invalidGroupSize(spec.groupSize)
        }
        guard supportedBits.contains(spec.bits) else {
            throw VibeVoiceQuantizationError.invalidBits(spec.bits)
        }

        let fm = FileManager.default

        do {
            try fm.createDirectory(at: outputURL, withIntermediateDirectories: true)
        } catch {
            throw VibeVoiceQuantizationError.outputDirectoryCreationFailed(outputURL)
        }

        let resolvedSourceURL = sourceURL.resolvingSymlinksInPath()

        let contents = try fm.contentsOfDirectory(at: resolvedSourceURL, includingPropertiesForKeys: nil)
        let safetensorFiles = contents.filter { $0.pathExtension == "safetensors" }

        if safetensorFiles.isEmpty {
            throw VibeVoiceQuantizationError.noSafetensorsFound(resolvedSourceURL)
        }

        var allWeights: [String: MLXArray] = [:]
        for file in safetensorFiles {
            let weights = try MLX.loadArrays(url: file)
            for (key, value) in weights {
                allWeights[key] = value
            }
        }

        if verbose {
            print("Loaded \(allWeights.count) tensors from \(safetensorFiles.count) file(s)")
        }

        var quantizedWeights: [String: MLXArray] = [:]
        var quantizedLayers: [VibeVoiceQuantizationManifest.QuantizedLayerInfo] = []
        var quantizedCount = 0
        var skippedCount = 0

        for (key, tensor) in allWeights {
            let isEmbedding = key.contains("embed_tokens") || key.contains("tts_input_types")
            let isNorm = key.contains("norm") || key.contains("layernorm")
            let isEosClassifier = key.contains("eos_classifier")
            let isAcousticConv = key.contains("acoustic_tokenizer") &&
                (key.contains(".conv.") || key.contains(".convtr.") || key.contains(".mixer."))

            if key.hasSuffix(".weight") && tensor.ndim == 2 && !isEmbedding && !isNorm && !isAcousticConv && !isEosClassifier {
                let outDim = tensor.dim(0)
                let inDim = tensor.dim(1)

                if inDim % spec.groupSize == 0 {
                    let base = String(key.dropLast(".weight".count))

                    var f = tensor
                    if f.dtype != .float32 {
                        f = f.asType(.float32)
                    }

                    let (wq, scales, biases) = MLX.quantized(
                        f,
                        groupSize: spec.groupSize,
                        bits: spec.bits,
                        mode: spec.mode.mlxMode
                    )

                    quantizedWeights[key] = wq
                    quantizedWeights["\(base).scales"] = scales
                    if let b = biases {
                        quantizedWeights["\(base).biases"] = b
                    }

                    quantizedLayers.append(.init(
                        name: base,
                        shape: [outDim, inDim],
                        inDim: inDim,
                        outDim: outDim,
                        file: "model.safetensors",
                        quantFile: "model.safetensors",
                        groupSize: spec.groupSize,
                        bits: spec.bits,
                        mode: spec.mode.rawValue
                    ))

                    quantizedCount += 1

                    if verbose {
                        print("Quantized: \(key) [\(outDim), \(inDim)]")
                    }
                } else {
                    quantizedWeights[key] = tensor
                    skippedCount += 1
                    if verbose {
                        print("Skipped (group size mismatch): \(key)")
                    }
                }
            } else {
                quantizedWeights[key] = tensor
                if key.hasSuffix(".weight") && verbose {
                    if tensor.ndim != 2 {
                        print("Skipped (ndim=\(tensor.ndim)): \(key)")
                    } else if isEmbedding {
                        print("Skipped (embedding): \(key)")
                    } else if isNorm {
                        print("Skipped (norm): \(key)")
                    } else if isAcousticConv {
                        print("Skipped (acoustic_tokenizer conv): \(key)")
                    } else if isEosClassifier {
                        print("Skipped (eos_classifier): \(key)")
                    }
                }
            }
        }

        if verbose {
            print("\nQuantized \(quantizedCount) layers, skipped \(skippedCount) layers")
        }

        try saveWeights(
            quantizedWeights,
            to: outputURL,
            verbose: verbose
        )

        let manifest = VibeVoiceQuantizationManifest(
            modelId: modelId,
            revision: revision,
            groupSize: spec.groupSize,
            bits: spec.bits,
            mode: spec.mode.rawValue,
            layers: quantizedLayers
        )

        let manifestURL = outputURL.appendingPathComponent("quantization.json")
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let manifestData = try encoder.encode(manifest)
        try manifestData.write(to: manifestURL)

        if verbose {
            print("Saved manifest to: \(manifestURL.path)")
        }

        try copyAncillaryFiles(from: resolvedSourceURL, to: outputURL, verbose: verbose)
    }

    private static func saveWeights(
        _ weights: [String: MLXArray],
        to outputDir: URL,
        verbose: Bool
    ) throws {
        let maxShardBytes = 4_500_000_000

        func bytes(of array: MLXArray) -> Int {
            array.shape.reduce(1, *) * array.dtype.size
        }

        let ordered = weights.keys.sorted().map { ($0, weights[$0]!) }
        var chunks: [[(String, MLXArray)]] = []
        var current: [(String, MLXArray)] = []
        var currentBytes = 0

        for (k, v) in ordered {
            let sz = bytes(of: v)
            if currentBytes > 0 && currentBytes + sz > maxShardBytes {
                chunks.append(current)
                current = []
                currentBytes = 0
            }
            current.append((k, v))
            currentBytes += sz
        }
        if !current.isEmpty {
            chunks.append(current)
        }

        let total = max(1, chunks.count)

        for (i, chunk) in chunks.enumerated() {
            let shardName: String
            if total == 1 {
                shardName = "model.safetensors"
            } else {
                shardName = String(format: "model-%05d-of-%05d.safetensors", i + 1, total)
            }
            let dstURL = outputDir.appendingPathComponent(shardName)

            var dict: [String: MLXArray] = [:]
            for (k, v) in chunk {
                dict[k] = v
            }

            try MLX.save(arrays: dict, metadata: [:], url: dstURL)

            if verbose {
                print("Saved: \(shardName) (\(chunk.count) tensors)")
            }
        }
    }

    private static func copyAncillaryFiles(
        from sourceURL: URL,
        to outputURL: URL,
        verbose: Bool
    ) throws {
        let fm = FileManager.default
        let contents = try fm.contentsOfDirectory(at: sourceURL, includingPropertiesForKeys: nil)

        let copyExtensions: Set<String> = ["json", "txt", "md"]
        let skipFiles: Set<String> = ["quantization.json"]

        for file in contents {
            let name = file.lastPathComponent
            let ext = file.pathExtension.lowercased()

            if ext == "safetensors" {
                continue
            }

            if skipFiles.contains(name) {
                continue
            }

            if copyExtensions.contains(ext) {
                if let data = try? Data(contentsOf: file) {
                    let destURL = outputURL.appendingPathComponent(name)
                    try data.write(to: destURL)
                    if verbose {
                        print("Copied: \(name)")
                    }
                }
            }
        }
    }

    public static func hasQuantization(at directory: URL) -> Bool {
        let manifestURL = directory.appendingPathComponent("quantization.json")
        return FileManager.default.fileExists(atPath: manifestURL.path)
    }

    public static func applyQuantization(
        to model: Module,
        manifest: VibeVoiceQuantizationManifest,
        availableKeys: Set<String>
    ) {
        let defaultSpec = (manifest.groupSize, manifest.bits, manifest.mode)

        MLXNN.quantize(model: model) { path, _ in
            let scalesKey = "\(path).scales"
            guard availableKeys.contains(scalesKey) else { return nil }

            let (groupSize, bits, modeStr) =
                manifest.layers.first(where: { $0.name == path }).map {
                    ($0.groupSize ?? defaultSpec.0, $0.bits ?? defaultSpec.1, $0.mode ?? defaultSpec.2)
                } ?? defaultSpec
            let mode: QuantizationMode = modeStr == "mxfp4" ? .mxfp4 : .affine
            return (groupSize, bits, mode)
        }
    }
}
