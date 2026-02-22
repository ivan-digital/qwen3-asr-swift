import Foundation
import MLX
import MLXNN
import Qwen3Common

@inline(__always) private func product(_ xs: [Int]) -> Int { xs.reduce(1, *) }

// MARK: - Mimi

public final class Mimi: Module {
    public let cfg: MimiConfig

    @ModuleInfo public var encoder: SeanetEncoder
    @ModuleInfo public var decoder: SeanetDecoder
    @ModuleInfo public var quantizer: SplitResidualVectorQuantizer

    @ModuleInfo public var encoder_transformer: ProjectedTransformer
    @ModuleInfo public var decoder_transformer: ProjectedTransformer

    @ModuleInfo public var downsample: ConvDownsample1d
    @ModuleInfo public var upsample: ConvTrUpsample1d

    public private(set) var encoderCache: [KVCacheSimple]
    public private(set) var decoderCache: [KVCacheSimple]

    private let downsampleStride: Int

    public init(cfg: MimiConfig) {
        self.cfg = cfg

        let encFPS = cfg.sampleRate / Double(product(cfg.seanet.ratios))
        self.downsampleStride = Int(encFPS / cfg.frameRate)

        self._encoder = ModuleInfo(wrappedValue: SeanetEncoder(cfg: cfg.seanet))
        self._decoder = ModuleInfo(wrappedValue: SeanetDecoder(cfg: cfg.seanet))

        self._quantizer = ModuleInfo(wrappedValue: SplitResidualVectorQuantizer(
            dim: cfg.codebookDim, inputDim: cfg.seanet.dimension,
            outputDim: cfg.seanet.dimension, nq: cfg.numCodebooks, bins: cfg.codebookSize))

        let txCfg = cfg.transformer
        self._encoder_transformer = ModuleInfo(wrappedValue: ProjectedTransformer(
            cfg: txCfg, inputDim: cfg.seanet.dimension,
            outputDims: [cfg.seanet.dimension]))
        self._decoder_transformer = ModuleInfo(wrappedValue: ProjectedTransformer(
            cfg: txCfg, inputDim: cfg.seanet.dimension,
            outputDims: [cfg.seanet.dimension]))

        self._downsample = ModuleInfo(wrappedValue: ConvDownsample1d(
            stride: downsampleStride, dim: cfg.seanet.dimension, causal: true))
        self._upsample = ModuleInfo(wrappedValue: ConvTrUpsample1d(
            stride: downsampleStride, dim: cfg.seanet.dimension, causal: true))

        self.encoderCache = _encoder_transformer.wrappedValue.makeCache()
        self.decoderCache = _decoder_transformer.wrappedValue.makeCache()
    }

    public func resetState() {
        encoder.resetState()
        decoder.resetState()
        for c in decoderCache { c.trim(c.offset) }
        for c in encoderCache { c.trim(c.offset) }
    }

    public var frameRate: Double { cfg.frameRate }
    public var sampleRate: Double { cfg.sampleRate }

    public func encode(_ xs: MLXArray) -> MLXArray {
        encoder.resetState()
        for c in encoderCache { c.trim(c.offset) }

        var z = encoder(xs)
        z = encoder_transformer(z, cache: encoderCache)[0]
        z = downsample(z)
        return quantizer.encode(z)
    }

    public func decode(_ codes: MLXArray) -> MLXArray {
        decoder.resetState()
        for c in decoderCache { c.trim(c.offset) }

        var z = quantizer.decode(codes)
        z = upsample(z)
        z = decoder_transformer(z, cache: decoderCache)[0]
        return decoder(z)
    }

    public func encodeStep(_ xs: MLXArray) -> MLXArray {
        var z = encoder.step(xs)
        z = encoder_transformer(z, cache: encoderCache)[0]
        z = downsample.step(z)
        z = quantizer.encode(z)
        return z
    }

    public func decodeStep(_ codes: MLXArray) -> MLXArray {
        var z = quantizer.decode(codes)
        z = upsample.step(z)
        z = decoder_transformer(z, cache: decoderCache)[0]
        z = decoder.step(z)
        return z
    }
}

// MARK: - Mimi Weight Sanitization

extension Mimi {
    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]

        for (rawKey, rawVal) in weights {
            var k = rawKey
                .split(separator: ".")
                .map { seg -> String in
                    if seg.hasPrefix("_") { return String(seg.dropFirst()) }
                    return String(seg)
                }
                .joined(separator: ".")

            if k.hasPrefix("encoder.model.") {
                k = k.replacingOccurrences(of: "encoder.model.", with: "encoder.")
            }
            if k.hasPrefix("decoder.model.") {
                k = k.replacingOccurrences(of: "decoder.model.", with: "decoder.")
            }

            if k.hasSuffix(".in_proj_weight") {
                k = k.replacingOccurrences(of: ".in_proj_weight", with: ".in_proj.weight")
            }
            if k.hasSuffix(".linear1.weight") {
                k = k.replacingOccurrences(of: ".linear1.weight", with: ".gating.linear1.weight")
            }
            if k.hasSuffix(".linear2.weight") {
                k = k.replacingOccurrences(of: ".linear2.weight", with: ".gating.linear2.weight")
            }

            // Decoder layer index remapping
            let decIdx = [2, 5, 8, 11]
            for (layerIdx, decoderIdx) in decIdx.enumerated() {
                k = k.replacingOccurrences(of: "decoder.\(decoderIdx).",
                                           with: "decoder.layers.\(layerIdx).upsample.")
                k = k.replacingOccurrences(of: "decoder.\(decoderIdx + 1).",
                                           with: "decoder.layers.\(layerIdx).residuals.0.")
            }
            let encIdx = [1, 4, 7, 10]
            for (layerIdx, encoderIdx) in encIdx.enumerated() {
                k = k.replacingOccurrences(of: "encoder.\(encoderIdx).",
                                           with: "encoder.layers.\(layerIdx).residuals.0.")
                k = k.replacingOccurrences(of: "encoder.\(encoderIdx + 2).",
                                           with: "encoder.layers.\(layerIdx).downsample.")
            }

            k = k.replacingOccurrences(of: "decoder.0.", with: "decoder.init_conv1d.")
            k = k.replacingOccurrences(of: "decoder.14.", with: "decoder.final_conv1d.")
            k = k.replacingOccurrences(of: "encoder.0.", with: "encoder.init_conv1d.")
            k = k.replacingOccurrences(of: "encoder.14.", with: "encoder.final_conv1d.")
            k = k.replacingOccurrences(of: ".block.1.", with: ".block.0.")
            k = k.replacingOccurrences(of: ".block.3.", with: ".block.1.")

            var v = rawVal
            if k.hasSuffix(".conv.weight")
                || k.hasSuffix(".output_proj.weight")
                || k.hasSuffix(".input_proj.weight") {
                if v.ndim >= 2 { v = swappedAxes(v, v.ndim - 1, v.ndim - 2) }
            }
            if k.hasSuffix(".convtr.weight") {
                if v.ndim == 3 {
                    if v.shape[1] == 1 {
                        v = swappedAxes(v, 1, 2) // Depthwise
                    } else {
                        v = v.transposed(1, 2, 0) // Regular
                    }
                }
            }

            out[k] = v
        }
        return out
    }
}

// MARK: - MimiStreamingDecoder

public final class MimiStreamingDecoder {
    private let mimi: Mimi

    public init(_ mimi: Mimi) {
        self.mimi = mimi
        reset()
    }

    public func reset() {
        mimi.decoder.resetState()
        mimi.upsample.resetState()
        for c in mimi.decoderCache { c.trim(c.offset) }
    }

    public func decodeFrames(_ tokens: MLXArray) -> MLXArray {
        let tok = (tokens.ndim == 2) ? tokens.expandedDimensions(axes: [0]) : tokens
        let T = tok.shape[2]

        var pcs: [MLXArray] = []
        for t in 0..<T {
            let left = split(tok, indices: [t], axis: 2)
            let mid = split(left[1], indices: [1], axis: 2)[0]
            pcs.append(mimi.decodeStep(mid))
        }
        return concatenated(pcs, axis: 2)
    }
}

// MARK: - Mimi Loading

public extension Mimi {
    static func fromPretrained(
        repoId: String = "kyutai/moshiko-pytorch-bf16",
        filename: String = "tokenizer-e351c8d8-checkpoint125.safetensors",
        numCodebooks: Int = 16,
        progressHandler: ((Double, String) -> Void)? = nil
    ) async throws -> Mimi {
        let cfg = MimiConfig.moshiko(numCodebooks: numCodebooks)
        let model = Mimi(cfg: cfg)

        progressHandler?(0.1, "Downloading Mimi codec...")
        let mimiDir = try HuggingFaceDownloader.getCacheDirectory(for: repoId)
        let weightFile = mimiDir.appendingPathComponent(filename)

        if !FileManager.default.fileExists(atPath: weightFile.path) {
            let url = URL(string: "https://huggingface.co/\(repoId)/resolve/main/\(filename)")!
            let config = URLSessionConfiguration.default
            config.timeoutIntervalForResource = 600
            let session = URLSession(configuration: config)
            defer { session.finishTasksAndInvalidate() }

            let (tempURL, response) = try await session.download(from: url)
            guard let httpResponse = response as? HTTPURLResponse,
                  httpResponse.statusCode == 200 else {
                throw DownloadError.failedToDownload(filename)
            }
            try FileManager.default.moveItem(at: tempURL, to: weightFile)
        }

        progressHandler?(0.5, "Loading Mimi weights...")
        var weights = try MLX.loadArrays(url: weightFile)
        weights = model.sanitize(weights: weights)

        progressHandler?(0.8, "Applying Mimi parameters...")
        let parameters = ModuleParameters.unflattened(weights)
        try model.update(parameters: parameters, verify: .all)

        func updateCodebooks(_ module: Module) {
            if let codebook = module as? EuclideanCodebook {
                codebook.updateInPlace()
            }
            for (_, child) in module.children().flattened() {
                updateCodebooks(child)
            }
        }
        updateCodebooks(model)
        eval(model)

        progressHandler?(1.0, "Mimi codec ready")
        return model
    }
}
