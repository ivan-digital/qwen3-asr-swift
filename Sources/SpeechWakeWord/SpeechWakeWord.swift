import CoreML
import Foundation
import AudioCommon

/// On-device wake-word / keyword spotter built on icefall's KWS Zipformer
/// (gigaspeech, 3.49M params) exported to CoreML.
///
/// Ships as three ``.mlmodelc`` bundles (encoder / decoder / joiner), a
/// SentencePiece BPE vocabulary (500 pieces), and a tuned threshold/boost pair.
/// See ``docs/models/kws-zipformer.md`` and ``docs/inference/wake-word.md``
/// for the full pipeline description.
///
/// - Warning: Not thread-safe. Create one detector per streaming audio source.
public final class WakeWordDetector {
    public let config: KWSZipformerConfig
    public let vocabulary: KWSVocabulary
    public let keywords: [KeywordSpec]
    public let contextGraph: ContextGraph

    public static let defaultModelId = "aufklarer/KWS-Zipformer-3M-CoreML-INT8"

    private let bpeTokenizer: BPETokenizer
    private let fbank: KaldiFbank

    var _isLoaded = true
    var encoder: MLModel?
    var decoder: MLModel?
    var joiner: MLModel?

    private init(
        config: KWSZipformerConfig,
        vocabulary: KWSVocabulary,
        bpeTokenizer: BPETokenizer,
        keywords: [KeywordSpec],
        contextGraph: ContextGraph,
        fbank: KaldiFbank,
        encoder: MLModel,
        decoder: MLModel,
        joiner: MLModel
    ) {
        self.config = config
        self.vocabulary = vocabulary
        self.bpeTokenizer = bpeTokenizer
        self.keywords = keywords
        self.contextGraph = contextGraph
        self.fbank = fbank
        self.encoder = encoder
        self.decoder = decoder
        self.joiner = joiner
    }

    // MARK: - Streaming API

    /// Create a new streaming detection session sharing this detector's models
    /// and context graph. Call ``WakeWordSession/reset()`` between unrelated
    /// audio streams.
    public func createSession() throws -> WakeWordSession {
        guard _isLoaded, let encoder, let decoder, let joiner else {
            throw AudioModelError.inferenceFailed(operation: "createSession", reason: "Model not loaded")
        }
        return try WakeWordSession(
            config: config,
            encoder: encoder,
            decoder: decoder,
            joiner: joiner,
            fbank: fbank,
            contextGraph: contextGraph
        )
    }

    /// Batch-detect keyword occurrences in a full audio buffer.
    /// Convenience wrapper around ``createSession``. Resamples to 16 kHz if needed.
    public func detect(audio: [Float], sampleRate: Int) throws -> [KeywordDetection] {
        let samples: [Float]
        if sampleRate != config.feature.sampleRate {
            samples = AudioFileLoader.resample(audio, from: sampleRate, to: config.feature.sampleRate)
        } else {
            samples = audio
        }
        let session = try createSession()
        var detections = try session.pushAudio(samples)
        detections.append(contentsOf: try session.finalize())
        return detections
    }

    public func warmUp() throws {
        let dummy = [Float](repeating: 0, count: config.feature.sampleRate) // 1s silence
        _ = try detect(audio: dummy, sampleRate: config.feature.sampleRate)
    }

    /// Build a ``StreamingKwsDecoder`` wired to the detector's CoreML
    /// decoder + joiner but without the encoder / fbank front-end. Used in
    /// parity tests that drive the beam search with reference encoder
    /// frames from an external source (e.g. Python).
    public func makeKwsDecoder(
        keywords: [KeywordSpec],
        contextScore: Double,
        acThreshold: Double,
        numTrailingBlanks: Int = 1,
        autoResetSeconds: Double = 1.5,
        beam: Int = 4
    ) throws -> StreamingKwsDecoder {
        guard let decoder = decoder, let joiner = joiner else {
            throw AudioModelError.inferenceFailed(operation: "makeKwsDecoder", reason: "Model not loaded")
        }
        let graph = ContextGraph(contextScore: contextScore, acThreshold: acThreshold)
        var ids: [[Int]] = []
        var phrases: [String] = []
        for kw in keywords {
            if let tokens = kw.tokens {
                let mapped = try tokens.map { piece -> Int in
                    guard let id = vocabulary.tokenToId[piece] else {
                        throw AudioModelError.invalidConfiguration(
                            model: "KWS-Zipformer", reason: "Unknown BPE piece '\(piece)'")
                    }
                    return id
                }
                ids.append(mapped)
            } else {
                ids.append(bpeTokenizer.encode(kw.phrase))
            }
            phrases.append(kw.phrase)
        }
        graph.build(
            tokenIds: ids, phrases: phrases,
            boosts: Array(repeating: 0.0, count: ids.count),
            thresholds: Array(repeating: 0.0, count: ids.count)
        )
        let ctxSize = config.decoder.contextSize
        return StreamingKwsDecoder(
            decoderFn: { ctx in
                WakeWordSession.runDecoder(model: decoder, contextTokens: ctx, contextSize: ctxSize)
            },
            joinerFn: { enc, dec in
                WakeWordSession.runJoiner(model: joiner, encoderFrame: enc, decoderOut: dec)
            },
            contextGraph: graph,
            blankId: config.decoder.blankId,
            unkId: nil,
            contextSize: ctxSize,
            beam: beam,
            numTrailingBlanks: numTrailingBlanks,
            blankPenalty: 0,
            frameShiftSeconds: 0.04,
            autoResetSeconds: autoResetSeconds
        )
    }

    // MARK: - Loading

    /// Load the model and build a context graph from a list of keywords.
    public static func fromPretrained(
        modelId: String? = nil,
        keywords: [KeywordSpec],
        progressHandler: ((Double, String) -> Void)? = nil
    ) async throws -> WakeWordDetector {
        let effectiveModelId = modelId ?? defaultModelId
        AudioLog.modelLoading.info("Loading KWS Zipformer: \(effectiveModelId)")

        let cacheDir = try HuggingFaceDownloader.getCacheDirectory(for: effectiveModelId)

        progressHandler?(0.0, "Downloading model...")
        try await HuggingFaceDownloader.downloadWeights(
            modelId: effectiveModelId,
            to: cacheDir,
            additionalFiles: [
                "encoder.mlmodelc/**",
                "decoder.mlmodelc/**",
                "joiner.mlmodelc/**",
                "tokens.txt",
                "bpe.model",
                "config.json",
                "commands_small.txt",
                "commands_large.txt"
            ]
        ) { fraction in progressHandler?(fraction * 0.7, "Downloading model...") }

        progressHandler?(0.70, "Loading configuration...")
        let config: KWSZipformerConfig
        let configURL = cacheDir.appendingPathComponent("config.json")
        if FileManager.default.fileExists(atPath: configURL.path) {
            let data = try Data(contentsOf: configURL)
            config = try JSONDecoder().decode(KWSZipformerConfig.self, from: data)
        } else {
            config = .default
        }

        progressHandler?(0.78, "Loading vocabulary...")
        let vocab = try KWSVocabulary.load(
            from: cacheDir.appendingPathComponent("tokens.txt"),
            blankId: config.decoder.blankId
        )

        progressHandler?(0.82, "Loading tokenizer...")
        let sp = try SentencePieceModel(contentsOf: cacheDir.appendingPathComponent("bpe.model"))
        let tokenizer = BPETokenizer(model: sp)

        progressHandler?(0.88, "Loading encoder...")
        let encoder = try loadMLModel(name: "encoder", from: cacheDir, units: .cpuAndNeuralEngine)
        progressHandler?(0.93, "Loading decoder...")
        let decoder = try loadMLModel(name: "decoder", from: cacheDir, units: .cpuAndNeuralEngine)
        progressHandler?(0.97, "Loading joiner...")
        let joiner = try loadMLModel(name: "joiner", from: cacheDir, units: .cpuAndNeuralEngine)

        let fbank = KaldiFbank(.init(
            sampleRate: config.feature.sampleRate,
            frameLengthMs: config.feature.frameLengthMs,
            frameShiftMs: config.feature.frameShiftMs,
            numMelBins: config.feature.numMelBins,
            highFreq: config.feature.highFreq,
            snipEdges: config.feature.snipEdges
        ))

        let graph = try buildContextGraph(
            keywords: keywords, tokenizer: tokenizer,
            vocabulary: vocab, defaults: config.kws
        )

        progressHandler?(1.0, "Model loaded")
        AudioLog.modelLoading.info("KWS Zipformer loaded (\(keywords.count) keywords, \(vocab.count) tokens)")

        return WakeWordDetector(
            config: config,
            vocabulary: vocab,
            bpeTokenizer: tokenizer,
            keywords: keywords,
            contextGraph: graph,
            fbank: fbank,
            encoder: encoder,
            decoder: decoder,
            joiner: joiner
        )
    }

    /// Build a context graph using this detector's tokenizer. Useful when
    /// you want to swap keyword lists without reloading the model.
    public func rebuildContextGraph(keywords: [KeywordSpec]) throws -> WakeWordDetector {
        guard let encoder = encoder, let decoder = decoder, let joiner = joiner else {
            throw AudioModelError.inferenceFailed(operation: "rebuildContextGraph", reason: "Model not loaded")
        }
        let graph = try Self.buildContextGraph(
            keywords: keywords, tokenizer: bpeTokenizer,
            vocabulary: vocabulary, defaults: config.kws
        )
        return WakeWordDetector(
            config: config,
            vocabulary: vocabulary,
            bpeTokenizer: bpeTokenizer,
            keywords: keywords,
            contextGraph: graph,
            fbank: fbank,
            encoder: encoder,
            decoder: decoder,
            joiner: joiner
        )
    }

    // MARK: - Helpers

    private static func buildContextGraph(
        keywords: [KeywordSpec],
        tokenizer: BPETokenizer,
        vocabulary: KWSVocabulary,
        defaults: KWSZipformerConfig.KWSDefaults
    ) throws -> ContextGraph {
        guard !keywords.isEmpty else {
            throw AudioModelError.invalidConfiguration(
                model: "KWS-Zipformer",
                reason: "At least one KeywordSpec is required"
            )
        }
        let graph = ContextGraph(
            contextScore: defaults.defaultContextScore,
            acThreshold: defaults.defaultThreshold
        )
        var tokenIds: [[Int]] = []
        var phrases: [String] = []
        var boosts: [Double] = []
        var thresholds: [Double] = []
        for kw in keywords {
            let ids: [Int]
            if let pieces = kw.tokens {
                ids = try pieces.map { piece -> Int in
                    guard let id = vocabulary.tokenToId[piece] else {
                        throw AudioModelError.invalidConfiguration(
                            model: "KWS-Zipformer",
                            reason: "Unknown BPE piece '\(piece)' in keyword '\(kw.phrase)'"
                        )
                    }
                    return id
                }
            } else {
                ids = tokenizer.encode(kw.phrase)
            }
            tokenIds.append(ids)
            phrases.append(kw.phrase)
            boosts.append(kw.boost)
            thresholds.append(kw.acThreshold)
        }
        graph.build(
            tokenIds: tokenIds,
            phrases: phrases,
            boosts: boosts,
            thresholds: thresholds
        )
        return graph
    }

    private static func loadMLModel(
        name: String, from directory: URL, units: MLComputeUnits
    ) throws -> MLModel {
        let url = directory.appendingPathComponent("\(name).mlmodelc", isDirectory: true)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw AudioModelError.modelLoadFailed(
                modelId: name, reason: "CoreML model not found at \(url.path)"
            )
        }
        let mlConfig = MLModelConfiguration()
        mlConfig.computeUnits = units
        return try MLModel(contentsOf: url, configuration: mlConfig)
    }
}
