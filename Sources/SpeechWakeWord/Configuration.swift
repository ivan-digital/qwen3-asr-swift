import Foundation

/// Configuration for the icefall KWS Zipformer (gigaspeech, 3.49M params) exported to CoreML.
///
/// The authoritative source is `config.json` shipped alongside the `.mlmodelc` bundle
/// (see `models/kws-zipformer/export/convert.py:544` in `soniqo/speech-models`). The
/// fields here mirror that schema exactly — keep in sync when re-exporting.
public struct KWSZipformerConfig: Codable, Sendable {
    public let modelName: String
    public let source: String
    public let checkpoint: String
    public let language: String
    public let license: String
    public let feature: FeatureConfig
    public let encoder: EncoderConfig
    public let decoder: DecoderConfig
    public let kws: KWSDefaults

    public struct FeatureConfig: Codable, Sendable {
        public let type: String
        public let sampleRate: Int
        public let numMelBins: Int
        public let frameLengthMs: Double
        public let frameShiftMs: Double
        public let dither: Double
        public let snipEdges: Bool
        public let normalizeSamples: Bool
        public let highFreq: Double
    }

    public struct EncoderConfig: Codable, Sendable {
        public let chunkSize: Int
        public let leftContextFrames: Int
        public let totalInputFrames: Int
        public let outputFrames: Int
        public let joinerDim: Int
        public let layerStateNames: [String]
        public let layerStateShapes: [[Int]]
        public let cachedEmbedLeftPadShape: [Int]
    }

    public struct DecoderConfig: Codable, Sendable {
        public let vocabSize: Int
        public let blankId: Int
        public let contextSize: Int
        public let decoderDim: Int
    }

    public struct KWSDefaults: Codable, Sendable {
        public let defaultThreshold: Double
        public let defaultContextScore: Double
        public let defaultNumTrailingBlanks: Int
        public let autoResetSeconds: Double
    }

    /// Defaults from the tuned export (ac_threshold=0.15, context_score=0.5).
    public static let `default` = KWSZipformerConfig(
        modelName: "kws-zipformer-gigaspeech",
        source: "pkufool/keyword-spotting-models@v0.11/icefall-kws-zipformer-gigaspeech-20240219",
        checkpoint: "finetune",
        language: "en",
        license: "Apache-2.0",
        feature: FeatureConfig(
            type: "kaldi-fbank",
            sampleRate: 16000,
            numMelBins: 80,
            frameLengthMs: 25.0,
            frameShiftMs: 10.0,
            dither: 0.0,
            snipEdges: false,
            normalizeSamples: true,
            highFreq: -400.0
        ),
        encoder: EncoderConfig(
            chunkSize: 16,
            leftContextFrames: 64,
            totalInputFrames: 45,  // chunkSize*2 + 13 (7 + 2*3 ConvNeXt pad)
            outputFrames: 8,        // chunkSize / 2 (Zipformer output downsamples by 2)
            joinerDim: 320,
            layerStateNames: [],    // Populated from config.json on load
            layerStateShapes: [],
            cachedEmbedLeftPadShape: [1, 128, 3, 19]
        ),
        decoder: DecoderConfig(
            vocabSize: 500,
            blankId: 0,
            contextSize: 2,
            decoderDim: 320
        ),
        kws: KWSDefaults(
            defaultThreshold: 0.15,
            defaultContextScore: 0.5,
            defaultNumTrailingBlanks: 1,
            autoResetSeconds: 1.5
        )
    )
}

/// One keyword registered with the detector.
public struct KeywordSpec: Sendable, Equatable {
    /// Space-separated display phrase, e.g. "hey soniqo".
    public let phrase: String
    /// Per-phrase acoustic probability threshold. 0 → use config default.
    public let acThreshold: Double
    /// Per-phrase boost score (0 → use config default). Positive values make
    /// the phrase easier to trigger; negative discourage it.
    public let boost: Double
    /// Optional explicit BPE piece sequence. When non-nil the detector uses
    /// these pieces directly (looked up by text in the model's ``tokens.txt``)
    /// instead of running the greedy BPE encoder over ``phrase``. Use this
    /// when you know the exact decomposition the model was trained on —
    /// sherpa-onnx keyword files ship in this format
    /// (``▁ L IGHT ▁UP`` for "light up", not ``▁LI GHT ▁UP``).
    public let tokens: [String]?

    public init(
        phrase: String,
        acThreshold: Double = 0.0,
        boost: Double = 0.0,
        tokens: [String]? = nil
    ) {
        self.phrase = phrase
        self.acThreshold = acThreshold
        self.boost = boost
        self.tokens = tokens
    }
}
