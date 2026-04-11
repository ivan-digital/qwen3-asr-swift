import Foundation

/// Configuration for Meta's Omnilingual ASR (CTC variant).
///
/// The published config.json files on HuggingFace (e.g.
/// `aufklarer/Omnilingual-ASR-CTC-300M-CoreML-INT8-10s`) are decoded into
/// this struct. Both the 5s and 10s window variants share the same
/// architecture; only `maxAudioSeconds` / `inputSamples` differ.
public struct OmnilingualConfig: Codable, Sendable {
    /// Expected audio sample rate in Hz (always 16000).
    public let sampleRate: Int
    /// Encoder output frame rate in Hz (= sampleRate / 320 for wav2vec2).
    public let frameRate: Int
    /// Maximum audio window baked into the CoreML graph, in seconds.
    public let maxAudioSeconds: Double
    /// Exact sample count the CoreML model was traced with (= maxAudioSeconds * sampleRate).
    public let inputSamples: Int
    /// Encoder architecture.
    public let encoder: Encoder
    /// CTC head.
    public let ctcHead: CTCHead
    /// Tokenizer metadata.
    public let tokenizer: Tokenizer

    public struct Encoder: Codable, Sendable {
        public let numLayers: Int
        public let modelDim: Int
        public let numHeads: Int

        enum CodingKeys: String, CodingKey {
            case numLayers = "num_layers"
            case modelDim = "model_dim"
            case numHeads = "num_heads"
        }
    }

    public struct CTCHead: Codable, Sendable {
        public let vocabSize: Int

        enum CodingKeys: String, CodingKey {
            case vocabSize = "vocab_size"
        }
    }

    public struct Tokenizer: Codable, Sendable {
        public let kind: String
        public let file: String
        public let bosIdx: Int
        public let padIdx: Int
        public let eosIdx: Int
        public let unkIdx: Int

        enum CodingKeys: String, CodingKey {
            case kind, file
            case bosIdx = "bos_idx"
            case padIdx = "pad_idx"
            case eosIdx = "eos_idx"
            case unkIdx = "unk_idx"
        }
    }

    enum CodingKeys: String, CodingKey {
        case sampleRate = "sample_rate"
        case frameRate = "frame_rate"
        case maxAudioSeconds = "max_audio_seconds"
        case inputSamples = "input_samples"
        case encoder
        case ctcHead = "ctc_head"
        case tokenizer
    }

    /// Default config matching the published 10s CoreML INT8 variant.
    public static let default10s = OmnilingualConfig(
        sampleRate: 16000,
        frameRate: 50,
        maxAudioSeconds: 10.0,
        inputSamples: 160_000,
        encoder: Encoder(numLayers: 24, modelDim: 1024, numHeads: 16),
        ctcHead: CTCHead(vocabSize: 10288),
        tokenizer: Tokenizer(
            kind: "sentencepiece",
            file: "tokenizer.model",
            bosIdx: 0, padIdx: 1, eosIdx: 2, unkIdx: 3
        )
    )

    /// Default config matching the published 5s CoreML INT8 variant.
    public static let default5s = OmnilingualConfig(
        sampleRate: 16000,
        frameRate: 50,
        maxAudioSeconds: 5.0,
        inputSamples: 80_000,
        encoder: Encoder(numLayers: 24, modelDim: 1024, numHeads: 16),
        ctcHead: CTCHead(vocabSize: 10288),
        tokenizer: Tokenizer(
            kind: "sentencepiece",
            file: "tokenizer.model",
            bosIdx: 0, padIdx: 1, eosIdx: 2, unkIdx: 3
        )
    )
}
