import Foundation

// MARK: - Top-level Configuration

public struct PersonaPlexConfig: Sendable {
    public var temporal: TemporalTransformerConfig
    public var depformer: DepformerConfig
    public var mimi: MimiConfig
    public var sampling: PersonaPlexSamplingConfig
    public var delays: [Int]
    public var sampleRate: Int

    public init(
        temporal: TemporalTransformerConfig = .default,
        depformer: DepformerConfig = .default,
        mimi: MimiConfig = .moshiko(),
        sampling: PersonaPlexSamplingConfig = .default,
        delays: [Int] = [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
        sampleRate: Int = 24000
    ) {
        self.temporal = temporal
        self.depformer = depformer
        self.mimi = mimi
        self.sampling = sampling
        self.delays = delays
        self.sampleRate = sampleRate
    }

    /// Number of streams: 1 text + 8 user audio + 8 agent audio = 17
    public var numStreams: Int { 1 + temporal.nQ + temporal.nQ }

    /// Number of user audio codebooks
    public var numUserAudioCodebooks: Int { temporal.nQ }

    /// Number of agent audio codebooks (same as depformer steps)
    public var numAgentAudioCodebooks: Int { depformer.numSteps }

    /// Maximum delay across all streams
    public var maxDelay: Int { delays.max() ?? 0 }

    public static var `default`: PersonaPlexConfig { PersonaPlexConfig() }
}

// MARK: - Temporal Transformer Config

public struct TemporalTransformerConfig: Sendable {
    public var dim: Int
    public var numLayers: Int
    public var numHeads: Int
    public var hiddenScale: Float
    public var nQ: Int              // Number of audio codebooks per side (8 user + 8 agent)
    public var card: Int            // Audio vocabulary size
    public var textCard: Int        // Text vocabulary size
    public var context: Int
    public var maxPeriod: Int
    public var rmsNormEps: Float
    public var groupSize: Int       // Quantization group size
    public var bits: Int            // Quantization bits

    /// FFN intermediate size: LLaMA-style formula dim * 2/3 * hiddenScale
    public var intermediateSize: Int { Int(Float(dim) * 2.0 / 3.0 * hiddenScale) }

    /// Head dimension
    public var headDim: Int { dim / numHeads }

    /// Number of audio embedding tables: nQ for user + nQ for agent = 2 * nQ = 16
    public var numAudioEmbeddings: Int { 2 * nQ }

    /// Total codebooks in delay array: 1 text + 16 audio
    public var numCodebooks: Int { 1 + numAudioEmbeddings }

    /// Initial token ID for audio (= card, one past valid range)
    public var initialTokenId: Int { card }

    /// Initial token ID for text (= textCard)
    public var textInitialTokenId: Int { textCard }

    /// Text padding token ID (existing_text_padding_id=3 in original model)
    public var textPaddingId: Int { 3 }

    /// Constant sine wave tokens (440Hz reference tone, used for user audio during prompting)
    public static let sineTokens: [Int32] = [430, 1268, 381, 1611, 1095, 1495, 56, 472]

    /// Constant silence tokens (used for agent audio during silence/text prompt phases)
    public static let silenceTokens: [Int32] = [948, 243, 1178, 546, 1736, 1030, 1978, 2008]

    /// Default system prompt: "<system> You are a helpful assistant. Answer questions clearly and concisely. <system>"
    /// Pre-tokenized with SentencePiece (tokenizer_spm_32k_3.model)
    public static let defaultSystemPromptTokens: [Int32] = [
        607, 4831, 578, 493, 298, 272, 3850, 5019, 263,
        506, 1292, 2366, 267, 22876, 362, 263, 607, 4831, 578
    ]

    public init(
        dim: Int = 4096,
        numLayers: Int = 32,
        numHeads: Int = 32,
        hiddenScale: Float = 4.125,
        nQ: Int = 8,
        card: Int = 2048,
        textCard: Int = 32000,
        context: Int = 3000,
        maxPeriod: Int = 10000,
        rmsNormEps: Float = 1e-8,
        groupSize: Int = 64,
        bits: Int = 4
    ) {
        self.dim = dim
        self.numLayers = numLayers
        self.numHeads = numHeads
        self.hiddenScale = hiddenScale
        self.nQ = nQ
        self.card = card
        self.textCard = textCard
        self.context = context
        self.maxPeriod = maxPeriod
        self.rmsNormEps = rmsNormEps
        self.groupSize = groupSize
        self.bits = bits
    }

    public static var `default`: TemporalTransformerConfig { TemporalTransformerConfig() }
}

// MARK: - Depformer Config

public struct DepformerConfig: Sendable {
    public var dim: Int
    public var numLayers: Int
    public var numHeads: Int
    public var dimFeedforward: Int
    public var numSteps: Int        // Number of codebooks to generate (8 for moshiko, 16 for personaplex)
    public var card: Int            // Audio vocabulary size
    public var textCard: Int        // Text vocabulary size (for text embedding in depformer)
    public var context: Int
    public var rmsNormEps: Float
    public var weightsPerStep: Bool // Each step uses different weights
    public var multiLinear: Bool    // Use multi-linear projections
    public var groupSize: Int       // Quantization group size
    public var bits: Int            // Quantization bits (4 = int4, 16 = BF16)

    /// Head dimension
    public var headDim: Int { dim / numHeads }

    public init(
        dim: Int = 1024,
        numLayers: Int = 6,
        numHeads: Int = 16,
        dimFeedforward: Int = 2816,
        numSteps: Int = 16,     // PersonaPlex uses dep_q=16
        card: Int = 2048,
        textCard: Int = 32000,
        context: Int = 8,
        rmsNormEps: Float = 1e-8,
        weightsPerStep: Bool = true,
        multiLinear: Bool = true,
        groupSize: Int = 64,
        bits: Int = 4
    ) {
        self.dim = dim
        self.numLayers = numLayers
        self.numHeads = numHeads
        self.dimFeedforward = dimFeedforward
        self.numSteps = numSteps
        self.card = card
        self.textCard = textCard
        self.context = context
        self.rmsNormEps = rmsNormEps
        self.weightsPerStep = weightsPerStep
        self.multiLinear = multiLinear
        self.groupSize = groupSize
        self.bits = bits
    }

    public static var `default`: DepformerConfig { DepformerConfig() }
}

// MARK: - Mimi Codec Config

public struct MimiConfig: Sendable {
    public let channels: Int
    public let sampleRate: Double
    public let frameRate: Double
    public let dimension: Int
    public let numCodebooks: Int
    public let codebookSize: Int
    public let codebookDim: Int
    public let seanet: SeanetConfig
    public let transformer: MimiTransformerConfig

    public static func moshiko(numCodebooks: Int = 16) -> MimiConfig {
        let seanet = SeanetConfig(
            dimension: 512, channels: 1, causal: true,
            nFilters: 64, nResidualLayers: 1, ratios: [8, 6, 5, 4],
            kernelSize: 7, residualKernelSize: 3, lastKernelSize: 3,
            dilationBase: 2, trueSkip: true, compress: 2,
            padMode: .edge
        )
        let transformer = MimiTransformerConfig(
            dModel: 512, numHeads: 8, numLayers: 8,
            causal: true, biasFF: false, biasAttn: false,
            layerScale: 0.01, context: 250, maxPeriod: 10000,
            dimFeedforward: 2048, gating: false
        )
        return MimiConfig(
            channels: 1, sampleRate: 24000, frameRate: 12.5,
            dimension: 512, numCodebooks: numCodebooks,
            codebookSize: 2048, codebookDim: 256,
            seanet: seanet, transformer: transformer
        )
    }
}

// MARK: - SEANet Config

public struct SeanetConfig: Sendable {
    public let dimension: Int
    public let channels: Int
    public let causal: Bool
    public let nFilters: Int
    public let nResidualLayers: Int
    public let ratios: [Int]
    public let kernelSize: Int
    public let residualKernelSize: Int
    public let lastKernelSize: Int
    public let dilationBase: Int
    public let trueSkip: Bool
    public let compress: Int
    public let padMode: PadMode

    // Aliases
    public var nfilters: Int { nFilters }
    public var nresidualLayers: Int { nResidualLayers }
    public var ksize: Int { kernelSize }
    public var residualKsize: Int { residualKernelSize }
    public var lastKsize: Int { lastKernelSize }
}

// MARK: - Mimi Transformer Config

public struct MimiTransformerConfig: Sendable {
    public let dModel: Int
    public let numHeads: Int
    public let numLayers: Int
    public let causal: Bool
    public let biasFF: Bool
    public let biasAttn: Bool
    public let layerScale: Float?
    public let context: Int
    public let maxPeriod: Int
    public let dimFeedforward: Int
    public let gating: Bool

    public var kvRepeat: Int = 1
    public var positionalEmbedding: String = "rope"
    public var useConvBlock: Bool = false
    public var crossAttention: Bool = false
    public var norm: String = "layer_norm"
    public var convLayout: Bool = true

    public var headDim: Int { dModel / numHeads }
}

// MARK: - Sampling Config

public struct PersonaPlexSamplingConfig: Sendable {
    public var audioTemp: Float
    public var audioTopK: Int
    public var textTemp: Float
    public var textTopK: Int
    public var audioRepetitionPenalty: Float
    public var repetitionWindow: Int

    public init(
        audioTemp: Float = 0.8,
        audioTopK: Int = 250,
        textTemp: Float = 0.7,
        textTopK: Int = 25,
        audioRepetitionPenalty: Float = 1.2,
        repetitionWindow: Int = 30
    ) {
        self.audioTemp = audioTemp
        self.audioTopK = audioTopK
        self.textTemp = textTemp
        self.textTopK = textTopK
        self.audioRepetitionPenalty = audioRepetitionPenalty
        self.repetitionWindow = repetitionWindow
    }

    public static var `default`: PersonaPlexSamplingConfig { PersonaPlexSamplingConfig() }
}

// MARK: - Voice Preset

public enum PersonaPlexVoice: String, CaseIterable, Sendable {
    // Natural Female
    case NATF0, NATF1, NATF2, NATF3
    // Natural Male
    case NATM0, NATM1, NATM2, NATM3
    // Variety Female
    case VARF0, VARF1, VARF2, VARF3, VARF4
    // Variety Male
    case VARM0, VARM1, VARM2, VARM3, VARM4

    public var displayName: String {
        switch self {
        case .NATF0: return "Natural Female 0"
        case .NATF1: return "Natural Female 1"
        case .NATF2: return "Natural Female 2"
        case .NATF3: return "Natural Female 3"
        case .NATM0: return "Natural Male 0"
        case .NATM1: return "Natural Male 1"
        case .NATM2: return "Natural Male 2"
        case .NATM3: return "Natural Male 3"
        case .VARF0: return "Variety Female 0"
        case .VARF1: return "Variety Female 1"
        case .VARF2: return "Variety Female 2"
        case .VARF3: return "Variety Female 3"
        case .VARF4: return "Variety Female 4"
        case .VARM0: return "Variety Male 0"
        case .VARM1: return "Variety Male 1"
        case .VARM2: return "Variety Male 2"
        case .VARM3: return "Variety Male 3"
        case .VARM4: return "Variety Male 4"
        }
    }
}

// MARK: - System Prompt Presets

/// Pre-tokenized system prompts for PersonaPlex (SentencePiece tokenizer_spm_32k_3.model).
/// Use `--system-prompt` CLI flag to select a preset.
public enum SystemPromptPreset: String, CaseIterable, Sendable {
    case focused
    case assistant
    case customerService = "customer-service"
    case teacher

    public var tokens: [Int32] {
        switch self {
        case .focused:
            // "<system> You are a helpful assistant. Listen carefully to what the user says,
            //  then respond directly to their question or request. Stay on topic. Be concise. <system>"
            return [
                607, 4831, 578, 493, 298, 272, 3850, 5019, 263,
                17453, 6716, 269, 419, 262, 819, 1182, 261, 409,
                4816, 1312, 269, 347, 560, 307, 2498, 263, 17308,
                291, 3398, 263, 1451, 22876, 263, 607, 4831, 578
            ]
        case .assistant:
            // "<system> You are a helpful assistant. Answer questions clearly and concisely. <system>"
            return [
                607, 4831, 578, 493, 298, 272, 3850, 5019, 263,
                506, 1292, 2366, 267, 22876, 362, 263, 607, 4831, 578
            ]
        case .customerService:
            // "<system> You are a customer service agent. Answer the customer question directly
            //  and helpfully. Do not change the subject. <system>"
            return [
                607, 4831, 578, 493, 298, 272, 3740, 844, 3022, 263,
                506, 262, 3740, 560, 1312, 267, 3850, 362, 263,
                958, 309, 652, 262, 1523, 263, 607, 4831, 578
            ]
        case .teacher:
            // "<system> You are a wise and friendly teacher. Answer questions or provide advice
            //  in a clear and engaging way. <system>"
            return [
                607, 4831, 578, 493, 298, 272, 11821, 267, 7514, 3290, 263,
                506, 1292, 307, 775, 3574, 271, 272, 1195, 267, 12250, 488, 263,
                607, 4831, 578
            ]
        }
    }

    public var description: String {
        switch self {
        case .focused: return "Focused responder — stays on topic, concise"
        case .assistant: return "General helpful assistant (default)"
        case .customerService: return "Customer service agent — direct, no topic changes"
        case .teacher: return "Friendly teacher — clear and engaging"
        }
    }
}
