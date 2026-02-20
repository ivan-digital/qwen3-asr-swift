import Foundation

// MARK: - LLM Config

public struct CosyVoiceLLMConfig: Codable, Sendable {
    public var hiddenSize: Int = 896
    public var numLayers: Int = 24
    public var numHeads: Int = 14
    public var numKVHeads: Int = 2
    public var headDim: Int = 64
    public var intermediateSize: Int = 4864
    public var ropeTheta: Float = 1_000_000.0
    public var rmsNormEps: Float = 1e-6
    public var textVocabSize: Int = 151936
    public var speechTokenSize: Int = 6561
    public var speechTokenExtra: Int = 200
    public var groupSize: Int = 64
    public var bits: Int = 4

    /// Total speech embedding/head size: speechTokenSize + speechTokenExtra
    public var totalSpeechVocabSize: Int { speechTokenSize + speechTokenExtra }

    /// Special token indices
    public var sosToken: Int { speechTokenSize }
    public var eosToken: Int { speechTokenSize + 1 }
    public var taskIdToken: Int { speechTokenSize + 2 }
    public var fillToken: Int { speechTokenSize + 3 }

    public init() {}
}

// MARK: - DiT Flow Config

public struct CosyVoiceDiTConfig: Codable, Sendable {
    public var dim: Int = 1024
    public var depth: Int = 22
    public var heads: Int = 16
    public var dimHead: Int = 64
    public var ffMult: Int = 2
    public var melDim: Int = 80
    public var muDim: Int = 80
    public var spkDim: Int = 80
    public var staticChunkSize: Int = 50
    public var freqEmbedDim: Int = 256

    /// Feedforward network dimension: dim * ffMult
    public var ffDim: Int { dim * ffMult }

    public init() {}
}

// MARK: - Flow Config

public struct CosyVoiceFlowConfig: Codable, Sendable {
    public var inputSize: Int = 512
    public var outputSize: Int = 80
    public var vocabSize: Int = 6561
    public var spkEmbedDim: Int = 192
    public var tokenFrameRate: Int = 25
    public var tokenMelRatio: Int = 2
    public var preLookaheadLen: Int = 3
    public var nTimesteps: Int = 10
    public var cfgRate: Float = 0.7
    public var dit: CosyVoiceDiTConfig = CosyVoiceDiTConfig()

    public init() {}
}

// MARK: - HiFi-GAN Config

public struct CosyVoiceHiFiGANConfig: Codable, Sendable {
    public var inChannels: Int = 80
    public var baseChannels: Int = 512
    public var nbHarmonics: Int = 8
    public var sampleRate: Int = 24000
    public var nsfAlpha: Float = 0.1
    public var nsfSigma: Float = 0.003
    public var nsfVoicedThreshold: Float = 10.0
    public var upsampleRates: [Int] = [8, 5, 3]
    public var upsampleKernelSizes: [Int] = [16, 11, 7]
    public var istftNFFT: Int = 16
    public var istftHopLen: Int = 4
    public var resblockKernelSizes: [Int] = [3, 7, 11]
    public var resblockDilationSizes: [[Int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
    public var sourceResblockKernelSizes: [Int] = [7, 7, 11]
    public var lreluSlope: Float = 0.1
    public var audioLimit: Float = 0.99
    public var convPreLookRight: Int = 4

    /// Total upsample factor: product of all upsample rates
    public var totalUpsampleFactor: Int { upsampleRates.reduce(1, *) }

    public init() {}
}

// MARK: - Mel Config

public struct CosyVoiceMelConfig: Codable, Sendable {
    public var nFFT: Int = 1920
    public var numMels: Int = 80
    public var hopSize: Int = 480
    public var winSize: Int = 1920
    public var fMin: Int = 0
    public var sampleRate: Int = 24000

    public init() {}
}

// MARK: - Sampling Config

public struct CosyVoiceSamplingConfig: Codable, Sendable {
    public var topP: Float = 0.8
    public var topK: Int = 25
    public var winSize: Int = 10
    public var tauR: Float = 0.1
    public var minTokenTextRatio: Float = 2.0
    public var maxTokenTextRatio: Float = 20.0

    public init() {}
}

// MARK: - Top-level Config

public struct CosyVoiceConfig: Codable, Sendable {
    public var llm: CosyVoiceLLMConfig
    public var flow: CosyVoiceFlowConfig
    public var hifigan: CosyVoiceHiFiGANConfig
    public var mel: CosyVoiceMelConfig
    public var sampling: CosyVoiceSamplingConfig
    public var sampleRate: Int = 24000
    public var chunkSize: Int = 25

    public init(
        llm: CosyVoiceLLMConfig = CosyVoiceLLMConfig(),
        flow: CosyVoiceFlowConfig = CosyVoiceFlowConfig(),
        hifigan: CosyVoiceHiFiGANConfig = CosyVoiceHiFiGANConfig(),
        mel: CosyVoiceMelConfig = CosyVoiceMelConfig(),
        sampling: CosyVoiceSamplingConfig = CosyVoiceSamplingConfig()
    ) {
        self.llm = llm
        self.flow = flow
        self.hifigan = hifigan
        self.mel = mel
        self.sampling = sampling
    }

    public static var `default`: CosyVoiceConfig { CosyVoiceConfig() }
}
