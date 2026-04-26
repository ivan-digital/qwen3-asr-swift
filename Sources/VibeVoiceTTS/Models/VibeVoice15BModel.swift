import Foundation
import MLX
import MLXNN
import MLXRandom

/// 1.5B model topology: one unified Qwen2 stack, dual audio encoders.
public class VibeVoice15BModel: Module {
    public let config: VibeVoiceConfiguration

    /// Single Qwen2.5 backbone — 28 layers @ hidden_size 1536 for 1.5B.
    @ModuleInfo(key: "language_model") public var languageModel: Qwen2Model

    @ModuleInfo(key: "acoustic_tokenizer") public var acousticTokenizer: VibeVoiceAcousticTokenizer
    @ModuleInfo(key: "acoustic_connector") public var acousticConnector: SpeechConnector

    @ModuleInfo(key: "semantic_tokenizer") public var semanticTokenizer: VibeVoiceSemanticTokenizer
    @ModuleInfo(key: "semantic_connector") public var semanticConnector: SpeechConnector

    @ModuleInfo(key: "prediction_head") public var predictionHead: VibeVoiceDiffusionHead

    public var speechScalingFactor: MLXArray = MLXArray(1.0)
    public var speechBiasFactor: MLXArray = MLXArray(0.0)
    public var noiseScheduler: DPMSolverMultistepScheduler

    public init(_ config: VibeVoiceConfiguration) throws {
        self.config = config

        guard let semCfg = config.semanticTokenizerConfig,
              let semDim = config.semanticVaeDim else {
            throw VibeVoiceError.modelNotInitialized(
                component: "1.5B requires semantic_tokenizer_config + semantic_vae_dim"
            )
        }

        // Unified LM uses ALL hidden layers (no split).
        _languageModel.wrappedValue = Qwen2Model(config.decoderConfig)

        _acousticTokenizer.wrappedValue = VibeVoiceAcousticTokenizer(config.acousticTokenizerConfig)
        _acousticConnector.wrappedValue = SpeechConnector(
            inputDim: config.acousticVaeDim,
            outputDim: config.decoderConfig.hiddenSize
        )

        _semanticTokenizer.wrappedValue = VibeVoiceSemanticTokenizer(semCfg)
        _semanticConnector.wrappedValue = SpeechConnector(
            inputDim: semDim,
            outputDim: config.decoderConfig.hiddenSize
        )

        _predictionHead.wrappedValue = VibeVoiceDiffusionHead(config.diffusionHeadConfig)

        self.noiseScheduler = try DPMSolverMultistepScheduler(
            numTrainTimesteps: config.diffusionHeadConfig.ddpmNumSteps,
            betaSchedule: config.diffusionHeadConfig.ddpmBetaSchedule,
            predictionType: config.diffusionHeadConfig.predictionType
        )

        super.init()
    }
}

/// 1.5B inference: structured prompt + dual-encoder prefill + LM token sampling
/// with branched generation (diffuse on `<speech_diffusion>`, terminate on
/// `<speech_end>`, advance on text tokens).
public class VibeVoice15BInference {
    public let model: VibeVoice15BModel
    public let numInferenceSteps: Int
    public let cfgScale: Float

    /// Per-layer KV cache for the unified LM.
    public var lmCache: [KVCacheSimple] = []
    /// Negative-conditioning KV cache (single-token <negative_text> baseline).
    public var negLmCache: [KVCacheSimple] = []

    private var cachedTimesteps: [Int32] = []

    public init(model: VibeVoice15BModel, numInferenceSteps: Int = 20, cfgScale: Float = 1.5) {
        self.model = model
        self.numInferenceSteps = numInferenceSteps
        self.cfgScale = cfgScale
        model.noiseScheduler.setTimesteps(numInferenceSteps: numInferenceSteps)
        self.cachedTimesteps = model.noiseScheduler.timesteps.asArray(Int32.self)
    }

    public func resetCaches() {
        let nLayers = model.config.decoderConfig.hiddenLayers
        lmCache = (0..<nLayers).map { _ in KVCacheSimple() }
        negLmCache = (0..<nLayers).map { _ in KVCacheSimple() }
    }

    /// Forward `input_ids` through the unified LM, with audio embeddings
    /// inserted at positions where `audioMask` is 1. Returns the final hidden
    /// state for all positions `[1, L, hidden_size]`.
    public func forwardWithAudio(
        inputIds: MLXArray,
        audioEmbeddings: MLXArray?,
        audioMask: MLXArray?,
        cache: inout [KVCacheSimple]
    ) -> MLXArray {
        var embeds = model.languageModel.embedTokens(inputIds)
        if let audio = audioEmbeddings, let mask = audioMask {
            embeds = mergeAudio(embeds: embeds, audio: audio, mask: mask)
        }
        return model.languageModel.forwardWithEmbeddings(embeds, cache: cache)
    }

    /// Diffuse a single acoustic latent conditioned on the LM hidden state.
    public func sampleSpeechLatent(condition: MLXArray, negCondition: MLXArray) throws -> MLXArray {
        let batchSize = condition.dim(0)
        let latentDim = model.config.diffusionHeadConfig.latentSize
        model.noiseScheduler.reset()

        let combinedCond = concatenated([condition, negCondition], axis: 0)
        var speech = MLXRandom.normal([batchSize, latentDim], dtype: condition.dtype)
        var prevX0: MLXArray? = nil

        for stepIdx in 0..<numInferenceSteps {
            let tVal = Float(cachedTimesteps[stepIdx])
            let timesteps = MLXArray([tVal, tVal])
            let combined = concatenated([speech, speech], axis: 0)
            let eps = model.predictionHead(
                noisyImages: combined, timesteps: timesteps, condition: combinedCond
            )
            let condEps = eps[0..<batchSize]
            let uncondEps = eps[batchSize...]
            let guidedEps = uncondEps + cfgScale * (condEps - uncondEps)
            let fullEps = concatenated([guidedEps, guidedEps], axis: 0)
            let (newSpeech, x0) = try model.noiseScheduler.stepGPU(
                modelOutput: fullEps,
                stepIdx: stepIdx,
                sample: concatenated([speech, speech], axis: 0),
                prevX0: prevX0
            )
            speech = newSpeech[0..<batchSize]
            prevX0 = x0[0..<batchSize]
        }
        return speech
    }

    private func mergeAudio(embeds: MLXArray, audio: MLXArray, mask: MLXArray) -> MLXArray {
        let L = embeds.dim(1)
        let D = embeds.dim(2)
        let A = audio.dim(1)
        let cumsum = mask.cumsum(axis: 1)
        let speechIdxRaw = cumsum - 1
        let speechIdxClipped = clip(speechIdxRaw, min: 0, max: A - 1)
        let gathered = audio[0, speechIdxClipped[0], 0...]
        let maskFloat = mask.asType(embeds.dtype).reshaped([1, L, 1])
        return maskFloat * gathered.reshaped([1, L, D]) + (1.0 - maskFloat) * embeds
    }
}
