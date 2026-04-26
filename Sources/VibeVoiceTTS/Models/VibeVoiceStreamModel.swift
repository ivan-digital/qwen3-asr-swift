import Foundation
import MLX
import MLXFast
import MLXNN
import MLXRandom

public class SpeechConnector: Module {
    @ModuleInfo(key: "fc1") public var fc1: Linear
    public let norm: RMSNorm
    @ModuleInfo(key: "fc2") public var fc2: Linear

    public init(inputDim: Int, outputDim: Int) {
        _fc1.wrappedValue = Linear(inputDim, outputDim, bias: true)
        self.norm = RMSNorm(dimensions: outputDim, eps: 1e-6)
        _fc2.wrappedValue = Linear(outputDim, outputDim, bias: true)
        super.init()
    }

    public func callAsFunction(_ features: MLXArray) -> MLXArray {
        var x = fc1(features)
        x = norm(x)
        x = fc2(x)
        return x
    }
}

public struct VibeVoiceConfiguration: Codable {
    public var decoderConfig: Qwen2Configuration
    public var acousticTokenizerConfig: AcousticTokenizerConfiguration
    public var diffusionHeadConfig: DiffusionHeadConfiguration
    public var ttsBackboneNumHiddenLayers: Int
    public var acousticVaeDim: Int
    /// Optional 1.5B-style semantic tokenizer config (encoder-only). Absent on
    /// 0.5B Realtime variants.
    public var semanticTokenizerConfig: AcousticTokenizerConfiguration?
    /// Optional 1.5B-style semantic VAE dim (typically 128). Absent on 0.5B.
    public var semanticVaeDim: Int?

    enum CodingKeys: String, CodingKey {
        case decoderConfig = "decoder_config"
        case acousticTokenizerConfig = "acoustic_tokenizer_config"
        case diffusionHeadConfig = "diffusion_head_config"
        case ttsBackboneNumHiddenLayers = "tts_backbone_num_hidden_layers"
        case acousticVaeDim = "acoustic_vae_dim"
        case semanticTokenizerConfig = "semantic_tokenizer_config"
        case semanticVaeDim = "semantic_vae_dim"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        decoderConfig = try container.decode(Qwen2Configuration.self, forKey: .decoderConfig)
        acousticTokenizerConfig = try container.decode(AcousticTokenizerConfiguration.self, forKey: .acousticTokenizerConfig)
        diffusionHeadConfig = try container.decode(DiffusionHeadConfiguration.self, forKey: .diffusionHeadConfig)
        acousticVaeDim = try container.decodeIfPresent(Int.self, forKey: .acousticVaeDim) ?? 64
        semanticTokenizerConfig = try container.decodeIfPresent(AcousticTokenizerConfiguration.self, forKey: .semanticTokenizerConfig)
        semanticVaeDim = try container.decodeIfPresent(Int.self, forKey: .semanticVaeDim)

        if let ttsLayers = try container.decodeIfPresent(Int.self, forKey: .ttsBackboneNumHiddenLayers) {
            ttsBackboneNumHiddenLayers = ttsLayers
        } else {
            let totalLayers = decoderConfig.hiddenLayers
            ttsBackboneNumHiddenLayers = max(totalLayers - 4, totalLayers * 3 / 4)
        }
    }

    public init(
        decoderConfig: Qwen2Configuration = Qwen2Configuration(),
        acousticTokenizerConfig: AcousticTokenizerConfiguration = AcousticTokenizerConfiguration(),
        diffusionHeadConfig: DiffusionHeadConfiguration = DiffusionHeadConfiguration(),
        ttsBackboneNumHiddenLayers: Int = 20,
        acousticVaeDim: Int = 64,
        semanticTokenizerConfig: AcousticTokenizerConfiguration? = nil,
        semanticVaeDim: Int? = nil
    ) {
        self.decoderConfig = decoderConfig
        self.acousticTokenizerConfig = acousticTokenizerConfig
        self.diffusionHeadConfig = diffusionHeadConfig
        self.ttsBackboneNumHiddenLayers = ttsBackboneNumHiddenLayers
        self.acousticVaeDim = acousticVaeDim
        self.semanticTokenizerConfig = semanticTokenizerConfig
        self.semanticVaeDim = semanticVaeDim
    }

    /// Convenience: does this config require dual-encoder voice prefill (1.5B)?
    public var hasSemanticTokenizer: Bool {
        semanticTokenizerConfig != nil && semanticVaeDim != nil
    }
}

public class VibeVoiceStreamModel: Module {
    public let config: VibeVoiceConfiguration

    @ModuleInfo(key: "language_model") public var languageModel: Qwen2Model

    @ModuleInfo(key: "tts_language_model") public var ttsLanguageModel: Qwen2Model

    @ModuleInfo(key: "tts_input_types") public var ttsInputTypes: Embedding

    @ModuleInfo(key: "acoustic_tokenizer") public var acousticTokenizer: VibeVoiceAcousticTokenizer

    @ModuleInfo(key: "acoustic_connector") public var acousticConnector: SpeechConnector

    /// Present only on 1.5B variants. Encodes reference audio into a 128-dim
    /// semantic latent (ASR-trained, captures phonemes/content).
    @ModuleInfo(key: "semantic_tokenizer") public var semanticTokenizer: VibeVoiceSemanticTokenizer?

    /// Present only on 1.5B variants. Projects semantic latents to hidden_size
    /// to be summed with acoustic embeddings during voice-prompt prefill.
    @ModuleInfo(key: "semantic_connector") public var semanticConnector: SpeechConnector?

    @ModuleInfo(key: "prediction_head") public var predictionHead: VibeVoiceDiffusionHead

    /// 1.5B does not ship an EOS classifier — generation stops via max_tokens.
    /// 0.5B Realtime ships one and we use it.
    @ModuleInfo(key: "tts_eos_classifier") public var eosClassifier: EOSClassifier?

    /// Set to true after weight load if the bundle contains eos_classifier weights.
    public var hasEosClassifier: Bool = false

    public var speechScalingFactor: MLXArray = MLXArray(1.0)
    public var speechBiasFactor: MLXArray = MLXArray(0.0)

    public var noiseScheduler: DPMSolverMultistepScheduler

    public init(_ config: VibeVoiceConfiguration) throws {
        self.config = config

        var lmConfig = config.decoderConfig
        let lmBackboneNumHiddenLayers = config.decoderConfig.hiddenLayers - config.ttsBackboneNumHiddenLayers
        lmConfig.hiddenLayers = lmBackboneNumHiddenLayers

        _languageModel.wrappedValue = Qwen2Model(lmConfig)

        var ttsLmConfig = config.decoderConfig
        ttsLmConfig.hiddenLayers = config.ttsBackboneNumHiddenLayers

        _ttsLanguageModel.wrappedValue = Qwen2Model(ttsLmConfig)

        _ttsInputTypes.wrappedValue = Embedding(embeddingCount: 2, dimensions: config.decoderConfig.hiddenSize)

        _acousticTokenizer.wrappedValue = VibeVoiceAcousticTokenizer(config.acousticTokenizerConfig)

        _acousticConnector.wrappedValue = SpeechConnector(
            inputDim: config.acousticVaeDim,
            outputDim: config.decoderConfig.hiddenSize
        )

        if let semConfig = config.semanticTokenizerConfig, let semDim = config.semanticVaeDim {
            // 1.5B variant — instantiate the semantic encoder + connector.
            _semanticTokenizer.wrappedValue = VibeVoiceSemanticTokenizer(semConfig)
            _semanticConnector.wrappedValue = SpeechConnector(
                inputDim: semDim,
                outputDim: config.decoderConfig.hiddenSize
            )
        }

        _predictionHead.wrappedValue = VibeVoiceDiffusionHead(config.diffusionHeadConfig)

        _eosClassifier.wrappedValue = EOSClassifier(hiddenSize: config.decoderConfig.hiddenSize)

        self.noiseScheduler = try DPMSolverMultistepScheduler(
            numTrainTimesteps: config.diffusionHeadConfig.ddpmNumSteps,
            betaSchedule: config.diffusionHeadConfig.ddpmBetaSchedule,
            predictionType: config.diffusionHeadConfig.predictionType
        )

        super.init()
    }

    public func getInputEmbeddings(_ inputIds: MLXArray) -> MLXArray {
        languageModel.embedTokens(inputIds)
    }

}
public class VibeVoiceStreamInference {
    public let model: VibeVoiceStreamModel
    public let numInferenceSteps: Int
    public let cfgScale: Float

    internal var lmCache: [KVCacheSimple] = []
    internal var ttsLmCache: [KVCacheSimple] = []
    internal var negLmCache: [KVCacheSimple] = []
    internal var negTtsLmCache: [KVCacheSimple] = []

    internal var lmLastHidden: MLXArray?
    internal var ttsLmLastHidden: MLXArray?
    internal var negTtsLmLastHidden: MLXArray?

    private var cachedTimesteps: [Int32] = []

    public init(model: VibeVoiceStreamModel, numInferenceSteps: Int = 20, cfgScale: Float = 3.0) {
        self.model = model
        self.numInferenceSteps = numInferenceSteps
        self.cfgScale = cfgScale

        model.noiseScheduler.setTimesteps(numInferenceSteps: numInferenceSteps)
        self.cachedTimesteps = model.noiseScheduler.timesteps.asArray(Int32.self)
    }

    public func loadVoiceCache(from path: String) throws {
        let url = URL(fileURLWithPath: path)

        let tensors = try MLX.loadArrays(url: url)

        let lmLayers = model.config.decoderConfig.hiddenLayers - model.config.ttsBackboneNumHiddenLayers
        let ttsLmLayers = model.config.ttsBackboneNumHiddenLayers

        lmCache = (0..<lmLayers).map { _ in KVCacheSimple() }
        ttsLmCache = (0..<ttsLmLayers).map { _ in KVCacheSimple() }
        negLmCache = (0..<lmLayers).map { _ in KVCacheSimple() }
        negTtsLmCache = (0..<ttsLmLayers).map { _ in KVCacheSimple() }

        lmLastHidden = tensors["lm_hidden"]
        for i in 0..<lmLayers {
            guard let key = tensors["lm_key_\(i)"], let value = tensors["lm_value_\(i)"] else {
                throw VibeVoiceError.weightsMissing(key: "lm_key_\(i) or lm_value_\(i)")
            }
            lmCache[i].initialize(keys: key, values: value)
        }

        ttsLmLastHidden = tensors["tts_lm_hidden"]
        for i in 0..<ttsLmLayers {
            guard let key = tensors["tts_lm_key_\(i)"], let value = tensors["tts_lm_value_\(i)"] else {
                throw VibeVoiceError.weightsMissing(key: "tts_lm_key_\(i) or tts_lm_value_\(i)")
            }
            ttsLmCache[i].initialize(keys: key, values: value)
        }

        for i in 0..<lmLayers {
            guard let key = tensors["neg_lm_key_\(i)"], let value = tensors["neg_lm_value_\(i)"] else {
                throw VibeVoiceError.weightsMissing(key: "neg_lm_key_\(i) or neg_lm_value_\(i)")
            }
            negLmCache[i].initialize(keys: key, values: value)
        }

        negTtsLmLastHidden = tensors["neg_tts_lm_hidden"]
        for i in 0..<ttsLmLayers {
            guard let key = tensors["neg_tts_lm_key_\(i)"], let value = tensors["neg_tts_lm_value_\(i)"] else {
                throw VibeVoiceError.weightsMissing(key: "neg_tts_lm_key_\(i) or neg_tts_lm_value_\(i)")
            }
            negTtsLmCache[i].initialize(keys: key, values: value)
        }
    }

    public func resetCaches() {
        let lmLayers = model.config.decoderConfig.hiddenLayers - model.config.ttsBackboneNumHiddenLayers
        lmCache = (0..<lmLayers).map { _ in KVCacheSimple() }
        ttsLmCache = (0..<model.config.ttsBackboneNumHiddenLayers).map { _ in KVCacheSimple() }
        negLmCache = (0..<lmLayers).map { _ in KVCacheSimple() }
        negTtsLmCache = (0..<model.config.ttsBackboneNumHiddenLayers).map { _ in KVCacheSimple() }

        lmLastHidden = nil
        ttsLmLastHidden = nil
        negTtsLmLastHidden = nil
    }

    internal func forwardLM(inputIds: MLXArray, cache: inout [KVCacheSimple]) -> MLXArray {
        let embeddings = model.languageModel.embedTokens(inputIds)
        return model.languageModel.forwardWithEmbeddings(embeddings, cache: cache, applyFinalNorm: false)
    }

    internal func forwardTTSLM(
        inputIds: MLXArray,
        lmHiddenState: MLXArray,
        ttsTextMask: MLXArray,
        cache: inout [KVCacheSimple]
    ) -> MLXArray {
        var inputsEmbeds = model.languageModel.embedTokens(inputIds)

        let startIdx = inputsEmbeds.dim(1) - lmHiddenState.dim(1)
        if startIdx > 0 {
            let prefix = inputsEmbeds[0..., 0..<startIdx, 0...]
            inputsEmbeds = concatenated([prefix, lmHiddenState], axis: 1)
        } else {
            inputsEmbeds = lmHiddenState
        }

        let ttsTypeEmbed = model.ttsInputTypes(ttsTextMask.asType(.int32))
        inputsEmbeds = inputsEmbeds + ttsTypeEmbed

        return model.ttsLanguageModel.forwardWithEmbeddings(inputsEmbeds, cache: cache)
    }

    internal func forwardTTSLMWithAcoustic(
        acousticEmbed: MLXArray,
        cache: inout [KVCacheSimple]
    ) -> MLXArray {
        let batchSize = acousticEmbed.dim(0)
        let speechTypeMask = MLXArray.zeros([batchSize, 1], dtype: .int32)
        let ttsTypeEmbed = model.ttsInputTypes(speechTypeMask.asType(.int32))
        let inputsEmbeds = acousticEmbed + ttsTypeEmbed

        return model.ttsLanguageModel.forwardWithEmbeddings(inputsEmbeds, cache: cache)
    }

    private func generateSpeechTokensCore(
        maxSpeechTokens: Int,
        acousticCache: StreamingConvCache?,
        collectLatentsOnly: Bool
    ) throws -> (scaledLatents: [MLXArray], audioChunks: [MLXArray], tokenCount: Int, eosDetected: Bool) {
        var scaledLatentChunks: [MLXArray] = []
        var audioChunks: [MLXArray] = []
        var tokenCount = 0
        var eosDetected = false

        while tokenCount < maxSpeechTokens {
            guard let ttsHidden = ttsLmLastHidden else {
                throw VibeVoiceError.modelNotInitialized(component: "TTS LM hidden state")
            }
            guard let negTtsHidden = negTtsLmLastHidden else {
                throw VibeVoiceError.modelNotInitialized(component: "Negative TTS LM hidden state")
            }

            let lastIdx = ttsHidden.dim(1) - 1
            let condition = ttsHidden[0..., lastIdx...(lastIdx), 0...].squeezed(axis: 1)

            let negLastIdx = negTtsHidden.dim(1) - 1
            let negCondition = negTtsHidden[0..., negLastIdx...(negLastIdx), 0...].squeezed(axis: 1)

            let speechLatent2D = try sampleSpeechLatent(
                condition: condition,
                negCondition: negCondition
            )

            let speechLatent = expandedDimensions(speechLatent2D, axis: 1)

            let scaledLatent = speechLatent / model.speechScalingFactor - model.speechBiasFactor

            if collectLatentsOnly {
                scaledLatentChunks.append(scaledLatent)
            } else if let cache = acousticCache {
                let audioChunk = model.acousticTokenizer.decode(scaledLatent, cache: cache, useCache: true)
                audioChunks.append(audioChunk)
            }

            let acousticEmbed = model.acousticConnector(speechLatent)

            ttsLmLastHidden = forwardTTSLMWithAcoustic(
                acousticEmbed: acousticEmbed,
                cache: &ttsLmCache
            )

            negTtsLmLastHidden = forwardTTSLMWithAcoustic(
                acousticEmbed: acousticEmbed,
                cache: &negTtsLmCache
            )

            tokenCount += 1

            if let ttsHidden = ttsLmLastHidden, let negTtsHidden = negTtsLmLastHidden {
                eval(ttsHidden, negTtsHidden)
            }

            if try checkEndOfSpeech() {
                eosDetected = true
                break
            }
        }

        return (scaledLatentChunks, audioChunks, tokenCount, eosDetected)
    }

    internal func checkEndOfSpeech() throws -> Bool {
        // 1.5B doesn't ship an EOS classifier — generation stops via maxSpeechTokens.
        guard let classifier = model.eosClassifier, model.hasEosClassifier else {
            return false
        }
        guard let ttsHidden = ttsLmLastHidden else {
            throw VibeVoiceError.modelNotInitialized(component: "TTS LM")
        }
        let lastIdx = ttsHidden.dim(1) - 1
        let eosHidden = ttsHidden[0..., lastIdx...(lastIdx), 0...].squeezed(axis: 1)

        let eosLogits = classifier(eosHidden)
        let eosProb = sigmoid(eosLogits)
        eval(eosProb)
        let prob = eosProb[0, 0].item(Float.self)

        return prob > 0.5
    }

    /// Generate audio from tokenized text using the pre-loaded voice cache.
    ///
    /// Accumulates all chunks into a single `[1, 1, samples]` MLXArray. For
    /// chunk-streaming (play while generating), use `generateWithVoiceCacheStream`.
    public func generateWithVoiceCache(
        tokenIds ttsTextIds: MLXArray,
        maxSpeechTokens: Int = 500
    ) throws -> MLXArray {
        var audioChunks: [MLXArray] = []
        try generateWithVoiceCacheCore(
            tokenIds: ttsTextIds,
            maxSpeechTokens: maxSpeechTokens
        ) { chunk in
            audioChunks.append(chunk)
        }
        if audioChunks.isEmpty {
            return MLXArray.zeros([ttsTextIds.dim(0), 1, 0])
        }
        return concatenated(audioChunks, axis: -1)
    }

    /// Generate audio from tokenized text and emit each acoustic-decoder chunk
    /// via the callback as it's produced. Use this for play-while-generating
    /// integrations (e.g. `AudioCommon.StreamingAudioPlayer.scheduleChunk(_:)`).
    public func generateWithVoiceCacheStream(
        tokenIds ttsTextIds: MLXArray,
        maxSpeechTokens: Int = 500,
        onChunk: (MLXArray) -> Void
    ) throws {
        try generateWithVoiceCacheCore(
            tokenIds: ttsTextIds,
            maxSpeechTokens: maxSpeechTokens,
            onChunk: onChunk
        )
    }

    /// Core inference loop. `onChunk` is called once per produced audio chunk
    /// (shape `[B, 1, samples]`).
    private func generateWithVoiceCacheCore(
        tokenIds ttsTextIds: MLXArray,
        maxSpeechTokens: Int,
        onChunk: (MLXArray) -> Void
    ) throws {
        guard ttsLmLastHidden != nil, negTtsLmLastHidden != nil else {
            throw VibeVoiceError.voiceCacheNotLoaded
        }

        let batchSize = ttsTextIds.dim(0)
        let totalTextTokens = ttsTextIds.dim(1)
        let acousticCache = StreamingConvCache()

        var textWindowIndex = 0
        var totalGeneratedSpeech = 0
        var finished = false

        while !finished {
            let windowStart = textWindowIndex * TTSConstants.textWindowSize
            let windowEnd = min((textWindowIndex + 1) * TTSConstants.textWindowSize, totalTextTokens)

            if windowStart < totalTextTokens {
                let curTextIds = ttsTextIds[0..., windowStart..<windowEnd]
                let curWindowSize = windowEnd - windowStart

                if curWindowSize > 0 {
                    lmLastHidden = forwardLM(inputIds: curTextIds, cache: &lmCache)

                    guard let lmHidden = lmLastHidden else {
                        throw VibeVoiceError.modelNotInitialized(component: "LM hidden state")
                    }

                    let textMask = MLXArray.ones([batchSize, curWindowSize], dtype: .int32)
                    ttsLmLastHidden = forwardTTSLM(
                        inputIds: curTextIds,
                        lmHiddenState: lmHidden,
                        ttsTextMask: textMask,
                        cache: &ttsLmCache
                    )
                }

                textWindowIndex += 1
            }

            for _ in 0..<TTSConstants.speechWindowSize {
                if totalGeneratedSpeech >= maxSpeechTokens {
                    finished = true
                    break
                }

                guard let ttsHidden = ttsLmLastHidden else {
                    throw VibeVoiceError.modelNotInitialized(component: "TTS LM hidden state")
                }
                guard let negTtsHidden = negTtsLmLastHidden else {
                    throw VibeVoiceError.modelNotInitialized(component: "Negative TTS LM hidden state")
                }

                let lastIdx = ttsHidden.dim(1) - 1
                let condition = ttsHidden[0..., lastIdx...(lastIdx), 0...].squeezed(axis: 1)

                let negLastIdx = negTtsHidden.dim(1) - 1
                let negCondition = negTtsHidden[0..., negLastIdx...(negLastIdx), 0...].squeezed(axis: 1)

                let speechLatent2D = try sampleSpeechLatent(
                    condition: condition,
                    negCondition: negCondition
                )

                let speechLatent = expandedDimensions(speechLatent2D, axis: 1)
                let scaledLatent = speechLatent / model.speechScalingFactor - model.speechBiasFactor
                let audioChunk = model.acousticTokenizer.decode(scaledLatent, cache: acousticCache, useCache: true)
                eval(audioChunk)
                onChunk(audioChunk)

                let acousticEmbed = model.acousticConnector(speechLatent)
                ttsLmLastHidden = forwardTTSLMWithAcoustic(
                    acousticEmbed: acousticEmbed,
                    cache: &ttsLmCache
                )
                negTtsLmLastHidden = forwardTTSLMWithAcoustic(
                    acousticEmbed: acousticEmbed,
                    cache: &negTtsLmCache
                )

                totalGeneratedSpeech += 1

                if try checkEndOfSpeech() {
                    finished = true
                    break
                }
            }

            if windowStart >= totalTextTokens {
                // No more text to consume; if speech tokens are also capped,
                // break to avoid infinite loop when EOS never fires.
                if totalGeneratedSpeech >= maxSpeechTokens { finished = true }
            }
        }
        _ = batchSize  // silence unused-warning when no chunks emitted
    }

    public func generate(tokenIds ttsTextIds: MLXArray, maxSpeechTokens: Int = 500) throws -> MLXArray {
        resetCaches()

        let batchSize = ttsTextIds.dim(0)
        let totalTextTokens = ttsTextIds.dim(1)

        let negTokenIds = MLXArray([Int32(TokenConstants.negativeTextId)]).reshaped([1, 1])
        let negLmHidden = forwardLM(inputIds: negTokenIds, cache: &negLmCache)

        let negTextMask = MLXArray.ones([batchSize, 1], dtype: .int32)
        negTtsLmLastHidden = forwardTTSLM(
            inputIds: negTokenIds,
            lmHiddenState: negLmHidden,
            ttsTextMask: negTextMask,
            cache: &negTtsLmCache
        )

        let acousticCache = StreamingConvCache()

        var audioChunks: [MLXArray] = []
        var textWindowIndex = 0
        var totalGeneratedSpeech = 0
        var finished = false

        while !finished {
            let windowStart = textWindowIndex * TTSConstants.textWindowSize
            let windowEnd = min((textWindowIndex + 1) * TTSConstants.textWindowSize, totalTextTokens)

            if windowStart >= totalTextTokens {
                break
            }

            let curTextIds = ttsTextIds[0..., windowStart..<windowEnd]
            let curWindowSize = windowEnd - windowStart

            if curWindowSize > 0 {
                lmLastHidden = forwardLM(inputIds: curTextIds, cache: &lmCache)

                guard let lmHidden = lmLastHidden else {
                    throw VibeVoiceError.modelNotInitialized(component: "LM hidden state")
                }

                let textMask = MLXArray.ones([batchSize, curWindowSize], dtype: .int32)
                ttsLmLastHidden = forwardTTSLM(
                    inputIds: curTextIds,
                    lmHiddenState: lmHidden,
                    ttsTextMask: textMask,
                    cache: &ttsLmCache
                )

                if let ttsHidden = ttsLmLastHidden {
                    eval(ttsHidden)
                }
            }

            textWindowIndex += 1

            for _ in 0..<TTSConstants.speechWindowSize {
                if totalGeneratedSpeech >= maxSpeechTokens {
                    finished = true
                    break
                }

                let (_, chunks, count, eosDetected) = try generateSpeechTokensCore(
                    maxSpeechTokens: 1,
                    acousticCache: acousticCache,
                    collectLatentsOnly: false
                )

                audioChunks.append(contentsOf: chunks)
                totalGeneratedSpeech += count

                if eosDetected {
                    finished = true
                    break
                }
            }
        }

        if audioChunks.isEmpty {
            return MLXArray.zeros([batchSize, 1, 0])
        }

        let audio = concatenated(audioChunks, axis: -1)
        eval(audio)
        return audio
    }

    internal func sampleSpeechLatent(condition: MLXArray, negCondition: MLXArray) throws -> MLXArray {
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
                noisyImages: combined,
                timesteps: timesteps,
                condition: combinedCond
            )

            let condEps = eps[0..<batchSize]
            let uncondEps = eps[batchSize...]
            let guidedEps = uncondEps + cfgScale * (condEps - uncondEps)

            let fullEps = concatenated([guidedEps, guidedEps], axis: 0)

            let (newSpeech, x0Pred) = try model.noiseScheduler.stepGPU(
                modelOutput: fullEps,
                stepIdx: stepIdx,
                sample: concatenated([speech, speech], axis: 0),
                prevX0: prevX0
            )

            speech = newSpeech[0..<batchSize]
            prevX0 = x0Pred[0..<batchSize]
        }

        return speech
    }

    public func generateSpeech(
        textEmbeddings: MLXArray,
        ttsInputTypes: MLXArray,
        numFrames: Int
    ) throws -> MLXArray {
        let batchSize = textEmbeddings.dim(0)

        let ttsTypeEmbed = model.ttsInputTypes(ttsInputTypes)
        let conditionedEmbeddings = textEmbeddings + ttsTypeEmbed

        let lmHidden = model.languageModel.forwardWithEmbeddings(conditionedEmbeddings)

        let ttsHidden = model.ttsLanguageModel.forwardWithEmbeddings(lmHidden)

        let latents = try generateLatentsDiffusion(
            condition: ttsHidden,
            numFrames: numFrames,
            batchSize: batchSize
        )

        let audio = model.acousticTokenizer.decode(latents)

        return audio
    }

    public func generateLatentsDiffusion(
        condition: MLXArray,
        negCondition: MLXArray? = nil,
        numFrames: Int,
        batchSize: Int
    ) throws -> MLXArray {
        let latentDim = model.config.diffusionHeadConfig.latentSize

        var latents = MLXRandom.normal([batchSize, numFrames, latentDim], dtype: condition.dtype)

        model.noiseScheduler.reset()

        var prevX0: MLXArray? = nil

        for stepIdx in 0..<numInferenceSteps {
            let tVal = Float(cachedTimesteps[stepIdx])
            let timesteps = MLXArray.ones([batchSize]) * tVal

            let modelOutput: MLXArray
            if let negCond = negCondition, cfgScale > 1.0 {
                let combinedCond = concatenated([condition, negCond], axis: 0)
                let combinedLatents = concatenated([latents, latents], axis: 0)
                let combinedTimesteps = concatenated([timesteps, timesteps], axis: 0)

                let combinedOutput = model.predictionHead(
                    noisyImages: combinedLatents,
                    timesteps: combinedTimesteps,
                    condition: combinedCond
                )

                let condOutput = combinedOutput[0..<batchSize]
                let uncondOutput = combinedOutput[batchSize...]
                modelOutput = uncondOutput + cfgScale * (condOutput - uncondOutput)
            } else {
                modelOutput = model.predictionHead(
                    noisyImages: latents,
                    timesteps: timesteps,
                    condition: condition
                )
            }

            let (newLatents, x0Pred) = try model.noiseScheduler.stepGPU(
                modelOutput: modelOutput,
                stepIdx: stepIdx,
                sample: latents,
                prevX0: prevX0
            )

            latents = newLatents
            prevX0 = x0Pred
        }

        eval(latents)
        return latents
    }

    public func scaleLatentsForDecoding(_ latents: MLXArray) -> MLXArray {
        return latents / model.speechScalingFactor - model.speechBiasFactor
    }
}

