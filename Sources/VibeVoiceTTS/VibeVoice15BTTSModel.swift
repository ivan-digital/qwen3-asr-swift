import AudioCommon
import Foundation
import MLX
import Tokenizers

/// Public 1.5B long-form TTS API. Uses the unified-LM `VibeVoice15BModel`
/// architecture (different from 0.5B Realtime — see VibeVoice15BModel.swift).
///
/// Usage:
/// ```swift
/// let tts = try await VibeVoice15BTTSModel.fromPretrained()
/// let pcm = try await tts.generate(
///     text: "Long-form English script.",
///     referenceAudio: pcm24k,
///     referenceTranscript: "the words spoken in the reference"
/// )
/// ```
public final class VibeVoice15BTTSModel {
    public struct Configuration: Sendable {
        public var modelId: String
        public var tokenizerModelId: String
        public var numInferenceSteps: Int
        public var cfgScale: Float
        public var maxSpeechTokens: Int

        public init(
            modelId: String = "aufklarer/VibeVoice-1.5B-MLX-INT4",
            tokenizerModelId: String = "Qwen/Qwen2.5-1.5B",
            numInferenceSteps: Int = 20,
            cfgScale: Float = 1.5,
            maxSpeechTokens: Int = 4000
        ) {
            self.modelId = modelId
            self.tokenizerModelId = tokenizerModelId
            self.numInferenceSteps = numInferenceSteps
            self.cfgScale = cfgScale
            self.maxSpeechTokens = maxSpeechTokens
        }
    }

    public let configuration: Configuration
    public let inference: VibeVoice15BInference
    private let tokenizer: Tokenizer

    private static let systemPrompt =
        " Transform the text provided by various speakers into speech output, utilizing the distinct voice of each respective speaker.\n"

    private init(
        configuration: Configuration,
        inference: VibeVoice15BInference,
        tokenizer: Tokenizer
    ) {
        self.configuration = configuration
        self.inference = inference
        self.tokenizer = tokenizer
    }

    public static func fromPretrained(
        configuration: Configuration = Configuration(),
        cacheDir: URL? = nil,
        offlineMode: Bool = false,
        progressHandler: ((Double, String) -> Void)? = nil
    ) async throws -> VibeVoice15BTTSModel {
        let modelCacheDir = try cacheDir
            ?? HuggingFaceDownloader.getCacheDirectory(for: configuration.modelId)
        if !HuggingFaceDownloader.weightsExist(in: modelCacheDir) {
            progressHandler?(0.0, "Downloading \(configuration.modelId)...")
            try await HuggingFaceDownloader.downloadWeights(
                modelId: configuration.modelId,
                to: modelCacheDir,
                additionalFiles: ["config.json", "preprocessor_config.json", "quantization.json"],
                offlineMode: offlineMode
            ) { fraction in
                progressHandler?(fraction * 0.7, "Downloading model...")
            }
        }

        progressHandler?(0.75, "Downloading tokenizer...")
        let tokenizerCacheDir = try HuggingFaceDownloader.getCacheDirectory(
            for: configuration.tokenizerModelId
        )
        if !HuggingFaceDownloader.weightsExist(in: tokenizerCacheDir) {
            try await HuggingFaceDownloader.downloadWeights(
                modelId: configuration.tokenizerModelId,
                to: tokenizerCacheDir,
                additionalFiles: [
                    "tokenizer.json", "tokenizer_config.json",
                    "vocab.json", "merges.txt",
                    "special_tokens_map.json", "added_tokens.json",
                ],
                offlineMode: offlineMode
            )
        }

        progressHandler?(0.85, "Loading model...")
        let model = try loadVibeVoice15BModel(from: modelCacheDir)
        let inference = VibeVoice15BInference(
            model: model,
            numInferenceSteps: configuration.numInferenceSteps,
            cfgScale: configuration.cfgScale
        )

        progressHandler?(0.95, "Loading tokenizer...")
        let tokenizer = try await AutoTokenizer.from(modelFolder: tokenizerCacheDir)

        progressHandler?(1.0, "Ready")
        return VibeVoice15BTTSModel(
            configuration: configuration,
            inference: inference,
            tokenizer: tokenizer
        )
    }

    /// Synthesize speech: encodes the reference audio through both encoders,
    /// builds the structured prompt, runs unified-LM forward + token sampling
    /// with `<speech_diffusion>`-branched diffusion until `<speech_end>` or
    /// `maxSpeechTokens` reached. Returns 24 kHz mono `[Float]` audio.
    public func generate(
        text: String,
        referenceAudio: [Float],
        referenceTranscript: String = "",
        sampleRate: Int = 24000
    ) async throws -> [Float] {
        let audio: [Float]
        if sampleRate != AudioConstants.sampleRate {
            audio = AudioFileLoader.resample(referenceAudio, from: sampleRate, to: AudioConstants.sampleRate)
        } else {
            audio = referenceAudio
        }
        guard !audio.isEmpty else {
            throw VibeVoiceError.modelNotInitialized(component: "reference audio empty")
        }

        inference.resetCaches()

        // 1) Encode through both audio encoders (use mean directly — 1.5B
        // convention per vllm_plugin/model.py).
        let audioArr = MLXArray(audio).reshaped([1, audio.count])
        let acMean = inference.model.acousticTokenizer.encode(audioArr)
        let semMean = inference.model.semanticTokenizer.encode(audioArr)
        eval(acMean, semMean)

        // 2) Compute combined audio embeddings = ac_connector + sem_connector.
        let acEmbeds = inference.model.acousticConnector(acMean)
        let semEmbeds = inference.model.semanticConnector(semMean)
        let audioEmbeds = acEmbeds + semEmbeds
        eval(audioEmbeds)
        let numVae = audioEmbeds.dim(1)

        // 3) Build structured prompt:
        //   <bos> system_prompt
        //   " Speaker 0:" <speech_start> [vae]*N <speech_end> "\n"
        //   " Text input:\n Speaker 0:<text>\n"
        //   " Speech output:\n" <speech_start>
        var ids: [Int32] = []
        var audioMask: [Bool] = []
        func append(_ chunk: [Int32]) {
            ids.append(contentsOf: chunk)
            audioMask.append(contentsOf: Array(repeating: false, count: chunk.count))
        }
        append(tokenizer.encode(text: Self.systemPrompt, addSpecialTokens: true).map(Int32.init))
        append(tokenizer.encode(text: " Speaker 0:", addSpecialTokens: false).map(Int32.init))
        append([Int32(TokenConstants.speechStartId)])
        ids.append(contentsOf: Array(repeating: Int32(TokenConstants.speechDiffusionId), count: numVae))
        audioMask.append(contentsOf: Array(repeating: true, count: numVae))
        append([Int32(TokenConstants.speechEndId)])
        append(tokenizer.encode(text: "\n", addSpecialTokens: false).map(Int32.init))
        append(tokenizer.encode(text: " Text input:\n Speaker 0:\(text)\n", addSpecialTokens: false).map(Int32.init))
        append(tokenizer.encode(text: " Speech output:\n", addSpecialTokens: false).map(Int32.init))
        append([Int32(TokenConstants.speechStartId)])

        let inputIds = MLXArray(ids).reshaped([1, ids.count])
        let mask = MLXArray(audioMask.map { $0 ? Int32(1) : Int32(0) }).reshaped([1, audioMask.count])

        // 4) Prefill — full unified-LM forward with audio replacement at vae positions.
        let prefillHidden = inference.forwardWithAudio(
            inputIds: inputIds,
            audioEmbeddings: audioEmbeds,
            audioMask: mask,
            cache: &inference.lmCache
        )

        // 5) Negative-conditioning prefill: single negative-text token through LM.
        let negToken = MLXArray([Int32(TokenConstants.negativeTextId)]).reshaped([1, 1])
        let negPrefillHidden = inference.forwardWithAudio(
            inputIds: negToken,
            audioEmbeddings: nil, audioMask: nil,
            cache: &inference.negLmCache
        )

        // 6) Generation loop: LM token sampling + branched on <speech_diffusion> / <speech_end> / text.
        let speechEnd = Int32(TokenConstants.speechEndId)
        let speechDiff = Int32(TokenConstants.speechDiffusionId)
        let embedTokens = inference.model.languageModel.embedTokens
        let acousticCache = StreamingConvCache()

        var audioChunks: [MLXArray] = []
        let lastIdx = prefillHidden.dim(1) - 1
        var currentHidden = prefillHidden[0..., lastIdx...(lastIdx), 0...]
        let negLastIdx = negPrefillHidden.dim(1) - 1
        var negCurrentHidden = negPrefillHidden[0..., negLastIdx...(negLastIdx), 0...]

        for _ in 0..<configuration.maxSpeechTokens {
            let logits = embedTokens.asLinear(currentHidden)  // [1, 1, vocab]
            let nextToken = logits.argMax(axis: -1)
            eval(nextToken)
            let tokenId = nextToken[0, 0].item(Int32.self)

            if tokenId == speechEnd { break }

            if tokenId == speechDiff {
                // Sample a new acoustic latent via diffusion conditioned on
                // current hidden. Decode to audio. Feed connector back as the
                // next input embedding.
                let cond = currentHidden.squeezed(axis: 1)
                let negCond = negCurrentHidden.squeezed(axis: 1)
                let latent2D = try inference.sampleSpeechLatent(condition: cond, negCondition: negCond)
                let latent = expandedDimensions(latent2D, axis: 1)  // [1,1,vae]
                let scaled = latent / inference.model.speechScalingFactor - inference.model.speechBiasFactor
                let chunk = inference.model.acousticTokenizer.decode(
                    scaled, cache: acousticCache, useCache: true
                )
                eval(chunk)
                audioChunks.append(chunk)

                // Feed acoustic_connector(latent) as next embed (no semantic
                // during generation — only acoustic per reference).
                let acEmbed = inference.model.acousticConnector(latent)
                currentHidden = inference.model.languageModel.forwardWithEmbeddings(
                    acEmbed, cache: inference.lmCache
                )
                negCurrentHidden = inference.model.languageModel.forwardWithEmbeddings(
                    acEmbed, cache: inference.negLmCache
                )
            } else {
                // Plain text token — embed + step.
                let tokenArr = MLXArray([tokenId]).reshaped([1, 1])
                currentHidden = inference.forwardWithAudio(
                    inputIds: tokenArr, audioEmbeddings: nil, audioMask: nil,
                    cache: &inference.lmCache
                )
                negCurrentHidden = inference.forwardWithAudio(
                    inputIds: tokenArr, audioEmbeddings: nil, audioMask: nil,
                    cache: &inference.negLmCache
                )
            }
        }

        if audioChunks.isEmpty { return [] }
        let full = concatenated(audioChunks, axis: -1)
        eval(full)
        return full.reshaped([-1]).asArray(Float.self)
    }
}

extension VibeVoice15BTTSModel: SpeechGenerationModel {
    public var sampleRate: Int { AudioConstants.sampleRate }

    /// Convenience for SpeechGenerationModel conformance — uses a no-op
    /// reference audio (silence). Real usage should pass real reference audio
    /// via `generate(text:referenceAudio:...)`.
    public func generate(text: String, language: String?) async throws -> [Float] {
        // Without reference audio the voice has no identity — produce 0.5 s
        // of mock silence as the prompt. Callers should use the explicit
        // `generate(text:referenceAudio:...)` overload for real use.
        let silence = [Float](repeating: 0, count: 12000)
        return try await generate(
            text: text,
            referenceAudio: silence,
            referenceTranscript: ""
        )
    }
}
