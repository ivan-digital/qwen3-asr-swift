import AudioCommon
import Foundation
import MLX
import Tokenizers

/// Top-level speech-swift wrapper around the VibeVoice streaming model.
///
/// Conforms to `AudioCommon.SpeechGenerationModel` so `VibeVoiceTTSModel`
/// drops into `ModelSet` and any API expecting a TTS model.
///
/// Usage:
/// ```swift
/// let tts = try await VibeVoiceTTSModel.fromPretrained()
/// try tts.loadVoice(from: voiceCacheURL)        // .safetensors voice cache
/// let pcm = try await tts.generate(text: "Hello world.", language: nil)
/// ```
public final class VibeVoiceTTSModel {

    public struct Configuration: Sendable {
        /// HuggingFace model id for the VibeVoice weights (MLX-compatible).
        public var modelId: String
        /// HuggingFace model id for the Qwen2.5 tokenizer.
        public var tokenizerModelId: String
        /// DPM-Solver inference steps. 10–20 typical; higher = quality over speed.
        public var numInferenceSteps: Int
        /// Classifier-free guidance scale. 1.3 is the Realtime-0.5B default.
        public var cfgScale: Float
        /// Cap on generated speech tokens per call.
        public var maxSpeechTokens: Int

        public init(
            modelId: String = "microsoft/VibeVoice-Realtime-0.5B",
            tokenizerModelId: String = "Qwen/Qwen2.5-0.5B",
            numInferenceSteps: Int = 20,
            cfgScale: Float = 1.3,
            maxSpeechTokens: Int = 500
        ) {
            self.modelId = modelId
            self.tokenizerModelId = tokenizerModelId
            self.numInferenceSteps = numInferenceSteps
            self.cfgScale = cfgScale
            self.maxSpeechTokens = maxSpeechTokens
        }

        /// Long-form VibeVoice 1.5B variant (EN/ZH, up to 90-min multi-speaker).
        public static let longForm1_5B = Configuration(
            modelId: "microsoft/VibeVoice-1.5B",
            tokenizerModelId: "Qwen/Qwen2.5-1.5B",
            numInferenceSteps: 20,
            cfgScale: 1.5,
            maxSpeechTokens: 4000
        )
    }

    public let configuration: Configuration
    public let inference: VibeVoiceStreamInference
    private let tokenizer: Tokenizer
    private var voiceLoaded: Bool = false

    private init(
        configuration: Configuration,
        inference: VibeVoiceStreamInference,
        tokenizer: Tokenizer
    ) {
        self.configuration = configuration
        self.inference = inference
        self.tokenizer = tokenizer
    }

    // MARK: - Loading

    /// Load a pretrained VibeVoice model.
    ///
    /// Downloads model weights + tokenizer from HuggingFace (or uses cache).
    public static func fromPretrained(
        configuration: Configuration = Configuration(),
        cacheDir: URL? = nil,
        offlineMode: Bool = false,
        progressHandler: ((Double, String) -> Void)? = nil
    ) async throws -> VibeVoiceTTSModel {
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
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "vocab.json",
                    "merges.txt",
                    "special_tokens_map.json",
                    "added_tokens.json",
                ],
                offlineMode: offlineMode
            )
        }

        progressHandler?(0.85, "Loading model...")
        let model = try loadVibeVoiceStreamModel(from: modelCacheDir)
        let inference = VibeVoiceStreamInference(
            model: model,
            numInferenceSteps: configuration.numInferenceSteps,
            cfgScale: configuration.cfgScale
        )

        progressHandler?(0.95, "Loading tokenizer...")
        let tokenizer = try await AutoTokenizer.from(modelFolder: tokenizerCacheDir)

        progressHandler?(1.0, "Ready")
        return VibeVoiceTTSModel(
            configuration: configuration,
            inference: inference,
            tokenizer: tokenizer
        )
    }

    // MARK: - Voice cache

    /// Load a voice cache (.safetensors) for speaker conditioning.
    ///
    /// Voice caches contain pre-computed KV caches + hidden states for a
    /// specific speaker, produced by running reference audio through the
    /// VibeVoice encoder offline. See `docs/models/vibevoice.md` for the format.
    public func loadVoice(from url: URL) throws {
        try inference.loadVoiceCache(from: url.path)
        voiceLoaded = true
    }

    /// Load a voice cache from a file path string.
    public func loadVoice(from path: String) throws {
        try loadVoice(from: URL(fileURLWithPath: path))
    }

    // MARK: - Generation

    /// Synthesize speech for the given text. Requires a voice cache to be
    /// loaded first via `loadVoice(from:)`.
    public func generate(text: String) async throws -> [Float] {
        guard voiceLoaded else {
            throw VibeVoiceError.voiceCacheNotLoaded
        }

        let ids = tokenizer.encode(text: text + "\n", addSpecialTokens: false)
        let int32Ids = ids.map { Int32($0) }
        let tokenIds = MLXArray(int32Ids).reshaped([1, int32Ids.count])

        let audio = try inference.generateWithVoiceCache(
            tokenIds: tokenIds,
            maxSpeechTokens: configuration.maxSpeechTokens
        )
        eval(audio)

        // Shape: [1, 1, samples] — flatten to [Float].
        let flat = audio.reshaped([-1])
        return flat.asArray(Float.self)
    }

    // MARK: - Voice cache creation

    /// Mint a voice cache from a reference audio recording + its transcript.
    ///
    /// Returns the cache as a tensor dict matching the format read by
    /// `loadVoice(from:)`. Save it via `saveVoiceCache(_:to:)`.
    ///
    /// The transcript should be the actual words spoken in `referenceAudio`.
    /// Audio is resampled to 24 kHz mono internally.
    ///
    /// - Parameters:
    ///   - referenceAudio: 1D PCM samples (mono).
    ///   - sampleRate: Source rate; resampled to 24 kHz if different.
    ///   - transcript: The text the speaker is saying in the audio.
    public func encodeVoice(
        referenceAudio: [Float],
        sampleRate: Int = 24000,
        transcript: String
    ) throws -> [String: MLXArray] {
        let audio: [Float]
        if sampleRate != AudioConstants.sampleRate {
            audio = AudioFileLoader.resample(
                referenceAudio,
                from: sampleRate,
                to: AudioConstants.sampleRate
            )
        } else {
            audio = referenceAudio
        }
        guard !audio.isEmpty else {
            throw VibeVoiceError.modelNotInitialized(component: "reference audio is empty")
        }

        inference.resetCaches()

        // 1) Tokenize transcript and run text through both LMs.
        let textIds = tokenizer.encode(text: transcript + "\n", addSpecialTokens: false)
        guard !textIds.isEmpty else {
            throw VibeVoiceError.modelNotInitialized(component: "transcript tokenized to empty")
        }
        let textTokens = MLXArray(textIds.map { Int32($0) }).reshaped([1, textIds.count])

        let lmHidden = inference.forwardLM(inputIds: textTokens, cache: &inference.lmCache)
        inference.lmLastHidden = lmHidden

        let textMask = MLXArray.ones([1, textIds.count], dtype: .int32)
        let textTtsHidden = inference.forwardTTSLM(
            inputIds: textTokens,
            lmHiddenState: lmHidden,
            ttsTextMask: textMask,
            cache: &inference.ttsLmCache
        )
        var accumTtsLmHidden = textTtsHidden  // accumulate full TTS-LM history

        // 2) Encode reference audio. 1.5B uses dual encoders (acoustic +
        // semantic) summed into the same audio-position embedding; 0.5B is
        // acoustic-only.
        //
        // Connector-input convention differs by variant (per Microsoft's reference
        // code paths):
        //   0.5B Realtime — connector takes RAW post-diffusion latent =
        //     `(encoder_mean + speech_bias_factor) * speech_scaling_factor`
        //     (matches modeling_vibevoice.py `forward_speech_features` / the
        //     domain that the diffusion produces during generation).
        //   1.5B long-form — connector takes the encoder MEAN directly, no
        //     scaling applied (matches `vllm_plugin/model.py` lines 360-379).
        let audioArr = MLXArray(audio).reshaped([1, audio.count])
        let acousticMean = inference.model.acousticTokenizer.encode(audioArr)
        let acousticForConnector: MLXArray
        if inference.model.config.hasSemanticTokenizer {
            // 1.5B: feed encoder mean directly.
            acousticForConnector = acousticMean
        } else {
            // 0.5B: apply scaling to enter the diffusion / connector domain.
            acousticForConnector = (acousticMean + inference.model.speechBiasFactor) * inference.model.speechScalingFactor
        }
        eval(acousticForConnector)

        var semanticLatents: MLXArray? = nil
        if let semTok = inference.model.semanticTokenizer {
            // Semantic is encoder-only and feeds the mean directly — no scaling.
            let s = semTok.encode(audioArr)
            eval(s)
            semanticLatents = s
        }

        // 3) Push each speech latent through the TTS LM (one at a time).
        let numLatents = acousticForConnector.dim(1)
        for i in 0..<numLatents {
            let oneAcoustic = acousticForConnector[0..., i...(i), 0...]  // [1, 1, vae_dim]
            var combinedEmbed = inference.model.acousticConnector(oneAcoustic)
            if let sem = semanticLatents, let semConn = inference.model.semanticConnector {
                let oneSem = sem[0..., i...(i), 0...]
                combinedEmbed = combinedEmbed + semConn(oneSem)
            }
            let stepHidden = inference.forwardTTSLMWithAcoustic(
                acousticEmbed: combinedEmbed,
                cache: &inference.ttsLmCache
            )
            accumTtsLmHidden = concatenated([accumTtsLmHidden, stepHidden], axis: 1)
        }
        eval(accumTtsLmHidden)
        inference.ttsLmLastHidden = accumTtsLmHidden

        // 4) Negative path: a single TokenConstants.negativeTextId, no audio.
        let negToken = MLXArray([Int32(TokenConstants.negativeTextId)]).reshaped([1, 1])
        let negLmHidden = inference.forwardLM(inputIds: negToken, cache: &inference.negLmCache)
        let negTextMask = MLXArray.ones([1, 1], dtype: .int32)
        let negTtsHidden = inference.forwardTTSLM(
            inputIds: negToken,
            lmHiddenState: negLmHidden,
            ttsTextMask: negTextMask,
            cache: &inference.negTtsLmCache
        )
        eval(negTtsHidden)
        inference.negTtsLmLastHidden = negTtsHidden

        // 5) Build the dict.
        var out: [String: MLXArray] = [:]
        out["lm_hidden"] = lmHidden
        out["tts_lm_hidden"] = accumTtsLmHidden
        out["neg_tts_lm_hidden"] = negTtsHidden

        for (idx, c) in inference.lmCache.enumerated() {
            if let (k, v) = sliceCache(c) {
                out["lm_key_\(idx)"] = k
                out["lm_value_\(idx)"] = v
            }
        }
        for (idx, c) in inference.ttsLmCache.enumerated() {
            if let (k, v) = sliceCache(c) {
                out["tts_lm_key_\(idx)"] = k
                out["tts_lm_value_\(idx)"] = v
            }
        }
        for (idx, c) in inference.negLmCache.enumerated() {
            if let (k, v) = sliceCache(c) {
                out["neg_lm_key_\(idx)"] = k
                out["neg_lm_value_\(idx)"] = v
            }
        }
        for (idx, c) in inference.negTtsLmCache.enumerated() {
            if let (k, v) = sliceCache(c) {
                out["neg_tts_lm_key_\(idx)"] = k
                out["neg_tts_lm_value_\(idx)"] = v
            }
        }

        eval(Array(out.values))
        return out
    }

    /// Save a previously encoded voice cache to a `.safetensors` file.
    public func saveVoiceCache(_ cache: [String: MLXArray], to url: URL) throws {
        try MLX.save(arrays: cache, url: url)
    }

    /// Convenience: encode + save in one call. Marks the cache loaded so the
    /// next `generate(text:)` call uses the new voice.
    @discardableResult
    public func encodeAndSaveVoice(
        referenceAudio: [Float],
        sampleRate: Int = 24000,
        transcript: String,
        to url: URL
    ) throws -> URL {
        let cache = try encodeVoice(
            referenceAudio: referenceAudio,
            sampleRate: sampleRate,
            transcript: transcript
        )
        try saveVoiceCache(cache, to: url)
        voiceLoaded = true
        return url
    }

    // MARK: - 1.5B long-form: structured-prompt single-shot synthesis
    //
    // VibeVoice 1.5B was trained on a structured prompt that the streaming /
    // 0.5B Realtime variant doesn't use. The full prompt is:
    //
    //   <system_prompt>
    //   " Speaker 0:" <speech_start_id> [vae_token_id]*N <speech_end_id> "\n"
    //   " Text input:\n Speaker 0: <text>\n"
    //   " Speech output:\n" <speech_start_id>
    //
    // The vae_token_id positions are placeholders — at forward time they get
    // their token-embedding REPLACED by the per-position audio embedding
    // computed as `acoustic_connector(ac_mean) + semantic_connector(sem_mean)`.
    //
    // This single-shot path is required for 1.5B; the voice-cache + generate
    // split that 0.5B uses doesn't apply because 1.5B was trained on the full
    // structured prompt as one input.

    private static let systemPromptVibeVoice =
        " Transform the text provided by various speakers into speech output, utilizing the distinct voice of each respective speaker.\n"

    /// 1.5B-specific text-to-speech with reference audio + transcript and
    /// generation text in a single call. Reference audio defines the speaker
    /// voice; `text` is what gets spoken.
    public func generateLongForm(
        referenceAudio: [Float],
        referenceTranscript: String,
        text: String,
        sampleRate: Int = 24000
    ) async throws -> [Float] {
        guard inference.model.config.hasSemanticTokenizer else {
            throw VibeVoiceError.modelNotInitialized(
                component: "generateLongForm requires the 1.5B variant — Configuration.longForm1_5B"
            )
        }
        let audio: [Float]
        if sampleRate != AudioConstants.sampleRate {
            audio = AudioFileLoader.resample(
                referenceAudio,
                from: sampleRate,
                to: AudioConstants.sampleRate
            )
        } else {
            audio = referenceAudio
        }
        guard !audio.isEmpty else {
            throw VibeVoiceError.modelNotInitialized(component: "reference audio empty")
        }

        inference.resetCaches()

        // 1) Encode reference audio through both tokenizers (1.5B uses MEAN
        // directly, no scaling — see vllm_plugin/model.py).
        let audioArr = MLXArray(audio).reshaped([1, audio.count])
        let acousticMean = inference.model.acousticTokenizer.encode(audioArr)
        guard let semTok = inference.model.semanticTokenizer,
              let semConn = inference.model.semanticConnector
        else {
            throw VibeVoiceError.modelNotInitialized(component: "semantic tokenizer / connector")
        }
        let semanticMean = semTok.encode(audioArr)
        eval(acousticMean, semanticMean)

        // Audio embeddings to inject at vae_token positions.
        // Both connectors output [B, T, hidden]. Sum at per-position level.
        let acousticEmbeds = inference.model.acousticConnector(acousticMean)
        let semanticEmbeds = semConn(semanticMean)
        let audioEmbeds = acousticEmbeds + semanticEmbeds
        eval(audioEmbeds)
        let numVaeTokens = audioEmbeds.dim(1)

        // 2) Build the full structured prompt as a token sequence.
        //
        // We can't ask the Tokenizer about the special VibeVoice tokens
        // (speech_start_id etc.) because they're outside the Qwen2.5 vocab in
        // some loaders. We construct the array manually using TokenConstants.
        var ids: [Int32] = []
        var audioMask: [Bool] = []  // true at vae_token positions
        func append(_ chunk: [Int32]) {
            ids.append(contentsOf: chunk)
            audioMask.append(contentsOf: Array(repeating: false, count: chunk.count))
        }
        // System prompt — encoded WITH special tokens (BOS), matching the
        // reference processor: vibevoice/processor/vibevoice_processor.py:273.
        append(tokenizer.encode(text: Self.systemPromptVibeVoice, addSpecialTokens: true).map(Int32.init))
        // Voice exemplar block: " Speaker 0:" + <speech_start> + vae*N + <speech_end> + "\n"
        // The audio embeddings replace the vae placeholders via the mask.
        append(tokenizer.encode(text: " Speaker 0:", addSpecialTokens: false).map(Int32.init))
        append([Int32(TokenConstants.speechStartId)])
        ids.append(contentsOf: Array(repeating: Int32(TokenConstants.speechDiffusionId), count: numVaeTokens))
        audioMask.append(contentsOf: Array(repeating: true, count: numVaeTokens))
        append([Int32(TokenConstants.speechEndId)])
        append(tokenizer.encode(text: "\n", addSpecialTokens: false).map(Int32.init))
        // Text input section: ONLY the text-to-speak (the reference transcript
        // isn't provided as text — the audio exemplar above conveys the voice).
        // Note: NO space between "Speaker 0:" and the text — matches the
        // reference processor format `f" Speaker {id}:{text}\n"`.
        append(tokenizer.encode(text: " Text input:\n Speaker 0:\(text)\n", addSpecialTokens: false).map(Int32.init))
        // Speech output cue
        append(tokenizer.encode(text: " Speech output:\n", addSpecialTokens: false).map(Int32.init))
        // Generation begins after this <speech_start>
        append([Int32(TokenConstants.speechStartId)])
        // Reference transcript intentionally unused at inference; kept in the
        // public API for forward-compat / multi-shot voice priming.
        _ = referenceTranscript

        let inputIds = MLXArray(ids).reshaped([1, ids.count])
        let mask = MLXArray(audioMask.map { $0 ? Int32(1) : Int32(0) }).reshaped([1, audioMask.count])

        // 3) Embed text tokens, then replace at audio-mask positions with
        // audioEmbeds (sequential mapping: i-th true position gets i-th audio embed).
        var embeds = inference.model.languageModel.embedTokens(inputIds)
        embeds = mergeAudioIntoEmbeds(embeds: embeds, audio: audioEmbeds, mask: mask)

        // 4) Forward through base LM.
        let lmHidden = inference.model.languageModel.forwardWithEmbeddings(
            embeds, cache: inference.lmCache, applyFinalNorm: false
        )
        inference.lmLastHidden = lmHidden

        // 5) Forward through TTS LM with proper type embeddings (text=1, audio=0).
        let textTypeMask = MLXArray.ones([1, ids.count], dtype: .int32) - mask
        let typeEmbed = inference.model.ttsInputTypes(textTypeMask)
        let ttsInputs = lmHidden + typeEmbed
        let ttsHidden = inference.model.ttsLanguageModel.forwardWithEmbeddings(
            ttsInputs, cache: inference.ttsLmCache
        )
        inference.ttsLmLastHidden = ttsHidden

        // 6) Negative conditioning (single negative-text-id token).
        let negToken = MLXArray([Int32(TokenConstants.negativeTextId)]).reshaped([1, 1])
        let negLmHidden = inference.forwardLM(inputIds: negToken, cache: &inference.negLmCache)
        let negTextMask = MLXArray.ones([1, 1], dtype: .int32)
        inference.negTtsLmLastHidden = inference.forwardTTSLM(
            inputIds: negToken,
            lmHiddenState: negLmHidden,
            ttsTextMask: negTextMask,
            cache: &inference.negTtsLmCache
        )

        // 7) Generation loop with LM token sampling (1.5B-specific):
        //
        // At each step we:
        //   a) Compute LM logits from the last TTS-LM hidden via the tied
        //      embed_tokens matrix (Qwen2.5 1.5B has tie_word_embeddings=true).
        //   b) Argmax over vocab for the next token.
        //   c) Branch:
        //        - speech_diffusion_id → run diffusion to sample acoustic latent,
        //          decode to audio chunk, feed acoustic_connector(latent) back as
        //          the next embedding (with type=0 / speech).
        //        - speech_end_id → stop.
        //        - any other token → embed it (with type=1 / text) and feed back.
        //
        // This mirrors the 1.5B reference where text and speech tokens are
        // interleaved by the LM, not externally orchestrated.
        let maxSpeech = configuration.maxSpeechTokens
        let acousticCache = StreamingConvCache()
        var audioChunks: [MLXArray] = []
        let speechEndId = Int32(TokenConstants.speechEndId)
        let speechDiffusionId = Int32(TokenConstants.speechDiffusionId)
        let embedTokens = inference.model.languageModel.embedTokens

        var stepNum = 0
        for _ in 0..<maxSpeech {
            guard let ttsHidden = inference.ttsLmLastHidden,
                  let negTtsHidden = inference.negTtsLmLastHidden else { break }

            // a) Sample next token via LM head (tied to embed_tokens for 1.5B).
            let lastTtsIdx = ttsHidden.dim(1) - 1
            let lastHidden = ttsHidden[0..., lastTtsIdx...(lastTtsIdx), 0...]  // [1,1,H]
            let logits = embedTokens.asLinear(lastHidden)  // [1, 1, vocab]
            let nextToken = logits.argMax(axis: -1)        // [1, 1]
            eval(nextToken)
            let tokenId = nextToken[0, 0].item(Int32.self)

            if stepNum < 10 || stepNum % 25 == 0 {
                let flat = logits.reshaped([-1])
                let maxV = flat.max().item(Float.self)
                let speechDiffLogit = flat[Int(speechDiffusionId)].item(Float.self)
                let speechEndLogit = flat[Int(speechEndId)].item(Float.self)
                print("[debug] step=\(stepNum) token=\(tokenId) maxLogit=\(maxV) | speech_diff(\(speechDiffusionId))=\(speechDiffLogit) | speech_end(\(speechEndId))=\(speechEndLogit)")
            }
            stepNum += 1

            if tokenId == speechEndId {
                break
            }

            if tokenId == speechDiffusionId {
                // Speech token: diffuse acoustic latent.
                let lastIdx = ttsHidden.dim(1) - 1
                let cond = ttsHidden[0..., lastIdx...(lastIdx), 0...].squeezed(axis: 1)
                let negLastIdx = negTtsHidden.dim(1) - 1
                let negCond = negTtsHidden[0..., negLastIdx...(negLastIdx), 0...].squeezed(axis: 1)

                let latent2D = try inference.sampleSpeechLatent(condition: cond, negCondition: negCond)
                let latent = expandedDimensions(latent2D, axis: 1)
                let scaled = latent / inference.model.speechScalingFactor - inference.model.speechBiasFactor
                let chunk = inference.model.acousticTokenizer.decode(
                    scaled, cache: acousticCache, useCache: true
                )
                eval(chunk)
                audioChunks.append(chunk)

                // Feed acoustic embedding back (type=0 / speech).
                let acEmbed = inference.model.acousticConnector(latent)
                inference.ttsLmLastHidden = inference.forwardTTSLMWithAcoustic(
                    acousticEmbed: acEmbed,
                    cache: &inference.ttsLmCache
                )
                inference.negTtsLmLastHidden = inference.forwardTTSLMWithAcoustic(
                    acousticEmbed: acEmbed,
                    cache: &inference.negTtsLmCache
                )
            } else {
                // Plain text token — embed it and feed forward through both
                // LMs with type=1 (text).
                let tokenArr = MLXArray([tokenId]).reshaped([1, 1])
                let lmStep = inference.forwardLM(inputIds: tokenArr, cache: &inference.lmCache)
                inference.lmLastHidden = lmStep

                let textMask = MLXArray.ones([1, 1], dtype: .int32)
                inference.ttsLmLastHidden = inference.forwardTTSLM(
                    inputIds: tokenArr,
                    lmHiddenState: lmStep,
                    ttsTextMask: textMask,
                    cache: &inference.ttsLmCache
                )

                // Negative path tracks the same text token so it stays aligned.
                let negLmStep = inference.forwardLM(inputIds: tokenArr, cache: &inference.negLmCache)
                inference.negTtsLmLastHidden = inference.forwardTTSLM(
                    inputIds: tokenArr,
                    lmHiddenState: negLmStep,
                    ttsTextMask: textMask,
                    cache: &inference.negTtsLmCache
                )
            }
        }

        guard !audioChunks.isEmpty else { return [] }
        let full = concatenated(audioChunks, axis: -1)
        eval(full)
        let flat = full.reshaped([-1])
        return flat.asArray(Float.self)
    }

    /// Replace embedding rows at positions where `mask` is 1 with the matching
    /// row from `audio`. `audio` shape `[1, A, D]`, `embeds` `[1, L, D]`,
    /// `mask` `[1, L]` int32 with exactly A ones.
    private func mergeAudioIntoEmbeds(
        embeds: MLXArray,
        audio: MLXArray,
        mask: MLXArray
    ) -> MLXArray {
        // Build a [1, L, D] tensor where audio rows are scattered into the
        // positions marked by `mask`, and other positions are zero. Then
        // additively combine: `embeds * (1 - mask) + scattered_audio`.
        let L = embeds.dim(1)
        let D = embeds.dim(2)
        let A = audio.dim(1)
        // For each output position l, compute speech_idx = cumsum(mask)[l] - 1,
        // clipped to [0, A-1]. When mask[l]==1 we gather audio[speech_idx];
        // otherwise we keep embeds[l].
        let cumsum = mask.cumsum(axis: 1)
        let speechIdxRaw = cumsum - 1
        let speechIdxClipped = clip(speechIdxRaw, min: 0, max: A - 1)
        // gather: audio[0, speechIdx[0, l], :] for each l
        // shape: [L, D]
        let gathered = audio[0, speechIdxClipped[0], 0...]
        // Broadcast mask over D dim and select
        let maskFloat = mask.asType(embeds.dtype).reshaped([1, L, 1])
        let result = maskFloat * gathered.reshaped([1, L, D])
                   + (1.0 - maskFloat) * embeds
        return result
    }

    private func sliceCache(_ c: KVCacheSimple) -> (MLXArray, MLXArray)? {
        guard let k = c.keys, let v = c.values, c.offset > 0 else { return nil }
        // Stored layout: (B, kv_heads, alloc_seq, head_dim). Slice to valid offset.
        let kSlice = k[0..., 0..., 0..<c.offset, 0...]
        let vSlice = v[0..., 0..., 0..<c.offset, 0...]
        return (kSlice, vSlice)
    }

    /// Synthesize speech and yield each acoustic-decoder chunk as it's produced.
    ///
    /// Each yielded `[Float]` is a 24 kHz mono audio chunk (typically 100–250 ms).
    /// Wire this into `AudioCommon.StreamingAudioPlayer.scheduleChunk(_:)` for
    /// play-while-generating UX.
    ///
    /// Requires a voice cache to be loaded first via `loadVoice(from:)`.
    public func generateChunkStream(text: String) -> AsyncThrowingStream<[Float], Error> {
        AsyncThrowingStream { continuation in
            Task {
                do {
                    guard voiceLoaded else {
                        throw VibeVoiceError.voiceCacheNotLoaded
                    }
                    let ids = tokenizer.encode(text: text + "\n", addSpecialTokens: false)
                    let int32Ids = ids.map { Int32($0) }
                    let tokenIds = MLXArray(int32Ids).reshaped([1, int32Ids.count])

                    try inference.generateWithVoiceCacheStream(
                        tokenIds: tokenIds,
                        maxSpeechTokens: configuration.maxSpeechTokens
                    ) { chunk in
                        // chunk shape: [B, 1, samples]; flatten to [Float].
                        let samples = chunk.reshaped([-1]).asArray(Float.self)
                        continuation.yield(samples)
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }
}
