import Foundation
import MLX
import MLXNN
import MLXFast
import Qwen3Common

/// Main Qwen3-TTS model for text-to-speech synthesis
public class Qwen3TTSModel {
    public let config: Qwen3TTSConfig
    public let talker: TalkerModel
    public let codePredictor: CodePredictorModel
    public let codecDecoder: SpeechTokenizerDecoder

    private var tokenizer: Qwen3Tokenizer?

    public init(config: Qwen3TTSConfig = .base06B) {
        self.config = config
        self.talker = TalkerModel(config: config.talker)
        self.codePredictor = CodePredictorModel(config: config.codePredictor)
        self.codecDecoder = SpeechTokenizerDecoder(config: config.speechTokenizerDecoder)
    }

    public func setTokenizer(_ tokenizer: Qwen3Tokenizer) {
        self.tokenizer = tokenizer
    }

    /// Synthesize speech from text
    /// - Parameters:
    ///   - text: Input text to synthesize
    ///   - language: Language tag (e.g., "english", "chinese")
    ///   - sampling: Sampling configuration
    /// - Returns: Audio samples at 24kHz
    public func synthesize(
        text: String,
        language: String = "english",
        sampling: SamplingConfig = .default
    ) -> [Float] {
        guard let tokenizer = tokenizer else {
            fatalError("Tokenizer not loaded. Call setTokenizer() first.")
        }

        guard let langId = CodecTokens.languageId(for: language) else {
            print("Warning: Unknown language '\(language)', defaulting to English")
            return synthesize(text: text, language: "english", sampling: sampling)
        }

        let t0 = CFAbsoluteTimeGetCurrent()

        // Stage 1: Prepare text tokens and codec prefix
        let textTokens = prepareTextTokens(text: text, tokenizer: tokenizer)
        let codecPrefixTokens = buildCodecPrefix(languageId: langId)

        // Stage 2: Build input embeddings with element-wise text+codec overlay
        let (prefillEmbeds, trailingTextHidden, ttsPadEmbed) = buildPrefillEmbeddings(
            textTokens: textTokens, codecPrefixTokens: codecPrefixTokens)

        eval(prefillEmbeds, trailingTextHidden, ttsPadEmbed)
        let t1 = CFAbsoluteTimeGetCurrent()

        // Stage 3: Autoregressive generation with per-step code predictor
        let (allCodebooks, numFrames) = generateWithCodePredictor(
            prefillEmbeds: prefillEmbeds,
            trailingTextHidden: trailingTextHidden,
            ttsPadEmbed: ttsPadEmbed,
            sampling: sampling)

        eval(allCodebooks)
        let t2 = CFAbsoluteTimeGetCurrent()

        guard numFrames > 0 else {
            print("Warning: Talker generated no tokens")
            return []
        }

        // Stage 4: Codec decode to waveform
        let outputSamples = numFrames * 1920
        print("  Decoding \(numFrames) frames -> \(outputSamples) samples (\(String(format: "%.1f", Double(outputSamples) / 24000.0))s)...")
        let waveform = codecDecoder.decode(codes: allCodebooks)
        let t3 = CFAbsoluteTimeGetCurrent()

        let audioDur = Double(waveform.count) / 24000.0
        print("  Timing: embed=\(String(format: "%.3f", t1-t0))s | " +
              "generate=\(String(format: "%.3f", t2-t1))s (\(numFrames) steps, " +
              "\(String(format: "%.0f", (t2-t1)/Double(numFrames)*1000))ms/step) | " +
              "decode=\(String(format: "%.3f", t3-t2))s | " +
              "total=\(String(format: "%.3f", t3-t0))s | " +
              "audio=\(String(format: "%.2f", audioDur))s | " +
              "RTF=\(String(format: "%.2f", (t3-t0)/audioDur))")

        return waveform
    }

    // MARK: - Text Preparation

    /// Prepare text tokens using chat template.
    /// Template: <|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n
    private func prepareTextTokens(text: String, tokenizer: Qwen3Tokenizer) -> [Int] {
        let imStartId = 151644
        let imEndId = 151645
        let newlineId = 198
        let assistantId = 77091

        var tokens: [Int] = []

        // <|im_start|>assistant\n
        tokens.append(contentsOf: [imStartId, assistantId, newlineId])

        // Encode text
        let textTokens = tokenizer.encode(text)
        tokens.append(contentsOf: textTokens)

        // <|im_end|>\n<|im_start|>assistant\n
        tokens.append(contentsOf: [imEndId, newlineId, imStartId, assistantId, newlineId])

        return tokens
    }

    // MARK: - Codec Prefix

    /// Build codec prefix: [think, think_bos, lang_id, think_eos, pad, bos]
    private func buildCodecPrefix(languageId: Int) -> [Int32] {
        [
            Int32(CodecTokens.codecThink),
            Int32(CodecTokens.codecThinkBos),
            Int32(languageId),
            Int32(CodecTokens.codecThinkEos),
            Int32(CodecTokens.codecPad),
            Int32(CodecTokens.codecBos),
        ]
    }

    // MARK: - Embedding Construction

    /// Build prefill embeddings with element-wise text+codec overlay.
    ///
    /// Python reference:
    /// ```
    /// # Text-side TTS special tokens
    /// tts_bos_embed = text_projection(text_embedding(151672))
    /// tts_pad_embed = text_projection(text_embedding(151671))
    /// tts_eos_embed = text_projection(text_embedding(151673))
    ///
    /// # Build text overlay for codec prefix (pad_count = codec_len - 2)
    /// pad_embeds = broadcast(tts_pad_embed, (1, pad_count, 1024))
    /// text_overlay = concat([pad_embeds, tts_bos_embed], axis=1)
    ///
    /// # Element-wise sum of text overlay + codec prefix (minus last token)
    /// combined = text_overlay + codec_embed[:, :-1, :]
    ///
    /// # Role embedding (first 3 text tokens: <|im_start|>assistant\n)
    /// role_embed = text_embed[:, :3, :]
    ///
    /// # First text token added to last codec token (codec_bos)
    /// first_text = text_embed[:, 3:4, :] + codec_embed[:, -1:, :]
    ///
    /// # Prefill input
    /// input_embeds = concat([role_embed, combined, first_text], axis=1)
    ///
    /// # Trailing text (tokens 4 to -5, plus tts_eos)
    /// trailing_text = concat([text_embed[:, 4:-5, :], tts_eos_embed], axis=1)
    /// ```
    private func buildPrefillEmbeddings(
        textTokens: [Int], codecPrefixTokens: [Int32]
    ) -> (prefillEmbeds: MLXArray, trailingTextHidden: MLXArray, ttsPadEmbed: MLXArray) {
        let hiddenSize = config.talker.hiddenSize

        // Embed all text tokens → project to hidden dim
        let textTokenArray = MLXArray(textTokens.map { Int32($0) }).expandedDimensions(axis: 0)  // [1, textLen]
        let textEmbeds = talker.embedText(textTokenArray)  // [1, textLen, hiddenSize]

        // Embed codec prefix tokens
        let codecArray = MLXArray(codecPrefixTokens).expandedDimensions(axis: 0)  // [1, codecLen]
        let codecEmbeds = talker.embedCodec(codecArray)  // [1, codecLen, hiddenSize]

        // TTS special token embeddings (text-side)
        let ttsPadTokens = MLXArray([Int32(CodecTokens.ttsPad)]).expandedDimensions(axis: 0)
        let ttsBosTokens = MLXArray([Int32(CodecTokens.ttsBos)]).expandedDimensions(axis: 0)
        let ttsEosTokens = MLXArray([Int32(CodecTokens.ttsEos)]).expandedDimensions(axis: 0)

        let ttsPadEmbed = talker.embedText(ttsPadTokens)  // [1, 1, hiddenSize]
        let ttsBosEmbed = talker.embedText(ttsBosTokens)  // [1, 1, hiddenSize]
        let ttsEosEmbed = talker.embedText(ttsEosTokens)  // [1, 1, hiddenSize]

        let codecLen = codecPrefixTokens.count  // 6

        // Text overlay for codec prefix:
        // pad_count = codecLen - 2 (the last 2 positions are: tts_bos overlay + first_text+codec_bos)
        // But actually: we overlay codecLen-1 positions with text, and the last codec token (bos) gets first_text
        let padCount = codecLen - 2  // 4 pad positions
        let padEmbeds = broadcast(ttsPadEmbed, to: [1, padCount, hiddenSize])
        let textOverlay = concatenated([padEmbeds, ttsBosEmbed], axis: 1)  // [1, codecLen-1, hiddenSize]

        // Element-wise sum: text overlay + codec prefix (all but last token)
        let codecWithoutLast = codecEmbeds[0..., 0..<(codecLen - 1), 0...]  // [1, codecLen-1, hiddenSize]
        let combined = textOverlay + codecWithoutLast  // [1, codecLen-1, hiddenSize]

        // Role embedding: first 3 text tokens (<|im_start|>assistant\n)
        let roleEmbed = textEmbeds[0..., 0..<3, 0...]  // [1, 3, hiddenSize]

        // First text token (index 3) added to last codec token (codec_bos)
        let firstTextEmbed = textEmbeds[0..., 3..<4, 0...]  // [1, 1, hiddenSize]
        let lastCodecEmbed = codecEmbeds[0..., (codecLen - 1)..<codecLen, 0...]  // [1, 1, hiddenSize]
        let firstTextPlusCodec = firstTextEmbed + lastCodecEmbed  // [1, 1, hiddenSize]

        // Prefill: [role_embed, combined, first_text+codec_bos]
        let prefillEmbeds = concatenated([roleEmbed, combined, firstTextPlusCodec], axis: 1)

        // Trailing text: tokens[4:-5] + tts_eos
        // textTokens length includes: [im_start, assistant, \n, ...text..., im_end, \n, im_start, assistant, \n]
        // Trailing = tokens from index 4 to (len-5), then append tts_eos
        let textLen = textTokens.count
        let trailStart = 4
        let trailEnd = textLen - 5
        if trailEnd > trailStart {
            let trailingSlice = textEmbeds[0..., trailStart..<trailEnd, 0...]
            let trailingTextHidden = concatenated([trailingSlice, ttsEosEmbed], axis: 1)
            return (prefillEmbeds, trailingTextHidden, ttsPadEmbed)
        } else {
            // Very short text: only tts_eos as trailing
            return (prefillEmbeds, ttsEosEmbed, ttsPadEmbed)
        }
    }

    // MARK: - Generation Loop

    /// Generate first codebook + predict remaining 15 codebooks per-step.
    ///
    /// Each generation step:
    /// 1. Run talker forward → get logits for first codebook + hidden states
    /// 2. Sample first codebook token
    /// 3. Run code predictor autoregressively to predict 15 remaining codebook tokens
    /// 4. Build next-step input: trailing_text_embed + sum(all 16 codebook embeddings)
    private func generateWithCodePredictor(
        prefillEmbeds: MLXArray,
        trailingTextHidden: MLXArray,
        ttsPadEmbed: MLXArray,
        sampling: SamplingConfig
    ) -> (allCodebooks: MLXArray, numFrames: Int) {
        let safeMaxTokens = min(sampling.maxTokens, 500)

        var talkerCache: [(MLXArray, MLXArray)]? = nil
        var generatedFirstCodebook: [Int32] = []
        var generatedAllCodebooks: [[Int32]] = (0..<config.codePredictor.numCodeGroups).map { _ in [] }

        // Pre-allocate code predictor sampling config (reused every step × 15 groups)
        let cpSamplingConfig = SamplingConfig(temperature: sampling.temperature, topK: sampling.topK)

        let prefillLen = prefillEmbeds.dim(1)
        let positionIds = buildTTSPositionIds(seqLen: prefillLen)

        // Prefill
        var (logits, hiddenStates, newCache) = talker(
            inputsEmbeds: prefillEmbeds,
            positionIds: positionIds,
            cache: talkerCache)
        talkerCache = newCache

        // Sample first token from last position
        let lastLogits = logits[0..., (prefillLen - 1)..<prefillLen, 0...]
        var nextToken = sampleToken(
            logits: lastLogits,
            config: sampling,
            generatedTokens: generatedFirstCodebook,
            suppressRange: (2048, 3072),
            eosTokenId: CodecTokens.codecEos)

        if nextToken == Int32(CodecTokens.codecEos) {
            return (MLXArray.zeros([1, 16, 0]), 0)
        }

        generatedFirstCodebook.append(nextToken)
        generatedAllCodebooks[0].append(nextToken)

        // Get hidden state for this step's code predictor
        let lastHidden = hiddenStates[0..., (prefillLen - 1)..<prefillLen, 0...]  // [1, 1, D]

        // Run code predictor for this timestep to get remaining 15 codebook tokens
        var codeTokens = predictCodebooksForTimestep(
            hiddenState: lastHidden,
            firstCodebookToken: nextToken,
            cpSamplingConfig: cpSamplingConfig)
        for (i, token) in codeTokens.enumerated() {
            generatedAllCodebooks[i + 1].append(token)
        }

        var trailingIdx = 0
        var step = prefillLen

        // Autoregressive generation
        for iterIdx in 1..<safeMaxTokens {
            // Text side: next trailing text embed or tts_pad
            let textEmbed: MLXArray
            let trailingLen = trailingTextHidden.dim(1)
            if trailingIdx < trailingLen {
                textEmbed = trailingTextHidden[0..., trailingIdx..<(trailingIdx + 1), 0...]
                trailingIdx += 1
            } else {
                textEmbed = ttsPadEmbed
            }

            // Codec side: sum all 16 codebook embeddings (codebook 0 + 15 code predictor groups)
            let codecEmbed = talker.embedCodec(
                MLXArray([nextToken]).expandedDimensions(axis: 0))  // [1, 1, D] — codebook 0
                + codePredictor.batchEmbedAllGroups(codeTokens)     // [1, 1, D] — groups 1-15

            // Next input = text + codec (element-wise sum)
            let stepEmbeds = textEmbed + codecEmbed  // [1, 1, D]
            let stepPosIds = buildTTSPositionIds(seqLen: 1, offset: step)

            (logits, hiddenStates, newCache) = talker(
                inputsEmbeds: stepEmbeds,
                positionIds: stepPosIds,
                cache: talkerCache)
            talkerCache = newCache

            nextToken = sampleToken(
                logits: logits,
                config: sampling,
                generatedTokens: generatedFirstCodebook,
                suppressRange: (2048, 3072),
                eosTokenId: CodecTokens.codecEos)

            if nextToken == Int32(CodecTokens.codecEos) { break }

            generatedFirstCodebook.append(nextToken)
            generatedAllCodebooks[0].append(nextToken)

            // Code predictor for this timestep
            let stepHidden = hiddenStates  // [1, 1, D]
            codeTokens = predictCodebooksForTimestep(
                hiddenState: stepHidden,
                firstCodebookToken: nextToken,
                cpSamplingConfig: cpSamplingConfig)
            for (i, token) in codeTokens.enumerated() {
                generatedAllCodebooks[i + 1].append(token)
            }

            step += 1

            if iterIdx % 50 == 0 {
                let estSec = Double(generatedFirstCodebook.count) / 12.5
                print("  Talker: \(generatedFirstCodebook.count) tokens (~\(String(format: "%.1f", estSec))s audio)...")
            }
        }

        let numFrames = generatedFirstCodebook.count

        if numFrames >= safeMaxTokens && nextToken != Int32(CodecTokens.codecEos) {
            let estSec = Double(numFrames) / 12.5
            print("Warning: Hit safety limit of \(safeMaxTokens) tokens (~\(String(format: "%.1f", estSec))s audio). "
                + "Increase SamplingConfig.maxTokens if you need longer output.")
        }

        let estAudioSec = Double(numFrames) / 12.5
        print("  Talker done: \(numFrames) codec tokens (~\(String(format: "%.1f", estAudioSec))s audio)")

        // Stack all codebooks: [1, 16, T]
        let codebookArrays = generatedAllCodebooks.map { tokens in
            MLXArray(tokens).expandedDimensions(axis: 0)  // [1, T]
        }
        let allCodebooks = stacked(codebookArrays, axis: 1)  // [1, 16, T]

        return (allCodebooks, numFrames)
    }

    // MARK: - Per-Timestep Code Prediction

    /// Predict 15 remaining codebook tokens for a single timestep.
    ///
    /// The code predictor runs autoregressively across codebook groups:
    /// - Step 0: prefill [hidden_state, code_0_embed] (length 2)
    /// - Steps 1-14: single embedding of previous code token (length 1), KV cache
    private func predictCodebooksForTimestep(
        hiddenState: MLXArray,
        firstCodebookToken: Int32,
        cpSamplingConfig: SamplingConfig
    ) -> [Int32] {
        var codeTokens: [Int32] = []
        codeTokens.reserveCapacity(config.codePredictor.numCodeGroups - 1)
        var cpCache: [(MLXArray, MLXArray)]? = nil

        // First codebook embedding (from talker's codec embedding)
        let code0Embed = talker.embedCodec(
            MLXArray([firstCodebookToken]).expandedDimensions(axis: 0))  // [1, 1, D]

        // Prefill: [hidden_state, code_0_embed] — length 2
        let prefillInput = concatenated([hiddenState, code0Embed], axis: 1)  // [1, 2, D]

        // Predict codebook group 0 (= codebook 2 overall)
        var (cpLogits, cpNewCache) = codePredictor(
            inputsEmbeds: prefillInput, groupIndex: 0, cache: nil)
        cpCache = cpNewCache

        // Sample from last position (prefill length is always 2)
        let lastCpLogits = cpLogits[0..., 1..<2, 0...]
        var prevToken = sampleToken(logits: lastCpLogits, config: cpSamplingConfig)
        codeTokens.append(prevToken)

        // Remaining 14 codebook groups
        for groupIdx in 1..<(config.codePredictor.numCodeGroups - 1) {
            // Embed previous group's token
            let prevEmbed = codePredictor.embedCodecGroup(
                MLXArray([prevToken]).expandedDimensions(axis: 0),
                groupIndex: groupIdx - 1)  // [1, 1, D]

            (cpLogits, cpNewCache) = codePredictor(
                inputsEmbeds: prevEmbed, groupIndex: groupIdx, cache: cpCache)
            cpCache = cpNewCache

            prevToken = sampleToken(logits: cpLogits, config: cpSamplingConfig)
            codeTokens.append(prevToken)
        }

        return codeTokens  // 15 tokens
    }
}

// MARK: - Model Loading

public extension Qwen3TTSModel {
    /// Load model from HuggingFace hub
    static func fromPretrained(
        modelId: String = "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-4bit",
        tokenizerModelId: String = "Qwen/Qwen3-TTS-Tokenizer-12Hz",
        progressHandler: ((Double, String) -> Void)? = nil
    ) async throws -> Qwen3TTSModel {
        progressHandler?(0.05, "Preparing download...")

        let model = Qwen3TTSModel()

        // Download main model weights
        let mainCacheDir = try HuggingFaceDownloader.getCacheDirectory(for: modelId)
        if !HuggingFaceDownloader.weightsExist(in: mainCacheDir) {
            progressHandler?(0.1, "Downloading TTS model weights...")
            try await HuggingFaceDownloader.downloadWeights(
                modelId: modelId,
                to: mainCacheDir,
                additionalFiles: ["vocab.json", "merges.txt", "tokenizer_config.json"],
                progressHandler: { progress in
                    progressHandler?(0.1 + progress * 0.3, "Downloading TTS model...")
                })
        }

        // Download tokenizer/codec weights
        let tokenizerCacheDir = try HuggingFaceDownloader.getCacheDirectory(for: tokenizerModelId)
        if !HuggingFaceDownloader.weightsExist(in: tokenizerCacheDir) {
            progressHandler?(0.4, "Downloading speech tokenizer...")
            try await HuggingFaceDownloader.downloadWeights(
                modelId: tokenizerModelId,
                to: tokenizerCacheDir,
                progressHandler: { progress in
                    progressHandler?(0.4 + progress * 0.2, "Downloading speech tokenizer...")
                })
        }

        // Load tokenizer
        progressHandler?(0.6, "Loading tokenizer...")
        let vocabPath = mainCacheDir.appendingPathComponent("vocab.json")
        if FileManager.default.fileExists(atPath: vocabPath.path) {
            let tokenizer = Qwen3Tokenizer()
            try tokenizer.load(from: vocabPath)
            model.setTokenizer(tokenizer)
        }

        // Load Talker + Code Predictor weights (single load of safetensors)
        progressHandler?(0.7, "Loading TTS model weights...")
        try TTSWeightLoader.loadTalkerAndCodePredictorWeights(
            talker: model.talker, codePredictor: model.codePredictor, from: mainCacheDir)

        // Load Speech Tokenizer Decoder weights
        progressHandler?(0.85, "Loading speech tokenizer decoder...")
        try TTSWeightLoader.loadSpeechTokenizerDecoderWeights(
            into: model.codecDecoder, from: tokenizerCacheDir)

        progressHandler?(1.0, "Ready")
        return model
    }
}
