import Foundation
import MLX
import MLXNN
import Qwen3Common

// MARK: - PersonaPlex Model

public final class PersonaPlexModel: Module {
    public let cfg: PersonaPlexConfig

    @ModuleInfo public var temporal: TemporalTransformer
    @ModuleInfo public var depformer: Depformer
    public let mimi: Mimi

    public init(cfg: PersonaPlexConfig = .default) {
        self.cfg = cfg
        self._temporal = ModuleInfo(wrappedValue: TemporalTransformer(cfg: cfg.temporal))
        self._depformer = ModuleInfo(wrappedValue: Depformer(cfg: cfg.depformer, temporalDim: cfg.temporal.dim))
        self.mimi = Mimi(cfg: cfg.mimi)
    }

    // MARK: - Offline Inference

    /// Process user audio and generate response audio.
    ///
    /// Stream layout (17 streams):
    ///   - Stream 0:    text (agent inner monologue)
    ///   - Streams 1-8: agent audio (8 codebooks, predicted by depformer)
    ///   - Streams 9-16: user audio (8 codebooks from Mimi encoder)
    ///
    /// Prompt sequence before user audio:
    ///   1. Voice prompt (pre-computed embeddings fed through temporal transformer)
    ///   2. 0.5s silence spacer
    ///   3. Text system prompt (SentencePiece tokens, one per frame)
    ///   4. 0.5s silence spacer
    ///   5. User audio frames, then autoregressive generation
    ///
    /// - Parameters:
    ///   - userAudio: [numSamples] float array of 24kHz mono audio
    ///   - voice: voice preset for the agent
    ///   - systemPromptTokens: SentencePiece-tokenized system prompt (nil = default)
    ///   - maxSteps: maximum generation steps (at 12.5 Hz)
    ///   - verbose: print timing info
    /// - Returns: [numSamples] float array of 24kHz response audio
    public func respond(
        userAudio: [Float],
        voice: PersonaPlexVoice = .NATM0,
        systemPromptTokens: [Int32]? = nil,
        maxSteps: Int = 500,
        verbose: Bool = false
    ) -> [Float] {
        let startTime = CFAbsoluteTimeGetCurrent()

        // 1. Encode user audio with Mimi
        let audioArray = MLXArray(userAudio).reshaped([1, 1, userAudio.count])
        let userCodes = mimi.encode(audioArray)  // [1, numCodebooks, T]
        eval(userCodes)

        let userFrameCount = userCodes.shape[2]
        if verbose {
            let encTime = CFAbsoluteTimeGetCurrent() - startTime
            print("  Mimi encode: \(String(format: "%.2f", encTime))s, \(userFrameCount) frames")
        }

        // 2. Load voice prompt embeddings
        let voiceStart = CFAbsoluteTimeGetCurrent()
        let voiceEmbeddings: MLXArray?
        let voiceCache: MLXArray?
        do {
            let modelDir = try HuggingFaceDownloader.getCacheDirectory(for: "aufklarer/PersonaPlex-7B-MLX-4bit")
            let voiceDir = modelDir.appendingPathComponent("voices")
            let voiceFile = voiceDir.appendingPathComponent("\(voice.rawValue).safetensors")
            if FileManager.default.fileExists(atPath: voiceFile.path) {
                let weights = try MLX.loadArrays(url: voiceFile)
                voiceEmbeddings = weights["embeddings"]  // [T, 1, 1, dim]
                voiceCache = weights["cache"]            // [1, 17, 4]
            } else {
                voiceEmbeddings = nil
                voiceCache = nil
            }
        } catch {
            voiceEmbeddings = nil
            voiceCache = nil
        }

        let voiceFrameCount = voiceEmbeddings?.shape[0] ?? 0
        let silenceFrameCount = Int(0.5 * cfg.mimi.frameRate)  // 0.5s silence = ~6 frames
        let textPromptTokens = systemPromptTokens ?? TemporalTransformerConfig.defaultSystemPromptTokens
        let textPromptLen = textPromptTokens.count

        if verbose {
            let voiceTime = CFAbsoluteTimeGetCurrent() - voiceStart
            print("  Voice prompt: \(voiceFrameCount) frames, text prompt: \(textPromptLen) tokens (\(String(format: "%.2f", voiceTime))s)")
        }

        // 3. Reset caches
        temporal.resetCache()
        mimi.resetState()

        // Total steps: voice + silence1 + text_prompt + silence2 + user audio + generation
        let promptLen = voiceFrameCount + silenceFrameCount + textPromptLen + silenceFrameCount
        let prefillLen = promptLen + userFrameCount
        let delays = cfg.delays
        let maxDelay = cfg.maxDelay
        let numStreams = cfg.numStreams
        let nQ = cfg.temporal.nQ
        let totalLen = prefillLen + maxSteps + maxDelay + 3

        // 4. Initialize token cache
        // Stream 0 = text, streams 1-8 = agent audio, streams 9-16 = user audio
        var tokenCache = [[Int32]](repeating: [Int32](repeating: -1, count: totalLen), count: numStreams)

        // --- Phase 1: Voice prompt tokens ---
        // During voice prompt: text=PAD, agent audio=silence tokens, user audio=sine tokens
        for t in 0..<voiceFrameCount {
            // Text: padding token
            tokenCache[0][t + delays[0]] = Int32(cfg.temporal.textPaddingId)
            // Agent audio: silence tokens (streams 1-8)
            for cb in 0..<nQ {
                let streamIdx = 1 + cb
                tokenCache[streamIdx][t + delays[streamIdx]] = TemporalTransformerConfig.silenceTokens[cb]
            }
            // User audio: sine tokens (streams 9-16)
            for cb in 0..<nQ {
                let streamIdx = 1 + nQ + cb
                tokenCache[streamIdx][t + delays[streamIdx]] = TemporalTransformerConfig.sineTokens[cb]
            }
        }

        // --- Phase 2: Silence spacer 1 ---
        var pos = voiceFrameCount
        for t in 0..<silenceFrameCount {
            tokenCache[0][pos + delays[0]] = Int32(cfg.temporal.textPaddingId)
            for cb in 0..<nQ {
                let agentIdx = 1 + cb
                tokenCache[agentIdx][pos + delays[agentIdx]] = TemporalTransformerConfig.silenceTokens[cb]
            }
            for cb in 0..<nQ {
                let userIdx = 1 + nQ + cb
                tokenCache[userIdx][pos + delays[userIdx]] = TemporalTransformerConfig.sineTokens[cb]
            }
            pos += 1
        }

        // --- Phase 3: Text prompt ---
        // Text stream gets the actual system prompt tokens (one per frame)
        // Agent audio = silence tokens, user audio = sine tokens
        for t in 0..<textPromptLen {
            tokenCache[0][pos + delays[0]] = textPromptTokens[t]
            for cb in 0..<nQ {
                let agentIdx = 1 + cb
                tokenCache[agentIdx][pos + delays[agentIdx]] = TemporalTransformerConfig.silenceTokens[cb]
            }
            for cb in 0..<nQ {
                let userIdx = 1 + nQ + cb
                tokenCache[userIdx][pos + delays[userIdx]] = TemporalTransformerConfig.sineTokens[cb]
            }
            pos += 1
        }

        // --- Phase 4: Silence spacer 2 ---
        for _ in 0..<silenceFrameCount {
            tokenCache[0][pos + delays[0]] = Int32(cfg.temporal.textPaddingId)
            for cb in 0..<nQ {
                let agentIdx = 1 + cb
                tokenCache[agentIdx][pos + delays[agentIdx]] = TemporalTransformerConfig.silenceTokens[cb]
            }
            for cb in 0..<nQ {
                let userIdx = 1 + nQ + cb
                tokenCache[userIdx][pos + delays[userIdx]] = TemporalTransformerConfig.sineTokens[cb]
            }
            pos += 1
        }

        // --- Phase 5: User audio ---
        // Fill user audio into streams 9-16, agent audio = silence, text = PAD
        let userCodesArr = userCodes.asType(.int32)
        eval(userCodesArr)
        for t in 0..<userFrameCount {
            tokenCache[0][pos + delays[0]] = Int32(cfg.temporal.textPaddingId)
            // Agent audio: silence during user turn
            for cb in 0..<nQ {
                let agentIdx = 1 + cb
                tokenCache[agentIdx][pos + delays[agentIdx]] = TemporalTransformerConfig.silenceTokens[cb]
            }
            // User audio from Mimi encoder
            for cb in 0..<min(nQ, userCodes.shape[1]) {
                let userIdx = 1 + nQ + cb
                tokenCache[userIdx][pos + delays[userIdx]] = userCodesArr[0, cb, t].item(Int32.self)
            }
            pos += 1
        }

        // 5. Autoregressive generation (full-duplex: agent generates while user speaks)
        //
        // Phase layout:
        //   steps 0..<voiceFrameCount:    voice prompt (embeddings only, no generation)
        //   steps voiceFrameCount..<promptLen: silence + text prompt + silence (forward, no generation)
        //   steps promptLen..<prefillLen:  user audio + simultaneous agent generation
        //   steps prefillLen..<prefillLen+maxSteps: post-user generation (user=sine)
        var agentTokens: [[Int32]] = Array(repeating: [], count: cfg.depformer.numSteps)
        let genStart = CFAbsoluteTimeGetCurrent()
        let generationStartStep = promptLen  // Start generating when user audio begins

        for step in 0..<(prefillLen + maxSteps) {
            // --- Voice prompt: use pre-computed embeddings ---
            if step < voiceFrameCount, let voiceEmb = voiceEmbeddings {
                let emb = voiceEmb[step].reshaped([1, 1, cfg.temporal.dim])
                temporal.forwardEmbedding(emb, offset: step)
                continue
            }

            // Build input tokens for this step
            let textTokenArr = MLXArray([tokenCache[0][step]]).reshaped([1, 1])
            var audioTokenArrs: [MLXArray] = []
            for stream in 1..<numStreams {
                let tok = tokenCache[stream][step]
                let effectiveTok = tok >= 0 ? tok : Int32(cfg.temporal.initialTokenId)
                audioTokenArrs.append(MLXArray([effectiveTok]))
            }
            let audioTokens = stacked(audioTokenArrs, axis: 0).reshaped([1, numStreams - 1, 1])

            // Forward through temporal transformer
            let (hidden, textLogits) = temporal.forward(
                textTokens: textTokenArr,
                audioTokens: audioTokens,
                offset: step
            )
            eval(hidden, textLogits)

            // During silence/text prompt, skip sampling
            if step < generationStartStep {
                continue
            }

            // Sample text token
            let textToken = sampleTopK(
                logits: textLogits.squeezed(axis: 1),
                temperature: cfg.sampling.textTemp,
                topK: cfg.sampling.textTopK
            )
            eval(textToken)

            // Generate agent audio tokens via depformer (with per-codebook repetition penalty)
            let agentCodes = depformer.generate(
                temporalHidden: hidden,
                textToken: textToken
            ) { logits, cbIdx in
                let windowSize = cfg.sampling.repetitionWindow
                let history = Array(agentTokens[cbIdx].suffix(windowSize))
                return sampleTopKWithPenalty(
                    logits: logits,
                    temperature: cfg.sampling.audioTemp,
                    topK: cfg.sampling.audioTopK,
                    pastTokens: history,
                    penalty: cfg.sampling.audioRepetitionPenalty
                )
            }
            eval(agentCodes)

            // Write generated tokens into cache for next step
            let nextStep = step + 1
            if nextStep < totalLen {
                // Text token
                let textVal = textToken[0].item(Int32.self)
                tokenCache[0][nextStep + delays[0]] = textVal

                // Agent audio tokens → streams 1-8 (first nQ codebooks feed back)
                let agentArr = agentCodes[0]  // [numSteps]
                for cb in 0..<cfg.depformer.numSteps {
                    let tok = agentArr[cb].item(Int32.self)
                    if cb < nQ {
                        let agentStreamIdx = 1 + cb
                        let delayedPos = nextStep + delays[agentStreamIdx]
                        if delayedPos < totalLen {
                            tokenCache[agentStreamIdx][delayedPos] = tok
                        }
                    }
                    agentTokens[cb].append(tok)
                }

                // User audio: fill with sine tokens after user audio ends
                if step >= prefillLen {
                    for cb in 0..<nQ {
                        let userIdx = 1 + nQ + cb
                        let delayedPos = nextStep + delays[userIdx]
                        if delayedPos < totalLen {
                            tokenCache[userIdx][delayedPos] = TemporalTransformerConfig.sineTokens[cb]
                        }
                    }
                }
            }
        }

        if verbose {
            let genTime = CFAbsoluteTimeGetCurrent() - genStart
            let totalSteps = prefillLen + maxSteps
            let msPerStep = genTime / Double(totalSteps) * 1000
            print("  Generation: \(String(format: "%.2f", genTime))s, \(String(format: "%.1f", msPerStep))ms/step (\(totalSteps) steps, \(maxSteps) gen)")
        }

        // 6. Decode agent tokens with Mimi
        let decStart = CFAbsoluteTimeGetCurrent()
        let numAgentFrames = agentTokens[0].count
        guard numAgentFrames > 0 else { return [] }

        // Build [1, numCodebooks, T] tensor for Mimi decoder
        var codeMatrix = [[Int32]](repeating: [Int32](repeating: 0, count: numAgentFrames),
                                   count: cfg.mimi.numCodebooks)
        for cb in 0..<min(cfg.depformer.numSteps, cfg.mimi.numCodebooks) {
            codeMatrix[cb] = agentTokens[cb]
        }

        let flatCodes = codeMatrix.flatMap { $0 }
        let codesArr = MLXArray(flatCodes).reshaped([1, cfg.mimi.numCodebooks, numAgentFrames])
        let decoded = mimi.decode(codesArr)  // [1, 1, numSamples]
        eval(decoded)

        if verbose {
            let decTime = CFAbsoluteTimeGetCurrent() - decStart
            print("  Mimi decode: \(String(format: "%.2f", decTime))s")
        }

        // Extract audio samples
        let numSamples = decoded.shape[2]
        var samples = [Float](repeating: 0, count: numSamples)
        let flatDecoded = decoded.reshaped([numSamples])
        eval(flatDecoded)
        for i in 0..<numSamples {
            samples[i] = flatDecoded[i].item(Float.self)
        }

        if verbose {
            let totalTime = CFAbsoluteTimeGetCurrent() - startTime
            let audioDuration = Double(numSamples) / Double(cfg.sampleRate)
            print("  Total: \(String(format: "%.2f", totalTime))s, audio: \(String(format: "%.2f", audioDuration))s, RTF: \(String(format: "%.2f", totalTime / max(audioDuration, 0.001)))")
        }

        return samples
    }

    // MARK: - Model Loading

    public static func fromPretrained(
        modelId: String = "aufklarer/PersonaPlex-7B-MLX-4bit",
        progressHandler: ((Double, String) -> Void)? = nil
    ) async throws -> PersonaPlexModel {
        let cfg = PersonaPlexConfig.default
        let model = PersonaPlexModel(cfg: cfg)

        // Download weights
        progressHandler?(0.05, "Downloading PersonaPlex weights...")
        let modelDir = try HuggingFaceDownloader.getCacheDirectory(for: modelId)

        let weightFiles = [
            "temporal.safetensors",
            "depformer.safetensors",
            "embeddings.safetensors",
            "mimi.safetensors",
            "config.json"
        ]

        try await HuggingFaceDownloader.downloadWeights(
            modelId: modelId,
            to: modelDir,
            additionalFiles: weightFiles
        ) { progress in
            progressHandler?(0.05 + progress * 0.5, "Downloading...")
        }

        // Load weights
        progressHandler?(0.55, "Loading model weights...")
        try PersonaPlexWeightLoader.loadWeights(
            model: model,
            from: modelDir
        ) { progress, status in
            progressHandler?(0.55 + progress * 0.25, status)
        }

        // Load Mimi
        progressHandler?(0.80, "Loading Mimi codec...")
        try PersonaPlexWeightLoader.loadMimi(
            model: model.mimi,
            from: modelDir
        ) { progress, status in
            progressHandler?(0.80 + progress * 0.15, status)
        }

        model.train(false)
        progressHandler?(1.0, "PersonaPlex ready")
        return model
    }
}
