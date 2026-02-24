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

        // 2. Load voice prompt embeddings + cache
        let voiceStart = CFAbsoluteTimeGetCurrent()
        let voiceEmbeddings: MLXArray?
        let voiceCache: MLXArray?  // [1, 17, CT] ring buffer with voice prompt tokens
        do {
            let modelDir = try HuggingFaceDownloader.getCacheDirectory(for: "aufklarer/PersonaPlex-7B-MLX-4bit")
            let voiceDir = modelDir.appendingPathComponent("voices")
            let voiceFile = voiceDir.appendingPathComponent("\(voice.rawValue).safetensors")
            if FileManager.default.fileExists(atPath: voiceFile.path) {
                let weights = try MLX.loadArrays(url: voiceFile)
                voiceEmbeddings = weights["embeddings"]  // [T, 1, 1, dim]
                voiceCache = weights["cache"]             // [1, 17, maxDelay+3]
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

        // Total steps: voice + silence1 + text_prompt + silence2 + user audio + gen
        // The reference model skips offset=0 in prepare_step_input (writes initial tokens,
        // no forward pass). The first real forward pass uses voice[0] at position 0.
        // So RoPE offset = step (matching PyTorch's transformer offset exactly).
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

        // --- Apply voice prompt cache ---
        // The voice .safetensors contains a ring buffer snapshot [1, 17, CT] with the actual
        // tokens that were in the delay buffer after voice prompt creation. This includes real
        // voice audio tokens for agent streams (not silence). We overwrite the last few
        // positions of the voice prompt phase so subsequent reads get correct token values.
        if let vc = voiceCache, voiceFrameCount > 0 {
            let CT = maxDelay + 3  // ring buffer size (4 for PersonaPlex)
            eval(vc)
            // Map ring buffer positions to flat array positions.
            // Python state.offset after voice prompt = V+1 (V voice steps + 1 init skip).
            // Python reads ring[(state.offset-1) % CT]; Swift reads tokenCache[step-1].
            // With RoPE offset=step, the transformer position matches Python's internal offset.
            // Mapping: tokenCache[flatPos] = cache[s, (flatPos + 1) % CT].
            for s in 0..<numStreams {
                let d = delays[s]
                for k in 0...d {
                    let flatPos = voiceFrameCount - 1 + k
                    let ringPos = (voiceFrameCount + k) % CT
                    if flatPos >= 0 && flatPos < totalLen {
                        tokenCache[s][flatPos] = Int32(vc[0, s, ringPos].item(Float.self))
                    }
                }
            }
        }

        // --- Phase 2: Silence spacer 1 ---
        var pos = voiceFrameCount
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

        // 5. Batched prefill
        //
        // Phase layout:
        //   steps 0..<voiceFrameCount:    voice prompt (batched embeddings)
        //   steps voiceFrameCount..<promptLen: silence + text + silence (batched tokens)
        //   steps promptLen..<prefillLen:  user audio + simultaneous agent generation (per-step)
        //   steps prefillLen..<prefillLen+maxSteps: post-user generation (per-step)
        var agentTokens: [[Int32]] = Array(repeating: [], count: cfg.depformer.numSteps)
        let genStart = CFAbsoluteTimeGetCurrent()

        // --- Batch voice prompt ---
        if let voiceEmb = voiceEmbeddings, voiceFrameCount > 0 {
            // voiceEmb: [T, 1, 1, dim] → [1, T, dim]
            let batchEmb = voiceEmb.reshaped([voiceFrameCount, cfg.temporal.dim])
                .expandedDimensions(axis: 0)
            temporal.forwardBatchEmbedding(batchEmb, offset: 0)
        }

        if verbose {
            let vpTime = CFAbsoluteTimeGetCurrent() - genStart
            print("  Voice prefill: \(voiceFrameCount) frames batched (\(String(format: "%.2f", vpTime))s)")
        }

        // --- Batch non-voice prefill (silence1 + text prompt + silence2) ---
        let nonVoicePrefillLen = silenceFrameCount + textPromptLen + silenceFrameCount
        if nonVoicePrefillLen > 0 {
            var batchText = [Int32](repeating: 0, count: nonVoicePrefillLen)
            var batchAudioFlat = [Int32](repeating: 0, count: (numStreams - 1) * nonVoicePrefillLen)

            for t in 0..<nonVoicePrefillLen {
                let globalStep = voiceFrameCount + t
                let readIdx = globalStep > 0 ? globalStep - 1 : 0
                batchText[t] = globalStep > 0 ? tokenCache[0][readIdx] : Int32(cfg.temporal.textPaddingId)
                for stream in 1..<numStreams {
                    let tok = globalStep > 0 ? tokenCache[stream][readIdx] : Int32(-1)
                    batchAudioFlat[(stream - 1) * nonVoicePrefillLen + t] = tok
                }
            }

            let textArr = MLXArray(batchText).reshaped([1, nonVoicePrefillLen])
            let audioArr = MLXArray(batchAudioFlat).reshaped([1, numStreams - 1, nonVoicePrefillLen])

            let (prefillHidden, _) = temporal.forward(
                textTokens: textArr,
                audioTokens: audioArr,
                offset: voiceFrameCount
            )
            eval(prefillHidden)  // populate KV caches
        }

        if verbose {
            let pfTime = CFAbsoluteTimeGetCurrent() - genStart
            print("  Total prefill: \(voiceFrameCount + nonVoicePrefillLen) steps batched (\(String(format: "%.2f", pfTime))s)")
        }

        // 6. Per-step generation loop (user audio + post-user)
        //
        // When compiled, we separate embedding computation (not compiled — uses Slice)
        // from transformer layers (compiled — pure matmuls/attention).
        let useCompiledStep = temporal.compiledStep != nil
        let b = 1  // batch size

        for step in promptLen..<(prefillLen + maxSteps) {
            // Build input tokens for this step.
            // Original Moshi reads (offset - 1) % CT — the PREVIOUS step's token.
            let readIdx = step - 1
            let textTok = tokenCache[0][readIdx]
            let textTokenArr = MLXArray([textTok]).reshaped([1, 1])
            var audioTokenArrs: [MLXArray] = []
            for stream in 1..<numStreams {
                let tok = tokenCache[stream][readIdx]
                audioTokenArrs.append(MLXArray([tok]))
            }
            let audioTokens = stacked(audioTokenArrs, axis: 0).reshaped([1, numStreams - 1, 1])

            let hidden: MLXArray
            let textLogits: MLXArray

            if useCompiledStep {
                // Compute embeddings (not compiled — uses per-stream Slice indexing)
                var embSum = temporal.text_emb(textTokenArr)  // [1, 1, dim]
                for i in 0..<cfg.temporal.numAudioEmbeddings {
                    let rawTok = audioTokens[0..<b, i, 0..<1]
                    let isValid = rawTok .>= MLXArray(Int32(0))
                    let safeTok = MLX.maximum(rawTok, MLXArray(Int32(0)))
                    let embResult = temporal.emb[i](safeTok)
                    let mask = isValid.expandedDimensions(axis: -1)
                    embSum = embSum + MLX.where(mask, embResult, MLXArray(Float(0)))
                }

                // Compiled transformer layers + out_norm + text_linear
                (hidden, textLogits) = temporal.executeStep(hidden: embSum, offset: step)
            } else {
                // Uncompiled path: full forward
                (hidden, textLogits) = temporal.forward(
                    textTokens: textTokenArr,
                    audioTokens: audioTokens,
                    offset: step
                )
            }

            // Sample text token — eval here forces textLogits (and hidden) computation.
            // Depformer needs textToken value for embedding lookup.
            let textToken = sampleTopK(
                logits: textLogits.squeezed(axis: 1),
                temperature: cfg.sampling.textTemp,
                topK: cfg.sampling.textTopK
            )
            eval(textToken)

            // Build provided tokens for depformer conditioning during user audio.
            // In Python Moshi, the depformer uses real user audio tokens (from the cache
            // target position) as conditioning for user audio codebook steps (8-15).
            // This ensures the autoregressive chain within the depformer sees real audio
            // rather than its own potentially wrong predictions.
            var providedTokens: [Int32]? = nil
            if step < prefillLen {
                var provided = [Int32](repeating: -1, count: cfg.depformer.numSteps)
                for cb in 0..<nQ {
                    let userStreamIdx = 1 + nQ + cb
                    // Read from position `step` — matches Python's target_position.
                    // For delay-0 streams: current step's user audio.
                    // For delay-1 streams: previous step's user audio (written at pos-1+1=pos).
                    if step >= 0 && step < totalLen {
                        let tok = tokenCache[userStreamIdx][step]
                        if tok >= 0 {
                            provided[nQ + cb] = tok  // depformer step nQ+cb = user cb
                        }
                    }
                }
                providedTokens = provided
            }

            // Generate audio tokens via depformer (with per-codebook repetition penalty)
            let agentCodes = depformer.generate(
                temporalHidden: hidden,
                textToken: textToken,
                providedTokens: providedTokens
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
            // No eval barrier — .item() calls below trigger eval implicitly

            // Write generated tokens into cache at position `step` (NO delay).
            // Critical: In Python Moshi, process_transformer_output() writes ALL depformer
            // tokens at target_position = offset % CT (same position for all streams).
            // The delay is only used for external input (user audio, prompt tokens).
            // Writing at `step` (not `step + delays[k]`) ensures the next step immediately
            // reads the depformer's output for ALL streams, including delay-1 streams.
            let textVal = textToken[0].item(Int32.self)
            if step < totalLen {
                tokenCache[0][step] = textVal
            }

            // Agent audio tokens → streams 1-8
            let agentArr = agentCodes[0]  // [numSteps]
            for cb in 0..<nQ {
                let tok = agentArr[cb].item(Int32.self)
                if step < totalLen {
                    tokenCache[1 + cb][step] = tok
                }
                agentTokens[cb].append(tok)
            }

            // User audio predictions (depformer steps 8-15) → streams 9-16
            // During user audio: don't write (user audio already pre-filled, matches
            //   Python's provided=True preventing depformer overwrites).
            // After user audio: write depformer predictions (matches Python's provided=False
            //   allowing depformer writes).
            for cb in nQ..<cfg.depformer.numSteps {
                let tok = agentArr[cb].item(Int32.self)
                if step >= prefillLen && step < totalLen {
                    tokenCache[1 + cb][step] = tok
                }
                agentTokens[cb].append(tok)
            }
        }

        if verbose {
            let genTime = CFAbsoluteTimeGetCurrent() - genStart
            let genSteps = userFrameCount + maxSteps
            let msPerGenStep = genSteps > 0 ? genTime / Double(genSteps) * 1000 : 0
            print("  Generation: \(String(format: "%.2f", genTime))s, \(String(format: "%.1f", msPerGenStep))ms/step (\(genSteps) gen steps)")
        }

        // 6. Decode agent tokens with Mimi
        // Only use first nQ (8) codebooks — the depformer generates dep_q=16 tokens per step,
        // but only the first 8 are agent audio codebooks (streams 1-8). The remaining 8 are
        // predictions for user audio codebooks (streams 9-16, unused for decoding).
        // Original PersonaPlex: mimi.set_num_codebooks(8); pcm = mimi.decode(tokens[:, 1:9])
        let decStart = CFAbsoluteTimeGetCurrent()
        let numAgentFrames = agentTokens[0].count
        guard numAgentFrames > 0 else { return [] }

        let numDecodeCodebooks = nQ  // 8 (matching set_num_codebooks(8) in original)
        var codeMatrix = [[Int32]](repeating: [Int32](repeating: 0, count: numAgentFrames),
                                   count: numDecodeCodebooks)
        for cb in 0..<numDecodeCodebooks {
            codeMatrix[cb] = agentTokens[cb]
        }

        let flatCodes = codeMatrix.flatMap { $0 }
        let codesArr = MLXArray(flatCodes).reshaped([1, numDecodeCodebooks, numAgentFrames])
        let decoded = mimi.decode(codesArr)  // [1, 1, numSamples]
        eval(decoded)

        if verbose {
            let decTime = CFAbsoluteTimeGetCurrent() - decStart
            print("  Mimi decode: \(String(format: "%.2f", decTime))s")
        }

        // Extract audio samples (bulk copy — single GPU sync instead of per-sample)
        let numSamples = decoded.shape[2]
        let flatDecoded = decoded.reshaped([numSamples])
        eval(flatDecoded)
        let samples = flatDecoded.asArray(Float.self)

        if verbose {
            let totalTime = CFAbsoluteTimeGetCurrent() - startTime
            let audioDuration = Double(numSamples) / Double(cfg.sampleRate)
            print("  Total: \(String(format: "%.2f", totalTime))s, audio: \(String(format: "%.2f", audioDuration))s, RTF: \(String(format: "%.2f", totalTime / max(audioDuration, 0.001)))")
        }

        return samples
    }

    // MARK: - Streaming Inference

    // AudioChunk is defined in Qwen3Common/Protocols.swift

    /// Streaming configuration.
    public struct PersonaPlexStreamingConfig: Sendable {
        /// Frames to accumulate before first chunk emission
        public var firstChunkFrames: Int
        /// Frames to accumulate for subsequent chunks
        public var chunkFrames: Int

        public init(firstChunkFrames: Int = 25, chunkFrames: Int = 25) {
            self.firstChunkFrames = firstChunkFrames
            self.chunkFrames = chunkFrames
        }

        public static let `default` = PersonaPlexStreamingConfig()
    }

    /// Stream response audio in chunks during generation.
    ///
    /// Same pipeline as `respond()` but emits audio chunks incrementally via Mimi's
    /// streaming decoder (`decodeStep()`), enabling playback before generation completes.
    ///
    /// - Parameters:
    ///   - userAudio: [numSamples] float array of 24kHz mono audio
    ///   - voice: voice preset for the agent
    ///   - systemPromptTokens: SentencePiece-tokenized system prompt (nil = default)
    ///   - maxSteps: maximum generation steps (at 12.5 Hz)
    ///   - streaming: streaming configuration (chunk sizes)
    ///   - verbose: print timing info
    /// - Returns: AsyncThrowingStream of audio chunks
    public func respondStream(
        userAudio: [Float],
        voice: PersonaPlexVoice = .NATM0,
        systemPromptTokens: [Int32]? = nil,
        maxSteps: Int = 500,
        streaming: PersonaPlexStreamingConfig = .default,
        verbose: Bool = false
    ) -> AsyncThrowingStream<AudioChunk, Error> {
        AsyncThrowingStream { continuation in
            Task {
                do {
                    let startTime = CFAbsoluteTimeGetCurrent()

                    // 1. Encode user audio with Mimi
                    let audioArray = MLXArray(userAudio).reshaped([1, 1, userAudio.count])
                    let userCodes = mimi.encode(audioArray)
                    eval(userCodes)
                    let userFrameCount = userCodes.shape[2]

                    // 2. Load voice prompt
                    let voiceEmbeddings: MLXArray?
                    let voiceCache: MLXArray?
                    do {
                        let modelDir = try HuggingFaceDownloader.getCacheDirectory(
                            for: "aufklarer/PersonaPlex-7B-MLX-4bit")
                        let voiceFile = modelDir.appendingPathComponent("voices")
                            .appendingPathComponent("\(voice.rawValue).safetensors")
                        if FileManager.default.fileExists(atPath: voiceFile.path) {
                            let weights = try MLX.loadArrays(url: voiceFile)
                            voiceEmbeddings = weights["embeddings"]
                            voiceCache = weights["cache"]
                        } else {
                            voiceEmbeddings = nil; voiceCache = nil
                        }
                    } catch { voiceEmbeddings = nil; voiceCache = nil }

                    let voiceFrameCount = voiceEmbeddings?.shape[0] ?? 0
                    let silenceFrameCount = Int(0.5 * cfg.mimi.frameRate)
                    let textPromptTokens = systemPromptTokens
                        ?? TemporalTransformerConfig.defaultSystemPromptTokens
                    let textPromptLen = textPromptTokens.count
                    let delays = cfg.delays
                    let maxDelay = cfg.maxDelay
                    let numStreams = cfg.numStreams
                    let nQ = cfg.temporal.nQ
                    let promptLen = voiceFrameCount + silenceFrameCount + textPromptLen + silenceFrameCount
                    let prefillLen = promptLen + userFrameCount
                    let totalLen = prefillLen + maxSteps + maxDelay + 3

                    // 3. Reset caches + initialize streaming decoder
                    temporal.resetCache()
                    mimi.resetState()
                    let streamingDecoder = MimiStreamingDecoder(mimi)

                    // 4. Initialize token cache (same as respond())
                    var tokenCache = [[Int32]](
                        repeating: [Int32](repeating: -1, count: totalLen), count: numStreams)

                    // Fill token cache phases (voice, silence, text, silence, user audio)
                    for t in 0..<voiceFrameCount {
                        tokenCache[0][t + delays[0]] = Int32(cfg.temporal.textPaddingId)
                        for cb in 0..<nQ {
                            let s = 1 + cb
                            tokenCache[s][t + delays[s]] = TemporalTransformerConfig.silenceTokens[cb]
                        }
                        for cb in 0..<nQ {
                            let s = 1 + nQ + cb
                            tokenCache[s][t + delays[s]] = TemporalTransformerConfig.sineTokens[cb]
                        }
                    }
                    if let vc = voiceCache, voiceFrameCount > 0 {
                        let CT = maxDelay + 3
                        eval(vc)
                        for s in 0..<numStreams {
                            let d = delays[s]
                            for k in 0...d {
                                let flatPos = voiceFrameCount - 1 + k
                                let ringPos = (voiceFrameCount + k) % CT
                                if flatPos >= 0 && flatPos < totalLen {
                                    tokenCache[s][flatPos] = Int32(vc[0, s, ringPos].item(Float.self))
                                }
                            }
                        }
                    }
                    var pos = voiceFrameCount
                    for _ in 0..<silenceFrameCount {
                        tokenCache[0][pos + delays[0]] = Int32(cfg.temporal.textPaddingId)
                        for cb in 0..<nQ {
                            let s = 1 + cb
                            tokenCache[s][pos + delays[s]] = TemporalTransformerConfig.silenceTokens[cb]
                        }
                        for cb in 0..<nQ {
                            let s = 1 + nQ + cb
                            tokenCache[s][pos + delays[s]] = TemporalTransformerConfig.sineTokens[cb]
                        }
                        pos += 1
                    }
                    for t in 0..<textPromptLen {
                        tokenCache[0][pos + delays[0]] = textPromptTokens[t]
                        for cb in 0..<nQ {
                            let s = 1 + cb
                            tokenCache[s][pos + delays[s]] = TemporalTransformerConfig.silenceTokens[cb]
                        }
                        for cb in 0..<nQ {
                            let s = 1 + nQ + cb
                            tokenCache[s][pos + delays[s]] = TemporalTransformerConfig.sineTokens[cb]
                        }
                        pos += 1
                    }
                    for _ in 0..<silenceFrameCount {
                        tokenCache[0][pos + delays[0]] = Int32(cfg.temporal.textPaddingId)
                        for cb in 0..<nQ {
                            let s = 1 + cb
                            tokenCache[s][pos + delays[s]] = TemporalTransformerConfig.silenceTokens[cb]
                        }
                        for cb in 0..<nQ {
                            let s = 1 + nQ + cb
                            tokenCache[s][pos + delays[s]] = TemporalTransformerConfig.sineTokens[cb]
                        }
                        pos += 1
                    }
                    let userCodesArr = userCodes.asType(.int32); eval(userCodesArr)
                    for t in 0..<userFrameCount {
                        tokenCache[0][pos + delays[0]] = Int32(cfg.temporal.textPaddingId)
                        for cb in 0..<nQ {
                            let s = 1 + cb
                            tokenCache[s][pos + delays[s]] = TemporalTransformerConfig.silenceTokens[cb]
                        }
                        for cb in 0..<min(nQ, userCodes.shape[1]) {
                            let s = 1 + nQ + cb
                            tokenCache[s][pos + delays[s]] = userCodesArr[0, cb, t].item(Int32.self)
                        }
                        pos += 1
                    }

                    // 5. Batched prefill (same as respond)
                    if let voiceEmb = voiceEmbeddings, voiceFrameCount > 0 {
                        let batchEmb = voiceEmb.reshaped([voiceFrameCount, cfg.temporal.dim])
                            .expandedDimensions(axis: 0)
                        temporal.forwardBatchEmbedding(batchEmb, offset: 0)
                    }
                    let nonVoicePrefillLen = silenceFrameCount + textPromptLen + silenceFrameCount
                    if nonVoicePrefillLen > 0 {
                        var batchText = [Int32](repeating: 0, count: nonVoicePrefillLen)
                        var batchAudioFlat = [Int32](
                            repeating: 0, count: (numStreams - 1) * nonVoicePrefillLen)
                        for t in 0..<nonVoicePrefillLen {
                            let globalStep = voiceFrameCount + t
                            let readIdx = globalStep > 0 ? globalStep - 1 : 0
                            batchText[t] = globalStep > 0
                                ? tokenCache[0][readIdx] : Int32(cfg.temporal.textPaddingId)
                            for stream in 1..<numStreams {
                                let tok = globalStep > 0 ? tokenCache[stream][readIdx] : Int32(-1)
                                batchAudioFlat[(stream - 1) * nonVoicePrefillLen + t] = tok
                            }
                        }
                        let textArr = MLXArray(batchText).reshaped([1, nonVoicePrefillLen])
                        let audioArr = MLXArray(batchAudioFlat)
                            .reshaped([1, numStreams - 1, nonVoicePrefillLen])
                        let (ph, _) = temporal.forward(
                            textTokens: textArr, audioTokens: audioArr, offset: voiceFrameCount)
                        eval(ph)
                    }

                    if verbose {
                        let pfTime = CFAbsoluteTimeGetCurrent() - startTime
                        print("  Stream prefill: \(String(format: "%.2f", pfTime))s")
                    }

                    // 6. Generation loop with streaming decode
                    var agentTokens: [[Int32]] = Array(repeating: [], count: cfg.depformer.numSteps)
                    var pendingCodes: [[Int32]] = Array(repeating: [], count: nQ)
                    var totalEmittedFrames = 0
                    let useCompiledStep = temporal.compiledStep != nil
                    let b = 1
                    let genStart = CFAbsoluteTimeGetCurrent()

                    for step in promptLen..<(prefillLen + maxSteps) {
                        let readIdx = step - 1
                        let textTok = tokenCache[0][readIdx]
                        let textTokenArr = MLXArray([textTok]).reshaped([1, 1])
                        var audioTokenArrs: [MLXArray] = []
                        for stream in 1..<numStreams {
                            audioTokenArrs.append(MLXArray([tokenCache[stream][readIdx]]))
                        }
                        let audioTokens = stacked(audioTokenArrs, axis: 0)
                            .reshaped([1, numStreams - 1, 1])

                        let hidden: MLXArray
                        let textLogits: MLXArray
                        if useCompiledStep {
                            var embSum = temporal.text_emb(textTokenArr)
                            for i in 0..<cfg.temporal.numAudioEmbeddings {
                                let rawTok = audioTokens[0..<b, i, 0..<1]
                                let isValid = rawTok .>= MLXArray(Int32(0))
                                let safeTok = MLX.maximum(rawTok, MLXArray(Int32(0)))
                                let embResult = temporal.emb[i](safeTok)
                                let mask = isValid.expandedDimensions(axis: -1)
                                embSum = embSum + MLX.where(mask, embResult, MLXArray(Float(0)))
                            }
                            (hidden, textLogits) = temporal.executeStep(
                                hidden: embSum, offset: step)
                        } else {
                            (hidden, textLogits) = temporal.forward(
                                textTokens: textTokenArr, audioTokens: audioTokens, offset: step)
                        }

                        let textToken = sampleTopK(
                            logits: textLogits.squeezed(axis: 1),
                            temperature: cfg.sampling.textTemp,
                            topK: cfg.sampling.textTopK)
                        eval(textToken)

                        var providedTokens: [Int32]? = nil
                        if step < prefillLen {
                            var provided = [Int32](repeating: -1, count: cfg.depformer.numSteps)
                            for cb in 0..<nQ {
                                let userStreamIdx = 1 + nQ + cb
                                if step < totalLen {
                                    let tok = tokenCache[userStreamIdx][step]
                                    if tok >= 0 { provided[nQ + cb] = tok }
                                }
                            }
                            providedTokens = provided
                        }

                        let agentCodes = depformer.generate(
                            temporalHidden: hidden, textToken: textToken,
                            providedTokens: providedTokens
                        ) { logits, cbIdx in
                            let history = Array(agentTokens[cbIdx].suffix(
                                cfg.sampling.repetitionWindow))
                            return sampleTopKWithPenalty(
                                logits: logits, temperature: cfg.sampling.audioTemp,
                                topK: cfg.sampling.audioTopK, pastTokens: history,
                                penalty: cfg.sampling.audioRepetitionPenalty)
                        }

                        // Extract tokens and write to cache
                        let textVal = textToken[0].item(Int32.self)
                        if step < totalLen { tokenCache[0][step] = textVal }
                        let agentArr = agentCodes[0]
                        for cb in 0..<nQ {
                            let tok = agentArr[cb].item(Int32.self)
                            if step < totalLen { tokenCache[1 + cb][step] = tok }
                            agentTokens[cb].append(tok)
                            pendingCodes[cb].append(tok)
                        }
                        for cb in nQ..<cfg.depformer.numSteps {
                            let tok = agentArr[cb].item(Int32.self)
                            if step >= prefillLen && step < totalLen {
                                tokenCache[1 + cb][step] = tok
                            }
                            agentTokens[cb].append(tok)
                        }

                        // Emit chunk when enough frames accumulated
                        let threshold = totalEmittedFrames == 0
                            ? streaming.firstChunkFrames : streaming.chunkFrames
                        if pendingCodes[0].count >= threshold {
                            let chunkFrames = pendingCodes[0].count
                            let flatCodes = pendingCodes.flatMap { $0 }
                            let codesArr = MLXArray(flatCodes)
                                .reshaped([1, nQ, chunkFrames])
                            let decoded = streamingDecoder.decodeFrames(codesArr)
                            eval(decoded)
                            let flat = decoded.reshaped([decoded.shape[2]])
                            let samples = flat.asArray(Float.self)

                            let elapsed = CFAbsoluteTimeGetCurrent() - genStart
                            continuation.yield(AudioChunk(
                                samples: samples,
                                sampleRate: cfg.sampleRate,
                                frameIndex: totalEmittedFrames,
                                isFinal: false,
                                elapsedTime: elapsed))

                            if verbose {
                                print("  Chunk: \(samples.count) samples at frame \(totalEmittedFrames) (\(String(format: "%.2f", elapsed))s)")
                            }

                            totalEmittedFrames += chunkFrames
                            pendingCodes = Array(repeating: [], count: nQ)
                        }
                    }

                    // Final chunk with remaining frames
                    if !pendingCodes[0].isEmpty {
                        let chunkFrames = pendingCodes[0].count
                        let flatCodes = pendingCodes.flatMap { $0 }
                        let codesArr = MLXArray(flatCodes).reshaped([1, nQ, chunkFrames])
                        let decoded = streamingDecoder.decodeFrames(codesArr)
                        eval(decoded)
                        let flat = decoded.reshaped([decoded.shape[2]])
                        let samples = flat.asArray(Float.self)

                        let elapsed = CFAbsoluteTimeGetCurrent() - genStart
                        continuation.yield(AudioChunk(
                            samples: samples,
                            sampleRate: cfg.sampleRate,
                            frameIndex: totalEmittedFrames,
                            isFinal: true,
                            elapsedTime: elapsed))

                        if verbose {
                            print("  Final chunk: \(samples.count) samples (\(String(format: "%.2f", elapsed))s)")
                        }
                    }

                    if verbose {
                        let totalTime = CFAbsoluteTimeGetCurrent() - startTime
                        print("  Stream total: \(String(format: "%.2f", totalTime))s")
                    }

                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    // MARK: - Diagnostic Info

    public struct DiagnosticInfo {
        public var textTokens: [Int32] = []
        public var agentTokensByCodebook: [[Int32]] = []
        public var hiddenStats: [(mean: Float, std: Float, min: Float, max: Float)] = []
        public var textLogitStats: [(topToken: Int32, topLogit: Float, entropy: Float)] = []
        public var inputTokenSnapshots: [[(stream: Int, token: Int32)]] = []
    }

    /// Same as respond() but captures diagnostic info for debugging.
    public func respondDiagnostic(
        userAudio: [Float],
        voice: PersonaPlexVoice = .NATM0,
        systemPromptTokens: [Int32]? = nil,
        maxSteps: Int = 500
    ) -> (audio: [Float], diag: DiagnosticInfo) {
        var diag = DiagnosticInfo()

        let audioArray = MLXArray(userAudio).reshaped([1, 1, userAudio.count])
        let userCodes = mimi.encode(audioArray)
        eval(userCodes)
        let userFrameCount = userCodes.shape[2]

        let voiceEmbeddings: MLXArray?
        let voiceCache2: MLXArray?
        do {
            let modelDir = try HuggingFaceDownloader.getCacheDirectory(for: "aufklarer/PersonaPlex-7B-MLX-4bit")
            let voiceDir = modelDir.appendingPathComponent("voices")
            let voiceFile = voiceDir.appendingPathComponent("\(voice.rawValue).safetensors")
            if FileManager.default.fileExists(atPath: voiceFile.path) {
                let weights = try MLX.loadArrays(url: voiceFile)
                voiceEmbeddings = weights["embeddings"]
                voiceCache2 = weights["cache"]
            } else { voiceEmbeddings = nil; voiceCache2 = nil }
        } catch { voiceEmbeddings = nil; voiceCache2 = nil }

        let voiceFrameCount = voiceEmbeddings?.shape[0] ?? 0
        let silenceFrameCount = Int(0.5 * cfg.mimi.frameRate)
        let textPromptTokens = systemPromptTokens ?? TemporalTransformerConfig.defaultSystemPromptTokens
        let textPromptLen = textPromptTokens.count

        temporal.resetCache()
        mimi.resetState()

        let promptLen = voiceFrameCount + silenceFrameCount + textPromptLen + silenceFrameCount
        let prefillLen = promptLen + userFrameCount
        let delays = cfg.delays
        let numStreams = cfg.numStreams
        let nQ = cfg.temporal.nQ
        let totalLen = prefillLen + maxSteps + cfg.maxDelay + 3

        var tokenCache = [[Int32]](repeating: [Int32](repeating: -1, count: totalLen), count: numStreams)

        // Pre-fill phases (same as respond)
        for t in 0..<voiceFrameCount {
            tokenCache[0][t + delays[0]] = Int32(cfg.temporal.textPaddingId)
            for cb in 0..<nQ {
                let s = 1 + cb; tokenCache[s][t + delays[s]] = TemporalTransformerConfig.silenceTokens[cb]
            }
            for cb in 0..<nQ {
                let s = 1 + nQ + cb; tokenCache[s][t + delays[s]] = TemporalTransformerConfig.sineTokens[cb]
            }
        }
        // Apply voice prompt cache (same as respond)
        if let vc = voiceCache2, voiceFrameCount > 0 {
            let CT = cfg.maxDelay + 3
            eval(vc)
            for s in 0..<numStreams {
                let d = delays[s]
                for k in 0...d {
                    let flatPos = voiceFrameCount - 1 + k
                    let ringPos = (voiceFrameCount + k) % CT
                    if flatPos >= 0 && flatPos < totalLen {
                        tokenCache[s][flatPos] = Int32(vc[0, s, ringPos].item(Float.self))
                    }
                }
            }
        }
        var pos = voiceFrameCount
        for _ in 0..<silenceFrameCount {
            tokenCache[0][pos + delays[0]] = Int32(cfg.temporal.textPaddingId)
            for cb in 0..<nQ { let s = 1+cb; tokenCache[s][pos+delays[s]] = TemporalTransformerConfig.silenceTokens[cb] }
            for cb in 0..<nQ { let s = 1+nQ+cb; tokenCache[s][pos+delays[s]] = TemporalTransformerConfig.sineTokens[cb] }
            pos += 1
        }
        for t in 0..<textPromptLen {
            tokenCache[0][pos + delays[0]] = textPromptTokens[t]
            for cb in 0..<nQ { let s = 1+cb; tokenCache[s][pos+delays[s]] = TemporalTransformerConfig.silenceTokens[cb] }
            for cb in 0..<nQ { let s = 1+nQ+cb; tokenCache[s][pos+delays[s]] = TemporalTransformerConfig.sineTokens[cb] }
            pos += 1
        }
        for _ in 0..<silenceFrameCount {
            tokenCache[0][pos + delays[0]] = Int32(cfg.temporal.textPaddingId)
            for cb in 0..<nQ { let s = 1+cb; tokenCache[s][pos+delays[s]] = TemporalTransformerConfig.silenceTokens[cb] }
            for cb in 0..<nQ { let s = 1+nQ+cb; tokenCache[s][pos+delays[s]] = TemporalTransformerConfig.sineTokens[cb] }
            pos += 1
        }
        let userCodesArr = userCodes.asType(.int32); eval(userCodesArr)
        for t in 0..<userFrameCount {
            tokenCache[0][pos + delays[0]] = Int32(cfg.temporal.textPaddingId)
            for cb in 0..<nQ { let s = 1+cb; tokenCache[s][pos+delays[s]] = TemporalTransformerConfig.silenceTokens[cb] }
            for cb in 0..<min(nQ, userCodes.shape[1]) {
                let s = 1+nQ+cb; tokenCache[s][pos+delays[s]] = userCodesArr[0, cb, t].item(Int32.self)
            }
            pos += 1
        }

        var agentTokens: [[Int32]] = Array(repeating: [], count: cfg.depformer.numSteps)
        let generationStartStep = promptLen

        for step in 0..<(prefillLen + maxSteps) {
            if step < voiceFrameCount, let voiceEmb = voiceEmbeddings {
                let emb = voiceEmb[step].reshaped([1, 1, cfg.temporal.dim])
                temporal.forwardEmbedding(emb, offset: step)
                continue
            }

            let readIdx = step > 0 ? step - 1 : 0
            let textTok = step > 0 ? tokenCache[0][readIdx] : Int32(cfg.temporal.textPaddingId)
            let textTokenArr = MLXArray([textTok]).reshaped([1, 1])
            var audioTokenArrs: [MLXArray] = []
            for stream in 1..<numStreams {
                let tok = step > 0 ? tokenCache[stream][readIdx] : Int32(-1)
                audioTokenArrs.append(MLXArray([tok]))
            }
            let audioTokens = stacked(audioTokenArrs, axis: 0).reshaped([1, numStreams - 1, 1])

            let (hidden, textLogits) = temporal.forward(
                textTokens: textTokenArr, audioTokens: audioTokens, offset: step)
            eval(hidden, textLogits)

            // Capture hidden state stats for first 20 gen steps
            if step >= generationStartStep && diag.hiddenStats.count < 20 {
                let h = hidden.reshaped([-1])
                let hMean = MLX.mean(h).item(Float.self)
                let hStd = MLX.sqrt(MLX.mean((h - MLXArray(hMean)) * (h - MLXArray(hMean)))).item(Float.self)
                let hMin = MLX.min(h).item(Float.self)
                let hMax = MLX.max(h).item(Float.self)
                diag.hiddenStats.append((mean: hMean, std: hStd, min: hMin, max: hMax))

                // Text logit stats
                let tl = textLogits.squeezed(axes: [0, 1])  // [vocabSize]
                let topIdx = argMax(tl).item(Int32.self)
                let topVal = tl[Int(topIdx)].item(Float.self)
                let probs = softMax(tl, axis: -1)
                let logProbs = log(probs + MLXArray(Float(1e-10)))
                let ent = -(probs * logProbs).sum().item(Float.self)
                diag.textLogitStats.append((topToken: topIdx, topLogit: topVal, entropy: ent))

                // Snapshot input tokens
                var snapshot: [(stream: Int, token: Int32)] = [(0, textTok)]
                for stream in 1..<min(5, numStreams) {
                    let tok = step > 0 ? tokenCache[stream][readIdx] : Int32(-1)
                    snapshot.append((stream, tok))
                }
                diag.inputTokenSnapshots.append(snapshot)
            }

            if step < generationStartStep { continue }

            let textToken = sampleTopK(
                logits: textLogits.squeezed(axis: 1),
                temperature: cfg.sampling.textTemp, topK: cfg.sampling.textTopK)
            eval(textToken)
            let textVal = textToken[0].item(Int32.self)
            diag.textTokens.append(textVal)

            // Depformer conditioning (same as respond)
            var providedTokensDiag: [Int32]? = nil
            if step < prefillLen {
                var provided = [Int32](repeating: -1, count: cfg.depformer.numSteps)
                for cb in 0..<nQ {
                    let userStreamIdx = 1 + nQ + cb
                    if step >= 0 && step < totalLen {
                        let tok = tokenCache[userStreamIdx][step]
                        if tok >= 0 { provided[nQ + cb] = tok }
                    }
                }
                providedTokensDiag = provided
            }

            let agentCodes = depformer.generate(
                temporalHidden: hidden, textToken: textToken,
                providedTokens: providedTokensDiag
            ) { logits, cbIdx in
                let history = Array(agentTokens[cbIdx].suffix(cfg.sampling.repetitionWindow))
                return sampleTopKWithPenalty(
                    logits: logits, temperature: cfg.sampling.audioTemp,
                    topK: cfg.sampling.audioTopK, pastTokens: history,
                    penalty: cfg.sampling.audioRepetitionPenalty)
            }
            // No eval barrier — .item() calls below trigger eval implicitly

            // Write at position `step` (no delay) — matches Python's target_position
            if step < totalLen { tokenCache[0][step] = textVal }
            let agentArr = agentCodes[0]
            for cb in 0..<nQ {
                let tok = agentArr[cb].item(Int32.self)
                if step < totalLen { tokenCache[1 + cb][step] = tok }
                agentTokens[cb].append(tok)
            }
            for cb in nQ..<cfg.depformer.numSteps {
                let tok = agentArr[cb].item(Int32.self)
                if step >= prefillLen && step < totalLen {
                    tokenCache[1 + cb][step] = tok
                }
                agentTokens[cb].append(tok)
            }
        }

        diag.agentTokensByCodebook = agentTokens

        // Decode
        let numAgentFrames = agentTokens[0].count
        guard numAgentFrames > 0 else { return ([], diag) }
        let numDecodeCodebooks = nQ
        var codeMatrix = [[Int32]](repeating: [Int32](repeating: 0, count: numAgentFrames), count: numDecodeCodebooks)
        for cb in 0..<numDecodeCodebooks { codeMatrix[cb] = agentTokens[cb] }
        let flatCodes = codeMatrix.flatMap { $0 }
        let codesArr = MLXArray(flatCodes).reshaped([1, numDecodeCodebooks, numAgentFrames])
        let decoded = mimi.decode(codesArr)
        eval(decoded)
        let flatDecoded = decoded.reshaped([decoded.shape[2]]); eval(flatDecoded)
        let samples = flatDecoded.asArray(Float.self)

        return (samples, diag)
    }

    // MARK: - Model Loading

    public static func fromPretrained(
        modelId: String = "aufklarer/PersonaPlex-7B-MLX-4bit",
        progressHandler: ((Double, String) -> Void)? = nil
    ) async throws -> PersonaPlexModel {
        // Download weights first to get config
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

        // Read config.json to detect quantization settings
        var cfg = PersonaPlexConfig.default
        let configFile = modelDir.appendingPathComponent("config.json")
        if FileManager.default.fileExists(atPath: configFile.path),
           let data = try? Data(contentsOf: configFile),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let quant = json["quantization"] as? [String: Any] {
            // Temporal quantization (always present when quantization section exists)
            let quantizedComponents = quant["quantized_components"] as? [String] ?? []
            if let bits = quant["bits"] as? Int {
                cfg.temporal.bits = bits
            }
            if let groupSize = quant["group_size"] as? Int {
                cfg.temporal.groupSize = groupSize
            }
            // Depformer quantization (if listed in quantized_components)
            if quantizedComponents.contains("depformer") {
                cfg.depformer.bits = quant["bits"] as? Int ?? 4
                cfg.depformer.groupSize = quant["group_size"] as? Int ?? 64
            } else {
                cfg.depformer.bits = 16
                cfg.depformer.groupSize = 1
            }
        } else {
            // No quantization section → BF16
            cfg.temporal.bits = 16
            cfg.temporal.groupSize = 1
            cfg.depformer.bits = 16
            cfg.depformer.groupSize = 1
        }
        let model = PersonaPlexModel(cfg: cfg)

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

    // MARK: - Compilation & Warmup

    /// Set up compiled temporal transformer for Metal kernel fusion (~30% speedup).
    /// Call after loading weights. Gracefully falls back to uncompiled on failure.
    public func setupCompilation() {
        temporal.setupCompilation()
    }

    /// Run warmup forward passes to trace the compiled graph and JIT-compile Metal shaders.
    /// This eliminates first-inference latency from shader compilation.
    public func warmUp() {
        setupCompilation()

        // Run a single compiled step to trace the graph
        temporal.resetCache()

        // First, do a prefill to initialize KV caches (compiled step needs non-empty cache)
        let dummyText = MLXArray([Int32(3)]).reshaped([1, 1])
        let dummyAudio = MLXArray.zeros([1, cfg.temporal.numAudioEmbeddings, 1], dtype: .int32)
        let (_, _) = temporal.forward(textTokens: dummyText, audioTokens: dummyAudio, offset: 0)

        // Now run a compiled step
        let dummyEmb = MLXArray.zeros([1, 1, cfg.temporal.dim])
        let (normed, logits) = temporal.executeStep(hidden: dummyEmb, offset: 1)
        eval(normed, logits)

        temporal.resetCache()
    }
}
