import Foundation
import MLX
import MLXNN
import MLXFast
import Qwen3Common

/// Errors thrown by streaming TTS synthesis.
public enum TTSError: Error, LocalizedError {
    case tokenizerNotLoaded
    case unknownLanguage(String)

    public var errorDescription: String? {
        switch self {
        case .tokenizerNotLoaded:
            return "Tokenizer not loaded. Call setTokenizer() first."
        case .unknownLanguage(let lang):
            return "Unknown language '\(lang)'"
        }
    }
}

/// A chunk of audio produced during streaming synthesis.
public struct TTSAudioChunk: Sendable {
    /// PCM audio samples at 24 kHz, Float32
    public let samples: [Float]
    /// Sample rate (always 24000)
    public let sampleRate: Int
    /// Index of the first codec frame in this chunk
    public let frameIndex: Int
    /// Cumulative number of codec frames decoded so far
    public let totalFrames: Int
    /// True if this is the last chunk (EOS or max tokens reached)
    public let isFinal: Bool
    /// Wall-clock seconds since synthesis started
    public let elapsedTime: Double
}

/// Main Qwen3-TTS model for text-to-speech synthesis
public class Qwen3TTSModel {
    public let config: Qwen3TTSConfig
    public let talker: TalkerModel
    public let codePredictor: CodePredictorModel
    public let codecDecoder: SpeechTokenizerDecoder

    /// Speaker configuration parsed from config.json (nil for Base model, populated for CustomVoice)
    public private(set) var speakerConfig: SpeakerConfig?

    /// Available speaker names (empty for Base model)
    public var availableSpeakers: [String] { speakerConfig?.availableSpeakers ?? [] }

    private var tokenizer: Qwen3Tokenizer?

    /// Compiled talker generation step (28-layer transformer + codec head) for kernel fusion.
    /// Fuses ~420 Metal kernel dispatches per step into fewer optimized kernels.
    ///
    /// Uses shapeless=true: handles growing KV cache without recompilation.
    /// RoPE offset is passed as a regular function input (compile treats inputs as variables).
    /// Batch dimension uses -1 reshapes so the same compiled graph works for any batch size.
    private var compiledTalkerStep: (([MLXArray]) -> [MLXArray])?

    /// Compiled code predictor transformer (layers + norm, no lm_head) for kernel fusion.
    /// Used for groups 1-14 of per-timestep code prediction (seqLen=1 with cache).
    ///
    /// Uses shapeless=false: one compiled graph per cache size (14 sizes, compiled once during warmup).
    /// Each group i always has cache seqLen=i+2, so compiled graphs are reused across timesteps.
    ///
    /// Talker is compiled with shapeless=true — RoPE offset passed as regular MLXArray input,
    /// growing KV cache handled by shapeless mode, batch dim uses -1 reshapes.
    private var compiledCPTransformer: (([MLXArray]) -> [MLXArray])?

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
    ///   - speaker: Speaker voice name (requires CustomVoice model, e.g., "vivian", "ryan")
    ///   - sampling: Sampling configuration
    /// - Returns: Audio samples at 24kHz
    public func synthesize(
        text: String,
        language: String = "english",
        speaker: String? = nil,
        sampling: SamplingConfig = .default
    ) -> [Float] {
        guard let tokenizer = tokenizer else {
            fatalError("Tokenizer not loaded. Call setTokenizer() first.")
        }

        // Resolve speaker → token ID and optional language override
        let (speakerTokenId, effectiveLanguage) = resolveSpeaker(speaker, language: language)

        guard let langId = CodecTokens.languageId(for: effectiveLanguage) else {
            print("Warning: Unknown language '\(effectiveLanguage)', defaulting to English")
            return synthesize(text: text, language: "english", speaker: speaker, sampling: sampling)
        }

        let t0 = CFAbsoluteTimeGetCurrent()

        // Stage 1: Prepare text tokens and codec prefix
        let textTokens = prepareTextTokens(text: text, tokenizer: tokenizer)
        let codecPrefixTokens = buildCodecPrefix(languageId: langId, speakerTokenId: speakerTokenId)

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

    // MARK: - Streaming Synthesis

    /// Synthesize speech as a stream of audio chunks with low first-packet latency.
    ///
    /// The architecture is fully causal (Talker, Code Predictor, Mimi decoder all use causal
    /// attention/convolutions), so streaming produces the same quality as batch synthesis.
    ///
    /// - Parameters:
    ///   - text: Input text to synthesize
    ///   - language: Language tag (e.g., "english", "chinese")
    ///   - speaker: Speaker voice name (requires CustomVoice model)
    ///   - sampling: Sampling configuration
    ///   - streaming: Streaming configuration (chunk sizes, decoder context)
    /// - Returns: An async stream of `TTSAudioChunk` values
    public func synthesizeStream(
        text: String,
        language: String = "english",
        speaker: String? = nil,
        sampling: SamplingConfig = .default,
        streaming: StreamingConfig = .default
    ) -> AsyncThrowingStream<TTSAudioChunk, Error> {
        AsyncThrowingStream { continuation in
            Task {
                do {
                    try self.runStreamingGeneration(
                        text: text,
                        language: language,
                        speaker: speaker,
                        sampling: sampling,
                        streaming: streaming,
                        continuation: continuation)
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    /// Internal streaming generation loop. Same structure as `synthesize()` but emits audio
    /// chunks via the continuation as soon as enough frames are accumulated.
    private func runStreamingGeneration(
        text: String,
        language: String,
        speaker: String?,
        sampling: SamplingConfig,
        streaming: StreamingConfig,
        continuation: AsyncThrowingStream<TTSAudioChunk, Error>.Continuation
    ) throws {
        guard let tokenizer = tokenizer else {
            throw TTSError.tokenizerNotLoaded
        }

        let (speakerTokenId, effectiveLanguage) = resolveSpeaker(speaker, language: language)

        guard let langId = CodecTokens.languageId(for: effectiveLanguage) else {
            throw TTSError.unknownLanguage(effectiveLanguage)
        }

        let t0 = CFAbsoluteTimeGetCurrent()
        let safeMaxTokens = min(sampling.maxTokens, 500)
        let samplesPerFrame = 1920  // 24000 / 12.5

        // Stage 1: Prepare embeddings (identical to synthesize)
        let textTokens = prepareTextTokens(text: text, tokenizer: tokenizer)
        let codecPrefixTokens = buildCodecPrefix(languageId: langId, speakerTokenId: speakerTokenId)
        let (prefillEmbeds, trailingTextHidden, ttsPadEmbed) = buildPrefillEmbeddings(
            textTokens: textTokens, codecPrefixTokens: codecPrefixTokens)
        eval(prefillEmbeds, trailingTextHidden, ttsPadEmbed)

        // Stage 2: Autoregressive generation with chunked decode + emit
        let cpSamplingConfig = SamplingConfig(temperature: sampling.temperature, topK: sampling.topK)
        let prefillLen = prefillEmbeds.dim(1)

        // Prefill
        var (logits, hiddenStates, newCache) = talker(
            inputsEmbeds: prefillEmbeds,
            offset: MLXArray(Int32(0)),
            cache: nil)
        var talkerCache = newCache

        // Sample first token
        let lastLogits = logits[0..., (prefillLen - 1)..<prefillLen, 0...]
        var nextToken = sampleToken(
            logits: lastLogits,
            config: sampling,
            generatedTokens: [],
            suppressRange: (2048, 3072),
            eosTokenId: CodecTokens.codecEos)

        if nextToken == Int32(CodecTokens.codecEos) {
            let chunk = TTSAudioChunk(
                samples: [], sampleRate: 24000, frameIndex: 0,
                totalFrames: 0, isFinal: true,
                elapsedTime: CFAbsoluteTimeGetCurrent() - t0)
            continuation.yield(chunk)
            return
        }

        var generatedFirstCodebook: [Int32] = [nextToken]
        var generatedAllCodebooks: [[Int32]] = (0..<config.codePredictor.numCodeGroups).map { _ in [] }
        generatedAllCodebooks[0].append(nextToken)

        // Code predictor for first timestep
        let lastHidden = hiddenStates[0..., (prefillLen - 1)..<prefillLen, 0...]
        var codeTokens = predictCodebooksForTimestep(
            hiddenState: lastHidden,
            firstCodebookToken: nextToken,
            cpSamplingConfig: cpSamplingConfig)
        for (i, token) in codeTokens.enumerated() {
            generatedAllCodebooks[i + 1].append(token)
        }

        var trailingIdx = 0
        var step = prefillLen
        var emittedFrames = 0

        var nextEmitThreshold = streaming.firstChunkFrames

        // Emit immediately if prefill already produced enough frames (e.g., firstChunkFrames=1)
        if generatedFirstCodebook.count >= nextEmitThreshold {
            let chunk = decodeAndEmitChunk(
                allCodebooks: generatedAllCodebooks,
                chunkStart: 0,
                chunkEnd: generatedFirstCodebook.count,
                decoderLeftContext: streaming.decoderLeftContext,
                samplesPerFrame: samplesPerFrame)
            let audioChunk = TTSAudioChunk(
                samples: chunk,
                sampleRate: 24000,
                frameIndex: 0,
                totalFrames: generatedFirstCodebook.count,
                isFinal: false,
                elapsedTime: CFAbsoluteTimeGetCurrent() - t0)
            continuation.yield(audioChunk)
            emittedFrames = generatedFirstCodebook.count
            nextEmitThreshold = emittedFrames + streaming.chunkFrames
        }

        // Autoregressive generation loop
        for iterIdx in 1..<safeMaxTokens {
            // Text side
            let textEmbed: MLXArray
            let trailingLen = trailingTextHidden.dim(1)
            if trailingIdx < trailingLen {
                textEmbed = trailingTextHidden[0..., trailingIdx..<(trailingIdx + 1), 0...]
                trailingIdx += 1
            } else {
                textEmbed = ttsPadEmbed
            }

            // Codec side
            let codecEmbed = talker.embedCodec(
                MLXArray([nextToken]).expandedDimensions(axis: 0))
                + codePredictor.batchEmbedAllGroups(codeTokens)

            let stepEmbeds = textEmbed + codecEmbed

            (logits, hiddenStates, newCache) = executeTalkerStep(
                embeds: stepEmbeds, offset: step, cache: talkerCache)
            talkerCache = newCache

            nextToken = sampleToken(
                logits: logits,
                config: sampling,
                generatedTokens: generatedFirstCodebook,
                suppressRange: (2048, 3072),
                eosTokenId: CodecTokens.codecEos)

            let isEos = nextToken == Int32(CodecTokens.codecEos)

            if !isEos {
                generatedFirstCodebook.append(nextToken)
                generatedAllCodebooks[0].append(nextToken)

                let stepHidden = hiddenStates
                codeTokens = predictCodebooksForTimestep(
                    hiddenState: stepHidden,
                    firstCodebookToken: nextToken,
                    cpSamplingConfig: cpSamplingConfig)
                for (i, token) in codeTokens.enumerated() {
                    generatedAllCodebooks[i + 1].append(token)
                }
            }

            step += 1
            let totalFrames = generatedFirstCodebook.count

            // Emit chunk when we have enough frames or on EOS/max
            let shouldEmit = isEos || totalFrames >= nextEmitThreshold || iterIdx == safeMaxTokens - 1
            if shouldEmit && totalFrames > emittedFrames {
                let chunkFrameStart = emittedFrames
                let chunkFrameEnd = totalFrames

                let chunk = decodeAndEmitChunk(
                    allCodebooks: generatedAllCodebooks,
                    chunkStart: chunkFrameStart,
                    chunkEnd: chunkFrameEnd,
                    decoderLeftContext: streaming.decoderLeftContext,
                    samplesPerFrame: samplesPerFrame)

                let audioChunk = TTSAudioChunk(
                    samples: chunk,
                    sampleRate: 24000,
                    frameIndex: chunkFrameStart,
                    totalFrames: chunkFrameEnd,
                    isFinal: isEos || iterIdx == safeMaxTokens - 1,
                    elapsedTime: CFAbsoluteTimeGetCurrent() - t0)
                continuation.yield(audioChunk)

                emittedFrames = chunkFrameEnd
                // After first emit, use regular chunk size
                nextEmitThreshold = emittedFrames + streaming.chunkFrames
            }

            if isEos { break }

            if iterIdx % 50 == 0 {
                let estSec = Double(generatedFirstCodebook.count) / 12.5
                print("  Streaming: \(generatedFirstCodebook.count) tokens (~\(String(format: "%.1f", estSec))s audio)...")
            }
        }

        let numFrames = generatedFirstCodebook.count
        if numFrames >= safeMaxTokens && nextToken != Int32(CodecTokens.codecEos) {
            let estSec = Double(numFrames) / 12.5
            print("Warning: Hit safety limit of \(safeMaxTokens) tokens (~\(String(format: "%.1f", estSec))s audio).")
        }

        // If we never emitted the final chunk (e.g. exactly at threshold), emit now
        if emittedFrames < numFrames {
            let chunk = decodeAndEmitChunk(
                allCodebooks: generatedAllCodebooks,
                chunkStart: emittedFrames,
                chunkEnd: numFrames,
                decoderLeftContext: streaming.decoderLeftContext,
                samplesPerFrame: samplesPerFrame)
            let audioChunk = TTSAudioChunk(
                samples: chunk,
                sampleRate: 24000,
                frameIndex: emittedFrames,
                totalFrames: numFrames,
                isFinal: true,
                elapsedTime: CFAbsoluteTimeGetCurrent() - t0)
            continuation.yield(audioChunk)
        }
    }

    /// Decode a chunk of codec frames to audio samples, using left context for decoder quality.
    ///
    /// Builds `[1, 16, contextFrames + chunkFrames]` from accumulated codebooks, runs the codec
    /// decoder, trims left-context and zero-pad samples, and returns Float PCM.
    ///
    /// The codec decoder (ConvNeXt kernel=7 after 2x pre-upsample) requires >= 4 input frames.
    /// When fewer real frames are available (e.g., 1-frame first chunk with no context), zeros
    /// are prepended as left padding. The decoder is fully causal (left-padded convolutions),
    /// so zero-padding produces silence that doesn't affect the real frames' output.
    private func decodeAndEmitChunk(
        allCodebooks: [[Int32]],
        chunkStart: Int,
        chunkEnd: Int,
        decoderLeftContext: Int,
        samplesPerFrame: Int
    ) -> [Float] {
        let contextStart = max(chunkStart - decoderLeftContext, 0)
        let actualContext = chunkStart - contextStart

        // Build [1, 16, contextFrames + chunkFrames] from accumulated codebooks
        let numGroups = allCodebooks.count
        let frameRange = contextStart..<chunkEnd
        var codebookArrays: [MLXArray] = []
        for g in 0..<numGroups {
            let slice = Array(allCodebooks[g][frameRange])
            codebookArrays.append(MLXArray(slice).expandedDimensions(axis: 0))  // [1, T]
        }
        var codes = stacked(codebookArrays, axis: 1)  // [1, 16, T]

        // Zero-pad if fewer than 4 frames (codec decoder minimum for ConvNeXt kernel=7)
        let minDecodeFrames = 4
        let realFrames = codes.dim(2)
        let zeroPadFrames = max(minDecodeFrames - realFrames, 0)
        if zeroPadFrames > 0 {
            let pad = MLXArray.zeros([1, numGroups, zeroPadFrames]).asType(.int32)
            codes = concatenated([pad, codes], axis: 2)  // prepend zeros on left
        }

        // Decode through codec (uses compiled path when available)
        let waveform = codecDecoder.executeDecoder(codes)  // [1, T_samples, 1]

        // Keep the last `realChunkFrames * samplesPerFrame` samples from the decoder output.
        // The decoder has a ~2880-sample startup overhead (causal conv warmup), so trimming
        // from the left by `(zeroPad + context) * samplesPerFrame` can overshoot. Instead,
        // compute the desired output size and trim from the right side of the waveform.
        let realChunkFrames = chunkEnd - chunkStart
        let expectedKept = realChunkFrames * samplesPerFrame
        let totalSamples = waveform.dim(1)
        let trimSamples = max(0, totalSamples - expectedKept)
        let kept = waveform[0..., trimSamples..<totalSamples, 0...]

        let flat = kept.squeezed()
        eval(flat)
        return flat.asArray(Float.self)
    }

    // MARK: - Batch Synthesis

    /// Synthesize speech from multiple texts in parallel using batched generation.
    ///
    /// All items generate tokens in lockstep. Items that finish early (hit EOS) receive
    /// padding tokens. Generation stops when all items are done or the safety cap is reached.
    ///
    /// Texts are sorted by length before batching so similar-length items are grouped together,
    /// minimizing wasted compute from padding. Results are returned in the original input order.
    ///
    /// **Memory:** Each item uses ~55 MB KV cache per 500 tokens. B=4 at 500 tokens ≈ 220 MB.
    ///
    /// **Limitations:**
    /// - Repetition penalty is not applied in batch mode (requires per-item token history).
    /// - Items with very different output lengths waste compute on padding steps.
    ///   If one item fails to hit EOS, all items in the batch run to the safety cap.
    ///
    /// - Parameters:
    ///   - texts: Array of texts to synthesize
    ///   - language: Language tag (e.g., "english", "chinese")
    ///   - sampling: Sampling configuration
    ///   - maxBatchSize: Maximum items per batch (default 4)
    /// - Returns: Array of audio samples at 24kHz, one per input text (same order as input)
    public func synthesizeBatch(
        texts: [String],
        language: String = "english",
        sampling: SamplingConfig = .default,
        maxBatchSize: Int = 4
    ) -> [[Float]] {
        guard !texts.isEmpty else { return [] }

        // Single item: delegate to existing method for zero overhead
        if texts.count == 1 {
            return [synthesize(text: texts[0], language: language, sampling: sampling)]
        }

        guard let tokenizer = tokenizer else {
            fatalError("Tokenizer not loaded. Call setTokenizer() first.")
        }

        guard let langId = CodecTokens.languageId(for: language) else {
            print("Warning: Unknown language '\(language)', defaulting to English")
            return synthesizeBatch(texts: texts, language: "english", sampling: sampling, maxBatchSize: maxBatchSize)
        }

        // Sort texts by length to group similar-length items together.
        // This minimizes padding waste: if one batch has all short texts and another
        // has all long texts, no short text is forced to wait for a long one.
        let indexed = texts.enumerated().map { ($0.offset, $0.element) }
        let sorted = indexed.sorted { $0.1.count < $1.1.count }
        let sortedTexts = sorted.map { $0.1 }
        let originalIndices = sorted.map { $0.0 }

        // Process in chunks if exceeding maxBatchSize
        var sortedResults: [[Float]]
        if sortedTexts.count > maxBatchSize {
            sortedResults = []
            for chunkStart in stride(from: 0, to: sortedTexts.count, by: maxBatchSize) {
                let chunkEnd = min(chunkStart + maxBatchSize, sortedTexts.count)
                let chunk = Array(sortedTexts[chunkStart..<chunkEnd])
                let chunkResults = synthesizeBatchInternal(texts: chunk, langId: langId, tokenizer: tokenizer, sampling: sampling)
                sortedResults.append(contentsOf: chunkResults)
            }
        } else {
            sortedResults = synthesizeBatchInternal(texts: sortedTexts, langId: langId, tokenizer: tokenizer, sampling: sampling)
        }

        // Restore original order
        var results = [[Float]](repeating: [], count: texts.count)
        for (sortedIdx, origIdx) in originalIndices.enumerated() {
            results[origIdx] = sortedResults[sortedIdx]
        }
        return results
    }

    /// Internal batch synthesis for a single chunk (already sorted, within maxBatchSize).
    private func synthesizeBatchInternal(
        texts: [String],
        langId: Int,
        tokenizer: Qwen3Tokenizer,
        sampling: SamplingConfig
    ) -> [[Float]] {
        let t0 = CFAbsoluteTimeGetCurrent()
        let batchSize = texts.count

        // Stage 1: Prepare per-item data
        let codecPrefixTokens = buildCodecPrefix(languageId: langId)

        var prefills: [MLXArray] = []
        var trailings: [MLXArray] = []
        var padEmbeds: [MLXArray] = []

        for text in texts {
            let textTokens = prepareTextTokens(text: text, tokenizer: tokenizer)
            let (prefill, trailing, padEmbed) = buildPrefillEmbeddings(
                textTokens: textTokens, codecPrefixTokens: codecPrefixTokens)
            prefills.append(prefill)
            trailings.append(trailing)
            padEmbeds.append(padEmbed)
        }

        // All prefills are length 9 — stack directly
        let batchPrefill = concatenated(prefills, axis: 0)  // [B, 9, D]
        let ttsPadEmbed = padEmbeds[0]  // All pad embeds are the same — use first

        // Pre-pad trailing texts to max length
        let maxTrailingLen = trailings.map { $0.dim(1) }.max()!
        var paddedTrailings: [MLXArray] = []
        for trailing in trailings {
            let trailLen = trailing.dim(1)
            if trailLen < maxTrailingLen {
                let padCount = maxTrailingLen - trailLen
                let padding = broadcast(ttsPadEmbed, to: [1, padCount, config.talker.hiddenSize])
                paddedTrailings.append(concatenated([trailing, padding], axis: 1))
            } else {
                paddedTrailings.append(trailing)
            }
        }
        let batchTrailing = concatenated(paddedTrailings, axis: 0)  // [B, maxTrailingLen, D]

        eval(batchPrefill, batchTrailing)
        let t1 = CFAbsoluteTimeGetCurrent()

        // Stage 2: Batch generation
        let (allCodebooksList, frameCounts) = generateBatchWithCodePredictor(
            batchPrefill: batchPrefill,
            batchTrailing: batchTrailing,
            ttsPadEmbed: ttsPadEmbed,
            sampling: sampling)

        let t2 = CFAbsoluteTimeGetCurrent()

        // Log padding waste: how many steps each item wasted after hitting EOS
        let maxFrames = frameCounts.max() ?? 0
        if maxFrames > 0 {
            let wastedSteps = frameCounts.map { maxFrames - $0 }
            let totalWaste = wastedSteps.reduce(0, +)
            let wasteRatio = Double(totalWaste) / Double(maxFrames * batchSize)
            if wasteRatio > 0.3 {
                print("  Warning: \(Int(wasteRatio * 100))% padding waste " +
                      "(items finished at: \(frameCounts.map { String($0) }.joined(separator: ", ")) steps). " +
                      "Batch similar-length texts for better efficiency.")
            }
        }

        // Stage 3: Decode each item
        var results: [[Float]] = []
        for i in 0..<batchSize {
            let numFrames = frameCounts[i]
            if numFrames == 0 {
                print("  Item \(i): no tokens generated")
                results.append([])
                continue
            }
            let codes = allCodebooksList[i]  // [1, 16, Ti]
            let outputSamples = numFrames * 1920
            print("  Item \(i): decoding \(numFrames) frames -> \(outputSamples) samples (\(String(format: "%.1f", Double(outputSamples) / 24000.0))s)...")
            let waveform = codecDecoder.decode(codes: codes)
            results.append(waveform)
        }
        let t3 = CFAbsoluteTimeGetCurrent()

        let totalAudio = results.reduce(0.0) { $0 + Double($1.count) / 24000.0 }
        let totalFrames = frameCounts.reduce(0, +)
        print("  Batch timing: embed=\(String(format: "%.3f", t1-t0))s | " +
              "generate=\(String(format: "%.3f", t2-t1))s (\(totalFrames) total steps, " +
              "\(batchSize) items) | " +
              "decode=\(String(format: "%.3f", t3-t2))s | " +
              "total=\(String(format: "%.3f", t3-t0))s | " +
              "audio=\(String(format: "%.2f", totalAudio))s | " +
              "RTF=\(String(format: "%.2f", (t3-t0)/max(totalAudio, 0.001)))")

        return results
    }

    // MARK: - Batch Generation Loop

    /// Batch autoregressive generation: all B items in lockstep.
    private func generateBatchWithCodePredictor(
        batchPrefill: MLXArray,
        batchTrailing: MLXArray,
        ttsPadEmbed: MLXArray,
        sampling: SamplingConfig
    ) -> (allCodebooksList: [MLXArray], frameCounts: [Int]) {
        let batchSize = batchPrefill.dim(0)
        let safeMaxTokens = min(sampling.maxTokens, 500)
        let maxTrailingLen = batchTrailing.dim(1)
        let cpSamplingConfig = SamplingConfig(temperature: sampling.temperature, topK: sampling.topK)
        let codecPadToken = Int32(CodecTokens.codecPad)

        // Prefill
        let prefillLen = batchPrefill.dim(1)

        var (logits, hiddenStates, talkerCache) = talker(
            inputsEmbeds: batchPrefill,
            offset: MLXArray(Int32(0)),
            cache: nil)

        // Sample first token for each item
        let firstLogits = logits[0..., (prefillLen - 1)..<prefillLen, 0...]  // [B, 1, vocab]
        var finished = MLXArray(Array(repeating: false, count: batchSize))  // [B]

        var nextTokens = sampleTokensBatch(
            logits: firstLogits,
            config: sampling,
            finishedMask: finished,
            padToken: codecPadToken,
            suppressRange: (2048, 3072),
            eosTokenId: CodecTokens.codecEos)

        // Check which items hit EOS immediately
        let eosCheck = nextTokens .== MLXArray(Int32(CodecTokens.codecEos))
        finished = logicalOr(finished, eosCheck)

        // Get hidden states for code predictor
        var lastHidden = hiddenStates[0..., (prefillLen - 1)..<prefillLen, 0...]  // [B, 1, D]

        // Predict remaining 15 codebooks for first timestep
        var cpTokens = predictCodebooksForTimestepBatch(
            hiddenStates: lastHidden,
            firstCodebookTokens: nextTokens,
            cpSamplingConfig: cpSamplingConfig)  // [B, 15]

        // Accumulate codebooks: list of [B, 16] per timestep
        var allCBSteps: [MLXArray] = []
        let firstStep = concatenated([nextTokens.expandedDimensions(axis: 1), cpTokens], axis: 1)  // [B, 16]
        allCBSteps.append(firstStep)

        var trailingIdx = 0
        var step = prefillLen

        // Autoregressive generation
        for iterIdx in 1..<safeMaxTokens {
            // Text side: next trailing text embed or pad (same index for all items since pre-padded)
            let textEmbed: MLXArray
            if trailingIdx < maxTrailingLen {
                textEmbed = batchTrailing[0..., trailingIdx..<(trailingIdx + 1), 0...]  // [B, 1, D]
                trailingIdx += 1
            } else {
                // Broadcast pad embed to [B, 1, D]
                textEmbed = broadcast(ttsPadEmbed, to: [batchSize, 1, config.talker.hiddenSize])
            }

            // Codec side: embed first codebook + sum of 15 predicted codebooks
            let codecEmbed = talker.embedCodec(
                nextTokens.expandedDimensions(axis: 1))  // [B, 1] → [B, 1, D]
                + codePredictor.batchEmbedAllGroupsBatch(cpTokens)  // [B, 1, D]

            let stepEmbeds = textEmbed + codecEmbed  // [B, 1, D]

            let newResult = executeTalkerStep(
                embeds: stepEmbeds, offset: step, cache: talkerCache)
            logits = newResult.0
            hiddenStates = newResult.1
            talkerCache = newResult.2

            nextTokens = sampleTokensBatch(
                logits: logits,
                config: sampling,
                finishedMask: finished,
                padToken: codecPadToken,
                suppressRange: (2048, 3072),
                eosTokenId: CodecTokens.codecEos)

            // Update finished mask
            let newEos = nextTokens .== MLXArray(Int32(CodecTokens.codecEos))
            finished = logicalOr(finished, newEos)

            // Code predictor for this timestep
            lastHidden = hiddenStates  // [B, 1, D]
            cpTokens = predictCodebooksForTimestepBatch(
                hiddenStates: lastHidden,
                firstCodebookTokens: nextTokens,
                cpSamplingConfig: cpSamplingConfig)

            let stepCB = concatenated([nextTokens.expandedDimensions(axis: 1), cpTokens], axis: 1)
            allCBSteps.append(stepCB)

            step += 1

            // Check if all items are done
            eval(finished)
            let finishedArray = finished.asArray(Bool.self)
            if finishedArray.allSatisfy({ $0 }) { break }

            if iterIdx % 50 == 0 {
                let estSec = Double(iterIdx) / 12.5
                let doneCount = finishedArray.filter { $0 }.count
                print("  Batch: \(iterIdx) steps (~\(String(format: "%.1f", estSec))s), \(doneCount)/\(batchSize) done...")
            }
        }

        let totalSteps = allCBSteps.count
        print("  Batch generation done: \(totalSteps) steps, \(batchSize) items")

        // Stack all timesteps: [B, 16, T]
        let stepsStacked = stacked(allCBSteps, axis: 0)  // [T, B, 16]
        let allCB = stepsStacked.transposed(1, 2, 0)  // [B, 16, T]
        eval(allCB)

        // Extract per-item codebooks, trimming at EOS
        var results: [MLXArray] = []
        var frameCounts: [Int] = []

        for i in 0..<batchSize {
            let itemCB = allCB[i..<(i + 1)]  // [1, 16, T]
            let firstCBRow = itemCB[0..., 0, 0...]  // [1, T] — first codebook
            eval(firstCBRow)
            let tokens = firstCBRow.squeezed().asArray(Int32.self)

            // Find EOS position
            var eosPos = tokens.count
            for (j, tok) in tokens.enumerated() {
                if tok == Int32(CodecTokens.codecEos) {
                    eosPos = j
                    break
                }
            }

            if eosPos == 0 {
                results.append(MLXArray.zeros([1, 16, 0]))
                frameCounts.append(0)
            } else {
                let trimmed = itemCB[0..., 0..., 0..<eosPos]  // [1, 16, eosPos]
                results.append(trimmed)
                frameCounts.append(eosPos)
            }
        }

        return (results, frameCounts)
    }

    /// Predict 15 remaining codebook tokens for B items at a single timestep.
    private func predictCodebooksForTimestepBatch(
        hiddenStates: MLXArray,
        firstCodebookTokens: MLXArray,
        cpSamplingConfig: SamplingConfig
    ) -> MLXArray {
        let batchSize = hiddenStates.dim(0)
        let numGroups = config.codePredictor.numCodeGroups - 1  // 15

        var cpCache: [(MLXArray, MLXArray)]? = nil

        // First codebook embedding (from talker's codec embedding)
        let code0Embed = talker.embedCodec(
            firstCodebookTokens.expandedDimensions(axis: 1))  // [B, 1, D]

        // Prefill: [hidden_state, code_0_embed] — length 2
        let prefillInput = concatenated([hiddenStates, code0Embed], axis: 1)  // [B, 2, D]

        // Predict codebook group 0
        var (cpLogits, cpNewCache) = codePredictor(
            inputsEmbeds: prefillInput, groupIndex: 0, cache: nil)
        cpCache = cpNewCache

        let lastCpLogits = cpLogits[0..., 1..<2, 0...]  // [B, 1, vocab]

        // No EOS/suppress needed for code predictor
        let noFinished = MLXArray(Array(repeating: false, count: batchSize))
        var prevTokens = sampleTokensBatch(
            logits: lastCpLogits,
            config: cpSamplingConfig,
            finishedMask: noFinished,
            padToken: 0)

        var groupTokens: [MLXArray] = [prevTokens]  // list of [B]

        // Remaining 14 codebook groups (compiled transformer + separate lm_head)
        for groupIdx in 1..<numGroups {
            let prevEmbed = codePredictor.embedCodecGroup(
                prevTokens.expandedDimensions(axis: 1),
                groupIndex: groupIdx - 1)  // [B, 1, D]

            let cpResult = executeCPTransformerStep(hidden: prevEmbed, cache: cpCache!)
            cpCache = cpResult.newCache
            cpLogits = codePredictor.lmHeads[groupIdx](cpResult.normed)

            prevTokens = sampleTokensBatch(
                logits: cpLogits,
                config: cpSamplingConfig,
                finishedMask: noFinished,
                padToken: 0)
            groupTokens.append(prevTokens)
        }

        // Stack: 15 × [B] → [B, 15]
        return stacked(groupTokens, axis: 1)
    }
    // MARK: - Warm-up

    /// Run minimal dummy forward passes to compile Metal shaders and allocate GPU buffers.
    /// This eliminates first-inference latency from shader compilation.
    public func warmUp() {
        guard let tokenizer = tokenizer else { return }

        // Set up compiled code predictor for kernel fusion
        setupCompilation()

        // Run a minimal prefill through the talker to compile all Metal shaders.
        let textTokens = prepareTextTokens(text: "hi", tokenizer: tokenizer)
        let codecPrefix = buildCodecPrefix(languageId: CodecTokens.languageEnglish)
        let (prefillEmbeds, trailingTextHidden, ttsPadEmbed) = buildPrefillEmbeddings(
            textTokens: textTokens, codecPrefixTokens: codecPrefix)
        eval(prefillEmbeds, trailingTextHidden, ttsPadEmbed)

        // Talker prefill: compiles all 28-layer attention + MLP shaders
        let prefillLen = prefillEmbeds.dim(1)
        let (logits, hiddenStates, talkerWarmupCache) = talker(
            inputsEmbeds: prefillEmbeds, offset: MLXArray(Int32(0)), cache: nil)
        eval(logits)

        // Pre-compile talker generation step (shapeless=true, traced once here).
        // Uses the cache from prefill so the compiled graph includes cache concatenation.
        let warmupCodecEmbed = talker.embedCodec(MLXArray([Int32(0)]).expandedDimensions(axis: 0))
        let (warmupLogits, _, _) = executeTalkerStep(
            embeds: warmupCodecEmbed, offset: prefillLen, cache: talkerWarmupCache)
        eval(warmupLogits)

        // Parallel code predictor: compiles 5-layer shaders + all 15 lm_heads
        let lastHidden = hiddenStates[0..., (prefillLen - 1)..<prefillLen, 0...]
        let code0Embed = talker.embedCodec(MLXArray([Int32(0)]).expandedDimensions(axis: 0))
        let cpInput = concatenated([lastHidden, code0Embed], axis: 1)
        let allLogits = codePredictor.predictAllGroupsParallel(inputsEmbeds: cpInput)
        eval(allLogits)

        // Pre-compile CP transformer for all 14 cache sizes (groups 1-14).
        // Each group i has cache seqLen = i+2, which is constant across timesteps.
        // This traces + compiles 14 graphs during warmup so generation pays zero compile cost.
        let (_, cpPrefillCache) = codePredictor(
            inputsEmbeds: cpInput, groupIndex: 0, cache: nil)
        var cpCache: [(MLXArray, MLXArray)] = cpPrefillCache
        let numGroups = config.codePredictor.numCodeGroups - 1  // 15
        for groupIdx in 1..<numGroups {
            let (cpNormed, newCPCache) = executeCPTransformerStep(
                hidden: code0Embed, cache: cpCache)
            cpCache = newCPCache
            let groupLogits = codePredictor.lmHeads[groupIdx](cpNormed)
            eval(groupLogits)
        }

        // Compile codec decoder for kernel fusion (different from shader JIT compilation).
        // compile() fuses multiple kernel dispatches into fewer optimized kernels per chunk.
        // Warmup adds ~300ms to load time but saves on every generation.
        codecDecoder.setupCompilation()
        codecDecoder.warmUp()
    }

    // MARK: - Compiled Generation Steps

    /// Initialize compiled talker and code predictor for Metal kernel fusion.
    ///
    /// MLX.compile() traces the computation graph on first call and replays it
    /// on subsequent calls, fusing small kernel calls into larger ones.
    ///
    /// Talker: compiled with shapeless=true. RoPE offset is passed as a regular MLXArray
    /// input (compile treats inputs as variables, not constants). Growing KV cache is
    /// handled by shapeless mode. Batch dimension uses -1 reshapes for any batch size.
    ///
    /// Code predictor: compiled with shapeless=false. 14 fixed cache sizes, compiled
    /// once during warmup, reused for all subsequent timesteps.
    func setupCompilation() {
        // Compiled talker: [embeds, offset, K0, V0, ..., K27, V27] →
        //                  [logits, hidden, K0, V0, ..., K27, V27]
        // Offset is a regular function input (compile treats inputs as variables, not constants).
        let talkerRef = talker
        let numTalkerLayers = config.talker.numLayers

        compiledTalkerStep = compile(
            inputs: [talkerRef], outputs: [talkerRef], shapeless: true
        ) { inputs in
            let embeds = inputs[0]
            let offset = inputs[1]  // MLXArray scalar — dynamic, not baked
            var cache: [(MLXArray, MLXArray)] = []
            for i in 0..<numTalkerLayers {
                cache.append((inputs[2 + i * 2], inputs[3 + i * 2]))
            }

            let (logits, hidden, newCache) = talkerRef(
                inputsEmbeds: embeds, offset: offset, cache: cache)

            var result = [logits, hidden]
            for (k, v) in newCache { result.append(k); result.append(v) }
            return result
        }
        let numCPLayers = config.codePredictor.numLayers
        let cpRef = codePredictor

        // Compiled CP transformer: hidden + 5×(K,V) cache → normed + 5×(K,V) cache
        // Does NOT include lm_head (applied separately per group index)
        compiledCPTransformer = compile(
            inputs: [cpRef], outputs: [cpRef], shapeless: false
        ) { inputs in
            var hidden = inputs[0]
            var cpCache: [(MLXArray, MLXArray)] = []
            for i in 0..<numCPLayers {
                cpCache.append((inputs[1 + i * 2], inputs[2 + i * 2]))
            }
            var newCache: [(MLXArray, MLXArray)] = []
            for (i, layer) in cpRef.layers.enumerated() {
                let (output, updated) = layer(hidden, attentionMask: nil, cache: cpCache[i])
                hidden = output
                newCache.append(updated)
            }
            hidden = cpRef.norm(hidden)
            var result = [hidden]
            for (k, v) in newCache { result.append(k); result.append(v) }
            return result
        }
    }

    /// Execute a talker generation step (compiled when available).
    ///
    /// The compiled path fuses ~420 Metal kernel dispatches (28 layers × ~15 ops) into
    /// fewer optimized kernels. Uses shapeless=true to handle growing KV cache without
    /// recompilation. RoPE offset is passed as a regular MLXArray input (not baked).
    private func executeTalkerStep(
        embeds: MLXArray, offset: Int, cache: [(MLXArray, MLXArray)]
    ) -> (MLXArray, MLXArray, [(MLXArray, MLXArray)]) {
        guard let compiled = compiledTalkerStep else {
            return talker(inputsEmbeds: embeds, offset: MLXArray(Int32(offset)), cache: cache)
        }

        // Flatten inputs: [embeds, offset, K0, V0, K1, V1, ..., K27, V27]
        let offsetArray = MLXArray(Int32(offset))
        var flatInputs = [embeds, offsetArray]
        for (k, v) in cache { flatInputs.append(k); flatInputs.append(v) }

        let out = compiled(flatInputs)

        // Unflatten: [logits, hidden, K0, V0, ..., K27, V27]
        var newCache: [(MLXArray, MLXArray)] = []
        for i in 0..<config.talker.numLayers {
            newCache.append((out[2 + i * 2], out[3 + i * 2]))
        }
        return (out[0], out[1], newCache)
    }

    /// Execute a code predictor transformer step (layers + norm, no lm_head).
    /// For single-token steps (seqLen=1) where no attention mask is needed.
    /// Apply codePredictor.lmHeads[groupIndex] to the normed output separately.
    private func executeCPTransformerStep(
        hidden: MLXArray, cache: [(MLXArray, MLXArray)]
    ) -> (normed: MLXArray, newCache: [(MLXArray, MLXArray)]) {
        guard let compiled = compiledCPTransformer else {
            var h = hidden
            var newCache: [(MLXArray, MLXArray)] = []
            for (i, layer) in codePredictor.layers.enumerated() {
                let (output, updated) = layer(h, attentionMask: nil, cache: cache[i])
                h = output
                newCache.append(updated)
            }
            return (codePredictor.norm(h), newCache)
        }
        var flatInputs = [hidden]
        for (k, v) in cache { flatInputs.append(k); flatInputs.append(v) }
        let out = compiled(flatInputs)
        var newCache: [(MLXArray, MLXArray)] = []
        for i in 0..<config.codePredictor.numLayers {
            newCache.append((out[1 + i * 2], out[2 + i * 2]))
        }
        return (out[0], newCache)
    }

    // MARK: - Speaker Resolution

    /// Resolve speaker name to token ID and effective language.
    /// - Returns: (speakerTokenId, effectiveLanguage) — speakerTokenId is nil if no speaker
    private func resolveSpeaker(_ speaker: String?, language: String) -> (Int?, String) {
        guard let speakerName = speaker else {
            return (nil, language)
        }

        guard let config = speakerConfig else {
            print("Warning: Speaker '\(speakerName)' requested but model has no speaker support. " +
                  "Use the CustomVoice model variant for speaker selection.")
            return (nil, language)
        }

        let normalizedName = speakerName.lowercased()
        guard let tokenId = config.speakerIds[normalizedName] else {
            let available = config.availableSpeakers.joined(separator: ", ")
            print("Warning: Unknown speaker '\(speakerName)'. Available speakers: \(available)")
            return (nil, language)
        }

        // Check if this speaker has a dialect override
        var effectiveLanguage = language
        if let dialect = config.speakerDialects[normalizedName] {
            effectiveLanguage = dialect
        }

        return (tokenId, effectiveLanguage)
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

    /// Build codec prefix: [think, think_bos, lang_id, think_eos, pad, bos] (6 tokens)
    /// With speaker: [think, think_bos, lang_id, think_eos, pad, bos, spk_token] (7 tokens)
    func buildCodecPrefix(languageId: Int, speakerTokenId: Int? = nil) -> [Int32] {
        var prefix: [Int32] = [
            Int32(CodecTokens.codecThink),
            Int32(CodecTokens.codecThinkBos),
            Int32(languageId),
            Int32(CodecTokens.codecThinkEos),
            Int32(CodecTokens.codecPad),
            Int32(CodecTokens.codecBos),
        ]
        if let spkId = speakerTokenId {
            prefix.append(Int32(spkId))
        }
        return prefix
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

        // Prefill
        var (logits, hiddenStates, newCache) = talker(
            inputsEmbeds: prefillEmbeds,
            offset: MLXArray(Int32(0)),
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

            (logits, hiddenStates, newCache) = executeTalkerStep(
                embeds: stepEmbeds, offset: step, cache: talkerCache!)
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
    ///
    /// Uses lazy evaluation: all 15 groups are chained as a single MLX computation graph
    /// with zero GPU sync barriers. One `eval()` at the end materializes all tokens.
    /// This reduces per-step GPU syncs from 15 to 1.
    private func predictCodebooksForTimestep(
        hiddenState: MLXArray,
        firstCodebookToken: Int32,
        cpSamplingConfig: SamplingConfig
    ) -> [Int32] {
        var cpCache: [(MLXArray, MLXArray)]? = nil

        // First codebook embedding (from talker's codec embedding)
        let code0Embed = talker.embedCodec(
            MLXArray([firstCodebookToken]).expandedDimensions(axis: 0))  // [1, 1, D]

        // Prefill: [hidden_state, code_0_embed] — length 2
        let prefillInput = concatenated([hiddenState, code0Embed], axis: 1)  // [1, 2, D]

        // Predict codebook group 0 (= codebook 2 overall)
        let (cpLogits, cpNewCache) = codePredictor(
            inputsEmbeds: prefillInput, groupIndex: 0, cache: nil)
        cpCache = cpNewCache

        // Sample lazily — returns MLXArray scalar, NO .item() sync
        let lastCpLogits = cpLogits[0..., 1..<2, 0...]
        var prevTokenArray = sampleTokenLazy(logits: lastCpLogits, config: cpSamplingConfig)
        var lazyTokens: [MLXArray] = [prevTokenArray]

        // Remaining 14 codebook groups — fully lazy chain, no GPU syncs
        for groupIdx in 1..<(config.codePredictor.numCodeGroups - 1) {
            // Embed previous group's token (lazy MLXArray → embedding, no sync needed)
            let prevEmbed = codePredictor.embedCodecGroup(
                prevTokenArray.reshaped(1, 1),
                groupIndex: groupIdx - 1)  // [1, 1, D]

            let cpResult = executeCPTransformerStep(hidden: prevEmbed, cache: cpCache!)
            cpCache = cpResult.newCache
            let groupLogits = codePredictor.lmHeads[groupIdx](cpResult.normed)

            prevTokenArray = sampleTokenLazy(logits: groupLogits, config: cpSamplingConfig)
            lazyTokens.append(prevTokenArray)
        }

        // ONE eval to materialize the entire 15-group computation graph
        let tokenStack = stacked(lazyTokens)  // [15]
        eval(tokenStack)
        return tokenStack.asArray(Int32.self)  // bulk extraction, no per-token sync
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

        // Parse config.json for speaker config
        let configPath = mainCacheDir.appendingPathComponent("config.json")
        if FileManager.default.fileExists(atPath: configPath.path) {
            model.speakerConfig = try? parseSpeakerConfig(from: configPath)
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

        progressHandler?(0.95, "Warming up model...")
        model.warmUp()

        progressHandler?(1.0, "Ready")
        return model
    }

    /// Parse speaker configuration from model config.json
    private static func parseSpeakerConfig(from path: URL) throws -> SpeakerConfig? {
        let data = try Data(contentsOf: path)
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let talkerConfig = json["talker_config"] as? [String: Any] else {
            return nil
        }

        // Parse spk_id: {"serena": 3066, "vivian": 3065, ...}
        let spkIdRaw = talkerConfig["spk_id"] as? [String: Any] ?? [:]
        var speakerIds: [String: Int] = [:]
        for (name, value) in spkIdRaw {
            if let id = value as? Int {
                speakerIds[name.lowercased()] = id
            }
        }

        // Return nil if no speakers defined (Base model has empty spk_id)
        guard !speakerIds.isEmpty else { return nil }

        // Parse spk_is_dialect: {"eric": "sichuan_dialect", "dylan": "beijing_dialect"}
        let dialectRaw = talkerConfig["spk_is_dialect"] as? [String: String] ?? [:]
        var speakerDialects: [String: String] = [:]
        for (name, dialect) in dialectRaw {
            speakerDialects[name.lowercased()] = dialect.lowercased()
        }

        // Parse codec_language_id: {"english": 2050, "chinese": 2055, ...}
        let langIdRaw = talkerConfig["codec_language_id"] as? [String: Any] ?? [:]
        var codecLanguageIds: [String: Int] = [:]
        for (name, value) in langIdRaw {
            if let id = value as? Int {
                codecLanguageIds[name.lowercased()] = id
            }
        }

        return SpeakerConfig(
            speakerIds: speakerIds,
            speakerDialects: speakerDialects,
            codecLanguageIds: codecLanguageIds)
    }
}
