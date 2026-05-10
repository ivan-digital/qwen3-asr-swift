import Foundation
import MLX
import MLXNN
import PersonaPlex   // sampleTextWithPenalty, sampleTopKWithPenalty
import AudioCommon

// MARK: - Streaming config

public struct HibikiStreamingConfig: Sendable {
    /// Frames accumulated before the first chunk is emitted.
    public var firstChunkFrames: Int
    /// Frames per subsequent chunk.
    public var chunkFrames: Int

    public init(firstChunkFrames: Int = 25, chunkFrames: Int = 25) {
        self.firstChunkFrames = firstChunkFrames
        self.chunkFrames = chunkFrames
    }

    public static let `default` = HibikiStreamingConfig()
}

// MARK: - Driver

public extension HibikiTranslateModel {

    /// Translate source-language speech to English speech (offline).
    ///
    /// Hibiki is **synchronous 1:1**: each Mimi frame of input (80 ms) produces
    /// exactly one Mimi frame of output. So `output_duration ≈ input_duration`.
    ///
    /// - Parameters:
    ///   - sourceAudio: PCM samples at 24 kHz, mono.
    ///   - sourceLanguage: hint only — Hibiki Zero auto-detects.
    ///   - verbose: print per-phase timings.
    /// - Returns: tuple of (English audio samples at 24 kHz, generated text token IDs).
    func translate(
        sourceAudio: [Float],
        sourceLanguage: HibikiSourceLanguage = .fr,
        verbose: Bool = false
    ) -> (audio: [Float], textTokens: [Int32]) {
        let totalStart = CFAbsoluteTimeGetCurrent()
        let nQ = cfg.temporal.nQ                                     // 16
        let numStreams = cfg.numStreams                              // 33
        let delays = cfg.delays                                      // 33 entries
        let maxDelay = cfg.maxDelay                                  // 2

        // 1. Encode source audio with Mimi.
        let encStart = CFAbsoluteTimeGetCurrent()
        let audioMLX = MLXArray(sourceAudio).reshaped([1, 1, sourceAudio.count])
        let sourceCodes = mimi.encode(audioMLX)   // [1, 16, T_src]
        eval(sourceCodes)
        let tSrc = sourceCodes.shape[2]
        guard tSrc > 0 else { return ([], []) }
        if verbose {
            print("  Mimi encode: \(String(format: "%.2f", CFAbsoluteTimeGetCurrent() - encStart))s, frames: \(tSrc)")
        }

        // 2. Initialize 33-stream token cache.
        //
        // Critical: pre-fill with **initial tokens** (the LAST index in each
        // embedding table — `textCard` for text, `card` for audio), NOT -1.
        // Upstream Moshi's `_step` reads `state.initial[k]` for any position
        // within the codebook's delay warm-up window. Those initial tokens
        // have trained embeddings; using -1 (masked to zero) leaves the model
        // running effectively unconditional during warm-up and badly
        // misaligned for the rest of the sequence. (The previous bug here
        // produced English subwords unrelated to the source content.)
        let totalLen = tSrc + maxDelay + 2
        let textInit = Int32(cfg.temporal.textInitialTokenId)        // = textCard
        let audioInit = Int32(cfg.temporal.initialTokenId)           // = card
        var tokenCache: [[Int32]] = []
        tokenCache.reserveCapacity(numStreams)
        tokenCache.append([Int32](repeating: textInit, count: totalLen))
        for _ in 1..<numStreams {
            tokenCache.append([Int32](repeating: audioInit, count: totalLen))
        }

        // Pre-populate text stream with padding tokens for the source-covered
        // window (the model emits SPM padding while audio is streaming).
        for t in 0..<tSrc {
            tokenCache[0][t + delays[0]] = Int32(cfg.temporal.textPaddingId)
        }

        // Pre-populate **source** (FR Mimi-encoded) codebook streams.
        // Following Moshi/PersonaPlex convention, generated tokens occupy
        // streams 1..nQ ("agent half") and input audio occupies streams
        // 1+nQ..1+2nQ-1 ("user half"). For Hibiki Zero-3B that means source
        // (FR) lives in streams 17..32 — NOT 1..16. Each stream is shifted
        // by `delays[stream]`.
        let sourceCodesHost = sourceCodes.reshaped([nQ, tSrc]).asArray(Int32.self)
        for cb in 0..<nQ {
            let sourceStreamIdx = 1 + nQ + cb        // 17..32
            let delay = delays[sourceStreamIdx]
            for t in 0..<tSrc {
                let writePos = t + delay
                if writePos < totalLen {
                    tokenCache[sourceStreamIdx][writePos] = sourceCodesHost[cb * tSrc + t]
                }
            }
        }

        // 3. Generation loop.
        var allTextTokens: [Int32] = []
        var targetTokens: [[Int32]] = (0..<nQ).map { _ in [] }
        var perCodebookHistory: [[Int32]] = (0..<cfg.depformer.numSteps).map { _ in [] }

        let genStart = CFAbsoluteTimeGetCurrent()
        temporal.resetCache()

        // Hibiki runs exactly tSrc steps (synchronous 1:1).
        for step in 0..<tSrc {
            if Task.isCancelled { break }

            // Read previous-step input for all 33 streams. The cache was
            // pre-filled with init tokens (textCard for text, card for audio),
            // so out-of-range or warm-up reads return the trained initial-
            // token embedding — same semantics as upstream `state.initial`.
            let readIdx = step - 1
            let textTok = readIdx >= 0 ? tokenCache[0][readIdx] : textInit
            let textTokenArr = MLXArray([textTok]).reshaped([1, 1])

            var audioStreamTokens: [Int32] = []
            for stream in 1..<numStreams {
                let tok = readIdx >= 0 ? tokenCache[stream][readIdx] : audioInit
                audioStreamTokens.append(tok)
            }
            let audioTokens = MLXArray(audioStreamTokens)
                .reshaped([1, numStreams - 1, 1])

            let (hidden, textLogits) = temporal.forward(
                textTokens: textTokenArr,
                audioTokens: audioTokens,
                offset: step
            )

            // Sample text token (with optional repetition penalty).
            // Setting HIBIKI_GREEDY=1 forces argmax for both text and audio
            // (deterministic, useful for debugging quality).
            let greedy = ProcessInfo.processInfo.environment["HIBIKI_GREEDY"] != nil
            let textHistory = Array(allTextTokens.suffix(cfg.sampling.repetitionWindow))
            let textToken: MLXArray
            if greedy {
                textToken = argMax(textLogits.squeezed(axis: 1), axis: -1).asType(.int32)
            } else {
                textToken = sampleTextWithPenalty(
                    logits: textLogits.squeezed(axis: 1),
                    temperature: cfg.sampling.textTemp,
                    topK: cfg.sampling.textTopK,
                    pastTokens: textHistory,
                    penalty: cfg.sampling.textRepetitionPenalty
                )
            }
            eval(textToken)
            let textVal = textToken[0].item(Int32.self)
            tokenCache[0][step] = textVal
            allTextTokens.append(textVal)

            // Generate the 16 target codebooks via depformer.
            let targetCodes = depformer.generate(
                temporalHidden: hidden,
                textToken: textToken
            ) { logits, cbIdx in
                if greedy {
                    return argMax(logits, axis: -1).asType(.int32)
                }
                let history = Array(perCodebookHistory[cbIdx].suffix(cfg.sampling.repetitionWindow))
                return sampleTopKWithPenalty(
                    logits: logits,
                    temperature: cfg.sampling.audioTemp,
                    topK: cfg.sampling.audioTopK,
                    pastTokens: history,
                    penalty: cfg.sampling.audioRepetitionPenalty
                )
            }

            // Write **target** (EN, depformer-generated) codebooks into the
            // "agent half" of the stream layout: streams 1..nQ (1..16 for
            // Zero-3B). Following Moshi/PersonaPlex convention, target tokens
            // are written at position `step` directly (no per-stream delay
            // shift) — the upstream Python code writes generated tokens at
            // target_position = offset % CT for ALL streams. The delay only
            // applies to externally provided input (the source codes above).
            let codes = targetCodes[0]   // [16]
            for cb in 0..<nQ {
                let tok = codes[cb].item(Int32.self)
                targetTokens[cb].append(tok)
                perCodebookHistory[cb].append(tok)
                let targetStreamIdx = 1 + cb               // 1..16
                if step < totalLen {
                    tokenCache[targetStreamIdx][step] = tok
                }
            }
        }
        if verbose {
            let genTime = CFAbsoluteTimeGetCurrent() - genStart
            let msPerStep = tSrc > 0 ? genTime / Double(tSrc) * 1000 : 0
            print("  Generation: \(String(format: "%.2f", genTime))s, " +
                  "\(String(format: "%.1f", msPerStep))ms/step (\(tSrc) steps)")
        }

        // 4. Decode target codebooks via Mimi → English audio.
        //
        // CRITICAL: align codebooks by their per-stream delay before decoding.
        // Upstream Moshi `_step` returns codes via `cache.gather(index = (offset
        // - max_delay + delays[k]) % CT)` (lm.py line 778-780). For Hibiki Zero
        // with delays = [text=0, target_cb0=0, target_cb1..15=2], this means
        // the time-T audio frame is composed of:
        //   - cb0 from generation step T
        //   - cb1..15 from generation step T + 2
        // Without this un-shift, cb1..15 would be 2 frames stale relative to
        // cb0, garbling Mimi's reconstruction. PersonaPlex (max_delay=1) is
        // less affected by skipping the un-shift; Hibiki (max_delay=2) is more
        // sensitive.
        let decStart = CFAbsoluteTimeGetCurrent()
        let totalGenSteps = targetTokens[0].count
        let alignedFrames = totalGenSteps - maxDelay   // skip max_delay warm-up
        guard alignedFrames > 0 else { return ([], allTextTokens) }

        // Per-codebook delays for the target half (streams 1..nQ).
        // delays[1 + cb] gives the delay of target codebook cb.
        var flat: [Int32] = []
        flat.reserveCapacity(nQ * alignedFrames)
        for cb in 0..<nQ {
            let delay = delays[1 + cb]
            for t in 0..<alignedFrames {
                let srcStep = t + delay
                flat.append(targetTokens[cb][srcStep])
            }
        }
        let codesArr = MLXArray(flat).reshaped([1, nQ, alignedFrames])
        let decoded = mimi.decode(codesArr)   // [1, 1, samples]
        eval(decoded)

        let numSamples = decoded.shape[2]
        let samples = decoded.reshaped([numSamples]).asArray(Float.self)

        if verbose {
            let decTime = CFAbsoluteTimeGetCurrent() - decStart
            print("  Mimi decode: \(String(format: "%.2f", decTime))s")
            let total = CFAbsoluteTimeGetCurrent() - totalStart
            let dur = Double(numSamples) / Double(cfg.sampleRate)
            print("  Total: \(String(format: "%.2f", total))s, " +
                  "audio: \(String(format: "%.2f", dur))s, " +
                  "RTF: \(String(format: "%.2f", total / max(dur, 0.001)))")
        }
        _ = sourceLanguage   // unused — model auto-detects, kept for API completeness
        return (samples, allTextTokens)
    }

    /// Streaming counterpart that emits `AudioChunk`s as Mimi finishes decoding
    /// each accumulated batch of target frames.
    func translateStream(
        sourceAudio: [Float],
        sourceLanguage: HibikiSourceLanguage = .fr,
        streaming: HibikiStreamingConfig = .default,
        verbose: Bool = false
    ) -> AsyncThrowingStream<AudioChunk, Error> {
        let cfg = self.cfg
        let model = self
        return AsyncThrowingStream<AudioChunk, Error> { continuation in
            Task {
                do {
                    // For v1, run offline translate then emit a single chunk
                    // covering the full output. True streaming via decodeStep
                    // can be added in a follow-up. Note that Hibiki's
                    // synchronous 1:1 generation means we know the full output
                    // duration up front.
                    let (audio, text) = model.translate(
                        sourceAudio: sourceAudio,
                        sourceLanguage: sourceLanguage,
                        verbose: verbose
                    )
                    let chunk = AudioChunk(
                        samples: audio,
                        sampleRate: cfg.sampleRate,
                        frameIndex: 0,
                        isFinal: true,
                        textTokens: text
                    )
                    continuation.yield(chunk)
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }
}
