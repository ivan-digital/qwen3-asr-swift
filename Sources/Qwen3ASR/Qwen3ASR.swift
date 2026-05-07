import Foundation
import MLXCommon
import MLX
import MLXNN
import MLXFast
import AudioCommon

/// Optional decoder tunables for `Qwen3ASRModel.transcribe(audio:options:)`.
///
/// Defaults match the historical greedy behaviour of `transcribe(audio:)`
/// so existing callers see zero change. Tune these when greedy decoding
/// collapses onto a single token (typical on silence or ambiguous phonemes).
public struct Qwen3DecodingOptions: Sendable {
    /// Cap on decoder output per chunk.
    public var maxTokens: Int = 448

    /// Optional language hint ("en", "zh", …). `nil` = auto-detect.
    public var language: String?

    /// Context hint prepended to the decoder prompt.
    public var context: String?

    /// HuggingFace-style repetition penalty. Divides the logits of tokens
    /// already generated this chunk by this factor before `argMax`.
    /// `1.0` disables; `1.1`–`1.3` is the common tuning range.
    public var repetitionPenalty: Float = 1.0

    /// If > 0, masks any next-token whose emission would form a repeated
    /// n-gram of this size. `0` disables.
    public var noRepeatNgramSize: Int = 0

    /// `0` = greedy (argmax). `> 0` = sample with this temperature via
    /// Gumbel-max. Higher = more random.
    public var temperature: Float = 0.0

    public init(
        maxTokens: Int = 448,
        language: String? = nil,
        context: String? = nil,
        repetitionPenalty: Float = 1.0,
        noRepeatNgramSize: Int = 0,
        temperature: Float = 0.0
    ) {
        self.maxTokens = maxTokens
        self.language = language
        self.context = context
        self.repetitionPenalty = repetitionPenalty
        self.noRepeatNgramSize = noRepeatNgramSize
        self.temperature = temperature
    }
}

/// Special token IDs for Qwen3-ASR
public struct Qwen3ASRTokens: Sendable {
    public static let audioTokenId = 151676        // <|audio_pad|>
    public static let audioStartTokenId = 151669   // <|audio_start|>
    public static let audioEndTokenId = 151670     // <|audio_end|>
    public static let eosTokenId = 151645          // <|im_end|>
    public static let padTokenId = 151643          // <|endoftext|>
    public static let imStartTokenId = 151644      // <|im_start|>
    public static let imEndTokenId = 151645        // <|im_end|>
    public static let timestampTokenId = 151705    // <|timestamp|>
    public static let asrTextTokenId = 151704      // <asr_text>
    public static let newlineTokenId = 198         // \n
    public static let systemTokenId = 8948         // system
    public static let userTokenId = 872            // user
    public static let assistantTokenId = 77091     // assistant
}

/// Main Qwen3-ASR model for speech recognition.
///
/// - Warning: This class is not thread-safe. Create separate instances for concurrent use.
public class Qwen3ASRModel {
    public let audioEncoder: Qwen3AudioEncoder
    public let featureExtractor: WhisperFeatureExtractor
    public var textDecoder: QuantizedTextModel?

    /// Tokenizer for decoding output tokens
    private var tokenizer: Qwen3Tokenizer?

    /// Text decoder config
    public let textConfig: TextDecoderConfig

    /// Whether the model weights are loaded and ready for inference.
    var _isLoaded = true

    init(
        audioConfig: Qwen3AudioEncoderConfig = .default,
        textConfig: TextDecoderConfig = .small
    ) {
        self.audioEncoder = Qwen3AudioEncoder(config: audioConfig)
        self.featureExtractor = WhisperFeatureExtractor()
        self.textConfig = textConfig
        // Text decoder will be initialized when loading weights
        self.textDecoder = nil
    }

    /// Set tokenizer for text decoding
    func setTokenizer(_ tokenizer: Qwen3Tokenizer) {
        self.tokenizer = tokenizer
    }

    /// Initialize text decoder (called after loading)
    func initializeTextDecoder() {
        self.textDecoder = QuantizedTextModel(config: textConfig)
    }

    // MARK: - Batched Transcription

    /// Transcribe multiple audio chunks in a single batched decoder pass.
    ///
    /// For greedy decoding (default options) this amortises decoder weight
    /// loads across `B` independent token streams on the MLX/GPU backend.
    ///
    /// Non-greedy options fall back to serial per-chunk decoding.
    public func transcribeBatch(
        audios: [[Float]],
        sampleRate: Int = 16000,
        language: String? = nil,
        maxTokens: Int? = nil,
        context: String? = nil,
        options: Qwen3DecodingOptions = Qwen3DecodingOptions()
    ) -> [String] {
        if audios.isEmpty { return [] }

        let resolvedOptions = Self.resolveBatchDecodingOptions(
            language: language, maxTokens: maxTokens, context: context, options: options
        )

        guard Self.isGreedyFastPath(resolvedOptions) else {
            return audios.map { transcribe(audio: $0, sampleRate: sampleRate, options: resolvedOptions) }
        }

        guard let textDecoder = textDecoder else {
            return audios.map { transcribe(audio: $0, sampleRate: sampleRate, options: resolvedOptions) }
        }

        let embedsPairs: [(embeds: MLXArray, seqLen: Int)] = audios.map { audio in
            let mel = featureExtractor.process(audio, sampleRate: sampleRate)
            let batched = mel.expandedDimensions(axis: 0)
            var ae = audioEncoder(batched)
            ae = ae.expandedDimensions(axis: 0)
            return (ae, ae.dim(1))
        }

        guard Self.isExperimentalBatchedDecodeEnabled() else {
            return generateTextBatchedBulkSync(
                audioEmbedsBatch: embedsPairs,
                textDecoder: textDecoder,
                language: resolvedOptions.language,
                maxTokens: resolvedOptions.maxTokens,
                context: resolvedOptions.context
            )
        }

        // Bucket by encoder sequence length. The batched decoder currently
        // requires equal-length prefill so cache offsets stay uniform across
        // rows; mixed-duration inputs are split into independent buckets and
        // restored to the caller's original order.
        let seqLenBatches = Self.makeSeqLenBatches(
            seqLens: embedsPairs.map(\.seqLen),
            maxBatchSize: embedsPairs.count
        )
        var results = Array(repeating: "", count: embedsPairs.count)
        for indices in seqLenBatches {
            if indices.count == 1, let index = indices.first {
                let pair = embedsPairs[index]
                results[index] = generateText(
                    audioEmbeds: pair.embeds,
                    textDecoder: textDecoder,
                    language: resolvedOptions.language,
                    maxTokens: resolvedOptions.maxTokens,
                    context: resolvedOptions.context,
                    decodingOptions: resolvedOptions
                )
                continue
            }

            let bucketPairs = indices.map { embedsPairs[$0] }
            let bucketResults = generateTextBatched(
                audioEmbedsBatch: bucketPairs,
                textDecoder: textDecoder,
                language: resolvedOptions.language,
                maxTokens: resolvedOptions.maxTokens,
                context: resolvedOptions.context
            )
            for (index, text) in zip(indices, bucketResults) {
                results[index] = text
            }
        }
        return results
    }

    /// Transcribe audio to text with explicit decoder options.
    ///
    /// The legacy `transcribe(audio:sampleRate:language:maxTokens:context:)`
    /// overload below forwards into this path with default (greedy) options.
    public func transcribe(
        audio: [Float],
        sampleRate: Int = 16000,
        options: Qwen3DecodingOptions
    ) -> String {
        let melFeatures = featureExtractor.process(audio, sampleRate: sampleRate)
        let batchedFeatures = melFeatures.expandedDimensions(axis: 0)
        var audioEmbeds = audioEncoder(batchedFeatures)
        audioEmbeds = audioEmbeds.expandedDimensions(axis: 0)
        guard let textDecoder = textDecoder else {
            let shape = audioEmbeds.shape
            return "[Audio encoded: \(shape)] - Text decoder not loaded"
        }
        return generateText(
            audioEmbeds: audioEmbeds,
            textDecoder: textDecoder,
            language: options.language,
            maxTokens: options.maxTokens,
            context: options.context,
            decodingOptions: options
        )
    }

    /// Transcribe audio to text
    public func transcribe(
        audio: [Float],
        sampleRate: Int = 16000,
        language: String? = nil,
        maxTokens: Int = 448,
        context: String? = nil
    ) -> String {
        // Extract mel features
        let melFeatures = featureExtractor.process(audio, sampleRate: sampleRate)

        // Add batch dimension: [mel, time] -> [1, mel, time]
        let batchedFeatures = melFeatures.expandedDimensions(axis: 0)

        // Encode audio - returns [time, features] without batch dim (matching Python)
        var audioEmbeds = audioEncoder(batchedFeatures)

        // Add batch dimension for consistency: [time, features] -> [1, time, features]
        audioEmbeds = audioEmbeds.expandedDimensions(axis: 0)

        // Check if text decoder is loaded
        guard let textDecoder = textDecoder else {
            let shape = audioEmbeds.shape
            return "[Audio encoded: \(shape)] - Text decoder not loaded"
        }

        // Generate text using the text decoder
        return generateText(
            audioEmbeds: audioEmbeds,
            textDecoder: textDecoder,
            language: language,
            maxTokens: maxTokens,
            context: context
        )
    }

    /// Generate text from audio embeddings.
    ///
    /// When `decodingOptions` is supplied, the decoder loop applies an
    /// HF-style repetition penalty, an optional no-repeat n-gram mask, and
    /// optional temperature sampling before each token selection. With the
    /// default `Qwen3DecodingOptions()` (repetition=1.0, no-repeat=0,
    /// temperature=0) behaviour is bit-identical to plain greedy.
    func generateText(
        audioEmbeds: MLXArray,
        textDecoder: QuantizedTextModel,
        language: String?,
        maxTokens: Int,
        context: String? = nil,
        decodingOptions: Qwen3DecodingOptions = Qwen3DecodingOptions()
    ) -> String {
        let T = Qwen3ASRTokens.self
        let numAudioTokens = audioEmbeds.dim(1)
        var inputIds: [Int32] = []

        // <|im_start|>system\n{context}<|im_end|>\n
        inputIds.append(contentsOf: [T.imStartTokenId, T.systemTokenId, T.newlineTokenId].map { Int32($0) })
        if let context = context, !context.isEmpty, let tokenizer = tokenizer {
            let contextTokens = tokenizer.encode(context)
            inputIds.append(contentsOf: contextTokens.map { Int32($0) })
        }
        inputIds.append(contentsOf: [T.imEndTokenId, T.newlineTokenId].map { Int32($0) })

        // <|im_start|>user\n<|audio_start|>
        inputIds.append(contentsOf: [T.imStartTokenId, T.userTokenId, T.newlineTokenId, T.audioStartTokenId].map { Int32($0) })

        // <|audio_pad|> * numAudioTokens (placeholder tokens that will be replaced)
        let audioStartIndex = inputIds.count
        for _ in 0..<numAudioTokens {
            inputIds.append(Int32(T.audioTokenId))
        }
        let audioEndIndex = inputIds.count

        // <|audio_end|><|im_end|>\n
        inputIds.append(contentsOf: [T.audioEndTokenId, T.imEndTokenId, T.newlineTokenId].map { Int32($0) })

        // <|im_start|>assistant\n
        inputIds.append(contentsOf: [T.imStartTokenId, T.assistantTokenId, T.newlineTokenId].map { Int32($0) })

        // Add language hint if specified, then always add <asr_text>. Without
        // <asr_text>, the model doesn't know it should transcribe. Without a
        // language hint, the model auto-detects and prepends "language XX".
        if let lang = language, let tokenizer = tokenizer {
            let langPrefix = "language \(lang)"
            let langTokens = tokenizer.encode(langPrefix)
            inputIds.append(contentsOf: langTokens.map { Int32($0) })
        }
        inputIds.append(Int32(T.asrTextTokenId))

        // Get text embeddings for all tokens
        let inputIdsTensor = MLXArray(inputIds).expandedDimensions(axis: 0)
        var inputEmbeds = textDecoder.embedTokens(inputIdsTensor)

        // Replace audio_pad token positions with actual audio embeddings
        let audioEmbedsTyped = audioEmbeds.asType(inputEmbeds.dtype)
        let beforeAudio = inputEmbeds[0..., 0..<audioStartIndex, 0...]
        let afterAudio = inputEmbeds[0..., audioEndIndex..., 0...]

        inputEmbeds = concatenated([beforeAudio, audioEmbedsTyped, afterAudio], axis: 1)

        // Initialize KV cache
        var cache: [(MLXArray, MLXArray)]? = nil

        // First pass: process the full input embeddings
        let (hiddenStates, newCache) = textDecoder(inputsEmbeds: inputEmbeds, cache: cache)
        cache = newCache

        // Get logits from the last position using embedding as LM head (tied weights)
        let seqLen = hiddenStates.dim(1)
        let lastHidden = hiddenStates[0..., (seqLen-1)..<seqLen, 0...]
        let logits = textDecoder.embedTokens.asLinear(lastHidden)

        // Greedy fast path uses a double-buffered asyncEval decode loop that
        // overlaps the GPU forward pass for token N+1 with the host-side
        // bookkeeping (EOS check + Swift array append) for token N. The
        // legacy slow path stays on the per-token CPU sync because it pulls
        // the full logits tensor to CPU for repetition / n-gram / temperature
        // manipulation, which would defeat the overlap.
        let generatedTokens: [Int32]
        if Self.isGreedyFastPath(decodingOptions) {
            generatedTokens = Self.generateGreedyAsyncEval(
                textDecoder: textDecoder,
                initialLogits: logits,
                cache: cache!,
                maxTokens: maxTokens
            )
        } else {
            generatedTokens = Self.generateSlow(
                textDecoder: textDecoder,
                initialLogits: logits,
                cache: cache!,
                maxTokens: maxTokens,
                options: decodingOptions
            )
        }

        // Decode tokens to text
        if let tokenizer = tokenizer {
            let rawText = tokenizer.decode(tokens: generatedTokens.map { Int($0) })
            // Strip "language XX<asr_text>" prefix if present (auto-detection output)
            if let range = rawText.range(of: "<asr_text>") {
                return String(rawText[range.upperBound...]).trimmingCharacters(in: .whitespaces)
            }
            return rawText
        } else {
            // Fallback: return token IDs
            return generatedTokens.map { String($0) }.joined(separator: " ")
        }
    }

    /// Greedy with default options: temperature 0, no repetition penalty,
    /// no n-gram blocking. The double-buffered asyncEval loop only kicks
    /// in for this configuration so we can guarantee bit-identical token
    /// sequences vs. the legacy `argMax(...).item()` decoder.
    static func isGreedyFastPath(_ options: Qwen3DecodingOptions) -> Bool {
        return options.repetitionPenalty == 1.0
            && options.noRepeatNgramSize == 0
            && options.temperature == 0.0
    }

    static func resolveBatchDecodingOptions(
        language: String?,
        maxTokens: Int?,
        context: String?,
        options: Qwen3DecodingOptions
    ) -> Qwen3DecodingOptions {
        var resolved = options
        if let language {
            resolved.language = language
        }
        if let maxTokens {
            resolved.maxTokens = maxTokens
        }
        if let context {
            resolved.context = context
        }
        return resolved
    }

    static func isExperimentalBatchedDecodeEnabled() -> Bool {
        ProcessInfo.processInfo.environment["QWEN3_ASR_EXPERIMENTAL_BATCH_DECODE"] == "1"
    }

    /// Return input indices grouped by encoder sequence length, bounded by
    /// `maxBatchSize`. Order within each bucket follows the input order so
    /// equal-length fixed chunks keep deterministic batching behaviour.
    static func makeSeqLenBatches(seqLens: [Int], maxBatchSize: Int) -> [[Int]] {
        guard !seqLens.isEmpty else { return [] }
        let cappedBatchSize = max(1, maxBatchSize)
        var bucketsBySeqLen: [Int: [[Int]]] = [:]
        var orderedSeqLens: [Int] = []

        for (index, seqLen) in seqLens.enumerated() {
            if bucketsBySeqLen[seqLen] == nil {
                bucketsBySeqLen[seqLen] = [[]]
                orderedSeqLens.append(seqLen)
            }
            var batches = bucketsBySeqLen[seqLen]!
            if batches[batches.count - 1].count >= cappedBatchSize {
                batches.append([])
            }
            batches[batches.count - 1].append(index)
            bucketsBySeqLen[seqLen] = batches
        }

        return orderedSeqLens.flatMap { bucketsBySeqLen[$0] ?? [] }
    }

    /// Double-buffered greedy decode loop. The key trick is to keep the
    /// "next token" as a lazy 0-D `MLXArray` (the result of `argMax`),
    /// build the *next* step's forward pass on top of it (still lazy),
    /// then call `MLX.asyncEval` so the GPU starts computing step N+1
    /// before we sync step N's int32 to CPU. The host-side EOS check
    /// and `generatedTokens.append` then overlap with the in-flight GPU
    /// work for step N+1 instead of stalling between every token.
    ///
    /// Greedy correctness invariant: argMax is deterministic, so this
    /// produces the exact same token sequence as the legacy loop on
    /// matching inputs.
    static func generateGreedyAsyncEval(
        textDecoder: QuantizedTextModel,
        initialLogits: MLXArray,
        cache initialCache: [(MLXArray, MLXArray)],
        maxTokens: Int
    ) -> [Int32] {
        var generatedTokens: [Int32] = []
        guard maxTokens > 0 else { return generatedTokens }

        // Stage 0: argmax of the prefill's last logits. Stays lazy until
        // the first `.item()` below.
        //
        // Cast to int32 explicitly: MLX's `argmax` returns uint32, but the
        // legacy loop fed `embedTokens` an int32 tensor (built from a Swift
        // `Int32`). Quantized embedding lookup observably dispatches
        // differently on uint32 vs. int32, producing tokens that diverge
        // from the legacy path on a small fraction of inputs. Casting here
        // restores exact dtype parity, so greedy stays token-for-token
        // identical to the pre-optimisation decoder.
        var nextTokenArr = argMax(initialLogits, axis: -1).squeezed().asType(.int32)
        var cache = initialCache
        // Kick off the GPU on the first token (and the prefill cache that
        // step 1's graph will read from).
        asyncEval(nextTokenArr, cache)

        let eosToken = Int32(Qwen3ASRTokens.eosTokenId)

        for step in 0..<maxTokens {
            // Stage N+1's graph BEFORE syncing N. embedTokens expects a
            // [batch, seq] int32 tensor; nextTokenArr is 0-D so we expand
            // twice to [1, 1].
            //
            // Skip the speculative graph build on the LAST iteration —
            // we'd never consume it, and on tiny `maxTokens` the saved
            // GPU/host work matters.
            var nextTokenArrN1: MLXArray? = nil
            var cacheN1: [(MLXArray, MLXArray)]? = nil
            if step + 1 < maxTokens {
                let nextEmbed = textDecoder.embedTokens(
                    nextTokenArr.expandedDimensions(axis: 0).expandedDimensions(axis: 0)
                )
                let (hiddenN1, newCacheN1) = textDecoder(inputsEmbeds: nextEmbed, cache: cache)
                let lastHiddenN1 = hiddenN1[0..., (-1)..., .ellipsis]
                let logitsN1 = textDecoder.embedTokens.asLinear(lastHiddenN1)
                let argN1 = argMax(logitsN1, axis: -1).squeezed().asType(.int32)
                // Kick GPU on N+1 (chains after the asyncEval that is
                // still computing N).
                asyncEval(argN1, newCacheN1)
                nextTokenArrN1 = argN1
                cacheN1 = newCacheN1
            }

            // Now sync N. By the time we reach `.item()` the GPU has
            // very likely finished N already, and the cost of this call
            // shrinks to a host-side memcpy of one int32 — which itself
            // overlaps with the in-flight N+1 forward pass.
            let nextToken = nextTokenArr.item(Int32.self)
            // Match the legacy loop semantics exactly: append the token
            // first, *then* break on EOS. The legacy loop emitted EOS
            // into `generatedTokens` whenever EOS was the most recent
            // pick, so greedy stays bit-identical.
            generatedTokens.append(nextToken)
            if nextToken == eosToken { break }

            guard let advancedCache = cacheN1, let advancedToken = nextTokenArrN1 else {
                // Final iteration without speculative work — nothing to
                // advance to; the loop will exit naturally next.
                break
            }
            cache = advancedCache
            nextTokenArr = advancedToken
        }
        return generatedTokens
    }

    // MARK: - Batched Greedy Decode
    //
    // Default path: serial per-row decoder forwards with one batched token
    // sync per step. This keeps row-level correctness identical to serial
    // transcribe while removing the naive B scalar `.item()` sync points.
    //
    // Experimental path (`QWEN3_ASR_EXPERIMENTAL_BATCH_DECODE=1`): true
    // [B,1,H] decoder forward with stacked KV cache, bucketed by encoder
    // sequence length. This is closer to vLLM/SGLang continuous batching but
    // remains gated until row-level correctness is resolved.

    func generateTextBatchedBulkSync(
        audioEmbedsBatch: [(embeds: MLXArray, seqLen: Int)],
        textDecoder: QuantizedTextModel,
        language: String?,
        maxTokens: Int,
        context: String? = nil
    ) -> [String] {
        let B = audioEmbedsBatch.count
        if B == 0 { return [] }
        if maxTokens <= 0 { return Self.decodeTokenBatch(allTokens: Array(repeating: [], count: B), tokenizer: tokenizer) }

        let (prefixTokenIds, suffixTokenIds) = Self.buildPromptTokenIds(
            language: language, context: context, tokenizer: tokenizer
        )
        let prefixLen = prefixTokenIds.count

        var itemCaches: [[(MLXArray, MLXArray)]] = []
        var currentItemTokenIds: [MLXArray] = []

        for pair in audioEmbedsBatch {
            let inputEmbeds = Self.buildInputEmbeds(
                audioEmbeds: pair.embeds,
                audioTokenCount: pair.seqLen,
                prefixTokenIds: prefixTokenIds,
                suffixTokenIds: suffixTokenIds,
                audioStartIndex: prefixLen,
                textDecoder: textDecoder
            )

            let (hiddenStates, itemCache) = textDecoder(inputsEmbeds: inputEmbeds)
            let seqLen = hiddenStates.dim(1)
            let lastHidden = hiddenStates[0..., (seqLen - 1)..<seqLen, 0...]
            let logits = textDecoder.embedTokens.asLinear(lastHidden)
            currentItemTokenIds.append(argMax(logits, axis: -1).asType(.int32))
            itemCaches.append(itemCache)
        }

        var currentBatchTokenIds = concatenated(currentItemTokenIds, axis: 0)  // [B, 1]
        asyncEval(currentBatchTokenIds)

        let eosToken = Int32(Qwen3ASRTokens.eosTokenId)
        var finished = [Bool](repeating: false, count: B)
        var allTokens: [[Int32]] = (0..<B).map { _ in [] }

        for step in 0..<maxTokens {
            var nextItemTokenIds: [MLXArray]?
            var nextItemCaches: [[(MLXArray, MLXArray)]]?
            var nextBatchTokenIds: MLXArray?

            if step + 1 < maxTokens {
                var stagedTokens: [MLXArray] = []
                var stagedCaches: [[(MLXArray, MLXArray)]] = []
                for b in 0..<B {
                    if finished[b] {
                        stagedTokens.append(currentItemTokenIds[b])
                        stagedCaches.append(itemCaches[b])
                        continue
                    }

                    let tokenEmbeds = textDecoder.embedTokens(currentItemTokenIds[b])
                    let (hiddenStates, newCache) = textDecoder(inputsEmbeds: tokenEmbeds, cache: itemCaches[b])
                    let lastHidden = hiddenStates[0..., (-1)..., .ellipsis]
                    let logits = textDecoder.embedTokens.asLinear(lastHidden)
                    stagedTokens.append(argMax(logits, axis: -1).asType(.int32))
                    stagedCaches.append(newCache)
                }

                let stagedBatch = concatenated(stagedTokens, axis: 0)
                asyncEval(stagedBatch)
                nextItemTokenIds = stagedTokens
                nextItemCaches = stagedCaches
                nextBatchTokenIds = stagedBatch
            }

            // One CPU synchronization per decode step for the whole batch,
            // rather than B scalar `.item()` calls. Decoder forwards remain
            // per-item, so this path preserves serial row correctness while
            // removing the dominant host sync pattern from naive batching.
            let tokenValues = currentBatchTokenIds.reshaped([B]).asArray(Int32.self)
            for b in 0..<B {
                if finished[b] { continue }
                let t = tokenValues[b]
                allTokens[b].append(t)
                if t == eosToken { finished[b] = true }
            }

            if finished.allSatisfy({ $0 }) { break }
            guard let nextItemTokenIds, let nextItemCaches, let nextBatchTokenIds else { break }
            currentItemTokenIds = nextItemTokenIds
            itemCaches = nextItemCaches
            currentBatchTokenIds = nextBatchTokenIds
        }

        return Self.decodeTokenBatch(allTokens: allTokens, tokenizer: tokenizer)
    }

    func generateTextBatched(
        audioEmbedsBatch: [(embeds: MLXArray, seqLen: Int)],
        textDecoder: QuantizedTextModel,
        language: String?,
        maxTokens: Int,
        context: String? = nil
    ) -> [String] {
        let B = audioEmbedsBatch.count
        if B == 0 { return [] }
        if maxTokens <= 0 { return Self.decodeTokenBatch(allTokens: Array(repeating: [], count: B), tokenizer: tokenizer) }

        precondition(
            audioEmbedsBatch.allSatisfy { $0.seqLen == audioEmbedsBatch[0].seqLen },
            "generateTextBatched currently requires equal-length audio encoder outputs"
        )

        let eosToken = Int32(Qwen3ASRTokens.eosTokenId)

        // Phase 1: Build prompt token IDs for each item.
        // All items share the same prefix tokens; only the audio_pad count differs.
        let (prefixTokenIds, suffixTokenIds) = Self.buildPromptTokenIds(
            language: language, context: context, tokenizer: tokenizer
        )
        let prefixLen = prefixTokenIds.count

        var perItemInputEmbeds: [MLXArray] = []

        for pair in audioEmbedsBatch {
            let inputEmbeds = Self.buildInputEmbeds(
                audioEmbeds: pair.embeds,
                audioTokenCount: pair.seqLen,
                prefixTokenIds: prefixTokenIds,
                suffixTokenIds: suffixTokenIds,
                audioStartIndex: prefixLen,
                textDecoder: textDecoder
            )
            perItemInputEmbeds.append(inputEmbeds.squeezed(axis: 0))
        }

        // Phase 2: Prefill each chunk independently, then stack the KV cache
        // into one batched cache. This preserves the exact single-item prefill
        // semantics while still letting the autoregressive decode loop run as
        // one [B, 1, H] stream. Batched prefill can be revisited once it has a
        // bit-exact regression test against the serial path.
        var itemCaches: [[(MLXArray, MLXArray)]] = []
        var itemFirstTokenIds: [MLXArray] = []
        for inputEmbeds in perItemInputEmbeds {
            let itemInput = inputEmbeds.expandedDimensions(axis: 0)
            let (hiddenStates, itemCache) = textDecoder(inputsEmbeds: itemInput)
            let seqLen = hiddenStates.dim(1)
            let lastHidden = hiddenStates[0..., (seqLen - 1)..<seqLen, 0...]
            let logits = textDecoder.embedTokens.asLinear(lastHidden)
            itemFirstTokenIds.append(argMax(logits, axis: -1).asType(.int32))
            itemCaches.append(itemCache)
        }

        let batchTokenIds = concatenated(itemFirstTokenIds, axis: 0)  // [B, 1]
        var batchCache: [(MLXArray, MLXArray)] = []
        for layerIndex in 0..<itemCaches[0].count {
            let keys = concatenated(itemCaches.map { $0[layerIndex].0 }, axis: 0)
            let values = concatenated(itemCaches.map { $0[layerIndex].1 }, axis: 0)
            batchCache.append((keys, values))
        }

        var finished = [Bool](repeating: false, count: B)
        var allTokens: [[Int32]] = (0..<B).map { _ in [] }

        // Phase 4: Batched autoregressive decode.
        //
        // Strategy (inspired by vLLM/SGLang continuous batching, simplified):
        //
        // We keep the batched KV cache as a single tensor [B, numKVHeads, T, headDim]
        // and grow it uniformly for ALL items—including finished ones that contribute
        // pad tokens. This avoids the PagedAttention complexity (variable cache lengths
        // per item) at the cost of a small amount of wasted compute on pad tokens for
        // items that already hit EOS. For ASR with equal-length audio chunks, this
        // waste is near-zero because all items finish within a few tokens of each other.
        //
        // The key win: each decode step does ONE batched [B,1,H] forward pass through
        // all 28 decoder layers. The quantized weight dequant + GEMM (the bottleneck)
        // is amortised across B items instead of being repeated B times.

        var batchKVCache = batchCache  // [(MLXArray, MLXArray)], each [B, numKVHeads, T, headDim]
        var currentTokenIds = batchTokenIds  // [B, 1]
        asyncEval(currentTokenIds, batchKVCache)

        for step in 0..<maxTokens {
            var nextTokenIdsN1: MLXArray?
            var cacheN1: [(MLXArray, MLXArray)]?

            if step + 1 < maxTokens {
                // Embed all B tokens (finished items get re-embedded; their
                // logits are ignored). Cache grows uniformly.
                let batchEmbeds = textDecoder.embedTokens(currentTokenIds)  // [B, 1, H]

                // Single batched forward for ALL items.
                let (hidden, newBatchCache) = textDecoder(inputsEmbeds: batchEmbeds, cache: batchKVCache)

                // Batched LM head on the last position.
                let lastH = hidden[0..., (-1)..., 0...]  // [B, 1, H]
                var perItemTokenIds: [MLXArray] = []
                for b in 0..<B {
                    let itemHidden = lastH[b..<(b + 1), 0..., 0...]  // [1, 1, H]
                    let itemLogits = textDecoder.embedTokens.asLinear(itemHidden)  // [1, 1, V]
                    perItemTokenIds.append(argMax(itemLogits, axis: -1).asType(.int32))
                }
                let nextBatchTokenIds = concatenated(perItemTokenIds, axis: 0)  // [B, 1]

                asyncEval(nextBatchTokenIds, newBatchCache)
                nextTokenIdsN1 = nextBatchTokenIds
                cacheN1 = newBatchCache
            }

            // Sync the whole [B, 1] token tensor once. This avoids B separate
            // scalar host/device sync points and keeps row extraction aligned
            // with MLX's contiguous flattened layout.
            let tokenValues = currentTokenIds.reshaped([B]).asArray(Int32.self)
            for b in 0..<B {
                if finished[b] { continue }
                let t = tokenValues[b]
                allTokens[b].append(t)
                if t == eosToken { finished[b] = true }
            }

            if finished.allSatisfy({ $0 }) { break }

            guard let nextTokenIdsN1, let cacheN1 else { break }
            currentTokenIds = nextTokenIdsN1
            batchKVCache = cacheN1
        }

        return Self.decodeTokenBatch(allTokens: allTokens, tokenizer: tokenizer)
    }

    /// Build the shared prompt prefix token IDs and per-item suffix.
    /// The prefix is everything up to (not including) the audio_pad tokens.
    /// The suffix is everything after them.
    static func buildPromptTokenIds(
        language: String?,
        context: String?,
        tokenizer: Qwen3Tokenizer?
    ) -> (prefix: [Int32], suffix: [Int32]) {
        let T = Qwen3ASRTokens.self

        // Prefix: everything before the audio_pad tokens.
        var prefix: [Int32] = []
        prefix.append(contentsOf: [T.imStartTokenId, T.systemTokenId, T.newlineTokenId].map { Int32($0) })
        if let context = context, !context.isEmpty, let tokenizer = tokenizer {
            let contextTokens = tokenizer.encode(context)
            prefix.append(contentsOf: contextTokens.map { Int32($0) })
        }
        prefix.append(contentsOf: [T.imEndTokenId, T.newlineTokenId].map { Int32($0) })
        prefix.append(contentsOf: [T.imStartTokenId, T.userTokenId, T.newlineTokenId, T.audioStartTokenId].map { Int32($0) })

        // Suffix: everything after the audio_pad tokens.
        var suffix: [Int32] = []
        suffix.append(contentsOf: [T.audioEndTokenId, T.imEndTokenId, T.newlineTokenId].map { Int32($0) })
        suffix.append(contentsOf: [T.imStartTokenId, T.assistantTokenId, T.newlineTokenId].map { Int32($0) })
        if let lang = language, let tokenizer = tokenizer {
            let langPrefix = "language \(lang)"
            let langTokens = tokenizer.encode(langPrefix)
            suffix.append(contentsOf: langTokens.map { Int32($0) })
        }
        suffix.append(Int32(T.asrTextTokenId))

        return (prefix, suffix)
    }

    static func buildInputEmbeds(
        audioEmbeds: MLXArray,
        audioTokenCount: Int,
        prefixTokenIds: [Int32],
        suffixTokenIds: [Int32],
        audioStartIndex: Int,
        textDecoder: QuantizedTextModel
    ) -> MLXArray {
        var inputIds = prefixTokenIds
        for _ in 0..<audioTokenCount {
            inputIds.append(Int32(Qwen3ASRTokens.audioTokenId))
        }
        inputIds.append(contentsOf: suffixTokenIds)

        let inputIdsTensor = MLXArray(inputIds).expandedDimensions(axis: 0)
        var inputEmbeds = textDecoder.embedTokens(inputIdsTensor)

        let audioEmbedsTyped = audioEmbeds.asType(inputEmbeds.dtype)
        let audioEndIndex = audioStartIndex + audioTokenCount
        let beforeAudio = inputEmbeds[0..., 0..<audioStartIndex, 0...]
        let afterAudio = inputEmbeds[0..., audioEndIndex..., 0...]
        inputEmbeds = concatenated([beforeAudio, audioEmbedsTyped, afterAudio], axis: 1)
        return inputEmbeds
    }

    static func decodeTokenBatch(
        allTokens: [[Int32]],
        tokenizer: Qwen3Tokenizer?
    ) -> [String] {
        guard let tokenizer = tokenizer else {
            return allTokens.map { $0.map { String($0) }.joined(separator: " ") }
        }
        return allTokens.map { tokens in
            let rawText = tokenizer.decode(tokens: tokens.map { Int($0) })
            if let range = rawText.range(of: "<asr_text>") {
                return String(rawText[range.upperBound...]).trimmingCharacters(in: .whitespaces)
            }
            return rawText
        }
    }

    /// Legacy decode loop kept verbatim for the non-greedy slow path.
    /// `pickNextToken` here pulls logits to CPU for repetition penalty,
    /// n-gram masking, and temperature sampling, so there's no benefit
    /// from `asyncEval` overlap.
    static func generateSlow(
        textDecoder: QuantizedTextModel,
        initialLogits: MLXArray,
        cache initialCache: [(MLXArray, MLXArray)],
        maxTokens: Int,
        options: Qwen3DecodingOptions
    ) -> [Int32] {
        var generatedTokens: [Int32] = []
        guard maxTokens > 0 else { return generatedTokens }
        var cache: [(MLXArray, MLXArray)]? = initialCache

        var nextToken = Self.pickNextToken(
            logits: initialLogits,
            generatedSoFar: generatedTokens,
            options: options
        )
        generatedTokens.append(nextToken)

        for _ in 1..<maxTokens {
            if nextToken == Int32(Qwen3ASRTokens.eosTokenId) { break }

            let tokenEmbeds = textDecoder.embedTokens(
                MLXArray([nextToken]).expandedDimensions(axis: 0)
            )
            let (hiddenStates, newCache) = textDecoder(inputsEmbeds: tokenEmbeds, cache: cache)
            cache = newCache

            let lastHiddenNext = hiddenStates[0..., (-1)..., .ellipsis]
            let logits = textDecoder.embedTokens.asLinear(lastHiddenNext)
            nextToken = Self.pickNextToken(
                logits: logits,
                generatedSoFar: generatedTokens,
                options: options
            )
            generatedTokens.append(nextToken)
        }
        return generatedTokens
    }

    // MARK: - Decoder knobs

    /// Pick the next token from a logits tensor, applying repetition
    /// penalty, no-repeat n-gram masking, and optional temperature sampling.
    ///
    /// With default options (repetition=1.0, noRepeat=0, temperature=0) the
    /// result is the same `argMax` the decoder used pre-refactor.
    /// Implementation pulls logits to CPU (a 1-D Float array of vocab size)
    /// so we can manipulate entries in-place without fighting MLX indexing.
    ///
    /// Access is `internal static` (not `private`) so
    /// ``Qwen3DecodingOptionsTests`` can exercise the sampler directly via
    /// ``@testable import Qwen3ASR`` — there is no GPU or model download
    /// involved so the path is trivially unit-testable once reachable.
    static func pickNextToken(
        logits: MLXArray,
        generatedSoFar: [Int32],
        options: Qwen3DecodingOptions
    ) -> Int32 {
        // Fast path — pure greedy, no modifications.
        if options.repetitionPenalty == 1.0,
           options.noRepeatNgramSize == 0,
           options.temperature == 0 {
            return argMax(logits, axis: -1).squeezed().item(Int32.self)
        }

        // Pull logits to CPU. `logits` is [1, 1, vocabSize]; after squeeze
        // and conversion we have a plain `[Float]` of length vocabSize.
        let flat = logits.squeezed().asType(.float32)
        let vocabSize = flat.size
        var scores: [Float] = flat.asArray(Float.self)
        precondition(scores.count == vocabSize, "pickNextToken: vocab size mismatch")

        // Repetition penalty: divide logits for already-generated tokens.
        if options.repetitionPenalty > 1.0 && !generatedSoFar.isEmpty {
            let penalty = options.repetitionPenalty
            for token in Set(generatedSoFar) {
                let idx = Int(token)
                guard idx >= 0, idx < vocabSize else { continue }
                let v = scores[idx]
                // Positive logits divide; negative logits multiply — matches
                // HuggingFace's implementation so the penalty always reduces
                // the probability of the repeated token.
                scores[idx] = v > 0 ? v / penalty : v * penalty
            }
        }

        // No-repeat-ngram: any next token whose emission would form a
        // repeated n-gram of size N gets pushed to -infinity.
        let n = options.noRepeatNgramSize
        if n > 0 && generatedSoFar.count >= n - 1 {
            let lastPrefix = Array(generatedSoFar.suffix(n - 1))
            // Walk every position where `lastPrefix` already appeared —
            // the token that followed it becomes forbidden as the NEXT
            // token now.
            if generatedSoFar.count >= n {
                for i in 0...(generatedSoFar.count - n) {
                    let window = Array(generatedSoFar[i..<(i + n - 1)])
                    guard window == lastPrefix else { continue }
                    let forbidden = Int(generatedSoFar[i + n - 1])
                    if forbidden >= 0 && forbidden < vocabSize {
                        scores[forbidden] = -.infinity
                    }
                }
            }
        }

        // Temperature sampling via Gumbel-max trick:
        // argmax(logits/T + Gumbel(0,1)) ~ categorical(softmax(logits/T)).
        if options.temperature > 0 {
            let t = options.temperature
            for i in 0..<vocabSize {
                let u = Float.random(in: 1e-6...1.0)
                scores[i] = scores[i] / t - Float.log(-Float.log(u))
            }
        }

        // Argmax of the adjusted scores.
        var bestIdx = 0
        var bestScore = -Float.infinity
        for i in 0..<vocabSize where scores[i] > bestScore {
            bestScore = scores[i]
            bestIdx = i
        }
        return Int32(bestIdx)
    }
}

// MARK: - Backward Compatibility (delegates to HuggingFaceDownloader)

public extension Qwen3ASRModel {
    static func sanitizedCacheKey(for modelId: String) -> String {
        HuggingFaceDownloader.sanitizedCacheKey(for: modelId)
    }

    static func validatedRemoteFileName(_ file: String) throws -> String {
        try HuggingFaceDownloader.validatedRemoteFileName(file)
    }

    static func validatedLocalPath(directory: URL, fileName: String) throws -> URL {
        try HuggingFaceDownloader.validatedLocalPath(directory: directory, fileName: fileName)
    }
}

// MARK: - Model Size Detection

/// Supported ASR model sizes
public enum ASRModelSize {
    case small  // 0.6B
    case large  // 1.7B

    /// Default model IDs on HuggingFace
    public var defaultModelId: String {
        switch self {
        case .small: return "aufklarer/Qwen3-ASR-0.6B-MLX-4bit"
        case .large: return "aufklarer/Qwen3-ASR-1.7B-MLX-8bit"
        }
    }

    /// Audio encoder config for this model size
    public var audioConfig: Qwen3AudioEncoderConfig {
        switch self {
        case .small: return .small
        case .large: return .large
        }
    }

    /// Text decoder config for this model size and quantization bits
    public func textConfig(bits: Int) -> TextDecoderConfig {
        switch (self, bits) {
        case (.small, 8): return .small8bit
        case (.small, _): return .small
        case (.large, 8): return .large8bit
        case (.large, _): return .large
        }
    }

    /// Text decoder config for this model size (default bits)
    public var textConfig: TextDecoderConfig {
        switch self {
        case .small: return .small
        case .large: return .large
        }
    }

    /// Detect model size from a HuggingFace model ID
    public static func detect(from modelId: String) -> ASRModelSize {
        if modelId.contains("1.7B") || modelId.contains("1.7b") {
            return .large
        }
        return .small
    }

    /// Detect quantization bits from a HuggingFace model ID.
    /// Returns 4 by default for 0.6B, 8 for 1.7B if not specified.
    public static func detectBits(from modelId: String) -> Int {
        let lower = modelId.lowercased()
        if lower.contains("8bit") || lower.contains("8-bit") {
            return 8
        }
        if lower.contains("4bit") || lower.contains("4-bit") {
            return 4
        }
        // Default: 4 for small, 8 for large (backwards-compatible)
        let size = detect(from: modelId)
        return size == .large ? 8 : 4
    }
}

// MARK: - Model Loading

public extension Qwen3ASRModel {
    /// Load model from HuggingFace hub with automatic weight downloading
    static func fromPretrained(
        modelId: String = "aufklarer/Qwen3-ASR-0.6B-MLX-4bit",
        cacheDir: URL? = nil,
        offlineMode: Bool = false,
        progressHandler: ((Double, String) -> Void)? = nil
    ) async throws -> Qwen3ASRModel {
        progressHandler?(0.0, "Downloading model...")

        // Auto-detect model size and quantization bits from model ID
        let modelSize = ASRModelSize.detect(from: modelId)
        let detectedBits = ASRModelSize.detectBits(from: modelId)

        // Get cache directory
        let cacheDir = try cacheDir ?? HuggingFaceDownloader.getCacheDirectory(for: modelId)

        // Download weights and tokenizer files (skips files that already exist on disk)
        // Download is the slowest part — give it 0-80% of progress
        try await HuggingFaceDownloader.downloadWeights(
            modelId: modelId,
            to: cacheDir,
            additionalFiles: ["vocab.json", "merges.txt", "tokenizer_config.json"],
            offlineMode: offlineMode,
            progressHandler: { progress in
                progressHandler?(progress * 0.8, "Downloading weights...")
            }
        )

        progressHandler?(0.80, "Loading tokenizer...")

        // Create model with appropriate config for detected size and bits
        let model = Qwen3ASRModel(
            audioConfig: modelSize.audioConfig,
            textConfig: modelSize.textConfig(bits: detectedBits)
        )

        // Load tokenizer from vocab.json
        let vocabPath = cacheDir.appendingPathComponent("vocab.json")
        if FileManager.default.fileExists(atPath: vocabPath.path) {
            let tokenizer = Qwen3Tokenizer()
            try tokenizer.load(from: vocabPath)
            model.setTokenizer(tokenizer)
        }

        progressHandler?(0.85, "Loading audio encoder weights...")

        // Load audio encoder weights
        try WeightLoader.loadWeights(into: model.audioEncoder, from: cacheDir)

        progressHandler?(0.92, "Loading text decoder weights...")

        // Initialize and load text decoder
        model.initializeTextDecoder()
        if let textDecoder = model.textDecoder {
            try WeightLoader.loadTextDecoderWeights(into: textDecoder, from: cacheDir)
        }

        MetalBudget.pinMemory()
        progressHandler?(1.0, "Ready")

        return model
    }
}
