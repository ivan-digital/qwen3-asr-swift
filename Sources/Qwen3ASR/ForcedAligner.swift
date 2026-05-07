import Foundation
import MLXCommon
import MLX
import MLXNN
import MLXFast
import AudioCommon

// AlignedWord is defined in AudioCommon/Protocols.swift and re-exported via AudioCommon import above.

/// Forced aligner model variant
public enum ForcedAlignerVariant: String, CaseIterable, Sendable {
    case mlx4bit = "aufklarer/Qwen3-ForcedAligner-0.6B-4bit"
    case mlx8bit = "aufklarer/Qwen3-ForcedAligner-0.6B-8bit"
    case bf16 = "aufklarer/Qwen3-ForcedAligner-0.6B-bf16"

    /// Detect variant from model ID string
    public static func detect(from modelId: String) -> ForcedAlignerVariant? {
        if let exact = Self.allCases.first(where: { $0.rawValue == modelId }) {
            return exact
        }
        if modelId.contains("bf16") || modelId.contains("float") { return .bf16 }
        if modelId.contains("8bit") { return .mlx8bit }
        if modelId.contains("4bit") { return .mlx4bit }
        return nil
    }

    public var textConfig: TextDecoderConfig {
        switch self {
        case .mlx4bit:
            var cfg = TextDecoderConfig.small
            cfg.bits = 4
            cfg.groupSize = 64
            return cfg
        case .mlx8bit:
            var cfg = TextDecoderConfig.small
            cfg.bits = 8
            cfg.groupSize = 64
            return cfg
        case .bf16:
            return .small
        }
    }

    public var usesFloatTextDecoder: Bool {
        self == .bf16
    }
}

/// Qwen3 Forced Aligner — predicts word-level timestamps for audio+text pairs.
///
/// Uses the same encoder-decoder architecture as Qwen3-ASR but replaces the
/// vocab lm_head with a 5000-class timestamp classification head.
/// Inference is non-autoregressive (single forward pass).
public class Qwen3ForcedAligner {
    public let audioEncoder: Qwen3AudioEncoder
    public let textDecoder: any ForcedAlignerTextDecoding
    public let classifyHead: Linear
    public let featureExtractor: WhisperFeatureExtractor
    public var tokenizer: Qwen3Tokenizer?

    private let config: Qwen3ASRConfig

    public init(
        audioConfig: Qwen3AudioEncoderConfig = .forcedAligner,
        textConfig: TextDecoderConfig = .small,
        classifyNum: Int = 5000,
        useFloatTextDecoder: Bool = false
    ) {
        self.audioEncoder = Qwen3AudioEncoder(config: audioConfig)
        if useFloatTextDecoder {
            self.textDecoder = FloatTextModel(config: textConfig)
        } else {
            self.textDecoder = QuantizedTextModel(config: textConfig)
        }
        self.classifyHead = Linear(textConfig.hiddenSize, classifyNum)
        self.featureExtractor = WhisperFeatureExtractor()

        var cfg = Qwen3ASRConfig()
        cfg.classifyNum = classifyNum
        self.config = cfg
    }

    /// Align text to audio with automatic chunking for long inputs.
    ///
    /// The underlying classifier head emits a fixed-resolution timestamp
    /// index (default `classifyNum=5000` × `0.08s` per slot = 400s
    /// addressable range) but in practice the model's reliable range is
    /// shorter (~270s on Qwen3-ForcedAligner-0.6B-4bit observed on TED-Ed
    /// material). Past that, it produces low/non-monotonic indices that
    /// LIS correction collapses into a flat plateau — every trailing word
    /// shares the same timestamp.
    ///
    /// `alignLong` runs `align` on the full audio, detects the trailing
    /// plateau, keeps the reliable prefix, then re-aligns the remaining
    /// audio + remaining words and offsets timestamps. Iterates until no
    /// plateau remains or the remaining work is below a minimum chunk size.
    ///
    /// For audio shorter than the threshold this is a one-pass call into
    /// `align`. For longer audio it pays one extra align pass per chunk.
    public func alignLong(
        audio: [Float],
        text: String,
        sampleRate: Int = 16000,
        language: String = "English",
        progressHandler: ((String) -> Void)? = nil
    ) -> [AlignedWord] {
        // The model is reliable up to ~270s on the bundles we ship; we
        // don't try to be too aggressive with the threshold so the
        // single-pass case stays the common path. The plateau detector
        // does the actual work — this is just a fast bypass when there's
        // no risk of saturation.
        let bypassThresholdSeconds: Float = 240
        let minChunkSeconds: Float = 5
        let plateauTolerance: Float = 0.1     // seconds; "same start time" if diff < this
        let plateauMinWords = 5               // need ≥ N stuck words to call it a plateau

        var allAligned: [AlignedWord] = []
        var remainingAudio = audio
        var remainingText = text
        var offsetSec: Float = 0
        var pass = 1

        while !remainingAudio.isEmpty && !remainingText.isEmpty {
            let durationSec = Float(remainingAudio.count) / Float(sampleRate)
            let aligned = align(
                audio: remainingAudio,
                text: remainingText,
                sampleRate: sampleRate,
                language: language
            )
            if aligned.isEmpty { break }

            // Skip plateau detection on small chunks — the model is reliable
            // there, and detecting plateau on tiny outputs creates spurious
            // splits.
            if durationSec <= bypassThresholdSeconds || aligned.count < plateauMinWords * 2 {
                allAligned.append(contentsOf: Self.offsetWords(aligned, by: offsetSec))
                break
            }

            let plateauStart = Self.findTrailingPlateauStart(
                aligned, tolerance: plateauTolerance, minSize: plateauMinWords
            )
            if plateauStart == aligned.count {
                // No plateau — alignment looks healthy.
                allAligned.append(contentsOf: Self.offsetWords(aligned, by: offsetSec))
                break
            }

            // Take the reliable prefix; recurse on the remainder.
            let reliable = aligned.prefix(plateauStart)
            let splitTime = reliable.last!.endTime
            allAligned.append(contentsOf: Self.offsetWords(Array(reliable), by: offsetSec))

            let splitSample = Int(splitTime * Float(sampleRate))
            guard splitSample < remainingAudio.count else { break }
            let nextAudio = Array(remainingAudio[splitSample...])
            let remainingDuration = Float(nextAudio.count) / Float(sampleRate)
            if remainingDuration < minChunkSeconds { break }

            // Pull the words from the remainder by name. We split on the
            // same boundary `align` used (whitespace) so words line up.
            let wordsAll = remainingText.split(separator: " ", omittingEmptySubsequences: true)
            guard plateauStart < wordsAll.count else { break }
            let nextText = wordsAll[plateauStart...].joined(separator: " ")

            progressHandler?(
                "Audio \(String(format: "%.1f", durationSec))s saturated after word \(plateauStart) "
                + "(\(String(format: "%.1f", splitTime))s); chunking remaining \(String(format: "%.1f", remainingDuration))s "
                + "(pass \(pass + 1))"
            )

            remainingAudio = nextAudio
            remainingText = nextText
            offsetSec += splitTime
            pass += 1
            if pass > 10 { break }  // belt-and-braces against pathological loops
        }

        return allAligned
    }

    static func offsetWords(_ words: [AlignedWord], by seconds: Float) -> [AlignedWord] {
        guard seconds != 0 else { return words }
        return words.map {
            AlignedWord(text: $0.text, startTime: $0.startTime + seconds, endTime: $0.endTime + seconds)
        }
    }

    /// Index of the first word in the trailing "stuck" plateau, or
    /// `aligned.count` if no plateau is detected.
    ///
    /// A plateau is `≥ minSize` consecutive trailing words whose start
    /// times differ by less than `tolerance`. This is the LIS-clamp
    /// signature: the model produced low/garbage indices for those
    /// positions and the monotonicity pass collapsed them onto the last
    /// reliable anchor.
    static func findTrailingPlateauStart(
        _ aligned: [AlignedWord], tolerance: Float, minSize: Int
    ) -> Int {
        let n = aligned.count
        guard n > minSize else { return n }
        // Walk backward: when `aligned[i].startTime ≈ aligned[i-1].startTime`,
        // both are in the plateau, so the plateau extends *to* index `i-1`.
        // Stop at the first big jump.
        var plateauStart = n
        for i in (1..<n).reversed() {
            let dt = abs(aligned[i].startTime - aligned[i - 1].startTime)
            if dt < tolerance {
                plateauStart = i - 1
            } else {
                break
            }
        }
        return (n - plateauStart) >= minSize ? plateauStart : n
    }

    /// Align text to audio, producing word-level timestamps.
    ///
    /// - Parameters:
    ///   - audio: Raw audio samples (mono)
    ///   - text: Text to align against the audio
    ///   - sampleRate: Sample rate of the audio (default 16000)
    ///   - language: Language hint for word splitting (default "English")
    /// - Returns: Array of words with start/end timestamps in seconds
    public func align(
        audio: [Float],
        text: String,
        sampleRate: Int = 16000,
        language: String = "English"
    ) -> [AlignedWord] {
        guard let tokenizer = tokenizer else {
            print("Error: tokenizer not loaded")
            return []
        }

        // 1. Extract mel features → audio encoder → audio embeddings
        let melFeatures = featureExtractor.process(audio, sampleRate: sampleRate)
        let batchedFeatures = melFeatures.expandedDimensions(axis: 0)
        var audioEmbeds = audioEncoder(batchedFeatures)
        audioEmbeds = audioEmbeds.expandedDimensions(axis: 0)  // [1, T_audio, hiddenSize]

        let numAudioTokens = audioEmbeds.dim(1)

        // 2. Prepare text with timestamp slots
        let slotted = TextPreprocessor.prepareForAlignment(
            text: text,
            tokenizer: tokenizer,
            language: language
        )

        guard !slotted.words.isEmpty else {
            print("Warning: no words found in text")
            return []
        }

        // 3. Build input_ids with chat template
        let inputIds = buildInputIds(
            slottedTokenIds: slotted.tokenIds,
            numAudioTokens: numAudioTokens,
            tokenizer: tokenizer,
            language: language
        )

        // Track where the slotted text starts in the full sequence
        let slottedTextStart = inputIds.count - slotted.tokenIds.count

        // 4. Embed all tokens and replace audio_pad with audio embeddings
        let inputIdsTensor = MLXArray(inputIds.map { Int32($0) }).expandedDimensions(axis: 0)
        var inputEmbeds = textDecoder.embeddings(for: inputIdsTensor)

        // Find audio_pad range and replace with audio embeddings
        let audioStartIndex = findAudioPadStart(inputIds)
        let audioEndIndex = audioStartIndex + numAudioTokens

        let audioEmbedsTyped = audioEmbeds.asType(inputEmbeds.dtype)
        let beforeAudio = inputEmbeds[0..., 0..<audioStartIndex, 0...]
        let afterAudio = inputEmbeds[0..., audioEndIndex..., 0...]
        inputEmbeds = concatenated([beforeAudio, audioEmbedsTyped, afterAudio], axis: 1)

        // 5. Single forward pass through decoder (no cache, no autoregressive loop)
        let (hiddenStates, _) = textDecoder.decode(inputsEmbeds: inputEmbeds, attentionMask: nil, cache: nil)

        // 6. Apply classify head to ALL hidden states → logits [1, seqLen, classifyNum]
        let logits = classifyHead(hiddenStates)

        // 7. Extract logits at timestamp positions → argmax → raw indices
        // Adjust timestamp positions to account for the chat template prefix
        let absoluteTimestampPositions = slotted.timestampPositions.map { $0 + slottedTextStart }

        var rawIndices: [Int] = []
        for pos in absoluteTimestampPositions {
            let posLogits = logits[0, pos, 0...]  // [classifyNum]
            let idx = argMax(posLogits).item(Int32.self)
            rawIndices.append(Int(idx))
        }

        // 8. Apply LIS monotonicity correction
        let correctedIndices = TimestampCorrection.enforceMonotonicity(rawIndices)

        // Optional raw/corrected dump for bug-triage of misaligned timestamps.
        if ProcessInfo.processInfo.environment["ALIGN_DEBUG"] == "1" {
            print("[align-debug] indices=\(rawIndices.count) numAudioTokens=\(numAudioTokens)")
            print("[align-debug] first 10 raw:       \(Array(rawIndices.prefix(10)))")
            print("[align-debug] first 10 corrected: \(Array(correctedIndices.prefix(10)))")
            print("[align-debug] last 60 raw:       \(Array(rawIndices.suffix(60)))")
            print("[align-debug] last 60 corrected: \(Array(correctedIndices.suffix(60)))")
        }

        // 9. Convert to seconds and pair as (start, end)
        let segmentTime = config.timestampSegmentTime
        var alignedWords: [AlignedWord] = []

        for (wordIdx, word) in slotted.words.enumerated() {
            let startIdx = wordIdx * 2       // even indices are start timestamps
            let endIdx = wordIdx * 2 + 1     // odd indices are end timestamps

            guard endIdx < correctedIndices.count else { break }

            let startTime = Float(correctedIndices[startIdx]) * segmentTime
            let endTime = Float(correctedIndices[endIdx]) * segmentTime

            alignedWords.append(AlignedWord(
                text: word,
                startTime: startTime,
                endTime: max(endTime, startTime)  // ensure end >= start
            ))
        }

        return alignedWords
    }

    // MARK: - Private Helpers

    /// Build full input_ids sequence with chat template
    private func buildInputIds(
        slottedTokenIds: [Int],
        numAudioTokens: Int,
        tokenizer: Qwen3Tokenizer,
        language: String
    ) -> [Int] {
        let imStartId = Qwen3ASRTokens.imStartTokenId
        let imEndId = Qwen3ASRTokens.imEndTokenId
        let audioStartId = Qwen3ASRTokens.audioStartTokenId
        let audioEndId = Qwen3ASRTokens.audioEndTokenId
        let audioPadId = Qwen3ASRTokens.audioTokenId
        let newlineId = 198

        // Token IDs for role names
        let systemId = 8948
        let userId = 872
        let assistantId = 77091

        var ids: [Int] = []

        // <|im_start|>system\n<|im_end|>\n
        ids.append(contentsOf: [imStartId, systemId, newlineId, imEndId, newlineId])

        // <|im_start|>user\n<|audio_start|>
        ids.append(contentsOf: [imStartId, userId, newlineId, audioStartId])

        // <|audio_pad|> * numAudioTokens
        for _ in 0..<numAudioTokens {
            ids.append(audioPadId)
        }

        // <|audio_end|><|im_end|>\n
        ids.append(contentsOf: [audioEndId, imEndId, newlineId])

        // <|im_start|>assistant\n
        ids.append(contentsOf: [imStartId, assistantId, newlineId])

        // Slotted text with timestamp tokens
        ids.append(contentsOf: slottedTokenIds)

        return ids
    }

    /// Find the start index of audio_pad tokens in input_ids
    private func findAudioPadStart(_ inputIds: [Int]) -> Int {
        let audioPadId = Qwen3ASRTokens.audioTokenId
        for (i, id) in inputIds.enumerated() {
            if id == audioPadId { return i }
        }
        return 0
    }
}

// MARK: - Model Loading

public extension Qwen3ForcedAligner {

    /// Load forced aligner model from HuggingFace hub
    static func fromPretrained(
        modelId: String = "aufklarer/Qwen3-ForcedAligner-0.6B-4bit",
        cacheDir: URL? = nil,
        offlineMode: Bool = false,
        progressHandler: ((Double, String) -> Void)? = nil
    ) async throws -> Qwen3ForcedAligner {
        progressHandler?(0.0, "Downloading model...")

        let cacheDir = try cacheDir ?? HuggingFaceDownloader.getCacheDirectory(for: modelId)

        // Download weights and tokenizer files
        try await HuggingFaceDownloader.downloadWeights(
            modelId: modelId,
            to: cacheDir,
            additionalFiles: ["vocab.json", "merges.txt", "tokenizer_config.json",
                              "quantize_config.json"],
            offlineMode: offlineMode,
            progressHandler: { progress in
                progressHandler?(progress * 0.8, "Downloading weights...")
            }
        )

        progressHandler?(0.80, "Loading tokenizer...")

        // Detect variant: try quantize_config.json first, fall back to model ID
        let variant: ForcedAlignerVariant
        let packaging = detectPackaging(in: cacheDir)
        if let detected = packaging {
            variant = detected
        } else if let detected = ForcedAlignerVariant.detect(from: modelId) {
            variant = detected
        } else {
            variant = .mlx4bit
        }

        let model = Qwen3ForcedAligner(
            audioConfig: .forcedAligner,
            textConfig: variant.textConfig,
            useFloatTextDecoder: variant.usesFloatTextDecoder
        )

        // Load tokenizer
        let vocabPath = cacheDir.appendingPathComponent("vocab.json")
        if FileManager.default.fileExists(atPath: vocabPath.path) {
            let tokenizer = Qwen3Tokenizer()
            try tokenizer.load(from: vocabPath)
            model.tokenizer = tokenizer
        }

        progressHandler?(0.85, "Loading audio encoder weights...")

        // Load weights
        try WeightLoader.loadForcedAlignerWeights(into: model, from: cacheDir)

        progressHandler?(1.0, "Ready")

        return model
    }

    /// Detect model variant from quantize_config.json
    private static func detectPackaging(in cacheDir: URL) -> ForcedAlignerVariant? {
        struct QuantizationFile: Decodable {
            struct Quantization: Decodable {
                let bits: Int?
                let groupSize: Int?
                enum CodingKeys: String, CodingKey {
                    case bits
                    case groupSize = "group_size"
                }
            }
            let quantization: Quantization?
        }

        let configPath = cacheDir.appendingPathComponent("quantize_config.json")
        guard let data = try? Data(contentsOf: configPath),
              let file = try? JSONDecoder().decode(QuantizationFile.self, from: data),
              let quant = file.quantization,
              let bits = quant.bits else {
            return nil
        }

        switch bits {
        case 0: return .bf16
        case 4: return .mlx4bit
        case 8: return .mlx8bit
        default: return nil
        }
    }
}
