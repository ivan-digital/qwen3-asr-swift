import Foundation
import MLX
import MLXNN
import MLXFast
import Qwen3Common

/// A word with its aligned start and end timestamps
public struct AlignedWord: Sendable {
    public let text: String
    public let startTime: Float   // seconds
    public let endTime: Float     // seconds

    public init(text: String, startTime: Float, endTime: Float) {
        self.text = text
        self.startTime = startTime
        self.endTime = endTime
    }
}

/// Qwen3 Forced Aligner — predicts word-level timestamps for audio+text pairs.
///
/// Uses the same encoder-decoder architecture as Qwen3-ASR but replaces the
/// vocab lm_head with a 5000-class timestamp classification head.
/// Inference is non-autoregressive (single forward pass).
public class Qwen3ForcedAligner {
    public let audioEncoder: Qwen3AudioEncoder
    public let textDecoder: QuantizedTextModel
    public let classifyHead: Linear
    public let featureExtractor: WhisperFeatureExtractor
    public var tokenizer: Qwen3Tokenizer?

    private let config: Qwen3ASRConfig

    public init(
        audioConfig: Qwen3AudioEncoderConfig = .forcedAligner,
        textConfig: TextDecoderConfig = .small,
        classifyNum: Int = 5000
    ) {
        self.audioEncoder = Qwen3AudioEncoder(config: audioConfig)
        self.textDecoder = QuantizedTextModel(config: textConfig)
        self.classifyHead = Linear(textConfig.hiddenSize, classifyNum)
        self.featureExtractor = WhisperFeatureExtractor()

        var cfg = Qwen3ASRConfig()
        cfg.classifyNum = classifyNum
        self.config = cfg
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
        var inputEmbeds = textDecoder.embedTokens(inputIdsTensor)

        // Find audio_pad range and replace with audio embeddings
        let audioStartIndex = findAudioPadStart(inputIds)
        let audioEndIndex = audioStartIndex + numAudioTokens

        let audioEmbedsTyped = audioEmbeds.asType(inputEmbeds.dtype)
        let beforeAudio = inputEmbeds[0..., 0..<audioStartIndex, 0...]
        let afterAudio = inputEmbeds[0..., audioEndIndex..., 0...]
        inputEmbeds = concatenated([beforeAudio, audioEmbedsTyped, afterAudio], axis: 1)

        // 5. Single forward pass through decoder (no cache, no autoregressive loop)
        let (hiddenStates, _) = textDecoder(inputsEmbeds: inputEmbeds, cache: nil)

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
        progressHandler: ((Double, String) -> Void)? = nil
    ) async throws -> Qwen3ForcedAligner {
        progressHandler?(0.0, "Downloading model...")

        let cacheDir = try HuggingFaceDownloader.getCacheDirectory(for: modelId)

        // Download weights and tokenizer files
        try await HuggingFaceDownloader.downloadWeights(
            modelId: modelId,
            to: cacheDir,
            additionalFiles: ["vocab.json", "merges.txt", "tokenizer_config.json"],
            progressHandler: { progress in
                progressHandler?(progress * 0.8, "Downloading weights...")
            }
        )

        progressHandler?(0.80, "Loading tokenizer...")

        // Detect quantization from model ID
        let textConfig: TextDecoderConfig
        if modelId.contains("4bit") {
            var cfg = TextDecoderConfig.small
            cfg.bits = 4
            cfg.groupSize = 64
            textConfig = cfg
        } else if modelId.contains("8bit") {
            var cfg = TextDecoderConfig.small
            cfg.bits = 8
            cfg.groupSize = 64
            textConfig = cfg
        } else {
            // bf16 — still use quantized text model structure but with default config
            textConfig = .small
        }

        let model = Qwen3ForcedAligner(
            audioConfig: .forcedAligner,
            textConfig: textConfig
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
}
