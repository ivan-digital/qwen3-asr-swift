#if canImport(CoreML)
import CoreML
import Foundation
import MLX
import AudioCommon

/// Full CoreML ASR model: CoreML encoder + CoreML text decoder.
///
/// Runs the entire Qwen3-ASR pipeline on CoreML (Neural Engine + CPU),
/// eliminating the MLX GPU dependency. Requires macOS 15+ / iOS 18+
/// for MLState KV cache support.
@available(macOS 15, iOS 18, *)
public class CoreMLASRModel {
    public let encoder: CoreMLASREncoder
    public let decoder: CoreMLTextDecoder
    public let featureExtractor: WhisperFeatureExtractor
    private var tokenizer: Qwen3Tokenizer?

    public init(encoder: CoreMLASREncoder, decoder: CoreMLTextDecoder) {
        self.encoder = encoder
        self.decoder = decoder
        self.featureExtractor = WhisperFeatureExtractor()
    }

    /// Load full CoreML ASR from HuggingFace.
    ///
    /// Downloads encoder and decoder models from `aufklarer/Qwen3-ASR-CoreML`.
    public static func fromPretrained(
        encoderModelId: String = CoreMLASREncoder.defaultModelId,
        decoderModelId: String = CoreMLASREncoder.defaultModelId,
        tokenizerModelId: String = "aufklarer/Qwen3-ASR-0.6B-MLX-4bit",
        computeUnits: MLComputeUnits = .all,
        progressHandler: ((Double, String) -> Void)? = nil
    ) async throws -> CoreMLASRModel {
        // Download encoder (0-30%)
        progressHandler?(0.0, "Loading CoreML encoder...")
        let enc = try await CoreMLASREncoder.fromPretrained(
            modelId: encoderModelId,
            computeUnits: computeUnits
        ) { p, msg in
            progressHandler?(p * 0.3, msg)
        }

        // Download decoder (30-80%)
        progressHandler?(0.3, "Loading CoreML decoder...")
        let dec = try await CoreMLTextDecoder.fromPretrained(
            modelId: decoderModelId,
            computeUnits: computeUnits
        ) { p, msg in
            progressHandler?(0.3 + p * 0.5, msg)
        }

        // Download tokenizer (80-90%)
        progressHandler?(0.8, "Loading tokenizer...")
        let tokenizerDir = try HuggingFaceDownloader.getCacheDirectory(for: tokenizerModelId)
        try await HuggingFaceDownloader.downloadWeights(
            modelId: tokenizerModelId,
            to: tokenizerDir,
            additionalFiles: ["vocab.json", "merges.txt", "tokenizer_config.json"]
        )

        let model = CoreMLASRModel(encoder: enc, decoder: dec)

        let vocabPath = tokenizerDir.appendingPathComponent("vocab.json")
        if FileManager.default.fileExists(atPath: vocabPath.path) {
            let tokenizer = Qwen3Tokenizer()
            try tokenizer.load(from: vocabPath)
            model.tokenizer = tokenizer
        }

        progressHandler?(1.0, "Ready")
        return model
    }

    /// Warm up both encoder and decoder.
    public func warmUp() throws {
        try encoder.warmUp()
        try decoder.warmUp()
    }

    /// Transcribe audio to text using full CoreML pipeline.
    ///
    /// The entire inference runs on CoreML (Neural Engine + CPU) without MLX GPU.
    public func transcribe(
        audio: [Float],
        sampleRate: Int = 16000,
        language: String? = nil,
        maxTokens: Int = 448
    ) throws -> String {
        // Extract mel features
        let melFeatures = featureExtractor.process(audio, sampleRate: sampleRate)

        // Encode audio → embeddings [1, T/8, 1024]
        let audioEmbeds = try encoder.encode(melFeatures)
        let numAudioTokens = audioEmbeds.dim(1)

        // Reset decoder KV cache
        decoder.resetCache()

        // Build chat template token sequence
        let imStartId: Int32 = 151644
        let imEndId: Int32 = 151645
        let audioStartId: Int32 = 151669
        let audioEndId: Int32 = 151670
        let asrTextId: Int32 = 151704
        let newlineId: Int32 = 198
        let systemId: Int32 = 8948
        let userId: Int32 = 872
        let assistantId: Int32 = 77091

        // <|im_start|>system\n<|im_end|>\n
        var prefixTokens: [Int32] = [imStartId, systemId, newlineId, imEndId, newlineId]
        // <|im_start|>user\n<|audio_start|>
        prefixTokens += [imStartId, userId, newlineId, audioStartId]

        // <|audio_end|><|im_end|>\n<|im_start|>assistant\n
        var suffixTokens: [Int32] = [audioEndId, imEndId, newlineId, imStartId, assistantId, newlineId]

        // Language hint + <|asr_text|>
        if let lang = language, let tokenizer = tokenizer {
            let langPrefix = "language \(lang)"
            let langTokens = tokenizer.encode(langPrefix)
            suffixTokens += langTokens.map { Int32($0) }
        }
        suffixTokens.append(asrTextId)

        // ── Prefill: process all prefix tokens ──
        var lastLogits: MLMultiArray?

        for token in prefixTokens {
            let embedding = try decoder.embed(tokenId: token)
            lastLogits = try decoder.decoderStep(embedding: embedding)
        }

        // ── Prefill: process audio embeddings ──
        for i in 0..<numAudioTokens {
            let audioEmbed = try decoder.audioEmbeddingToMultiArray(audioEmbeds, at: i)
            lastLogits = try decoder.decoderStep(embedding: audioEmbed)
        }

        // ── Prefill: process suffix tokens ──
        for token in suffixTokens {
            let embedding = try decoder.embed(tokenId: token)
            lastLogits = try decoder.decoderStep(embedding: embedding)
        }

        // ── Autoregressive generation ──
        guard var logits = lastLogits else {
            return "[CoreML decoder: no output]"
        }

        var generatedTokens: [Int32] = []
        var nextToken = decoder.argmax(logits: logits)
        generatedTokens.append(nextToken)

        for _ in 1..<maxTokens {
            if nextToken == imEndId { break }

            let embedding = try decoder.embed(tokenId: nextToken)
            logits = try decoder.decoderStep(embedding: embedding)
            nextToken = decoder.argmax(logits: logits)
            generatedTokens.append(nextToken)
        }

        // Decode tokens
        if let tokenizer = tokenizer {
            let rawText = tokenizer.decode(tokens: generatedTokens.map { Int($0) })
            if let range = rawText.range(of: "<asr_text>") {
                return String(rawText[range.upperBound...]).trimmingCharacters(in: .whitespaces)
            }
            return rawText
        } else {
            return generatedTokens.map { String($0) }.joined(separator: " ")
        }
    }

    // MARK: - MLX-Free Transcription

    /// Transcribe audio to text without any MLX/Metal dependency.
    ///
    /// Uses `featureExtractor.processRaw()` (CPU via Accelerate) and
    /// `encoder.encode(melData:melBins:timeFrames:)` (CoreML) to produce
    /// MLMultiArray embeddings, then decodes using `audioEmbeddingFromMultiArray()`.
    ///
    /// This method is safe for iOS background execution where Metal GPU eval
    /// (triggered by MLXArray operations) would cause a crash.
    ///
    /// - Note: Requires `processRaw()` on WhisperFeatureExtractor and
    ///   `encode(melData:melBins:timeFrames:)` on CoreMLASREncoder, both added by T2.
    public func transcribeWithoutMLX(
        audio: [Float],
        sampleRate: Int = 16000,
        language: String? = nil,
        maxTokens: Int = 448
    ) throws -> String {
        // 1. Extract mel features (pure CPU via Accelerate — no MLXArray)
        let melFeatures = featureExtractor.processRaw(audio, sampleRate: sampleRate)

        // 2. Encode audio → MLMultiArray embeddings [1, T/8, 1024]
        let audioEmbeds = try encoder.encode(
            melData: melFeatures.data,
            melBins: melFeatures.melBins,
            timeFrames: melFeatures.timeFrames
        )
        let numAudioTokens = audioEmbeds.shape[1].intValue

        // 3. Reset decoder KV cache
        decoder.resetCache()

        // 4. Build chat template token sequence (identical to transcribe())
        let imStartId: Int32 = 151644
        let imEndId: Int32 = 151645
        let audioStartId: Int32 = 151669
        let audioEndId: Int32 = 151670
        let asrTextId: Int32 = 151704
        let newlineId: Int32 = 198
        let systemId: Int32 = 8948
        let userId: Int32 = 872
        let assistantId: Int32 = 77091

        // <|im_start|>system\n<|im_end|>\n
        var prefixTokens: [Int32] = [imStartId, systemId, newlineId, imEndId, newlineId]
        // <|im_start|>user\n<|audio_start|>
        prefixTokens += [imStartId, userId, newlineId, audioStartId]

        // <|audio_end|><|im_end|>\n<|im_start|>assistant\n
        var suffixTokens: [Int32] = [audioEndId, imEndId, newlineId, imStartId, assistantId, newlineId]

        // Language hint + <|asr_text|>
        if let lang = language, let tokenizer = tokenizer {
            let langPrefix = "language \(lang)"
            let langTokens = tokenizer.encode(langPrefix)
            suffixTokens += langTokens.map { Int32($0) }
        }
        suffixTokens.append(asrTextId)

        // 5. Prefill: process all prefix tokens
        var lastLogits: MLMultiArray?

        for token in prefixTokens {
            let embedding = try decoder.embed(tokenId: token)
            lastLogits = try decoder.decoderStep(embedding: embedding)
        }

        // 6. Prefill: process audio embeddings (MLX-free path)
        for i in 0..<numAudioTokens {
            let audioEmbed = try decoder.audioEmbeddingFromMultiArray(audioEmbeds, at: i)
            lastLogits = try decoder.decoderStep(embedding: audioEmbed)
        }

        // Prefill: process suffix tokens
        for token in suffixTokens {
            let embedding = try decoder.embed(tokenId: token)
            lastLogits = try decoder.decoderStep(embedding: embedding)
        }

        // 7. Autoregressive generation
        guard var logits = lastLogits else {
            return "[CoreML decoder: no output]"
        }

        var generatedTokens: [Int32] = []
        var nextToken = decoder.argmax(logits: logits)
        generatedTokens.append(nextToken)

        for _ in 1..<maxTokens {
            if nextToken == imEndId { break }

            let embedding = try decoder.embed(tokenId: nextToken)
            logits = try decoder.decoderStep(embedding: embedding)
            nextToken = decoder.argmax(logits: logits)
            generatedTokens.append(nextToken)
        }

        // Decode tokens
        if let tokenizer = tokenizer {
            let rawText = tokenizer.decode(tokens: generatedTokens.map { Int($0) })
            if let range = rawText.range(of: "<asr_text>") {
                return String(rawText[range.upperBound...]).trimmingCharacters(in: .whitespaces)
            }
            return rawText
        } else {
            return generatedTokens.map { String($0) }.joined(separator: " ")
        }
    }
}

// MARK: - SpeechRecognitionModel

@available(macOS 15, iOS 18, *)
extension CoreMLASRModel: SpeechRecognitionModel {
    public var inputSampleRate: Int { 16000 }

    public func transcribe(audio: [Float], sampleRate: Int, language: String?) -> String {
        do {
            return try transcribe(audio: audio, sampleRate: sampleRate, language: language, maxTokens: 448)
        } catch {
            return "[CoreML error: \(error.localizedDescription)]"
        }
    }
}

// MARK: - Background-Safe Transcription

@available(macOS 15, iOS 18, *)
extension CoreMLASRModel {
    /// Background-safe transcription (no MLX/Metal dependency).
    ///
    /// Uses `transcribeWithoutMLX()` which avoids all MLXArray operations
    /// that would trigger Metal GPU eval. Safe to call from iOS background
    /// audio processing where GPU access is prohibited.
    public func transcribeBackgroundSafe(audio: [Float], sampleRate: Int, language: String?) -> String {
        do {
            return try transcribeWithoutMLX(audio: audio, sampleRate: sampleRate, language: language)
        } catch {
            return "[CoreML error: \(error.localizedDescription)]"
        }
    }
}
#endif
