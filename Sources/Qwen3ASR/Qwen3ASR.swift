import Foundation
import MLX
import MLXNN
import MLXFast

/// Special token IDs for Qwen3-ASR
public struct Qwen3ASRTokens {
    public static let audioTokenId = 151676        // <|audio_pad|>
    public static let audioStartTokenId = 151669   // <|audio_start|>
    public static let audioEndTokenId = 151670     // <|audio_end|>
    public static let eosTokenId = 151645          // <|im_end|>
    public static let padTokenId = 151643          // <|endoftext|>
    public static let imStartTokenId = 151644      // <|im_start|>
    public static let imEndTokenId = 151645        // <|im_end|>
}

/// Main Qwen3-ASR model for speech recognition
public class Qwen3ASRModel {
    public let audioEncoder: Qwen3AudioEncoder
    public let featureExtractor: WhisperFeatureExtractor
    public var textDecoder: QuantizedTextModel?

    /// Tokenizer for decoding output tokens
    private var tokenizer: Qwen3Tokenizer?

    /// Text decoder config
    public let textConfig: TextDecoderConfig

    public init(
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
    public func setTokenizer(_ tokenizer: Qwen3Tokenizer) {
        self.tokenizer = tokenizer
    }

    /// Initialize text decoder (called after loading)
    public func initializeTextDecoder() {
        self.textDecoder = QuantizedTextModel(config: textConfig)
    }

    /// Transcribe audio to text
    /// - Parameters:
    ///   - audio: Float audio samples
    ///   - sampleRate: Sample rate of input audio (default 24000)
    ///   - language: Target output language (nil = auto-detect and transcribe in source language)
    ///   - maxTokens: Maximum tokens to generate
    public func transcribe(
        audio: [Float],
        sampleRate: Int = 24000,
        language: String? = nil,
        maxTokens: Int = 448
    ) -> String {
        // Extract mel features
        let melFeatures = featureExtractor.process(audio, sampleRate: sampleRate)

        // Debug mel features
        let melFlat = melFeatures.flattened()
        print("DEBUG: Mel features shape: \(melFeatures.shape)")
        print("DEBUG: Mel features - mean: \(mean(melFlat).item(Float.self)), std: \(sqrt(variance(melFlat)).item(Float.self))")

        // Add batch dimension: [mel, time] -> [1, mel, time]
        let batchedFeatures = melFeatures.expandedDimensions(axis: 0)

        // Encode audio - returns [time, features] without batch dim (matching Python)
        var audioEmbeds = audioEncoder(batchedFeatures)

        // Add batch dimension for consistency: [time, features] -> [1, time, features]
        audioEmbeds = audioEmbeds.expandedDimensions(axis: 0)

        // Debug audio embeddings
        let embedsFlat = audioEmbeds.flattened()
        print("DEBUG: Audio embeds shape: \(audioEmbeds.shape)")
        print("DEBUG: Audio embeds - mean: \(mean(embedsFlat).item(Float.self)), std: \(sqrt(variance(embedsFlat)).item(Float.self))")
        print("DEBUG: Audio embeds - min: \(min(embedsFlat).item(Float.self)), max: \(max(embedsFlat).item(Float.self))")

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
            maxTokens: maxTokens
        )
    }

    /// Generate text from audio embeddings
    /// - Parameters:
    ///   - audioEmbeds: Audio embeddings from encoder
    ///   - textDecoder: Text decoder model
    ///   - language: Target language (nil = let model auto-detect and transcribe in source language)
    ///   - maxTokens: Maximum tokens to generate
    private func generateText(
        audioEmbeds: MLXArray,
        textDecoder: QuantizedTextModel,
        language: String?,
        maxTokens: Int
    ) -> String {
        // Qwen3-ASR prompt format (from mlx-audio implementation):
        // <|im_start|>system\n<|im_end|>\n
        // <|im_start|>user\n<|audio_start|><|audio_pad|>...<|audio_end|><|im_end|>\n
        // <|im_start|>assistant\n[language X<asr_text>] <- model generates this if not specified
        //
        // If language is specified: <|im_start|>assistant\nlanguage {lang}<asr_text>
        // If language is nil: <|im_start|>assistant\n (let model output "language X<asr_text>...")

        // Special token IDs
        let imStartId = 151644      // <|im_start|>
        let imEndId = 151645        // <|im_end|>
        let audioStartId = 151669   // <|audio_start|>
        let audioEndId = 151670     // <|audio_end|>
        let audioPadId = 151676     // <|audio_pad|> - placeholder for audio embeddings
        let asrTextId = 151704      // <asr_text>
        let newlineId = 198         // \n

        // Token IDs for "system", "user", "assistant"
        // Verified from vocab.json via tokenizer.debugTokenMappings()
        let systemId = 8948        // "system"
        let userId = 872           // "user"
        let assistantId = 77091    // "assistant"

        // Number of audio tokens (from audio encoder output)
        let numAudioTokens = audioEmbeds.dim(1)

        // Build input_ids array with audio_pad placeholder tokens
        var inputIds: [Int32] = []

        // <|im_start|>system\n<|im_end|>\n
        inputIds.append(contentsOf: [imStartId, systemId, newlineId, imEndId, newlineId].map { Int32($0) })

        // <|im_start|>user\n<|audio_start|>
        inputIds.append(contentsOf: [imStartId, userId, newlineId, audioStartId].map { Int32($0) })

        // <|audio_pad|> * numAudioTokens (placeholder tokens that will be replaced)
        let audioStartIndex = inputIds.count
        for _ in 0..<numAudioTokens {
            inputIds.append(Int32(audioPadId))
        }
        let audioEndIndex = inputIds.count

        // <|audio_end|><|im_end|>\n
        inputIds.append(contentsOf: [audioEndId, imEndId, newlineId].map { Int32($0) })

        // <|im_start|>assistant\n
        inputIds.append(contentsOf: [imStartId, assistantId, newlineId].map { Int32($0) })

        // Language handling:
        // - If language specified: add "language {lang}<asr_text>" - model outputs in that language
        // - If language is nil: don't add anything - model will generate "language X<asr_text>..."
        if let lang = language, let tokenizer = tokenizer {
            // Tokenize "language {lang}" and add <asr_text>
            let langPrefix = "language \(lang)"
            let langTokens = tokenizer.encode(langPrefix)
            inputIds.append(contentsOf: langTokens.map { Int32($0) })
            inputIds.append(Int32(asrTextId))
            print("DEBUG: Forcing language: \(lang)")
        } else {
            // Auto-detect: don't add anything after "assistant\n"
            // Model will generate: "language Russian<asr_text>Привет" (for example)
            print("DEBUG: Auto-detect mode - model generates full language tag + text")
        }

        print("DEBUG: Input IDs length: \(inputIds.count), audio tokens: \(numAudioTokens) at positions \(audioStartIndex)..<\(audioEndIndex)")

        // Get text embeddings for all tokens
        let inputIdsTensor = MLXArray(inputIds).expandedDimensions(axis: 0)  // [1, seq_len]
        var inputEmbeds = textDecoder.embedTokens(inputIdsTensor)  // [1, seq_len, hidden]

        // Debug: compare embedding scales
        let textEmbedsFlat = inputEmbeds.flattened()
        let textStd = sqrt(variance(textEmbedsFlat)).item(Float.self)
        print("DEBUG: Text embeds - mean: \(mean(textEmbedsFlat).item(Float.self)), std: \(textStd)")

        let audioEmbedsFlat = audioEmbeds.flattened()
        let audioStd = sqrt(variance(audioEmbedsFlat)).item(Float.self)
        print("DEBUG: Audio embeds (before scaling) - mean: \(mean(audioEmbedsFlat).item(Float.self)), std: \(audioStd)")

        // Scale audio embeddings to match text embedding variance
        var scaledAudioEmbeds = audioEmbeds
        if audioStd > 0 {
            let scaleFactor = textStd / audioStd
            scaledAudioEmbeds = audioEmbeds * scaleFactor
            print("DEBUG: Scaling audio embeds by \(scaleFactor) to match text std")
        }

        // Replace audio_pad token positions with actual audio embeddings
        // audioEmbeds shape: [1, numAudioTokens, hidden]
        // We need to replace inputEmbeds[0, audioStartIndex:audioEndIndex, :] with audioEmbeds[0, :, :]
        let hiddenSize = inputEmbeds.dim(2)

        // Build the final embeddings by concatenating parts
        let beforeAudio = inputEmbeds[0..., 0..<audioStartIndex, 0...]  // [1, audioStartIndex, hidden]
        let afterAudio = inputEmbeds[0..., audioEndIndex..., 0...]  // [1, remaining, hidden]

        inputEmbeds = concatenated([beforeAudio, scaledAudioEmbeds, afterAudio], axis: 1)

        print("DEBUG: Prompt structure - before_audio:\(audioStartIndex), audio:\(numAudioTokens), after_audio:\(inputIds.count - audioEndIndex)")

        // Initialize KV cache
        var cache: [(MLXArray, MLXArray)]? = nil

        // Generate tokens
        var generatedTokens: [Int32] = []

        // First pass: process the full input embeddings
        var (hiddenStates, newCache) = textDecoder(inputsEmbeds: inputEmbeds, cache: cache)
        cache = newCache

        // Get logits from the last position using embedding as LM head (tied weights)
        // hiddenStates shape: [1, seq_len, hidden]
        let seqLen = hiddenStates.dim(1)
        let lastHidden = hiddenStates[0..., (seqLen-1)..<seqLen, 0...]  // [1, 1, hidden]
        var logits = textDecoder.embedTokens.asLinear(lastHidden)
        var nextToken = argMax(logits, axis: -1).squeezed().item(Int32.self)
        generatedTokens.append(nextToken)

        // Debug: check logits stats (using MLX ops instead of slow loop)
        let logitsFlat = logits.squeezed()
        print("DEBUG: Logits shape: \(logitsFlat.shape)")
        print("DEBUG: Logits stats - mean: \(mean(logitsFlat).item(Float.self)), max: \(max(logitsFlat).item(Float.self)), min: \(min(logitsFlat).item(Float.self))")

        print("DEBUG: First token generated: \(nextToken) (EOS=\(Qwen3ASRTokens.eosTokenId))")
        print("DEBUG: Input embeds shape: \(inputEmbeds.shape), hidden states shape: \(hiddenStates.shape)")

        // Continue generating
        for iteration in 1..<maxTokens {
            // Check for EOS
            if nextToken == Int32(Qwen3ASRTokens.eosTokenId) {
                print("DEBUG: EOS token reached at iteration \(iteration)")
                break
            }

            // Get embedding for the new token
            let tokenEmbeds = textDecoder.embedTokens(MLXArray([nextToken]).expandedDimensions(axis: 0))

            // Forward pass with cache
            (hiddenStates, newCache) = textDecoder(inputsEmbeds: tokenEmbeds, cache: cache)
            cache = newCache

            // Get next token
            let lastHiddenNext = hiddenStates[0..., (-1)..., .ellipsis]
            logits = textDecoder.embedTokens.asLinear(lastHiddenNext)
            nextToken = argMax(logits, axis: -1).squeezed().item(Int32.self)
            generatedTokens.append(nextToken)
        }

        // Debug: print first few generated tokens
        print("DEBUG: Generated \(generatedTokens.count) tokens: \(Array(generatedTokens.prefix(20)))")

        // Decode tokens to text
        if let tokenizer = tokenizer {
            return tokenizer.decode(tokens: generatedTokens.map { Int($0) })
        } else {
            // Fallback: return token IDs
            return generatedTokens.map { String($0) }.joined(separator: " ")
        }
    }
}

// MARK: - Model Loading

public extension Qwen3ASRModel {
    /// Load model from HuggingFace hub with automatic weight downloading
    static func fromPretrained(
        modelId: String = "mlx-community/Qwen3-ASR-0.6B-4bit",
        progressHandler: ((Double, String) -> Void)? = nil
    ) async throws -> Qwen3ASRModel {
        progressHandler?(0.1, "Downloading model...")

        // Get cache directory
        let cacheDir = getCacheDirectory(for: modelId)

        // Download weights if needed
        if !weightsExist(in: cacheDir) {
            try await downloadWeights(modelId: modelId, to: cacheDir, progressHandler: { progress in
                progressHandler?(0.1 + progress * 0.4, "Downloading weights...")
            })
        }

        progressHandler?(0.5, "Loading tokenizer...")

        // Create model with default config
        let model = Qwen3ASRModel()

        // Load tokenizer from vocab.json
        let vocabPath = cacheDir.appendingPathComponent("vocab.json")
        if FileManager.default.fileExists(atPath: vocabPath.path) {
            let tokenizer = Qwen3Tokenizer()
            try tokenizer.load(from: vocabPath)
            tokenizer.debugTokenMappings()  // Debug: verify token IDs
            model.setTokenizer(tokenizer)
        }

        progressHandler?(0.6, "Loading audio encoder weights...")

        // Load audio encoder weights
        try WeightLoader.loadWeights(into: model.audioEncoder, from: cacheDir)

        progressHandler?(0.75, "Loading text decoder weights...")

        // Initialize and load text decoder
        model.initializeTextDecoder()
        if let textDecoder = model.textDecoder {
            try WeightLoader.loadTextDecoderWeights(into: textDecoder, from: cacheDir)
        }

        progressHandler?(1.0, "Ready")

        return model
    }

    private static func getCacheDirectory(for modelId: String) -> URL {
        let cacheDir = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first!
            .appendingPathComponent("qwen3-asr")
            .appendingPathComponent(modelId.replacingOccurrences(of: "/", with: "_"))

        try? FileManager.default.createDirectory(at: cacheDir, withIntermediateDirectories: true)
        return cacheDir
    }

    private static func weightsExist(in directory: URL) -> Bool {
        let fm = FileManager.default
        let contents = (try? fm.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)) ?? []
        return contents.contains { $0.pathExtension == "safetensors" }
    }

    private static func downloadWeights(
        modelId: String,
        to directory: URL,
        progressHandler: ((Double) -> Void)?
    ) async throws {
        let baseURL = "https://huggingface.co/\(modelId)/resolve/main"

        // Files to download (config and tokenizer)
        var filesToDownload = [
            "config.json",
            "vocab.json",
            "tokenizer_config.json"
        ]

        // Determine model file(s) to download
        let indexPath = directory.appendingPathComponent("model.safetensors.index.json")

        if !FileManager.default.fileExists(atPath: indexPath.path) {
            let indexURL = URL(string: "\(baseURL)/model.safetensors.index.json")!
            if let (indexData, indexResponse) = try? await URLSession.shared.data(from: indexURL),
               let httpResponse = indexResponse as? HTTPURLResponse,
               httpResponse.statusCode == 200 {
                try indexData.write(to: indexPath)
            }
        }

        // Check if we have an index file and get model files from it
        var modelFiles: [String] = []
        if FileManager.default.fileExists(atPath: indexPath.path),
           let indexData = try? Data(contentsOf: indexPath),
           let index = try? JSONSerialization.jsonObject(with: indexData) as? [String: Any],
           let weightMap = index["weight_map"] as? [String: String] {
            let uniqueFiles = Set(weightMap.values)
            modelFiles = Array(uniqueFiles).sorted()
            print("Found \(modelFiles.count) model file(s) from index: \(modelFiles)")
        } else {
            modelFiles = ["model.safetensors"]
        }

        filesToDownload.append(contentsOf: modelFiles)

        for (index, file) in filesToDownload.enumerated() {
            let localPath = directory.appendingPathComponent(file)

            if FileManager.default.fileExists(atPath: localPath.path) {
                progressHandler?(Double(index + 1) / Double(filesToDownload.count))
                continue
            }

            print("Downloading: \(file)")

            let url = URL(string: "\(baseURL)/\(file)")!
            let (data, response) = try await URLSession.shared.data(from: url)

            guard let httpResponse = response as? HTTPURLResponse,
                  httpResponse.statusCode == 200 else {
                throw DownloadError.failedToDownload(file)
            }

            try data.write(to: localPath)

            progressHandler?(Double(index + 1) / Double(filesToDownload.count))
        }
    }
}

// MARK: - Errors

public enum DownloadError: Error, LocalizedError {
    case failedToDownload(String)

    public var errorDescription: String? {
        switch self {
        case .failedToDownload(let file):
            return "Failed to download: \(file)"
        }
    }
}
