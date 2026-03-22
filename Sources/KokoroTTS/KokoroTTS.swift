import CoreML
import Foundation
import AudioCommon

/// Kokoro-82M text-to-speech — CoreML-based, runs on Neural Engine.
///
/// Lightweight (82M params) non-autoregressive TTS model.
/// Supports 8 languages with 50 preset voices. Designed for iOS/iPad deployment.
///
/// Uses pre-converted CoreML models from aufklarer/Kokoro-82M-CoreML.
///
/// ```swift
/// let tts = try await KokoroTTSModel.fromPretrained()
/// let audio = try tts.synthesize(text: "Hello world", voice: "af_heart")
/// ```
public final class KokoroTTSModel {

    /// Default HuggingFace model ID.
    public static let defaultModelId = "aufklarer/Kokoro-82M-CoreML"

    /// iOS-optimized INT8 palettized variant (single 5s bucket, ~89MB total).
    public static let int8iOSModelId = "aufklarer/Kokoro-82M-CoreML-INT8"

    /// Output sample rate (24kHz).
    public static let outputSampleRate = 24000

    /// Model configuration.
    public let config: KokoroConfig

    /// Whether the model is loaded and ready for inference.
    var _isLoaded = true

    var network: KokoroNetwork?
    private let phonemizer: KokoroPhonemizer
    var voiceEmbeddings: [String: [Float]]

    init(
        config: KokoroConfig,
        network: KokoroNetwork,
        phonemizer: KokoroPhonemizer,
        voiceEmbeddings: [String: [Float]]
    ) {
        self.config = config
        self.network = network
        self.phonemizer = phonemizer
        self.voiceEmbeddings = voiceEmbeddings
    }

    // MARK: - Synthesis

    /// Synthesize speech from text.
    ///
    /// - Parameters:
    ///   - text: Input text to speak
    ///   - voice: Voice preset name (default: "af_heart")
    ///   - language: Language code (default: "en")
    /// - Returns: Audio samples at 24kHz, Float32
    public func synthesize(
        text: String,
        voice: String = "af_heart",
        language: String = "en"
    ) throws -> [Float] {
        guard _isLoaded, let network else {
            throw AudioModelError.inferenceFailed(
                operation: "kokoro-synthesize", reason: "Model not loaded")
        }

        // Step 1: Phonemize text → token IDs
        let tokenIds = phonemizer.tokenize(text, maxLength: config.maxPhonemeLength)

        // Step 2: Select model bucket based on token count and loaded models
        let available = Set(network.availableBuckets)
        guard let activeBucket = ModelBucket.select(
            forTokenCount: tokenIds.count, available: available
        ) else {
            throw AudioModelError.inferenceFailed(
                operation: "kokoro-synthesize",
                reason: "No suitable model bucket for \(tokenIds.count) tokens "
                    + "(available: \(available.map(\.modelName)))")
        }

        // Step 3: Get voice style embedding (256-dim)
        guard let styleVector = voiceEmbeddings[voice] else {
            let available = Array(voiceEmbeddings.keys).sorted().prefix(5)
            throw AudioModelError.voiceNotFound(
                voice: voice,
                searchPath: "Available: \(available.joined(separator: ", "))...")
        }

        // Step 4: Pad tokens to bucket's max length
        let maxTokens = activeBucket.maxTokens
        let paddedIds = phonemizer.pad(tokenIds, to: maxTokens)

        // Step 5: Create MLMultiArray inputs
        let inputIds = try MLMultiArray(shape: [1, maxTokens as NSNumber], dataType: .int32)
        let maskArray = try MLMultiArray(shape: [1, maxTokens as NSNumber], dataType: .int32)
        let inputPtr = inputIds.dataPointer.assumingMemoryBound(to: Int32.self)
        let maskPtr = maskArray.dataPointer.assumingMemoryBound(to: Int32.self)
        for i in 0..<maxTokens {
            inputPtr[i] = Int32(paddedIds[i])
            maskPtr[i] = (i < tokenIds.count) ? 1 : 0
        }

        let refS = try MLMultiArray(shape: [1, config.styleDim as NSNumber], dataType: .float32)
        let refPtr = refS.dataPointer.assumingMemoryBound(to: Float.self)
        for i in 0..<min(styleVector.count, config.styleDim) {
            refPtr[i] = styleVector[i]
        }

        let randomPhases = try MLMultiArray(shape: [1, config.numPhases as NSNumber], dataType: .float32)
        let phasePtr = randomPhases.dataPointer.assumingMemoryBound(to: Float.self)
        for i in 0..<config.numPhases {
            phasePtr[i] = Float.random(in: 0..<(2 * .pi))
        }

        // Step 6: Run end-to-end model
        AudioLog.inference.debug("Kokoro: \(tokenIds.count) tokens → \(activeBucket.modelName) bucket")
        let t0 = CFAbsoluteTimeGetCurrent()
        let output = try network.predict(
            inputIds: inputIds,
            attentionMask: maskArray,
            refS: refS,
            randomPhases: randomPhases,
            bucket: activeBucket
        )
        let t1 = CFAbsoluteTimeGetCurrent()

        // Step 7: Extract audio samples
        let validSamples: Int
        if let lengthArray = output.audioLengthSamples {
            validSamples = lengthArray[0].intValue
        } else {
            validSamples = output.audio.count
        }
        let audio = extractAudio(from: output.audio, sampleCount: validSamples)

        let elapsedMs = (t1 - t0) * 1000
        let duration = Double(audio.count) / Double(config.sampleRate)
        print("Kokoro: \(String(format: "%.1f", elapsedMs))ms, " +
              "\(String(format: "%.2f", duration))s audio, " +
              "RTFx=\(String(format: "%.1f", duration / (elapsedMs / 1000)))")

        return audio
    }

    /// List available voice presets.
    public var availableVoices: [String] {
        Array(voiceEmbeddings.keys).sorted()
    }

    // MARK: - Audio Extraction

    private func extractAudio(from array: MLMultiArray, sampleCount: Int) -> [Float] {
        let count = min(array.count, max(0, sampleCount))
        guard count > 0 else { return [] }

        var samples = [Float](repeating: 0, count: count)
        if array.dataType == .float16 {
            let ptr = array.dataPointer.assumingMemoryBound(to: Float16.self)
            for i in 0..<count { samples[i] = Float(ptr[i]) }
        } else {
            let ptr = array.dataPointer.assumingMemoryBound(to: Float.self)
            samples.withUnsafeMutableBufferPointer { dst in
                dst.baseAddress!.update(from: ptr, count: count)
            }
        }
        return samples
    }

    // MARK: - Warmup

    /// Warm up CoreML models by running a dummy inference.
    public func warmUp() throws {
        _ = try? synthesize(text: "hello", voice: availableVoices.first ?? "af_heart")
    }

    // MARK: - Model Loading

    /// Load a pretrained Kokoro model from HuggingFace.
    ///
    /// Downloads CoreML models and voice embeddings on first use, then caches locally.
    /// Default voice preset.
    public static let defaultVoice = "af_heart"

    /// - Parameters:
    ///   - loadG2P: Load neural G2P models for OOV words (~80MB extra memory).
    ///     Set to `false` for voice pipelines where responses use common words.
    public static func fromPretrained(
        modelId: String = defaultModelId,
        voice: String = defaultVoice,
        maxBuckets: Int = 1,
        computeUnits: MLComputeUnits = .all,
        loadG2P: Bool = true,
        progressHandler: ((Double, String) -> Void)? = nil
    ) async throws -> KokoroTTSModel {
        AudioLog.modelLoading.info("Loading Kokoro model: \(modelId)")

        let cacheDir: URL
        do {
            cacheDir = try HuggingFaceDownloader.getCacheDirectory(for: modelId)
        } catch {
            throw AudioModelError.modelLoadFailed(
                modelId: modelId, reason: "Failed to resolve cache directory", underlying: error)
        }

        // Download single model variant + config + single voice.
        // kokoro_21_5s is the smallest bucket (max 124 tokens, ~5s audio)
        // — sufficient for short TTS and uses less memory than larger buckets.
        progressHandler?(0.0, "Downloading model...")
        do {
            try await HuggingFaceDownloader.downloadWeights(
                modelId: modelId,
                to: cacheDir,
                additionalFiles: [
                    "kokoro_21_5s.mlmodelc/**",
                    "G2PEncoder.mlmodelc/**",
                    "G2PDecoder.mlmodelc/**",
                    "vocab_index.json",
                    "g2p_vocab.json",
                    "us_gold.json",
                    "us_silver.json",
                    "voices/\(voice).json",
                ]
            ) { fraction in
                progressHandler?(fraction * 0.7, "Downloading model...")
            }
        } catch {
            throw AudioModelError.modelLoadFailed(
                modelId: modelId, reason: "Download failed", underlying: error)
        }

        let voicesDir = cacheDir.appendingPathComponent("voices")

        // Load config
        progressHandler?(0.70, "Loading configuration...")
        let config = KokoroConfig.default

        // Load vocabulary (vocab_index.json maps IPA symbols → token IDs)
        progressHandler?(0.72, "Loading vocabulary...")
        let vocabURL = cacheDir.appendingPathComponent("vocab_index.json")
        let phonemizer: KokoroPhonemizer
        if FileManager.default.fileExists(atPath: vocabURL.path) {
            phonemizer = try KokoroPhonemizer.loadVocab(from: vocabURL)
        } else {
            throw AudioModelError.modelLoadFailed(
                modelId: modelId, reason: "vocab_index.json not found")
        }

        // Load pronunciation dictionaries
        progressHandler?(0.74, "Loading pronunciation dictionaries...")
        try phonemizer.loadDictionaries(from: cacheDir)

        // Load G2P models (separate encoder + decoder) — optional, ~80MB extra
        if loadG2P {
            progressHandler?(0.76, "Loading G2P models...")
            let g2pEncoderURL = cacheDir.appendingPathComponent("G2PEncoder.mlmodelc", isDirectory: true)
            let g2pDecoderURL = cacheDir.appendingPathComponent("G2PDecoder.mlmodelc", isDirectory: true)
            let g2pVocabURL = cacheDir.appendingPathComponent("g2p_vocab.json")
            if FileManager.default.fileExists(atPath: g2pEncoderURL.path) &&
               FileManager.default.fileExists(atPath: g2pDecoderURL.path) {
                try phonemizer.loadG2PModels(
                    encoderURL: g2pEncoderURL,
                    decoderURL: g2pDecoderURL,
                    vocabURL: g2pVocabURL
                )
                AudioLog.modelLoading.debug("Loaded CoreML G2P encoder + decoder")
            }
        } else {
            AudioLog.modelLoading.debug("Skipped G2P models (loadG2P=false)")
        }

        // Load voice embeddings from per-voice JSON files
        progressHandler?(0.78, "Loading voice embeddings...")
        var voiceEmbeddings = [String: [Float]]()
        if FileManager.default.fileExists(atPath: voicesDir.path) {
            let files = try FileManager.default.contentsOfDirectory(
                at: voicesDir, includingPropertiesForKeys: nil)
            for file in files where file.pathExtension == "json" {
                let voiceName = file.deletingPathExtension().lastPathComponent
                if let embedding = try? loadVoiceEmbedding(from: file, styleDim: config.styleDim) {
                    voiceEmbeddings[voiceName] = embedding
                }
            }
            AudioLog.modelLoading.debug("Loaded \(voiceEmbeddings.count) voice presets")
        }

        // Load CoreML TTS models
        progressHandler?(0.85, "Loading CoreML models...")
        let network: KokoroNetwork
        do {
            network = try KokoroNetwork(directory: cacheDir, computeUnits: computeUnits, maxBuckets: maxBuckets)
            AudioLog.modelLoading.debug("Loaded buckets: \(network.availableBuckets.map { $0.modelName })")
        } catch {
            throw AudioModelError.modelLoadFailed(
                modelId: modelId, reason: "Failed to load CoreML models", underlying: error)
        }

        progressHandler?(1.0, "Model loaded")
        AudioLog.modelLoading.info("Kokoro model loaded successfully")

        return KokoroTTSModel(
            config: config,
            network: network,
            phonemizer: phonemizer,
            voiceEmbeddings: voiceEmbeddings
        )
    }

    /// Load voice embedding from a per-voice JSON file.
    ///
    /// Voice JSON contains: {"embedding": [256 floats], "1": [256], ..., "510": [256]}
    /// The model's ref_s input is [1, 256]. We use the "embedding" key.
    private static func loadVoiceEmbedding(from url: URL, styleDim: Int) throws -> [Float] {
        let data = try Data(contentsOf: url)
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let embedding = json["embedding"] as? [Double] else {
            return []
        }
        return embedding.prefix(styleDim).map { Float($0) }
    }
}
