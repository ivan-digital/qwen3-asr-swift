import CoreML
import Foundation
import AudioCommon

/// Kokoro-82M text-to-speech — CoreML-based, runs on Neural Engine.
///
/// Lightweight (82M params) non-autoregressive TTS model.
/// Supports 8 languages with 50 preset voices. Designed for iOS/iPad deployment.
///
/// Uses a 3-stage CoreML pipeline:
/// 1. Duration model: predicts phoneme durations
/// 2. Prosody model: predicts F0 and noise features
/// 3. Decoder: generates audio waveform
///
/// ```swift
/// let tts = try await KokoroTTSModel.fromPretrained()
/// let audio = try tts.synthesize(text: "Hello world", voice: "af_heart")
/// ```
public final class KokoroTTSModel {

    /// Default HuggingFace model ID.
    public static let defaultModelId = "aufklarer/Kokoro-82M-CoreML"

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
    ///   - speed: Speech speed multiplier (1.0 = normal, < 1.0 = slower, > 1.0 = faster)
    /// - Returns: Audio samples at 24kHz, Float32
    public func synthesize(
        text: String,
        voice: String = "af_heart",
        language: String = "en",
        speed: Float = 1.0
    ) throws -> [Float] {
        guard _isLoaded, let network else {
            throw AudioModelError.inferenceFailed(
                operation: "kokoro-synthesize", reason: "Model not loaded")
        }

        // Step 1: Phonemize text → token IDs
        let tokenIds = phonemizer.tokenize(text, maxLength: config.maxPhonemeLength)
        let tokenCount = tokenIds.count

        // Step 2: Get voice style embedding (256-dim)
        guard let styleVector = voiceEmbeddings[voice] else {
            let available = Array(voiceEmbeddings.keys).sorted().prefix(5)
            throw AudioModelError.voiceNotFound(
                voice: voice,
                searchPath: "Available: \(available.joined(separator: ", "))...")
        }

        // Step 3: Select phoneme bucket (minimizes LSTM padding contamination)
        guard let phonemeBucket = PhonemeBucket.select(forTokenCount: tokenCount) else {
            throw AudioModelError.inferenceFailed(
                operation: "kokoro-synthesize",
                reason: "Text too long (\(tokenCount) tokens), max \(PhonemeBucket.p128.rawValue)")
        }
        let padTo = phonemeBucket.rawValue

        // Step 4: Create duration model inputs
        let paddedIds = phonemizer.pad(tokenIds, to: padTo)
        let inputIds = try createInt32Array(shape: [1, padTo], values: paddedIds.map { Int32($0) })
        let maskArray = try createInt32Array(shape: [1, padTo], values: (0..<padTo).map { Int32($0 < tokenCount ? 1 : 0) })
        let refS = try createFloatArray(shape: [1, config.styleDim], values: styleVector)
        let speedArray = try createFloatArray(shape: [1], values: [speed])

        AudioLog.inference.debug("Kokoro: \(tokenCount) tokens → pad \(padTo)")
        let t0 = CFAbsoluteTimeGetCurrent()

        // Stage 1: Predict durations
        let durOutput = try network.predictDuration(
            inputIds: inputIds, attentionMask: maskArray, refS: refS, speed: speedArray)

        // Extract valid durations (only first tokenCount positions)
        let durations = extractDurations(from: durOutput.predDur, validCount: tokenCount)
        let totalFrames = durations.reduce(0, +)

        // Select decoder bucket
        guard let decoderBucket = DecoderBucket.select(forFrameCount: totalFrames) else {
            throw AudioModelError.inferenceFailed(
                operation: "kokoro-synthesize",
                reason: "Audio too long (\(totalFrames) frames, \(Double(totalFrames * 600) / 24000)s)")
        }
        let bucketFrames = decoderBucket.maxFrames

        // Stage 1→2: Build alignment matrix and compute aligned features
        let (en, asr) = buildAlignedFeatures(
            durations: durations,
            dTransposed: durOutput.dTransposed,
            tEn: durOutput.tEn,
            padTo: padTo,
            totalFrames: totalFrames,
            bucketFrames: bucketFrames
        )

        // Stage 2: Predict prosody (F0 + N)
        let sArray = try createFloatArray(shape: [1, 128], values: Array(styleVector[128...]))
        let prosOutput = try network.predictProsody(en: en, s: sArray)

        // Stage 3: Decode to audio
        let refSDec = try createFloatArray(shape: [1, 128], values: Array(styleVector[..<128]))
        let audioOutput = try network.decode(
            asr: asr,
            f0Pred: prosOutput.f0Pred,
            nPred: prosOutput.nPred,
            refS: refSDec,
            bucket: decoderBucket
        )

        // Extract valid audio samples
        let validSamples = totalFrames * config.samplesPerFrame
        let audio = extractAudio(from: audioOutput, sampleCount: validSamples)

        let t1 = CFAbsoluteTimeGetCurrent()
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

    // MARK: - Alignment

    /// Build aligned features from duration predictions.
    ///
    /// Creates an alignment matrix from predicted durations, then computes:
    /// - en = d_transposed @ alignment → [1, 640, bucketFrames] prosody features
    /// - asr = t_en @ alignment → [1, 512, bucketFrames] text features
    private func buildAlignedFeatures(
        durations: [Int],
        dTransposed: MLMultiArray,
        tEn: MLMultiArray,
        padTo: Int,
        totalFrames: Int,
        bucketFrames: Int
    ) -> (en: MLMultiArray, asr: MLMultiArray) {
        // Build alignment matrix [padTo, totalFrames]
        // aln[n, f] = 1.0 if frame f corresponds to phoneme n
        var alignment = [Float](repeating: 0, count: padTo * totalFrames)
        var frameIdx = 0
        for (phonemeIdx, dur) in durations.enumerated() {
            for _ in 0..<dur {
                if frameIdx < totalFrames {
                    alignment[phonemeIdx * totalFrames + frameIdx] = 1.0
                    frameIdx += 1
                }
            }
        }

        // Matrix multiply: en = d_transposed @ alignment
        // d_transposed: [1, 640, padTo], alignment: [padTo, totalFrames] → en: [1, 640, totalFrames]
        let dDim = 640
        let tDim = 512

        let en = matmul3D(a: dTransposed, aDim2: padTo, b: alignment,
                          bRows: padTo, bCols: totalFrames, outDim1: dDim,
                          padCols: bucketFrames)

        let asr = matmul3D(a: tEn, aDim2: padTo, b: alignment,
                           bRows: padTo, bCols: totalFrames, outDim1: tDim,
                           padCols: bucketFrames)

        return (en, asr)
    }

    /// Matrix multiply [1, D, M] × [M, N] → [1, D, padCols] with zero-padding.
    private func matmul3D(
        a: MLMultiArray, aDim2: Int,
        b: [Float], bRows: Int, bCols: Int,
        outDim1: Int, padCols: Int
    ) -> MLMultiArray {
        let result = try! MLMultiArray(shape: [1, outDim1 as NSNumber, padCols as NSNumber],
                                       dataType: .float32)
        let resultPtr = result.dataPointer.assumingMemoryBound(to: Float.self)

        // Handle both float16 and float32 input
        if a.dataType == .float16 {
            let aPtr = a.dataPointer.assumingMemoryBound(to: Float16.self)
            for d in 0..<outDim1 {
                for f in 0..<bCols {
                    var sum: Float = 0
                    for m in 0..<bRows {
                        sum += Float(aPtr[d * aDim2 + m]) * b[m * bCols + f]
                    }
                    resultPtr[d * padCols + f] = sum
                }
            }
        } else {
            let aPtr = a.dataPointer.assumingMemoryBound(to: Float.self)
            for d in 0..<outDim1 {
                for f in 0..<bCols {
                    var sum: Float = 0
                    for m in 0..<bRows {
                        sum += aPtr[d * aDim2 + m] * b[m * bCols + f]
                    }
                    resultPtr[d * padCols + f] = sum
                }
            }
        }

        return result
    }

    // MARK: - Helpers

    private func extractDurations(from array: MLMultiArray, validCount: Int) -> [Int] {
        var durations = [Int]()
        if array.dataType == .float16 {
            let ptr = array.dataPointer.assumingMemoryBound(to: Float16.self)
            for i in 0..<validCount {
                durations.append(max(1, Int(Float(ptr[i]).rounded())))
            }
        } else {
            let ptr = array.dataPointer.assumingMemoryBound(to: Float.self)
            for i in 0..<validCount {
                durations.append(max(1, Int(ptr[i].rounded())))
            }
        }
        return durations
    }

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

    private func createInt32Array(shape: [Int], values: [Int32]) throws -> MLMultiArray {
        let arr = try MLMultiArray(shape: shape.map { $0 as NSNumber }, dataType: .int32)
        let ptr = arr.dataPointer.assumingMemoryBound(to: Int32.self)
        for i in 0..<values.count { ptr[i] = values[i] }
        return arr
    }

    private func createFloatArray(shape: [Int], values: [Float]) throws -> MLMultiArray {
        let arr = try MLMultiArray(shape: shape.map { $0 as NSNumber }, dataType: .float32)
        let ptr = arr.dataPointer.assumingMemoryBound(to: Float.self)
        for i in 0..<values.count { ptr[i] = values[i] }
        return arr
    }

    // MARK: - Warmup

    /// Warm up CoreML models by running a dummy inference.
    public func warmUp() throws {
        _ = try? synthesize(text: "hello", voice: availableVoices.first ?? "af_heart")
    }

    // MARK: - Model Loading

    /// Load a pretrained Kokoro model from HuggingFace.
    ///
    /// Downloads 3-stage CoreML models and voice embeddings on first use.
    public static let defaultVoice = "af_heart"

    public static func fromPretrained(
        modelId: String = defaultModelId,
        voice: String = defaultVoice,
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

        // Download 3-stage models + G2P + voice
        progressHandler?(0.0, "Downloading model...")
        do {
            try await HuggingFaceDownloader.downloadWeights(
                modelId: modelId,
                to: cacheDir,
                additionalFiles: [
                    "duration.mlmodelc/**",
                    "prosody.mlmodelc/**",
                    "decoder_5s.mlmodelc/**",
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

        // Load vocabulary
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

        // Load G2P models
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

        // Load voice embeddings
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

        // Load 3-stage CoreML models
        progressHandler?(0.85, "Loading CoreML models...")
        let network: KokoroNetwork
        do {
            network = try KokoroNetwork(directory: cacheDir)
            AudioLog.modelLoading.debug("Loaded 3-stage pipeline: \(network.availableDecoderBuckets.map { $0.modelName })")
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
    private static func loadVoiceEmbedding(from url: URL, styleDim: Int) throws -> [Float] {
        let data = try Data(contentsOf: url)
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let embedding = json["embedding"] as? [Double] else {
            return []
        }
        return embedding.prefix(styleDim).map { Float($0) }
    }
}
