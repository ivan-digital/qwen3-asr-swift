import CoreML
import Foundation
import AudioCommon

/// Meta Omnilingual ASR — CTC variant, running on CoreML / Neural Engine.
///
/// This Swift port matches Meta's `ASRInferencePipeline.transcribe(...)`:
///
/// 1. Resample to 16 kHz mono Float32 (via `AudioFileLoader.resample`).
/// 2. Pad or truncate the audio to exactly `config.inputSamples` — the CoreML
///    graph is traced with a fixed window (5 s or 10 s).
/// 3. Apply layer-norm-style waveform normalization:
///    `(x - mean(x)) / sqrt(var(x) + eps)`
///    This mirrors fairseq2's `apply_audio_normalization(waveform)`.
/// 4. Run the single CoreML model → `[1, T, vocab_size]` logits.
/// 5. Greedy CTC: argmax per frame, collapse consecutive duplicates.
/// 6. SentencePiece detokenize with `skip_special_tokens=True`.
///
/// The CTC variant is **language-agnostic**: it covers all 1600+ languages in
/// Meta's catalog and ignores the `language` parameter. For language-conditioned
/// inference, use the LLM variant (separate follow-up module).
///
/// - Warning: This class is not thread-safe. Create separate instances for
///   concurrent use.
public class OmnilingualASRModel {
    /// Runtime configuration loaded from the published `config.json`.
    public let config: OmnilingualConfig

    /// Default HuggingFace model id (10s window, INT8 palettized).
    public static let defaultModelId = "aufklarer/Omnilingual-ASR-CTC-300M-CoreML-INT8-10s"

    /// Shorter window variant (5s) — lower memory, faster cold start, but
    /// requires more chunking for long-form audio.
    public static let shortWindowModelId = "aufklarer/Omnilingual-ASR-CTC-300M-CoreML-INT8"

    /// Numerical stability epsilon for layer-norm waveform normalization.
    /// Matches PyTorch's `layer_norm` default `eps=1e-5`.
    public static let layerNormEpsilon: Float = 1e-5

    /// Maximum utterance length the reference pipeline accepts. Matches
    /// `MAX_ALLOWED_AUDIO_SEC = 40` in Meta's `ASRInferencePipeline`.
    /// For longer audio, segment with VAD (`SpeechVAD`) or use `ParakeetStreamingASR`.
    public static let maxAudioSeconds: Double = 40.0

    var _isLoaded = true
    var model: MLModel?
    private let vocabulary: OmnilingualVocabulary

    private init(
        config: OmnilingualConfig,
        model: MLModel,
        vocabulary: OmnilingualVocabulary
    ) {
        self.config = config
        self.model = model
        self.vocabulary = vocabulary
    }

    // MARK: - Loading

    /// Download and load the model from HuggingFace.
    public static func fromPretrained(
        modelId: String? = nil,
        cacheDir: URL? = nil,
        offlineMode: Bool = false,
        progressHandler: ((Double, String) -> Void)? = nil
    ) async throws -> OmnilingualASRModel {
        let effectiveModelId = modelId ?? defaultModelId
        AudioLog.modelLoading.info("Loading Omnilingual ASR model: \(effectiveModelId)")

        let resolvedCacheDir: URL
        do {
            resolvedCacheDir = try cacheDir ?? HuggingFaceDownloader.getCacheDirectory(for: effectiveModelId)
        } catch {
            throw AudioModelError.modelLoadFailed(
                modelId: effectiveModelId, reason: "Failed to resolve cache directory", underlying: error)
        }

        progressHandler?(0.0, "Downloading model...")
        do {
            try await HuggingFaceDownloader.downloadWeights(
                modelId: effectiveModelId,
                to: resolvedCacheDir,
                additionalFiles: [
                    "config.json",
                    "tokenizer.model",
                    "omnilingual-ctc-300m-int8.mlpackage/**",
                ],
                offlineMode: offlineMode
            ) { fraction in
                progressHandler?(fraction * 0.8, "Downloading model...")
            }
        } catch {
            throw AudioModelError.modelLoadFailed(
                modelId: effectiveModelId, reason: "Download failed", underlying: error)
        }

        // Config
        progressHandler?(0.80, "Loading configuration...")
        let configURL = resolvedCacheDir.appendingPathComponent("config.json")
        let config: OmnilingualConfig
        if FileManager.default.fileExists(atPath: configURL.path) {
            let data = try Data(contentsOf: configURL)
            config = try JSONDecoder().decode(OmnilingualConfig.self, from: data)
        } else {
            AudioLog.modelLoading.debug("config.json not found, using default10s")
            config = .default10s
        }

        // Vocabulary
        progressHandler?(0.85, "Loading tokenizer...")
        let tokenizerURL = resolvedCacheDir.appendingPathComponent(config.tokenizer.file)
        guard FileManager.default.fileExists(atPath: tokenizerURL.path) else {
            throw AudioModelError.modelLoadFailed(
                modelId: effectiveModelId,
                reason: "tokenizer.model not found at \(tokenizerURL.path)",
                underlying: nil)
        }
        let vocabulary = try OmnilingualVocabulary.load(from: tokenizerURL, tokenizer: config.tokenizer)
        AudioLog.modelLoading.debug("Loaded SentencePiece vocabulary: \(vocabulary.count) pieces")

        // CoreML model
        progressHandler?(0.92, "Loading CoreML model...")
        let mlModel = try loadCoreMLModel(from: resolvedCacheDir)

        progressHandler?(1.0, "Model loaded")
        AudioLog.modelLoading.info("Omnilingual ASR model loaded successfully")

        return OmnilingualASRModel(config: config, model: mlModel, vocabulary: vocabulary)
    }

    private static func loadCoreMLModel(from directory: URL) throws -> MLModel {
        // The published repos use the .mlpackage layout. Prefer the compiled
        // .mlmodelc if it exists (first-run cached), otherwise compile on the fly.
        let packageURL = directory.appendingPathComponent(
            "omnilingual-ctc-300m-int8.mlpackage", isDirectory: true)
        guard FileManager.default.fileExists(atPath: packageURL.path) else {
            throw AudioModelError.modelLoadFailed(
                modelId: "omnilingual-ctc-300m",
                reason: "mlpackage not found at \(packageURL.path)",
                underlying: nil)
        }

        let configuration = MLModelConfiguration()
        // Match ParakeetASR default — ANE compilation is unreliable on some
        // devices, so target CPU+GPU for portability.
        configuration.computeUnits = .cpuAndGPU

        do {
            let compiledURL = try MLModel.compileModel(at: packageURL)
            return try MLModel(contentsOf: compiledURL, configuration: configuration)
        } catch {
            throw AudioModelError.modelLoadFailed(
                modelId: "omnilingual-ctc-300m",
                reason: "CoreML model load failed",
                underlying: error)
        }
    }

    // MARK: - Warmup

    /// Warm up CoreML by running inference on silence. First prediction is
    /// ~4× slower due to compute-graph compilation.
    public func warmUp() throws {
        let dummy = [Float](repeating: 0, count: config.inputSamples)
        _ = try transcribeAudio(dummy, sampleRate: config.sampleRate)
    }

    // MARK: - Transcription

    /// Transcribe an audio buffer to text. Input at any sample rate; the
    /// model resamples to 16 kHz internally. Audio longer than the CoreML
    /// window (`config.maxAudioSeconds`) is chunked into non-overlapping
    /// windows and the transcripts are concatenated. The whole utterance
    /// is hard-capped at `Self.maxAudioSeconds` (40 s), matching Meta's
    /// `ASRInferencePipeline`. For longer audio, segment with `SpeechVAD`.
    public func transcribeAudio(_ audio: [Float], sampleRate: Int, language: String? = nil) throws -> String {
        guard model != nil else {
            throw AudioModelError.inferenceFailed(operation: "transcribe", reason: "Model not loaded")
        }

        // 1. Resample to 16 kHz if needed.
        let samples: [Float]
        if sampleRate != config.sampleRate {
            samples = AudioFileLoader.resample(audio, from: sampleRate, to: config.sampleRate)
        } else {
            samples = audio
        }

        let durationSec = Double(samples.count) / Double(config.sampleRate)
        if durationSec > Self.maxAudioSeconds {
            throw AudioModelError.inferenceFailed(
                operation: "transcribe",
                reason: "Input \(String(format: "%.1f", durationSec))s exceeds Omnilingual cap of \(Int(Self.maxAudioSeconds))s. Segment with SpeechVAD or use ParakeetStreamingASR.")
        }

        // 2. Chunk into `inputSamples`-sized windows.
        let windowSize = config.inputSamples
        guard windowSize > 0 else {
            throw AudioModelError.inferenceFailed(operation: "transcribe", reason: "Invalid inputSamples=0")
        }

        var offset = 0
        var transcripts: [String] = []
        while offset < samples.count {
            let end = min(offset + windowSize, samples.count)
            // Normalize on the actual content first (matches reference: layer_norm
            // is applied to the raw waveform, not the zero-padded window — otherwise
            // padding silence skews the mean/variance for sub-window inputs).
            var chunk = Self.layerNormalize(Array(samples[offset..<end]), eps: Self.layerNormEpsilon)
            if chunk.count < windowSize {
                chunk.append(contentsOf: repeatElement(0, count: windowSize - chunk.count))
            }

            let text = try runWindow(chunk)
            if !text.isEmpty {
                transcripts.append(text)
            }
            offset = end
        }

        return transcripts.joined(separator: " ")
    }

    /// Run inference on exactly one fixed-length, already-normalized window.
    /// Caller is responsible for chunking, normalizing, and zero-padding to
    /// `config.inputSamples`.
    private func runWindow(_ window: [Float]) throws -> String {
        precondition(window.count == config.inputSamples,
                     "window.count=\(window.count) must equal config.inputSamples=\(config.inputSamples)")
        guard let model else {
            throw AudioModelError.inferenceFailed(operation: "transcribe", reason: "Model not loaded")
        }

        // Build CoreML input [1, inputSamples].
        let tMel0 = CFAbsoluteTimeGetCurrent()
        let inputArray = try MLMultiArray(
            shape: [1, NSNumber(value: config.inputSamples)], dataType: .float32)
        let inputPtr = inputArray.dataPointer.bindMemory(to: Float.self, capacity: config.inputSamples)
        window.withUnsafeBufferPointer { src in
            inputPtr.update(from: src.baseAddress!, count: config.inputSamples)
        }

        // 5. Run the model.
        let provider = try MLDictionaryFeatureProvider(dictionary: ["audio": inputArray])
        let tEnc0 = CFAbsoluteTimeGetCurrent()
        let output = try model.prediction(from: provider)
        let tEnc1 = CFAbsoluteTimeGetCurrent()

        guard let logits = output.featureValue(for: "logits")?.multiArrayValue else {
            throw AudioModelError.inferenceFailed(operation: "transcribe",
                                                   reason: "Model did not produce `logits` output")
        }

        // Expected shape [1, T, vocab_size]; pull T and V defensively.
        let shape = logits.shape.map { $0.intValue }
        guard shape.count == 3 else {
            throw AudioModelError.inferenceFailed(
                operation: "transcribe",
                reason: "Unexpected logits shape \(shape) — expected [1, T, V]")
        }
        let T = shape[1]
        let V = shape[2]
        guard V == config.ctcHead.vocabSize else {
            throw AudioModelError.inferenceFailed(
                operation: "transcribe",
                reason: "Vocab mismatch: logits V=\(V), config vocabSize=\(config.ctcHead.vocabSize)")
        }

        // 6. Flatten logits to `[T, V]` (Float32) and greedy CTC.
        //
        // CoreML MLMultiArrays with `compute_precision=FLOAT16` produce Float16
        // outputs, and may use non-dense strides. Use MLMultiArray's own flat
        // subscript which transparently handles both dtype conversion and
        // stride math. ~5M elements per 10 s window runs in <10 ms on ANE.
        let elementCount = T * V
        var flat = [Float](repeating: 0, count: elementCount)
        AudioLog.inference.debug("Omnilingual: logits shape=\(shape) dtype=\(logits.dataType.rawValue) strides=\(logits.strides)")
        for i in 0..<elementCount {
            flat[i] = logits[i].floatValue
        }
        let tDec0 = CFAbsoluteTimeGetCurrent()
        let tokenIds = CTCGreedyDecoder.decode(logits: flat, timeSteps: T, vocabSize: V)
        let tDec1 = CFAbsoluteTimeGetCurrent()

        // 7. Detokenize with skip_special_tokens.
        let text = vocabulary.decode(tokenIds)

        let inMs = (tEnc0 - tMel0) * 1000
        let encMs = (tEnc1 - tEnc0) * 1000
        let decMs = (tDec1 - tDec0) * 1000
        AudioLog.inference.info("Omnilingual: input=\(String(format: "%.1f", inMs))ms enc=\(String(format: "%.1f", encMs))ms dec=\(String(format: "%.1f", decMs))ms (T=\(T), \(tokenIds.count) tokens)")

        return text
    }

    // MARK: - Waveform normalization (layer_norm)

    /// Match PyTorch's `layer_norm(waveform, waveform.shape)` — zero mean,
    /// unit variance over the entire buffer with biased variance estimate.
    /// Made `internal` + testable so the unit tests can verify the math.
    static func layerNormalize(_ samples: [Float], eps: Float) -> [Float] {
        let n = samples.count
        guard n > 0 else { return samples }

        var sum: Float = 0
        var sumSq: Float = 0
        for s in samples {
            sum += s
            sumSq += s * s
        }
        let mean = sum / Float(n)
        // PyTorch uses biased variance (divide by N, not N-1) in layer_norm.
        let variance = max(0, sumSq / Float(n) - mean * mean)
        let invStd = 1 / (variance + eps).squareRoot()

        var out = [Float](repeating: 0, count: n)
        for i in 0..<n {
            out[i] = (samples[i] - mean) * invStd
        }
        return out
    }
}
