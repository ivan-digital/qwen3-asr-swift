import CoreML
import Foundation
import AudioCommon

/// Parakeet TDT 0.6B v3 — CoreML-based automatic speech recognition.
///
/// Uses a FastConformer encoder with a Token-and-Duration Transducer (TDT) decoder.
/// Mel preprocessing is done in Swift using Accelerate/vDSP. The encoder, decoder, and
/// joint network run on CoreML with INT4-quantized encoder for Neural Engine acceleration.
///
/// - Warning: This class is not thread-safe. Create separate instances for concurrent use.
public class ParakeetASRModel {
    /// Model configuration.
    public let config: ParakeetConfig

    /// Default HuggingFace model ID.
    public static let defaultModelId = "aufklarer/Parakeet-TDT-v3-CoreML-INT4"

    private let melPreprocessor: MelPreprocessor
    private let encoder: MLModel
    private let decoder: MLModel
    private let joint: MLModel
    private let vocabulary: ParakeetVocabulary

    private init(
        config: ParakeetConfig,
        encoder: MLModel,
        decoder: MLModel,
        joint: MLModel,
        vocabulary: ParakeetVocabulary
    ) {
        self.config = config
        self.melPreprocessor = MelPreprocessor(config: config)
        self.encoder = encoder
        self.decoder = decoder
        self.joint = joint
        self.vocabulary = vocabulary
    }

    // MARK: - Warmup

    /// Warm up CoreML models by running a dummy inference.
    ///
    /// CoreML compiles computation graphs on the first `prediction()` call, which causes
    /// a ~4x latency penalty on cold inference. Calling `warmUp()` triggers this compilation
    /// on a minimal 1-second silence input so that subsequent real transcriptions run at full speed.
    public func warmUp() throws {
        let dummyAudio = [Float](repeating: 0, count: config.sampleRate)  // 1s silence
        _ = try transcribeAudio(dummyAudio, sampleRate: config.sampleRate)
    }

    // MARK: - Transcription

    /// Transcribe audio to text.
    ///
    /// - Parameters:
    ///   - audio: PCM Float32 audio samples
    ///   - sampleRate: Sample rate of the input audio in Hz
    ///   - language: Language hint (unused, Parakeet auto-detects from 25 European languages)
    /// - Returns: Transcribed text
    /// - Throws: `AudioModelError` on CoreML inference failure
    public func transcribeAudio(_ audio: [Float], sampleRate: Int, language: String? = nil) throws -> String {
        // Resample to 16kHz if needed
        let samples: [Float]
        if sampleRate != config.sampleRate {
            samples = AudioFileLoader.resample(audio, from: sampleRate, to: config.sampleRate)
        } else {
            samples = audio
        }

        // Step 1: Mel spectrogram extraction (Swift/Accelerate)
        AudioLog.inference.debug("Parakeet: preprocessing \(samples.count) samples")
        let tMel0 = CFAbsoluteTimeGetCurrent()
        let (mel, melLength) = try melPreprocessor.extract(samples)
        let tMel1 = CFAbsoluteTimeGetCurrent()

        // Step 2: Encoder — mel → encoded representations
        // Pad mel to nearest enumerated shape (EnumeratedShapes avoids BNNS crash)
        let paddedMel = try padMelToEnumeratedShape(mel: mel, actualLength: melLength)
        AudioLog.inference.debug("Parakeet: encoding \(melLength) mel frames, padded to \(paddedMel.shape[2])")
        let tEnc0 = CFAbsoluteTimeGetCurrent()
        let encoderOutput = try runEncoder(mel: paddedMel, length: melLength)
        let tEnc1 = CFAbsoluteTimeGetCurrent()
        let encoded = encoderOutput.featureValue(for: "encoded")!.multiArrayValue!
        let encodedLengthArray = encoderOutput.featureValue(for: "encoded_length")!.multiArrayValue!
        let encodedLength = encodedLengthArray[0].intValue

        // Step 3: TDT greedy decode
        let tdtDecoder = TDTGreedyDecoder(config: config, decoder: decoder, joint: joint)
        let tDec0 = CFAbsoluteTimeGetCurrent()
        let tokenIds = try tdtDecoder.decode(encoded: encoded, encodedLength: encodedLength)
        let tDec1 = CFAbsoluteTimeGetCurrent()

        // Step 4: Vocabulary decode
        let text = vocabulary.decode(tokenIds)

        let melMs = (tMel1 - tMel0) * 1000
        let encMs = (tEnc1 - tEnc0) * 1000
        let decMs = (tDec1 - tDec0) * 1000
        let totalMs = melMs + encMs + decMs
        print("Parakeet timing: mel=\(String(format: "%.1f", melMs))ms, encoder=\(String(format: "%.1f", encMs))ms, decode=\(String(format: "%.1f", decMs))ms, total=\(String(format: "%.1f", totalMs))ms (\(tokenIds.count) tokens, \(encodedLength) frames)")

        return text
    }

    // MARK: - CoreML Inference Helpers

    /// Enumerated mel frame lengths supported by the encoder CoreML model.
    private static let enumeratedMelLengths = [100, 200, 300, 400, 500, 750, 1000, 1500, 2000, 3000]

    /// Pad mel spectrogram to the nearest enumerated shape.
    /// The encoder uses EnumeratedShapes to avoid a BNNS crash with dynamic shapes.
    private func padMelToEnumeratedShape(mel: MLMultiArray, actualLength: Int) throws -> MLMultiArray {
        let melFrames = mel.shape[2].intValue

        // Find the smallest enumerated length >= melFrames
        guard let targetLength = Self.enumeratedMelLengths.first(where: { $0 >= melFrames }) else {
            throw AudioModelError.inferenceFailed(
                operation: "mel padding",
                reason: "Audio too long: \(melFrames) mel frames exceeds max \(Self.enumeratedMelLengths.last!) (30s)")
        }

        if targetLength == melFrames {
            return mel
        }

        // Create zero-padded mel array [1, 128, targetLength]
        let padded = try MLMultiArray(
            shape: [1, 128, targetLength as NSNumber], dataType: mel.dataType)
        let srcPtr = mel.dataPointer.assumingMemoryBound(to: Float16.self)
        let dstPtr = padded.dataPointer.assumingMemoryBound(to: Float16.self)

        let numMelBins = config.numMelBins
        for bin in 0..<numMelBins {
            let srcOffset = bin * melFrames
            let dstOffset = bin * targetLength
            dstPtr.advanced(by: dstOffset)
                .update(from: srcPtr.advanced(by: srcOffset), count: melFrames)
            dstPtr.advanced(by: dstOffset + melFrames)
                .update(repeating: Float16(0), count: targetLength - melFrames)
        }

        return padded
    }

    private func runEncoder(mel: MLMultiArray, length: Int) throws -> MLFeatureProvider {
        let lengthArray = try MLMultiArray(shape: [1], dataType: .int32)
        lengthArray[0] = NSNumber(value: Int32(length))

        let input = try MLDictionaryFeatureProvider(dictionary: [
            "mel": MLFeatureValue(multiArray: mel),
            "length": MLFeatureValue(multiArray: lengthArray),
        ])
        return try encoder.prediction(from: input)
    }

    // MARK: - Model Loading

    /// Load a pretrained Parakeet model from HuggingFace.
    ///
    /// - Parameters:
    ///   - modelId: HuggingFace model identifier
    ///   - progressHandler: Optional callback for download/load progress `(fraction, status)`
    /// - Returns: Initialized model ready for transcription
    public static func fromPretrained(
        modelId: String = defaultModelId,
        progressHandler: ((Double, String) -> Void)? = nil
    ) async throws -> ParakeetASRModel {
        AudioLog.modelLoading.info("Loading Parakeet model: \(modelId)")

        // Step 1: Get/create cache directory
        let cacheDir: URL
        do {
            cacheDir = try HuggingFaceDownloader.getCacheDirectory(for: modelId)
        } catch {
            throw AudioModelError.modelLoadFailed(
                modelId: modelId, reason: "Failed to resolve cache directory", underlying: error)
        }

        // Step 2: Download model files (no preprocessor needed — mel is computed in Swift)
        progressHandler?(0.0, "Downloading model...")
        do {
            try await HuggingFaceDownloader.downloadWeights(
                modelId: modelId,
                to: cacheDir,
                additionalFiles: [
                    "encoder.mlmodelc/**",
                    "decoder.mlmodelc/**",
                    "joint.mlmodelc/**",
                    "vocab.json",
                    "config.json",
                ]
            ) { fraction in
                progressHandler?(fraction * 0.7, "Downloading model...")
            }
        } catch {
            throw AudioModelError.modelLoadFailed(
                modelId: modelId, reason: "Download failed", underlying: error)
        }

        // Step 3: Load config
        progressHandler?(0.70, "Loading configuration...")
        let config: ParakeetConfig
        let configURL = cacheDir.appendingPathComponent("config.json")
        if FileManager.default.fileExists(atPath: configURL.path) {
            let data = try Data(contentsOf: configURL)
            config = try JSONDecoder().decode(ParakeetConfig.self, from: data)
            AudioLog.modelLoading.debug("Loaded config from \(configURL.path)")
        } else {
            config = .default
            AudioLog.modelLoading.debug("Using default config")
        }

        // Step 4: Load vocabulary
        progressHandler?(0.75, "Loading vocabulary...")
        let vocabURL = cacheDir.appendingPathComponent("vocab.json")
        let vocabulary: ParakeetVocabulary
        do {
            vocabulary = try ParakeetVocabulary.load(from: vocabURL)
            AudioLog.modelLoading.debug("Loaded vocabulary: \(vocabulary.count) tokens")
        } catch {
            throw AudioModelError.modelLoadFailed(
                modelId: modelId, reason: "Failed to load vocabulary", underlying: error)
        }

        // Step 5: Load CoreML models (encoder, decoder, joint — no preprocessor)
        progressHandler?(0.80, "Loading CoreML models...")
        let encoder = try loadCoreMLModel(
            name: "encoder", from: cacheDir, computeUnits: .all)
        progressHandler?(0.90, "Loading decoder...")
        let decoder = try loadCoreMLModel(
            name: "decoder", from: cacheDir, computeUnits: .cpuAndNeuralEngine)
        progressHandler?(0.95, "Loading joint network...")
        let joint = try loadCoreMLModel(
            name: "joint", from: cacheDir, computeUnits: .cpuAndNeuralEngine)

        progressHandler?(1.0, "Model loaded")
        AudioLog.modelLoading.info("Parakeet model loaded successfully")

        return ParakeetASRModel(
            config: config,
            encoder: encoder,
            decoder: decoder,
            joint: joint,
            vocabulary: vocabulary
        )
    }

    /// Load a compiled CoreML model from a `.mlmodelc` directory.
    private static func loadCoreMLModel(
        name: String,
        from directory: URL,
        computeUnits: MLComputeUnits
    ) throws -> MLModel {
        let modelURL = directory.appendingPathComponent("\(name).mlmodelc", isDirectory: true)

        guard FileManager.default.fileExists(atPath: modelURL.path) else {
            throw AudioModelError.modelLoadFailed(
                modelId: name, reason: "CoreML model not found at \(modelURL.path)")
        }

        let mlConfig = MLModelConfiguration()
        mlConfig.computeUnits = computeUnits

        do {
            let model = try MLModel(contentsOf: modelURL, configuration: mlConfig)
            AudioLog.modelLoading.debug("Loaded CoreML model: \(name) (compute: \(String(describing: computeUnits)))")
            return model
        } catch {
            throw AudioModelError.modelLoadFailed(
                modelId: name, reason: "Failed to compile/load CoreML model", underlying: error)
        }
    }
}
