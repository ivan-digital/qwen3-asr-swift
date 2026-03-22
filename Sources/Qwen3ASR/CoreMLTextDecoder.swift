#if canImport(CoreML)
import CoreML
import Foundation
import MLX
import AudioCommon

/// CoreML text decoder for Qwen3-ASR with MLState KV cache.
///
/// Runs the full text decoder on Neural Engine via CoreML instead of GPU via MLX.
/// Uses fixed-size KV cache with attention masking. Requires macOS 15+ / iOS 18+
/// for MLState support.
///
/// The decoder consists of two CoreML models:
///   - **embedding**: Token ID → embedding vector lookup
///   - **decoder**: Embedding → logits, with 56 KV cache states (28 layers × 2)
@available(macOS 15, iOS 18, *)
public class CoreMLTextDecoder {
    private let embeddingModel: MLModel
    private let decoderModel: MLModel
    private let maxSeqLength: Int
    private let vocabSize: Int
    private let hiddenSize: Int

    /// MLState holds the KV cache buffers managed by CoreML.
    private var decoderState: MLState

    /// Current position in the KV cache (incremented per step).
    private var currentPosition: Int = 0

    public static let defaultModelId = "aufklarer/Qwen3-ASR-CoreML"

    public init(
        embeddingModel: MLModel,
        decoderModel: MLModel,
        maxSeqLength: Int = 1024,
        vocabSize: Int = 151936,
        hiddenSize: Int = 1024
    ) {
        self.embeddingModel = embeddingModel
        self.decoderModel = decoderModel
        self.maxSeqLength = maxSeqLength
        self.vocabSize = vocabSize
        self.hiddenSize = hiddenSize
        self.decoderState = decoderModel.makeState()
    }

    /// Load decoder models from a directory containing `embedding.mlmodelc` and `decoder.mlmodelc`.
    public static func load(
        from directory: URL,
        computeUnits: MLComputeUnits = .all
    ) throws -> CoreMLTextDecoder {
        let config = MLModelConfiguration()
        config.computeUnits = computeUnits

        // Load config.json for model parameters
        var maxSeq = 1024
        var vocabSize = 151936
        var hiddenSize = 1024
        let configPath = directory.appendingPathComponent("config.json")
        if let data = try? Data(contentsOf: configPath),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            maxSeq = json["max_seq_length"] as? Int ?? 1024
            vocabSize = json["vocab_size"] as? Int ?? 151936
            hiddenSize = json["hidden_size"] as? Int ?? 1024
        }

        // Try compiled first, then mlpackage
        let embURL = findModel(named: "embedding", in: directory)
        let decURL = findModel(named: "decoder", in: directory)

        guard let embURL else {
            throw AudioModelError.modelLoadFailed(
                modelId: "embedding",
                reason: "CoreML embedding not found in \(directory.path)")
        }
        guard let decURL else {
            throw AudioModelError.modelLoadFailed(
                modelId: "decoder",
                reason: "CoreML decoder not found in \(directory.path)")
        }

        let embModel = try MLModel(contentsOf: embURL, configuration: config)
        let decModel = try MLModel(contentsOf: decURL, configuration: config)

        return CoreMLTextDecoder(
            embeddingModel: embModel,
            decoderModel: decModel,
            maxSeqLength: maxSeq,
            vocabSize: vocabSize,
            hiddenSize: hiddenSize
        )
    }

    /// Load from HuggingFace.
    public static func fromPretrained(
        modelId: String = defaultModelId,
        computeUnits: MLComputeUnits = .all,
        progressHandler: ((Double, String) -> Void)? = nil
    ) async throws -> CoreMLTextDecoder {
        let cacheDir = try HuggingFaceDownloader.getCacheDirectory(for: modelId)

        progressHandler?(0.0, "Downloading CoreML decoder...")
        try await HuggingFaceDownloader.downloadWeights(
            modelId: modelId,
            to: cacheDir,
            additionalFiles: [
                "embedding.mlmodelc/**",
                "decoder.mlmodelc/**",
                "config.json",
            ]
        ) { fraction in
            progressHandler?(fraction * 0.8, "Downloading CoreML decoder...")
        }

        progressHandler?(0.9, "Loading CoreML decoder...")
        let decoder = try load(from: cacheDir, computeUnits: computeUnits)
        progressHandler?(1.0, "Ready")
        return decoder
    }

    /// Warm up both models with dummy inputs.
    public func warmUp() throws {
        // Warm embedding
        let dummyToken = try MLMultiArray(shape: [1, 1], dataType: .int32)
        dummyToken[0] = 0
        let embInput = try MLDictionaryFeatureProvider(dictionary: [
            "token_id": MLFeatureValue(multiArray: dummyToken),
        ])
        _ = try embeddingModel.prediction(from: embInput)

        // Warm decoder with a single step (using a temporary state so we don't pollute the real cache)
        let warmupState = decoderModel.makeState()
        let dummyEmbed = try MLMultiArray(shape: [1, 1, hiddenSize as NSNumber], dataType: .float32)
        let dummyPos = try MLMultiArray(shape: [1], dataType: .int32)
        dummyPos[0] = 0
        let dummyMask = try MLMultiArray(shape: [1, 1, 1, maxSeqLength as NSNumber], dataType: .float32)
        let ptr = dummyMask.dataPointer.assumingMemoryBound(to: Float.self)
        for i in 0..<maxSeqLength { ptr[i] = -1e4 }
        ptr[0] = 0  // Allow position 0

        let decInput = try MLDictionaryFeatureProvider(dictionary: [
            "input_embeds": MLFeatureValue(multiArray: dummyEmbed),
            "position": MLFeatureValue(multiArray: dummyPos),
            "attention_mask": MLFeatureValue(multiArray: dummyMask),
        ])
        _ = try decoderModel.prediction(from: decInput, using: warmupState)
    }

    /// Reset the KV cache for a new transcription.
    public func resetCache() {
        currentPosition = 0
        decoderState = decoderModel.makeState()
    }

    // MARK: - Token Operations

    /// Look up embedding for a token ID.
    public func embed(tokenId: Int32) throws -> MLMultiArray {
        let tokenArray = try MLMultiArray(shape: [1, 1], dataType: .int32)
        tokenArray[0] = NSNumber(value: tokenId)

        let input = try MLDictionaryFeatureProvider(dictionary: [
            "token_id": MLFeatureValue(multiArray: tokenArray),
        ])
        let output = try embeddingModel.prediction(from: input)

        guard let embedding = output.featureValue(for: "embedding")?.multiArrayValue else {
            throw AudioModelError.inferenceFailed(
                operation: "CoreML embedding", reason: "Missing embedding output")
        }
        return embedding
    }

    /// Run one decoder step with the given embedding.
    ///
    /// Updates the KV cache at the current position and returns logits.
    public func decoderStep(embedding: MLMultiArray) throws -> MLMultiArray {
        guard currentPosition < maxSeqLength else {
            throw AudioModelError.inferenceFailed(
                operation: "CoreML decoder",
                reason: "Sequence length \(currentPosition) exceeds max \(maxSeqLength)")
        }

        // Build attention mask: 0 for valid positions, -1e4 for invalid
        let mask = try MLMultiArray(shape: [1, 1, 1, maxSeqLength as NSNumber], dataType: .float32)
        let maskPtr = mask.dataPointer.assumingMemoryBound(to: Float.self)
        for i in 0...currentPosition {
            maskPtr[i] = 0
        }
        for i in (currentPosition + 1)..<maxSeqLength {
            maskPtr[i] = -1e4
        }

        let position = try MLMultiArray(shape: [1], dataType: .int32)
        position[0] = NSNumber(value: Int32(currentPosition))

        let input = try MLDictionaryFeatureProvider(dictionary: [
            "input_embeds": MLFeatureValue(multiArray: embedding),
            "position": MLFeatureValue(multiArray: position),
            "attention_mask": MLFeatureValue(multiArray: mask),
        ])

        let output = try decoderModel.prediction(from: input, using: decoderState)
        currentPosition += 1

        guard let logits = output.featureValue(for: "logits")?.multiArrayValue else {
            throw AudioModelError.inferenceFailed(
                operation: "CoreML decoder", reason: "Missing logits output")
        }
        return logits
    }

    /// Get argmax token ID from logits.
    public func argmax(logits: MLMultiArray) -> Int32 {
        let count = logits.count
        var maxVal: Float = -Float.infinity
        var maxIdx: Int32 = 0

        switch logits.dataType {
        case .float16:
            let ptr = logits.dataPointer.assumingMemoryBound(to: Float16.self)
            for i in 0..<count {
                let val = Float(ptr[i])
                if val > maxVal {
                    maxVal = val
                    maxIdx = Int32(i)
                }
            }
        case .float32:
            let ptr = logits.dataPointer.assumingMemoryBound(to: Float.self)
            for i in 0..<count {
                if ptr[i] > maxVal {
                    maxVal = ptr[i]
                    maxIdx = Int32(i)
                }
            }
        default:
            let ptr = logits.dataPointer.assumingMemoryBound(to: Float.self)
            for i in 0..<count {
                if ptr[i] > maxVal {
                    maxVal = ptr[i]
                    maxIdx = Int32(i)
                }
            }
        }
        return maxIdx
    }

    // MARK: - Audio Embedding Injection

    /// Convert MLXArray audio embeddings to MLMultiArray for decoder input.
    ///
    /// Audio embeddings from the CoreML encoder are [1, T, 1024].
    /// Each position is fed to the decoder one at a time during prefill.
    public func audioEmbeddingToMultiArray(_ embedding: MLXArray, at index: Int) throws -> MLMultiArray {
        // embedding: [1, T, hidden_size] — extract [1, 1, hidden_size] at index
        let hidden = embedding.dim(2)
        let result = try MLMultiArray(shape: [1, 1, hidden as NSNumber], dataType: .float32)
        let ptr = result.dataPointer.assumingMemoryBound(to: Float.self)

        // Extract the slice at index
        let slice = embedding[0..., index..<(index + 1), 0...]
        let data: [Float] = slice.asArray(Float.self)
        for i in 0..<hidden {
            ptr[i] = data[i]
        }
        return result
    }

    /// Extract audio embedding at index from MLMultiArray (no MLX dependency).
    ///
    /// This is the MLX-free equivalent of `audioEmbeddingToMultiArray(_:at:)`.
    /// Used by `transcribeWithoutMLX()` to avoid any Metal GPU evaluation,
    /// making it safe for iOS background execution.
    ///
    /// - Parameters:
    ///   - embeddings: Audio embeddings as MLMultiArray with shape `[1, T, hidden_size]`
    ///   - index: Time-step index to extract (0..<T)
    /// - Returns: MLMultiArray with shape `[1, 1, hidden_size]` for a single time step
    public func audioEmbeddingFromMultiArray(_ embeddings: MLMultiArray, at index: Int) throws -> MLMultiArray {
        let hidden = embeddings.shape[2].intValue
        let result = try MLMultiArray(shape: [1, 1, hidden as NSNumber], dataType: .float32)
        let srcPtr = embeddings.dataPointer.assumingMemoryBound(to: Float.self)
        let dstPtr = result.dataPointer.assumingMemoryBound(to: Float.self)
        let offset = index * hidden
        for i in 0..<hidden {
            dstPtr[i] = srcPtr[offset + i]
        }
        return result
    }

    // MARK: - Helpers

    private static func findModel(named name: String, in directory: URL) -> URL? {
        let compiled = directory.appendingPathComponent("\(name).mlmodelc", isDirectory: true)
        if FileManager.default.fileExists(atPath: compiled.path) {
            return compiled
        }
        let pkg = directory.appendingPathComponent("\(name).mlpackage", isDirectory: true)
        if FileManager.default.fileExists(atPath: pkg.path) {
            return pkg
        }
        return nil
    }
}
#endif
