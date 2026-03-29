#if canImport(CoreML)
import CoreML
import Foundation
import AudioCommon

/// Qwen3-TTS via CoreML — runs on Neural Engine for iOS/macOS.
///
/// Three CoreML models:
/// - Talker: 28-layer transformer, autoregressive codec token generation
/// - CodePredictor: 5-layer transformer, predicts 15 residual codebooks per step
/// - MimiDecoder: Convolutional vocoder, codebook indices → 24kHz waveform
public final class Qwen3TTSCoreMLModel {

    public static let defaultModelId = "aufklarer/Qwen3-TTS-CoreML"

    private var talker: TalkerGenerator?
    private var codePredictor: CodePredictorCoreML?
    private var mimiDecoder: MimiDecoderCoreML?
    private var embeddings: EmbeddingManager?
    private var tokenizer: Qwen3Tokenizer?

    private let hiddenSize = 1024
    private let codecVocabSize = 3072
    private let codecEos = 2150
    private let codecPad = 2148

    /// Load pretrained model from HuggingFace.
    public static func fromPretrained(
        modelId: String = defaultModelId,
        localPath: String? = nil,
        computeUnits: MLComputeUnits = .all,
        progressHandler: ((Double, String) -> Void)? = nil
    ) async throws -> Qwen3TTSCoreMLModel {
        let cacheDir: URL
        if let localPath {
            cacheDir = URL(fileURLWithPath: localPath, isDirectory: true)
        } else {
            cacheDir = try HuggingFaceDownloader.getCacheDirectory(for: modelId)
            progressHandler?(0.0, "Downloading model...")
            try await HuggingFaceDownloader.downloadWeights(
                modelId: modelId,
                to: cacheDir,
                additionalFiles: [
                    "Talker.mlmodelc/**",
                    "CodePredictor.mlmodelc/**",
                    "MimiDecoder.mlmodelc/**",
                    "embeddings.safetensors",
                    "config.json",
                    "vocab.json",
                    "merges.txt",
                ]
            ) { progress in
                progressHandler?(progress * 0.7, "Downloading model...")
            }
        }

        let config = MLModelConfiguration()
        config.computeUnits = computeUnits

        progressHandler?(0.7, "Loading Talker...")
        let talkerURL = cacheDir.appendingPathComponent("Talker.mlmodelc", isDirectory: true)
        let talkerMLModel = try MLModel(contentsOf: talkerURL, configuration: config)

        progressHandler?(0.8, "Loading CodePredictor...")
        let cpURL = cacheDir.appendingPathComponent("CodePredictor.mlmodelc", isDirectory: true)
        // CP uses cpuAndNeuralEngine to match Talker, or cpuOnly for debugging NaN
        let cpConfig = MLModelConfiguration()
        cpConfig.computeUnits = .cpuAndNeuralEngine
        let cpModel = try MLModel(contentsOf: cpURL, configuration: cpConfig)

        progressHandler?(0.85, "Loading MimiDecoder...")
        let decoderURL = cacheDir.appendingPathComponent("MimiDecoder.mlmodelc", isDirectory: true)
        // MimiDecoder uses cpuAndNeuralEngine to avoid MPS graph compiler crash
        // on GreaterThanOp constant folding (Apple bug in MetalPerformanceShadersGraph)
        let decoderConfig = MLModelConfiguration()
        decoderConfig.computeUnits = .cpuAndNeuralEngine
        let decoderModel = try MLModel(contentsOf: decoderURL, configuration: decoderConfig)

        progressHandler?(0.9, "Loading embeddings...")
        let embURL = cacheDir.appendingPathComponent("embeddings.safetensors")
        let embeddings = try EmbeddingManager(embeddingsURL: embURL)

        // Load tokenizer
        let tokenizer = Qwen3Tokenizer()
        let vocabURL = cacheDir.appendingPathComponent("vocab.json")
        if FileManager.default.fileExists(atPath: vocabURL.path) {
            try tokenizer.load(from: vocabURL)
        }

        let model = Qwen3TTSCoreMLModel()
        model.talker = TalkerGenerator(model: talkerMLModel)
        let cp = CodePredictorCoreML(model: cpModel)
        cp.loadWeights(lmHeads: embeddings.cpLMHeadWeights,
                        codecEmbeddings: embeddings.cpCodecEmbeddings)
        model.codePredictor = cp
        model.mimiDecoder = MimiDecoderCoreML(model: decoderModel)
        model.embeddings = embeddings
        model.tokenizer = tokenizer

        progressHandler?(1.0, "Ready")
        return model
    }

    /// Synthesize speech from text.
    ///
    /// - Parameters:
    ///   - text: Text to synthesize
    ///   - language: Language name (e.g., "english", "chinese")
    ///   - temperature: Sampling temperature (0 = greedy)
    ///   - topK: Top-k filtering
    ///   - maxTokens: Maximum codec tokens to generate
    /// - Returns: Audio samples at 24kHz, mono Float32
    public func synthesize(
        text: String,
        language: String = "english",
        temperature: Float = 0.6,
        topK: Int = 50,
        maxTokens: Int = 500,
        eosLogitBias: Float = 0.0,
        repetitionPenalty: Float = 1.3
    ) throws -> [Float] {
        guard let talker, let codePredictor, let mimiDecoder,
              let embeddings, let tokenizer else {
            throw TTSCoreMLError.modelNotLoaded
        }

        // Build prompt embeddings
        let (prefillEmbeds, trailingTextEmbeds, ttsPadEmbed, _) = PromptBuilder.build(
            text: text, language: language, tokenizer: tokenizer, embeddings: embeddings)

        // Reset KV cache
        talker.resetCache()

        // Prefill: run all prompt tokens through the Talker sequentially
        let prefillLen = prefillEmbeds.shape[1].intValue
        var prefillSequence = [[Float16]]()
        let prefillPtr = prefillEmbeds.dataPointer.assumingMemoryBound(to: Float16.self)
        for t in 0..<prefillLen {
            prefillSequence.append(Array(UnsafeBufferPointer(
                start: prefillPtr.advanced(by: t * hiddenSize), count: hiddenSize)))
        }
        let (prefillLogits, prefillHidden) = try talker.prefill(embeds: prefillSequence)

        // Sample first codec token
        var nextToken = TTSSampler.sample(
            logits: prefillLogits, temperature: temperature, topK: topK,
            suppressRange: (2048, 3072), eosTokenId: codecEos)

        // Initialize codebook accumulator: [16][T]
        var allCodebooks = (0..<16).map { _ in [Int32]() }
        allCodebooks[0].append(nextToken)

        // Code Predictor for first timestep
        let firstHiddenArray = makeFloat16Array(prefillHidden)
        let firstCodeEmbed = makeFloat16Array(embeddings.codecEmbed(Int(nextToken)))
        let cpTokens = try codePredictor.predict(
            hiddenState: firstHiddenArray, firstCodeEmbed: firstCodeEmbed,
            temperature: temperature, topK: topK,
            repetitionPenalty: repetitionPenalty)
        for (i, token) in cpTokens.enumerated() {
            allCodebooks[i + 1].append(token)
        }

        // Autoregressive decode loop
        var trailingIdx = 0
        var prevCPTokens = cpTokens

        for step in 1..<maxTokens {
            // Text embedding for this step
            let textEmbed: [Float16]
            if trailingIdx < trailingTextEmbeds.count {
                textEmbed = trailingTextEmbeds[trailingIdx]
                trailingIdx += 1
            } else {
                textEmbed = ttsPadEmbed
            }

            // Codec embedding: first codebook + sum of 15 CP group embeddings
            let codecEmbed = embeddings.codecEmbed(Int(nextToken))
            let cpGroupEmbed = embeddings.cpGroupEmbedSum(prevCPTokens)

            // Combined step embedding: text + codec + cp_group (element-wise sum)
            var stepEmbed = [Float16](repeating: 0, count: hiddenSize)
            for i in 0..<hiddenSize {
                stepEmbed[i] = Float16(Float(textEmbed[i]) + Float(codecEmbed[i]) + Float(cpGroupEmbed[i]))
            }

            // Run one decode step
            let (stepLogits, stepHidden) = try talker.forward(embed: stepEmbed)

            // EOS bias: ramp after text tokens exhausted
            let textDone = trailingIdx >= trailingTextEmbeds.count
            let stepsAfterText = textDone ? max(0, step - trailingTextEmbeds.count) : 0
            let minTokens = max(trailingTextEmbeds.count + 3, 8)
            let biasActive = textDone && step >= minTokens
            let dynamicBias = eosLogitBias + (biasActive ? 20.0 + Float(stepsAfterText) * 3.0 : 0.0)

            nextToken = TTSSampler.sample(
                logits: stepLogits, temperature: temperature, topK: topK,
                repetitionPenalty: repetitionPenalty, generatedTokens: allCodebooks[0],
                suppressRange: (2048, 3072), eosTokenId: codecEos, eosLogitBias: dynamicBias)
            if nextToken == Int32(codecEos) { break }

            allCodebooks[0].append(nextToken)

            // Code Predictor
            let hiddenArray = makeFloat16Array(stepHidden)
            let codeEmbedArray = makeFloat16Array(embeddings.codecEmbed(Int(nextToken)))
            prevCPTokens = try codePredictor.predict(
                hiddenState: hiddenArray, firstCodeEmbed: codeEmbedArray,
                temperature: temperature, topK: topK,
                repetitionPenalty: repetitionPenalty)
            for (i, token) in prevCPTokens.enumerated() {
                allCodebooks[i + 1].append(token)
            }
        }

        // Decode all codebooks to audio
        guard !allCodebooks[0].isEmpty else {
            return []
        }

        var audio = try mimiDecoder.decode(codes: allCodebooks)

        // Normalize amplitude: CoreML FP16 decoder produces ~2.5x lower
        // amplitude than MLX. Scale to match expected output level.
        let peak = audio.map { abs($0) }.max() ?? 0
        if peak > 0.001 {
            let targetPeak: Float = 0.9
            let gain = min(targetPeak / peak, 10.0)  // cap at 10x to avoid amplifying noise
            for i in 0..<audio.count { audio[i] *= gain }
        }

        return audio
    }

    /// Unload all models to free memory.
    public func unload() {
        talker = nil
        codePredictor = nil
        mimiDecoder = nil
        embeddings = nil
        tokenizer = nil
    }

    // MARK: - Helpers

    /// Convert [Float16] array to MLMultiArray [1, 1, hiddenSize] for Code Predictor input.
    private func makeFloat16Array(_ embed: [Float16]) -> MLMultiArray {
        let array = try! MLMultiArray(shape: [1, 1, NSNumber(value: hiddenSize)], dataType: .float16)
        let ptr = array.dataPointer.assumingMemoryBound(to: Float16.self)
        for i in 0..<hiddenSize { ptr[i] = embed[i] }
        return array
    }

    public enum TTSCoreMLError: Error {
        case modelNotLoaded
        case generationFailed(String)
    }
}
#endif
