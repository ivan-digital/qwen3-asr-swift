#if canImport(CoreML)
import CoreML
import Foundation
import AudioCommon

/// Qwen3-TTS CoreML inference with 6-model ANE-optimized architecture.
///
/// Models: TextProjector, CodeEmbedder, MultiCodeEmbedder, CodeDecoder,
///         MultiCodeDecoder, SpeechDecoder
public final class Qwen3TTSCoreMLModel {
    public static let defaultModelId = "aufklarer/Qwen3-TTS-CoreML"

    private var codeDecoder: TalkerGenerator?
    private var multiCodeDecoder: MultiCodeDecoderCoreML?
    private var speechDecoder: SpeechDecoderCoreML?
    private var textProjector: TextProjectorModel?
    private var codeEmbedder: CodeEmbedderModel?
    private var multiCodeEmbedder: MultiCodeEmbedderModel?
    private var tokenizer: Qwen3Tokenizer?

    // Pre-computed special embeddings [1, 1024, 1, 1]
    private var ttsPadEmbed: MLMultiArray?
    private var ttsBosEmbed: MLMultiArray?
    private var ttsEosEmbed: MLMultiArray?
    public var speakerEmbedding: MLMultiArray?

    private let hiddenSize = 1024
    private let codecVocabSize = 3072
    private let codecEos = 2150

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
                modelId: modelId, to: cacheDir,
                additionalFiles: [
                    "TextProjector.mlmodelc/**", "CodeEmbedder.mlmodelc/**",
                    "MultiCodeEmbedder.mlmodelc/**", "CodeDecoder.mlmodelc/**",
                    "MultiCodeDecoder.mlmodelc/**", "SpeechDecoder.mlmodelc/**",
                    "speaker_embedding.npy", "tts_pad_embed.npy",
                    "tts_bos_embed.npy", "tts_eos_embed.npy",
                    "config.json", "vocab.json", "merges.txt",
                ]
            ) { progress in progressHandler?(progress * 0.7, "Downloading model...") }
        }

        // Embedders on CPU (FP32 precision for accumulation, matching TTSKit)
        // CodeDecoder/MCD/SD on default (let CoreML decide CPU/GPU/ANE)
        let cpuConfig = MLModelConfiguration()
        cpuConfig.computeUnits = .cpuOnly

        // With MLState, KV cache stays on ANE — use all compute units
        let defaultConfig = MLModelConfiguration()
        defaultConfig.computeUnits = .cpuAndGPU

        let model = Qwen3TTSCoreMLModel()

        progressHandler?(0.7, "Loading models...")

        // Load 6 models
        func loadML(_ name: String, _ cfg: MLModelConfiguration = defaultConfig) throws -> MLModel {
            // Try pre-compiled .mlmodelc first
            let compiledURL = cacheDir.appendingPathComponent("\(name).mlmodelc", isDirectory: true)
            if FileManager.default.fileExists(atPath: compiledURL.path) {
                return try MLModel(contentsOf: compiledURL, configuration: cfg)
            }
            // Compile from .mlpackage at runtime
            let pkgURL = cacheDir.appendingPathComponent("\(name).mlpackage", isDirectory: true)
            let compiled = try MLModel.compileModel(at: pkgURL)
            return try MLModel(contentsOf: compiled, configuration: cfg)
        }

        model.textProjector = TextProjectorModel(model: try loadML("TextProjector", cpuConfig))
        model.codeEmbedder = CodeEmbedderModel(model: try loadML("CodeEmbedder", cpuConfig))
        model.multiCodeEmbedder = MultiCodeEmbedderModel(model: try loadML("MultiCodeEmbedder", cpuConfig))
        model.codeDecoder = TalkerGenerator(model: try loadML("CodeDecoder"))
        model.multiCodeDecoder = MultiCodeDecoderCoreML(model: try loadML("MultiCodeDecoder"))
        model.speechDecoder = SpeechDecoderCoreML(model: try loadML("SpeechDecoder"))

        progressHandler?(0.9, "Loading embeddings...")

        // Load special embeddings from .npy or compute from TextProjector
        func loadNpy(_ name: String) -> MLMultiArray? {
            let url = cacheDir.appendingPathComponent("\(name).npy")
            guard let data = try? Data(contentsOf: url), data.count > 10 else { return nil }
            var headerEnd = 10
            for i in 8..<min(256, data.count) { if data[i] == 0x0A { headerEnd = i + 1; break } }
            let floatData = data.subdata(in: headerEnd..<data.count)
            let count = floatData.count / 4
            // Keep FP32 to match Python pipeline (cast to FP16 happens at model input)
            let result = try! MLMultiArray(shape: [1, NSNumber(value: count), 1, 1], dataType: .float32)
            let dst = result.dataPointer.assumingMemoryBound(to: Float.self)
            floatData.withUnsafeBytes { raw in
                let src = raw.bindMemory(to: Float.self)
                for i in 0..<count { dst[i] = src[i] }
            }
            return result
        }

        // Load from npy (PyTorch FP32 precision, matches Python inference pipeline)
        model.ttsPadEmbed = loadNpy("tts_pad_embed")
        model.ttsBosEmbed = loadNpy("tts_bos_embed")
        model.ttsEosEmbed = loadNpy("tts_eos_embed")
        model.speakerEmbedding = loadNpy("speaker_embedding")

        // Load tokenizer
        let tokenizer = Qwen3Tokenizer()
        let vocabURL = cacheDir.appendingPathComponent("vocab.json")
        if FileManager.default.fileExists(atPath: vocabURL.path) {
            try tokenizer.load(from: vocabURL)
        }
        model.tokenizer = tokenizer

        progressHandler?(1.0, "Ready")
        return model
    }

    // MARK: - Synthesis

    public func synthesize(
        text: String,
        language: String = "english",
        temperature: Float = 0.8,
        topK: Int = 50,
        maxTokens: Int = 125,
        repetitionPenalty: Float = 1.05
    ) throws -> [Float] {
        guard let codeDecoder, let multiCodeDecoder, let speechDecoder,
              let textProjector, let codeEmbedder, let multiCodeEmbedder,
              let tokenizer, let ttsPadEmbed, let ttsBosEmbed, let ttsEosEmbed else {
            throw TTSCoreMLError.modelNotLoaded
        }

        // Build non-streaming prefill (all text in prefill)
        let prefillEmbeds = try PromptBuilder.build(
            text: text, language: language, tokenizer: tokenizer,
            textProjector: textProjector, codeEmbedder: codeEmbedder,
            ttsPadEmbed: ttsPadEmbed, ttsBosEmbed: ttsBosEmbed,
            ttsEosEmbed: ttsEosEmbed, speakerEmbedding: speakerEmbedding)

        // Cap decode tokens: min(requested, 8× prefill, remaining KV cache slots)
        let maxStepsByPrefill = 8 * prefillEmbeds.count
        let cacheSlots = 256 - prefillEmbeds.count
        let effectiveMaxTokens = min(maxTokens, min(maxStepsByPrefill, cacheSlots))

        // Reset CodeDecoder KV cache
        codeDecoder.resetCache()

        // Prefill: run all positions through CodeDecoder
        var lastLogits = [Float]()
        var lastHidden = try MLMultiArray(shape: [1, 1024, 1, 1], dataType: .float16)
        for embed in prefillEmbeds {
            (lastLogits, _) = try codeDecoder.forward(embedArray: embed)
        }
        lastHidden = codeDecoder.lastHiddenState!

        // Build suppress token set: [2048, 3072) except EOS (matching TTSKit)
        // Suppress EOS for first token (min_new_tokens=2)
        lastLogits[codecEos] = -1e9
        var nextToken = TTSSampler.sample(
            logits: lastLogits, temperature: temperature, topK: topK,
            suppressRange: (2048, 3072), eosTokenId: codecEos)

        var allCodebooks = (0..<16).map { _ in [Int32]() }
        allCodebooks[0].append(nextToken)
        var generatedCB0 = [nextToken]

        // MultiCodeDecoder: predict CB1-15 for first frame
        var cpTokens = try multiCodeDecoder.predict(
            hiddenState: lastHidden, cb0Token: nextToken,
            codeEmbedder: codeEmbedder, multiCodeEmbedder: multiCodeEmbedder,
            temperature: temperature, topK: topK)
        for (i, token) in cpTokens.enumerated() {
            allCodebooks[i + 1].append(token)
        }

        // Autoregressive decode loop (capped at effectiveMaxTokens)
        for step in 1..<effectiveMaxTokens {
            // Step input = sum(all 16 codec embeddings) + tts_pad
            // Accumulate in FP32 to match Python's numpy precision, cast to FP16 at the end
            var sum32 = [Float](repeating: 0, count: hiddenSize)

            // Add CB0 embedding
            let cb0Emb = ensureNCHW(try codeEmbedder.embed(Int(nextToken)), channels: hiddenSize)
            let cb0Ptr = cb0Emb.dataPointer.assumingMemoryBound(to: Float16.self)
            for i in 0..<hiddenSize { sum32[i] += Float(cb0Ptr[i]) }

            // Add CB1-15 embeddings
            for (cbIdx, token) in cpTokens.enumerated() {
                let mceEmb = ensureNCHW(
                    try multiCodeEmbedder.embed(codebookIdx: cbIdx, tokenId: Int(token)),
                    channels: hiddenSize)
                let ptr = mceEmb.dataPointer.assumingMemoryBound(to: Float16.self)
                for i in 0..<hiddenSize { sum32[i] += Float(ptr[i]) }
            }

            // Add tts_pad (FP32 from npy)
            if ttsPadEmbed.dataType == .float32 {
                let padPtr = ttsPadEmbed.dataPointer.assumingMemoryBound(to: Float.self)
                for i in 0..<hiddenSize { sum32[i] += padPtr[i] }
            } else {
                let padPtr = ttsPadEmbed.dataPointer.assumingMemoryBound(to: Float16.self)
                for i in 0..<hiddenSize { sum32[i] += Float(padPtr[i]) }
            }

            // Cast to FP16 for model input
            let stepInput = try MLMultiArray(shape: [1, NSNumber(value: hiddenSize), 1, 1], dataType: .float16)
            let outPtr = stepInput.dataPointer.assumingMemoryBound(to: Float16.self)
            for i in 0..<hiddenSize { outPtr[i] = Float16(sum32[i]) }

            // CodeDecoder forward
            (lastLogits, _) = try codeDecoder.forward(embedArray: stepInput)
            lastHidden = codeDecoder.lastHiddenState!

            // Sample CB0 — suppress control tokens [2048, 3072) except EOS
            let eosLogit = lastLogits[codecEos]
            for i in 2048..<codecVocabSize { if i != codecEos { lastLogits[i] = -1e9 } }
            if step < 2 { lastLogits[codecEos] = -1e9 }  // min_new_tokens=2
            else { lastLogits[codecEos] = eosLogit }

            // Repetition penalty
            for t in Set(generatedCB0) {
                let idx = Int(t)
                if lastLogits[idx] > 0 { lastLogits[idx] /= repetitionPenalty }
                else { lastLogits[idx] *= repetitionPenalty }
            }

            nextToken = TTSSampler.sample(
                logits: lastLogits, temperature: temperature, topK: topK)
            generatedCB0.append(nextToken)

            if nextToken == Int32(codecEos) { break }
            allCodebooks[0].append(nextToken)

            // MultiCodeDecoder: predict CB1-15
            cpTokens = try multiCodeDecoder.predict(
                hiddenState: lastHidden, cb0Token: nextToken,
                codeEmbedder: codeEmbedder, multiCodeEmbedder: multiCodeEmbedder,
                temperature: temperature, topK: topK)
            for (i, token) in cpTokens.enumerated() {
                allCodebooks[i + 1].append(token)
            }
        }

        // SpeechDecoder: all codes → audio
        guard !allCodebooks[0].isEmpty else { return [] }
        var audio = try speechDecoder.decode(codes: allCodebooks)

        // Normalize amplitude
        let peak = audio.map { abs($0) }.max() ?? 0
        if peak > 0.001 {
            let gain = min(0.9 / peak, 10.0)
            for i in 0..<audio.count { audio[i] *= gain }
        }

        return audio
    }

    public func unload() {
        codeDecoder = nil; multiCodeDecoder = nil; speechDecoder = nil
        textProjector = nil; codeEmbedder = nil; multiCodeEmbedder = nil
        tokenizer = nil
    }

    public enum TTSCoreMLError: Error {
        case modelNotLoaded
    }
}
#endif
