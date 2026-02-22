import Foundation
import MLX
import MLXNN
import Qwen3Common

// MARK: - PersonaPlex Model

public final class PersonaPlexModel: Module {
    public let cfg: PersonaPlexConfig

    @ModuleInfo public var temporal: TemporalTransformer
    @ModuleInfo public var depformer: Depformer
    public let mimi: Mimi

    public init(cfg: PersonaPlexConfig = .default) {
        self.cfg = cfg
        self._temporal = ModuleInfo(wrappedValue: TemporalTransformer(cfg: cfg.temporal))
        self._depformer = ModuleInfo(wrappedValue: Depformer(cfg: cfg.depformer, temporalDim: cfg.temporal.dim))
        self.mimi = Mimi(cfg: cfg.mimi)
    }

    // MARK: - Offline Inference

    /// Process user audio and generate response audio.
    /// - Parameters:
    ///   - userAudio: [numSamples] float array of 24kHz mono audio
    ///   - voice: voice preset for the agent
    ///   - maxSteps: maximum generation steps (at 12.5 Hz)
    ///   - verbose: print timing info
    /// - Returns: [numSamples] float array of 24kHz response audio
    public func respond(
        userAudio: [Float],
        voice: PersonaPlexVoice = .NATM0,
        maxSteps: Int = 500,
        verbose: Bool = false
    ) -> [Float] {
        let startTime = CFAbsoluteTimeGetCurrent()

        // 1. Encode user audio with Mimi
        let audioArray = MLXArray(userAudio).reshaped([1, 1, userAudio.count])
        let userCodes = mimi.encode(audioArray)  // [1, numCodebooks, T]
        eval(userCodes)

        let prefillLen = userCodes.shape[2]
        if verbose {
            let encTime = CFAbsoluteTimeGetCurrent() - startTime
            print("  Mimi encode: \(String(format: "%.2f", encTime))s, \(prefillLen) frames")
        }

        // 2. Reset caches
        temporal.resetCache()
        mimi.resetState()

        // 3. Initialize delay buffer
        let delays = cfg.delays
        let maxDelay = cfg.maxDelay
        let numStreams = cfg.numStreams

        // Token cache: [numStreams, maxLen]
        // Stream 0 = text, streams 1-8 = user audio, streams 9-16 = agent audio
        let totalLen = prefillLen + maxSteps + maxDelay + 3
        var tokenCache = [[Int32]](repeating: [Int32](repeating: -1, count: totalLen), count: numStreams)

        // Fill user audio tokens into cache
        let userCodesArr = userCodes.asType(.int32)
        eval(userCodesArr)
        for cb in 0..<min(cfg.temporal.nQ, userCodes.shape[1]) {
            let row = userCodesArr[0, cb]  // [T]
            let rowValues: [Int32] = (0..<prefillLen).map { t in
                row[t].item(Int32.self)
            }
            for t in 0..<prefillLen {
                tokenCache[1 + cb][t + delays[1 + cb]] = rowValues[t]
            }
        }

        // Initialize text stream with padding
        for t in 0..<prefillLen {
            tokenCache[0][t + delays[0]] = Int32(cfg.temporal.textPaddingId)
        }

        // Initialize agent audio streams with initial tokens
        for cb in 0..<cfg.temporal.nQ {
            let streamIdx = 1 + cfg.temporal.nQ + cb  // 9-16
            for t in 0..<prefillLen {
                tokenCache[streamIdx][t + delays[streamIdx]] = Int32(cfg.temporal.initialTokenId)
            }
        }

        // 4. Autoregressive generation
        var agentTokens: [[Int32]] = Array(repeating: [], count: cfg.depformer.numSteps)
        let genStart = CFAbsoluteTimeGetCurrent()

        for step in 0..<(prefillLen + maxSteps) {
            // Build input tokens for this step
            let textTokenArr = MLXArray([tokenCache[0][step]]).reshaped([1, 1])
            var audioTokenArrs: [MLXArray] = []
            for stream in 1..<numStreams {
                let tok = tokenCache[stream][step]
                let effectiveTok = tok >= 0 ? tok : Int32(cfg.temporal.initialTokenId)
                audioTokenArrs.append(MLXArray([effectiveTok]))
            }
            let audioTokens = stacked(audioTokenArrs, axis: 0).reshaped([1, numStreams - 1, 1])

            // Forward through temporal transformer
            let (hidden, textLogits) = temporal.forward(
                textTokens: textTokenArr,
                audioTokens: audioTokens,
                offset: step
            )
            eval(hidden, textLogits)

            // Sample text token
            let textToken = sampleTopK(
                logits: textLogits.squeezed(axis: 1),
                temperature: cfg.sampling.textTemp,
                topK: cfg.sampling.textTopK
            )
            eval(textToken)

            // Generate agent audio tokens via depformer
            let agentCodes = depformer.generate(
                temporalHidden: hidden,
                textToken: textToken
            ) { logits in
                sampleTopK(
                    logits: logits,
                    temperature: cfg.sampling.audioTemp,
                    topK: cfg.sampling.audioTopK
                )
            }
            eval(agentCodes)

            // Write generated tokens into cache for next step
            let nextStep = step + 1
            if nextStep < totalLen {
                // Text token
                let textVal = textToken[0].item(Int32.self)
                tokenCache[0][nextStep + delays[0]] = textVal

                // Agent audio tokens
                let agentArr = agentCodes[0]  // [numSteps]
                for cb in 0..<cfg.depformer.numSteps {
                    let agentStreamIdx = 1 + cfg.temporal.nQ + cb
                    if agentStreamIdx < numStreams {
                        let tok = agentArr[cb].item(Int32.self)
                        let delayedPos = nextStep + delays[agentStreamIdx]
                        if delayedPos < totalLen {
                            tokenCache[agentStreamIdx][delayedPos] = tok
                        }
                    }
                    // Only collect agent tokens after prefill
                    if step >= prefillLen {
                        let tok = agentArr[cb].item(Int32.self)
                        agentTokens[cb].append(tok)
                    }
                }
            }
        }

        if verbose {
            let genTime = CFAbsoluteTimeGetCurrent() - genStart
            let genSteps = prefillLen + maxSteps
            let msPerStep = genTime / Double(genSteps) * 1000
            print("  Generation: \(String(format: "%.2f", genTime))s, \(String(format: "%.1f", msPerStep))ms/step")
        }

        // 5. Decode agent tokens with Mimi
        let decStart = CFAbsoluteTimeGetCurrent()
        let numAgentFrames = agentTokens[0].count
        guard numAgentFrames > 0 else { return [] }

        // Build [1, numCodebooks, T] tensor for Mimi decoder
        // Use only the first 8 codebooks (Mimi uses 16, but agent only generates 8)
        // Pad remaining codebooks with zeros
        var codeMatrix = [[Int32]](repeating: [Int32](repeating: 0, count: numAgentFrames),
                                   count: cfg.mimi.numCodebooks)
        for cb in 0..<min(cfg.depformer.numSteps, cfg.mimi.numCodebooks) {
            codeMatrix[cb] = agentTokens[cb]
        }

        let flatCodes = codeMatrix.flatMap { $0 }
        let codesArr = MLXArray(flatCodes).reshaped([1, cfg.mimi.numCodebooks, numAgentFrames])
        let decoded = mimi.decode(codesArr)  // [1, 1, numSamples]
        eval(decoded)

        if verbose {
            let decTime = CFAbsoluteTimeGetCurrent() - decStart
            print("  Mimi decode: \(String(format: "%.2f", decTime))s")
        }

        // Extract audio samples
        let numSamples = decoded.shape[2]
        var samples = [Float](repeating: 0, count: numSamples)
        let flatDecoded = decoded.reshaped([numSamples])
        eval(flatDecoded)
        for i in 0..<numSamples {
            samples[i] = flatDecoded[i].item(Float.self)
        }

        if verbose {
            let totalTime = CFAbsoluteTimeGetCurrent() - startTime
            let audioDuration = Double(numSamples) / Double(cfg.sampleRate)
            print("  Total: \(String(format: "%.2f", totalTime))s, audio: \(String(format: "%.2f", audioDuration))s, RTF: \(String(format: "%.2f", totalTime / max(audioDuration, 0.001)))")
        }

        return samples
    }

    // MARK: - Model Loading

    public static func fromPretrained(
        modelId: String = "aufklarer/PersonaPlex-7B-MLX-4bit",
        progressHandler: ((Double, String) -> Void)? = nil
    ) async throws -> PersonaPlexModel {
        let cfg = PersonaPlexConfig.default
        let model = PersonaPlexModel(cfg: cfg)

        // Download weights
        progressHandler?(0.05, "Downloading PersonaPlex weights...")
        let modelDir = try HuggingFaceDownloader.getCacheDirectory(for: modelId)

        let weightFiles = [
            "temporal.safetensors",
            "depformer.safetensors",
            "embeddings.safetensors",
            "mimi.safetensors",
            "config.json"
        ]

        try await HuggingFaceDownloader.downloadWeights(
            modelId: modelId,
            to: modelDir,
            additionalFiles: weightFiles
        ) { progress in
            progressHandler?(0.05 + progress * 0.5, "Downloading...")
        }

        // Load weights
        progressHandler?(0.55, "Loading model weights...")
        try PersonaPlexWeightLoader.loadWeights(
            model: model,
            from: modelDir
        ) { progress, status in
            progressHandler?(0.55 + progress * 0.25, status)
        }

        // Load Mimi
        progressHandler?(0.80, "Loading Mimi codec...")
        try PersonaPlexWeightLoader.loadMimi(
            model: model.mimi,
            from: modelDir
        ) { progress, status in
            progressHandler?(0.80 + progress * 0.15, status)
        }

        model.train(false)
        progressHandler?(1.0, "PersonaPlex ready")
        return model
    }
}
