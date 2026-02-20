import Foundation
import MLX
import MLXNN
import Qwen3Common
import Tokenizers
import Hub

// MARK: - Public Types

public enum MarvisTTSError: Error, LocalizedError {
    case invalidArgument(String)
    case voiceNotFound
    case invalidRefAudio(String)
    case downloadFailed(String)
    case modelLoadFailed(String)

    public var errorDescription: String? {
        switch self {
        case .invalidArgument(let msg): return "Invalid argument: \(msg)"
        case .voiceNotFound: return "Voice prompt files not found"
        case .invalidRefAudio(let msg): return "Invalid reference audio: \(msg)"
        case .downloadFailed(let msg): return "Download failed: \(msg)"
        case .modelLoadFailed(let msg): return "Model load failed: \(msg)"
        }
    }
}

public struct AudioChunk: Sendable {
    public let audio: [Float]
    public let sampleRate: Int
    public let sampleCount: Int
    public let frameCount: Int
    public let audioDuration: TimeInterval
    public let realTimeFactor: Double
    public let processingTime: Double
}

// MARK: - MarvisTTSModel

public final class MarvisTTSModel {
    public enum Voice: String, Sendable {
        case conversationalA = "conversational_a"
        case conversationalB = "conversational_b"
    }

    public enum QualityLevel: Int, Sendable {
        case low = 8
        case medium = 16
        case high = 24
        case maximum = 32
    }

    public let sampleRate: Int

    private let model: CSMModel
    private let promptURLs: [URL]
    private let textTokenizer: Tokenizer
    private let audioTokenizer: MimiTokenizer
    private let streamingDecoder: MimiStreamingDecoder

    public init(
        model: CSMModel,
        promptURLs: [URL],
        textTokenizer: Tokenizer,
        audioTokenizer: MimiTokenizer
    ) {
        self.model = model
        self.promptURLs = promptURLs
        self.textTokenizer = textTokenizer
        self.audioTokenizer = audioTokenizer
        self.streamingDecoder = MimiStreamingDecoder(audioTokenizer.codec)
        self.sampleRate = Int(audioTokenizer.codec.cfg.sampleRate)
        model.resetCaches()
    }

    // MARK: - Public API

    public static func fromPretrained(
        modelId: String = "Marvis-AI/marvis-tts-250m-v0.2-MLX-8bit",
        progressHandler: ((Double, String) -> Void)? = nil
    ) async throws -> MarvisTTSModel {
        progressHandler?(0.05, "Downloading CSM model...")

        // Download model files
        let modelDir = try HuggingFaceDownloader.getCacheDirectory(for: modelId)
        try await HuggingFaceDownloader.downloadWeights(
            modelId: modelId, to: modelDir,
            additionalFiles: [
                "tokenizer.json", "tokenizer_config.json",
                "special_tokens_map.json",
            ]
        ) { p in progressHandler?(0.05 + p * 0.25, "Downloading model...") }

        // Download prompt files
        try await downloadPromptFiles(modelId: modelId, to: modelDir)

        progressHandler?(0.35, "Loading config...")
        let configURL = modelDir.appendingPathComponent("config.json")
        let args = try JSONDecoder().decode(CSMModelArgs.self, from: Data(contentsOf: configURL))

        // Determine quantization
        var groupSize: Int? = nil
        var bits: Int? = nil
        if let q = args.quantization,
           let gs = q["group_size"], case .number(let g) = gs,
           let b = q["bits"], case .number(let bVal) = b {
            groupSize = Int(g)
            bits = Int(bVal)
        }

        progressHandler?(0.40, "Building model...")
        let model = CSMModel(config: args, groupSize: groupSize, bits: bits)

        progressHandler?(0.45, "Loading weights...")
        var weights = try loadMarvisWeights(from: modelDir)
        weights = sanitizeCSMWeights(weights)

        let parameters = ModuleParameters.unflattened(weights)
        try model.update(parameters: parameters, verify: .noUnusedKeys)
        eval(model)

        // Load tokenizer
        progressHandler?(0.70, "Loading tokenizer...")
        let textTokenizer = try await AutoTokenizer.from(modelFolder: modelDir)

        // Load Mimi codec
        progressHandler?(0.75, "Loading Mimi codec...")
        let codec = try await Mimi.fromPretrained { p, status in
            progressHandler?(0.75 + p * 0.20, status)
        }
        let audioTokenizer = MimiTokenizer(codec)

        // Find prompt files
        let promptDir = modelDir.appendingPathComponent("prompts", isDirectory: true)
        var audioPromptURLs: [URL] = []
        if FileManager.default.fileExists(atPath: promptDir.path) {
            let contents = try FileManager.default.contentsOfDirectory(at: promptDir, includingPropertiesForKeys: nil)
            audioPromptURLs = contents.filter { $0.pathExtension == "wav" }
        }

        progressHandler?(1.0, "Ready")

        return MarvisTTSModel(
            model: model, promptURLs: audioPromptURLs,
            textTokenizer: textTokenizer, audioTokenizer: audioTokenizer)
    }

    public func synthesize(
        text: String,
        voice: Voice? = .conversationalA,
        quality: QualityLevel = .maximum,
        refAudio: MLXArray? = nil,
        refText: String? = nil
    ) async throws -> [Float] {
        var allAudio: [Float] = []
        for try await chunk in synthesizeStream(
            text: text, voice: voice, quality: quality,
            refAudio: refAudio, refText: refText, streamingInterval: 999.0
        ) {
            allAudio.append(contentsOf: chunk.audio)
        }
        return allAudio
    }

    public func synthesizeStream(
        text: String,
        voice: Voice? = .conversationalA,
        quality: QualityLevel = .maximum,
        refAudio: MLXArray? = nil,
        refText: String? = nil,
        streamingInterval: Double = 0.5
    ) -> AsyncThrowingStream<AudioChunk, Error> {
        let (stream, continuation) = AsyncThrowingStream<AudioChunk, Error>.makeStream()

        Task { [weak self] in
            guard let self else { return }
            do {
                guard voice != nil || refAudio != nil else {
                    throw MarvisTTSError.invalidArgument("voice or refAudio/refText must be specified")
                }

                let context = try self.loadContext(voice: voice, refAudio: refAudio, refText: refText)

                let maxAudioFrames = Int(60000 / 80.0) // ~12.5 fps
                let streamingIntervalTokens = Int(streamingInterval * 12.5)

                let generationText = (context.text + " " + text).trimmingCharacters(in: .whitespaces)
                let seg = Segment(speaker: 0, text: generationText, audio: context.audio)

                model.resetCaches()
                streamingDecoder.reset()

                let (toks, masks) = try tokenizeSegment(seg, addEOS: false)
                let promptTokens = toks.asType(Int32.self)
                let promptMask = masks.asType(Bool.self)

                var samplesFrames: [MLXArray] = []
                var currTokens = expandedDimensions(promptTokens, axis: 0)
                var currMask = expandedDimensions(promptMask, axis: 0)
                var currPos = expandedDimensions(MLXArray(Array(0..<promptTokens.shape[0])), axis: 0)
                var generatedCount = 0
                var yieldedCount = 0

                let sampleFn = TopPSampler(temperature: 0.9, topP: 0.8).sample
                var startTime = CFAbsoluteTimeGetCurrent()

                for _ in 0..<maxAudioFrames {
                    if Task.isCancelled { break }

                    let frame = model.generateFrame(
                        maxCodebooks: quality.rawValue,
                        tokens: currTokens, tokensMask: currMask,
                        inputPos: currPos, sampler: sampleFn)

                    // EOS if every codebook is 0
                    if frame.sum().item(Int32.self) == 0 { break }

                    samplesFrames.append(frame)

                    let zerosText = MLXArray.zeros([1, 1], type: Int32.self)
                    let nextFrame = concatenated([frame, zerosText], axis: 1)
                    currTokens = expandedDimensions(nextFrame, axis: 1)

                    let K = frame.shape[1]
                    let onesK = MLXArray.ones([1, K], type: Bool.self)
                    let zero1 = MLXArray.zeros([1, 1], type: Bool.self)
                    let nextMask = concatenated([onesK, zero1], axis: 1)
                    currMask = expandedDimensions(nextMask, axis: 1)

                    currPos = split(currPos, indices: [currPos.shape[1] - 1], axis: 1)[1] + MLXArray(1)

                    generatedCount += 1

                    if (generatedCount - yieldedCount) >= streamingIntervalTokens {
                        yieldedCount = generatedCount
                        let chunk = makeChunk(samplesFrames, start: startTime)
                        continuation.yield(chunk)
                        samplesFrames.removeAll(keepingCapacity: true)
                        startTime = CFAbsoluteTimeGetCurrent()
                    }
                }

                if !samplesFrames.isEmpty {
                    let chunk = makeChunk(samplesFrames, start: startTime)
                    continuation.yield(chunk)
                }

                continuation.finish()
            } catch {
                continuation.finish(throwing: error)
            }
        }
        return stream
    }

    // MARK: - Private

    private struct Segment {
        let speaker: Int
        let text: String
        let audio: MLXArray
    }

    private func loadContext(voice: Voice?, refAudio: MLXArray?, refText: String?) throws -> Segment {
        if let refAudio, let refText {
            return Segment(speaker: 0, text: refText, audio: refAudio)
        }
        guard let voice else { throw MarvisTTSError.voiceNotFound }

        var refAudioURL: URL?
        for promptURL in promptURLs {
            if promptURL.lastPathComponent == "\(voice.rawValue).wav" {
                refAudioURL = promptURL
            }
        }
        guard let refAudioURL else { throw MarvisTTSError.voiceNotFound }

        let refTextURL = refAudioURL.deletingPathExtension().appendingPathExtension("txt")
        guard let refText = try? String(contentsOf: refTextURL, encoding: .utf8) else {
            throw MarvisTTSError.voiceNotFound
        }

        let samples = try AudioFileLoader.load(url: refAudioURL, targetSampleRate: sampleRate)
        let audio = MLXArray(samples)
        return Segment(speaker: 0, text: refText, audio: audio)
    }

    private func tokenizeTextSegment(text: String, speaker: Int) -> (MLXArray, MLXArray) {
        let K = model.args.audioNumCodebooks
        let frameW = K + 1

        let prompt = "[\(speaker)]" + text
        let ids = MLXArray(textTokenizer.encode(text: prompt).map { Int32($0) })

        let T = ids.shape[0]
        var frame = MLXArray.zeros([T, frameW], type: Int32.self)
        var mask = MLXArray.zeros([T, frameW], type: Bool.self)

        // Put text IDs in the last column
        let lastCol = frameW - 1
        let left = split(frame, indices: [lastCol], axis: 1)[0]
        frame = concatenated([left, ids.reshaped([T, 1])], axis: 1)

        let maskLeft = split(mask, indices: [lastCol], axis: 1)[0]
        mask = concatenated([maskLeft, MLXArray.ones([T, 1], type: Bool.self)], axis: 1)

        return (frame, mask)
    }

    private func tokenizeAudio(_ audio: MLXArray, addEOS: Bool = true) -> (MLXArray, MLXArray) {
        let K = model.args.audioNumCodebooks
        let x = audio.reshaped([1, 1, audio.shape[0]])
        let encoded = encodeChunked(x)
        var codes = split(encoded, indices: [1], axis: 0)[0].reshaped([K, encoded.shape[2]])

        if addEOS {
            let eos = MLXArray.zeros([K, 1], type: Int32.self)
            codes = concatenated([codes, eos], axis: 1)
        }

        let T = codes.shape[1]
        let codesT = swappedAxes(codes, 0, 1) // [T, K]

        let frame = concatenated([codesT, MLXArray.zeros([T, 1], type: Int32.self)], axis: 1) // [T, K+1]
        let mask = concatenated([MLXArray.ones([T, K], type: Bool.self), MLXArray.zeros([T, 1], type: Bool.self)], axis: 1) // [T, K+1]

        return (frame, mask)
    }

    private func tokenizeSegment(_ segment: Segment, addEOS: Bool = true) throws -> (MLXArray, MLXArray) {
        let (txt, txtMask) = tokenizeTextSegment(text: segment.text, speaker: segment.speaker)
        let (aud, audMask) = tokenizeAudio(segment.audio, addEOS: addEOS)
        return (concatenated([txt, aud], axis: 0), concatenated([txtMask, audMask], axis: 0))
    }

    private func encodeChunked(_ xs: MLXArray, chunkSize: Int = 48_000) -> MLXArray {
        audioTokenizer.codec.resetState()
        var codes: [MLXArray] = []
        let totalLen = xs.shape[2]
        for start in stride(from: 0, to: totalLen, by: chunkSize) {
            let end = min(totalLen, start + chunkSize)
            let chunk = xs[0..<xs.shape[0], 0..<xs.shape[1], start..<end]
            codes.append(audioTokenizer.codec.encodeStep(chunk))
        }
        return MLX.concatenated(codes, axis: 2)
    }

    private func makeChunk(_ frames: [MLXArray], start: CFTimeInterval) -> AudioChunk {
        let frameCount = frames.count

        var stk = stacked(frames, axis: 0) // [F, 1, K]
        stk = swappedAxes(stk, 0, 1) // [1, F, K]
        stk = swappedAxes(stk, 1, 2) // [1, K, F]

        let audio1x1x = streamingDecoder.decodeFrames(stk)
        let sampleCount = audio1x1x.shape[2]
        let audio = audio1x1x.reshaped([sampleCount])

        let elapsed = CFAbsoluteTimeGetCurrent() - start
        let audioSeconds = Double(sampleCount) / Double(sampleRate)
        let rtf = audioSeconds > 0 ? elapsed / audioSeconds : 0.0

        return AudioChunk(
            audio: audio.asArray(Float.self),
            sampleRate: sampleRate,
            sampleCount: sampleCount,
            frameCount: frameCount,
            audioDuration: audioSeconds,
            realTimeFactor: (rtf * 100).rounded() / 100,
            processingTime: elapsed)
    }
}

// MARK: - Prompt File Download

private func downloadPromptFiles(modelId: String, to directory: URL) async throws {
    let promptDir = directory.appendingPathComponent("prompts", isDirectory: true)
    try FileManager.default.createDirectory(at: promptDir, withIntermediateDirectories: true)

    let promptFiles = [
        "prompts/conversational_a.wav", "prompts/conversational_a.txt",
        "prompts/conversational_b.wav", "prompts/conversational_b.txt",
    ]

    let baseURL = "https://huggingface.co/\(modelId)/resolve/main"
    let config = URLSessionConfiguration.default
    config.timeoutIntervalForResource = 120
    let session = URLSession(configuration: config)
    defer { session.finishTasksAndInvalidate() }

    for file in promptFiles {
        let localPath = directory.appendingPathComponent(file)
        if FileManager.default.fileExists(atPath: localPath.path) { continue }

        let url = URL(string: "\(baseURL)/\(file)")!
        let (tempURL, response) = try await session.download(from: url)
        guard let httpResponse = response as? HTTPURLResponse,
              httpResponse.statusCode == 200 else { continue }
        try FileManager.default.moveItem(at: tempURL, to: localPath)
    }
}
