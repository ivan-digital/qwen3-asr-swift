import Foundation
import ArgumentParser
import PersonaPlex
import AudioCommon
import MLX

public struct RespondCommand: ParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "respond",
        abstract: "Full-duplex speech-to-speech inference using PersonaPlex 7B"
    )

    @Option(name: .shortAndLong, help: "Input audio WAV file (24kHz mono)")
    public var input: String?

    @Option(name: .shortAndLong, help: "Output response WAV file")
    public var output: String = "response.wav"

    @Option(name: .long, help: "Voice preset (e.g. NATM0, NATF1, VARF0)")
    public var voice: String = "NATM0"

    @Option(name: .long, help: "System prompt preset: assistant, focused, customer-service, teacher")
    public var systemPrompt: String = "assistant"

    @Option(name: .long, help: "Maximum generation steps at 12.5Hz (default: 200 = ~16s)")
    public var maxSteps: Int = 200

    @Option(name: .long, help: "HuggingFace model ID")
    public var modelId: String = "aufklarer/PersonaPlex-7B-MLX-4bit"

    @Flag(name: .long, help: "Enable streaming output (emit audio chunks during generation)")
    public var stream: Bool = false

    @Option(name: .long, help: "Frames per streaming chunk (default: 25 = ~2s)")
    public var chunkFrames: Int = 25

    @Flag(name: .long, help: "Enable compiled temporal transformer (warmup + kernel fusion)")
    public var compile: Bool = false

    @Flag(name: .long, help: "List available voices and exit")
    public var listVoices: Bool = false

    @Flag(name: .long, help: "List available system prompt presets and exit")
    public var listPrompts: Bool = false

    @Flag(name: .long, help: "Show detailed timing info")
    public var verbose: Bool = false

    @Flag(name: .long, help: "Decode and print the model's inner monologue text")
    public var transcript: Bool = false

    @Flag(name: .long, help: "Output results as JSON (includes transcript, latency, audio path)")
    public var json: Bool = false

    // Sampling config overrides
    @Option(name: .long, help: "Audio sampling temperature (default: 0.8)")
    public var audioTemp: Float?

    @Option(name: .long, help: "Text sampling temperature (default: 0.7)")
    public var textTemp: Float?

    @Option(name: .long, help: "Audio top-k candidates (default: 250)")
    public var audioTopK: Int?

    @Option(name: .long, help: "Audio repetition penalty (default: 1.2, 1.0 = disabled)")
    public var repetitionPenalty: Float?

    @Option(name: .long, help: "Text repetition penalty (default: 1.2, 1.0 = disabled)")
    public var textRepetitionPenalty: Float?

    @Option(name: .long, help: "Repetition penalty window in frames (default: 30)")
    public var repetitionWindow: Int?

    @Option(name: .long, help: "Silence frames before early stop (default: 15, 0 = disabled)")
    public var silenceEarlyStop: Int?

    @Option(name: .long, help: "Text entropy threshold for early stop (default: 0 = disabled, try 1.0)")
    public var entropyThreshold: Float?

    @Option(name: .long, help: "Consecutive low-entropy steps before early stop (default: 10)")
    public var entropyWindow: Int?

    public init() {}

    public func run() throws {
        if listVoices {
            print("Available voices:")
            for v in PersonaPlexVoice.allCases {
                print("  \(v.rawValue) - \(v.displayName)")
            }
            return
        }

        if listPrompts {
            print("Available system prompts:")
            for p in SystemPromptPreset.allCases {
                print("  \(p.rawValue) - \(p.description)")
            }
            return
        }

        guard let input = input else {
            print("Error: --input is required for inference.")
            throw ExitCode(1)
        }

        guard let selectedVoice = PersonaPlexVoice(rawValue: voice) else {
            print("Unknown voice: \(voice)")
            print("Use --list-voices to see available options.")
            throw ExitCode(1)
        }

        guard let selectedPrompt = SystemPromptPreset(rawValue: systemPrompt) else {
            print("Unknown system prompt: \(systemPrompt)")
            print("Use --list-prompts to see available options.")
            throw ExitCode(1)
        }

        try runAsync {
            if verbose {
                print("Build: \(buildVersion)")
            }
            print("Loading PersonaPlex 7B model...")
            let model = try await PersonaPlexModel.fromPretrained(
                modelId: modelId, progressHandler: reportProgress)

            // Apply sampling config overrides
            if let v = audioTemp { model.cfg.sampling.audioTemp = v }
            if let v = textTemp { model.cfg.sampling.textTemp = v }
            if let v = audioTopK { model.cfg.sampling.audioTopK = v }
            if let v = repetitionPenalty { model.cfg.sampling.audioRepetitionPenalty = v }
            if let v = textRepetitionPenalty { model.cfg.sampling.textRepetitionPenalty = v }
            if let v = repetitionWindow { model.cfg.sampling.repetitionWindow = v }
            if let v = silenceEarlyStop { model.cfg.sampling.silenceEarlyStopFrames = v }
            if let v = entropyThreshold { model.cfg.sampling.entropyEarlyStopThreshold = v }
            if let v = entropyWindow { model.cfg.sampling.entropyWindow = v }

            if compile {
                print("Warming up compiled temporal transformer...")
                let warmStart = CFAbsoluteTimeGetCurrent()
                model.warmUp()
                let warmTime = CFAbsoluteTimeGetCurrent() - warmStart
                print("  Warmup: \(String(format: "%.2f", warmTime))s")
            }

            // Load SentencePiece decoder for transcript
            var spmDecoder: SentencePieceDecoder?
            if transcript || json {
                do {
                    let cacheDir = try HuggingFaceDownloader.getCacheDirectory(for: modelId)
                    let spmPath = cacheDir.appendingPathComponent("tokenizer_spm_32k_3.model").path
                    if FileManager.default.fileExists(atPath: spmPath) {
                        spmDecoder = try SentencePieceDecoder(modelPath: spmPath)
                    }
                } catch {
                    if !json { print("Warning: Could not load SentencePiece decoder: \(error)") }
                }
            }

            if !json { print("Loading input audio: \(input)") }
            let inputURL = URL(fileURLWithPath: input)
            let audio = try AudioFileLoader.load(
                url: inputURL, targetSampleRate: 24000)
            let duration = Double(audio.count) / 24000.0
            if !json {
                print("  Duration: \(String(format: "%.2f", duration))s (\(audio.count) samples)")
                print("Generating response with voice \(selectedVoice.rawValue), prompt: \(selectedPrompt.rawValue)")
            }
            let startTime = CFAbsoluteTimeGetCurrent()

            var responseSamples: [Float] = []
            var textTokens: [Int32] = []

            if stream {
                let streamingConfig = PersonaPlexModel.PersonaPlexStreamingConfig(
                    firstChunkFrames: chunkFrames, chunkFrames: chunkFrames)
                let audioStream = model.respondStream(
                    userAudio: audio,
                    voice: selectedVoice,
                    systemPromptTokens: selectedPrompt.tokens,
                    maxSteps: maxSteps,
                    streaming: streamingConfig,
                    verbose: verbose)

                var chunkCount = 0
                for try await chunk in audioStream {
                    responseSamples.append(contentsOf: chunk.samples)
                    chunkCount += 1
                    if chunk.isFinal { textTokens = chunk.textTokens }
                    if verbose {
                        let chunkDuration = Double(chunk.samples.count) / 24000.0
                        print("  Chunk \(chunkCount): \(chunk.samples.count) samples (\(String(format: "%.2f", chunkDuration))s) at \(String(format: "%.2f", chunk.elapsedTime ?? 0))s")
                    }
                }
            } else {
                let result = model.respond(
                    userAudio: audio,
                    voice: selectedVoice,
                    systemPromptTokens: selectedPrompt.tokens,
                    maxSteps: maxSteps,
                    verbose: verbose)
                responseSamples = result.audio
                textTokens = result.textTokens
            }

            let elapsed = CFAbsoluteTimeGetCurrent() - startTime
            let responseDuration = Double(responseSamples.count) / 24000.0
            let rtf = responseDuration > 0 ? elapsed / responseDuration : 0

            let outputURL = URL(fileURLWithPath: output)
            try WAVWriter.write(samples: responseSamples, sampleRate: 24000, to: outputURL)

            // Decode transcript
            var decodedTranscript: String?
            if let dec = spmDecoder, !textTokens.isEmpty {
                decodedTranscript = dec.decode(textTokens)
            }

            if json {
                var result: [String: Any] = [
                    "audio": output,
                    "voice": selectedVoice.rawValue,
                    "system_prompt": selectedPrompt.rawValue,
                    "input_duration": round(duration * 100) / 100,
                    "output_duration": round(responseDuration * 100) / 100,
                    "elapsed": round(elapsed * 100) / 100,
                    "rtf": round(rtf * 100) / 100,
                    "text_tokens": textTokens.count
                ]
                if let t = decodedTranscript { result["transcript"] = t }
                let jsonData = try JSONSerialization.data(
                    withJSONObject: result, options: [.prettyPrinted, .sortedKeys])
                print(String(data: jsonData, encoding: .utf8) ?? "{}")
            } else {
                print("Response: \(String(format: "%.2f", responseDuration))s (\(textTokens.count) text tokens)")
                print("Time: \(String(format: "%.2f", elapsed))s")
                if responseDuration > 0 {
                    print("RTF: \(String(format: "%.2f", rtf))")
                }
                if let t = decodedTranscript, transcript {
                    print("Transcript: \(t)")
                }
                print("Saved to \(output)")
            }
        }
    }
}
