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

    @Option(name: .long, help: "Custom system prompt text (overrides --system-prompt preset)")
    public var systemPromptText: String?

    @Option(name: .long, help: "Maximum generation steps at 12.5Hz (default: 200 = ~16s)")
    public var maxSteps: Int = 200

    @Option(name: .long, help: "HuggingFace model ID (e.g. aufklarer/PersonaPlex-7B-MLX-4bit or aufklarer/PersonaPlex-7B-MLX-8bit)")
    public var modelId: String = PersonaPlexModel.defaultModelId

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

    @Flag(name: .long, help: "Full-duplex mode: feed input WAV into ring buffer, record response. Saves input + output WAVs for offline analysis.")
    public var fullDuplex: Bool = false

    @Option(name: .long, help: "Full-duplex debug output directory (default: /tmp/personaplex_debug)")
    public var debugDir: String = "/tmp/personaplex_debug"

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

        var selectedPrompt: SystemPromptPreset?
        if systemPromptText == nil {
            guard let preset = SystemPromptPreset(rawValue: systemPrompt) else {
                print("Unknown system prompt: \(systemPrompt)")
                print("Use --list-prompts to see available options.")
                throw ExitCode(1)
            }
            selectedPrompt = preset
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

            // Use the model's built-in tokenizer for transcript decoding
            let spmDecoder = model.tokenizer

            // Resolve system prompt tokens
            let promptTokens: [Int32]?
            let promptLabel: String
            if let customText = systemPromptText {
                guard let tokens = model.tokenizeSystemPrompt(customText) else {
                    print("Error: SentencePiece tokenizer not available for custom system prompt.")
                    throw ExitCode(1)
                }
                promptTokens = tokens
                promptLabel = "custom (\(tokens.count) tokens)"
            } else {
                promptTokens = selectedPrompt?.tokens
                promptLabel = selectedPrompt?.rawValue ?? "default"
            }

            if !json { print("Loading input audio: \(input)") }
            let inputURL = URL(fileURLWithPath: input)
            let audio = try AudioFileLoader.load(
                url: inputURL, targetSampleRate: 24000)
            let duration = Double(audio.count) / 24000.0
            if !json {
                print("  Duration: \(String(format: "%.2f", duration))s (\(audio.count) samples)")
                print("Generating response with voice \(selectedVoice.rawValue), prompt: \(promptLabel)")
            }
            let startTime = CFAbsoluteTimeGetCurrent()

            var responseSamples: [Float] = []
            var textTokens: [Int32] = []

            if fullDuplex {
                // Full-duplex debug mode: feed input into ring buffer, record all output
                let dir = URL(fileURLWithPath: debugDir)
                try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)

                print("Full-duplex mode: feeding \(String(format: "%.1f", duration))s input...")
                let ringBuffer = AudioRingBuffer(capacity: 24000 * 10)

                // Feed input audio into ring buffer in real-time-ish chunks
                let chunkSize = 1920  // 80ms at 24kHz (one Mimi frame)
                Task.detached {
                    var offset = 0
                    while offset < audio.count {
                        let end = min(offset + chunkSize, audio.count)
                        let chunk = Array(audio[offset..<end])
                        ringBuffer.write(chunk)
                        offset = end
                        // Simulate real-time: 80ms per chunk
                        try? await Task.sleep(nanoseconds: 80_000_000)
                    }
                    print("  Input feed complete (\(audio.count) samples)")
                }

                let stream = model.respondRealtime(
                    voice: selectedVoice,
                    systemPromptTokens: promptTokens,
                    userAudioBuffer: ringBuffer,
                    maxSteps: maxSteps,
                    verbose: verbose
                )

                var frameCount = 0
                for try await agentSamples in stream {
                    responseSamples.append(contentsOf: agentSamples)
                    frameCount += 1
                    if frameCount % 25 == 0 {
                        let elapsed = CFAbsoluteTimeGetCurrent() - startTime
                        let audioSec = Double(responseSamples.count) / 24000.0
                        print("  Frame \(frameCount): \(String(format: "%.1f", audioSec))s audio, \(String(format: "%.1f", elapsed))s elapsed")
                    }
                }

                let elapsed = CFAbsoluteTimeGetCurrent() - startTime
                let outDuration = Double(responseSamples.count) / 24000.0
                print("Full-duplex complete: \(frameCount) frames, \(String(format: "%.1f", outDuration))s audio in \(String(format: "%.1f", elapsed))s")

                // Save debug files
                let inputPath = dir.appendingPathComponent("input.wav")
                let outputPath = dir.appendingPathComponent("output.wav")
                try WAVWriter.write(samples: audio, sampleRate: 24000, to: inputPath)
                try WAVWriter.write(samples: responseSamples, sampleRate: 24000, to: outputPath)
                print("Saved: \(inputPath.path)")
                print("Saved: \(outputPath.path)")
                return
            } else if stream {
                let streamingConfig = PersonaPlexModel.PersonaPlexStreamingConfig(
                    firstChunkFrames: chunkFrames, chunkFrames: chunkFrames)
                let audioStream = model.respondStream(
                    userAudio: audio,
                    voice: selectedVoice,
                    systemPromptTokens: promptTokens,
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
                    systemPromptTokens: promptTokens,
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
                    "system_prompt": promptLabel,
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
