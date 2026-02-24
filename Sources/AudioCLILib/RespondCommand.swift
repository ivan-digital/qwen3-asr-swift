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

    @Option(name: .long, help: "Maximum generation steps at 12.5Hz (default: 500 = ~40s)")
    public var maxSteps: Int = 500

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
            print("Loading PersonaPlex 7B model...")
            let model = try await PersonaPlexModel.fromPretrained(
                modelId: modelId, progressHandler: reportProgress)

            if compile {
                print("Warming up compiled temporal transformer...")
                let warmStart = CFAbsoluteTimeGetCurrent()
                model.warmUp()
                let warmTime = CFAbsoluteTimeGetCurrent() - warmStart
                print("  Warmup: \(String(format: "%.2f", warmTime))s")
            }

            print("Loading input audio: \(input)")
            let inputURL = URL(fileURLWithPath: input)
            let audio = try AudioFileLoader.load(
                url: inputURL, targetSampleRate: 24000)
            let duration = Double(audio.count) / 24000.0
            print("  Duration: \(String(format: "%.2f", duration))s (\(audio.count) samples)")

            print("Generating response with voice \(selectedVoice.rawValue), prompt: \(selectedPrompt.rawValue)")
            let startTime = CFAbsoluteTimeGetCurrent()

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

                var allSamples: [Float] = []
                var chunkCount = 0
                for try await chunk in audioStream {
                    allSamples.append(contentsOf: chunk.samples)
                    chunkCount += 1
                    if verbose {
                        let chunkDuration = Double(chunk.samples.count) / 24000.0
                        print("  Chunk \(chunkCount): \(chunk.samples.count) samples (\(String(format: "%.2f", chunkDuration))s) at \(String(format: "%.2f", chunk.elapsedTime ?? 0))s")
                    }
                }

                let elapsed = CFAbsoluteTimeGetCurrent() - startTime
                let responseDuration = Double(allSamples.count) / 24000.0

                print("Response: \(String(format: "%.2f", responseDuration))s (\(chunkCount) chunks)")
                print("Time: \(String(format: "%.2f", elapsed))s")
                if responseDuration > 0 {
                    print("RTF: \(String(format: "%.2f", elapsed / responseDuration))")
                }

                let outputURL = URL(fileURLWithPath: output)
                try WAVWriter.write(samples: allSamples, sampleRate: 24000, to: outputURL)
                print("Saved to \(output)")
            } else {
                let response = model.respond(
                    userAudio: audio,
                    voice: selectedVoice,
                    systemPromptTokens: selectedPrompt.tokens,
                    maxSteps: maxSteps,
                    verbose: verbose)

                let elapsed = CFAbsoluteTimeGetCurrent() - startTime
                let responseDuration = Double(response.count) / 24000.0

                print("Response: \(String(format: "%.2f", responseDuration))s")
                print("Time: \(String(format: "%.2f", elapsed))s")
                if responseDuration > 0 {
                    print("RTF: \(String(format: "%.2f", elapsed / responseDuration))")
                }

                let outputURL = URL(fileURLWithPath: output)
                try WAVWriter.write(samples: response, sampleRate: 24000, to: outputURL)
                print("Saved to \(output)")
            }
        }
    }
}
