import Foundation
import ArgumentParser
import PersonaPlex
import Qwen3Common
import MLX

struct PersonaPlexCLI: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "personaplex-cli",
        abstract: "PersonaPlex 7B full-duplex speech-to-speech inference"
    )

    @Option(name: .long, help: "Input audio WAV file (24kHz mono)")
    var input: String?

    @Option(name: .long, help: "Output response WAV file")
    var output: String = "response.wav"

    @Option(name: .long, help: "Voice preset (e.g. NATM0, NATF1, VARF0)")
    var voice: String = "NATM0"

    @Option(name: .long, help: "System prompt preset: assistant (default), focused, customer-service, teacher")
    var systemPrompt: String = "assistant"

    @Option(name: .long, help: "Maximum generation steps at 12.5Hz (default: 500 = ~40s)")
    var maxSteps: Int = 500

    @Option(name: .long, help: "Model ID on HuggingFace")
    var modelId: String = "aufklarer/PersonaPlex-7B-MLX-4bit"

    @Flag(name: .long, help: "Enable streaming output (emit audio chunks during generation)")
    var stream: Bool = false

    @Option(name: .long, help: "Frames per streaming chunk (default: 25 = ~2s)")
    var chunkFrames: Int = 25

    @Flag(name: .long, help: "Enable compiled temporal transformer (warmup + kernel fusion)")
    var compile: Bool = false

    @Flag(name: .long, help: "List available voices and exit")
    var listVoices: Bool = false

    @Flag(name: .long, help: "List available system prompt presets and exit")
    var listPrompts: Bool = false

    @Flag(name: .long, help: "Show detailed timing info")
    var verbose: Bool = false

    func run() throws {
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

        let semaphore = DispatchSemaphore(value: 0)
        var exitCode: Int32 = 0

        Task {
            do {
                // Load model
                print("Loading PersonaPlex 7B model...")
                let model = try await PersonaPlexModel.fromPretrained(
                    modelId: modelId
                ) { progress, status in
                    print("  [\(Int(progress * 100))%] \(status)")
                }

                // Warmup + compilation if requested
                if compile {
                    print("Warming up compiled temporal transformer...")
                    let warmStart = CFAbsoluteTimeGetCurrent()
                    model.warmUp()
                    let warmTime = CFAbsoluteTimeGetCurrent() - warmStart
                    print("  Warmup: \(String(format: "%.2f", warmTime))s")
                }

                // Load input audio
                print("Loading input audio: \(input)")
                let inputURL = URL(fileURLWithPath: input)
                let audio = try AudioFileLoader.load(
                    url: inputURL, targetSampleRate: 24000)
                let duration = Double(audio.count) / 24000.0
                print("  Duration: \(String(format: "%.2f", duration))s (\(audio.count) samples)")

                // Generate response
                print("Generating response with voice \(selectedVoice.rawValue), prompt: \(selectedPrompt.rawValue)")
                let startTime = CFAbsoluteTimeGetCurrent()

                if stream {
                    // Streaming mode: emit chunks during generation, concatenate for output
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
                    // Offline mode
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

            } catch {
                print("Error: \(error)")
                exitCode = 1
            }
            semaphore.signal()
        }

        semaphore.wait()
        if exitCode != 0 { throw ExitCode(exitCode) }
    }
}

PersonaPlexCLI.main()
