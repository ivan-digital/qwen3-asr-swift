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
    var input: String

    @Option(name: .long, help: "Output response WAV file")
    var output: String = "response.wav"

    @Option(name: .long, help: "Voice preset (e.g. NATM0, NATF1, VARF0)")
    var voice: String = "NATM0"

    @Option(name: .long, help: "Maximum generation steps at 12.5Hz (default: 500 = ~40s)")
    var maxSteps: Int = 500

    @Option(name: .long, help: "Model ID on HuggingFace")
    var modelId: String = "aufklarer/PersonaPlex-7B-MLX-4bit"

    @Flag(name: .long, help: "List available voices and exit")
    var listVoices: Bool = false

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

        guard let selectedVoice = PersonaPlexVoice(rawValue: voice) else {
            print("Unknown voice: \(voice)")
            print("Use --list-voices to see available options.")
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

                // Load input audio
                print("Loading input audio: \(input)")
                let inputURL = URL(fileURLWithPath: input)
                let audio = try AudioFileLoader.load(
                    url: inputURL, targetSampleRate: 24000)
                let duration = Double(audio.count) / 24000.0
                print("  Duration: \(String(format: "%.2f", duration))s (\(audio.count) samples)")

                // Generate response
                print("Generating response with voice \(selectedVoice.rawValue)...")
                let startTime = CFAbsoluteTimeGetCurrent()

                let response = model.respond(
                    userAudio: audio,
                    voice: selectedVoice,
                    maxSteps: maxSteps,
                    verbose: verbose
                )

                let elapsed = CFAbsoluteTimeGetCurrent() - startTime
                let responseDuration = Double(response.count) / 24000.0

                print("Response: \(String(format: "%.2f", responseDuration))s")
                print("Time: \(String(format: "%.2f", elapsed))s")
                if responseDuration > 0 {
                    print("RTF: \(String(format: "%.2f", elapsed / responseDuration))")
                }

                // Save output
                let outputURL = URL(fileURLWithPath: output)
                try WAVWriter.write(samples: response, sampleRate: 24000, to: outputURL)
                print("Saved to \(output)")

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
