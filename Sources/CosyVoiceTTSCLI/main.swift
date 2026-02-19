import Foundation
import ArgumentParser
import CosyVoiceTTS
import Qwen3Common

struct CosyVoiceTTSCLI: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "cosyvoice-tts-cli",
        abstract: "CosyVoice3 text-to-speech synthesis (9 languages, streaming)"
    )

    @Argument(help: "Text to synthesize")
    var text: String

    @Option(name: .long, help: "Output WAV file path")
    var output: String = "output.wav"

    @Option(name: .long, help: "Language (english, chinese, german, japanese, korean, spanish, french, italian, russian)")
    var language: String = "english"

    @Option(name: .long, help: "Model ID on HuggingFace")
    var modelId: String = "aufklarer/CosyVoice3-0.5B-MLX-4bit"

    @Option(name: .long, help: "Streaming chunk interval in seconds (0 = non-streaming)")
    var streamingInterval: Double = 0

    func run() throws {
        let semaphore = DispatchSemaphore(value: 0)
        var exitCode: Int32 = 0

        Task {
            do {
                print("Loading CosyVoice3 model...")
                let model = try await CosyVoiceTTSModel.fromPretrained(
                    modelId: modelId
                ) { progress, status in
                    print("  [\(Int(progress * 100))%] \(status)")
                }

                print("Synthesizing: \"\(text)\"")
                print("  Language: \(language)")

                let startTime = CFAbsoluteTimeGetCurrent()

                if streamingInterval > 0 {
                    // Streaming mode
                    var allSamples: [Float] = []
                    var chunkCount = 0
                    for try await chunk in model.synthesizeStream(text: text, language: language) {
                        allSamples.append(contentsOf: chunk.samples)
                        chunkCount += 1
                        let chunkDuration = Double(chunk.samples.count) / Double(chunk.sampleRate)
                        print("  Chunk \(chunkCount): \(String(format: "%.2f", chunkDuration))s (\(chunk.samples.count) samples)")
                    }

                    let elapsed = CFAbsoluteTimeGetCurrent() - startTime
                    let duration = Double(allSamples.count) / 24000.0
                    print(String(format: "  Duration: %.2fs, Time: %.2fs, RTF: %.2f", duration, elapsed, elapsed / max(duration, 0.001)))

                    let outputURL = URL(fileURLWithPath: output)
                    try WAVWriter.write(samples: allSamples, sampleRate: 24000, to: outputURL)
                    print("Saved to \(output)")
                } else {
                    // Non-streaming mode
                    let samples = model.synthesize(text: text, language: language)

                    let elapsed = CFAbsoluteTimeGetCurrent() - startTime
                    let duration = Double(samples.count) / 24000.0
                    print(String(format: "  Duration: %.2fs, Time: %.2fs, RTF: %.2f", duration, elapsed, elapsed / max(duration, 0.001)))

                    let outputURL = URL(fileURLWithPath: output)
                    try WAVWriter.write(samples: samples, sampleRate: 24000, to: outputURL)
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

CosyVoiceTTSCLI.main()
