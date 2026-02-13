import Foundation
import Qwen3TTS
import Qwen3Common
import ArgumentParser

struct Qwen3TTSCLI: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "qwen3-tts-cli",
        abstract: "Qwen3-TTS text-to-speech synthesis"
    )

    @Argument(help: "Text to synthesize (ignored if --batch-file is provided)")
    var text: String?

    @Option(name: .long, help: "Output WAV file path")
    var output: String = "output.wav"

    @Option(name: .long, help: "Language (english, chinese, german, japanese)")
    var language: String = "english"

    @Option(name: .long, help: "Sampling temperature")
    var temperature: Float = 0.9

    @Option(name: .long, help: "Top-k sampling")
    var topK: Int = 50

    @Option(name: .long, help: "Maximum tokens to generate (500 = ~40s audio)")
    var maxTokens: Int = 500

    @Option(name: .long, help: "File with one text per line for batch synthesis")
    var batchFile: String?

    @Option(name: .long, help: "Maximum batch size for parallel generation")
    var batchSize: Int = 4

    func validate() throws {
        if text == nil && batchFile == nil {
            throw ValidationError("Either a text argument or --batch-file must be provided")
        }
    }

    func run() throws {
        let semaphore = DispatchSemaphore(value: 0)
        var exitCode: Int32 = 0

        Task {
            do {
                print("Loading model...")
                let model = try await Qwen3TTSModel.fromPretrained { progress, status in
                    print("  [\(Int(progress * 100))%] \(status)")
                }

                let config = SamplingConfig(
                    temperature: temperature,
                    topK: topK,
                    maxTokens: maxTokens)

                if let batchFile = batchFile {
                    // Batch mode: read texts from file
                    let content = try String(contentsOfFile: batchFile, encoding: .utf8)
                    let texts = content.components(separatedBy: .newlines)
                        .map { $0.trimmingCharacters(in: .whitespaces) }
                        .filter { !$0.isEmpty }

                    guard !texts.isEmpty else {
                        print("Error: No texts found in \(batchFile)")
                        exitCode = 1
                        semaphore.signal()
                        return
                    }

                    print("Batch synthesizing \(texts.count) texts...")
                    let audioList = model.synthesizeBatch(
                        texts: texts,
                        language: language,
                        sampling: config,
                        maxBatchSize: batchSize)

                    // Write each output as output_0.wav, output_1.wav, etc.
                    let basePath = (output as NSString).deletingPathExtension
                    let ext = (output as NSString).pathExtension.isEmpty ? "wav" : (output as NSString).pathExtension

                    for (i, audio) in audioList.enumerated() {
                        guard !audio.isEmpty else {
                            print("Warning: Item \(i) produced no audio")
                            continue
                        }
                        let path = "\(basePath)_\(i).\(ext)"
                        let url = URL(fileURLWithPath: path)
                        try WAVWriter.write(samples: audio, sampleRate: 24000, to: url)
                        print("Saved item \(i): \(audio.count) samples (\(String(format: "%.2f", Double(audio.count) / 24000.0))s) to \(path)")
                    }
                } else if let text = text {
                    // Single text mode
                    print("Synthesizing: \"\(text)\"")
                    let audio = model.synthesize(
                        text: text,
                        language: language,
                        sampling: config)

                    guard !audio.isEmpty else {
                        print("Error: No audio generated")
                        exitCode = 1
                        semaphore.signal()
                        return
                    }

                    let outputURL = URL(fileURLWithPath: output)
                    try WAVWriter.write(samples: audio, sampleRate: 24000, to: outputURL)
                    print("Saved \(audio.count) samples (\(String(format: "%.2f", Double(audio.count) / 24000.0))s) to \(output)")
                }

                exitCode = 0
            } catch {
                print("Error: \(error)")
                exitCode = 1
            }
            semaphore.signal()
        }

        semaphore.wait()
        if exitCode != 0 {
            throw ExitCode(exitCode)
        }
    }
}

Qwen3TTSCLI.main()
