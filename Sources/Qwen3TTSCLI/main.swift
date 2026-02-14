import Foundation
import Qwen3TTS
import Qwen3Common
import ArgumentParser

struct Qwen3TTSCLI: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "qwen3-tts-cli",
        abstract: "Qwen3-TTS text-to-speech synthesis"
    )

    @Argument(help: "Text to synthesize (omit when using --list-speakers or --batch-file)")
    var text: String?

    @Option(name: .long, help: "Output WAV file path")
    var output: String = "output.wav"

    @Option(name: .long, help: "Language (english, chinese, german, japanese, spanish, french, korean, russian, italian, portuguese)")
    var language: String = "english"

    @Option(name: .long, help: "Speaker voice (requires CustomVoice model, e.g., vivian, ryan, serena)")
    var speaker: String?

    @Option(name: .long, help: "Model variant: base (default) or customVoice, or a full HuggingFace model ID")
    var model: String = "base"

    @Flag(name: .long, help: "List available speakers for the loaded model")
    var listSpeakers: Bool = false

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

    @Flag(name: .long, help: "Stream audio generation (lower latency, incremental file output)")
    var stream: Bool = false

    func validate() throws {
        if text == nil && batchFile == nil && !listSpeakers {
            throw ValidationError("Either a text argument, --batch-file, or --list-speakers must be provided")
        }
    }

    func run() throws {
        let semaphore = DispatchSemaphore(value: 0)
        var exitCode: Int32 = 0

        Task {
            do {
                // Resolve model ID from variant name or full ID
                let modelId: String
                switch model.lowercased() {
                case "base":
                    modelId = TTSModelVariant.base.rawValue
                case "customvoice", "custom_voice", "custom-voice":
                    modelId = TTSModelVariant.customVoice.rawValue
                default:
                    modelId = model  // Treat as full HuggingFace model ID
                }

                print("Loading model (\(modelId))...")
                let ttsModel = try await Qwen3TTSModel.fromPretrained(
                    modelId: modelId
                ) { progress, status in
                    print("  [\(Int(progress * 100))%] \(status)")
                }

                // Handle --list-speakers
                if listSpeakers {
                    let speakers = ttsModel.availableSpeakers
                    if speakers.isEmpty {
                        print("No speakers available for this model.")
                        print("Use --model customVoice to load a model with speaker support.")
                    } else {
                        print("Available speakers:")
                        for name in speakers {
                            let dialect = ttsModel.speakerConfig?.speakerDialects[name]
                            let suffix = dialect != nil ? " (\(dialect!))" : ""
                            print("  - \(name)\(suffix)")
                        }
                    }
                    semaphore.signal()
                    return
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
                    let audioList = ttsModel.synthesizeBatch(
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
                } else if let inputText = text {
                    if stream {
                        try await runStreaming(model: ttsModel, text: inputText, config: config)
                    } else {
                        try runStandard(model: ttsModel, text: inputText, config: config)
                    }
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

    private func runStandard(model: Qwen3TTSModel, text: String, config: SamplingConfig) throws {
        if let spk = speaker {
            print("Synthesizing: \"\(text)\" [speaker: \(spk)]")
        } else {
            print("Synthesizing: \"\(text)\"")
        }
        let audio = model.synthesize(
            text: text,
            language: language,
            speaker: speaker,
            sampling: config)

        guard !audio.isEmpty else {
            print("Error: No audio generated")
            throw ExitCode(1)
        }

        let outputURL = URL(fileURLWithPath: output)
        try WAVWriter.write(samples: audio, sampleRate: 24000, to: outputURL)
        print("Saved \(audio.count) samples (\(String(format: "%.2f", Double(audio.count) / 24000.0))s) to \(output)")
    }

    private func runStreaming(model: Qwen3TTSModel, text: String, config: SamplingConfig) async throws {
        if let spk = speaker {
            print("Streaming synthesis: \"\(text)\" [speaker: \(spk)]")
        } else {
            print("Streaming synthesis: \"\(text)\"")
        }
        let t0 = CFAbsoluteTimeGetCurrent()

        let outputURL = URL(fileURLWithPath: output)
        let writer = try StreamingWAVWriter(to: outputURL)
        var totalSamples = 0
        var chunkCount = 0
        var firstChunkTime: Double?

        let stream = model.synthesizeStream(
            text: text,
            language: language,
            speaker: speaker,
            sampling: config)

        for try await chunk in stream {
            if chunk.isFinal && chunk.samples.isEmpty { break }

            if !chunk.samples.isEmpty {
                writer.write(samples: chunk.samples)
                totalSamples += chunk.samples.count
                chunkCount += 1

                if firstChunkTime == nil {
                    firstChunkTime = CFAbsoluteTimeGetCurrent() - t0
                    print("  First audio chunk in \(String(format: "%.2f", firstChunkTime!))s")
                }

                let dur = Double(chunk.samples.count) / 24000.0
                print("  Chunk \(chunkCount): \(chunk.samples.count) samples (\(String(format: "%.2f", dur))s)")
            }
        }

        let result = writer.finalize()
        let t1 = CFAbsoluteTimeGetCurrent()
        let audioDur = Double(result.sampleCount) / 24000.0

        print("Saved \(result.sampleCount) samples (\(String(format: "%.2f", audioDur))s) to \(output)")
        print("  Total time: \(String(format: "%.2f", t1 - t0))s, " +
              "RTF: \(String(format: "%.2f", (t1 - t0) / max(audioDur, 0.001))), " +
              "\(chunkCount) chunks")
    }
}

Qwen3TTSCLI.main()
