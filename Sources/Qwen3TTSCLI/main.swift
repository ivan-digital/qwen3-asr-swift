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

    @Option(name: .long, help: "Instruction text for style control (requires CustomVoice model, e.g., \"Speak cheerfully\")")
    var instruct: String?

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

    @Flag(name: .long, help: "Enable streaming synthesis (emit audio chunks incrementally)")
    var stream: Bool = false

    @Option(name: .long, help: "Codec frames in first streamed chunk (default 3 = 240ms, use 1 for ~70ms latency)")
    var firstChunkFrames: Int = 3

    @Option(name: .long, help: "Codec frames per subsequent streamed chunk (default 25 = 2s)")
    var chunkFrames: Int = 25

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

                // Resolve effective instruct for display purposes
                let effectiveInstruct: String?
                let instructIsDefault: Bool
                if let explicit = instruct {
                    effectiveInstruct = explicit
                    instructIsDefault = false
                } else if ttsModel.speakerConfig != nil {
                    effectiveInstruct = Qwen3TTSModel.defaultInstruct
                    instructIsDefault = true
                } else {
                    effectiveInstruct = nil
                    instructIsDefault = false
                }

                if stream, let inputText = text {
                    try await runStreaming(model: ttsModel, text: inputText, instruct: effectiveInstruct, instructIsDefault: instructIsDefault, config: config)
                } else if let batchFile = batchFile {
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
                        instruct: instruct,
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
                    try runStandard(model: ttsModel, text: inputText, instruct: effectiveInstruct, instructIsDefault: instructIsDefault, config: config)
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

    private func runStreaming(model: Qwen3TTSModel, text: String, instruct: String?, instructIsDefault: Bool = false, config: SamplingConfig) async throws {
        let streamingConfig = StreamingConfig(
            firstChunkFrames: firstChunkFrames,
            chunkFrames: chunkFrames)

        var info = "Streaming synthesis: \"\(text)\""
        if let spk = speaker { info += " [speaker: \(spk)]" }
        if let inst = instruct { info += " [instruct: \(inst)\(instructIsDefault ? " (default)" : "")]" }
        print(info)
        print("  First chunk: \(firstChunkFrames) frames, subsequent: \(chunkFrames) frames")

        var allSamples: [Float] = []
        var chunkCount = 0
        var firstPacketLatency: Double?

        let stream = model.synthesizeStream(
            text: text,
            language: language,
            speaker: speaker,
            instruct: instruct,
            sampling: config,
            streaming: streamingConfig)

        for try await chunk in stream {
            chunkCount += 1
            allSamples.append(contentsOf: chunk.samples)

            if firstPacketLatency == nil {
                firstPacketLatency = chunk.elapsedTime
            }

            let chunkDuration = Double(chunk.samples.count) / 24000.0
            let marker = chunk.isFinal ? " [FINAL]" : ""
            print("  Chunk \(chunkCount): \(chunk.samples.count) samples " +
                  "(\(String(format: "%.3f", chunkDuration))s) | " +
                  "frames \(chunk.frameIndex)..<\(chunk.totalFrames) | " +
                  "elapsed \(String(format: "%.3f", chunk.elapsedTime))s\(marker)")
        }

        guard !allSamples.isEmpty else {
            print("Error: No audio generated")
            throw ExitCode(1)
        }

        let totalDuration = Double(allSamples.count) / 24000.0
        print("  First-packet latency: \(String(format: "%.0f", (firstPacketLatency ?? 0) * 1000))ms")
        print("  Total: \(chunkCount) chunks, \(allSamples.count) samples (\(String(format: "%.2f", totalDuration))s)")

        let outputURL = URL(fileURLWithPath: output)
        try WAVWriter.write(samples: allSamples, sampleRate: 24000, to: outputURL)
        print("Saved to \(output)")
    }

    private func runStandard(model: Qwen3TTSModel, text: String, instruct: String?, instructIsDefault: Bool = false, config: SamplingConfig) throws {
        var info = "Synthesizing: \"\(text)\""
        if let spk = speaker { info += " [speaker: \(spk)]" }
        if let inst = instruct { info += " [instruct: \(inst)\(instructIsDefault ? " (default)" : "")]" }
        print(info)
        let audio = model.synthesize(
            text: text,
            language: language,
            speaker: speaker,
            instruct: instruct,
            sampling: config)

        guard !audio.isEmpty else {
            print("Error: No audio generated")
            throw ExitCode(1)
        }

        let outputURL = URL(fileURLWithPath: output)
        try WAVWriter.write(samples: audio, sampleRate: 24000, to: outputURL)
        print("Saved \(audio.count) samples (\(String(format: "%.2f", Double(audio.count) / 24000.0))s) to \(output)")
    }
}

Qwen3TTSCLI.main()
