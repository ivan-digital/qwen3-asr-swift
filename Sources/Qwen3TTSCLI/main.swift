import Foundation
import Qwen3TTS
import Qwen3Common
import ArgumentParser

struct Qwen3TTSCLI: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "qwen3-tts-cli",
        abstract: "Qwen3-TTS text-to-speech synthesis"
    )

    @Argument(help: "Text to synthesize")
    var text: String

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

    @Flag(name: .long, help: "Stream audio generation (lower latency, incremental file output)")
    var stream: Bool = false

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

                if stream {
                    try await runStreaming(model: model, config: config)
                } else {
                    try runStandard(model: model, config: config)
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

    private func runStandard(model: Qwen3TTSModel, config: SamplingConfig) throws {
        print("Synthesizing: \"\(text)\"")
        let audio = model.synthesize(
            text: text,
            language: language,
            sampling: config)

        guard !audio.isEmpty else {
            print("Error: No audio generated")
            throw ExitCode(1)
        }

        let outputURL = URL(fileURLWithPath: output)
        try WAVWriter.write(samples: audio, sampleRate: 24000, to: outputURL)
        print("Saved \(audio.count) samples (\(String(format: "%.2f", Double(audio.count) / 24000.0))s) to \(output)")
    }

    private func runStreaming(model: Qwen3TTSModel, config: SamplingConfig) async throws {
        print("Streaming synthesis: \"\(text)\"")
        let t0 = CFAbsoluteTimeGetCurrent()

        let outputURL = URL(fileURLWithPath: output)
        let writer = try StreamingWAVWriter(to: outputURL)
        var totalSamples = 0
        var chunkCount = 0
        var firstChunkTime: Double?

        let stream = model.synthesizeStream(
            text: text,
            language: language,
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
