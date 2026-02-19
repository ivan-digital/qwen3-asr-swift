import Foundation
import MLX
import MarvisTTS
import Qwen3Common
import ArgumentParser

struct MarvisTTSCLI: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "marvis-tts-cli",
        abstract: "Marvis TTS text-to-speech synthesis with streaming support"
    )

    @Argument(help: "Text to synthesize")
    var text: String

    @Option(name: .long, help: "Output WAV file path")
    var output: String = "output.wav"

    @Option(name: .long, help: "Voice: conversational_a or conversational_b")
    var voice: String = "conversational_a"

    @Option(name: .long, help: "Quality: low (8), medium (16), high (24), maximum (32)")
    var quality: String = "maximum"

    @Option(name: .long, help: "Streaming interval in seconds (0 = no streaming output)")
    var streamingInterval: Double = 0.5

    @Option(name: .long, help: "Reference audio WAV for voice cloning")
    var refAudio: String?

    @Option(name: .long, help: "Reference text matching ref-audio")
    var refText: String?

    @Option(name: .long, help: "HuggingFace model ID")
    var model: String = "Marvis-AI/marvis-tts-250m-v0.2-MLX-8bit"

    func run() throws {
        let semaphore = DispatchSemaphore(value: 0)
        var exitCode: Int32 = 0

        Task {
            do {
                print("Loading model (\(model))...")
                let ttsModel = try await MarvisTTSModel.fromPretrained(
                    modelId: model
                ) { progress, status in
                    print("  [\(Int(progress * 100))%] \(status)")
                }

                let resolvedVoice: MarvisTTSModel.Voice? = {
                    switch voice.lowercased() {
                    case "conversational_a", "a": return .conversationalA
                    case "conversational_b", "b": return .conversationalB
                    default: return .conversationalA
                    }
                }()

                let resolvedQuality: MarvisTTSModel.QualityLevel = {
                    switch quality.lowercased() {
                    case "low", "8": return .low
                    case "medium", "16": return .medium
                    case "high", "24": return .high
                    case "maximum", "32", "max": return .maximum
                    default: return .maximum
                    }
                }()

                // Handle reference audio for voice cloning
                var refAudioArray: MLXArray? = nil
                if let refAudioPath = refAudio {
                    let url = URL(fileURLWithPath: refAudioPath)
                    let samples = try AudioFileLoader.load(url: url, targetSampleRate: ttsModel.sampleRate)
                    refAudioArray = MLXArray(samples)
                    guard refText != nil else {
                        print("Error: --ref-text is required when using --ref-audio")
                        exitCode = 1
                        semaphore.signal()
                        return
                    }
                }

                print("Synthesizing: \"\(text)\" [voice: \(voice), quality: \(quality)]")
                let startTime = CFAbsoluteTimeGetCurrent()

                var allAudio: [Float] = []
                var chunkCount = 0

                for try await chunk in ttsModel.synthesizeStream(
                    text: text,
                    voice: refAudioArray != nil ? nil : resolvedVoice,
                    quality: resolvedQuality,
                    refAudio: refAudioArray,
                    refText: refText,
                    streamingInterval: streamingInterval
                ) {
                    chunkCount += 1
                    allAudio.append(contentsOf: chunk.audio)
                    let elapsed = CFAbsoluteTimeGetCurrent() - startTime
                    print("  Chunk \(chunkCount): \(chunk.sampleCount) samples "
                          + "(\(String(format: "%.2f", chunk.audioDuration))s audio, "
                          + "RTF: \(String(format: "%.2f", chunk.realTimeFactor)), "
                          + "elapsed: \(String(format: "%.2f", elapsed))s)")
                }

                guard !allAudio.isEmpty else {
                    print("Error: No audio generated")
                    exitCode = 1
                    semaphore.signal()
                    return
                }

                let totalDuration = Double(allAudio.count) / Double(ttsModel.sampleRate)
                let totalElapsed = CFAbsoluteTimeGetCurrent() - startTime
                let totalRTF = totalElapsed / totalDuration

                let outputURL = URL(fileURLWithPath: output)
                try WAVWriter.write(samples: allAudio, sampleRate: ttsModel.sampleRate, to: outputURL)
                print("Saved \(allAudio.count) samples (\(String(format: "%.2f", totalDuration))s) to \(output)")
                print("Total RTF: \(String(format: "%.2f", totalRTF))")
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

MarvisTTSCLI.main()
