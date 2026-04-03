import Foundation
import ArgumentParser
import Qwen3TTSCoreML
import AudioCommon

public struct Qwen3TTSCoreMLCommand: ParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "qwen3-tts-coreml",
        abstract: "Text-to-speech using Qwen3-TTS (CoreML, Neural Engine)"
    )

    @Argument(help: "Text to synthesize")
    public var text: String

    @Option(name: .shortAndLong, help: "Output WAV file path")
    public var output: String = "output.wav"

    @Option(name: .long, help: "Language: english, chinese, japanese, korean, german, french, russian, spanish")
    public var language: String = "english"

    @Option(name: .long, help: "Maximum codec tokens (1 token ≈ 80ms)")
    public var maxTokens: Int = 125

    @Option(name: .long, help: "Sampling temperature")
    public var temperature: Double = 0.8

    @Option(name: .long, help: "Top-K sampling")
    public var topK: Int = 50

    @Option(name: .long, help: "HuggingFace model ID")
    public var model: String = Qwen3TTSCoreMLModel.defaultModelId

    public init() {}

    public func run() throws {
        try runAsync {
            print("Loading Qwen3-TTS CoreML models...")
            let ttsModel = try await Qwen3TTSCoreMLModel.fromPretrained(
                modelId: model,
                progressHandler: { progress, status in
                    let pct = Int(progress * 100)
                    print("\r  \(status) \(pct)%", terminator: "")
                    fflush(stdout)
                }
            )
            print()

            print("Synthesizing: \"\(text)\"")
            print("  Language: \(language), maxTokens: \(maxTokens)")

            let startTime = CFAbsoluteTimeGetCurrent()
            let audio = try ttsModel.synthesize(
                text: text,
                language: language,
                temperature: Float(temperature),
                topK: topK,
                maxTokens: maxTokens
            )
            let elapsed = CFAbsoluteTimeGetCurrent() - startTime

            guard !audio.isEmpty else {
                print("Error: No audio generated")
                throw ExitCode(1)
            }

            let duration = Double(audio.count) / 24000.0
            print(String(format: "  Duration: %.2fs, Time: %.3fs, RTFx: %.1f",
                         duration, elapsed, duration / elapsed))

            let outputURL = URL(fileURLWithPath: output)
            try WAVWriter.write(samples: audio, sampleRate: 24000, to: outputURL)
            print("Saved to \(output)")
        }
    }
}
