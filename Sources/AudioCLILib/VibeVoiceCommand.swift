import Foundation
import ArgumentParser
import VibeVoiceTTS
import AudioCommon

public struct VibeVoiceCommand: ParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "vibevoice",
        abstract: "Text-to-speech synthesis using Microsoft VibeVoice (MLX, EN/ZH)",
        discussion: """
        VibeVoice is a long-form / multi-speaker TTS model from Microsoft. This
        subcommand targets the Realtime-0.5B variant by default. Voice identity
        comes from a voice-cache .safetensors file (pre-computed from reference
        audio) — pass one via --voice-cache.
        """
    )

    @Argument(help: "Text to synthesize")
    public var text: String

    @Option(name: [.customLong("voice-cache"), .customShort("v")],
            help: "Path to a .safetensors voice cache (required)")
    public var voiceCache: String

    @Option(name: .shortAndLong, help: "Output WAV file path")
    public var output: String = "vibevoice.wav"

    @Option(name: .long, help: "HuggingFace model ID")
    public var model: String = "microsoft/VibeVoice-Realtime-0.5B"

    @Option(name: .long, help: "Qwen2.5 tokenizer model ID")
    public var tokenizer: String = "Qwen/Qwen2.5-0.5B"

    @Option(name: .long, help: "DPM-Solver inference steps (higher = better quality, slower)")
    public var steps: Int = 20

    @Option(name: .long, help: "Classifier-free guidance scale")
    public var cfg: Float = 1.3

    @Option(name: .long, help: "Cap on generated speech tokens")
    public var maxTokens: Int = 500

    @Flag(name: .long, help: "Use the long-form 1.5B variant preset")
    public var longForm: Bool = false

    @Option(name: .long, help: "1.5B-only: reference audio file for the structured-prompt single-shot path")
    public var referenceAudio: String?

    @Option(name: .long, help: "1.5B-only: transcript of --reference-audio")
    public var referenceTranscript: String?

    @Flag(name: .long, help: "Show detailed timing info")
    public var verbose: Bool = false

    public init() {}

    public func validate() throws {
        // 1.5B can run two ways:
        //   1. --reference-audio + --reference-transcript (single-shot, recommended)
        //   2. --voice-cache (uses the simpler streaming path, lower quality on 1.5B)
        if longForm, let ref = referenceAudio, let _ = referenceTranscript {
            guard FileManager.default.fileExists(atPath: ref) else {
                throw ValidationError("Reference audio not found at \(ref)")
            }
            return
        }
        guard FileManager.default.fileExists(atPath: voiceCache) else {
            throw ValidationError("Voice cache not found at \(voiceCache) (or supply --reference-audio + --reference-transcript with --long-form)")
        }
    }

    public func run() throws {
        try runAsync {
            var config: VibeVoiceTTSModel.Configuration = longForm
                ? .longForm1_5B
                : VibeVoiceTTSModel.Configuration()
            // Allow CLI overrides.
            if !longForm {
                config.modelId = model
                config.tokenizerModelId = tokenizer
            }
            config.numInferenceSteps = steps
            config.cfgScale = cfg
            config.maxSpeechTokens = maxTokens

            print("Loading VibeVoice model (\(config.modelId))...")
            let tts = try await VibeVoiceTTSModel.fromPretrained(
                configuration: config,
                progressHandler: reportProgress
            )

            let start = CFAbsoluteTimeGetCurrent()
            let audio: [Float]
            let outRate: Int
            if longForm, let refPath = referenceAudio, let refText = referenceTranscript {
                print("Loading reference audio: \(refPath)")
                let refURL = URL(fileURLWithPath: refPath)
                let refSamples = try AudioFileLoader.load(url: refURL, targetSampleRate: 24000)
                // Re-load via the proper unified-LM 1.5B model class.
                print("Loading 1.5B unified-LM model (\(model))...")
                var cfg15 = VibeVoice15BTTSModel.Configuration()
                cfg15.modelId = model
                cfg15.tokenizerModelId = tokenizer
                cfg15.numInferenceSteps = steps
                cfg15.cfgScale = cfg
                cfg15.maxSpeechTokens = maxTokens
                let tts15 = try await VibeVoice15BTTSModel.fromPretrained(
                    configuration: cfg15,
                    progressHandler: reportProgress
                )
                print("Synthesizing (1.5B unified LM): \"\(text)\"")
                audio = try await tts15.generate(
                    text: text,
                    referenceAudio: refSamples,
                    referenceTranscript: refText,
                    sampleRate: 24000
                )
                outRate = tts15.sampleRate
            } else {
                print("Loading voice cache: \(voiceCache)")
                try tts.loadVoice(from: voiceCache)
                print("Synthesizing: \"\(text)\"")
                audio = try await tts.generate(text: text)
                outRate = tts.sampleRate
            }
            let elapsed = CFAbsoluteTimeGetCurrent() - start

            guard !audio.isEmpty else {
                print("Error: no audio generated")
                throw ExitCode(1)
            }

            let duration = Double(audio.count) / Double(outRate)
            print(String(format: "  Duration: %.2fs, Time: %.3fs, RTFx: %.1f",
                         duration, elapsed, duration / elapsed))

            let outputURL = URL(fileURLWithPath: output)
            try WAVWriter.write(samples: audio, sampleRate: outRate, to: outputURL)
            print("Saved to \(output)")
        }
    }
}
