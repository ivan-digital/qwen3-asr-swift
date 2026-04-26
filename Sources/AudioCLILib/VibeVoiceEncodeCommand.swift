import Foundation
import ArgumentParser
import VibeVoiceTTS
import AudioCommon

public struct VibeVoiceEncodeCommand: ParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "vibevoice-encode-voice",
        abstract: "Mint a VibeVoice voice cache from a reference audio recording + transcript",
        discussion: """
        Encodes the reference audio through VibeVoice's acoustic tokenizer and
        runs both the text and audio through the TTS LM to produce a
        precomputed .safetensors voice cache. The result can be loaded by
        `audio vibevoice ... --voice-cache <out>` to synthesize new speech in
        the reference speaker's voice.

        Audio is resampled to 24 kHz mono internally. Provide the actual
        spoken text as the transcript for best speaker fidelity.
        """
    )

    @Argument(help: "Reference audio file (WAV / m4a / etc.)")
    public var input: String

    @Argument(help: "Transcript — the actual words spoken in the input audio")
    public var transcript: String

    @Option(name: .shortAndLong, help: "Output voice cache (.safetensors)")
    public var output: String = "voice.safetensors"

    @Option(name: .long, help: "HuggingFace model ID")
    public var model: String = "microsoft/VibeVoice-Realtime-0.5B"

    @Option(name: .long, help: "Qwen2.5 tokenizer model ID")
    public var tokenizer: String = "Qwen/Qwen2.5-0.5B"

    @Flag(name: .long, help: "Use the long-form 1.5B variant preset")
    public var longForm: Bool = false

    public init() {}

    public func validate() throws {
        guard FileManager.default.fileExists(atPath: input) else {
            throw ValidationError("Input audio file not found at \(input)")
        }
    }

    public func run() throws {
        try runAsync {
            var config: VibeVoiceTTSModel.Configuration = longForm
                ? .longForm1_5B
                : VibeVoiceTTSModel.Configuration()
            if !longForm {
                config.modelId = model
                config.tokenizerModelId = tokenizer
            }

            print("Loading VibeVoice model (\(config.modelId))...")
            let tts = try await VibeVoiceTTSModel.fromPretrained(
                configuration: config,
                progressHandler: reportProgress
            )

            print("Loading reference audio: \(input)")
            let inputURL = URL(fileURLWithPath: input)
            // AudioFileLoader.load resamples to 24 kHz internally.
            let samples = try AudioFileLoader.load(url: inputURL, targetSampleRate: 24000)

            print("Encoding voice — \(samples.count) samples @ 24000 Hz, transcript: \"\(transcript)\"")
            let start = CFAbsoluteTimeGetCurrent()
            let outputURL = URL(fileURLWithPath: output)
            try tts.encodeAndSaveVoice(
                referenceAudio: samples,
                sampleRate: 24000,
                transcript: transcript,
                to: outputURL
            )
            let elapsed = CFAbsoluteTimeGetCurrent() - start
            print(String(format: "  Encoded in %.2fs", elapsed))
            print("Saved to \(output)")
        }
    }
}
