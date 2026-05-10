import Foundation
import ArgumentParser
import VibeVoiceTTS
import AudioCommon

public struct VibeVoiceEncodeCommand: ParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "vibevoice-encode-voice",
        abstract: "Mint a VibeVoice voice cache from a reference audio recording + transcript (Realtime-0.5B path; English only).",
        discussion: """
        Encodes the reference audio through VibeVoice's acoustic tokenizer and
        runs both the text and audio through the TTS LM to produce a
        precomputed .safetensors voice cache that `audio vibevoice ...
        --voice-cache <out>` can load.

        IMPORTANT — checkpoint availability:
        Microsoft's `VibeVoice-Realtime-0.5B` is distributed inference-only
        and does not ship the acoustic encoder, so this command currently
        fails fast and points at the only real workflow speech-swift can
        run end-to-end on its own: clone an arbitrary speaker from raw
        audio via `audio vibevoice ... --long-form --reference-audio <wav>
        --reference-transcript "..."`. That path runs the full
        VibeVoice-1.5B pipeline (which does ship the encoder) and inlines
        the encoding on every synthesis call.

        Audio is resampled to 24 kHz mono internally. Provide the actual
        spoken text as the transcript for best speaker fidelity.

        Language: English only — Realtime-0.5B is English-trained; per
        Microsoft's model card, other languages may produce unpredictable
        results.
        """
    )

    @Argument(help: "Reference audio file (WAV / m4a / etc.)")
    public var input: String

    @Argument(help: "Transcript of --input audio (English; Realtime-0.5B is EN-only)")
    public var transcript: String

    @Option(name: .shortAndLong, help: "Output voice cache (.safetensors)")
    public var output: String = "voice.safetensors"

    @Option(name: .long, help: "HuggingFace model ID (default: microsoft/VibeVoice-Realtime-0.5B)")
    public var model: String?

    @Option(name: .long, help: "Qwen2.5 tokenizer model ID (default: Qwen/Qwen2.5-0.5B)")
    public var tokenizer: String?

    public init() {}

    public func validate() throws {
        guard FileManager.default.fileExists(atPath: input) else {
            throw ValidationError("Input audio file not found at \(input)")
        }
    }

    public func run() throws {
        try runAsync {
            var config = VibeVoiceTTSModel.Configuration()
            // Only override the preset's defaults when the caller passed
            // --model / --tokenizer explicitly.
            if let m = model { config.modelId = m }
            if let t = tokenizer { config.tokenizerModelId = t }

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
