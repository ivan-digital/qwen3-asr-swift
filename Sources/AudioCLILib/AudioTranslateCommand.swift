import Foundation
import ArgumentParser
import HibikiTranslate
import AudioCommon
import MLX

public struct AudioTranslateCommand: ParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "audio-translate",
        abstract: "Translate speech with Hibiki Zero-3B (FR/ES/PT/DE → EN, on-device, MLX)"
    )

    @Argument(help: "Input audio WAV file (mono; resampled to 24 kHz internally).")
    public var input: String

    @Option(name: .shortAndLong, help: "Output WAV file path.")
    public var output: String = "translated.wav"

    @Option(name: .long, help: "Source language hint (fr, es, pt, de). Hibiki Zero auto-detects; this is metadata only.")
    public var sourceLang: String = "fr"

    @Option(name: .long, help: "Quantization variant: 4bit (default) or 8bit.")
    public var quantization: String = "4bit"

    @Option(name: .long, help: "HuggingFace model id (overrides --quantization).")
    public var modelId: String?

    @Flag(name: .long, help: "Pre-compile temporal transformer (warmup). Currently no-op for Hibiki driver.")
    public var compile: Bool = false

    @Flag(name: .long, help: "Print per-phase timings.")
    public var verbose: Bool = false

    @Flag(name: .long, help: "Print model's inner monologue (English transcript).")
    public var transcript: Bool = false

    public init() {}

    public func run() throws {
        guard let language = HibikiSourceLanguage(rawValue: sourceLang) else {
            throw ValidationError("Unknown --source-lang: '\(sourceLang)'. Supported: fr, es, pt, de.")
        }

        let resolvedModelId = modelId ?? {
            switch quantization.lowercased() {
            case "8bit": return HibikiTranslateModel.modelId8bit
            case "4bit", "": return HibikiTranslateModel.defaultModelId
            default: return HibikiTranslateModel.defaultModelId
            }
        }()

        let inputURL = URL(fileURLWithPath: (input as NSString).expandingTildeInPath)
        let outputURL = URL(fileURLWithPath: (output as NSString).expandingTildeInPath)

        let pcm = try AudioFileLoader.load(url: inputURL, targetSampleRate: 24000)
        if verbose {
            let duration = Double(pcm.count) / 24000.0
            print("Loaded \(input): \(pcm.count) samples, \(String(format: "%.2f", duration))s @ 24 kHz, lang=\(language.displayName)")
            print("Loading Hibiki Zero-3B (\(resolvedModelId))...")
        }

        let semaphore = DispatchSemaphore(value: 0)
        var loadedModel: HibikiTranslateModel?
        var loadError: Error?
        Task {
            do {
                loadedModel = try await HibikiTranslateModel.fromPretrained(
                    modelId: resolvedModelId,
                    progressHandler: verbose ? { p, msg in
                        print("  \(Int(p * 100))%  \(msg)")
                    } : nil
                )
                semaphore.signal()
            } catch {
                loadError = error
                semaphore.signal()
            }
        }
        semaphore.wait()
        if let e = loadError { throw e }
        guard let model = loadedModel else {
            throw ValidationError("Failed to load model.")
        }

        if compile { model.warmUp() }

        if verbose { print("Translating...") }
        let (audio, textTokens) = model.translate(
            sourceAudio: pcm,
            sourceLanguage: language,
            verbose: verbose
        )

        try WAVWriter.write(samples: audio, sampleRate: 24000, to: outputURL)
        if verbose {
            let duration = Double(audio.count) / 24000.0
            print("Wrote \(output): \(audio.count) samples, \(String(format: "%.2f", duration))s")
        }

        if transcript {
            let count = textTokens.count
            print("English transcript token count: \(count)")
            // Decoding requires SentencePieceModel — emit raw ids for v1.
            // SPM-48k decode wiring is a v2 follow-up.
            print("Token IDs (first 50): \(Array(textTokens.prefix(50)))")
        }
    }
}
