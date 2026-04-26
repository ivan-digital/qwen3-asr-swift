import Foundation
import ArgumentParser
import MADLADTranslation

public struct TranslateCommand: ParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "translate",
        abstract: "Translate text into a target language using MADLAD-400 (MLX, Apple Silicon)"
    )

    @Argument(help: "Text to translate. Omit to read from stdin (e.g. `audio transcribe x.wav | audio translate --to es`).")
    public var text: String?

    @Option(name: [.short, .long], help: "Target language code (ISO 639-1, e.g. es, zh, fr, ja, de).")
    public var to: String

    @Option(name: .long, help: "HuggingFace model id.")
    public var model: String = MADLADTranslator.defaultModelId

    @Option(name: .long, help: "Quantization variant: int4 (default, smaller) or int8 (higher quality).")
    public var quantization: String = "int4"

    @Option(name: .long, help: "Maximum tokens to generate.")
    public var maxTokens: Int = 256

    @Option(name: .long, help: "Sampling temperature (0 = greedy, recommended for translation).")
    public var temperature: Double = 0.0

    @Option(name: .long, help: "Top-K sampling cutoff (0 = disabled).")
    public var topK: Int = 0

    @Option(name: .long, help: "Top-P (nucleus) sampling cutoff.")
    public var topP: Double = 1.0

    @Flag(name: .long, help: "Stream translated tokens as they are produced.")
    public var stream: Bool = false

    @Flag(name: .long, help: "Output JSON with translation + metrics.")
    public var json: Bool = false

    public init() {}

    public func run() throws {
        try runAsync {
            let source = try resolveInput()
            guard !source.isEmpty else {
                throw ValidationError("No input text. Pass as argument or pipe via stdin.")
            }

            let quant: MADLADTranslator.Quantization = (quantization == "int8") ? .int8 : .int4

            FileHandle.standardError.write(
                "Loading MADLAD (\(quantization))...\n".data(using: .utf8)!)
            let translator = try await MADLADTranslator.fromPretrained(
                modelId: model,
                quantization: quant,
                progressHandler: { progress, status in
                    let pct = Int(progress * 100)
                    FileHandle.standardError.write(
                        "\r  \(status) \(pct)%   ".data(using: .utf8)!)
                }
            )
            FileHandle.standardError.write("\n".data(using: .utf8)!)

            let sampling = TranslationSamplingConfig(
                temperature: Float(temperature),
                topK: topK,
                topP: Float(topP),
                maxTokens: maxTokens
            )

            if stream && !json {
                let stream = translator.translateStream(
                    source, to: to, sampling: sampling)
                for try await piece in stream {
                    print(piece, terminator: "")
                    fflush(stdout)
                }
                print()
            } else {
                let translated = try translator.translate(
                    source, to: to, sampling: sampling)
                if json {
                    let metrics = translator.lastMetrics
                    let payload: [String: Any] = [
                        "source": source,
                        "target_language": to,
                        "translation": translated,
                        "metrics": [
                            "source_tokens": metrics.sourceTokens,
                            "generated_tokens": metrics.generatedTokens,
                            "encode_ms": metrics.encodeTimeMs,
                            "decode_ms": metrics.decodeTimeMs,
                            "tokens_per_sec": metrics.tokensPerSecond,
                        ],
                    ]
                    let data = try JSONSerialization.data(
                        withJSONObject: payload, options: [.prettyPrinted, .sortedKeys])
                    if let s = String(data: data, encoding: .utf8) { print(s) }
                } else {
                    print(translated)
                }
            }
        }
    }

    private func resolveInput() throws -> String {
        if let text = text { return text }
        // Read all of stdin.
        let data = FileHandle.standardInput.readDataToEndOfFile()
        return String(data: data, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
    }
}
