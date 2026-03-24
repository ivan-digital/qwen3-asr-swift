import Foundation
import ArgumentParser
import SourceSeparation
import AudioCommon

public struct SeparateCommand: ParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "separate",
        abstract: "Separate a music track into stems (vocals, drums, bass, other)"
    )

    @Argument(help: "Input audio file (WAV, stereo 44.1kHz)")
    public var input: String

    @Option(name: .long, help: "Output directory (default: <input>_stems/)")
    public var outputDir: String?

    @Option(name: .long, help: "Stems to extract: vocals,drums,bass,other (default: all)")
    public var stems: String?

    @Option(name: .long, help: "HuggingFace model ID")
    public var modelId: String = SourceSeparator.defaultModelId

    @Flag(name: .long, help: "Show timing info")
    public var verbose: Bool = false

    public init() {}

    public func run() throws {
        let inputURL = URL(fileURLWithPath: input)
        let baseName = inputURL.deletingPathExtension().lastPathComponent
        let outDir = URL(fileURLWithPath: outputDir ?? "\(baseName)_stems")

        // Parse target stems
        let targets: [SeparationTarget]
        if let stemsStr = stems {
            targets = stemsStr.split(separator: ",").compactMap {
                SeparationTarget(rawValue: String($0).trimmingCharacters(in: .whitespaces))
            }
        } else {
            targets = SeparationTarget.allCases
        }

        try FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)

        try runAsync {
            print("Loading model...")
            let separator = try await SourceSeparator.fromPretrained(
                modelId: modelId, progressHandler: reportProgress)

            print("Loading audio: \(input)")
            let mono = try AudioFileLoader.load(url: inputURL, targetSampleRate: 44100)
            let audio = [mono, mono]  // Duplicate mono to stereo (TODO: proper stereo loading)
            let duration = Double(mono.count) / 44100.0
            print("  Duration: \(String(format: "%.1f", duration))s")

            print("Separating into \(targets.map(\.rawValue).joined(separator: ", "))...")
            let startTime = CFAbsoluteTimeGetCurrent()

            let results = separator.separate(audio: audio, targets: targets)

            let elapsed = CFAbsoluteTimeGetCurrent() - startTime
            let rtf = elapsed / duration

            for (target, stemAudio) in results {
                let outputURL = outDir.appendingPathComponent("\(target.rawValue).wav")
                // Mix stereo to mono for WAV output (TODO: stereo WAV writer)
                let mixedMono = zip(stemAudio[0], stemAudio[1]).map { ($0 + $1) / 2 }
                try WAVWriter.write(samples: mixedMono, sampleRate: 44100, to: outputURL)
                print("  Saved: \(outputURL.lastPathComponent)")
            }

            print("Done in \(String(format: "%.1f", elapsed))s (RTF: \(String(format: "%.2f", rtf)))")
        }
    }
}
