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

    @Option(name: .long, help: "Model variant: hq (default, 8.9M/stem) or l (28.3M/stem)")
    public var model: String = "hq"

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
            let modelId = model.lowercased() == "l" ? SourceSeparator.largeModelId : SourceSeparator.defaultModelId
            print("Loading model (\(model.lowercased() == "l" ? "UMX-L" : "UMX-HQ"))...")
            let separator = try await SourceSeparator.fromPretrained(
                modelId: modelId, progressHandler: reportProgress)

            print("Loading audio: \(input)")
            let audio = try AudioFileLoader.loadStereo(url: inputURL, targetSampleRate: 44100)
            let duration = Double(audio[0].count) / 44100.0
            print("  Duration: \(String(format: "%.1f", duration))s")

            print("Separating into \(targets.map(\.rawValue).joined(separator: ", "))...")
            let startTime = CFAbsoluteTimeGetCurrent()

            let results = separator.separate(audio: audio, targets: targets)

            let elapsed = CFAbsoluteTimeGetCurrent() - startTime
            let rtf = elapsed / duration

            for (target, stemAudio) in results {
                let outputURL = outDir.appendingPathComponent("\(target.rawValue).wav")
                try WAVWriter.writeStereo(left: stemAudio[0], right: stemAudio[1], sampleRate: 44100, to: outputURL)
                print("  Saved: \(outputURL.lastPathComponent)")
            }

            print("Done in \(String(format: "%.1f", elapsed))s (RTF: \(String(format: "%.2f", rtf)))")
        }
    }
}
