import Foundation
import ArgumentParser
import Qwen3ASR
import ParakeetASR
import AudioCommon

public struct TranscribeBatchCommand: ParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "transcribe-batch",
        abstract: "Transcribe a directory of audio files (model loaded once)"
    )

    @Argument(help: "Directory containing audio files (WAV, FLAC, etc.)")
    public var inputDir: String

    @Option(name: .long, help: "Output directory for transcriptions (one .txt per file)")
    public var outputDir: String?

    @Option(name: .long, help: "ASR engine: qwen3 (default), parakeet")
    public var engine: String = "qwen3"

    @Option(name: .shortAndLong, help: "[qwen3] Model: 0.6B (default), 0.6B-8bit, 1.7B, 1.7B-4bit")
    public var model: String = "0.6B"

    @Option(name: .long, help: "Language hint (optional)")
    public var language: String?

    @Option(name: .long, help: "Audio file extensions to process (default: wav,flac,mp3)")
    public var extensions: String = "wav,flac,mp3"

    @Flag(name: .long, help: "Output results as JSON lines (one per file)")
    public var jsonl: Bool = false

    public init() {}

    public func run() throws {
        switch engine.lowercased() {
        case "parakeet":
            try runParakeetBatch()
        default:
            try runQwen3Batch()
        }
    }

    private func runQwen3Batch() throws {
        try runAsync {
            let modelId = resolveASRModelId(model)
            let sizeLabel = ASRModelSize.detect(from: modelId) == .large ? "1.7B" : "0.6B"

            // Discover audio files
            let files = findAudioFiles()
            guard !files.isEmpty else {
                print("No audio files found in \(inputDir)")
                return
            }
            print("Found \(files.count) audio files")

            // Load model once
            print("Loading model (\(sizeLabel)): \(modelId)")
            let loadStart = CFAbsoluteTimeGetCurrent()
            let asrModel = try await Qwen3ASRModel.fromPretrained(
                modelId: modelId, progressHandler: reportProgress)
            let loadTime = CFAbsoluteTimeGetCurrent() - loadStart
            print(String(format: "  Model loaded in %.2fs", loadTime))

            // Prepare output directory
            let outDir = prepareOutputDir()

            // Warmup
            let warmupStart = CFAbsoluteTimeGetCurrent()
            let warmupAudio = try AudioFileLoader.load(
                url: files[0], targetSampleRate: 24000)
            _ = asrModel.transcribe(audio: warmupAudio, sampleRate: 24000, language: language)
            let warmupTime = CFAbsoluteTimeGetCurrent() - warmupStart
            print(String(format: "  Warmup: %.2fs", warmupTime))

            // Transcribe all files
            var totalInference = 0.0
            var totalAudio = 0.0
            let batchStart = CFAbsoluteTimeGetCurrent()

            for (idx, fileURL) in files.enumerated() {
                let name = fileURL.deletingPathExtension().lastPathComponent
                let pct = Double(idx + 1) / Double(files.count) * 100

                do {
                    let audio = try AudioFileLoader.load(
                        url: fileURL, targetSampleRate: 24000)
                    let duration = Double(audio.count) / 24000.0

                    let t0 = CFAbsoluteTimeGetCurrent()
                    let result = asrModel.transcribe(
                        audio: audio, sampleRate: 24000, language: language)
                    let elapsed = CFAbsoluteTimeGetCurrent() - t0
                    let rtf = elapsed / max(duration, 0.001)

                    totalInference += elapsed
                    totalAudio += duration

                    // Output
                    if jsonl {
                        let escaped = result
                            .replacingOccurrences(of: "\\", with: "\\\\")
                            .replacingOccurrences(of: "\"", with: "\\\"")
                        print("{\"file\":\"\(name)\",\"text\":\"\(escaped)\","
                            + String(format: "\"time\":%.3f,\"rtf\":%.4f,\"duration\":%.2f}",
                                     elapsed, rtf, duration))
                    } else {
                        print(String(format: "  [%d/%d] (%.0f%%) %@: %@  (%.2fs, RTF=%.3f)",
                                     idx + 1, files.count, pct, name, result, elapsed, rtf))
                    }

                    // Save transcript
                    if let outDir = outDir {
                        let outFile = outDir.appendingPathComponent("\(name).txt")
                        try result.write(to: outFile, atomically: true, encoding: .utf8)
                    }
                } catch {
                    if jsonl {
                        print("{\"file\":\"\(name)\",\"error\":\"\(error)\"}")
                    } else {
                        print("  [\(idx + 1)/\(files.count)] \(name): ERROR - \(error)")
                    }
                }
            }

            let batchTime = CFAbsoluteTimeGetCurrent() - batchStart
            let aggRTF = totalInference / max(totalAudio, 0.001)

            print(String(format: "\nBatch complete: %d files, %.1fs audio",
                         files.count, totalAudio))
            print(String(format: "  Total inference: %.2fs, Aggregate RTF: %.4f",
                         totalInference, aggRTF))
            print(String(format: "  Wall time: %.2fs (includes I/O)",
                         batchTime))
            print(String(format: "  Model load: %.2fs, Warmup: %.2fs",
                         loadTime, warmupTime))
        }
    }

    private func runParakeetBatch() throws {
        try runAsync {
            let files = findAudioFiles()
            guard !files.isEmpty else {
                print("No audio files found in \(inputDir)")
                return
            }
            print("Found \(files.count) audio files")

            let parakeetModelId = model.lowercased().contains("8")
                ? ParakeetASRModel.int8ModelId
                : ParakeetASRModel.defaultModelId
            print("Loading Parakeet-TDT model: \(parakeetModelId)")
            let loadStart = CFAbsoluteTimeGetCurrent()
            let parakeet = try await ParakeetASRModel.fromPretrained(
                modelId: parakeetModelId, progressHandler: reportProgress)
            let loadTime = CFAbsoluteTimeGetCurrent() - loadStart

            print("Warming up CoreML...")
            let warmupStart = CFAbsoluteTimeGetCurrent()
            try parakeet.warmUp()
            let warmupTime = CFAbsoluteTimeGetCurrent() - warmupStart
            print(String(format: "  Model loaded in %.2fs, warmup: %.2fs",
                         loadTime, warmupTime))

            let outDir = prepareOutputDir()
            var totalInference = 0.0
            var totalAudio = 0.0

            for (idx, fileURL) in files.enumerated() {
                try autoreleasepool {
                let name = fileURL.deletingPathExtension().lastPathComponent
                let pct = Double(idx + 1) / Double(files.count) * 100

                do {
                    let audio = try AudioFileLoader.load(
                        url: fileURL, targetSampleRate: 16000)
                    let duration = Double(audio.count) / 16000.0

                    let t0 = CFAbsoluteTimeGetCurrent()
                    let result = try parakeet.transcribeAudio(
                        audio, sampleRate: 16000, language: language)
                    let elapsed = CFAbsoluteTimeGetCurrent() - t0
                    let rtf = elapsed / max(duration, 0.001)

                    totalInference += elapsed
                    totalAudio += duration

                    if jsonl {
                        let escaped = result
                            .replacingOccurrences(of: "\\", with: "\\\\")
                            .replacingOccurrences(of: "\"", with: "\\\"")
                        print("{\"file\":\"\(name)\",\"text\":\"\(escaped)\","
                            + String(format: "\"time\":%.3f,\"rtf\":%.4f,\"duration\":%.2f}",
                                     elapsed, rtf, duration))
                    } else {
                        print(String(format: "  [%d/%d] (%.0f%%) %@: %@  (%.2fs, RTF=%.3f)",
                                     idx + 1, files.count, pct, name, result, elapsed, rtf))
                    }

                    if let outDir = outDir {
                        let outFile = outDir.appendingPathComponent("\(name).txt")
                        try result.write(to: outFile, atomically: true, encoding: .utf8)
                    }
                } catch {
                    print("  [\(idx + 1)/\(files.count)] \(name): ERROR - \(error)")
                }
                } // autoreleasepool
            }

            let aggRTF = totalInference / max(totalAudio, 0.001)
            print(String(format: "\nBatch complete: %d files, %.1fs audio",
                         files.count, totalAudio))
            print(String(format: "  Total inference: %.2fs, Aggregate RTF: %.4f",
                         totalInference, aggRTF))
        }
    }

    private func findAudioFiles() -> [URL] {
        let dirURL = URL(fileURLWithPath: inputDir)
        let exts = Set(extensions.split(separator: ",").map { String($0).lowercased() })
        let fm = FileManager.default

        guard let enumerator = fm.enumerator(
            at: dirURL, includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles]) else {
            return []
        }

        var files: [URL] = []
        for case let fileURL as URL in enumerator {
            if exts.contains(fileURL.pathExtension.lowercased()) {
                files.append(fileURL)
            }
        }
        return files.sorted { $0.lastPathComponent < $1.lastPathComponent }
    }

    private func prepareOutputDir() -> URL? {
        guard let dir = outputDir else { return nil }
        let url = URL(fileURLWithPath: dir)
        try? FileManager.default.createDirectory(
            at: url, withIntermediateDirectories: true)
        return url
    }
}
