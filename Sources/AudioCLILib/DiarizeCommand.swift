import Foundation
import ArgumentParser
import SpeechVAD
import AudioCommon

public struct DiarizeCommand: ParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "diarize",
        abstract: "Identify speakers and their speech segments in audio"
    )

    @Argument(help: "Audio file to analyze (WAV, any sample rate)")
    public var audioFile: String

    @Option(name: .long, help: "Clustering distance threshold (0.0-2.0)")
    public var threshold: Float = DiarizationConfig.default.clusteringThreshold

    @Option(name: .long, help: "Maximum number of speakers (0 = auto)")
    public var maxSpeakers: Int = 0

    @Option(name: .long, help: "Enrollment audio for target speaker extraction")
    public var targetSpeaker: String?

    @Flag(name: .long, help: "Output as JSON")
    public var json: Bool = false

    public init() {}

    public func run() throws {
        try runAsync {
            print("Loading audio: \(audioFile)")
            let audio = try AudioFileLoader.load(
                url: URL(fileURLWithPath: audioFile), targetSampleRate: 16000)
            let duration = formatDuration(audio.count, sampleRate: 16000)
            print("  Loaded \(audio.count) samples (\(duration)s)")

            let config = DiarizationConfig(
                clusteringThreshold: threshold,
                maxSpeakers: maxSpeakers
            )

            print("Loading diarization models...")
            let pipeline = try await DiarizationPipeline.fromPretrained(
                progressHandler: reportProgress
            )

            if let enrollmentFile = targetSpeaker {
                // Speaker extraction mode
                print("Loading enrollment audio: \(enrollmentFile)")
                let enrollAudio = try AudioFileLoader.load(
                    url: URL(fileURLWithPath: enrollmentFile), targetSampleRate: 16000)

                print("Extracting target speaker embedding...")
                let targetEmb = pipeline.embeddingModel.embed(
                    audio: enrollAudio, sampleRate: 16000)

                print("Extracting target speaker segments...")
                let start = Date()
                let segments = pipeline.extractSpeaker(
                    audio: audio, sampleRate: 16000,
                    targetEmbedding: targetEmb, config: config
                )
                let elapsed = Date().timeIntervalSince(start)

                if json {
                    printSpeechJSON(segments)
                } else {
                    if segments.isEmpty {
                        print("Target speaker not found.")
                    } else {
                        for seg in segments {
                            let s = String(format: "%.2f", seg.startTime)
                            let e = String(format: "%.2f", seg.endTime)
                            let d = String(format: "%.2f", seg.duration)
                            print("Target: [\(s)s - \(e)s] (\(d)s)")
                        }
                        let totalSpeech = segments.reduce(Float(0)) { $0 + $1.duration }
                        print("\n\(segments.count) segment(s), \(String(format: "%.2f", totalSpeech))s total")
                    }
                    print("Extraction took \(String(format: "%.2f", elapsed))s")
                }
            } else {
                // Full diarization mode
                print("Running diarization...")
                let start = Date()
                let result = pipeline.diarize(
                    audio: audio, sampleRate: 16000, config: config)
                let elapsed = Date().timeIntervalSince(start)

                if json {
                    printDiarizeJSON(result)
                } else {
                    if result.segments.isEmpty {
                        print("No speech detected.")
                    } else {
                        for seg in result.segments {
                            let s = String(format: "%.2f", seg.startTime)
                            let e = String(format: "%.2f", seg.endTime)
                            let d = String(format: "%.2f", seg.duration)
                            print("Speaker \(seg.speakerId): [\(s)s - \(e)s] (\(d)s)")
                        }
                        print("\n--- \(result.numSpeakers) speaker(s) ---")
                    }
                    print("Diarization took \(String(format: "%.2f", elapsed))s")
                }
            }
        }
    }

    private func printDiarizeJSON(_ result: DiarizationResult) {
        var items = [[String: Any]]()
        for seg in result.segments {
            items.append([
                "start": Double(String(format: "%.3f", seg.startTime))!,
                "end": Double(String(format: "%.3f", seg.endTime))!,
                "duration": Double(String(format: "%.3f", seg.duration))!,
                "speaker": seg.speakerId,
            ])
        }
        let output: [String: Any] = [
            "segments": items,
            "num_speakers": result.numSpeakers,
        ]
        if let data = try? JSONSerialization.data(withJSONObject: output, options: .prettyPrinted),
           let str = String(data: data, encoding: .utf8) {
            print(str)
        }
    }

    private func printSpeechJSON(_ segments: [SpeechSegment]) {
        var items = [[String: Any]]()
        for seg in segments {
            items.append([
                "start": Double(String(format: "%.3f", seg.startTime))!,
                "end": Double(String(format: "%.3f", seg.endTime))!,
                "duration": Double(String(format: "%.3f", seg.duration))!,
            ])
        }
        if let data = try? JSONSerialization.data(withJSONObject: items, options: .prettyPrinted),
           let str = String(data: data, encoding: .utf8) {
            print(str)
        }
    }
}
