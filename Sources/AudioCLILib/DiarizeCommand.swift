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

    @Option(name: .long, help: "Minimum number of speakers (0 = auto)")
    public var minSpeakers: Int = 0

    @Option(name: .long, help: "Maximum number of speakers (0 = auto)")
    public var maxSpeakers: Int = 0

    @Option(name: .long, help: "Enrollment audio for target speaker extraction")
    public var targetSpeaker: String?

    @Option(name: .long, help: "Speaker embedding engine: mlx (default) or coreml")
    public var embeddingEngine: String = "mlx"

    @Flag(name: .long, help: "Output as JSON")
    public var json: Bool = false

    @Flag(name: .long, help: "Output as RTTM (standard diarization evaluation format)")
    public var rttm: Bool = false

    @Option(name: .long, help: "Reference RTTM file to score against (computes DER)")
    public var scoreAgainst: String?

    public init() {}

    public func run() throws {
        try runAsync {
            print("Loading audio: \(audioFile)")
            let audio = try AudioFileLoader.load(
                url: URL(fileURLWithPath: audioFile), targetSampleRate: 16000)
            let duration = formatDuration(audio.count, sampleRate: 16000)
            print("  Loaded \(audio.count) samples (\(duration)s)")

            guard let embEngine = WeSpeakerEngine(rawValue: embeddingEngine) else {
                print("Error: unknown embedding engine '\(embeddingEngine)'. Use 'mlx' or 'coreml'.")
                return
            }

            let config = DiarizationConfig(
                minSpeakers: minSpeakers,
                maxSpeakers: maxSpeakers
            )

            print("Loading diarization models (embedding engine: \(embEngine.rawValue))...")
            let pipeline = try await DiarizationPipeline.fromPretrained(
                embeddingEngine: embEngine,
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

                if rttm {
                    let basename = URL(fileURLWithPath: audioFile).deletingPathExtension().lastPathComponent
                    let rttmSegments = toRTTM(segments: result.segments, filename: basename)
                    print(formatRTTM(rttmSegments))
                } else if json {
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

                // DER scoring against reference
                if let refFile = scoreAgainst {
                    let refContent = try String(contentsOfFile: refFile, encoding: .utf8)
                    let refSegments = parseRTTM(refContent)
                    let derResult = computeDERWithOptimalMapping(
                        reference: refSegments, hypothesis: result.segments
                    )
                    print("\n--- DER Scoring ---")
                    print("Total speech: \(String(format: "%.2f", derResult.totalSpeech))s")
                    print("Missed speech: \(String(format: "%.2f", derResult.missedSpeech))s")
                    print("False alarm: \(String(format: "%.2f", derResult.falseAlarm))s")
                    print("Confusion: \(String(format: "%.2f", derResult.confusion))s")
                    print("DER: \(String(format: "%.1f", derResult.derPercent))%")
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
