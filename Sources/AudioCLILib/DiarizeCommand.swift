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

    @Option(name: .long, help: "Enrollment audio for target speaker extraction")
    public var targetSpeaker: String?

    @Option(name: .long, help: "Diarization engine: pyannote (default) or sortformer")
    public var engine: String = "pyannote"

    @Option(name: .long, help: "Speaker embedding engine: mlx (default) or coreml")
    public var embeddingEngine: String = "mlx"

    @Flag(name: .long, help: "Output as JSON")
    public var json: Bool = false

    @Flag(name: .long, help: "Output as RTTM (standard diarization evaluation format)")
    public var rttm: Bool = false

    @Flag(name: .long, help: "Pre-filter with Silero VAD to reduce false alarms")
    public var vadFilter: Bool = false

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

            let config = DiarizationConfig()

            if engine == "sortformer" {
                #if canImport(CoreML)
                try await runSortformer(audio: audio, config: config)
                #else
                print("Error: Sortformer requires CoreML (not available on this platform).")
                #endif
            } else if engine == "pyannote" {
                try await runPyannote(audio: audio, config: config)
            } else {
                print("Error: unknown engine '\(engine)'. Use 'pyannote' or 'sortformer'.")
            }
        }
    }

    #if canImport(CoreML)
    private func runSortformer(audio: [Float], config: DiarizationConfig) async throws {
        if targetSpeaker != nil {
            print("Warning: --target-speaker is not supported with Sortformer (no speaker embeddings). Ignoring.")
        }

        print("Loading Sortformer model...")
        let diarizer = try await SortformerDiarizer.fromPretrained(
            progressHandler: reportProgress
        )

        print("Running diarization (Sortformer)...")
        let start = Date()
        let result = diarizer.diarize(audio: audio, sampleRate: 16000, config: config)
        let elapsed = Date().timeIntervalSince(start)

        outputResult(result, elapsed: elapsed)

        if let refFile = scoreAgainst {
            try scoreDER(result: result, refFile: refFile)
        }
    }
    #endif

    private func runPyannote(audio: [Float], config: DiarizationConfig) async throws {
        guard let embEngine = WeSpeakerEngine(rawValue: embeddingEngine) else {
            print("Error: unknown embedding engine '\(embeddingEngine)'. Use 'mlx' or 'coreml'.")
            return
        }

        print("Loading diarization models (embedding engine: \(embEngine.rawValue)\(vadFilter ? ", VAD filter" : ""))...")
        let pipeline = try await DiarizationPipeline.fromPretrained(
            embeddingEngine: embEngine,
            useVADFilter: vadFilter,
            progressHandler: reportProgress
        )

        if let enrollmentFile = targetSpeaker {
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
            print("Running diarization...")
            let start = Date()
            let result = pipeline.diarize(
                audio: audio, sampleRate: 16000, config: config)
            let elapsed = Date().timeIntervalSince(start)

            outputResult(result, elapsed: elapsed)

            if let refFile = scoreAgainst {
                try scoreDER(result: result, refFile: refFile)
            }
        }
    }

    private func outputResult(_ result: DiarizationResult, elapsed: TimeInterval) {
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
    }

    private func scoreDER(result: DiarizationResult, refFile: String) throws {
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
