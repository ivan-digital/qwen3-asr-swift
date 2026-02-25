import Foundation
import ArgumentParser
import SpeechVAD
import AudioCommon

public struct VadStreamCommand: ParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "vad-stream",
        abstract: "Detect speech segments using streaming Silero VAD (32ms chunks)"
    )

    @Argument(help: "Audio file to analyze (WAV, any sample rate)")
    public var audioFile: String

    @Option(name: .shortAndLong, help: "Model ID on HuggingFace")
    public var model: String = SileroVADModel.defaultModelId

    @Option(name: .long, help: "Onset threshold (speech start)")
    public var onset: Float = VADConfig.sileroDefault.onset

    @Option(name: .long, help: "Offset threshold (speech end)")
    public var offset: Float = VADConfig.sileroDefault.offset

    @Option(name: .long, help: "Minimum speech duration in seconds")
    public var minSpeech: Float = VADConfig.sileroDefault.minSpeechDuration

    @Option(name: .long, help: "Minimum silence duration in seconds")
    public var minSilence: Float = VADConfig.sileroDefault.minSilenceDuration

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

            print("Loading Silero VAD model: \(model)")
            let vadModel = try await SileroVADModel.fromPretrained(
                modelId: model,
                progressHandler: reportProgress
            )

            let vadConfig = VADConfig(
                onset: onset,
                offset: offset,
                minSpeechDuration: minSpeech,
                minSilenceDuration: minSilence,
                windowDuration: VADConfig.sileroDefault.windowDuration,
                stepRatio: VADConfig.sileroDefault.stepRatio
            )

            let processor = StreamingVADProcessor(model: vadModel, config: vadConfig)

            print("Processing in 32ms chunks...")
            let start = Date()
            var allEvents = [VADEvent]()

            // Process audio in 512-sample chunks (simulating real-time streaming)
            let chunkSize = SileroVADModel.chunkSize
            var offset = 0

            while offset + chunkSize <= audio.count {
                let chunk = Array(audio[offset ..< (offset + chunkSize)])
                let events = processor.process(samples: chunk)
                for event in events {
                    if !json { printEvent(event) }
                }
                allEvents.append(contentsOf: events)
                offset += chunkSize
            }

            // Handle remaining samples
            if offset < audio.count {
                let remaining = Array(audio[offset...])
                let events = processor.process(samples: remaining)
                for event in events {
                    if !json { printEvent(event) }
                }
                allEvents.append(contentsOf: events)
            }

            // Flush any pending segment
            let flushEvents = processor.flush()
            for event in flushEvents {
                if !json { printEvent(event) }
            }
            allEvents.append(contentsOf: flushEvents)

            let elapsed = Date().timeIntervalSince(start)

            if json {
                printJSON(allEvents)
            } else {
                // Summary
                let segments = allEvents.compactMap { event -> SpeechSegment? in
                    if case .speechEnded(let seg) = event { return seg }
                    return nil
                }
                let totalSpeech = segments.reduce(Float(0)) { $0 + $1.duration }
                print("\n\(segments.count) segment(s), \(String(format: "%.2f", totalSpeech))s total speech")
                print("Processing took \(String(format: "%.3f", elapsed))s (\(String(format: "%.0f", Double(audio.count) / 16000.0 / elapsed))x real-time)")
            }
        }
    }

    private func printEvent(_ event: VADEvent) {
        switch event {
        case .speechStarted(let time):
            print("[\(String(format: "%.2f", time))s] Speech started")
        case .speechEnded(let seg):
            let d = String(format: "%.2f", seg.duration)
            print("[\(String(format: "%.2f", seg.endTime))s] Speech ended (\(d)s)")
        }
    }

    private func printJSON(_ events: [VADEvent]) {
        let segments = events.compactMap { event -> SpeechSegment? in
            if case .speechEnded(let seg) = event { return seg }
            return nil
        }
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
