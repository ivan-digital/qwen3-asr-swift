import Foundation
import AudioCommon

/// Events emitted by the streaming VAD processor.
public enum VADEvent: Sendable {
    /// Speech has been detected and confirmed (duration ≥ minSpeechDuration).
    case speechStarted(time: Float)
    /// Speech has ended (silence ≥ minSilenceDuration).
    case speechEnded(segment: SpeechSegment)
}

/// Event-driven streaming VAD processor.
///
/// Wraps a `SileroVADModel` to provide event-based speech detection.
/// Accepts audio samples of any length, buffers them into 512-sample chunks,
/// runs the model, and applies hysteresis with duration filtering via a
/// four-state machine.
///
/// - Warning: This class is not thread-safe. Create separate instances for concurrent use.
///
/// ```swift
/// let model = try await SileroVADModel.fromPretrained()
/// let processor = StreamingVADProcessor(model: model)
///
/// // Feed audio samples (any length)
/// let events = processor.process(samples: audioBuffer)
/// for event in events {
///     switch event {
///     case .speechStarted(let time):
///         print("Speech started at \(time)s")
///     case .speechEnded(let segment):
///         print("Speech: \(segment.startTime)s - \(segment.endTime)s")
///     }
/// }
///
/// // At end of stream, flush any pending segment
/// let finalEvents = processor.flush()
/// ```
public final class StreamingVADProcessor {

    private let model: SileroVADModel
    private let config: VADConfig
    private let chunkDuration: Float  // seconds per chunk (0.032)

    /// Buffer for accumulating samples until we have a full chunk
    private var buffer: [Float] = []
    /// Number of chunks processed so far
    private var chunkCount: Int = 0

    /// State machine for hysteresis + duration filtering
    private enum State {
        /// No speech detected
        case silence
        /// Onset threshold crossed, waiting for minSpeechDuration
        case pendingSpeech(startTime: Float)
        /// Speech confirmed and speechStarted emitted
        case speech(startTime: Float)
        /// Offset threshold crossed, waiting for minSilenceDuration
        case pendingSilence(speechStart: Float, silenceStart: Float)
    }

    private var state: State = .silence

    /// Create a streaming VAD processor.
    ///
    /// - Parameters:
    ///   - model: Silero VAD model instance
    ///   - config: VAD configuration (thresholds, durations)
    public init(model: SileroVADModel, config: VADConfig = .sileroDefault) {
        self.model = model
        self.config = config
        self.chunkDuration = Float(SileroVADModel.chunkSize) / Float(SileroVADModel.sampleRate)
    }

    /// Feed audio samples and get VAD events back.
    ///
    /// Samples are buffered internally. Events are emitted as soon as the
    /// state machine confirms speech start/end with the configured thresholds
    /// and duration constraints.
    ///
    /// - Parameter samples: PCM Float32 samples at 16kHz (any length)
    /// - Returns: zero or more VAD events
    public func process(samples: [Float]) -> [VADEvent] {
        buffer.append(contentsOf: samples)
        var events = [VADEvent]()

        while buffer.count >= SileroVADModel.chunkSize {
            let chunk = Array(buffer.prefix(SileroVADModel.chunkSize))
            buffer.removeFirst(SileroVADModel.chunkSize)

            let prob = model.processChunk(chunk)
            let time = Float(chunkCount) * chunkDuration
            chunkCount += 1

            events.append(contentsOf: processProb(prob, time: time))
        }

        return events
    }

    /// Flush any pending speech segment at end of stream.
    ///
    /// Call this when the audio stream ends to close any open speech segment.
    ///
    /// - Returns: zero or more final VAD events
    public func flush() -> [VADEvent] {
        // Process any remaining buffered samples (zero-padded)
        var events = [VADEvent]()
        if !buffer.isEmpty {
            var lastChunk = buffer
            lastChunk.append(contentsOf: [Float](repeating: 0, count: SileroVADModel.chunkSize - lastChunk.count))
            buffer.removeAll()

            let prob = model.processChunk(lastChunk)
            let time = Float(chunkCount) * chunkDuration
            chunkCount += 1
            events.append(contentsOf: processProb(prob, time: time))
        }

        let endTime = Float(chunkCount) * chunkDuration

        // Close any open state
        switch state {
        case .silence:
            break
        case .pendingSpeech(let startTime):
            // Check if pending speech meets minimum duration
            if endTime - startTime >= config.minSpeechDuration {
                events.append(.speechStarted(time: startTime))
                events.append(.speechEnded(segment: SpeechSegment(
                    startTime: startTime, endTime: endTime)))
            }
        case .speech(let startTime):
            events.append(.speechEnded(segment: SpeechSegment(
                startTime: startTime, endTime: endTime)))
        case .pendingSilence(let speechStart, let silenceStart):
            // End at the silence start point
            events.append(.speechEnded(segment: SpeechSegment(
                startTime: speechStart, endTime: silenceStart)))
        }

        state = .silence
        return events
    }

    /// Reset all state (model + processor).
    ///
    /// Call between processing different audio streams.
    public func reset() {
        buffer.removeAll()
        chunkCount = 0
        state = .silence
        model.resetState()
    }

    /// Current time position in seconds.
    public var currentTime: Float {
        Float(chunkCount) * chunkDuration
    }

    // MARK: - State Machine

    private func processProb(_ prob: Float, time: Float) -> [VADEvent] {
        var events = [VADEvent]()
        let nextTime = time + chunkDuration

        switch state {
        case .silence:
            if prob >= config.onset {
                state = .pendingSpeech(startTime: time)
            }

        case .pendingSpeech(let startTime):
            if prob < config.offset {
                // False alarm — speech too brief, return to silence
                state = .silence
            } else if nextTime - startTime >= config.minSpeechDuration {
                // Speech confirmed
                events.append(.speechStarted(time: startTime))
                state = .speech(startTime: startTime)
            }
            // else: still pending, keep waiting

        case .speech(let startTime):
            if prob < config.offset {
                // Speech may be ending
                state = .pendingSilence(speechStart: startTime, silenceStart: time)
            }

        case .pendingSilence(let speechStart, let silenceStart):
            if prob >= config.onset {
                // Speech resumed — cancel silence
                state = .speech(startTime: speechStart)
            } else if nextTime - silenceStart >= config.minSilenceDuration {
                // Silence confirmed — emit speechEnded
                events.append(.speechEnded(segment: SpeechSegment(
                    startTime: speechStart, endTime: silenceStart)))
                // Check if new speech is starting
                if prob >= config.onset {
                    state = .pendingSpeech(startTime: time)
                } else {
                    state = .silence
                }
            }
            // else: still waiting for silence confirmation
        }

        return events
    }
}
