import Foundation
import AudioCommon

/// A wake-word detector that exposes its trained keywords as a simple protocol.
/// Mirrors ``StreamingVADProvider`` in shape so a voice pipeline can gate
/// activation on either VAD or wake-word triggers.
public protocol WakeWordProvider: AnyObject {
    /// Expected input sample rate in Hz.
    var inputSampleRate: Int { get }
    /// Keywords the provider currently watches for.
    var registeredKeywords: [String] { get }
    /// Push a chunk of audio and receive any keyword detections that fired.
    func processAudio(_ samples: [Float]) throws -> [KeywordDetection]
    /// Reset streaming state between unrelated audio sources.
    func reset() throws
}

/// A ``WakeWordDetector`` holds the models; this session-wrapping adapter
/// gives the provider protocol a stable object to hand to a pipeline.
public final class WakeWordStreamingAdapter: WakeWordProvider {
    public let detector: WakeWordDetector
    private let session: WakeWordSession

    public init(detector: WakeWordDetector) throws {
        self.detector = detector
        self.session = try detector.createSession()
    }

    public var inputSampleRate: Int { detector.config.feature.sampleRate }

    public var registeredKeywords: [String] {
        detector.keywords.map { $0.phrase }
    }

    public func processAudio(_ samples: [Float]) throws -> [KeywordDetection] {
        try session.pushAudio(samples)
    }

    public func reset() throws {
        try session.reset()
    }
}
