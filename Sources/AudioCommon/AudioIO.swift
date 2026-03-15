#if canImport(AVFoundation)
import AVFoundation
import os

/// Reusable audio I/O manager — handles mic capture, resampling, and playback.
///
/// Eliminates AVAudioEngine boilerplate that every demo app reimplements.
///
/// ```swift
/// let audio = AudioIO()
/// try audio.startMicrophone(targetSampleRate: 16000) { samples in
///     pipeline.pushAudio(samples)
/// }
/// audio.player.scheduleChunk(ttsOutput)
/// audio.stopMicrophone()
/// ```
public final class AudioIO {
    /// Microphone state.
    public enum MicrophoneState: Sendable {
        case stopped, running, error(String)
    }

    /// Audio player for TTS output. Attached to the engine when mic starts.
    public let player = StreamingAudioPlayer()

    /// Current microphone state.
    public private(set) var microphoneState: MicrophoneState = .stopped

    /// RMS audio level (0.0–1.0) for UI meters. Updated on each mic buffer.
    public private(set) var audioLevel: Float = 0

    /// Whether to enable Voice Processing I/O for echo cancellation.
    public let enableAEC: Bool

    /// Playback sample rate (for TTS output).
    public let playbackSampleRate: Double

    private var engine: AVAudioEngine?
    private static let log = Logger(subsystem: "audio.soniqo", category: "AudioIO")

    public init(enableAEC: Bool = false, playbackSampleRate: Double = 24000) {
        self.enableAEC = enableAEC
        self.playbackSampleRate = playbackSampleRate
    }

    /// Start microphone capture, resampled to targetSampleRate.
    ///
    /// Also attaches the player to the engine for simultaneous playback.
    /// Call `player.scheduleChunk()` to play audio while recording.
    ///
    /// - Parameters:
    ///   - targetSampleRate: Output sample rate for onSamples (default 16kHz for VAD/ASR)
    ///   - onSamples: Callback with resampled mono Float32 samples (called on audio thread)
    public func startMicrophone(
        targetSampleRate: Int = 16000,
        onSamples: @escaping ([Float]) -> Void
    ) throws {
        stopMicrophone()

        #if os(iOS)
        let session = AVAudioSession.sharedInstance()
        if enableAEC {
            try session.setCategory(.playAndRecord, mode: .voiceChat, options: [.defaultToSpeaker, .allowBluetooth])
        } else {
            try session.setCategory(.playAndRecord, mode: .default, options: [.defaultToSpeaker, .allowBluetooth])
        }
        try session.setActive(true)
        #endif

        let engine = AVAudioEngine()
        let inputNode = engine.inputNode
        let hwFormat = inputNode.outputFormat(forBus: 0)

        // Mono intermediate at hardware rate
        guard let monoFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: hwFormat.sampleRate,
            channels: 1,
            interleaved: false
        ) else {
            microphoneState = .error("Cannot create mono format")
            return
        }

        // Target format for VAD/ASR
        guard let targetFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: Double(targetSampleRate),
            channels: 1,
            interleaved: false
        ) else {
            microphoneState = .error("Cannot create target format")
            return
        }

        guard let resampler = AVAudioConverter(from: monoFormat, to: targetFormat) else {
            microphoneState = .error("Cannot create resampler")
            return
        }

        inputNode.installTap(onBus: 0, bufferSize: 1024, format: hwFormat) { [weak self] buffer, _ in
            guard let self else { return }
            guard let srcData = buffer.floatChannelData else { return }
            let frameLen = Int(buffer.frameLength)
            guard frameLen > 0 else { return }

            // Extract channel 0 into mono buffer
            guard let monoBuffer = AVAudioPCMBuffer(pcmFormat: monoFormat, frameCapacity: buffer.frameCapacity) else { return }
            monoBuffer.frameLength = buffer.frameLength
            memcpy(monoBuffer.floatChannelData![0], srcData[0], frameLen * MemoryLayout<Float>.size)

            // Resample
            let outFrameCount = AVAudioFrameCount(Double(frameLen) * Double(targetSampleRate) / hwFormat.sampleRate)
            guard outFrameCount > 0,
                  let outBuffer = AVAudioPCMBuffer(pcmFormat: targetFormat, frameCapacity: outFrameCount) else { return }

            var error: NSError?
            resampler.convert(to: outBuffer, error: &error) { _, outStatus in
                outStatus.pointee = .haveData
                return monoBuffer
            }
            if error != nil { return }

            guard let outData = outBuffer.floatChannelData else { return }
            let count = Int(outBuffer.frameLength)
            guard count > 0 else { return }
            let samples = Array(UnsafeBufferPointer(start: outData[0], count: count))

            // RMS for audio level
            var sum: Float = 0
            for s in samples { sum += s * s }
            self.audioLevel = sqrt(sum / max(Float(count), 1))

            onSamples(samples)
        }

        // Attach player for TTS output
        guard let playerFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: playbackSampleRate,
            channels: 1,
            interleaved: false
        ) else { return }
        player.attach(to: engine, format: playerFormat)

        do {
            try engine.start()
            player.startPlayback()
            self.engine = engine
            microphoneState = .running
            Self.log.info("Microphone started at \(targetSampleRate)Hz, player at \(self.playbackSampleRate)Hz")
        } catch {
            microphoneState = .error(error.localizedDescription)
            throw error
        }
    }

    /// Stop microphone capture and detach player.
    public func stopMicrophone() {
        if let engine {
            engine.inputNode.removeTap(onBus: 0)
            player.detach(from: engine)
            engine.stop()
        }
        engine = nil
        audioLevel = 0
        microphoneState = .stopped
    }

    deinit {
        stopMicrophone()
    }
}
#endif
