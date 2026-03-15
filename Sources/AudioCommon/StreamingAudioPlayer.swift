#if canImport(AVFoundation)
import AVFoundation
import os

/// Streams audio chunks via AVAudioEngine for low-latency playback.
///
/// Manages its own AVAudioEngine and player node. Schedule audio chunks
/// as they arrive — the engine handles buffering and output.
///
/// - Important: Not safe for concurrent access. Use from a single task.
public final class StreamingAudioPlayer: @unchecked Sendable {
    private var engine: AVAudioEngine?
    private var playerNode: AVAudioPlayerNode?
    private var format: AVAudioFormat?
    private var pendingBuffers = 0
    private let lock = NSLock()

    public var isPlaying: Bool { playerNode?.isPlaying ?? false }

    /// Callback when all scheduled buffers finish playing.
    public var onPlaybackFinished: (() -> Void)?

    public init() {}

    // MARK: - Standalone mode (player manages its own engine)

    /// Start playback engine at the given sample rate.
    public func start(sampleRate: Double = 24000) throws {
        stop()
        let engine = AVAudioEngine()
        let node = AVAudioPlayerNode()
        guard let fmt = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: sampleRate,
            channels: 1,
            interleaved: false
        ) else { return }
        engine.attach(node)
        engine.connect(node, to: engine.mainMixerNode, format: fmt)
        try engine.start()
        node.play()
        self.engine = engine
        self.playerNode = node
        self.format = fmt
    }

    // MARK: - Attached mode (player uses an external engine)

    /// Attach to an existing AVAudioEngine (e.g. one that also has mic input).
    /// Call before `engine.start()`.
    public func attach(to engine: AVAudioEngine, format: AVAudioFormat) {
        let node = AVAudioPlayerNode()
        engine.attach(node)
        engine.connect(node, to: engine.mainMixerNode, format: format)
        self.playerNode = node
        self.format = format
    }

    /// Start the player node. Call after `engine.start()`.
    public func startPlayback() {
        playerNode?.play()
    }

    /// Detach from an external engine.
    public func detach(from engine: AVAudioEngine) {
        playerNode?.stop()
        if let node = playerNode {
            engine.disconnectNodeOutput(node)
            engine.detach(node)
        }
        playerNode = nil
        format = nil
    }

    // MARK: - Audio Scheduling

    /// Schedule a chunk of audio samples for playback.
    public func scheduleChunk(_ samples: [Float]) {
        guard let node = playerNode, let fmt = format, !samples.isEmpty else { return }
        guard let buffer = AVAudioPCMBuffer(pcmFormat: fmt, frameCapacity: AVAudioFrameCount(samples.count)) else { return }
        buffer.frameLength = AVAudioFrameCount(samples.count)
        samples.withUnsafeBufferPointer { ptr in
            buffer.floatChannelData![0].update(from: ptr.baseAddress!, count: samples.count)
        }
        lock.lock()
        pendingBuffers += 1
        lock.unlock()
        node.scheduleBuffer(buffer) { [weak self] in
            guard let self else { return }
            self.lock.lock()
            self.pendingBuffers -= 1
            let remaining = self.pendingBuffers
            self.lock.unlock()
            if remaining == 0 {
                self.onPlaybackFinished?()
            }
        }
    }

    /// Schedule samples with resampling from sourceSampleRate to the player's rate.
    public func play(samples: [Float], sampleRate: Int) throws {
        guard let fmt = format else { return }
        if Double(sampleRate) == fmt.sampleRate {
            scheduleChunk(samples)
        } else {
            // Resample using AVAudioConverter
            guard let srcFmt = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: Double(sampleRate), channels: 1, interleaved: false),
                  let converter = AVAudioConverter(from: srcFmt, to: fmt) else { return }
            guard let inputBuffer = AVAudioPCMBuffer(pcmFormat: srcFmt, frameCapacity: AVAudioFrameCount(samples.count)) else { return }
            inputBuffer.frameLength = AVAudioFrameCount(samples.count)
            samples.withUnsafeBufferPointer { ptr in
                inputBuffer.floatChannelData![0].update(from: ptr.baseAddress!, count: samples.count)
            }
            let outFrameCount = AVAudioFrameCount(Double(samples.count) * fmt.sampleRate / Double(sampleRate))
            guard let outputBuffer = AVAudioPCMBuffer(pcmFormat: fmt, frameCapacity: outFrameCount) else { return }
            var consumed = false
            var error: NSError?
            converter.convert(to: outputBuffer, error: &error) { _, outStatus in
                if consumed { outStatus.pointee = .noDataNow; return nil }
                consumed = true
                outStatus.pointee = .haveData
                return inputBuffer
            }
            let count = Int(outputBuffer.frameLength)
            guard count > 0, let data = outputBuffer.floatChannelData else { return }
            let resampled = Array(UnsafeBufferPointer(start: data[0], count: count))
            scheduleChunk(resampled)
        }
    }

    /// Wait until all scheduled buffers have finished playing.
    public func waitForCompletion() async {
        guard let node = playerNode, let fmt = format else { return }
        guard let silence = AVAudioPCMBuffer(pcmFormat: fmt, frameCapacity: 1) else { return }
        silence.frameLength = 1
        silence.floatChannelData![0][0] = 0
        await withCheckedContinuation { (cont: CheckedContinuation<Void, Never>) in
            node.scheduleBuffer(silence, completionCallbackType: .dataPlayedBack) { _ in
                cont.resume()
            }
        }
    }

    /// Fade out and stop immediately.
    public func fadeOutAndStop() {
        playerNode?.stop()
        lock.lock()
        pendingBuffers = 0
        lock.unlock()
    }

    /// Stop and release resources.
    public func stop() {
        playerNode?.stop()
        engine?.stop()
        engine = nil
        playerNode = nil
        format = nil
        lock.lock()
        pendingBuffers = 0
        lock.unlock()
    }
}
#endif
