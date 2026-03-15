import AVFoundation
import os

private let log = Logger(subsystem: "audio.soniqo.StreamingAudioPlayer", category: "Player")

/// Streams TTS audio chunks through an AVAudioPlayerNode attached to a shared AVAudioEngine.
///
/// Usage:
/// ```swift
/// let player = StreamingAudioPlayer()
/// player.attach(to: engine, format: playerFormat)
/// try engine.start()
/// player.startPlayback()  // must be after engine.start()
/// ```
///
/// Call `play(samples:sampleRate:)` to schedule audio chunks for playback.
/// Resamples automatically via `AVAudioConverter` when the TTS sample rate
/// differs from the engine output rate (e.g. 24kHz → 48kHz).
public final class StreamingAudioPlayer {
    public private(set) var isPlaying = false
    public var onPlaybackFinished: (() -> Void)?

    private var playerNode: AVAudioPlayerNode?
    private var outputFormat: AVAudioFormat?
    private var upsampler: AVAudioConverter?
    private var pendingBuffers = 0
    private let lock = NSLock()

    public init() {}

    /// Attach a player node to an existing engine. Call before `engine.start()`.
    ///
    /// Attaching before start avoids audio graph reconfiguration on a running engine.
    /// Call `startPlayback()` after the engine is started.
    public func attach(to engine: AVAudioEngine, format: AVAudioFormat) {
        let node = AVAudioPlayerNode()
        engine.attach(node)
        engine.connect(node, to: engine.mainMixerNode, format: format)
        self.playerNode = node
        self.outputFormat = format
        log.warning("attach: format=\(format.sampleRate)Hz \(format.channelCount)ch")
    }

    /// Begin playback — call after `AVAudioEngine.start()`.
    ///
    /// `AVAudioPlayerNode.play()` is a no-op when the engine isn't running,
    /// so this must be called after the engine has been started.
    public func startPlayback() {
        playerNode?.play()
        log.warning("startPlayback: node.isPlaying=\(self.playerNode?.isPlaying ?? false)")
    }

    /// Schedule TTS audio samples for playback. Resamples via AVAudioConverter if needed.
    public func play(samples: [Float], sampleRate: Int = 24000) throws {
        guard let playerNode, let outputFormat else {
            log.error("play: no playerNode or outputFormat — was attach() called?")
            return
        }
        // Log sample range to diagnose TTS output issues
        var minVal: Float = 0, maxVal: Float = 0
        if !samples.isEmpty {
            minVal = samples.min()!
            maxVal = samples.max()!
        }
        log.warning("play: \(samples.count) samples @ \(sampleRate)Hz, range=[\(minVal)...\(maxVal)], node.isPlaying=\(playerNode.isPlaying), outRate=\(outputFormat.sampleRate)")

        let outputRate = outputFormat.sampleRate
        let inputRate = Double(sampleRate)

        if abs(inputRate - outputRate) < 1.0 {
            scheduleBuffer(samples, format: outputFormat)
        } else {
            // Resample (e.g. 24kHz TTS → 48kHz engine)
            guard let inputFormat = AVAudioFormat(
                commonFormat: .pcmFormatFloat32,
                sampleRate: inputRate,
                channels: 1,
                interleaved: false
            ) else { return }

            if upsampler == nil || upsampler?.inputFormat.sampleRate != inputRate {
                upsampler = AVAudioConverter(from: inputFormat, to: outputFormat)
            }
            guard let converter = upsampler else { return }

            guard let inputBuffer = AVAudioPCMBuffer(
                pcmFormat: inputFormat,
                frameCapacity: AVAudioFrameCount(samples.count)
            ) else { return }
            inputBuffer.frameLength = AVAudioFrameCount(samples.count)
            memcpy(inputBuffer.floatChannelData![0], samples, samples.count * MemoryLayout<Float>.size)

            let ratio = outputRate / inputRate
            let outFrames = AVAudioFrameCount(Double(samples.count) * ratio)
            guard let outputBuffer = AVAudioPCMBuffer(
                pcmFormat: outputFormat,
                frameCapacity: outFrames
            ) else { return }

            var error: NSError?
            var consumed = false
            converter.convert(to: outputBuffer, error: &error) { _, outStatus in
                if consumed {
                    outStatus.pointee = .noDataNow
                    return nil
                }
                consumed = true
                outStatus.pointee = .haveData
                return inputBuffer
            }
            guard error == nil, outputBuffer.frameLength > 0 else { return }

            let resampled = Array(UnsafeBufferPointer(
                start: outputBuffer.floatChannelData![0],
                count: Int(outputBuffer.frameLength)))
            scheduleBuffer(resampled, format: outputFormat)
        }
    }

    private func scheduleBuffer(_ samples: [Float], format: AVAudioFormat) {
        guard let playerNode else { return }
        guard let buffer = AVAudioPCMBuffer(
            pcmFormat: format,
            frameCapacity: AVAudioFrameCount(samples.count)
        ) else { return }
        buffer.frameLength = AVAudioFrameCount(samples.count)
        memcpy(buffer.floatChannelData![0], samples, samples.count * MemoryLayout<Float>.size)

        lock.lock()
        pendingBuffers += 1
        isPlaying = true
        lock.unlock()

        log.warning("scheduleBuffer: \(samples.count) frames, pending=\(self.pendingBuffers)")
        playerNode.scheduleBuffer(buffer) { [weak self] in
            guard let self else { return }
            self.lock.lock()
            self.pendingBuffers -= 1
            let done = self.pendingBuffers <= 0
            self.lock.unlock()

            if done {
                DispatchQueue.main.async {
                    self.lock.lock()
                    let stillDone = self.pendingBuffers <= 0
                    self.lock.unlock()
                    guard stillDone else { return }
                    self.isPlaying = false
                    self.onPlaybackFinished?()
                }
            }
        }
    }

    /// Stop playback and clear scheduled buffers. Re-arms the node for new buffers.
    public func stop() {
        playerNode?.stop()
        playerNode?.play()
        lock.lock()
        pendingBuffers = 0
        lock.unlock()
        isPlaying = false
    }

    /// Fade out and stop (currently equivalent to `stop()`).
    public func fadeOutAndStop() {
        stop()
    }

    /// Detach from engine. Call when tearing down the pipeline.
    public func detach(from engine: AVAudioEngine) {
        playerNode?.stop()
        if let node = playerNode {
            engine.disconnectNodeOutput(node)
            engine.detach(node)
        }
        playerNode = nil
        outputFormat = nil
        upsampler = nil
        lock.lock()
        pendingBuffers = 0
        lock.unlock()
        isPlaying = false
    }
}
