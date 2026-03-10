import AVFoundation
import Observation

/// Streams TTS audio chunks through an externally-provided AVAudioPlayerNode.
/// Designed for use with a shared AVAudioEngine that also handles mic input + AEC.
@Observable
final class AudioPlayer {
    private(set) var isPlaying = false
    var onPlaybackFinished: (() -> Void)?

    private var playerNode: AVAudioPlayerNode?
    private var outputFormat: AVAudioFormat?
    private var upsampler: AVAudioConverter?
    private var pendingBuffers = 0
    private let lock = NSLock()

    /// Attach to an existing engine. Call once before play().
    /// - Parameters:
    ///   - engine: The shared AVAudioEngine (with VP/AEC on input)
    ///   - format: Output format matching engine sample rate (e.g. 48kHz mono)
    func attach(to engine: AVAudioEngine, format: AVAudioFormat) {
        let node = AVAudioPlayerNode()
        engine.attach(node)
        engine.connect(node, to: engine.mainMixerNode, format: format)
        self.playerNode = node
        self.outputFormat = format
        node.play()
    }

    /// Schedule TTS audio samples for playback. Resamples if needed.
    func play(samples: [Float], sampleRate: Int = 24000) throws {
        guard let playerNode, let outputFormat else { return }

        let outputRate = outputFormat.sampleRate
        let inputRate = Double(sampleRate)

        if abs(inputRate - outputRate) < 1.0 {
            // Same rate — schedule directly
            scheduleBuffer(samples, format: outputFormat)
        } else {
            // Resample (e.g. 24kHz TTS → 48kHz engine)
            guard let inputFormat = AVAudioFormat(
                commonFormat: .pcmFormatFloat32,
                sampleRate: inputRate,
                channels: 1,
                interleaved: false
            ) else { return }

            // Create or reuse converter
            if upsampler == nil || upsampler?.inputFormat.sampleRate != inputRate {
                upsampler = AVAudioConverter(from: inputFormat, to: outputFormat)
            }
            guard let converter = upsampler else { return }

            guard let inputBuffer = AVAudioPCMBuffer(pcmFormat: inputFormat, frameCapacity: AVAudioFrameCount(samples.count)) else { return }
            inputBuffer.frameLength = AVAudioFrameCount(samples.count)
            memcpy(inputBuffer.floatChannelData![0], samples, samples.count * MemoryLayout<Float>.size)

            let ratio = outputRate / inputRate
            let outFrames = AVAudioFrameCount(Double(samples.count) * ratio)
            guard let outputBuffer = AVAudioPCMBuffer(pcmFormat: outputFormat, frameCapacity: outFrames) else { return }

            var error: NSError?
            converter.convert(to: outputBuffer, error: &error) { _, outStatus in
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
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(samples.count)) else { return }
        buffer.frameLength = AVAudioFrameCount(samples.count)
        memcpy(buffer.floatChannelData![0], samples, samples.count * MemoryLayout<Float>.size)

        lock.lock()
        pendingBuffers += 1
        isPlaying = true
        lock.unlock()

        playerNode.scheduleBuffer(buffer) { [weak self] in
            guard let self else { return }
            self.lock.lock()
            self.pendingBuffers -= 1
            let done = self.pendingBuffers <= 0
            self.lock.unlock()

            if done {
                DispatchQueue.main.async {
                    // Re-check: new chunks may have been scheduled between the
                    // completion handler and this main-queue dispatch.
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

    func stop() {
        playerNode?.stop()
        playerNode?.play()  // re-arm for next schedule
        lock.lock()
        pendingBuffers = 0
        lock.unlock()
        isPlaying = false
    }

    func fadeOutAndStop() {
        stop()
    }

    /// Detach from engine (call on pipeline stop).
    func detach(from engine: AVAudioEngine) {
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
