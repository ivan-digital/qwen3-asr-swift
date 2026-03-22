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
    private var generationComplete = false
    private var isFirstBuffer = true
    private let lock = NSLock()

    /// Engine owned by this player (standalone mode).
    private var ownedEngine: AVAudioEngine?

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

    /// Create a standalone engine for playback (no mic, no AEC).
    /// Used by Speak tab where no shared engine exists.
    func ensureStandaloneEngine() {
        guard playerNode == nil else { return }
        let engine = AVAudioEngine()
        let node = AVAudioPlayerNode()
        let mixerFormat = engine.mainMixerNode.outputFormat(forBus: 0)
        guard let monoFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: mixerFormat.sampleRate,
            channels: 1,
            interleaved: false
        ) else { return }

        engine.attach(node)
        engine.connect(node, to: engine.mainMixerNode, format: monoFormat)

        do {
            try engine.start()
            node.play()
            self.ownedEngine = engine
            self.playerNode = node
            self.outputFormat = monoFormat
        } catch {
            // Engine failed to start — playback won't work
        }
    }

    /// Schedule TTS audio samples for playback. Resamples if needed.
    func play(samples: [Float], sampleRate: Int = 24000) throws {
        guard playerNode != nil, let outputFormat else { return }

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

        var output = samples

        // Drop near-silent warmup chunks at the start of a generation cycle.
        // TTS codec decoders emit multiple low-energy chunks before real speech.
        if isFirstBuffer && !samples.isEmpty {
            var sumSq: Float = 0
            for s in samples { sumSq += s * s }
            let rms = sqrt(sumSq / Float(samples.count))
            if rms < 0.03 {
                return  // Still warmup — drop and keep isFirstBuffer=true
            }

            // First real audio chunk — apply fade-in to prevent pop
            isFirstBuffer = false
            let fadeFrames = min(samples.count, Int(format.sampleRate * 0.005))  // 5ms
            for i in 0..<fadeFrames {
                output[i] *= Float(i) / Float(fadeFrames)
            }
        }

        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(output.count)) else { return }
        buffer.frameLength = AVAudioFrameCount(output.count)
        memcpy(buffer.floatChannelData![0], output, output.count * MemoryLayout<Float>.size)

        lock.lock()
        pendingBuffers += 1
        isPlaying = true
        lock.unlock()

        // Use .dataPlayedBack to ensure the callback fires only after the audio
        // has actually been output through the speakers, not just consumed by the
        // render thread. This prevents cutting off the last buffer.
        playerNode.scheduleBuffer(buffer, completionCallbackType: .dataPlayedBack) { [weak self] _ in
            guard let self else { return }
            self.lock.lock()
            self.pendingBuffers -= 1
            let done = self.pendingBuffers <= 0 && self.generationComplete
            self.lock.unlock()

            if done {
                DispatchQueue.main.async {
                    self.lock.lock()
                    let stillDone = self.pendingBuffers <= 0 && self.generationComplete
                    self.lock.unlock()
                    guard stillDone else { return }
                    self.isPlaying = false
                    self.onPlaybackFinished?()
                }
            }
        }
    }

    /// Signal that TTS generation is complete — no more chunks will arrive.
    /// Fires onPlaybackFinished if all buffers already drained.
    func markGenerationComplete() {
        lock.lock()
        generationComplete = true
        let done = pendingBuffers <= 0
        lock.unlock()

        if done {
            DispatchQueue.main.async {
                self.lock.lock()
                let stillDone = self.pendingBuffers <= 0 && self.generationComplete
                self.lock.unlock()
                guard stillDone else { return }
                self.isPlaying = false
                self.onPlaybackFinished?()
            }
        }
    }

    /// Reset for a new generation cycle (call before first chunk).
    func resetGeneration() {
        lock.lock()
        generationComplete = false
        isFirstBuffer = true
        lock.unlock()
    }

    func stop() {
        playerNode?.stop()
        playerNode?.play()  // re-arm for next schedule
        lock.lock()
        pendingBuffers = 0
        generationComplete = false
        isFirstBuffer = true
        lock.unlock()
        isPlaying = false
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
        ownedEngine?.stop()
        ownedEngine = nil
        lock.lock()
        pendingBuffers = 0
        lock.unlock()
        isPlaying = false
    }
}
