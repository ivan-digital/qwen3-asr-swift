import AVFoundation

/// Streams audio chunks via AVAudioEngine for low-latency playback.
final class StreamingAudioPlayer {
    private var engine: AVAudioEngine?
    private var playerNode: AVAudioPlayerNode?
    private var format: AVAudioFormat?

    var isPlaying: Bool { playerNode?.isPlaying ?? false }

    func start(sampleRate: Double = 24000) throws {
        stop()
        let engine = AVAudioEngine()
        let node = AVAudioPlayerNode()
        let fmt = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: sampleRate,
            channels: 1,
            interleaved: false
        )!
        engine.attach(node)
        engine.connect(node, to: engine.mainMixerNode, format: fmt)
        try engine.start()
        node.play()
        self.engine = engine
        self.playerNode = node
        self.format = fmt
    }

    func scheduleChunk(_ samples: [Float]) {
        guard let node = playerNode, let fmt = format, !samples.isEmpty else { return }
        let buffer = AVAudioPCMBuffer(pcmFormat: fmt, frameCapacity: AVAudioFrameCount(samples.count))!
        buffer.frameLength = AVAudioFrameCount(samples.count)
        samples.withUnsafeBufferPointer { ptr in
            buffer.floatChannelData![0].update(from: ptr.baseAddress!, count: samples.count)
        }
        node.scheduleBuffer(buffer)
    }

    /// Waits until all scheduled buffers have finished playing.
    func waitForCompletion() async {
        guard let node = playerNode, let fmt = format else { return }
        let silence = AVAudioPCMBuffer(pcmFormat: fmt, frameCapacity: 1)!
        silence.frameLength = 1
        silence.floatChannelData![0][0] = 0
        await withCheckedContinuation { (cont: CheckedContinuation<Void, Never>) in
            node.scheduleBuffer(silence, completionCallbackType: .dataPlayedBack) { _ in
                cont.resume()
            }
        }
    }

    func stop() {
        playerNode?.stop()
        engine?.stop()
        engine = nil
        playerNode = nil
        format = nil
    }
}
