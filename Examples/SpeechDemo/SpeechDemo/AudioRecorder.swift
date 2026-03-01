import AVFoundation
import Observation

@Observable
final class AudioRecorder {
    private(set) var isRecording = false
    private(set) var audioLevel: Float = 0

    private var audioEngine: AVAudioEngine?
    private var samples: [Float] = []
    private let lock = NSLock()

    func startRecording() {
        samples = []
        audioLevel = 0

        let engine = AVAudioEngine()
        let inputNode = engine.inputNode
        let hwFormat = inputNode.outputFormat(forBus: 0)

        // Target 16kHz mono
        guard let targetFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: 16000,
            channels: 1,
            interleaved: false
        ) else { return }

        // Install converter if sample rates differ
        guard let converter = AVAudioConverter(from: hwFormat, to: targetFormat) else { return }

        inputNode.installTap(onBus: 0, bufferSize: 1024, format: hwFormat) { [weak self] buffer, _ in
            guard let self else { return }

            let frameCount = AVAudioFrameCount(
                Double(buffer.frameLength) * 16000.0 / hwFormat.sampleRate
            )
            guard let convertedBuffer = AVAudioPCMBuffer(
                pcmFormat: targetFormat,
                frameCapacity: frameCount
            ) else { return }

            var error: NSError?
            converter.convert(to: convertedBuffer, error: &error) { _, outStatus in
                outStatus.pointee = .haveData
                return buffer
            }
            if error != nil { return }

            guard let channelData = convertedBuffer.floatChannelData else { return }
            let count = Int(convertedBuffer.frameLength)
            let data = Array(UnsafeBufferPointer(start: channelData[0], count: count))

            // RMS level
            let rms = sqrt(data.reduce(0) { $0 + $1 * $1 } / max(Float(count), 1))

            self.lock.lock()
            self.samples.append(contentsOf: data)
            self.lock.unlock()

            DispatchQueue.main.async {
                self.audioLevel = rms
            }
        }

        do {
            try engine.start()
            audioEngine = engine
            isRecording = true
        } catch {
            print("Failed to start audio engine: \(error)")
        }
    }

    func stopRecording() -> [Float] {
        audioEngine?.inputNode.removeTap(onBus: 0)
        audioEngine?.stop()
        audioEngine = nil
        isRecording = false
        audioLevel = 0

        lock.lock()
        let captured = samples
        samples = []
        lock.unlock()

        return captured
    }
}
