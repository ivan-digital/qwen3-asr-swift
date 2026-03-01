import AVFoundation

@Observable
final class AudioRecorder {
    private(set) var isRecording = false
    private(set) var audioLevel: Float = 0

    /// Called (on main thread) when speech is detected followed by sustained silence.
    var onSilenceAfterSpeech: (() -> Void)?

    /// RMS threshold below which audio is considered silence.
    var silenceThreshold: Float = 0.015

    /// Seconds of continuous silence after speech before firing callback.
    var silenceDuration: TimeInterval = 1.5

    private var audioEngine: AVAudioEngine?
    private var samples: [Float] = []
    private let lock = NSLock()
    private let targetSampleRate: Double

    // Silence detection state
    private var speechDetected = false
    private var silenceStartTime: Date?
    private var silenceCallbackFired = false

    init(targetSampleRate: Double = 24000) {
        self.targetSampleRate = targetSampleRate
    }

    func startRecording() {
        lock.lock()
        samples.removeAll()
        lock.unlock()

        speechDetected = false
        silenceStartTime = nil
        silenceCallbackFired = false

        let engine = AVAudioEngine()
        let inputNode = engine.inputNode
        let hwFormat = inputNode.outputFormat(forBus: 0)

        guard let targetFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: targetSampleRate,
            channels: 1,
            interleaved: false
        ) else { return }

        guard let converter = AVAudioConverter(from: hwFormat, to: targetFormat) else { return }

        inputNode.installTap(onBus: 0, bufferSize: 1024, format: hwFormat) { [weak self] buffer, _ in
            guard let self else { return }
            let frameCount = AVAudioFrameCount(
                Double(buffer.frameLength) * self.targetSampleRate / hwFormat.sampleRate
            )
            guard frameCount > 0,
                  let converted = AVAudioPCMBuffer(pcmFormat: targetFormat, frameCapacity: frameCount)
            else { return }

            var error: NSError?
            converter.convert(to: converted, error: &error) { _, outStatus in
                outStatus.pointee = .haveData
                return buffer
            }
            if error != nil { return }

            if let channelData = converted.floatChannelData?[0] {
                let count = Int(converted.frameLength)
                let ptr = UnsafeBufferPointer(start: channelData, count: count)
                let chunk = Array(ptr)

                // RMS level
                var sum: Float = 0
                for s in chunk { sum += s * s }
                let rms = sqrt(sum / max(Float(count), 1))

                self.lock.lock()
                self.samples.append(contentsOf: chunk)
                self.lock.unlock()

                // Silence detection
                let speechThreshold = self.silenceThreshold * 2
                if rms > speechThreshold {
                    if !self.speechDetected {
                        print("[AudioRecorder] Speech detected (rms=\(String(format: "%.4f", rms)))")
                    }
                    self.speechDetected = true
                    self.silenceStartTime = nil
                } else if self.speechDetected && rms < self.silenceThreshold {
                    if self.silenceStartTime == nil {
                        self.silenceStartTime = Date()
                        print("[AudioRecorder] Silence started (rms=\(String(format: "%.4f", rms)))")
                    } else if !self.silenceCallbackFired,
                              Date().timeIntervalSince(self.silenceStartTime!) >= self.silenceDuration {
                        self.silenceCallbackFired = true
                        print("[AudioRecorder] Silence callback fired after \(self.silenceDuration)s")
                        DispatchQueue.main.async {
                            self.onSilenceAfterSpeech?()
                        }
                    }
                }

                DispatchQueue.main.async {
                    self.audioLevel = rms
                }
            }
        }

        do {
            try engine.start()
            audioEngine = engine
            isRecording = true
        } catch {
            inputNode.removeTap(onBus: 0)
        }
    }

    func stopRecording() -> [Float] {
        audioEngine?.inputNode.removeTap(onBus: 0)
        audioEngine?.stop()
        audioEngine = nil
        isRecording = false
        audioLevel = 0

        lock.lock()
        let result = samples
        samples.removeAll()
        lock.unlock()
        return result
    }
}
