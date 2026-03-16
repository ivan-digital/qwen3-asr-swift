#if os(macOS)
import AVFoundation
import Foundation
import Observation
import ParakeetASR
import Qwen3TTS
import SpeechCore
import SpeechVAD
import AudioCommon

@Observable
@MainActor
final class EchoViewModel {
    var isLoading = false
    var loadingStatus = ""
    var errorMessage: String?
    var isRunning = false
    var pipelineState: String = "idle"
    var log: [String] = []

    private var vad: SileroVADModel?
    private var asr: ParakeetASRModel?
    private var tts: Qwen3TTSModel?
    private var pipeline: VoicePipeline?
    private var audioEngine: AVAudioEngine?
    private let player = StreamingAudioPlayer()
    private var debugRecordBuffer: [Float] = []
    private var debugTTSBuffer: [Float] = []
    private var isRecordingDebug = false

    var modelsLoaded: Bool { vad != nil && asr != nil && tts != nil }
    var isPlaying: Bool { player.isPlaying }

    func loadModels() async {
        isLoading = true
        errorMessage = nil
        log = []

        do {
            loadingStatus = "Loading VAD (CoreML)..."
            appendLog("Loading Silero VAD (CoreML)...")
            vad = try await Task.detached {
                try await SileroVADModel.fromPretrained(engine: .coreml)
            }.value

            loadingStatus = "Loading ASR (Parakeet CoreML)..."
            appendLog("Loading Parakeet TDT (CoreML)...")
            asr = try await Task.detached {
                let model = try await ParakeetASRModel.fromPretrained()
                try model.warmUp()
                return model
            }.value

            loadingStatus = "Loading TTS (Qwen3 Base)..."
            appendLog("Loading Qwen3-TTS (Base)...")
            tts = try await Task.detached {
                try await Qwen3TTSModel.fromPretrained(
                    modelId: TTSModelVariant.base.rawValue)
            }.value

            appendLog("All models loaded.")
            loadingStatus = ""
        } catch {
            errorMessage = "Load failed: \(error.localizedDescription)"
            appendLog("ERROR: \(error.localizedDescription)")
        }

        isLoading = false
    }

    func startPipeline() {
        guard let vad, let asr, let tts else { return }
        guard !isRunning else { return }

        var config = PipelineConfig()
        config.mode = .echo
        config.allowInterruptions = true
        config.minSilenceDuration = 0.6
        config.maxResponseDuration = 5.0

        pipeline = VoicePipeline(
            stt: asr,
            tts: tts,
            vad: vad,
            config: config,
            onEvent: { [weak self] event in
                DispatchQueue.main.async {
                    self?.handleEvent(event)
                }
            }
        )

        player.onPlaybackFinished = { [weak self] in
            self?.playbackDidFinish()
        }
        pipeline?.start()
        isRunning = true
        isRecordingDebug = true
        debugRecordBuffer = []
        pipelineState = "listening"
        appendLog("Pipeline started — speak into the mic...")
        startMicrophone()
    }

    func stopPipeline() {
        isRecordingDebug = false
        stopMicrophone()
        pipeline?.stop()
        pipeline = nil
        isRunning = false
        pipelineState = "idle"
        saveDebugRecording()
        appendLog("Pipeline stopped.")
    }

    private func saveDebugRecording() {
        let tmpDir = FileManager.default.temporaryDirectory
        if !debugRecordBuffer.isEmpty {
            let url = tmpDir.appendingPathComponent("echo_debug_mic.wav")
            do {
                try WAVWriter.write(samples: debugRecordBuffer, sampleRate: 16000, to: url)
                appendLog("[Debug] Saved mic: \(url.path) (\(String(format: "%.1f", Double(debugRecordBuffer.count) / 16000.0))s)")
            } catch {
                appendLog("[Debug] Failed to save mic: \(error)")
            }
            debugRecordBuffer = []
        }
        if !debugTTSBuffer.isEmpty {
            let url = tmpDir.appendingPathComponent("echo_debug_tts.wav")
            do {
                try WAVWriter.write(samples: debugTTSBuffer, sampleRate: 24000, to: url)
                appendLog("[Debug] Saved TTS: \(url.path) (\(String(format: "%.1f", Double(debugTTSBuffer.count) / 24000.0))s)")
            } catch {
                appendLog("[Debug] Failed to save TTS: \(error)")
            }
            debugTTSBuffer = []
        }
    }

    // MARK: - Event Handling

    private func handleEvent(_ event: PipelineEvent) {
        switch event {
        case .sessionCreated:
            break
        case .speechStarted:
            pipelineState = "speech detected"
            appendLog("[VAD] Speech started")
        case .speechEnded:
            pipelineState = "transcribing..."
            appendLog("[VAD] Speech ended")
        case .transcriptionCompleted(let text, let language, _):
            pipelineState = "synthesizing..."
            let langTag = language.map { " [\($0)]" } ?? ""
            appendLog("[STT\(langTag)] \(text)")
        case .responseCreated:
            pipelineState = "speaking..."
            player.resetGeneration()
        case .responseInterrupted:
            // User interrupted — fade out immediately
            player.fadeOutAndStop()
            pipelineState = "interrupted → listening"
            appendLog("[Interrupted] Stopping playback")
        case .responseAudioDelta(let samples):
            appendLog("[TTS chunk] \(samples.count) samples (\(String(format: "%.1f", Double(samples.count) / 24000.0))s)")
            if isRecordingDebug { debugTTSBuffer.append(contentsOf: samples) }
            do {
                try player.play(samples: samples, sampleRate: 24000)
            } catch {
                appendLog("[ERROR] Playback: \(error.localizedDescription)")
            }
        case .responseDone:
            appendLog("[Pipeline] Response done")
            player.markGenerationComplete()
        case .error(let msg):
            pipelineState = "error"
            appendLog("[ERROR] \(msg)")
            pipeline?.resumeListening()
        }
    }

    func playbackDidFinish() {
        resumeAfterResponse()
    }

    private func resumeAfterResponse() {
        guard isRunning else { return }
        // Non-blocking — post-playback guard is handled inside the turn
        // detector by suppressing VAD events for the configured duration.
        pipeline?.resumeListening()
        pipelineState = "listening"
        appendLog("Listening...")
    }

    // MARK: - Microphone (Full-duplex with AEC)

    private var aecEnabled = false

    private func startMicrophone() {
        let engine = AVAudioEngine()
        let inputNode = engine.inputNode

        // Enable Apple's AEC BEFORE starting engine and BEFORE reading format.
        // This removes the agent's speaker output from the mic signal,
        // enabling full-duplex: user can interrupt while agent speaks.
        do {
            try inputNode.setVoiceProcessingEnabled(true)
            aecEnabled = true
            appendLog("[AEC] Voice processing enabled")
        } catch {
            aecEnabled = false
            appendLog("[AEC] Failed: \(error.localizedDescription)")
        }

        // Voice processing changes the format — must read AFTER enabling.
        let hwFormat = inputNode.outputFormat(forBus: 0)
        appendLog("[Mic] Input format: \(Int(hwFormat.sampleRate))Hz, \(hwFormat.channelCount)ch, \(hwFormat.commonFormat.rawValue)")

        // If VP gave us an invalid format, fall back
        if hwFormat.sampleRate < 1 || hwFormat.channelCount < 1 {
            appendLog("[Mic] Invalid format — disabling voice processing")
            try? inputNode.setVoiceProcessingEnabled(false)
            aecEnabled = false
        }

        // VP produces multi-channel (e.g., 9ch). Extract ch0 then resample to 16kHz.
        let vpFormat = inputNode.outputFormat(forBus: 0)
        let vpRate = vpFormat.sampleRate
        appendLog("[Mic] Tap format: \(Int(vpRate))Hz, \(vpFormat.channelCount)ch")

        // Mono intermediate format at VP sample rate
        guard let monoFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: vpRate,
            channels: 1,
            interleaved: false
        ) else { return }

        guard let targetFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: 16000,
            channels: 1,
            interleaved: false
        ) else { return }

        guard let resampler = AVAudioConverter(from: monoFormat, to: targetFormat) else {
            appendLog("[Mic] Cannot create resampler")
            return
        }

        var sampleCounter = 0
        inputNode.installTap(onBus: 0, bufferSize: 1024, format: vpFormat) { [weak self] buffer, _ in
            guard let self else { return }
            guard let srcData = buffer.floatChannelData else { return }
            let frameLen = Int(buffer.frameLength)
            guard frameLen > 0 else { return }

            // Extract channel 0 (echo-cancelled signal) into a mono buffer
            guard let monoBuffer = AVAudioPCMBuffer(pcmFormat: monoFormat, frameCapacity: buffer.frameCapacity) else { return }
            monoBuffer.frameLength = buffer.frameLength
            memcpy(monoBuffer.floatChannelData![0], srcData[0], frameLen * MemoryLayout<Float>.size)

            // Resample mono to 16kHz
            let outFrameCount = AVAudioFrameCount(Double(frameLen) * 16000.0 / vpRate)
            guard outFrameCount > 0, let outBuffer = AVAudioPCMBuffer(pcmFormat: targetFormat, frameCapacity: outFrameCount) else { return }

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

            sampleCounter += 1
            if sampleCounter <= 3 {
                let rms = sqrt(samples.map { $0 * $0 }.reduce(0, +) / Float(count))
                DispatchQueue.main.async {
                    self.appendLog("[Mic] Buffer #\(sampleCounter): \(count) samples, RMS=\(String(format: "%.6f", rms))")
                }
            }

            // Record for debug
            if self.isRecordingDebug {
                self.debugRecordBuffer.append(contentsOf: samples)
            }

            self.pipeline?.pushAudio(samples)
        }

        do {
            try engine.start()
            audioEngine = engine
            appendLog("[Mic] Engine started")

            // Attach TTS player AFTER engine starts — VP constrains the graph.
            // Use mono format at engine sample rate — AVAudioEngine auto-upmixes
            // mono→stereo at the mixer. This avoids one-channel-only playback.
            let mixerFormat = engine.mainMixerNode.outputFormat(forBus: 0)
            let engineRate = mixerFormat.sampleRate
            guard let playerFormat = AVAudioFormat(
                commonFormat: .pcmFormatFloat32,
                sampleRate: engineRate,
                channels: 1,
                interleaved: false
            ) else { return }
            appendLog("[Player] Mixer: \(Int(engineRate))Hz, \(mixerFormat.channelCount)ch → Player: mono \(Int(engineRate))Hz")
            player.attach(to: engine, format: playerFormat)
            player.startPlayback()
            appendLog("[Player] Attached to shared engine")
        } catch {
            appendLog("[Mic] Engine error: \(error.localizedDescription)")
        }
    }

    private func stopMicrophone() {
        audioEngine?.inputNode.removeTap(onBus: 0)
        if let engine = audioEngine {
            player.detach(from: engine)
            engine.stop()
        }
        audioEngine = nil
    }

    private static let logFileURL: URL = {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("echo_debug.log")
        try? "".write(to: url, atomically: true, encoding: .utf8)
        return url
    }()

    private func appendLog(_ message: String) {
        log.append(message)
        if log.count > 50 {
            log.removeFirst(log.count - 50)
        }
        let line = "[\(Date())] \(message)\n"
        if let handle = try? FileHandle(forWritingTo: Self.logFileURL) {
            handle.seekToEndOfFile()
            handle.write(line.data(using: .utf8)!)
            handle.closeFile()
        }
    }
}

#endif
