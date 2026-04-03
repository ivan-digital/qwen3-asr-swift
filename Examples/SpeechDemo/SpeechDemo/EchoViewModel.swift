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
    var lastTranscription: String = ""
    var lastLanguage: String = ""
    var log: [String] = []
    var vadLevel: Float = 0

    private var vad: SileroVADModel?
    private var asr: ParakeetASRModel?
    private var tts: Qwen3TTSModel?
    private var pipeline: VoicePipeline?
    private var audioEngine: AVAudioEngine?
    private let player = StreamingAudioPlayer()
    private var debugMicBuffer: [Float] = []
    private var debugTTSBuffer: [Float] = []
    private var speechStartTime: Date?

    var modelsLoaded: Bool { vad != nil && asr != nil && tts != nil }

    // MARK: - Model Loading

    func loadModels() async {
        isLoading = true
        errorMessage = nil
        log = []

        do {
            loadingStatus = "Loading VAD (CoreML)..."
            vad = try await Task.detached {
                try await SileroVADModel.fromPretrained(engine: .coreml)
            }.value

            loadingStatus = "Loading ASR (Parakeet CoreML)..."
            asr = try await Task.detached {
                let model = try await ParakeetASRModel.fromPretrained()
                try model.warmUp()
                return model
            }.value

            loadingStatus = "Loading TTS (Qwen3 Base)..."
            tts = try await Task.detached {
                try await Qwen3TTSModel.fromPretrained(
                    modelId: TTSModelVariant.base.rawValue)
            }.value

            appendLog("All models loaded.")
            loadingStatus = ""
        } catch {
            errorMessage = "Load failed: \(error.localizedDescription)"
        }
        isLoading = false
    }

    // MARK: - Pipeline Lifecycle

    func startPipeline() {
        guard let vad, let asr, let tts else { return }
        guard !isRunning else { return }

        var config = PipelineConfig()
        config.mode = .echo
        config.allowInterruptions = false
        config.minSilenceDuration = 0.6
        config.eagerSTT = false
        config.maxResponseDuration = 15.0

        pipeline = VoicePipeline(
            stt: asr, tts: tts, vad: vad, config: config,
            onEvent: { [weak self] event in
                DispatchQueue.main.async { self?.handleEvent(event) }
            }
        )

        player.onPlaybackFinished = { [weak self] in
            guard let self, self.isRunning else { return }
            self.pipeline?.resumeListening()
            self.pipelineState = "listening"
            self.appendLog("Listening...")
        }

        pipeline?.start()
        isRunning = true
        debugMicBuffer = []
        debugTTSBuffer = []
        pipelineState = "listening"
        appendLog("Pipeline started — speak into the mic...")
        startMicrophone()
    }

    func stopPipeline() {
        stopMicrophone()
        pipeline?.stop()
        pipeline = nil
        isRunning = false
        pipelineState = "idle"
        saveDebugFiles()
        appendLog("Pipeline stopped.")
    }

    // MARK: - Event Handling

    private func handleEvent(_ event: PipelineEvent) {
        switch event {
        case .sessionCreated:
            break
        case .speechStarted:
            pipelineState = "speech detected"
            speechStartTime = Date()
            appendLog("[VAD] Speech started")
        case .speechEnded:
            pipelineState = "transcribing..."
            let duration = Date().timeIntervalSince(speechStartTime ?? Date())
            if duration > 13 {
                appendLog("[VAD] Speech ended (\(String(format: "%.0f", duration))s — max duration may have cut your phrase)")
            } else {
                appendLog("[VAD] Speech ended")
            }
        case .transcriptionCompleted(let text, let language, _):
            pipelineState = "synthesizing..."
            lastTranscription = text
            lastLanguage = language ?? ""
            appendLog("[STT\(language.map { " [\($0)]" } ?? "")] \(text)")
        case .responseCreated:
            pipelineState = "speaking..."
            player.resetGeneration()
        case .responseInterrupted:
            player.stop()
            pipelineState = "listening"
        case .responseAudioDelta(let samples):
            debugTTSBuffer.append(contentsOf: samples)
            try? player.play(samples: samples, sampleRate: 24000)
        case .responseDone:
            appendLog("[TTS] Done")
            player.markGenerationComplete()
        case .toolCallStarted, .toolCallCompleted:
            break
        case .error(let msg):
            pipelineState = "error"
            appendLog("[ERROR] \(msg)")
            pipeline?.resumeListening()
        }
    }

    // MARK: - Microphone

    private func startMicrophone() {
        let engine = AVAudioEngine()
        let inputNode = engine.inputNode

        try? inputNode.setVoiceProcessingEnabled(true)

        let vpFormat = inputNode.outputFormat(forBus: 0)
        guard vpFormat.sampleRate > 0, vpFormat.channelCount > 0 else {
            appendLog("[Mic] Invalid format")
            return
        }

        guard let monoFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32, sampleRate: vpFormat.sampleRate,
            channels: 1, interleaved: false
        ), let targetFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32, sampleRate: 16000,
            channels: 1, interleaved: false
        ), let resampler = AVAudioConverter(from: monoFormat, to: targetFormat) else {
            appendLog("[Mic] Cannot create audio formats")
            return
        }

        var bufCount = 0
        inputNode.installTap(onBus: 0, bufferSize: 1024, format: vpFormat) { [weak self] buffer, _ in
            guard let self, let srcData = buffer.floatChannelData else { return }
            let frameLen = Int(buffer.frameLength)
            guard frameLen > 0 else { return }

            guard let mono = AVAudioPCMBuffer(pcmFormat: monoFormat, frameCapacity: buffer.frameCapacity) else { return }
            mono.frameLength = buffer.frameLength
            memcpy(mono.floatChannelData![0], srcData[0], frameLen * MemoryLayout<Float>.size)

            let outCount = AVAudioFrameCount(Double(frameLen) * 16000.0 / vpFormat.sampleRate)
            guard outCount > 0, let out = AVAudioPCMBuffer(pcmFormat: targetFormat, frameCapacity: outCount) else { return }
            var err: NSError?
            resampler.convert(to: out, error: &err) { _, status in status.pointee = .haveData; return mono }
            guard err == nil, let outData = out.floatChannelData, out.frameLength > 0 else { return }

            let count = Int(out.frameLength)
            let samples = Array(UnsafeBufferPointer(start: outData[0], count: count))
            let rms = sqrt(samples.reduce(0) { $0 + $1 * $1 } / Float(count))

            bufCount += 1
            if bufCount <= 3 {
                DispatchQueue.main.async {
                    self.appendLog("[Mic] Buffer #\(bufCount): \(count) samples, RMS=\(String(format: "%.6f", rms))")
                }
            }
            if bufCount % 5 == 0 {
                DispatchQueue.main.async { self.vadLevel = min(rms / 0.05, 1.0) }
            }

            self.debugMicBuffer.append(contentsOf: samples)
            self.pipeline?.pushAudio(samples)
        }

        do {
            try engine.start()
            audioEngine = engine

            let mixerRate = engine.mainMixerNode.outputFormat(forBus: 0).sampleRate
            guard let playerFmt = AVAudioFormat(
                commonFormat: .pcmFormatFloat32, sampleRate: mixerRate,
                channels: 1, interleaved: false
            ) else { return }
            player.attach(to: engine, format: playerFmt)
            appendLog("[Mic] Started (\(Int(vpFormat.sampleRate))Hz, \(vpFormat.channelCount)ch)")
        } catch {
            appendLog("[Mic] Error: \(error.localizedDescription)")
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

    // MARK: - Debug

    private func saveDebugFiles() {
        let tmp = FileManager.default.temporaryDirectory
        for (buf, name, sr) in [
            (debugMicBuffer, "echo_debug_mic.wav", 16000),
            (debugTTSBuffer, "echo_debug_tts.wav", 24000),
        ] where !buf.isEmpty {
            let url = tmp.appendingPathComponent(name)
            try? WAVWriter.write(samples: buf, sampleRate: sr, to: url)
            appendLog("[Debug] Saved \(name) (\(String(format: "%.1f", Double(buf.count) / Double(sr)))s)")
        }
        debugMicBuffer = []
        debugTTSBuffer = []
    }

    private static let logFileURL: URL = {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("echo_debug.log")
        try? "".write(to: url, atomically: true, encoding: .utf8)
        return url
    }()

    private func appendLog(_ message: String) {
        log.append(message)
        if log.count > 50 { log.removeFirst(log.count - 50) }
        let line = "[\(Date())] \(message)\n"
        if let handle = try? FileHandle(forWritingTo: Self.logFileURL) {
            handle.seekToEndOfFile()
            handle.write(line.data(using: .utf8)!)
            handle.closeFile()
        }
    }
}

#endif
