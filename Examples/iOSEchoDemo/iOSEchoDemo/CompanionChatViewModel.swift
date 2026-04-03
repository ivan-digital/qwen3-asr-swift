import AVFoundation
import CoreML
import Foundation
import os
import Observation
import KokoroTTS
import ParakeetASR
import SpeechVAD
import SpeechCore
import AudioCommon

/// Apple built-in TTS for simulator — uses AVSpeechSynthesizer.speak() which plays
/// directly through speakers. The .write() API produces empty buffers on simulator.
final class AppleTTSModel: NSObject, SpeechGenerationModel, AVSpeechSynthesizerDelegate {
    var sampleRate: Int { 24000 }
    private let synthesizer = AVSpeechSynthesizer()
    private var continuation: CheckedContinuation<[Float], Error>?

    override init() {
        super.init()
        synthesizer.delegate = self
    }

    func generate(text: String, language: String?) async throws -> [Float] {
        let utterance = AVSpeechUtterance(string: text)
        utterance.voice = AVSpeechSynthesisVoice(language: language ?? "en-US")
        utterance.rate = AVSpeechUtteranceDefaultSpeechRate

        return try await withCheckedThrowingContinuation { cont in
            self.continuation = cont
            synthesizer.speak(utterance)
        }
    }

    // AVSpeechSynthesizer plays audio directly — return empty samples
    // since the pipeline doesn't need to play them via StreamingAudioPlayer
    func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didFinish utterance: AVSpeechUtterance) {
        continuation?.resume(returning: [Float](repeating: 0, count: 2400))
        continuation = nil
    }
}

enum MessageRole { case user, assistant, system }

/// Message displayed in chat UI.
struct ChatBubbleMessage: Identifiable {
    let id = UUID()
    let role: MessageRole
    var text: String
    let timestamp = Date()
}

private let pipelineLog = Logger(subsystem: "audio.soniqo.iOSEchoDemo", category: "Pipeline")

@Observable
@MainActor
final class CompanionChatViewModel {
    // MARK: - UI State

    var messages: [ChatBubbleMessage] = []
    var inputText = ""
    var isLoading = false
    var isGenerating = false
    var isListening = false
    var isSpeechDetected = false
    var pipelineState = "idle"
    var audioLevel: Float = 0
    var loadProgress: Double = 0
    var loadingStatus = ""
    var errorMessage: String?

    private var _modelsLoaded = false
    var modelsLoaded: Bool { _modelsLoaded }

    let diagnostics = DiagnosticsMonitor()

    // MARK: - Private State

    private var vadModel: SileroVADModel?
    private var sttModel: ParakeetASRModel?
    private var ttsModel: (any SpeechGenerationModel)?
    private var pipeline: VoicePipeline?
    private var pipelinePostPlaybackGuard: Double = 2.0
    private var audioEngine: AVAudioEngine?
    private let player = StreamingAudioPlayer()
    private var isSpeaking = false
    private var speechStartTime: CFAbsoluteTime = 0
    private var wasForceCut = false
    private var forceCutCooldownEnd: CFAbsoluteTime = 0
    private var lastResponseAudioDuration: Double = 0
    private var responseAudioStartTime: CFAbsoluteTime = 0
    private var micRecordBuffer: [Float] = []
    private var ttsRecordBuffer: [Float] = []
    private var debugLog: [String] = []

    private func dbg(_ msg: String) {
        let ts = String(format: "%.3f", CFAbsoluteTimeGetCurrent().truncatingRemainder(dividingBy: 1000))
        let line = "[\(ts)] \(msg)"
        debugLog.append(line)
        pipelineLog.warning("\(line, privacy: .public)")
    }

    // MARK: - Load Models

    func loadModels() async {
        isLoading = true
        errorMessage = nil
        loadProgress = 0

        do {
            loadingStatus = "Loading VAD..."
            loadProgress = 0.05
            vadModel = try await Task.detached {
                try await SileroVADModel.fromPretrained(engine: .coreml) { progress, status in
                    DispatchQueue.main.async { [weak self] in
                        self?.loadProgress = 0.05 + progress * 0.15
                        if !status.isEmpty { self?.loadingStatus = "VAD: \(status)" }
                    }
                }
            }.value

            // Pre-download and compile STT model (~500MB).
            // Without this, the first speech triggers a download that blocks
            // the pipeline worker thread — no transcriptions until complete.
            loadingStatus = "Downloading ASR model..."
            loadProgress = 0.2
            sttModel = try await Task.detached {
                try await ParakeetASRModel.fromPretrained(
                    modelId: ParakeetASRModel.defaultModelId) { progress, status in
                    DispatchQueue.main.async { [weak self] in
                        self?.loadProgress = 0.2 + progress * 0.4
                        if !status.isEmpty { self?.loadingStatus = "ASR: \(status)" }
                    }
                }
            }.value

            // Load TTS model
            loadingStatus = "Loading TTS..."
            loadProgress = 0.6
            #if targetEnvironment(simulator)
            // Simulator: Apple built-in TTS (CoreML models too slow on CPU-only simulator)
            ttsModel = AppleTTSModel()
            loadProgress = 0.95
            #else
            // Device: Kokoro CoreML (fast on ANE/GPU)
            ttsModel = try await Task.detached {
                try await KokoroTTSModel.fromPretrained { progress, status in
                    DispatchQueue.main.async { [weak self] in
                        self?.loadProgress = 0.6 + progress * 0.35
                        if !status.isEmpty { self?.loadingStatus = "TTS: \(status)" }
                    }
                }
            }.value
            #endif

            loadProgress = 1.0
            loadingStatus = "Ready"
            _modelsLoaded = true
        } catch {
            errorMessage = "Load failed: \(error.localizedDescription)"
        }

        isLoading = false
    }

    // MARK: - Pipeline Start/Stop

    func startListening() {
        guard !isListening, let vad = vadModel,
              let stt = sttModel, let tts = ttsModel else { return }

        var config = PipelineConfig()
        config.mode = .echo  // ASR → TTS, no LLM
        config.allowInterruptions = false  // No AEC — can't distinguish user from speaker
        config.minSilenceDuration = 0.6
        config.maxUtteranceDuration = 5.0   // Force-submit to STT after 5s (longer causes OOM without ANE)
        config.maxResponseDuration = 5.0   // Cap TTS output to prevent repetition loops
        config.eagerSTT = true  // Start transcribing during speech, don't wait for silence
        config.warmupSTT = false
        config.preSpeechBufferDuration = 2.0  // Capture beginning of sentences before VAD fires
        config.postPlaybackGuard = 2.0  // Suppress VAD for 2s after TTS to prevent echo feedback (no AEC yet)

        pipeline = VoicePipeline(
            stt: stt,
            tts: tts,
            vad: vad,
            config: config,
            onEvent: { [weak self] event in
                DispatchQueue.main.async { self?.handleEvent(event) }
            }
        )


        pipelineLog.warning("[START] echo pipeline created")

        pipeline?.start()
        isListening = true
        pipelineState = "listening"
        diagnostics.start()
        startMicrophone()
        pipelineLog.warning("[START] mic started, pipeline running")
    }

    func stopListening() {
        diagnostics.stop()
        stopMicrophone()
        pipeline?.stop()
        pipeline = nil
        isListening = false
        isGenerating = false
        isSpeechDetected = false
        isSpeaking = false
        audioLevel = 0
        pipelineState = "idle"
        saveDebugRecording()
    }

    private func saveDebugRecording() {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let dir = docs.appendingPathComponent("debug_audio")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)

        if !micRecordBuffer.isEmpty {
            let url = dir.appendingPathComponent("mic_debug.wav")
            writeWAV(samples: micRecordBuffer, sampleRate: 16000, to: url)
            pipelineLog.warning("DEBUG MIC: \(url.path) (\(self.micRecordBuffer.count / 16000)s)")
            micRecordBuffer.removeAll()
        }

        if !ttsRecordBuffer.isEmpty {
            let url = dir.appendingPathComponent("tts_debug.wav")
            writeWAV(samples: ttsRecordBuffer, sampleRate: 24000, to: url)
            pipelineLog.warning("DEBUG TTS: \(url.path) (\(self.ttsRecordBuffer.count / 24000)s)")
            ttsRecordBuffer.removeAll()
        }

        if !debugLog.isEmpty {
            let logUrl = dir.appendingPathComponent("pipeline_debug.log")
            try? debugLog.joined(separator: "\n").write(to: logUrl, atomically: true, encoding: .utf8)
            debugLog.removeAll()
        }
    }

    private func writeWAV(samples: [Float], sampleRate: Int, to url: URL) {
        var data = Data()
        let dataSize = samples.count * 2
        data.append(contentsOf: "RIFF".utf8)
        var fileSize = UInt32(36 + dataSize); data.append(Data(bytes: &fileSize, count: 4))
        data.append(contentsOf: "WAVE".utf8)
        data.append(contentsOf: "fmt ".utf8)
        var fmtSize: UInt32 = 16; data.append(Data(bytes: &fmtSize, count: 4))
        var fmt: UInt16 = 1; data.append(Data(bytes: &fmt, count: 2))
        var ch: UInt16 = 1; data.append(Data(bytes: &ch, count: 2))
        var sr = UInt32(sampleRate); data.append(Data(bytes: &sr, count: 4))
        var byteRate = UInt32(sampleRate * 2); data.append(Data(bytes: &byteRate, count: 4))
        var blockAlign: UInt16 = 2; data.append(Data(bytes: &blockAlign, count: 2))
        var bps: UInt16 = 16; data.append(Data(bytes: &bps, count: 2))
        data.append(contentsOf: "data".utf8)
        var dSize = UInt32(dataSize); data.append(Data(bytes: &dSize, count: 4))
        for s in samples {
            var pcm = Int16(max(-1, min(1, s)) * 32767)
            data.append(Data(bytes: &pcm, count: 2))
        }
        try? data.write(to: url)
    }

    // MARK: - Pipeline Events

    private func handleEvent(_ event: PipelineEvent) {
        switch event {
        case .sessionCreated:
            dbg("sessionCreated")

        case .speechStarted:
            // After force-cut, ignore speech until TTS response finishes + postPlaybackGuard
            if wasForceCut && CFAbsoluteTimeGetCurrent() < forceCutCooldownEnd {
                dbg("speechStarted IGNORED (force-cut cooldown, \(String(format: "%.1f", forceCutCooldownEnd - CFAbsoluteTimeGetCurrent()))s remaining)")
                speechStartTime = CFAbsoluteTimeGetCurrent()
                return
            }
            wasForceCut = false
            dbg("speechStarted")
            isSpeechDetected = true
            speechStartTime = CFAbsoluteTimeGetCurrent()
            pipelineState = "listening..."

        case .speechEnded:
            // Only check force-cut if we're actually tracking speech
            guard isSpeechDetected else {
                dbg("speechEnded IGNORED (not tracking)")
                return
            }
            let elapsed = CFAbsoluteTimeGetCurrent() - speechStartTime
            wasForceCut = elapsed >= 4.5  // ~maxUtteranceDuration (5s)
            if wasForceCut {
                // Initial cooldown until responseDone extends it
                forceCutCooldownEnd = .greatestFiniteMagnitude
                dbg("speechEnded (MAX LENGTH after \(String(format: "%.1f", elapsed))s)")
                pipelineState = "max length reached, transcribing..."
                messages.append(ChatBubbleMessage(role: .system,
                    text: "Recording limit reached (\(Int(elapsed))s). Transcribing what was captured."))
            } else {
                dbg("speechEnded (\(String(format: "%.1f", elapsed))s)")
                pipelineState = "transcribing..."
            }
            isSpeechDetected = false

        case .transcriptionCompleted(let text, let lang, let conf):
            let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
            dbg("transcription: '\(trimmed)' lang=\(lang ?? "-") conf=\(String(format: "%.2f", conf))")
            guard !trimmed.isEmpty else {
                // Empty transcription after force-cut → end cooldown so pipeline isn't stuck
                if wasForceCut {
                    wasForceCut = false
                    forceCutCooldownEnd = 0
                    pipelineState = "listening"
                    dbg("empty transcription after force-cut — cooldown cleared")
                }
                return
            }
            messages.append(ChatBubbleMessage(role: .user, text: trimmed))
            // Echo mode: transcription is sent directly to TTS by the pipeline
            pipelineState = "speaking..."
            isGenerating = true

        case .responseCreated:
            dbg("responseCreated")
            lastResponseAudioDuration = 0
            // In echo mode, the response IS the transcription — show it as assistant echo
            if let lastUser = messages.last, lastUser.role == .user {
                messages.append(ChatBubbleMessage(role: .assistant, text: "🔊 \(lastUser.text)"))
            }

        case .responseInterrupted:
            dbg("responseInterrupted")
            player.fadeOutAndStop()
            isSpeaking = false
            isGenerating = false
            pipelineState = "listening"

        case .responseAudioDelta(let samples):
            if !isSpeaking { responseAudioStartTime = CFAbsoluteTimeGetCurrent() }
            isSpeaking = true
            lastResponseAudioDuration += Double(samples.count) / 24000.0
            pipelineState = "speaking..."
            dbg("audioDelta: \(samples.count) samples (\(String(format: "%.2f", Double(samples.count)/24000))s)")
            do { try player.play(samples: samples, sampleRate: 24000) }
            catch { dbg("playback error: \(error)") }

        case .responseDone:
            // Guard = remaining playback time + postPlaybackGuard
            let elapsedSinceAudioStart = CFAbsoluteTimeGetCurrent() - responseAudioStartTime
            let remainingPlayback = max(0, lastResponseAudioDuration - elapsedSinceAudioStart)
            let guard_ = remainingPlayback + pipelinePostPlaybackGuard
            dbg("responseDone (audio=\(String(format: "%.1f", lastResponseAudioDuration))s, guard=\(String(format: "%.1f", guard_))s)")
            isGenerating = false
            isSpeaking = false
            player.markGenerationComplete()
            if wasForceCut {
                forceCutCooldownEnd = CFAbsoluteTimeGetCurrent() + guard_
            }
            lastResponseAudioDuration = 0
            resumeAfterResponse()

        case .toolCallStarted, .toolCallCompleted:
            break

        case .error(let msg):
            dbg("ERROR: \(msg)")
            errorMessage = msg
            pipelineState = "error"
            isGenerating = false
            pipeline?.resumeListening()
        }
    }

    private func resumeAfterResponse() {
        guard isListening else { return }
        isSpeaking = false
        pipeline?.resumeListening()
        pipelineState = "listening"
    }

    // MARK: - Microphone

    private func startMicrophone() {
        #if os(iOS)
        let session = AVAudioSession.sharedInstance()

        switch AVAudioApplication.shared.recordPermission {
        case .undetermined:
            AVAudioApplication.requestRecordPermission { [weak self] granted in
                DispatchQueue.main.async {
                    if granted {
                        self?.startMicrophone()
                    } else {
                        self?.errorMessage = "Microphone permission denied"
                    }
                }
            }
            return
        case .denied:
            errorMessage = "Microphone permission denied. Enable in Settings."
            return
        case .granted:
            break
        @unknown default:
            break
        }

        do {
            try session.setCategory(.playAndRecord, mode: .default,
                                    options: [.defaultToSpeaker, .allowBluetoothHFP])
            try session.setActive(true)
        } catch {
            errorMessage = "Mic access failed: \(error.localizedDescription)"
            return
        }
        #endif

        let engine = AVAudioEngine()
        let inputNode = engine.inputNode
        let hwFormat = inputNode.outputFormat(forBus: 0)

        guard let targetFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32, sampleRate: 16000,
            channels: 1, interleaved: false
        ) else { return }

        guard let monoFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32, sampleRate: hwFormat.sampleRate,
            channels: 1, interleaved: false
        ) else { return }

        guard let resampler = AVAudioConverter(from: monoFormat, to: targetFormat) else { return }

        inputNode.installTap(onBus: 0, bufferSize: 1024, format: hwFormat) { [weak self] buffer, _ in
            guard let self else { return }
            guard let srcData = buffer.floatChannelData else { return }
            let frameLen = Int(buffer.frameLength)
            guard frameLen > 0 else { return }

            guard let monoBuffer = AVAudioPCMBuffer(pcmFormat: monoFormat,
                                                     frameCapacity: buffer.frameCapacity) else { return }
            monoBuffer.frameLength = buffer.frameLength
            memcpy(monoBuffer.floatChannelData![0], srcData[0], frameLen * MemoryLayout<Float>.size)

            let outFrameCount = AVAudioFrameCount(Double(frameLen) * 16000.0 / hwFormat.sampleRate)
            guard outFrameCount > 0,
                  let outBuffer = AVAudioPCMBuffer(pcmFormat: targetFormat,
                                                    frameCapacity: outFrameCount) else { return }

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

            var sum: Float = 0
            for s in samples { sum += s * s }
            let rms = sqrt(sum / max(Float(count), 1))
            DispatchQueue.main.async {
                self.audioLevel = rms
                self.diagnostics.updateVAD(rms)
            }

            self.micRecordBuffer.append(contentsOf: samples)
            let maxMicSamples = 16000 * 60
            if self.micRecordBuffer.count > maxMicSamples {
                self.micRecordBuffer.removeFirst(self.micRecordBuffer.count - maxMicSamples)
            }

            // Don't feed audio during force-cut cooldown — prevents echo/recurring
            if self.wasForceCut && CFAbsoluteTimeGetCurrent() < self.forceCutCooldownEnd {
                return
            }
            self.pipeline?.pushAudio(samples)
        }

        guard let playerFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32, sampleRate: 24000,
            channels: 1, interleaved: false
        ) else { return }
        player.attach(to: engine, format: playerFormat)

        do {
            try engine.start()
            player.startPlayback()
            audioEngine = engine
        } catch {
            errorMessage = "Mic error: \(error.localizedDescription)"
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

    // MARK: - Text Input

    func send(_ text: String) {
        inputText = ""
        guard isListening else {
            messages.append(ChatBubbleMessage(role: .user, text: text))
            return
        }
        pipeline?.pushText(text)
    }

    func clearChat() {
        messages = []
    }
}
