import AVFoundation
import CoreML
import Foundation
import os
import Observation
import Qwen3Chat
import KokoroTTS
import ParakeetASR
import SpeechVAD
import SpeechCore
import AudioCommon

/// Message displayed in chat UI.
struct ChatBubbleMessage: Identifiable {
    let id = UUID()
    let role: ChatMessage.Role
    var text: String
    let timestamp = Date()
}

private let pipelineLog = Logger(subsystem: "audio.soniqo.TamagotchiDemo", category: "Pipeline")

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

    var modelsLoaded: Bool { vadModel != nil }

    let diagnostics = DiagnosticsMonitor()

    // MARK: - Private State

    private var vadModel: SileroVADModel?
    private var pipeline: VoicePipeline?
    private var audioEngine: AVAudioEngine?
    private let player = StreamingAudioPlayer()
    private var waitingForPlaybackEnd = false
    /// True while TTS is playing — used for echo suppression.
    private var isSpeaking = false
    private var currentResponseText = ""
    private var currentAssistantIdx: Int?

    private let systemPrompt = "You are Tama. Answer questions helpfully in one short sentence."
    /// Echo suppression threshold. On device with speaker echo, 0.06 works.
    /// On simulator (no echo), set lower to avoid blocking real speech.
    #if targetEnvironment(simulator)
    private let echoThreshold: Float = 0.01
    #else
    private let echoThreshold: Float = 0.06
    #endif

    // MARK: - Load Models

    func loadModels() async {
        isLoading = true
        errorMessage = nil
        loadProgress = 0

        do {
            loadingStatus = "Loading VAD..."
            loadProgress = 0.3
            vadModel = try await Task.detached {
                try await SileroVADModel.fromPretrained(engine: .coreml) { progress, status in
                    DispatchQueue.main.async { [weak self] in
                        self?.loadProgress = 0.3 + progress * 0.6
                        if !status.isEmpty { self?.loadingStatus = status }
                    }
                }
            }.value
            loadProgress = 1.0
            loadingStatus = "Ready"
        } catch {
            errorMessage = "Load failed: \(error.localizedDescription)"
        }

        isLoading = false
    }

    // MARK: - Pipeline Start/Stop

    func startListening() {
        guard !isListening, let vad = vadModel else { return }

        let sampling = ChatSamplingConfig(
            temperature: 0.3, topK: 20, maxTokens: 15, repetitionPenalty: 1.5)
        let sysPrompt = systemPrompt

        var config = PipelineConfig()
        config.mode = .voicePipeline
        config.allowInterruptions = true
        config.minInterruptionDuration = 1.0
        config.minSilenceDuration = 0.8
        config.maxResponseDuration = 10.0
        config.warmupSTT = true
        config.preSpeechBufferDuration = 1.5  // Keep 1.5s before VAD trigger to capture phrase start
        config.autoUnloadModels = false

        pipeline = VoicePipeline(
            sttFactory: {
                try await ParakeetASRModel.fromPretrained(
                    modelId: ParakeetASRModel.int8iOSModelId,
                    computeUnits: .cpuAndNeuralEngine  // Skip GPU plan compilation — saves memory + startup
                ) { _, _ in }
            },
            ttsFactory: {
                try await KokoroTTSModel.fromPretrained(
                    modelId: KokoroTTSModel.int8iOSModelId,
                    computeUnits: .cpuAndNeuralEngine  // Skip GPU plan — saves memory
                ) { _, _ in }
            },
            vad: vad,
            llmFactory: { [weak self] in
                let chat = try await Qwen3ChatModel.fromPretrained(computeUnits: .cpuAndNeuralEngine) { _, _ in }
                let llm = Qwen3PipelineLLM(model: chat, systemPrompt: sysPrompt, sampling: sampling)
                llm.onToken = { token in
                    DispatchQueue.main.async { self?.appendToken(token) }
                }
                return llm
            },
            config: config,
            onEvent: { [weak self] event in
                DispatchQueue.main.async { self?.handleEvent(event) }
            }
        )

        player.onPlaybackFinished = { [weak self] in
            DispatchQueue.main.async { self?.playbackDidFinish() }
        }

        pipelineLog.warning("[START] pipeline created, starting...")
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
    }

    // MARK: - Pipeline Events

    private func handleEvent(_ event: PipelineEvent) {
        switch event {
        case .sessionCreated:
            pipelineLog.warning("[EVT] sessionCreated")

        case .speechStarted:
            pipelineLog.warning("[EVT] speechStarted (isGenerating=\(self.isGenerating) isSpeaking=\(self.isSpeaking))")
            isSpeechDetected = true
            pipelineState = "speech detected"
            if isGenerating || isSpeaking {
                pipeline?.cancelLLM()
                player.fadeOutAndStop()
                isSpeaking = false
                isGenerating = false
            }

        case .speechEnded:
            pipelineLog.warning("[EVT] speechEnded")
            isSpeechDetected = false
            pipelineState = "transcribing..."

        case .transcriptionCompleted(let text, let lang, let conf):
            let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
            pipelineLog.warning("[EVT] transcription: '\(trimmed)' lang=\(lang ?? "-") conf=\(conf)")
            guard !trimmed.isEmpty else { return }
            messages.append(ChatBubbleMessage(role: .user, text: trimmed))
            pipelineState = "thinking..."
            isGenerating = true

        case .responseCreated:
            pipelineLog.warning("[EVT] responseCreated")
            if currentAssistantIdx == nil {
                currentResponseText = ""
                currentAssistantIdx = messages.count
                messages.append(ChatBubbleMessage(role: .assistant, text: "..."))
                pipelineState = "generating..."
            }

        case .responseInterrupted:
            pipelineLog.warning("[EVT] responseInterrupted")
            player.fadeOutAndStop()
            isSpeaking = false
            isGenerating = false
            currentAssistantIdx = nil
            pipelineState = "interrupted"

        case .responseAudioDelta(let samples):
            pipelineLog.warning("[EVT] audioDelta \(samples.count) samples")
            pipelineState = "speaking..."
            isSpeaking = true
            do { try player.play(samples: samples, sampleRate: 24000) }
            catch { pipelineLog.error("[EVT] playback error: \(error)") }

        case .responseDone:
            pipelineLog.warning("[EVT] responseDone")
            isGenerating = false
            currentAssistantIdx = nil
            if player.isPlaying {
                waitingForPlaybackEnd = true
            } else {
                resumeAfterResponse()
            }

        case .toolCallStarted(let name):
            pipelineLog.warning("[EVT] toolCall: \(name)")

        case .toolCallCompleted:
            break

        case .error(let msg):
            pipelineLog.error("[EVT] error: \(msg)")
            errorMessage = msg
            pipelineState = "error"
            isGenerating = false
            pipeline?.resumeListening()
        }
    }

    private func playbackDidFinish() {
        guard waitingForPlaybackEnd else { return }
        waitingForPlaybackEnd = false
        resumeAfterResponse()
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
        do {
            try session.setCategory(.playAndRecord, mode: .default,
                                    options: [.defaultToSpeaker, .allowBluetooth])
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

            // Echo suppression: drop low-energy frames during TTS playback.
            // Speaker echo is quiet (RMS < 0.06), real speech is louder.
            if self.isSpeaking {
                var sum: Float = 0
                let ptr = srcData[0]
                for i in 0..<frameLen { sum += ptr[i] * ptr[i] }
                let rms = sqrt(sum / Float(frameLen))
                if rms < self.echoThreshold { return }
            }

            // Resample to 16kHz mono
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

            // RMS for visual level
            var sum: Float = 0
            for s in samples { sum += s * s }
            let rms = sqrt(sum / max(Float(count), 1))
            DispatchQueue.main.async {
                self.audioLevel = rms
                self.diagnostics.updateVAD(rms)
            }

            self.pipeline?.pushAudio(samples)
        }

        // Attach TTS player at 24kHz before starting engine
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

    // MARK: - LLM Token Streaming

    func appendToken(_ token: String) {
        if currentAssistantIdx == nil {
            currentResponseText = ""
            currentAssistantIdx = messages.count
            messages.append(ChatBubbleMessage(role: .assistant, text: ""))
            pipelineState = "generating..."
            isGenerating = true
        }
        guard let idx = currentAssistantIdx, idx < messages.count else { return }
        currentResponseText += token
        messages[idx].text = currentResponseText
    }
}
