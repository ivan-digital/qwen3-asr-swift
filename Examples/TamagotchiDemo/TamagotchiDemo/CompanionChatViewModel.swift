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
    var speakEnabled = true

    var modelsLoaded: Bool { chatModel != nil && ttsModel != nil && asrModel != nil && vadModel != nil }

    let diagnostics = DiagnosticsMonitor()

    // MARK: - Models

    private var chatModel: Qwen3ChatModel?
    private var ttsModel: KokoroTTSModel?
    private var asrModel: ParakeetASRModel?
    private var vadModel: SileroVADModel?

    // MARK: - Pipeline

    private var pipeline: VoicePipeline?
    private var pipelineLLM: Qwen3PipelineLLM?
    private var audioEngine: AVAudioEngine?
    private let player = StreamingAudioPlayer()
    private var waitingForPlaybackEnd = false
    #if targetEnvironment(simulator)
    private let speechSynth = AVSpeechSynthesizer()
    #endif

    private let systemPrompt = "You are a helpful assistant. Reply in one short sentence."

    // MARK: - Load Models

    /// CoreML `.all` lets the framework pick the best backend per model.
    /// On simulator, GPU/ANE fallback errors are non-fatal — models run on CPU.
    private let coreMLUnits: MLComputeUnits = .all

    func loadModels() async {
        isLoading = true
        errorMessage = nil
        loadProgress = 0

        // Models are loaded sequentially to avoid peak memory from concurrent
        // CoreML compilation. Each model: download → compile → settle before next.
        // Total footprint: ~960 MB (VAD 1 + ASR 332 + LLM 318 + TTS 310).
        // iPhone allows ~2-3 GB — no warmup runs to avoid doubling memory.
        let units = coreMLUnits

        do {
            // 1. VAD (~1 MB)
            loadingStatus = "VAD..."
            loadProgress = 0.05
            vadModel = try await Task.detached {
                try await SileroVADModel.fromPretrained(engine: .coreml) { progress, status in
                    DispatchQueue.main.async { [weak self] in
                        self?.loadProgress = 0.05 + progress * 0.05
                        if !status.isEmpty { self?.loadingStatus = "VAD: \(status)" }
                    }
                }
            }.value

            // 2. ASR (~332 MB)
            loadingStatus = "Parakeet ASR..."
            loadProgress = 0.1
            asrModel = try await Task.detached {
                try await ParakeetASRModel.fromPretrained { progress, status in
                    DispatchQueue.main.async { [weak self] in
                        self?.loadProgress = 0.1 + progress * 0.3
                        if !status.isEmpty { self?.loadingStatus = "Parakeet: \(status)" }
                    }
                }
            }.value

            // 3. LLM (~318 MB)
            loadingStatus = "Qwen3 Chat..."
            loadProgress = 0.4
            chatModel = try await Task.detached {
                try await Qwen3ChatModel.fromPretrained(computeUnits: units) { progress, status in
                    DispatchQueue.main.async { [weak self] in
                        self?.loadProgress = 0.4 + progress * 0.2
                        if !status.isEmpty { self?.loadingStatus = "Qwen3: \(status)" }
                    }
                }
            }.value

            // 4. TTS (~310 MB)
            loadingStatus = "Kokoro TTS..."
            loadProgress = 0.6
            ttsModel = try await Task.detached {
                try await KokoroTTSModel.fromPretrained { progress, status in
                    DispatchQueue.main.async { [weak self] in
                        self?.loadProgress = 0.6 + progress * 0.3
                        if !status.isEmpty { self?.loadingStatus = "Kokoro: \(status)" }
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
        guard !isListening else { return }
        guard let vad = vadModel, let asr = asrModel, let tts = ttsModel, let chat = chatModel else { return }

        let sampling = ChatSamplingConfig(
            temperature: 0.6, topK: 40, maxTokens: 15, repetitionPenalty: 1.2)

        let llm = Qwen3PipelineLLM(
            model: chat,
            systemPrompt: systemPrompt,
            sampling: sampling
        )
        pipelineLLM = llm
        llm.onToken = { [weak self] token in
            DispatchQueue.main.async {
                self?.appendToken(token)
            }
        }

        var config = PipelineConfig()
        config.mode = .voicePipeline
        config.allowInterruptions = true
        config.minSilenceDuration = 0.4
        config.maxResponseDuration = 10.0
        config.warmupSTT = false  // Warmup done during model loading

        pipeline = VoicePipeline(
            stt: asr,
            tts: tts,
            vad: vad,
            llm: llm,
            config: config,
            onEvent: { [weak self] event in
                DispatchQueue.main.async {
                    self?.handleEvent(event)
                }
            }
        )

        player.onPlaybackFinished = { [weak self] in
            DispatchQueue.main.async {
                self?.playbackDidFinish()
            }
        }

        pipeline?.start()
        isListening = true
        pipelineState = "listening"
        diagnostics.start()
        startMicrophone()
    }

    func stopListening() {
        diagnostics.stop()
        stopMicrophone()
        pipelineLLM?.cancel()
        pipeline?.stop()
        pipeline = nil
        pipelineLLM = nil
        isListening = false
        isGenerating = false
        isSpeechDetected = false
        audioLevel = 0
        pipelineState = "idle"
        saveDebugRecordings()
    }

    // MARK: - Pipeline Events

    private var currentResponseText = ""
    private var currentAssistantIdx: Int?

    private func handleEvent(_ event: PipelineEvent) {
        switch event {
        case .sessionCreated:
            break

        case .speechStarted:
            isSpeechDetected = true
            pipelineState = "speech detected"

        case .speechEnded:
            isSpeechDetected = false
            pipelineState = "transcribing..."

        case .transcriptionCompleted(let text, _, _):
            let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
            pipelineLog.warning("Transcription: '\(trimmed)'")
            guard !trimmed.isEmpty else { return }
            messages.append(ChatBubbleMessage(role: .user, text: trimmed))
            pipelineState = "thinking..."
            isGenerating = true

        case .responseCreated:
            pipelineLog.warning("Event: responseCreated")
            // Assistant message may already exist from first LLM token.
            if currentAssistantIdx == nil {
                currentResponseText = ""
                currentAssistantIdx = messages.count
                messages.append(ChatBubbleMessage(role: .assistant, text: "..."))
                pipelineState = "generating..."
            }

        case .responseInterrupted:
            player.fadeOutAndStop()
            pipelineLLM?.cancel()  // Cancel LLM to unblock worker thread
            isGenerating = false
            currentAssistantIdx = nil
            pipelineState = "interrupted"

        case .responseAudioDelta(let samples):
            pipelineLog.warning("Event: responseAudioDelta \(samples.count) samples")
            pipelineState = "speaking..."
            recordTTSSamples(samples, sampleRate: 24000)
            #if !targetEnvironment(simulator)
            do {
                try player.play(samples: samples, sampleRate: 24000)
            } catch {
                pipelineLog.error("Playback error: \(error)")
            }
            #endif

        case .responseDone:
            pipelineLog.warning("Event: responseDone")
            isGenerating = false
            let responseText = currentResponseText
            currentAssistantIdx = nil
            #if targetEnvironment(simulator)
            if speakEnabled, !responseText.isEmpty {
                let utterance = AVSpeechUtterance(string: responseText)
                utterance.rate = AVSpeechUtteranceDefaultSpeechRate
                speechSynth.speak(utterance)
            }
            resumeAfterResponse()
            #else
            if player.isPlaying {
                waitingForPlaybackEnd = true
            } else {
                resumeAfterResponse()
            }
            #endif

        case .toolCallStarted(let name):
            pipelineLog.info("Tool call: \(name)")
            pipelineState = "calling \(name)..."

        case .toolCallCompleted(let name, _):
            pipelineLog.info("Tool done: \(name)")

        case .error(let msg):
            pipelineLog.error("Event: error '\(msg)'")
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
        pipeline?.resumeListening()
        pipelineState = "listening"
    }

    // MARK: - Microphone (AVAudioConverter resampling)

    private func startMicrophone() {
        #if os(iOS)
        let session = AVAudioSession.sharedInstance()
        do {
            try session.setCategory(.playAndRecord, mode: .default, options: [.defaultToSpeaker, .allowBluetooth])
            try session.setActive(true)
        } catch {
            errorMessage = "Mic access failed: \(error.localizedDescription)"
            return
        }
        #endif

        let engine = AVAudioEngine()
        let inputNode = engine.inputNode
        let hwFormat = inputNode.outputFormat(forBus: 0)

        // 16kHz mono target for VAD/ASR (Apple AVAudioConverter resampling)
        guard let targetFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: 16000,
            channels: 1,
            interleaved: false
        ) else { return }

        // Mono intermediate at hardware rate
        guard let monoFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: hwFormat.sampleRate,
            channels: 1,
            interleaved: false
        ) else { return }

        guard let resampler = AVAudioConverter(from: monoFormat, to: targetFormat) else { return }

        inputNode.installTap(onBus: 0, bufferSize: 1024, format: hwFormat) { [weak self] buffer, _ in
            guard let self else { return }
            guard let srcData = buffer.floatChannelData else { return }
            let frameLen = Int(buffer.frameLength)
            guard frameLen > 0 else { return }

            // Extract channel 0 into mono buffer
            guard let monoBuffer = AVAudioPCMBuffer(pcmFormat: monoFormat, frameCapacity: buffer.frameCapacity) else { return }
            monoBuffer.frameLength = buffer.frameLength
            memcpy(monoBuffer.floatChannelData![0], srcData[0], frameLen * MemoryLayout<Float>.size)

            // Resample to 16kHz using AVAudioConverter
            let outFrameCount = AVAudioFrameCount(Double(frameLen) * 16000.0 / hwFormat.sampleRate)
            guard outFrameCount > 0,
                  let outBuffer = AVAudioPCMBuffer(pcmFormat: targetFormat, frameCapacity: outFrameCount) else { return }

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

            // Debug: record all mic audio continuously
            DispatchQueue.main.async {
                self.recordMicSamples(samples)
            }

            // Feed resampled audio to pipeline
            self.pipeline?.pushAudio(samples)
        }

        // Attach TTS player at 24kHz (Kokoro TTS native rate) BEFORE starting engine.
        // AVAudioEngine handles the 24kHz→hardware rate conversion internally.
        guard let playerFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: 24000,
            channels: 1,
            interleaved: false
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

    // MARK: - Text Input (fallback, bypasses STT)

    func send(_ text: String) {
        inputText = ""
        guard isListening else {
            // If pipeline not running, just show message
            messages.append(ChatBubbleMessage(role: .user, text: text))
            return
        }
        pipeline?.pushText(text)
    }

    // MARK: - Actions

    func clearChat() {
        messages = []
        chatModel?.resetConversation()
    }

    // MARK: - LLM token streaming → UI

    /// Called from pipeline events to update the assistant message bubble.
    func appendToken(_ token: String) {
        // Create assistant message on first token (before responseCreated,
        // which only fires when TTS starts after LLM is done).
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

    // MARK: - Debug Audio Recording
    //
    // One continuous mic recording (like EchoDemo) — saved on stop to debug lost speech.

    private var micRecordBuffer: [Float] = []
    private var ttsRecordBuffer: [Float] = []

    private func recordMicSamples(_ samples: [Float]) {
        micRecordBuffer.append(contentsOf: samples)
    }

    private func recordTTSSamples(_ samples: [Float], sampleRate: Int) {
        ttsRecordBuffer.append(contentsOf: samples)
    }

    /// Save all debug recordings (called on stopListening).
    private func saveDebugRecordings() {
        if !micRecordBuffer.isEmpty {
            let url = debugAudioURL("mic_full.wav")
            writeWAV(samples: micRecordBuffer, sampleRate: 16000, to: url)
            let durMs = micRecordBuffer.count * 1000 / 16000
            pipelineLog.warning("Saved full mic recording: \(url.path) (\(durMs)ms)")
            micRecordBuffer.removeAll()
        }
        if !ttsRecordBuffer.isEmpty {
            let url = debugAudioURL("tts_full.wav")
            writeWAV(samples: ttsRecordBuffer, sampleRate: 24000, to: url)
            let durMs = ttsRecordBuffer.count * 1000 / 24000
            pipelineLog.warning("Saved full TTS recording: \(url.path) (\(durMs)ms)")
            ttsRecordBuffer.removeAll()
        }
    }

    private func debugAudioURL(_ name: String) -> URL {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let dir = docs.appendingPathComponent("debug_audio")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir.appendingPathComponent(name)
    }

    private func writeWAV(samples: [Float], sampleRate: Int, to url: URL) {
        let numSamples = samples.count
        let dataSize = numSamples * 2  // 16-bit PCM
        var data = Data()
        // RIFF header
        data.append(contentsOf: "RIFF".utf8)
        var fileSize = UInt32(36 + dataSize)
        data.append(Data(bytes: &fileSize, count: 4))
        data.append(contentsOf: "WAVE".utf8)
        // fmt chunk
        data.append(contentsOf: "fmt ".utf8)
        var fmtSize: UInt32 = 16
        data.append(Data(bytes: &fmtSize, count: 4))
        var audioFormat: UInt16 = 1  // PCM
        data.append(Data(bytes: &audioFormat, count: 2))
        var channels: UInt16 = 1
        data.append(Data(bytes: &channels, count: 2))
        var sr = UInt32(sampleRate)
        data.append(Data(bytes: &sr, count: 4))
        var byteRate = UInt32(sampleRate * 2)
        data.append(Data(bytes: &byteRate, count: 4))
        var blockAlign: UInt16 = 2
        data.append(Data(bytes: &blockAlign, count: 2))
        var bitsPerSample: UInt16 = 16
        data.append(Data(bytes: &bitsPerSample, count: 2))
        // data chunk
        data.append(contentsOf: "data".utf8)
        var dataChunkSize = UInt32(dataSize)
        data.append(Data(bytes: &dataChunkSize, count: 4))
        for s in samples {
            let clamped = max(-1.0, min(1.0, s))
            var pcm = Int16(clamped * 32767)
            data.append(Data(bytes: &pcm, count: 2))
        }
        try? data.write(to: url)
    }
}
