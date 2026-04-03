import CSpeechCore
import AudioCommon
import Foundation
import os

// MARK: - Pipeline Types

/// Voice pipeline mode.
public enum PipelineMode: Int {
    /// Full voice agent: audio → VAD → STT → LLM → TTS → audio
    case voicePipeline = 0
    /// Speech-to-text only: audio → VAD → STT → text
    case transcribeOnly = 1
    /// Echo test: audio → VAD → STT → TTS → audio
    case echo = 2
}

/// Pipeline state.
public enum PipelineState: Int {
    case idle = 0
    case listening = 1
    case transcribing = 2
    case thinking = 3
    case speaking = 4
}

/// Events emitted by the voice pipeline.
public enum PipelineEvent {
    case sessionCreated
    case speechStarted
    case speechEnded
    case transcriptionCompleted(text: String, language: String?, confidence: Float)
    case responseCreated
    case responseInterrupted
    case responseAudioDelta(samples: [Float])
    case responseDone
    case toolCallStarted(name: String)
    case toolCallCompleted(name: String, result: String)
    case error(String)
}

// PipelineTool is defined in AudioCommon/PipelineLLM.swift

/// Pipeline configuration.
public struct PipelineConfig {
    public var vadOnset: Float = 0.5
    public var vadOffset: Float = 0.35
    public var minSpeechDuration: Float = 0.25
    public var minSilenceDuration: Float = 0.1
    public var allowInterruptions: Bool = true
    public var minInterruptionDuration: Float = 1.0
    public var interruptionRecoveryTimeout: Float = 0.4
    public var maxUtteranceDuration: Float = 15.0
    public var preSpeechBufferDuration: Float = 0.6
    /// Max TTS response duration (seconds). Prevents hallucination loops. 0 = unlimited.
    public var maxResponseDuration: Float = 10.0
    /// Post-playback guard (seconds). Delay after resume_listening() to let AEC settle.
    public var postPlaybackGuard: Float = 0.3
    /// Start STT on first silence frame instead of waiting for silence confirmation (~0.6s savings).
    public var eagerSTT: Bool = true
    /// Minimum time in PendingSilence (seconds) before firing eager STT. Filters mid-sentence pauses.
    public var eagerSTTDelay: Float = 0.3
    /// Run dummy STT transcription at pipeline start to warm up Neural Engine.
    public var warmupSTT: Bool = true
    public var language: String = ""
    public var mode: PipelineMode = .echo

    public static let `default` = PipelineConfig()

    public init() {}
}

// PipelineLLM and MessageRole are defined in AudioCommon.
// Re-exported here for backward compatibility.
// import AudioCommon to access them directly.

// MARK: - Bridge Classes

/// Bridges a SpeechRecognitionModel to the C vtable.
private final class STTBridge {
    let model: SpeechRecognitionModel
    var lastText: [CChar] = []  // keeps C string alive between calls
    var lastLanguage: [CChar] = []  // keeps language C string alive
    init(_ model: SpeechRecognitionModel) { self.model = model }
}

/// Bridges a PipelineTool to the C callback.
private final class ToolBridge {
    let tool: PipelineTool
    var lastResult: [CChar] = []

    init(_ tool: PipelineTool) { self.tool = tool }
}

/// Bridges a SpeechGenerationModel to the C vtable.
private final class TTSBridge {
    let model: SpeechGenerationModel
    private let _cancelled = OSAllocatedUnfairLock(initialState: false)
    var cancelled: Bool {
        get { _cancelled.withLock { $0 } }
        set { _cancelled.withLock { $0 = newValue } }
    }

    init(_ model: SpeechGenerationModel) { self.model = model }
}

/// Bridges a StreamingVADProvider to the C vtable.
private final class VADBridge {
    let model: StreamingVADProvider

    init(_ model: StreamingVADProvider) { self.model = model }
}

/// Bridges a PipelineLLM to the C vtable.
private final class LLMBridge {
    let model: PipelineLLM

    init(_ model: PipelineLLM) { self.model = model }
}

/// Bridges the event callback.
private final class EventBridge {
    let handler: (PipelineEvent) -> Void
    var sttBridge: STTBridge?

    init(_ handler: @escaping (PipelineEvent) -> Void) { self.handler = handler }
}

// MARK: - VoicePipeline

/// Swift wrapper around speech-core's C voice pipeline.
///
/// Connects Swift ML models (STT, TTS, VAD, LLM) to the C++ pipeline engine
/// via the C vtable FFI. The pipeline manages the full conversational loop:
///
/// ```
/// audio → VAD → STT → LLM → TTS → audio
/// ```
///
/// Usage:
/// ```swift
/// let pipeline = VoicePipeline(
///     stt: asrModel,
///     tts: ttsModel,
///     vad: sileroVAD,
///     config: .init(mode: .echo),
///     onEvent: { event in print(event) }
/// )
/// pipeline.start()
/// pipeline.pushAudio(micSamples)
/// ```
public final class VoicePipeline {

    private var handle: sc_pipeline_t?

    // Bridges must stay alive for the lifetime of the pipeline
    private let sttBridge: STTBridge
    private let ttsBridge: TTSBridge
    private let vadBridge: VADBridge
    private var llmBridge: LLMBridge?
    private let eventBridge: EventBridge

    /// Create a voice pipeline.
    ///
    /// - Parameters:
    ///   - stt: Speech recognition model (implements `SpeechRecognitionModel`)
    ///   - tts: Speech generation model (implements `SpeechGenerationModel`)
    ///   - vad: Streaming VAD model (implements `StreamingVADProvider`)
    ///   - llm: Language model (optional, required for `.voicePipeline` mode)
    ///   - config: Pipeline configuration
    ///   - onEvent: Event callback
    public init(
        stt: SpeechRecognitionModel,
        tts: SpeechGenerationModel,
        vad: StreamingVADProvider,
        llm: PipelineLLM? = nil,
        config: PipelineConfig = .default,
        onEvent: @escaping (PipelineEvent) -> Void
    ) {
        self.sttBridge = STTBridge(stt)
        self.ttsBridge = TTSBridge(tts)
        self.vadBridge = VADBridge(vad)
        self.eventBridge = EventBridge(onEvent)
        self.eventBridge.sttBridge = self.sttBridge
        if let llm { self.llmBridge = LLMBridge(llm) }

        let sttVtable = Self.makeSTTVtable(sttBridge)
        let ttsVtable = Self.makeTTSVtable(ttsBridge)
        let vadVtable = Self.makeVADVtable(vadBridge)

        var cConfig = sc_config_default()
        cConfig.vad_onset = config.vadOnset
        cConfig.vad_offset = config.vadOffset
        cConfig.min_speech_duration = config.minSpeechDuration
        cConfig.min_silence_duration = config.minSilenceDuration
        cConfig.allow_interruptions = config.allowInterruptions
        cConfig.min_interruption_duration = config.minInterruptionDuration
        cConfig.interruption_recovery_timeout = config.interruptionRecoveryTimeout
        cConfig.max_utterance_duration = config.maxUtteranceDuration
        cConfig.pre_speech_buffer_duration = config.preSpeechBufferDuration
        cConfig.max_response_duration = config.maxResponseDuration
        cConfig.post_playback_guard = config.postPlaybackGuard
        cConfig.eager_stt = config.eagerSTT
        cConfig.eager_stt_delay = config.eagerSTTDelay
        cConfig.warmup_stt = config.warmupSTT
        cConfig.mode = sc_mode_t(rawValue: UInt32(config.mode.rawValue))

        let eventCtx = Unmanaged.passRetained(eventBridge).toOpaque()

        if let llmBridge {
            var llmVtable = Self.makeLLMVtable(llmBridge)
            config.language.withCString { langPtr in
                cConfig.language = langPtr
                handle = sc_pipeline_create(
                    sttVtable, ttsVtable, &llmVtable, vadVtable,
                    cConfig, Self.eventCallback, eventCtx)
            }
        } else {
            config.language.withCString { langPtr in
                cConfig.language = langPtr
                handle = sc_pipeline_create(
                    sttVtable, ttsVtable, nil, vadVtable,
                    cConfig, Self.eventCallback, eventCtx)
            }
        }
    }

    deinit {
        if handle != nil {
            sc_pipeline_stop(handle)
            sc_pipeline_destroy(handle)
        }
        Unmanaged.passUnretained(sttBridge).release()
        Unmanaged.passUnretained(ttsBridge).release()
        Unmanaged.passUnretained(vadBridge).release()
        if let llmBridge {
            Unmanaged.passUnretained(llmBridge).release()
        }
        Unmanaged.passUnretained(eventBridge).release()
        for bridge in toolBridges {
            Unmanaged.passUnretained(bridge).release()
        }
    }

    // MARK: - Public API

    public func start() {
        sc_pipeline_start(handle)
    }

    public func stop() {
        sc_pipeline_stop(handle)
    }

    /// Feed microphone audio (Float32 PCM at VAD sample rate).
    public func pushAudio(_ samples: [Float]) {
        samples.withUnsafeBufferPointer { buf in
            sc_pipeline_push_audio(handle, buf.baseAddress, buf.count)
        }
    }

    /// Signal that response playback has finished.
    /// Transitions from speaking back to idle.
    public func resumeListening() {
        sc_pipeline_resume_listening(handle)
    }

    /// Inject text directly — bypasses STT, sent to LLM.
    public func pushText(_ text: String) {
        sc_pipeline_push_text(handle, text)
    }

    public var state: PipelineState {
        PipelineState(rawValue: Int(sc_pipeline_state(handle).rawValue)) ?? .idle
    }

    public var isRunning: Bool {
        sc_pipeline_is_running(handle)
    }

    // MARK: - Tool Calling

    private var toolBridges: [ToolBridge] = []

    /// Register tools that the LLM can invoke during conversation.
    /// Must be called before `start()`.
    public func setTools(_ tools: [PipelineTool]) {
        sc_pipeline_clear_tools(handle)
        toolBridges.removeAll()

        for tool in tools {
            let bridge = ToolBridge(tool)
            toolBridges.append(bridge)

            let ctx = Unmanaged.passRetained(bridge).toOpaque()
            tool.name.withCString { namePtr in
                tool.description.withCString { descPtr in
                    var def = sc_tool_definition_t()
                    def.name = namePtr
                    def.description = descPtr
                    def.triggers = nil
                    def.handler = { namePtr, argsPtr, ctx in
                        guard let ctx else { return nil }
                        let bridge = Unmanaged<ToolBridge>.fromOpaque(ctx).takeUnretainedValue()
                        let args = argsPtr.map { String(cString: $0) } ?? ""
                        let result = bridge.tool.handler(args)
                        bridge.lastResult = Array(result.utf8CString)
                        return bridge.lastResult.withUnsafeBufferPointer { $0.baseAddress }
                    }
                    def.handler_context = ctx
                    def.command = nil
                    def.timeout = 0
                    def.cooldown = Int32(tool.cooldown)
                    sc_pipeline_add_tool(handle, def)
                }
            }
        }
    }

    // MARK: - Event Callback

    private static let eventCallback: sc_event_fn = { eventPtr, ctx in
        guard let eventPtr, let ctx else { return }
        let bridge = Unmanaged<EventBridge>.fromOpaque(ctx).takeUnretainedValue()
        let e = eventPtr.pointee

        let event: PipelineEvent
        switch e.type {
        case SC_EVENT_SESSION_CREATED:
            event = .sessionCreated
        case SC_EVENT_SPEECH_STARTED:
            event = .speechStarted
        case SC_EVENT_SPEECH_ENDED:
            event = .speechEnded
        case SC_EVENT_TRANSCRIPTION_COMPLETED:
            let text = e.text.map { String(cString: $0) } ?? ""
            // Read detected language from STT bridge (set during transcribe callback)
            var lang: String?
            if let sttBridge = bridge.sttBridge, !sttBridge.lastLanguage.isEmpty {
                let langStr = String(cString: sttBridge.lastLanguage)
                if !langStr.isEmpty { lang = langStr }
            }
            event = .transcriptionCompleted(text: text, language: lang, confidence: e.confidence)
        case SC_EVENT_RESPONSE_CREATED:
            event = .responseCreated
        case SC_EVENT_RESPONSE_INTERRUPTED:
            event = .responseInterrupted
        case SC_EVENT_RESPONSE_AUDIO_DELTA:
            event = .responseAudioDelta(samples: pcm16ToFloat32(e.audio_data, count: e.audio_data_length))
        case SC_EVENT_RESPONSE_DONE:
            event = .responseDone
        case SC_EVENT_TOOL_CALL_STARTED:
            let name = e.text.map { String(cString: $0) } ?? ""
            event = .toolCallStarted(name: name)
        case SC_EVENT_TOOL_CALL_COMPLETED:
            // text contains "tool_name: result" for completed events
            let text = e.text.map { String(cString: $0) } ?? ""
            let parts = text.split(separator: ":", maxSplits: 1)
            let name = parts.first.map(String.init) ?? ""
            let result = parts.count > 1 ? String(parts[1]).trimmingCharacters(in: .whitespaces) : ""
            event = .toolCallCompleted(name: name, result: result)
        case SC_EVENT_ERROR:
            let text = e.text.map { String(cString: $0) } ?? "Unknown error"
            event = .error(text)
        default:
            return
        }

        bridge.handler(event)
    }

    // MARK: - STT Vtable

    private static func makeSTTVtable(_ bridge: STTBridge) -> sc_stt_vtable_t {
        let ctx = Unmanaged.passRetained(bridge).toOpaque()
        return sc_stt_vtable_t(
            context: ctx,
            transcribe: { ctx, audio, length, sampleRate in
                let bridge = Unmanaged<STTBridge>.fromOpaque(ctx!).takeUnretainedValue()
                let samples = Array(UnsafeBufferPointer(start: audio, count: length))
                let duration = Double(length) / Double(sampleRate)
                let rms = sqrt(samples.map { $0 * $0 }.reduce(0, +) / Float(max(length, 1)))
                AudioLog.pipeline.info("STT input: \(length) samples, \(String(format: "%.2f", duration))s, RMS=\(String(format: "%.4f", rms))")
                let result = bridge.model.transcribeWithLanguage(
                    audio: samples,
                    sampleRate: Int(sampleRate),
                    language: nil)
                // Store as C strings on bridge — pointers valid until next transcribe call
                bridge.lastText = Array(result.text.utf8CString)
                bridge.lastLanguage = Array((result.language ?? "").utf8CString)
                let textPtr = bridge.lastText.withUnsafeBufferPointer { $0.baseAddress }
                let langPtr = bridge.lastLanguage.withUnsafeBufferPointer { $0.baseAddress }
                return sc_transcription_result_t(
                    text: textPtr,
                    language: langPtr,
                    confidence: result.confidence,
                    start_time: 0,
                    end_time: 0)
            },
            input_sample_rate: { ctx in
                let bridge = Unmanaged<STTBridge>.fromOpaque(ctx!).takeUnretainedValue()
                return Int32(bridge.model.inputSampleRate)
            },
            begin_stream: nil,
            push_chunk: nil,
            flush_stream: nil,
            end_stream: nil,
            cancel_stream: nil
        )
    }

    // MARK: - TTS Vtable

    private static func makeTTSVtable(_ bridge: TTSBridge) -> sc_tts_vtable_t {
        let ctx = Unmanaged.passRetained(bridge).toOpaque()
        return sc_tts_vtable_t(
            context: ctx,
            synthesize: { ctx, text, language, onChunk, chunkCtx in
                let bridge = Unmanaged<TTSBridge>.fromOpaque(ctx!).takeUnretainedValue()
                bridge.cancelled = false
                let textStr = String(cString: text!)
                let langStr = language.map { String(cString: $0) } ?? ""
                AudioLog.pipeline.info("TTS synthesize: text='\(textStr)', language='\(langStr)'")

                // Single TTS call for the full text — consistent voice, no
                // chunk boundary artifacts (with noChunking streaming config).
                let sem = DispatchSemaphore(value: 0)
                DispatchQueue.global(qos: .userInitiated).async {
                    let group = DispatchGroup()
                    group.enter()
                    Task {
                        defer { group.leave() }
                        do {
                            let stream = bridge.model.generateStream(
                                text: textStr, language: langStr.isEmpty ? nil : langStr)
                            var sentFinal = false
                            for try await chunk in stream {
                                guard !bridge.cancelled else { break }
                                let isFinal = chunk.isFinal
                                chunk.samples.withUnsafeBufferPointer { buf in
                                    onChunk?(buf.baseAddress, buf.count, isFinal, chunkCtx)
                                }
                                if isFinal { sentFinal = true; break }
                            }
                            if !sentFinal && !bridge.cancelled {
                                onChunk?(nil, 0, true, chunkCtx)
                            }
                        } catch {
                            onChunk?(nil, 0, true, chunkCtx)
                        }
                    }
                    group.wait()
                    sem.signal()
                }
                sem.wait()
            },
            output_sample_rate: { ctx in
                let bridge = Unmanaged<TTSBridge>.fromOpaque(ctx!).takeUnretainedValue()
                return Int32(bridge.model.sampleRate)
            },
            cancel: { ctx in
                let bridge = Unmanaged<TTSBridge>.fromOpaque(ctx!).takeUnretainedValue()
                bridge.cancelled = true
            }
        )
    }

    // MARK: - VAD Vtable

    private static func makeVADVtable(_ bridge: VADBridge) -> sc_vad_vtable_t {
        let ctx = Unmanaged.passRetained(bridge).toOpaque()
        return sc_vad_vtable_t(
            context: ctx,
            process_chunk: { ctx, samples, length in
                let bridge = Unmanaged<VADBridge>.fromOpaque(ctx!).takeUnretainedValue()
                let chunk = Array(UnsafeBufferPointer(start: samples, count: length))
                return bridge.model.processChunk(chunk)
            },
            reset: { ctx in
                let bridge = Unmanaged<VADBridge>.fromOpaque(ctx!).takeUnretainedValue()
                bridge.model.resetState()
            },
            input_sample_rate: { ctx in
                let bridge = Unmanaged<VADBridge>.fromOpaque(ctx!).takeUnretainedValue()
                return Int32(bridge.model.inputSampleRate)
            },
            chunk_size: { ctx in
                let bridge = Unmanaged<VADBridge>.fromOpaque(ctx!).takeUnretainedValue()
                return bridge.model.chunkSize
            }
        )
    }

    // MARK: - LLM Vtable

    private static func makeLLMVtable(_ bridge: LLMBridge) -> sc_llm_vtable_t {
        let ctx = Unmanaged.passRetained(bridge).toOpaque()
        return sc_llm_vtable_t(
            context: ctx,
            chat: { ctx, messages, count, onToken, tokenCtx in
                let bridge = Unmanaged<LLMBridge>.fromOpaque(ctx!).takeUnretainedValue()

                // Convert C messages to Swift
                var swiftMessages: [(role: MessageRole, content: String)] = []
                for i in 0..<count {
                    let msg = messages![i]
                    let role = MessageRole(rawValue: Int(msg.role.rawValue)) ?? .user
                    let content = msg.content.map { String(cString: $0) } ?? ""
                    swiftMessages.append((role: role, content: content))
                }

                bridge.model.chat(messages: swiftMessages) { token, isFinal in
                    token.withCString { cstr in
                        onToken?(cstr, isFinal, tokenCtx)
                    }
                }
            },
            cancel: { ctx in
                let bridge = Unmanaged<LLMBridge>.fromOpaque(ctx!).takeUnretainedValue()
                bridge.model.cancel()
            },
            count_tokens: nil  // Token counting not implemented yet
        )
    }

    // MARK: - PCM Conversion

    /// Convert PCM16-LE bytes to Float32 samples.
    private static func pcm16ToFloat32(_ data: UnsafePointer<UInt8>?, count: Int) -> [Float] {
        guard let data, count >= 2 else { return [] }
        let sampleCount = count / 2
        return (0..<sampleCount).map { i in
            let lo = UInt16(data[i * 2])
            let hi = UInt16(data[i * 2 + 1])
            let raw = Int16(bitPattern: lo | (hi << 8))
            return Float(raw) / 32768.0
        }
    }
}

/// Split text into sentences for per-sentence TTS synthesis.
private func splitTTSSentences(_ text: String) -> [String] {
    var sentences = [String]()
    text.enumerateSubstrings(in: text.startIndex..., options: .bySentences) { sub, _, _, _ in
        if let s = sub?.trimmingCharacters(in: .whitespaces), !s.isEmpty {
            sentences.append(s)
        }
    }
    if sentences.isEmpty && !text.trimmingCharacters(in: .whitespaces).isEmpty {
        sentences.append(text)
    }
    return sentences
}
