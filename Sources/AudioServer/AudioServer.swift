import Foundation
import Hummingbird
import HummingbirdCore
import HummingbirdWebSocket
import NIOCore
import Qwen3ASR
import Qwen3TTS
import CosyVoiceTTS
import PersonaPlex
import SpeechEnhancement
import AudioCommon

// MARK: - Server

public struct AudioServer {
    let state: ModelState
    let host: String
    let port: Int

    public init(host: String = "127.0.0.1", port: Int = 8080, preload: Bool = false) {
        self.state = ModelState()
        self.host = host
        self.port = port
    }

    public func run() async throws {
        let router = buildRouter()
        let state = self.state
        let wsConfig = WebSocketServerConfiguration(maxFrameSize: 1 << 24)  // 16 MB max frame
        let wsServer: HTTPServerBuilder = .http1WebSocketUpgrade(configuration: wsConfig) { head, _, _ in
            let path = head.path ?? ""
            guard path == "/v1/realtime" else { return .dontUpgrade }
            return .upgrade([:]) { inbound, outbound, _ in
                try await handleRealtimeWS(inbound: inbound, outbound: outbound, state: state)
            }
        }
        let app = Application(
            router: router,
            server: wsServer,
            configuration: .init(address: .hostname(host, port: port)))
        try await app.run()
    }

    public func preloadModels() async throws {
        _ = try await state.loadASR()
        _ = try await state.loadTTS()
        _ = try await state.loadPersonaPlex()
        _ = try await state.loadEnhancer()
    }

    // Individual preload entry points so callers can warm only the
    // models they actually use. Loading every model at boot (the old
    // --preload behaviour) can saturate the Metal wired pool on smaller
    // devices and deadlock the allocator on the first real request.
    public func preloadASR() async throws {
        _ = try await state.loadASR()
    }

    public func preloadTTS() async throws {
        _ = try await state.loadTTS()
    }

    public func preloadPersonaPlex() async throws {
        _ = try await state.loadPersonaPlex()
    }

    public func preloadEnhancer() async throws {
        _ = try await state.loadEnhancer()
    }

    // MARK: - HTTP Routes

    func buildRouter() -> Router<BasicRequestContext> {
        let router = Router()
        let state = self.state

        router.get("/health") { _, _ in
            Response(
                status: .ok,
                headers: [.contentType: "application/json"],
                body: .init(byteBuffer: .init(string: "{\"status\":\"ok\"}")))
        }

        router.post("/transcribe") { request, _ in
            let body = try await request.body.collect(upTo: 50 * 1024 * 1024)
            let params = try RequestParams.parse(body, contentType: request.headers[.contentType])

            guard let audioData = params.audioData else {
                return errorResponse("Missing audio data", status: .badRequest)
            }

            let sampleRate = params.int("sample_rate") ?? 16000
            let model = try await state.loadASR()
            let audio = try decodeWAVData(audioData, targetSampleRate: sampleRate)
            let text = model.transcribe(audio: audio, sampleRate: sampleRate)

            return jsonResponse([
                "text": text,
                "duration": round(Double(audio.count) / Double(sampleRate) * 100) / 100
            ] as [String: Any])
        }

        router.post("/speak") { request, _ in
            let body = try await request.body.collect(upTo: 1024 * 1024)
            let params = try RequestParams.parse(body, contentType: request.headers[.contentType])

            guard let text = params.text else {
                return errorResponse("Missing 'text' field", status: .badRequest)
            }

            let engine = params.string("engine") ?? "cosyvoice"
            let language = params.string("language") ?? "english"

            let samples: [Float]

            if engine == "qwen3" {
                let model = try await state.loadTTS()
                samples = model.synthesize(text: text, language: language)
            } else {
                let model = try await state.loadCosyVoice()
                samples = model.synthesize(text: text, language: language)
            }

            let wavData = try encodeWAV(samples: samples, sampleRate: 24000)
            return Response(
                status: .ok,
                headers: [.contentType: "audio/wav"],
                body: .init(byteBuffer: .init(data: wavData)))
        }

        router.post("/respond") { request, _ in
            let body = try await request.body.collect(upTo: 50 * 1024 * 1024)
            let params = try RequestParams.parse(body, contentType: request.headers[.contentType])

            guard let audioData = params.audioData else {
                return errorResponse("Missing audio data", status: .badRequest)
            }

            let voiceName = params.string("voice") ?? "NATM0"
            let maxSteps = params.int("max_steps") ?? 200

            guard let voice = PersonaPlexVoice(rawValue: voiceName) else {
                return errorResponse("Unknown voice: \(voiceName)", status: .badRequest)
            }

            let model = try await state.loadPersonaPlex()
            let audio = try decodeWAVData(audioData, targetSampleRate: 24000)
            let result = model.respond(
                userAudio: audio,
                voice: voice,
                maxSteps: maxSteps)

            var transcript: String?
            if let dec = state.spmDecoder, !result.textTokens.isEmpty {
                transcript = dec.decode(result.textTokens)
            }

            let wavData = try encodeWAV(samples: result.audio, sampleRate: 24000)
            let duration = Double(result.audio.count) / 24000.0

            if params.string("format") == "json" {
                var json: [String: Any] = [
                    "duration": round(duration * 100) / 100,
                    "text_tokens": result.textTokens.count
                ]
                if let t = transcript { json["transcript"] = t }
                json["audio_base64"] = wavData.base64EncodedString()
                return jsonResponse(json)
            }

            return Response(
                status: .ok,
                headers: [.contentType: "audio/wav"],
                body: .init(byteBuffer: .init(data: wavData)))
        }

        router.post("/enhance") { request, _ in
            let body = try await request.body.collect(upTo: 50 * 1024 * 1024)
            let params = try RequestParams.parse(body, contentType: request.headers[.contentType])

            guard let audioData = params.audioData else {
                return errorResponse("Missing audio data", status: .badRequest)
            }

            let enhancer = try await state.loadEnhancer()
            let audio = try decodeWAVData(audioData, targetSampleRate: 48000)
            let enhanced = try enhancer.enhance(audio: audio, sampleRate: 48000)

            let wavData = try encodeWAV(samples: enhanced, sampleRate: 48000)
            return Response(
                status: .ok,
                headers: [.contentType: "audio/wav"],
                body: .init(byteBuffer: .init(data: wavData)))
        }

        return router
    }
}

// MARK: - Lazy Model State

final class ModelState: @unchecked Sendable {
    private var asr: Qwen3ASRModel?
    private var tts: Qwen3TTSModel?
    private var cosyvoice: CosyVoiceTTSModel?
    private var personaplex: PersonaPlexModel?
    private var enhancer: SpeechEnhancer?
    var spmDecoder: SentencePieceDecoder?

    func loadASR() async throws -> Qwen3ASRModel {
        if let m = asr { return m }
        print("[server] Loading Qwen3-ASR...")
        let m = try await Qwen3ASRModel.fromPretrained(progressHandler: logProgress)
        asr = m
        return m
    }

    func loadTTS() async throws -> Qwen3TTSModel {
        if let m = tts { return m }
        print("[server] Loading Qwen3-TTS...")
        let m = try await Qwen3TTSModel.fromPretrained(progressHandler: logProgress)
        tts = m
        return m
    }

    func loadCosyVoice() async throws -> CosyVoiceTTSModel {
        if let m = cosyvoice { return m }
        print("[server] Loading CosyVoice...")
        let m = try await CosyVoiceTTSModel.fromPretrained(progressHandler: logProgress)
        cosyvoice = m
        return m
    }

    func loadPersonaPlex() async throws -> PersonaPlexModel {
        if let m = personaplex { return m }
        print("[server] Loading PersonaPlex 7B...")
        let m = try await PersonaPlexModel.fromPretrained(progressHandler: logProgress)
        personaplex = m
        do {
            let cacheDir = try HuggingFaceDownloader.getCacheDirectory(
                for: "aufklarer/PersonaPlex-7B-MLX-4bit")
            let spmPath = cacheDir.appendingPathComponent("tokenizer_spm_32k_3.model").path
            if FileManager.default.fileExists(atPath: spmPath) {
                spmDecoder = try SentencePieceDecoder(modelPath: spmPath)
            }
        } catch {}
        return m
    }

    func loadEnhancer() async throws -> SpeechEnhancer {
        if let m = enhancer { return m }
        print("[server] Loading DeepFilterNet3...")
        let m = try await SpeechEnhancer.fromPretrained(progressHandler: logProgress)
        enhancer = m
        return m
    }
}

private func logProgress(_ progress: Double, _ status: String) {
    print("  [\(Int(progress * 100))%] \(status)")
}

// MARK: - OpenAI Realtime API Handler

/// Per-connection session state for the OpenAI Realtime protocol.
private final class RealtimeSession {
    var engine: String = "cosyvoice"
    var language: String = "english"
    var inputAudioBuffer = Data()
    var inputSampleRate: Int = 24000
}

/// Handle /v1/realtime: OpenAI Realtime API compatible protocol.
/// All messages are JSON with a "type" field. Audio is base64-encoded PCM16 24kHz.
func handleRealtimeWS(
    inbound: WebSocketInboundStream,
    outbound: WebSocketOutboundWriter,
    state: ModelState
) async throws {
    let session = RealtimeSession()
    let sessionId = UUID().uuidString

    // Send session.created
    try await outbound.write(.text(formatJSON([
        "type": "session.created",
        "session": [
            "id": sessionId,
            "model": "qwen3-speech",
            "modalities": ["audio", "text"],
            "input_audio_format": "pcm16",
            "output_audio_format": "pcm16"
        ]
    ] as [String: Any])))

    for try await message in inbound.messages(maxSize: 50 * 1024 * 1024) {
        guard case .text(let string) = message else { continue }
        guard let jsonData = string.data(using: .utf8),
              let json = try? JSONSerialization.jsonObject(with: jsonData) as? [String: Any],
              let eventType = json["type"] as? String else {
            try await sendRealtimeError(outbound: outbound, message: "Invalid message format")
            continue
        }

        switch eventType {

        case "session.update":
            if let sessionConfig = json["session"] as? [String: Any] {
                if let engine = sessionConfig["engine"] as? String {
                    session.engine = engine
                }
                if let lang = sessionConfig["language"] as? String {
                    session.language = lang
                }
                if let fmt = sessionConfig["input_audio_format"] as? String, fmt == "pcm16" {
                    session.inputSampleRate = 24000
                }
            }
            try await outbound.write(.text(formatJSON([
                "type": "session.updated",
                "session": [
                    "id": sessionId,
                    "engine": session.engine,
                    "language": session.language,
                    "input_audio_format": "pcm16",
                    "output_audio_format": "pcm16"
                ]
            ] as [String: Any])))

        case "input_audio_buffer.append":
            guard let audioB64 = json["audio"] as? String,
                  let audioData = Data(base64Encoded: audioB64) else {
                try await sendRealtimeError(outbound: outbound, message: "Missing or invalid 'audio' field")
                continue
            }
            session.inputAudioBuffer.append(audioData)

        case "input_audio_buffer.clear":
            session.inputAudioBuffer.removeAll()
            try await outbound.write(.text(formatJSON([
                "type": "input_audio_buffer.cleared"
            ])))

        case "input_audio_buffer.commit":
            let audioData = session.inputAudioBuffer
            session.inputAudioBuffer.removeAll()

            guard !audioData.isEmpty else {
                try await sendRealtimeError(outbound: outbound, message: "Audio buffer is empty")
                continue
            }

            let itemId = UUID().uuidString
            try await outbound.write(.text(formatJSON([
                "type": "input_audio_buffer.committed",
                "item_id": itemId
            ])))

            // Transcribe: PCM16 24kHz → resample to 16kHz for ASR
            let floats = pcm16LEToFloat(audioData)
            let audio16k = resample(floats, from: session.inputSampleRate, to: 16000)
            let model = try await state.loadASR()
            let text = model.transcribe(audio: audio16k, sampleRate: 16000)

            let responseId = UUID().uuidString
            try await outbound.write(.text(formatJSON([
                "type": "conversation.item.input_audio_transcription.completed",
                "item_id": itemId,
                "transcript": text
            ])))

            // Also emit as a response for clients expecting response.* events
            try await outbound.write(.text(formatJSON([
                "type": "response.created",
                "response": ["id": responseId, "status": "in_progress"]
            ] as [String: Any])))
            try await outbound.write(.text(formatJSON([
                "type": "response.audio_transcript.delta",
                "response_id": responseId,
                "delta": text
            ])))
            try await outbound.write(.text(formatJSON([
                "type": "response.audio_transcript.done",
                "response_id": responseId,
                "transcript": text
            ])))
            try await outbound.write(.text(formatJSON([
                "type": "response.done",
                "response": ["id": responseId, "status": "completed"]
            ] as [String: Any])))

        case "response.create":
            let input = json["response"] as? [String: Any]
            let instructions = input?["instructions"] as? String
            let modalities = input?["modalities"] as? [String] ?? ["audio", "text"]
            let responseId = UUID().uuidString

            // If there's text to speak (from instructions or input items)
            var textToSpeak: String?

            if let instructions = instructions, !instructions.isEmpty {
                textToSpeak = instructions
            }

            // Check for input items with text content
            if textToSpeak == nil, let inputItems = input?["input"] as? [[String: Any]] {
                for item in inputItems {
                    if let content = item["content"] as? [[String: Any]] {
                        for part in content {
                            if part["type"] as? String == "input_text",
                               let text = part["text"] as? String {
                                textToSpeak = text
                            }
                        }
                    }
                }
            }

            // Also check conversation.item.create pattern — text in input
            if textToSpeak == nil, let text = input?["text"] as? String {
                textToSpeak = text
            }

            guard let text = textToSpeak else {
                try await sendRealtimeError(outbound: outbound, message: "No text to synthesize")
                continue
            }

            try await outbound.write(.text(formatJSON([
                "type": "response.created",
                "response": ["id": responseId, "status": "in_progress"]
            ] as [String: Any])))

            // Stream TTS audio as base64 PCM16 24kHz chunks
            let engine = (input?["engine"] as? String) ?? session.engine
            let language = (input?["language"] as? String) ?? session.language
            var totalSamples = 0

            if engine == "qwen3" {
                let model = try await state.loadTTS()
                let stream = model.synthesizeStream(text: text, language: language)
                for try await chunk in stream {
                    if !chunk.samples.isEmpty {
                        totalSamples += chunk.samples.count
                        let pcm = floatToPCM16LE(chunk.samples)
                        try await outbound.write(.text(formatJSON([
                            "type": "response.audio.delta",
                            "response_id": responseId,
                            "delta": pcm.base64EncodedString()
                        ])))
                    }
                }
            } else {
                let model = try await state.loadCosyVoice()
                let stream = model.synthesizeStream(text: text, language: language)
                for try await chunk in stream {
                    if !chunk.samples.isEmpty {
                        totalSamples += chunk.samples.count
                        let pcm = floatToPCM16LE(chunk.samples)
                        try await outbound.write(.text(formatJSON([
                            "type": "response.audio.delta",
                            "response_id": responseId,
                            "delta": pcm.base64EncodedString()
                        ])))
                    }
                }
            }

            if modalities.contains("text") {
                try await outbound.write(.text(formatJSON([
                    "type": "response.audio_transcript.done",
                    "response_id": responseId,
                    "transcript": text
                ])))
            }

            try await outbound.write(.text(formatJSON([
                "type": "response.audio.done",
                "response_id": responseId
            ])))

            let duration = Double(totalSamples) / 24000.0
            try await outbound.write(.text(formatJSON([
                "type": "response.done",
                "response": [
                    "id": responseId,
                    "status": "completed",
                    "usage": [
                        "total_tokens": 0,
                        "output_tokens": 0
                    ],
                    "output": [
                        ["type": "audio", "duration": round(duration * 100) / 100, "sample_rate": 24000]
                    ]
                ]
            ] as [String: Any])))

        case "conversation.item.create":
            // Accept text items for TTS via response.create flow
            if let item = json["item"] as? [String: Any],
               let content = item["content"] as? [[String: Any]] {
                for part in content {
                    if part["type"] as? String == "input_text" || part["type"] as? String == "text",
                       let _ = part["text"] as? String {
                        try await outbound.write(.text(formatJSON([
                            "type": "conversation.item.created",
                            "item": item
                        ] as [String: Any])))
                    }
                }
            }

        default:
            try await sendRealtimeError(outbound: outbound,
                message: "Unknown event type: \(eventType)")
        }
    }
}

private func sendRealtimeError(outbound: WebSocketOutboundWriter, message: String) async throws {
    try await outbound.write(.text(formatJSON([
        "type": "error",
        "error": ["type": "invalid_request_error", "message": message]
    ] as [String: Any])))
}

/// Resample audio via AVAudioConverter (delegates to AudioFileLoader).
func resample(_ samples: [Float], from sourceSR: Int, to targetSR: Int) -> [Float] {
    AudioFileLoader.resample(samples, from: sourceSR, to: targetSR)
}

// MARK: - Request Parsing

struct RequestParams {
    var audioData: Data?
    var text: String?
    var fields: [String: String] = [:]

    func string(_ key: String) -> String? { fields[key] }
    func int(_ key: String) -> Int? { fields[key].flatMap(Int.init) }

    static func parse(_ body: ByteBuffer, contentType: String?) throws -> RequestParams {
        var params = RequestParams()

        if let ct = contentType, ct.contains("application/json") {
            let data = Data(buffer: body)
            if let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] {
                if let text = json["text"] as? String { params.text = text }
                if let b64 = json["audio_base64"] as? String {
                    params.audioData = Data(base64Encoded: b64)
                }
                for (k, v) in json {
                    if let s = v as? String { params.fields[k] = s }
                    else if let n = v as? Int { params.fields[k] = String(n) }
                    else if let n = v as? Double { params.fields[k] = String(n) }
                }
            }
            return params
        }

        // Raw audio body (WAV)
        let data = Data(buffer: body)
        if data.count > 44 {
            params.audioData = data
        }
        return params
    }
}

// MARK: - Audio Encoding/Decoding

func decodeWAVData(_ data: Data, targetSampleRate: Int) throws -> [Float] {
    let tmpURL = FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString + ".wav")
    try data.write(to: tmpURL)
    defer { try? FileManager.default.removeItem(at: tmpURL) }
    return try AudioFileLoader.load(url: tmpURL, targetSampleRate: targetSampleRate)
}

func encodeWAV(samples: [Float], sampleRate: Int) throws -> Data {
    let tmpURL = FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString + ".wav")
    try WAVWriter.write(samples: samples, sampleRate: sampleRate, to: tmpURL)
    defer { try? FileManager.default.removeItem(at: tmpURL) }
    return try Data(contentsOf: tmpURL)
}

// MARK: - Response Helpers

func jsonResponse(_ dict: [String: Any]) -> Response {
    let data = (try? JSONSerialization.data(
        withJSONObject: dict, options: [.prettyPrinted, .sortedKeys])) ?? Data()
    return Response(
        status: .ok,
        headers: [.contentType: "application/json"],
        body: .init(byteBuffer: .init(data: data)))
}

func errorResponse(_ message: String, status: HTTPResponse.Status) -> Response {
    let data = (try? JSONSerialization.data(
        withJSONObject: ["error": message], options: [])) ?? Data()
    return Response(
        status: status,
        headers: [.contentType: "application/json"],
        body: .init(byteBuffer: .init(data: data)))
}

// MARK: - PCM Conversion

func pcm16LEToFloat(_ data: Data) -> [Float] {
    let sampleCount = data.count / 2
    var result = [Float](repeating: 0, count: sampleCount)
    data.withUnsafeBytes { raw in
        let int16s = raw.bindMemory(to: Int16.self)
        for i in 0..<sampleCount {
            result[i] = Float(Int16(littleEndian: int16s[i])) / 32768.0
        }
    }
    return result
}

func floatToPCM16LE(_ samples: [Float]) -> Data {
    var data = Data(count: samples.count * 2)
    data.withUnsafeMutableBytes { raw in
        let int16s = raw.bindMemory(to: Int16.self)
        for i in 0..<samples.count {
            let clamped = max(-1.0, min(1.0, samples[i]))
            int16s[i] = Int16(clamped * 32767.0).littleEndian
        }
    }
    return data
}

func formatJSON(_ dict: [String: Any]) -> String {
    guard let data = try? JSONSerialization.data(withJSONObject: dict, options: [.sortedKeys]) else {
        return "{}"
    }
    return String(data: data, encoding: .utf8) ?? "{}"
}
