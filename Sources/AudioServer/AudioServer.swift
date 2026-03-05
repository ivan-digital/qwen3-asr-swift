import Foundation
import Hummingbird
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
        let app = Application(
            router: router,
            configuration: .init(address: .hostname(host, port: port)))
        try await app.run()
    }

    public func preloadModels() async throws {
        _ = try await state.loadASR()
        _ = try await state.loadTTS()
        _ = try await state.loadPersonaPlex()
        _ = try await state.loadEnhancer()
    }

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
