import Foundation
import ArgumentParser
import AudioServer

@main
struct AudioServerCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "audio-server",
        abstract: "HTTP API server for speech models on Apple Silicon"
    )

    @Option(name: .long, help: "Host to bind (default: 127.0.0.1)")
    var host: String = "127.0.0.1"

    @Option(name: .long, help: "Port to bind (default: 8080)")
    var port: Int = 8080

    @Flag(name: .long, help: "Preload every model (ASR + TTS + PersonaPlex + Enhancer). Can saturate the Metal wired pool on smaller devices — prefer the per-model flags below unless you really need all four resident.")
    var preload: Bool = false

    @Flag(name: [.customLong("preload-asr")], help: "Preload Qwen3-ASR on startup (for /transcribe-only workloads).")
    var preloadAsr: Bool = false

    @Flag(name: [.customLong("preload-tts")], help: "Preload Qwen3-TTS on startup (for /speak workloads).")
    var preloadTts: Bool = false

    @Flag(name: [.customLong("preload-personaplex")], help: "Preload PersonaPlex 7B on startup (for /respond workloads).")
    var preloadPersonaPlex: Bool = false

    @Flag(name: [.customLong("preload-enhancer")], help: "Preload SpeechEnhancement on startup (for /enhance workloads).")
    var preloadEnhancer: Bool = false

    func run() async throws {
        let server = AudioServer(host: host, port: port)

        let loadAll = preload
        if loadAll || preloadAsr {
            print("Preloading Qwen3-ASR...")
            try await server.preloadASR()
        }
        if loadAll || preloadTts {
            print("Preloading Qwen3-TTS...")
            try await server.preloadTTS()
        }
        if loadAll || preloadPersonaPlex {
            print("Preloading PersonaPlex...")
            try await server.preloadPersonaPlex()
        }
        if loadAll || preloadEnhancer {
            print("Preloading SpeechEnhancement...")
            try await server.preloadEnhancer()
        }

        print("Starting server on http://\(host):\(port)")
        print("Endpoints:")
        print("  POST /transcribe     - Speech-to-text (WAV body or JSON with audio_base64)")
        print("  POST /speak          - Text-to-speech (JSON: {text, engine?, language?})")
        print("  POST /respond        - Speech-to-speech (WAV body, voice/max_steps via query)")
        print("  POST /enhance        - Speech enhancement (WAV body)")
        print("  GET  /health         - Health check")
        print("  WS   /v1/realtime    - OpenAI Realtime API (JSON events, base64 PCM16 audio)")

        try await server.run()
    }
}
