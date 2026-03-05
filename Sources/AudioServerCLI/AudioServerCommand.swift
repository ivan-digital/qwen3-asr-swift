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

    @Flag(name: .long, help: "Load all models on startup (slower start, faster first request)")
    var preload: Bool = false

    func run() async throws {
        let server = AudioServer(host: host, port: port)

        if preload {
            print("Preloading models...")
            try await server.preloadModels()
            print("All models loaded.")
        }

        print("Starting server on http://\(host):\(port)")
        print("Endpoints:")
        print("  POST /transcribe  - Speech-to-text (WAV body or JSON with audio_base64)")
        print("  POST /speak       - Text-to-speech (JSON: {text, engine?, language?})")
        print("  POST /respond     - Speech-to-speech (WAV body, voice/max_steps via query)")
        print("  POST /enhance     - Speech enhancement (WAV body)")
        print("  GET  /health      - Health check")

        try await server.run()
    }
}
