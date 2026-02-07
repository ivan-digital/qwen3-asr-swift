import Foundation
import MLX
import Qwen3ASR
#if canImport(Metal)
import Metal
#endif

private func runMain(audioPath: String) async throws {
    print("Loading model...")

    let model = try await Qwen3ASRModel.fromPretrained(
        modelId: "mlx-community/Qwen3-ASR-0.6B-4bit"
    ) { progress, status in
        print("  [\(Int(progress * 100))%] \(status)")
    }

    print("Loading audio: \(audioPath)")
    let audio = try AudioFileLoader.load(url: URL(fileURLWithPath: audioPath), targetSampleRate: 24000)
    let seconds = Double(audio.count) / 24000.0
    print(String(format: "  Loaded %d samples (%.2fs)", audio.count, seconds))

    print("Transcribing...")
    let result = model.transcribe(
        audio: audio,
        sampleRate: 24000
    )
    print("Result: \(result)")
}

private func requireMetalOrExit() {
    #if canImport(Metal)
    if MTLCreateSystemDefaultDevice() == nil {
        fputs("Error: Metal device is unavailable. MLX on macOS requires Metal.\n", stderr)
        fputs("If you are running inside a restricted sandbox/CI, run outside the sandbox or on a Mac with Metal enabled.\n", stderr)
        exit(1)
    }
    #endif
}

private func requireMLXMetallibOrExit() {
    guard let exeURL = Bundle.main.executableURL else { return }
    let binDir = exeURL.deletingLastPathComponent()

    let candidates: [URL] = [
        binDir.appendingPathComponent("mlx.metallib"),
        binDir.appendingPathComponent("default.metallib"),
        binDir.appendingPathComponent("Resources/mlx.metallib"),
        binDir.appendingPathComponent("Resources/default.metallib"),
    ]

    if candidates.contains(where: { FileManager.default.fileExists(atPath: $0.path) }) {
        return
    }

    fputs("Error: MLX Metal library (mlx.metallib) not found next to the executable.\n", stderr)
    fputs("Fix:\n", stderr)
    fputs("  1) Install Metal toolchain (once): xcodebuild -downloadComponent MetalToolchain\n", stderr)
    fputs("  2) From the repo root, run:\n", stderr)
    fputs("     swift build -c release --disable-sandbox\n", stderr)
    fputs("     ./scripts/build_mlx_metallib.sh release\n", stderr)
    exit(1)
}

// Entry point
Task {
    requireMetalOrExit()
    requireMLXMetallibOrExit()

    // Get the audio file from command line argument
    let args = CommandLine.arguments

    guard args.count >= 2 else {
        print("Usage: qwen3-asr-cli <audio-file>")
        print("Example: qwen3-asr-cli /path/to/audio.wav")
        exit(1)
    }

    let audioPath = args[1]
    let deviceOverride = ProcessInfo.processInfo.environment["QWEN3_ASR_DEVICE"]?.lowercased()

    do {
        if deviceOverride == "cpu" {
            try await Device.withDefaultDevice(.cpu) {
                try await runMain(audioPath: audioPath)
            }
        } else {
            try await runMain(audioPath: audioPath)
        }
        exit(0)
    } catch {
        print("Error: \(error)")
        exit(1)
    }
}

// Keep the main thread alive
RunLoop.main.run()
