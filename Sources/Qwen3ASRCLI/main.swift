import Foundation
import Qwen3ASR

// Entry point
Task {
    // Get the audio file from command line argument
    let args = CommandLine.arguments

    guard args.count >= 2 else {
        print("Usage: qwen3-asr-cli <audio-file>")
        print("Example: qwen3-asr-cli /path/to/audio.wav")
        exit(1)
    }

    let audioPath = args[1]

    print("Loading model...")
    do {
        let model = try await Qwen3ASRModel.fromPretrained(
            modelId: "mlx-community/Qwen3-ASR-0.6B-4bit"
        ) { progress, status in
            print("  [\(Int(progress * 100))%] \(status)")
        }

        print("Loading audio: \(audioPath)")
        let audio = try AudioFileLoader.load(url: URL(fileURLWithPath: audioPath), targetSampleRate: 24000)
        print("  Loaded \(audio.count) samples (\(String(format: "%.2f", Double(audio.count) / 24000.0))s)")

        print("Transcribing...")
        let result = model.transcribe(
            audio: audio,
            sampleRate: 24000
        )
        print("Result: \(result)")
        exit(0)
    } catch {
        print("Error: \(error)")
        exit(1)
    }
}

// Keep the main thread alive
RunLoop.main.run()
