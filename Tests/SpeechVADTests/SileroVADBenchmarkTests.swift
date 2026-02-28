import XCTest
import MLX
@testable import SpeechVAD
import AudioCommon

/// Benchmarks comparing MLX vs CoreML Silero VAD latency.
///
/// Requires both model variants to be cached locally:
/// - `aufklarer/Silero-VAD-v5-MLX` for MLX
/// - `aufklarer/Silero-VAD-v5-CoreML` for CoreML
///
/// Run with: `swift test --filter SileroVADBenchmarkTests -c release`
final class SileroVADBenchmarkTests: XCTestCase {

    func testMLXvsCoreMLLatency() async throws {
        let audioURL = URL(fileURLWithPath: "Tests/Qwen3ASRTests/Resources/test_audio.wav")
        guard FileManager.default.fileExists(atPath: audioURL.path) else {
            throw XCTSkip("Test audio file not found")
        }

        let (rawSamples, sampleRate) = try AudioFileLoader.loadWAV(url: audioURL)
        let audio = sampleRate != 16000
            ? AudioFileLoader.resample(rawSamples, from: sampleRate, to: 16000)
            : rawSamples

        let chunkSize = SileroVADModel.chunkSize
        let numChunks = audio.count / chunkSize

        // Load both models
        let mlxModel: SileroVADModel
        do {
            mlxModel = try await SileroVADModel.fromPretrained(engine: .mlx)
        } catch {
            throw XCTSkip("MLX model not cached: \(error)")
        }

        let coremlModel: SileroVADModel
        do {
            coremlModel = try await SileroVADModel.fromPretrained(engine: .coreml)
        } catch {
            throw XCTSkip("CoreML model not cached: \(error)")
        }

        // Warmup: 3 chunks each
        let warmupChunks = min(3, numChunks)
        for i in 0..<warmupChunks {
            let chunk = Array(audio[i * chunkSize ..< (i + 1) * chunkSize])
            _ = mlxModel.processChunk(chunk)
            _ = coremlModel.processChunk(chunk)
        }
        mlxModel.resetState()
        coremlModel.resetState()

        // Benchmark MLX
        var mlxProbs = [Float]()
        let mlxStart = CFAbsoluteTimeGetCurrent()
        for i in 0..<numChunks {
            let chunk = Array(audio[i * chunkSize ..< (i + 1) * chunkSize])
            mlxProbs.append(mlxModel.processChunk(chunk))
        }
        let mlxTime = CFAbsoluteTimeGetCurrent() - mlxStart

        // Benchmark CoreML
        var coremlProbs = [Float]()
        let coremlStart = CFAbsoluteTimeGetCurrent()
        for i in 0..<numChunks {
            let chunk = Array(audio[i * chunkSize ..< (i + 1) * chunkSize])
            coremlProbs.append(coremlModel.processChunk(chunk))
        }
        let coremlTime = CFAbsoluteTimeGetCurrent() - coremlStart

        // Report results
        let audioDuration = Double(audio.count) / 16000.0
        let mlxPerChunk = mlxTime / Double(numChunks) * 1000.0  // ms
        let coremlPerChunk = coremlTime / Double(numChunks) * 1000.0  // ms
        let mlxRTF = mlxTime / audioDuration
        let coremlRTF = coremlTime / audioDuration

        print("\n=== Silero VAD Latency Benchmark ===")
        print("Audio: \(String(format: "%.1f", audioDuration))s, \(numChunks) chunks")
        print("")
        print("MLX:")
        print("  Total:     \(String(format: "%.3f", mlxTime * 1000))ms")
        print("  Per chunk: \(String(format: "%.3f", mlxPerChunk))ms")
        print("  RTF:       \(String(format: "%.4f", mlxRTF))")
        print("")
        print("CoreML:")
        print("  Total:     \(String(format: "%.3f", coremlTime * 1000))ms")
        print("  Per chunk: \(String(format: "%.3f", coremlPerChunk))ms")
        print("  RTF:       \(String(format: "%.4f", coremlRTF))")
        print("")

        if coremlPerChunk < mlxPerChunk {
            print("CoreML is \(String(format: "%.1f", mlxPerChunk / coremlPerChunk))x faster")
        } else {
            print("MLX is \(String(format: "%.1f", coremlPerChunk / mlxPerChunk))x faster")
        }

        // Verify probabilities match within tolerance
        var maxDiff: Float = 0
        var totalDiff: Float = 0
        for i in 0..<min(mlxProbs.count, coremlProbs.count) {
            let diff = abs(mlxProbs[i] - coremlProbs[i])
            maxDiff = max(maxDiff, diff)
            totalDiff += diff
        }
        let avgDiff = totalDiff / Float(min(mlxProbs.count, coremlProbs.count))

        print("")
        print("Probability agreement:")
        print("  Max diff: \(String(format: "%.6f", maxDiff))")
        print("  Avg diff: \(String(format: "%.6f", avgDiff))")

        XCTAssertLessThan(maxDiff, 0.05,
                           "MLX and CoreML probabilities should agree within ±0.05")
    }
}
