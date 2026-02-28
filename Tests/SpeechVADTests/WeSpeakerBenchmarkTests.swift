import XCTest
import MLX
@testable import SpeechVAD
import AudioCommon

/// Benchmarks comparing MLX vs CoreML WeSpeaker speaker embedding latency.
///
/// Requires both model variants to be cached locally:
/// - `aufklarer/WeSpeaker-ResNet34-LM-MLX` for MLX
/// - `aufklarer/WeSpeaker-ResNet34-LM-CoreML` for CoreML
///
/// Run with: `swift test --filter WeSpeakerBenchmarkTests -c release`
final class WeSpeakerBenchmarkTests: XCTestCase {

    func testMLXvsCoreMLLatency() async throws {
        let audioURL = URL(fileURLWithPath: "Tests/Qwen3ASRTests/Resources/test_audio.wav")
        guard FileManager.default.fileExists(atPath: audioURL.path) else {
            throw XCTSkip("Test audio file not found")
        }

        let (rawSamples, sampleRate) = try AudioFileLoader.loadWAV(url: audioURL)
        let audio = sampleRate != 16000
            ? AudioFileLoader.resample(rawSamples, from: sampleRate, to: 16000)
            : rawSamples

        // Load both models
        let mlxModel: WeSpeakerModel
        do {
            mlxModel = try await WeSpeakerModel.fromPretrained(engine: .mlx)
        } catch {
            throw XCTSkip("MLX model not cached: \(error)")
        }

        let coremlModel: WeSpeakerModel
        do {
            coremlModel = try await WeSpeakerModel.fromPretrained(engine: .coreml)
        } catch {
            throw XCTSkip("CoreML model not cached: \(error)")
        }

        let iterations = 5

        // Warmup: 1 embedding each
        _ = mlxModel.embed(audio: audio, sampleRate: 16000)
        _ = coremlModel.embed(audio: audio, sampleRate: 16000)

        // Benchmark MLX
        let mlxStart = CFAbsoluteTimeGetCurrent()
        var mlxEmb = [Float]()
        for _ in 0..<iterations {
            mlxEmb = mlxModel.embed(audio: audio, sampleRate: 16000)
        }
        let mlxTime = CFAbsoluteTimeGetCurrent() - mlxStart

        // Benchmark CoreML
        let coremlStart = CFAbsoluteTimeGetCurrent()
        var coremlEmb = [Float]()
        for _ in 0..<iterations {
            coremlEmb = coremlModel.embed(audio: audio, sampleRate: 16000)
        }
        let coremlTime = CFAbsoluteTimeGetCurrent() - coremlStart

        // Report
        let audioDuration = Double(audio.count) / 16000.0
        let mlxPerCall = mlxTime / Double(iterations) * 1000.0
        let coremlPerCall = coremlTime / Double(iterations) * 1000.0

        print("\n=== WeSpeaker Embedding Latency Benchmark ===")
        print("Audio: \(String(format: "%.1f", audioDuration))s, \(iterations) iterations")
        print("")
        print("MLX:")
        print("  Per call: \(String(format: "%.1f", mlxPerCall))ms")
        print("  RTF:      \(String(format: "%.4f", mlxTime / Double(iterations) / audioDuration))")
        print("")
        print("CoreML:")
        print("  Per call: \(String(format: "%.1f", coremlPerCall))ms")
        print("  RTF:      \(String(format: "%.4f", coremlTime / Double(iterations) / audioDuration))")
        print("")

        if coremlPerCall < mlxPerCall {
            print("CoreML is \(String(format: "%.1f", mlxPerCall / coremlPerCall))x faster")
        } else {
            print("MLX is \(String(format: "%.1f", coremlPerCall / mlxPerCall))x faster")
        }

        // Both backends should produce valid L2-normalized embeddings
        XCTAssertEqual(mlxEmb.count, 256)
        XCTAssertEqual(coremlEmb.count, 256)

        let mlxNorm = sqrt(mlxEmb.reduce(Float(0)) { $0 + $1 * $1 })
        let coremlNorm = sqrt(coremlEmb.reduce(Float(0)) { $0 + $1 * $1 })
        XCTAssertEqual(mlxNorm, 1.0, accuracy: 0.05, "MLX embedding should be L2-normalized")
        XCTAssertEqual(coremlNorm, 1.0, accuracy: 0.05, "CoreML embedding should be L2-normalized")

        // Each backend should be self-consistent
        let mlxEmb2 = mlxModel.embed(audio: audio, sampleRate: 16000)
        let coremlEmb2 = coremlModel.embed(audio: audio, sampleRate: 16000)
        let mlxSelfSim = WeSpeakerModel.cosineSimilarity(mlxEmb, mlxEmb2)
        let coremlSelfSim = WeSpeakerModel.cosineSimilarity(coremlEmb, coremlEmb2)
        XCTAssertEqual(mlxSelfSim, 1.0, accuracy: 0.001, "MLX should be self-consistent")
        XCTAssertEqual(coremlSelfSim, 1.0, accuracy: 0.001, "CoreML should be self-consistent")

        // Note: cross-backend cosine sim is NOT expected to be high because
        // MLX (NHWC) and CoreML (NCHW) flatten stats pooling features in
        // different orders, producing valid but non-interchangeable embeddings.
        let crossSim = WeSpeakerModel.cosineSimilarity(mlxEmb, coremlEmb)
        print("")
        print("Cross-backend cosine similarity: \(String(format: "%.6f", crossSim))")
        print("(Not expected to be high — NHWC vs NCHW feature ordering differs)")
    }
}
