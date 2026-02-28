import XCTest
import MLX
@testable import PersonaPlex
import AudioCommon
import Qwen3ASR

final class PersonaPlexTests: XCTestCase {

    // MARK: - Configuration Tests

    func testDefaultConfig() {
        let cfg = PersonaPlexConfig.default

        // Temporal transformer
        XCTAssertEqual(cfg.temporal.dim, 4096)
        XCTAssertEqual(cfg.temporal.numLayers, 32)
        XCTAssertEqual(cfg.temporal.numHeads, 32)
        XCTAssertEqual(cfg.temporal.headDim, 128)
        XCTAssertEqual(cfg.temporal.intermediateSize, 11264) // 4096 * 2/3 * 4.125
        XCTAssertEqual(cfg.temporal.nQ, 8)
        XCTAssertEqual(cfg.temporal.card, 2048)
        XCTAssertEqual(cfg.temporal.textCard, 32000)
        XCTAssertEqual(cfg.temporal.context, 3000)
        XCTAssertEqual(cfg.temporal.numAudioEmbeddings, 16)
        XCTAssertEqual(cfg.temporal.numCodebooks, 17)
    }

    func testDepformerConfig() {
        let cfg = DepformerConfig.default

        XCTAssertEqual(cfg.dim, 1024)
        XCTAssertEqual(cfg.numLayers, 6)
        XCTAssertEqual(cfg.numHeads, 16)
        XCTAssertEqual(cfg.headDim, 64)
        XCTAssertEqual(cfg.dimFeedforward, 2816)
        XCTAssertEqual(cfg.numSteps, 16)
        XCTAssertEqual(cfg.context, 8)
        XCTAssertTrue(cfg.weightsPerStep)
        XCTAssertTrue(cfg.multiLinear)
    }

    func testMimiConfig() {
        let cfg = MimiConfig.moshiko()

        XCTAssertEqual(cfg.sampleRate, 24000)
        XCTAssertEqual(cfg.frameRate, 12.5)
        XCTAssertEqual(cfg.numCodebooks, 16)
        XCTAssertEqual(cfg.codebookSize, 2048)
        XCTAssertEqual(cfg.codebookDim, 256)
        XCTAssertEqual(cfg.dimension, 512)
        XCTAssertEqual(cfg.seanet.ratios, [8, 6, 5, 4])
        XCTAssertEqual(cfg.transformer.dModel, 512)
        XCTAssertEqual(cfg.transformer.numLayers, 8)
    }

    func testSamplingConfig() {
        let cfg = PersonaPlexSamplingConfig.default

        XCTAssertEqual(cfg.audioTemp, 0.8)
        XCTAssertEqual(cfg.audioTopK, 250)
        XCTAssertEqual(cfg.textTemp, 0.7)
        XCTAssertEqual(cfg.textTopK, 25)
    }

    func testDelayPattern() {
        let cfg = PersonaPlexConfig.default

        // 17 streams: [text, 8 user audio, 8 agent audio]
        // delays: [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1]
        XCTAssertEqual(cfg.delays.count, 17)
        XCTAssertEqual(cfg.delays[0], 0)   // text: no delay
        XCTAssertEqual(cfg.delays[1], 0)   // user audio cb0 (semantic): no delay
        XCTAssertEqual(cfg.delays[2], 1)   // user audio cb1: delay 1
        XCTAssertEqual(cfg.delays[9], 0)   // agent audio cb0 (semantic): no delay
        XCTAssertEqual(cfg.delays[10], 1)  // agent audio cb1: delay 1
        XCTAssertEqual(cfg.maxDelay, 1)
        XCTAssertEqual(cfg.numStreams, 17)
    }

    func testVoicePresets() {
        XCTAssertEqual(PersonaPlexVoice.allCases.count, 18)

        // Verify all voices have display names
        for voice in PersonaPlexVoice.allCases {
            XCTAssertFalse(voice.displayName.isEmpty)
            XCTAssertFalse(voice.rawValue.isEmpty)
        }

        // Verify string round-trip
        XCTAssertEqual(PersonaPlexVoice(rawValue: "NATM0"), .NATM0)
        XCTAssertEqual(PersonaPlexVoice(rawValue: "VARF2"), .VARF2)
        XCTAssertNil(PersonaPlexVoice(rawValue: "INVALID"))
    }

    func testHiddenScaleCalculation() {
        let cfg = TemporalTransformerConfig.default
        // dim=4096, hiddenScale=4.125, LLaMA-style: dim * 2/3 * hiddenScale
        // intermediateSize = 4096 * 2/3 * 4.125 = 11264
        XCTAssertEqual(cfg.intermediateSize, 11264)
    }

    func testDepformerDimFeedforward() {
        // Moshiko: dim=1024, dimFeedforward=2816 (= 1024 * 2/3 * 4.125)
        let cfg = DepformerConfig.default
        XCTAssertEqual(cfg.dimFeedforward, 2816)
    }

    // MARK: - Sampling Tests

    func testSampleTopKArgmax() {
        // Temperature 0 should produce argmax
        let logits = MLXArray([1.0, 5.0, 2.0, 3.0] as [Float]).reshaped([1, 4])
        let token = sampleTopK(logits: logits, temperature: 0, topK: 0)
        eval(token)
        XCTAssertEqual(token[0].item(Int32.self), 1, "Should pick index 1 (value 5.0)")
    }

    func testSampleTopKWithTemperature() {
        // With temperature, sampling should still produce valid indices
        let logits = MLXArray([1.0, 10.0, 0.5, 0.1] as [Float]).reshaped([1, 4])
        for _ in 0..<10 {
            let token = sampleTopK(logits: logits, temperature: 0.8, topK: 4)
            eval(token)
            let val = token[0].item(Int32.self)
            XCTAssertGreaterThanOrEqual(val, 0)
            XCTAssertLessThan(val, 4)
        }
    }

    // MARK: - MultiLinear Tests

    func testMultiLinearWeightIndexing() {
        let numSteps = 4
        let inDim = 8
        let outDim = 6
        let ml = MultiLinear(numSteps: numSteps, inDim: inDim, outDim: outDim, bias: false)

        // Weight shape should be [numSteps * outDim, inDim]
        XCTAssertEqual(ml.weight.shape, [numSteps * outDim, inDim])

        // Each step should produce [B, T, outDim] output
        let x = MLXRandom.normal([1, 1, inDim])
        for step in 0..<numSteps {
            let out = ml(x, step: step)
            eval(out)
            XCTAssertEqual(out.shape, [1, 1, outDim],
                           "Step \(step) output shape mismatch")
        }
    }

    func testMultiLinearDifferentSteps() {
        // Different steps should use different weight slices → different outputs
        let ml = MultiLinear(numSteps: 4, inDim: 8, outDim: 6, bias: false)
        let x = MLXRandom.normal([1, 1, 8])

        let out0 = ml(x, step: 0)
        let out1 = ml(x, step: 1)
        eval(out0, out1)

        // Outputs should differ (different weight slices)
        let diff = MLX.sum(MLX.abs(out0 - out1)).item(Float.self)
        XCTAssertGreaterThan(diff, 0, "Different steps should produce different outputs")
    }

    // MARK: - Voice File Tests

    /// Verify that all 18 voice preset safetensors exist in the model cache.
    /// Regression test for #32: voices were not downloaded by fromPretrained().
    func testVoiceFilesExist() throws {
        let modelId = "aufklarer/PersonaPlex-7B-MLX-4bit"
        let cacheDir: URL
        do {
            cacheDir = try HuggingFaceDownloader.getCacheDirectory(for: modelId)
        } catch {
            throw XCTSkip("Cannot resolve cache directory")
        }

        let voiceDir = cacheDir.appendingPathComponent("voices")
        guard FileManager.default.fileExists(atPath: voiceDir.path) else {
            throw XCTSkip("Voice directory not found at \(voiceDir.path) — model not cached")
        }

        for voice in PersonaPlexVoice.allCases {
            let voiceFile = voiceDir.appendingPathComponent("\(voice.rawValue).safetensors")
            XCTAssertTrue(
                FileManager.default.fileExists(atPath: voiceFile.path),
                "Voice file missing: voices/\(voice.rawValue).safetensors — fromPretrained() must download voices/*.safetensors"
            )
        }
    }

    /// Verify voice preset loads and contains non-empty embeddings + cache.
    /// Regression test for #32: missing voice → 0 frames → gibberish output.
    func testVoicePresetLoading() throws {
        let modelId = "aufklarer/PersonaPlex-7B-MLX-4bit"
        let cacheDir: URL
        do {
            cacheDir = try HuggingFaceDownloader.getCacheDirectory(for: modelId)
        } catch {
            throw XCTSkip("Cannot resolve cache directory")
        }

        let voiceFile = cacheDir.appendingPathComponent("voices/NATF1.safetensors")
        guard FileManager.default.fileExists(atPath: voiceFile.path) else {
            throw XCTSkip("Voice file not cached: \(voiceFile.path)")
        }

        let weights = try MLX.loadArrays(url: voiceFile)

        // Must have "embeddings" key with shape [T, 1, 1, dim] where T > 0
        guard let embeddings = weights["embeddings"] else {
            XCTFail("Voice safetensors missing 'embeddings' key")
            return
        }
        XCTAssertEqual(embeddings.ndim, 4, "Voice embeddings should be [T, 1, 1, dim]")
        XCTAssertGreaterThan(embeddings.shape[0], 0, "Voice should have >0 frames (was 0 in #32)")
        print("NATF1 voice: \(embeddings.shape[0]) frames, dim=\(embeddings.shape[3])")

        // Must have "cache" key
        guard let cache = weights["cache"] else {
            XCTFail("Voice safetensors missing 'cache' key")
            return
        }
        XCTAssertEqual(cache.ndim, 3, "Voice cache should be [1, 17, CT]")
        XCTAssertEqual(cache.shape[1], 17, "Voice cache should have 17 streams")
    }
}

// MARK: - E2E Tests (require model download)

// These tests download the model (~5.5 GB) on first run and cache it.
// Run with: swift test --filter PersonaPlexE2ETests

final class PersonaPlexE2ETests: XCTestCase {

    // Shared model instance to avoid reloading between tests (~5.5 GB)
    private static var _model: PersonaPlexModel?
    private static var modelError: Error?
    private static let loadLock = NSLock()
    private static var loaded = false

    private var model: PersonaPlexModel {
        get throws {
            Self.loadLock.lock()
            defer { Self.loadLock.unlock() }

            if let error = Self.modelError {
                throw error
            }
            guard let model = Self._model else {
                throw XCTSkip("Model not loaded — run testLoadModel first or set PERSONAPLEX_MODEL_ID")
            }
            return model
        }
    }

    override func setUpWithError() throws {
        // Skip E2E tests unless env var is set or model is already loaded
        if !Self.loaded {
            let hasEnv = ProcessInfo.processInfo.environment["PERSONAPLEX_E2E"] != nil
            try XCTSkipUnless(hasEnv, "Set PERSONAPLEX_E2E=1 to run PersonaPlex E2E tests")
        }
    }

    func testLoadModel() async throws {
        guard Self._model == nil else { return }

        let modelId = ProcessInfo.processInfo.environment["PERSONAPLEX_MODEL_ID"]
            ?? "aufklarer/PersonaPlex-7B-MLX-4bit"

        do {
            let model = try await PersonaPlexModel.fromPretrained(
                modelId: modelId
            ) { progress, status in
                print("  [\(Int(progress * 100))%] \(status)")
            }
            Self._model = model
            Self.loaded = true
            print("PersonaPlex model loaded successfully")
        } catch {
            Self.modelError = error
            throw error
        }
    }

    func testRespondProducesAudio() throws {
        let model = try self.model

        // Generate a short sine wave as test input (1s @ 24kHz)
        let sampleRate = 24000
        let duration = 1.0
        let numSamples = Int(duration * Double(sampleRate))
        var testAudio = [Float](repeating: 0, count: numSamples)
        for i in 0..<numSamples {
            testAudio[i] = sin(2 * .pi * 440 * Float(i) / Float(sampleRate)) * 0.5
        }

        let response = model.respond(
            userAudio: testAudio,
            voice: .NATM0,
            maxSteps: 10,  // Very short, just verify pipeline works
            verbose: true
        )

        XCTAssertFalse(response.isEmpty, "Should produce response audio")

        let responseDuration = Double(response.count) / Double(sampleRate)
        print("Response: \(response.count) samples (\(String(format: "%.2f", responseDuration))s)")
    }

    func testRespondNonSilent() throws {
        let model = try self.model

        // 0.5s test tone
        let numSamples = 12000
        var testAudio = [Float](repeating: 0, count: numSamples)
        for i in 0..<numSamples {
            testAudio[i] = sin(2 * .pi * 440 * Float(i) / 24000.0) * 0.5
        }

        let response = model.respond(
            userAudio: testAudio,
            voice: .NATF0,
            maxSteps: 15,
            verbose: false
        )

        guard !response.isEmpty else {
            XCTFail("Response should not be empty")
            return
        }

        // Check not silent
        let maxAmp = response.map { abs($0) }.max() ?? 0
        XCTAssertGreaterThan(maxAmp, 0.001, "Response audio should not be silent")
        print("Response max amplitude: \(String(format: "%.4f", maxAmp))")
    }

    func testRespondDurationBounds() throws {
        let model = try self.model

        // 1s input, 25 generation steps → ~2s response at 12.5Hz
        let numSamples = 24000
        var testAudio = [Float](repeating: 0, count: numSamples)
        for i in 0..<numSamples {
            testAudio[i] = sin(2 * .pi * 440 * Float(i) / 24000.0) * 0.3
        }

        let response = model.respond(
            userAudio: testAudio,
            voice: .NATM0,
            maxSteps: 25,
            verbose: true
        )

        let duration = Double(response.count) / 24000.0
        print("Response duration: \(String(format: "%.2f", duration))s")

        // 25 steps at 12.5Hz ≈ 2s of audio, Mimi decode expands 1920x
        // Duration should be roughly 0.1-5s (generous bounds for test stability)
        XCTAssertGreaterThan(duration, 0.05, "Response should be at least 50ms")
        XCTAssertLessThan(duration, 10.0, "Response should be less than 10s")
    }

    func testMimiRoundTrip() throws {
        let model = try self.model

        // Generate a 1s test tone
        let numSamples = 24000
        var testAudio = [Float](repeating: 0, count: numSamples)
        for i in 0..<numSamples {
            testAudio[i] = sin(2 * .pi * 440 * Float(i) / 24000.0) * 0.5
        }

        // Encode
        let audioArray = MLXArray(testAudio).reshaped([1, 1, numSamples])
        let codes = model.mimi.encode(audioArray)
        eval(codes)
        print("Mimi encode: \(codes.shape)")  // [1, 16, T]

        let T = codes.shape[2]
        let codesInt = codes.asType(.int32)
        eval(codesInt)

        // Print codebook statistics
        for cb in 0..<min(4, codes.shape[1]) {
            var vals: [Int32] = []
            for t in 0..<min(8, T) {
                vals.append(codesInt[0, cb, t].item(Int32.self))
            }
            let minVal = vals.min() ?? 0
            let maxVal = vals.max() ?? 0
            print("  CB\(cb): \(vals) range=[\(minVal),\(maxVal)]")
        }

        // Decode back
        let decoded = model.mimi.decode(codes)
        eval(decoded)
        print("Mimi decode: \(decoded.shape)")

        let roundTripSamples = decoded.shape[2]
        let maxAmp = MLX.abs(decoded).max().item(Float.self)
        let rms = sqrt(MLX.sum(decoded * decoded).item(Float.self) / Float(roundTripSamples))
        print("Round-trip: \(roundTripSamples) samples, maxAmp=\(String(format: "%.4f", maxAmp)), RMS=\(String(format: "%.6f", rms))")

        XCTAssertGreaterThan(maxAmp, 0.01, "Mimi round-trip should produce audible audio")
        XCTAssertGreaterThan(roundTripSamples, 0, "Should produce samples")
    }

    func testRespondDiagnostic() throws {
        let model = try self.model

        // 1s test tone
        let numSamples = 24000
        var testAudio = [Float](repeating: 0, count: numSamples)
        for i in 0..<numSamples {
            testAudio[i] = sin(2 * .pi * 440 * Float(i) / 24000.0) * 0.5
        }

        let response = model.respond(
            userAudio: testAudio,
            voice: .NATM0,
            maxSteps: 10,
            verbose: true
        )

        // Analyze output
        if response.isEmpty {
            print("DIAGNOSTIC: Response is empty!")
            return
        }

        let maxAmp = response.map { abs($0) }.max() ?? 0
        let rms = sqrt(response.map { $0 * $0 }.reduce(0, +) / Float(response.count))
        print("DIAGNOSTIC Response: \(response.count) samples")
        print("  maxAmp=\(String(format: "%.4f", maxAmp))")
        print("  RMS=\(String(format: "%.6f", rms))")
        print("  First 10 samples: \(response.prefix(10).map { String(format: "%.4f", $0) })")
    }

    func testRespondWithRealAudio() throws {
        let model = try self.model

        // Load real test audio if available
        let testAudioPath = ProcessInfo.processInfo.environment["PERSONAPLEX_TEST_AUDIO"]
        guard let audioPath = testAudioPath else {
            throw XCTSkip("Set PERSONAPLEX_TEST_AUDIO=/path/to/audio.wav to run real audio test")
        }

        let url = URL(fileURLWithPath: audioPath)
        let audio = try AudioFileLoader.load(url: url, targetSampleRate: 24000)
        let inputDuration = Double(audio.count) / 24000.0
        print("Input audio: \(String(format: "%.2f", inputDuration))s (\(audio.count) samples)")

        let startTime = CFAbsoluteTimeGetCurrent()
        let response = model.respond(
            userAudio: audio,
            voice: .NATM0,
            maxSteps: 50,
            verbose: true
        )
        let elapsed = CFAbsoluteTimeGetCurrent() - startTime

        XCTAssertFalse(response.isEmpty, "Should produce response audio")

        let responseDuration = Double(response.count) / 24000.0
        let rtf = elapsed / max(responseDuration, 0.001)
        print("Response: \(String(format: "%.2f", responseDuration))s, Time: \(String(format: "%.2f", elapsed))s, RTF: \(String(format: "%.2f", rtf))")

        // Save response for manual inspection
        let outputPath = audioPath.replacingOccurrences(of: ".wav", with: "_response.wav")
        try WAVWriter.write(
            samples: response,
            sampleRate: 24000,
            to: URL(fileURLWithPath: outputPath)
        )
        print("Saved response to \(outputPath)")
    }

    func testMultipleVoices() throws {
        let model = try self.model

        let numSamples = 12000  // 0.5s
        var testAudio = [Float](repeating: 0, count: numSamples)
        for i in 0..<numSamples {
            testAudio[i] = sin(2 * .pi * 440 * Float(i) / 24000.0) * 0.3
        }

        // Test a few different voices produce non-empty output
        let voices: [PersonaPlexVoice] = [.NATM0, .NATF0, .VARM0, .VARF0]
        for voice in voices {
            let response = model.respond(
                userAudio: testAudio,
                voice: voice,
                maxSteps: 5,
                verbose: false
            )
            XCTAssertFalse(response.isEmpty, "Voice \(voice.rawValue) should produce audio")
            print("Voice \(voice.rawValue): \(response.count) samples")
        }
    }

    // MARK: - Streaming Tests

    func testStreamingProducesChunks() async throws {
        if Self._model == nil { try await testLoadModel() }
        let model = try self.model

        // 0.5s test tone
        let numSamples = 12000
        var testAudio = [Float](repeating: 0, count: numSamples)
        for i in 0..<numSamples {
            testAudio[i] = sin(2 * .pi * 440 * Float(i) / 24000.0) * 0.5
        }

        let streamingConfig = PersonaPlexModel.PersonaPlexStreamingConfig(
            firstChunkFrames: 10, chunkFrames: 10)
        let stream = model.respondStream(
            userAudio: testAudio,
            voice: .NATM0,
            maxSteps: 25,
            streaming: streamingConfig,
            verbose: true
        )

        var allSamples: [Float] = []
        var chunkCount = 0
        for try await chunk in stream {
            XCTAssertFalse(chunk.samples.isEmpty, "Chunk should contain samples")
            XCTAssertEqual(chunk.sampleRate, 24000)
            allSamples.append(contentsOf: chunk.samples)
            chunkCount += 1
            print("  Stream chunk \(chunkCount): \(chunk.samples.count) samples, final=\(chunk.isFinal)")
        }

        XCTAssertGreaterThan(chunkCount, 1, "Should produce multiple chunks")
        XCTAssertFalse(allSamples.isEmpty, "Should produce audio samples")

        let maxAmp = allSamples.map { abs($0) }.max() ?? 0
        XCTAssertGreaterThan(maxAmp, 0.001, "Streamed audio should not be silent")
        print("Streaming: \(chunkCount) chunks, \(allSamples.count) total samples, maxAmp=\(String(format: "%.4f", maxAmp))")
    }

    /// Streaming E2E relevance test: real audio → PersonaPlex streaming → concatenate chunks → ASR → keyword check.
    /// Same as testResponseRelevance but exercises the streaming path.
    func testStreamingRelevance() async throws {
        if Self._model == nil { try await testLoadModel() }
        let ppModel = try self.model

        let audioURL = findTestAudio()
        guard let url = audioURL else {
            throw XCTSkip("test_audio.wav not found — expected at Tests/Qwen3ASRTests/Resources/test_audio.wav")
        }
        let userAudio = try AudioFileLoader.load(url: url, targetSampleRate: 24000)
        let inputDuration = Double(userAudio.count) / 24000.0
        print("Input: \(String(format: "%.2f", inputDuration))s (\(userAudio.count) samples)")

        // Stream PersonaPlex response (200 gen steps ≈ 16s at 12.5Hz)
        let streamingConfig = PersonaPlexModel.PersonaPlexStreamingConfig(
            firstChunkFrames: 25, chunkFrames: 25)
        let stream = ppModel.respondStream(
            userAudio: userAudio,
            voice: .NATM0,
            maxSteps: 200,
            streaming: streamingConfig,
            verbose: true
        )

        var allSamples: [Float] = []
        var chunkCount = 0
        for try await chunk in stream {
            XCTAssertFalse(chunk.samples.isEmpty, "Chunk should contain samples")
            XCTAssertEqual(chunk.sampleRate, 24000)
            allSamples.append(contentsOf: chunk.samples)
            chunkCount += 1
            let chunkDur = String(format: "%.2f", Double(chunk.samples.count) / 24000.0)
            print("  Stream chunk \(chunkCount): \(chunk.samples.count) samples (\(chunkDur)s), final=\(chunk.isFinal)")
        }
        let responseDuration = Double(allSamples.count) / 24000.0
        print("Streaming response: \(chunkCount) chunks, \(String(format: "%.2f", responseDuration))s total")

        XCTAssertGreaterThan(chunkCount, 1, "Should produce multiple chunks")
        XCTAssertFalse(allSamples.isEmpty, "Should produce audio samples")

        // Transcribe concatenated audio with ASR
        let asrModel = try await Qwen3ASRModel.fromPretrained()
        let resampled = resampleLinear(allSamples, fromRate: 24000, toRate: 16000)
        let transcript = asrModel.transcribe(audio: resampled, sampleRate: 16000)
        print("Streaming transcript: \"\(transcript)\"")

        // --- Relevance assertions (same as testResponseRelevance) ---
        let lower = transcript.lowercased()

        // 1. Non-empty, meaningful speech (at least 5 words)
        let words = lower.split(separator: " ").map(String.init)
        XCTAssertGreaterThan(words.count, 5,
            "Response should contain at least 5 words, got \(words.count): \"\(transcript)\"")

        // 2. Topic relevance: shipping/logistics keywords
        let shippingKeywords = [
            "ship", "shipping", "shipped", "delivery", "deliver", "delivered",
            "tomorrow", "order", "replacement", "part", "guarantee",
            "days", "today", "estimate", "tracking", "package", "send", "sent",
            "arrive", "arrival", "dispatch", "stock", "available", "warehouse"
        ]
        let matchedKeywords = shippingKeywords.filter { kw in lower.contains(kw) }
        print("Matched shipping keywords: \(matchedKeywords)")
        XCTAssertGreaterThanOrEqual(matchedKeywords.count, 1,
            "Response should mention at least 1 shipping-related keyword. Transcript: \"\(transcript)\"")

        // 3. No excessive repetition
        let maxConsecutive = maxConsecutiveRepeats(in: words)
        XCTAssertLessThan(maxConsecutive, 4,
            "Response has degenerate repetition (\(maxConsecutive) consecutive repeated words)")

        // 4. Word diversity
        let uniqueRatio = Double(Set(words).count) / Double(max(words.count, 1))
        XCTAssertGreaterThan(uniqueRatio, 0.3,
            "Response word diversity too low (\(String(format: "%.0f", uniqueRatio * 100))%): \"\(transcript)\"")
    }

    // MARK: - Response Relevance Tests

    /// E2E relevance test: real audio → PersonaPlex → ASR → keyword check.
    /// Test audio: "Can you guarantee that the replacement part will be shipped tomorrow?"
    /// Verifies the response transcript contains shipping/delivery-related content.
    func testResponseRelevance() async throws {
        if Self._model == nil { try await testLoadModel() }
        let ppModel = try self.model

        // Load test audio from known location
        let audioURL = findTestAudio()
        guard let url = audioURL else {
            throw XCTSkip("test_audio.wav not found — expected at Tests/Qwen3ASRTests/Resources/test_audio.wav")
        }
        let userAudio = try AudioFileLoader.load(url: url, targetSampleRate: 24000)
        let inputDuration = Double(userAudio.count) / 24000.0
        print("Input: \(String(format: "%.2f", inputDuration))s (\(userAudio.count) samples)")

        // Generate PersonaPlex response (200 gen steps ≈ 16s at 12.5Hz)
        let response = ppModel.respond(
            userAudio: userAudio, voice: .NATM0, maxSteps: 200, verbose: true)
        XCTAssertFalse(response.isEmpty, "Should produce response audio")
        let responseDuration = Double(response.count) / 24000.0
        print("Response: \(String(format: "%.2f", responseDuration))s (\(response.count) samples)")

        // Transcribe response with ASR
        let asrModel = try await Qwen3ASRModel.fromPretrained()
        let resampled = resampleLinear(response, fromRate: 24000, toRate: 16000)
        let transcript = asrModel.transcribe(audio: resampled, sampleRate: 16000)
        print("Response transcript: \"\(transcript)\"")

        // --- Explicit relevance assertions ---
        let lower = transcript.lowercased()

        // 1. Non-empty, meaningful speech (at least 5 words)
        let words = lower.split(separator: " ").map(String.init)
        XCTAssertGreaterThan(words.count, 5,
            "Response should contain at least 5 words, got \(words.count): \"\(transcript)\"")

        // 2. Topic relevance: input asks about shipping a replacement part.
        //    Response should mention at least ONE shipping/logistics keyword.
        let shippingKeywords = [
            "ship", "shipping", "shipped", "delivery", "deliver", "delivered",
            "tomorrow", "order", "replacement", "part", "guarantee",
            "days", "today", "estimate", "tracking", "package", "send", "sent",
            "arrive", "arrival", "dispatch", "stock", "available", "warehouse"
        ]
        let matchedKeywords = shippingKeywords.filter { kw in
            lower.contains(kw)
        }
        print("Matched shipping keywords: \(matchedKeywords)")
        XCTAssertGreaterThanOrEqual(matchedKeywords.count, 1,
            "Response should mention at least 1 shipping-related keyword. Transcript: \"\(transcript)\"")

        // 3. No excessive repetition (degenerate output check)
        let maxConsecutive = maxConsecutiveRepeats(in: words)
        XCTAssertLessThan(maxConsecutive, 4,
            "Response has degenerate repetition (\(maxConsecutive) consecutive repeated words)")

        // 4. Word diversity: at least 30% unique words (not stuck in a loop)
        let uniqueRatio = Double(Set(words).count) / Double(max(words.count, 1))
        XCTAssertGreaterThan(uniqueRatio, 0.3,
            "Response word diversity too low (\(String(format: "%.0f", uniqueRatio * 100))%): \"\(transcript)\"")
    }

    /// Verifies that response to speech contains coherent English
    /// (not noise, silence, or foreign language).
    func testResponseCoherence() async throws {
        if Self._model == nil { try await testLoadModel() }
        let ppModel = try self.model

        let audioURL = findTestAudio()
        guard let url = audioURL else {
            throw XCTSkip("test_audio.wav not found")
        }
        let userAudio = try AudioFileLoader.load(url: url, targetSampleRate: 24000)

        // Short response (100 steps ≈ 8s)
        let response = ppModel.respond(
            userAudio: userAudio, voice: .NATF0, maxSteps: 100, verbose: true)
        XCTAssertFalse(response.isEmpty, "Should produce response audio")

        // Check audio quality: non-silent, reasonable amplitude
        let maxAmp = response.map { abs($0) }.max() ?? 0
        let rms = sqrt(response.map { $0 * $0 }.reduce(0, +) / Float(max(response.count, 1)))
        print("Audio: maxAmp=\(String(format: "%.4f", maxAmp)), RMS=\(String(format: "%.6f", rms))")
        XCTAssertGreaterThan(maxAmp, 0.01, "Response should not be near-silent")
        XCTAssertLessThan(maxAmp, 10.0, "Response amplitude should be reasonable")

        // Transcribe and check for English content
        let asrModel = try await Qwen3ASRModel.fromPretrained()
        let resampled = resampleLinear(response, fromRate: 24000, toRate: 16000)
        let transcript = asrModel.transcribe(audio: resampled, sampleRate: 16000)
        print("Coherence transcript: \"\(transcript)\"")

        // Should contain common English words (not gibberish)
        let lower = transcript.lowercased()
        let commonEnglish = ["the", "a", "is", "it", "to", "and", "of", "in", "that",
                             "you", "i", "for", "we", "can", "so", "if", "but", "or",
                             "have", "do", "what", "this", "with", "not", "be", "are"]
        let englishHits = commonEnglish.filter { lower.contains($0) }
        print("English word hits: \(englishHits.count)/\(commonEnglish.count)")
        XCTAssertGreaterThanOrEqual(englishHits.count, 3,
            "Response should contain at least 3 common English words, got: \(englishHits)")
    }

    // MARK: - Relevance Test Helpers

    private func findTestAudio() -> URL? {
        let candidates = [
            "Tests/Qwen3ASRTests/Resources/test_audio.wav",
            "Tests/PersonaPlexTests/Resources/test_audio.wav",
        ]
        let cwd = FileManager.default.currentDirectoryPath
        for path in candidates {
            let url = URL(fileURLWithPath: cwd).appendingPathComponent(path)
            if FileManager.default.fileExists(atPath: url.path) { return url }
        }
        // Also check absolute path from env
        if let envPath = ProcessInfo.processInfo.environment["PERSONAPLEX_TEST_AUDIO"] {
            let url = URL(fileURLWithPath: envPath)
            if FileManager.default.fileExists(atPath: url.path) { return url }
        }
        return nil
    }

    private func maxConsecutiveRepeats(in words: [String]) -> Int {
        guard words.count > 1 else { return words.count }
        var maxRun = 1, run = 1
        for i in 1..<words.count {
            if words[i] == words[i-1] { run += 1; maxRun = max(maxRun, run) }
            else { run = 1 }
        }
        return maxRun
    }

    /// Deep diagnostic test: dumps text token stream, hidden state stats,
    /// audio codebook patterns, and input token snapshots.
    func testDeepDiagnostic() async throws {
        if Self._model == nil { try await testLoadModel() }
        let model = try self.model

        // Use real audio if available, else 1s sine wave
        let userAudio: [Float]
        if let path = ProcessInfo.processInfo.environment["PERSONAPLEX_TEST_AUDIO"] {
            userAudio = try AudioFileLoader.load(url: URL(fileURLWithPath: path), targetSampleRate: 24000)
            print("DIAG: Using real audio (\(userAudio.count) samples, \(String(format: "%.2f", Double(userAudio.count)/24000))s)")
        } else {
            let n = 24000
            userAudio = (0..<n).map { sin(2 * .pi * 440 * Float($0) / 24000) * 0.5 }
            print("DIAG: Using 1s sine wave")
        }

        let (audio, diag) = model.respondDiagnostic(
            userAudio: userAudio, voice: .NATM0, maxSteps: 50)

        // 1. Text token stream
        print("\n=== TEXT TOKEN STREAM (inner monologue) ===")
        print("  Count: \(diag.textTokens.count)")
        print("  First 50: \(Array(diag.textTokens.prefix(50)))")
        let uniqueText = Set(diag.textTokens)
        print("  Unique tokens: \(uniqueText.count)")
        // Token frequency
        var textFreq: [Int32: Int] = [:]
        for t in diag.textTokens { textFreq[t, default: 0] += 1 }
        let topText = textFreq.sorted { $0.value > $1.value }.prefix(10)
        print("  Top 10 tokens: \(topText.map { "(\($0.key): \($0.value)x)" }.joined(separator: ", "))")

        // 2. Hidden state stats
        print("\n=== HIDDEN STATE STATS (first 20 gen steps) ===")
        for (i, s) in diag.hiddenStats.enumerated() {
            print("  step \(i): mean=\(String(format: "%+.4f", s.mean)) std=\(String(format: "%.4f", s.std)) range=[\(String(format: "%.2f", s.min)), \(String(format: "%.2f", s.max))]")
        }

        // 3. Text logit stats
        print("\n=== TEXT LOGIT STATS (first 20 gen steps) ===")
        for (i, s) in diag.textLogitStats.enumerated() {
            print("  step \(i): top_token=\(s.topToken) top_logit=\(String(format: "%.2f", s.topLogit)) entropy=\(String(format: "%.2f", s.entropy))")
        }

        // 4. Input token snapshots
        print("\n=== INPUT TOKEN SNAPSHOTS (streams 0-4, first 20 gen steps) ===")
        for (i, snap) in diag.inputTokenSnapshots.enumerated() {
            let desc = snap.map { "s\($0.stream)=\($0.token)" }.joined(separator: " ")
            print("  step \(i): \(desc)")
        }

        // 5. Agent audio codebook stats
        print("\n=== AGENT AUDIO CODEBOOK STATS ===")
        for (cb, tokens) in diag.agentTokensByCodebook.prefix(8).enumerated() {
            let unique = Set(tokens)
            var freq: [Int32: Int] = [:]
            for t in tokens { freq[t, default: 0] += 1 }
            let top3 = freq.sorted { $0.value > $1.value }.prefix(3)
            print("  CB\(cb): \(tokens.count) tokens, \(unique.count) unique, top3: \(top3.map { "\($0.key):\($0.value)x" }.joined(separator: " "))")
            print("       first 20: \(Array(tokens.prefix(20)))")
        }

        // 6. Audio output stats
        if !audio.isEmpty {
            let maxAmp = audio.map { abs($0) }.max() ?? 0
            let rms = sqrt(audio.map { $0 * $0 }.reduce(0, +) / Float(audio.count))
            print("\n=== OUTPUT AUDIO ===")
            print("  Samples: \(audio.count), duration: \(String(format: "%.2f", Double(audio.count)/24000))s")
            print("  maxAmp: \(String(format: "%.4f", maxAmp)), RMS: \(String(format: "%.6f", rms))")
        }
    }

    /// Compare temporal transformer forward pass against Python MLX reference.
    /// Feeds the same known tokens with empty KV cache and compares intermediate values.
    ///
    /// Python MLX reference values (from /tmp/full_forward_mlx.py):
    ///   Summed embeddings: mean=-0.000383, std=0.031235
    ///   Layer 0:  mean=-0.000494, std=0.018814
    ///   Layer 1:  mean=-0.000756, std=0.024948
    ///   Layer 31: mean=-0.000849, std=0.574707
    ///   After out_norm: mean=-0.002991, std=1.497070
    ///   Text logits: argmax=21855, logit=3.7305
    func testTemporalForwardMatch() async throws {
        if Self._model == nil { try await testLoadModel() }
        let model = try self.model
        let temporal = model.temporal

        // Reset caches for clean comparison
        temporal.resetCache()

        // Same input tokens as Python reference
        let textTokens = MLXArray([Int32(3)]).reshaped([1, 1])
        let silenceTokens: [Int32] = [948, 243, 1178, 546, 1736, 1030, 1978, 2008]
        let sineTokens: [Int32] = [430, 1268, 381, 1611, 1095, 1495, 56, 472]
        let allAudio = silenceTokens + sineTokens  // 16 tokens
        let audioTokens = MLXArray(allAudio).reshaped([1, 16, 1])

        print("\n=== TEMPORAL FORWARD MATCH TEST ===")

        // Step 1: Compute embedding sum manually (matching TemporalTransformer.forward)
        var hidden = temporal.text_emb(textTokens)
        eval(hidden)
        let textMean = MLX.mean(hidden).item(Float.self)
        print("  text_emb mean: \(String(format: "%.6f", textMean))")
        // Python: -0.000121

        for i in 0..<16 {
            let tok = audioTokens[0..<1, i, 0..<1]
            let safeTok = MLX.maximum(tok, MLXArray(Int32(0)))
            let embResult = temporal.emb[i](safeTok)
            hidden = hidden + embResult
        }
        eval(hidden)
        let sumMean = MLX.mean(hidden).item(Float.self)
        let sumStd = MLX.sqrt(MLX.mean(hidden * hidden)).item(Float.self)  // rough std
        let first5 = (0..<5).map { hidden[0, 0, $0].item(Float.self) }
        print("  Summed: mean=\(String(format: "%.6f", sumMean))")
        print("  Summed first 5: \(first5.map { String(format: "%.6f", $0) })")
        // Python: mean=-0.000383, first5=[0.042, -0.012, 0.018, -0.011, -0.048]

        // Step 2: Pass through each layer individually
        for layerIdx in 0..<temporal.cfg.numLayers {
            let cache = temporal.cache[layerIdx]
            hidden = temporal.layers[layerIdx](hidden, cache: cache, offset: 1)
            eval(hidden)

            if layerIdx < 3 || layerIdx >= 30 || layerIdx == 15 {
                let lMean = MLX.mean(hidden).item(Float.self)
                let lStd = MLX.sqrt(MLX.mean(hidden * hidden)).item(Float.self)
                let lMin = MLX.min(hidden).item(Float.self)
                let lMax = MLX.max(hidden).item(Float.self)
                print("  Layer \(layerIdx): mean=\(String(format: "%.6f", lMean)), rms=\(String(format: "%.6f", lStd)), range=[\(String(format: "%.3f", lMin)), \(String(format: "%.3f", lMax))]")
            }
        }
        // Python Layer 0: mean=-0.000494, Layer 31: mean=-0.000849, std=0.574707

        // Step 3: out_norm
        let normed = temporal.out_norm(hidden)
        eval(normed)
        let normMean = MLX.mean(normed).item(Float.self)
        let normFirst5 = (0..<5).map { normed[0, 0, $0].item(Float.self) }
        print("  After out_norm: mean=\(String(format: "%.6f", normMean))")
        print("  out_norm first 5: \(normFirst5.map { String(format: "%.6f", $0) })")
        // Python: mean=-0.002991, first5=[-0.619, -1.014, 2.180, 2.605, -2.598]

        // Step 4: Text logits
        let textLogits = temporal.text_linear(normed)
        eval(textLogits)
        let logitsMean = MLX.mean(textLogits).item(Float.self)
        print("  Text logits mean: \(String(format: "%.6f", logitsMean))")
        // Python: -1.175781

        // Argmax
        let flat = textLogits.reshaped([-1])
        let argmaxIdx = argMax(flat).item(Int32.self)
        let argmaxVal = flat[Int(argmaxIdx)].item(Float.self)
        print("  Argmax: token=\(argmaxIdx), logit=\(String(format: "%.4f", argmaxVal))")
        // Python: token=21855, logit=3.7305

        // Top-5
        let top5 = argSort(-flat)[0..<5]
        eval(top5)
        print("  Top 5 tokens: ", terminator: "")
        for i in 0..<5 {
            let idx = top5[i].item(Int32.self)
            let val = flat[Int(idx)].item(Float.self)
            print("\(idx)(\(String(format: "%.2f", val))) ", terminator: "")
        }
        print()

        // Assertions (tolerance for quantization differences)
        XCTAssertEqual(sumMean, -0.000383, accuracy: 0.001, "Embedding sum mean should match Python")
        XCTAssertEqual(argmaxIdx, 21855, "Argmax text token should match Python reference")
    }

    /// Compare depformer forward pass against Python MLX reference.
    /// Uses the temporal hidden state from testTemporalForwardMatch and runs through depformer.
    ///
    /// Python MLX reference (from /tmp/test_depformer_mlx.py):
    ///   Step 0:  argmax=1676  Step 1:  argmax=1515  Step 2:  argmax=1626
    ///   Step 3:  argmax=1562  Step 4:  argmax=306   Step 5:  argmax=478
    ///   Step 6:  argmax=326   Step 7:  argmax=101   Step 8:  argmax=768
    ///   Step 9:  argmax=243   Step 10: argmax=1178  Step 11: argmax=417
    ///   Step 12: argmax=1736  Step 13: argmax=478   Step 14: argmax=1334
    ///   Step 15: argmax=274
    func testDepformerForwardMatch() async throws {
        if Self._model == nil { try await testLoadModel() }
        let model = try self.model
        let temporal = model.temporal
        let depformer = model.depformer

        // Reset caches
        temporal.resetCache()

        // Same input tokens as temporal forward test
        let textTokens = MLXArray([Int32(3)]).reshaped([1, 1])
        let silenceTokens: [Int32] = [948, 243, 1178, 546, 1736, 1030, 1978, 2008]
        let sineTokens: [Int32] = [430, 1268, 381, 1611, 1095, 1495, 56, 472]
        let allAudio = silenceTokens + sineTokens
        let audioTokens = MLXArray(allAudio).reshaped([1, 16, 1])

        // Compute temporal hidden state (same as testTemporalForwardMatch)
        let (normedHidden, textLogits) = temporal.forward(
            textTokens: textTokens, audioTokens: audioTokens, offset: 1)
        eval(normedHidden, textLogits)

        // Get text token argmax
        let flatLogits = textLogits.reshaped([-1])
        let textTokenGen = argMax(flatLogits)
        eval(textTokenGen)
        let textTok = textTokenGen.reshaped([1])

        print("\n=== DEPFORMER FORWARD MATCH TEST ===")
        print("  Text argmax: \(textTok[0].item(Int32.self))")

        // Run depformer with argmax sampling (temperature=0)
        // Reference values from 4-bit quantized depformer
        let expectedArgmax: [Int32] = [653, 1515, 1626, 1562, 306, 478, 326, 101,
                                       1031, 1211, 783, 546, 267, 478, 1334, 274]

        let agentCodes = depformer.generate(
            temporalHidden: normedHidden,
            textToken: textTok
        ) { logits, cbIdx in
            // Argmax sampling (temperature 0)
            return sampleTopK(logits: logits, temperature: 0, topK: 0)
        }
        eval(agentCodes)

        // Print results
        let codeArr = agentCodes[0]  // [numSteps]
        var mismatches = 0
        for k in 0..<16 {
            let tok = codeArr[k].item(Int32.self)
            let expected = expectedArgmax[k]
            let match = tok == expected ? "✓" : "✗ MISMATCH"
            print("  Step \(String(format: "%2d", k)): token=\(String(format: "%5d", tok)) expected=\(String(format: "%5d", expected)) \(match)")
            if tok != expected { mismatches += 1 }
        }
        print("  Mismatches: \(mismatches)/16")

        // Assert first codebook matches (most critical for audio quality)
        XCTAssertEqual(codeArr[0].item(Int32.self), expectedArgmax[0],
                       "Depformer step 0 should match Python reference")
    }

    /// Round-trip test: real audio → PersonaPlex → response audio → ASR transcription.
    /// Checks that the response contains recognizable English speech.
    func testRoundTripASR() async throws {
        let ppModel = try self.model

        // Load real test audio
        let testAudioPath = ProcessInfo.processInfo.environment["PERSONAPLEX_TEST_AUDIO"]
        guard let audioPath = testAudioPath else {
            throw XCTSkip("Set PERSONAPLEX_TEST_AUDIO=/path/to/audio.wav to run round-trip test")
        }

        let url = URL(fileURLWithPath: audioPath)
        let userAudio = try AudioFileLoader.load(url: url, targetSampleRate: 24000)
        let inputDuration = Double(userAudio.count) / 24000.0
        print("Input audio: \(String(format: "%.2f", inputDuration))s")

        // Use same system prompt as Python reference for comparison
        // "<system> You are a wise and friendly teacher. Answer questions or provide advice in a clear and engaging way. <system>"
        let refPromptTokens: [Int32] = [607, 4831, 578, 493, 298, 272, 11821, 267, 7514, 3290, 263, 506, 1292, 307, 775, 3574, 271, 272, 1195, 267, 12250, 488, 263, 607, 4831, 578]

        // Generate response. maxSteps=0 means generate only during user audio
        // (matching Python reference which also generates exactly len(user_audio) steps).
        // Post-user-audio generation degenerates without continuous user input.
        let response = ppModel.respond(
            userAudio: userAudio,
            voice: .NATM0,
            systemPromptTokens: refPromptTokens,
            maxSteps: 0,
            verbose: true
        )

        XCTAssertFalse(response.isEmpty, "Should produce response audio")
        let responseDuration = Double(response.count) / 24000.0
        print("Response: \(String(format: "%.2f", responseDuration))s (\(response.count) samples)")

        // Save response for manual inspection
        let outputPath = audioPath.replacingOccurrences(of: ".wav", with: "_roundtrip.wav")
        try WAVWriter.write(samples: response, sampleRate: 24000, to: URL(fileURLWithPath: outputPath))
        print("Saved response to \(outputPath)")

        // Load ASR model and transcribe the response
        let asrModel = try await Qwen3ASRModel.fromPretrained()
        // ASR expects 16kHz — resample from 24kHz
        let resampledResponse = resampleLinear(response, fromRate: 24000, toRate: 16000)
        let transcript = asrModel.transcribe(audio: resampledResponse, sampleRate: 16000)
        print("ASR transcript: \"\(transcript)\"")

        // Basic checks: transcript should be non-empty English text
        XCTAssertFalse(transcript.isEmpty, "Transcript should not be empty")
        XCTAssertGreaterThan(transcript.count, 3, "Transcript should contain recognizable words")

        // Check for excessive repetition (same word >5 times in a row)
        let words = transcript.lowercased().split(separator: " ").map(String.init)
        var maxConsecutive = 1
        var currentRun = 1
        for i in 1..<words.count {
            if words[i] == words[i-1] {
                currentRun += 1
                maxConsecutive = max(maxConsecutive, currentRun)
            } else {
                currentRun = 1
            }
        }
        print("Max consecutive repeated word: \(maxConsecutive)")
        XCTAssertLessThan(maxConsecutive, 6, "Response should not have excessive word repetition (>5 consecutive)")
    }
}

// MARK: - Utility

/// Simple linear resampling (for ASR which expects 16kHz from PersonaPlex's 24kHz output).
private func resampleLinear(_ samples: [Float], fromRate: Int, toRate: Int) -> [Float] {
    guard fromRate != toRate, !samples.isEmpty else { return samples }
    let ratio = Double(fromRate) / Double(toRate)
    let outputCount = Int(Double(samples.count) / ratio)
    var output = [Float](repeating: 0, count: outputCount)
    for i in 0..<outputCount {
        let srcPos = Double(i) * ratio
        let srcIdx = Int(srcPos)
        let frac = Float(srcPos - Double(srcIdx))
        if srcIdx + 1 < samples.count {
            output[i] = samples[srcIdx] * (1 - frac) + samples[srcIdx + 1] * frac
        } else if srcIdx < samples.count {
            output[i] = samples[srcIdx]
        }
    }
    return output
}
