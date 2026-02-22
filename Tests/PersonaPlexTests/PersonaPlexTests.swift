import XCTest
import MLX
@testable import PersonaPlex
import Qwen3Common
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

        // Generate response
        let response = ppModel.respond(
            userAudio: userAudio,
            voice: .NATM0,
            maxSteps: 200,
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
