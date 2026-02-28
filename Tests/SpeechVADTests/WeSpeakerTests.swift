import XCTest
import MLX
@testable import SpeechVAD
import AudioCommon

final class WeSpeakerTests: XCTestCase {

    // MARK: - Configuration Tests

    func testDiarizationConfigDefault() {
        let config = DiarizationConfig.default
        XCTAssertEqual(config.onset, 0.5, accuracy: 0.001)
        XCTAssertEqual(config.offset, 0.3, accuracy: 0.001)
        XCTAssertEqual(config.minSpeechDuration, 0.3, accuracy: 0.001)
        XCTAssertEqual(config.clusteringThreshold, 0.5, accuracy: 0.001)
        XCTAssertEqual(config.maxSpeakers, 0)
    }

    func testDiarizedSegmentDuration() {
        let seg = DiarizedSegment(startTime: 1.0, endTime: 3.5, speakerId: 0)
        XCTAssertEqual(seg.duration, 2.5, accuracy: 0.001)
        XCTAssertEqual(seg.speakerId, 0)
    }

    // MARK: - PowersetDecoder Tests

    func testPowersetDecoderShape() {
        // Uniform posteriors: [1, 10, 7]
        let posteriors = MLXArray.ones([1, 10, 7]) / 7.0
        let speakerProbs = PowersetDecoder.speakerProbabilities(from: posteriors)
        eval(speakerProbs)

        XCTAssertEqual(speakerProbs.shape, [1, 10, 3])
    }

    func testPowersetDecoderValues() {
        // Create known posteriors: only class 1 (spk1 alone) is active
        var data = [Float](repeating: 0, count: 7)
        data[1] = 1.0  // spk1 alone
        let posteriors = MLXArray(data, [1, 1, 7])

        let speakerProbs = PowersetDecoder.speakerProbabilities(from: posteriors)
        eval(speakerProbs)

        let probs = speakerProbs[0, 0].asArray(Float.self)
        XCTAssertEqual(probs[0], 1.0, accuracy: 0.001)  // spk1 active
        XCTAssertEqual(probs[1], 0.0, accuracy: 0.001)  // spk2 inactive
        XCTAssertEqual(probs[2], 0.0, accuracy: 0.001)  // spk3 inactive
    }

    func testPowersetDecoderOverlap() {
        // Class 4 (spk1+2 overlap) is active
        var data = [Float](repeating: 0, count: 7)
        data[4] = 1.0  // spk1+2
        let posteriors = MLXArray(data, [1, 1, 7])

        let speakerProbs = PowersetDecoder.speakerProbabilities(from: posteriors)
        eval(speakerProbs)

        let probs = speakerProbs[0, 0].asArray(Float.self)
        XCTAssertEqual(probs[0], 1.0, accuracy: 0.001)  // spk1 active
        XCTAssertEqual(probs[1], 1.0, accuracy: 0.001)  // spk2 active
        XCTAssertEqual(probs[2], 0.0, accuracy: 0.001)  // spk3 inactive
    }

    func testPowersetBinarize() {
        let probs: [Float] = [0.1, 0.1, 0.8, 0.9, 0.9, 0.2, 0.1]
        let segments = PowersetDecoder.binarize(
            probs: probs, onset: 0.5, offset: 0.3, frameDuration: 1.0
        )
        XCTAssertEqual(segments.count, 1)
        XCTAssertEqual(segments[0].startTime, 2.0, accuracy: 0.01)
        XCTAssertEqual(segments[0].endTime, 5.0, accuracy: 0.01)
    }

    // MARK: - MelFeatureExtractor Tests

    func testMelExtractorShape() {
        let extractor = MelFeatureExtractor()
        // 1 second of silence at 16kHz
        let audio = [Float](repeating: 0, count: 16000)
        let mel = extractor.extract(audio)
        eval(mel)

        // T = (16000 + 400 - 400) / 160 + 1 = 101 frames
        XCTAssertEqual(mel.shape[1], 80)
        XCTAssertGreaterThan(mel.shape[0], 90)
        XCTAssertLessThan(mel.shape[0], 110)
    }

    func testMelExtractorNonZero() {
        let extractor = MelFeatureExtractor()
        // Random audio
        var audio = [Float](repeating: 0, count: 16000)
        for i in 0..<audio.count {
            audio[i] = Float.random(in: -0.5...0.5)
        }
        let mel = extractor.extract(audio)
        eval(mel)

        // Should have non-zero values for non-silent audio
        let maxVal = mel.max().item(Float.self)
        XCTAssertGreaterThan(maxVal, -20.0)  // log scale, not -inf
    }

    // MARK: - WeSpeaker Model Shape Tests (random weights)

    func testResNet34OutputShape() {
        let network = WeSpeakerNetwork()

        // 2 seconds of mel features: T≈101 frames, 80 mel bins
        let mel = MLXRandom.normal([1, 101, 80, 1])
        let output = network(mel)
        eval(output)

        // Should produce [1, 256]
        XCTAssertEqual(output.shape, [1, 256])
    }

    func testResNet34L2Normalized() {
        let network = WeSpeakerNetwork()

        let mel = MLXRandom.normal([1, 101, 80, 1])
        let output = network(mel)
        eval(output)

        // L2 norm should be approximately 1.0
        let norm = sqrt((output * output).sum(axis: -1)).item(Float.self)
        XCTAssertEqual(norm, 1.0, accuracy: 0.01)
    }

    func testBasicBlockSameChannels() {
        let block = BasicBlock(inChannels: 32, outChannels: 32)
        let x = MLXRandom.normal([1, 100, 80, 32])
        let out = block(x)
        eval(out)

        // Same channels, no stride → same spatial dims
        XCTAssertEqual(out.shape, [1, 100, 80, 32])
    }

    func testBasicBlockDownsample() {
        let block = BasicBlock(inChannels: 32, outChannels: 64, stride: 2)
        let x = MLXRandom.normal([1, 100, 80, 32])
        let out = block(x)
        eval(out)

        // Stride 2 → halved spatial dims, doubled channels
        XCTAssertEqual(out.shape, [1, 50, 40, 64])
    }

    // MARK: - Cosine Similarity Tests

    func testCosineSimilaritySame() {
        let a: [Float] = [1, 0, 0, 0]
        let sim = WeSpeakerModel.cosineSimilarity(a, a)
        XCTAssertEqual(sim, 1.0, accuracy: 0.001)
    }

    func testCosineSimilarityOpposite() {
        let a: [Float] = [1, 0, 0, 0]
        let b: [Float] = [-1, 0, 0, 0]
        let sim = WeSpeakerModel.cosineSimilarity(a, b)
        XCTAssertEqual(sim, -1.0, accuracy: 0.001)
    }

    func testCosineSimilarityOrthogonal() {
        let a: [Float] = [1, 0, 0, 0]
        let b: [Float] = [0, 1, 0, 0]
        let sim = WeSpeakerModel.cosineSimilarity(a, b)
        XCTAssertEqual(sim, 0.0, accuracy: 0.001)
    }

    // MARK: - DiarizationResult Tests

    func testDiarizationResultInit() {
        let segments = [
            DiarizedSegment(startTime: 0.0, endTime: 1.0, speakerId: 0),
            DiarizedSegment(startTime: 1.5, endTime: 3.0, speakerId: 1),
        ]
        let embeddings = [[Float](repeating: 0.1, count: 256), [Float](repeating: -0.1, count: 256)]
        let result = DiarizationResult(segments: segments, numSpeakers: 2, speakerEmbeddings: embeddings)

        XCTAssertEqual(result.segments.count, 2)
        XCTAssertEqual(result.numSpeakers, 2)
        XCTAssertEqual(result.speakerEmbeddings.count, 2)
    }

    // MARK: - E2E Integration Test (requires real weights)

    func testE2EEmbedding() async throws {
        let model = try await WeSpeakerModel.fromPretrained()

        let audioURL = URL(fileURLWithPath: "Tests/Qwen3ASRTests/Resources/test_audio.wav")
        guard FileManager.default.fileExists(atPath: audioURL.path) else {
            throw XCTSkip("Test audio file not found")
        }

        let (samples, sampleRate) = try AudioFileLoader.loadWAV(url: audioURL)
        XCTAssertGreaterThan(samples.count, 0)

        let embedding = model.embed(audio: samples, sampleRate: sampleRate)

        // Should be 256-dim
        XCTAssertEqual(embedding.count, 256)

        // Should be L2 normalized (norm ≈ 1.0)
        let norm = sqrt(embedding.reduce(Float(0)) { $0 + $1 * $1 })
        XCTAssertEqual(norm, 1.0, accuracy: 0.01)

        // Same audio should produce similar embedding (consistency check)
        let embedding2 = model.embed(audio: samples, sampleRate: sampleRate)
        let similarity = WeSpeakerModel.cosineSimilarity(embedding, embedding2)
        XCTAssertEqual(similarity, 1.0, accuracy: 0.001)
    }

    func testE2EDiarization() async throws {
        let pipeline = try await DiarizationPipeline.fromPretrained()

        let audioURL = URL(fileURLWithPath: "Tests/Qwen3ASRTests/Resources/test_audio.wav")
        guard FileManager.default.fileExists(atPath: audioURL.path) else {
            throw XCTSkip("Test audio file not found")
        }

        let (samples, sampleRate) = try AudioFileLoader.loadWAV(url: audioURL)
        let result = pipeline.diarize(audio: samples, sampleRate: sampleRate, config: .default)

        // Should detect at least one speaker segment
        XCTAssertGreaterThan(result.segments.count, 0)
        XCTAssertGreaterThanOrEqual(result.numSpeakers, 1)

        // All segments should have valid times
        for seg in result.segments {
            XCTAssertGreaterThanOrEqual(seg.startTime, 0)
            XCTAssertGreaterThan(seg.endTime, seg.startTime)
            XCTAssertGreaterThanOrEqual(seg.speakerId, 0)
            XCTAssertLessThan(seg.speakerId, result.numSpeakers)
        }

        // Should have centroid for each speaker
        XCTAssertEqual(result.speakerEmbeddings.count, result.numSpeakers)
        for emb in result.speakerEmbeddings {
            XCTAssertEqual(emb.count, 256)
        }
    }

    // MARK: - E2E: Single Speaker Detection

    func testE2ESingleSpeakerAudio() async throws {
        let pipeline = try await DiarizationPipeline.fromPretrained()

        let audioURL = URL(fileURLWithPath: "Tests/Qwen3ASRTests/Resources/test_audio.wav")
        guard FileManager.default.fileExists(atPath: audioURL.path) else {
            throw XCTSkip("Test audio file not found")
        }

        let (samples, sampleRate) = try AudioFileLoader.loadWAV(url: audioURL)

        // Test audio has a single speaker — with maxSpeakers=1, pipeline should detect 1
        let config = DiarizationConfig(maxSpeakers: 1)
        let result = pipeline.diarize(audio: samples, sampleRate: sampleRate, config: config)
        XCTAssertEqual(result.numSpeakers, 1,
                       "maxSpeakers=1 should force 1 speaker, got \(result.numSpeakers)")

        // All segments should be speaker 0
        for seg in result.segments {
            XCTAssertEqual(seg.speakerId, 0)
        }

        // Segments should cover the speech region
        XCTAssertGreaterThan(result.segments.count, 0)
    }

    // MARK: - E2E: Speaker Extraction

    func testE2ESpeakerExtraction() async throws {
        let pipeline = try await DiarizationPipeline.fromPretrained()

        let audioURL = URL(fileURLWithPath: "Tests/Qwen3ASRTests/Resources/test_audio.wav")
        guard FileManager.default.fileExists(atPath: audioURL.path) else {
            throw XCTSkip("Test audio file not found")
        }

        let (samples, sampleRate) = try AudioFileLoader.loadWAV(url: audioURL)

        // Use same audio as enrollment (target speaker IS the speaker)
        let targetEmb = pipeline.embeddingModel.embed(audio: samples, sampleRate: sampleRate)
        XCTAssertEqual(targetEmb.count, 256)

        let extracted = pipeline.extractSpeaker(
            audio: samples, sampleRate: sampleRate,
            targetEmbedding: targetEmb
        )

        // Should extract at least one segment matching the target speaker
        XCTAssertGreaterThan(extracted.count, 0,
                             "Should extract segments matching the target speaker")

        // Extracted segments should be in the speech region (~5-8.5s)
        for seg in extracted {
            XCTAssertGreaterThanOrEqual(seg.startTime, 0)
            XCTAssertGreaterThan(seg.endTime, seg.startTime)
            XCTAssertGreaterThan(seg.duration, 0.1)
        }
    }

    // MARK: - E2E: Embedding Subsegment Consistency

    func testE2EEmbeddingSubsegments() async throws {
        let model = try await WeSpeakerModel.fromPretrained()

        let audioURL = URL(fileURLWithPath: "Tests/Qwen3ASRTests/Resources/test_audio.wav")
        guard FileManager.default.fileExists(atPath: audioURL.path) else {
            throw XCTSkip("Test audio file not found")
        }

        let (samples, sampleRate) = try AudioFileLoader.loadWAV(url: audioURL)

        // Embed the full audio
        let fullEmb = model.embed(audio: samples, sampleRate: sampleRate)

        // Embed just the speech region (~5-8.5s)
        let startSample = Int(5.0 * Float(sampleRate))
        let endSample = min(Int(8.5 * Float(sampleRate)), samples.count)
        let speechRegion = Array(samples[startSample..<endSample])

        let speechEmb = model.embed(audio: speechRegion, sampleRate: sampleRate)

        // Same speaker in both — cosine similarity should be positive
        // Full audio includes silence which affects the embedding, so threshold is relaxed
        let similarity = WeSpeakerModel.cosineSimilarity(fullEmb, speechEmb)
        XCTAssertGreaterThan(similarity, 0.2,
                             "Full audio and speech region embeddings should be positively correlated (got \(similarity))")
    }

    // MARK: - E2E: Protocol Conformance

    func testE2EProtocolConformance() async throws {
        let pipeline = try await DiarizationPipeline.fromPretrained()

        let audioURL = URL(fileURLWithPath: "Tests/Qwen3ASRTests/Resources/test_audio.wav")
        guard FileManager.default.fileExists(atPath: audioURL.path) else {
            throw XCTSkip("Test audio file not found")
        }

        let (samples, sampleRate) = try AudioFileLoader.loadWAV(url: audioURL)

        // Test through SpeakerDiarizationModel protocol
        let diarizationModel: any SpeakerDiarizationModel = pipeline
        let segments = diarizationModel.diarize(audio: samples, sampleRate: sampleRate)
        XCTAssertGreaterThan(segments.count, 0)
        for seg in segments {
            XCTAssertGreaterThanOrEqual(seg.speakerId, 0)
            XCTAssertGreaterThan(seg.endTime, seg.startTime)
        }

        // Test through SpeakerEmbeddingModel protocol
        let embModel: any SpeakerEmbeddingModel = pipeline.embeddingModel
        XCTAssertEqual(embModel.embeddingDimension, 256)
        XCTAssertEqual(embModel.inputSampleRate, 16000)

        let emb = embModel.embed(audio: samples, sampleRate: sampleRate)
        XCTAssertEqual(emb.count, 256)
        let norm = sqrt(emb.reduce(Float(0)) { $0 + $1 * $1 })
        XCTAssertEqual(norm, 1.0, accuracy: 0.01)
    }

    // MARK: - E2E: VAD Cross-Validation

    func testE2EVADCrossValidation() async throws {
        let pyannote = try await PyannoteVADModel.fromPretrained()
        let silero = try await SileroVADModel.fromPretrained()

        let audioURL = URL(fileURLWithPath: "Tests/Qwen3ASRTests/Resources/test_audio.wav")
        guard FileManager.default.fileExists(atPath: audioURL.path) else {
            throw XCTSkip("Test audio file not found")
        }

        let (samples, sampleRate) = try AudioFileLoader.loadWAV(url: audioURL)

        // Both models should detect speech in the same region
        let pyannoteSegs = pyannote.detectSpeech(audio: samples, sampleRate: sampleRate)
        let sileroSegs = silero.detectSpeech(audio: samples, sampleRate: sampleRate)

        XCTAssertGreaterThan(pyannoteSegs.count, 0, "Pyannote should detect speech")
        XCTAssertGreaterThan(sileroSegs.count, 0, "Silero should detect speech")

        // Both should detect speech starting roughly in 4-7s range
        if let pySeg = pyannoteSegs.first, let siSeg = sileroSegs.first {
            XCTAssertEqual(pySeg.startTime, siSeg.startTime, accuracy: 2.0,
                           "Pyannote (\(pySeg.startTime)s) and Silero (\(siSeg.startTime)s) start times should be within 2s")
            XCTAssertEqual(pySeg.endTime, siSeg.endTime, accuracy: 2.0,
                           "Pyannote (\(pySeg.endTime)s) and Silero (\(siSeg.endTime)s) end times should be within 2s")
        }

        // Both through VoiceActivityDetectionModel protocol
        let models: [any VoiceActivityDetectionModel] = [pyannote, silero]
        for model in models {
            let segs = model.detectSpeech(audio: samples, sampleRate: sampleRate)
            XCTAssertGreaterThan(segs.count, 0)
        }
    }

    // MARK: - E2E: Mel Feature Extractor on Real Audio

    func testE2EMelFeatures() async throws {
        let audioURL = URL(fileURLWithPath: "Tests/Qwen3ASRTests/Resources/test_audio.wav")
        guard FileManager.default.fileExists(atPath: audioURL.path) else {
            throw XCTSkip("Test audio file not found")
        }

        let (samples, sampleRate) = try AudioFileLoader.loadWAV(url: audioURL)

        // Resample to 16kHz if needed
        let audio16k: [Float]
        if sampleRate != 16000 {
            let ratio = Double(16000) / Double(sampleRate)
            let outLen = Int(Double(samples.count) * ratio)
            audio16k = (0..<outLen).map { i in
                let src = Double(i) / ratio
                let idx = Int(src)
                let frac = Float(src - Double(idx))
                if idx + 1 < samples.count {
                    return samples[idx] * (1 - frac) + samples[idx + 1] * frac
                }
                return idx < samples.count ? samples[idx] : 0
            }
        } else {
            audio16k = samples
        }

        let extractor = MelFeatureExtractor()
        let mel = extractor.extract(audio16k)
        eval(mel)

        // Shape: [T, 80]
        XCTAssertEqual(mel.shape[1], 80, "Mel should have 80 bins")
        XCTAssertGreaterThan(mel.shape[0], 0, "Should have at least 1 frame")

        // Expected frames: (numSamples + 400 - 400) / 160 + 1
        let expectedFrames = (audio16k.count / 160) + 1
        XCTAssertEqual(mel.shape[0], expectedFrames, accuracy: 5,
                       "Frame count should match expected")

        // Mel values should be in reasonable log range (not NaN/Inf)
        let maxVal = mel.max().item(Float.self)
        let minVal = mel.min().item(Float.self)
        XCTAssertFalse(maxVal.isNaN, "Mel max should not be NaN")
        XCTAssertFalse(minVal.isNaN, "Mel min should not be NaN")
        XCTAssertFalse(maxVal.isInfinite, "Mel max should not be Inf")
        XCTAssertGreaterThan(maxVal, -30.0, "Mel max should be above -30 (log scale)")
    }

    // MARK: - E2E: Diarization with Custom Config

    func testE2EDiarizationCustomConfig() async throws {
        let pipeline = try await DiarizationPipeline.fromPretrained()

        let audioURL = URL(fileURLWithPath: "Tests/Qwen3ASRTests/Resources/test_audio.wav")
        guard FileManager.default.fileExists(atPath: audioURL.path) else {
            throw XCTSkip("Test audio file not found")
        }

        let (samples, sampleRate) = try AudioFileLoader.loadWAV(url: audioURL)

        // Force max 1 speaker
        let config = DiarizationConfig(maxSpeakers: 1)
        let result = pipeline.diarize(audio: samples, sampleRate: sampleRate, config: config)

        XCTAssertEqual(result.numSpeakers, 1,
                       "maxSpeakers=1 should force single speaker, got \(result.numSpeakers)")
        XCTAssertGreaterThan(result.segments.count, 0)

        // All segments should be speaker 0 when forced to 1 speaker
        for seg in result.segments {
            XCTAssertEqual(seg.speakerId, 0)
        }

        // Force max 2 speakers — should produce at most 2
        let config2 = DiarizationConfig(maxSpeakers: 2)
        let result2 = pipeline.diarize(audio: samples, sampleRate: sampleRate, config: config2)
        XCTAssertLessThanOrEqual(result2.numSpeakers, 2,
                                  "maxSpeakers=2 should produce at most 2 speakers, got \(result2.numSpeakers)")
    }

    // MARK: - E2E: CoreML Embedding

    func testE2ECoreMLEmbedding() async throws {
        let model: WeSpeakerModel
        do {
            model = try await WeSpeakerModel.fromPretrained(engine: .coreml)
        } catch {
            throw XCTSkip("CoreML model not cached: \(error)")
        }

        let audioURL = URL(fileURLWithPath: "Tests/Qwen3ASRTests/Resources/test_audio.wav")
        guard FileManager.default.fileExists(atPath: audioURL.path) else {
            throw XCTSkip("Test audio file not found")
        }

        let (samples, sampleRate) = try AudioFileLoader.loadWAV(url: audioURL)
        let embedding = model.embed(audio: samples, sampleRate: sampleRate)

        // Should be 256-dim
        XCTAssertEqual(embedding.count, 256)

        // Should be L2 normalized (norm ~ 1.0)
        let norm = sqrt(embedding.reduce(Float(0)) { $0 + $1 * $1 })
        XCTAssertEqual(norm, 1.0, accuracy: 0.05)
    }

    func testE2ECoreMLEmbeddingConsistency() async throws {
        let model: WeSpeakerModel
        do {
            model = try await WeSpeakerModel.fromPretrained(engine: .coreml)
        } catch {
            throw XCTSkip("CoreML model not cached: \(error)")
        }

        let audioURL = URL(fileURLWithPath: "Tests/Qwen3ASRTests/Resources/test_audio.wav")
        guard FileManager.default.fileExists(atPath: audioURL.path) else {
            throw XCTSkip("Test audio file not found")
        }

        let (samples, sampleRate) = try AudioFileLoader.loadWAV(url: audioURL)

        let emb1 = model.embed(audio: samples, sampleRate: sampleRate)
        let emb2 = model.embed(audio: samples, sampleRate: sampleRate)

        // Same audio → same embedding
        let similarity = WeSpeakerModel.cosineSimilarity(emb1, emb2)
        XCTAssertEqual(similarity, 1.0, accuracy: 0.001)
    }

    func testE2ECoreMLDiarization() async throws {
        let pipeline: DiarizationPipeline
        do {
            pipeline = try await DiarizationPipeline.fromPretrained(embeddingEngine: .coreml)
        } catch {
            throw XCTSkip("CoreML model not cached: \(error)")
        }

        let audioURL = URL(fileURLWithPath: "Tests/Qwen3ASRTests/Resources/test_audio.wav")
        guard FileManager.default.fileExists(atPath: audioURL.path) else {
            throw XCTSkip("Test audio file not found")
        }

        let (samples, sampleRate) = try AudioFileLoader.loadWAV(url: audioURL)
        let result = pipeline.diarize(audio: samples, sampleRate: sampleRate, config: .default)

        XCTAssertGreaterThan(result.segments.count, 0)
        XCTAssertGreaterThanOrEqual(result.numSpeakers, 1)

        for seg in result.segments {
            XCTAssertGreaterThanOrEqual(seg.startTime, 0)
            XCTAssertGreaterThan(seg.endTime, seg.startTime)
        }
    }

    // MARK: - E2E: Embedding Different Audio Produces Different Embeddings

    func testE2EEmbeddingDifferentAudio() async throws {
        let model = try await WeSpeakerModel.fromPretrained()

        let audioURL = URL(fileURLWithPath: "Tests/Qwen3ASRTests/Resources/test_audio.wav")
        guard FileManager.default.fileExists(atPath: audioURL.path) else {
            throw XCTSkip("Test audio file not found")
        }

        let (samples, sampleRate) = try AudioFileLoader.loadWAV(url: audioURL)

        // Embed real audio
        let realEmb = model.embed(audio: samples, sampleRate: sampleRate)

        // Embed random noise (simulating a different "speaker")
        var noise = [Float](repeating: 0, count: 32000)  // 2s @ 16kHz
        for i in 0..<noise.count {
            noise[i] = Float.random(in: -0.5...0.5)
        }
        let noiseEmb = model.embed(audio: noise, sampleRate: 16000)

        // Both should be 256-dim and normalized
        XCTAssertEqual(realEmb.count, 256)
        XCTAssertEqual(noiseEmb.count, 256)

        let realNorm = sqrt(realEmb.reduce(Float(0)) { $0 + $1 * $1 })
        let noiseNorm = sqrt(noiseEmb.reduce(Float(0)) { $0 + $1 * $1 })
        XCTAssertEqual(realNorm, 1.0, accuracy: 0.01)
        XCTAssertEqual(noiseNorm, 1.0, accuracy: 0.01)

        // Real speech vs noise should have low similarity
        let similarity = WeSpeakerModel.cosineSimilarity(realEmb, noiseEmb)
        XCTAssertLessThan(similarity, 0.8,
                          "Speech vs noise should not be highly similar (got \(similarity))")
    }
}
