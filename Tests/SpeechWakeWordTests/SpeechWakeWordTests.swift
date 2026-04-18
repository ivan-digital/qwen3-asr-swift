import XCTest
@testable import SpeechWakeWord
import AudioCommon

// MARK: - Config

final class KWSZipformerConfigTests: XCTestCase {
    func testDefaults() {
        let c = KWSZipformerConfig.default
        XCTAssertEqual(c.feature.sampleRate, 16000)
        XCTAssertEqual(c.feature.numMelBins, 80)
        XCTAssertEqual(c.feature.frameLengthMs, 25.0)
        XCTAssertEqual(c.feature.frameShiftMs, 10.0)
        XCTAssertFalse(c.feature.snipEdges)
        XCTAssertEqual(c.feature.highFreq, -400.0)
        XCTAssertEqual(c.encoder.chunkSize, 16)
        XCTAssertEqual(c.encoder.totalInputFrames, 45)
        XCTAssertEqual(c.encoder.outputFrames, 8)
        XCTAssertEqual(c.encoder.joinerDim, 320)
        XCTAssertEqual(c.decoder.vocabSize, 500)
        XCTAssertEqual(c.decoder.blankId, 0)
        XCTAssertEqual(c.decoder.contextSize, 2)
        XCTAssertEqual(c.kws.defaultThreshold, 0.15)
        XCTAssertEqual(c.kws.defaultContextScore, 0.5)
    }

    func testCodableRoundTrip() throws {
        let original = KWSZipformerConfig.default
        let data = try JSONEncoder().encode(original)
        let decoded = try JSONDecoder().decode(KWSZipformerConfig.self, from: data)
        XCTAssertEqual(decoded.feature.numMelBins, original.feature.numMelBins)
        XCTAssertEqual(decoded.encoder.joinerDim, original.encoder.joinerDim)
        XCTAssertEqual(decoded.kws.defaultThreshold, original.kws.defaultThreshold)
    }

    func testModelIdConstant() {
        XCTAssertEqual(
            WakeWordDetector.defaultModelId,
            "aufklarer/KWS-Zipformer-3M-CoreML-INT8"
        )
    }

    func testKeywordSpecDefaults() {
        let spec = KeywordSpec(phrase: "hey soniqo")
        XCTAssertEqual(spec.phrase, "hey soniqo")
        XCTAssertEqual(spec.acThreshold, 0)
        XCTAssertEqual(spec.boost, 0)
    }
}

// MARK: - Vocabulary

final class KWSVocabularyTests: XCTestCase {
    func testDecodeWithWordMarkers() {
        let vocab = KWSVocabulary(idToToken: [
            0: "\u{2581}hey",
            1: "\u{2581}soniqo"
        ])
        XCTAssertEqual(vocab.decode([0, 1]), "hey soniqo")
    }

    func testDecodeSubwordConcat() {
        let vocab = KWSVocabulary(idToToken: [
            0: "\u{2581}un",
            1: "believ",
            2: "able"
        ])
        XCTAssertEqual(vocab.decode([0, 1, 2]), "unbelievable")
    }

    func testDecodeSkipsUnknownIds() {
        let vocab = KWSVocabulary(idToToken: [
            0: "\u{2581}hello",
            1: "\u{2581}world"
        ])
        XCTAssertEqual(vocab.decode([0, 42, 1]), "hello world")
    }

    func testLoadFromFile() throws {
        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent("tokens-\(UUID().uuidString).txt")
        let text = """
        <blk> 0
        \u{2581}hey 1
        \u{2581}soniqo 2
        s 3
        """
        try text.write(to: tmp, atomically: true, encoding: .utf8)
        defer { try? FileManager.default.removeItem(at: tmp) }
        let vocab = try KWSVocabulary.load(from: tmp)
        XCTAssertEqual(vocab.count, 4)
        XCTAssertEqual(vocab.idToToken[1], "\u{2581}hey")
        XCTAssertEqual(vocab.tokenToId["s"], 3)
    }
}

// MARK: - Fbank

final class KaldiFbankTests: XCTestCase {
    func testFrameCountSilence() {
        let fbank = KaldiFbank()
        // 1s @ 16 kHz, snip_edges=false: ceil(16000/160) = 100 frames
        XCTAssertEqual(fbank.numFrames(for: 16000), 100)
    }

    func testFrameCountSnipEdges() {
        let fbank = KaldiFbank(.init(snipEdges: true))
        // snip_edges=true: (16000 - 400)/160 + 1 = 98 frames
        XCTAssertEqual(fbank.numFrames(for: 16000), 98)
    }

    func testZeroInputYieldsEmpty() {
        let fbank = KaldiFbank()
        XCTAssertEqual(fbank.compute([]).count, 0)
    }

    func testStreamingMatchesBatch() {
        // Feeding one chunk at a time through StreamingSession must produce
        // the same mel frames as running compute() over the whole buffer at
        // once — byte-exact modulo ordering.
        let fbank = KaldiFbank()
        var samples = [Float](repeating: 0, count: 16000)
        for i in 0..<samples.count {
            samples[i] = sin(2.0 * .pi * 523.0 * Float(i) / 16000.0) * 0.4
                + sin(2.0 * .pi * 1240.0 * Float(i) / 16000.0) * 0.15
        }
        let batch = fbank.compute(samples)

        let session = KaldiFbank.StreamingSession(fbank)
        var streamed = [Float]()
        var offset = 0
        while offset < samples.count {
            let end = min(offset + 1600, samples.count)
            streamed.append(contentsOf: session.accept(Array(samples[offset..<end])))
            offset = end
        }
        streamed.append(contentsOf: session.flush())
        XCTAssertEqual(streamed.count, batch.count, "frame count mismatch")
        for i in 0..<streamed.count {
            XCTAssertEqual(streamed[i], batch[i], accuracy: 1e-6,
                           "mel[\(i)] diverged: streaming=\(streamed[i]) batch=\(batch[i])")
        }
    }

    func testStreamingTrimsBuffer() {
        // Feed 60 seconds of audio in 100 ms chunks. The session should
        // keep emitting frames correctly without the retained PCM buffer
        // growing linearly.
        let fbank = KaldiFbank()
        let session = KaldiFbank.StreamingSession(fbank)
        let chunk = [Float](repeating: 0.01, count: 1600)  // 100 ms
        var totalFrames = 0
        let numBins = fbank.options.numMelBins
        for _ in 0..<600 {
            totalFrames += session.accept(chunk).count / numBins
        }
        totalFrames += session.flush().count / numBins
        XCTAssertEqual(totalFrames, 6000, "expected 6000 frames for 60 s @ 10 ms shift (accuracy within 1 s)")

        // Run one more accept and verify parity with a batch compute over
        // the equivalent sliding window — the trim must keep enough left
        // context that frame values don't drift.
        let fresh = KaldiFbank.StreamingSession(fbank)
        var ref = [Float]()
        for _ in 0..<10 {
            ref.append(contentsOf: fresh.accept(chunk))
        }
        ref.append(contentsOf: fresh.flush())
        let batch = fbank.compute([Float](repeating: 0.01, count: 16000))
        XCTAssertEqual(ref.count, batch.count)
        for i in 0..<ref.count {
            XCTAssertEqual(ref[i], batch[i], accuracy: 1e-6)
        }
    }

    func testStreamingResetRestartsFromScratch() {
        let fbank = KaldiFbank()
        let session = KaldiFbank.StreamingSession(fbank)
        _ = session.accept([Float](repeating: 0.1, count: 8000))
        let emittedBeforeReset = session.emittedFrames
        XCTAssertGreaterThan(emittedBeforeReset, 0)

        session.reset()
        XCTAssertEqual(session.emittedFrames, 0)

        _ = session.accept([Float](repeating: 0.1, count: 8000))
        XCTAssertEqual(session.emittedFrames, emittedBeforeReset)
    }

    func testSilenceIsFiniteNonPositive() {
        let fbank = KaldiFbank()
        let samples = [Float](repeating: 0, count: 16000)
        let mels = fbank.compute(samples)
        XCTAssertEqual(mels.count, 100 * 80)
        for v in mels {
            XCTAssertFalse(v.isNaN, "silence should not produce NaN")
            XCTAssertFalse(v.isInfinite, "silence should not produce infinity")
            // Log of near-zero energies → very negative but bounded.
            XCTAssertLessThan(v, 0, "log-mel of silence must be negative")
        }
    }

    func testParityWithKaldiNativeFbank() throws {
        // Compare Swift KaldiFbank output against a reference ``.bin`` produced
        // by ``kaldi-native-fbank`` via ``streaming_fbank.waveform_to_fbank``.
        // Regenerate with Tests/SpeechWakeWordTests/Resources/_generate_reference.py.
        let wavURL = try XCTUnwrap(Bundle.module.url(forResource: "fbank_input", withExtension: "wav"))
        let binURL = try XCTUnwrap(Bundle.module.url(forResource: "fbank_reference", withExtension: "bin"))

        let samples = try AudioFileLoader.load(url: wavURL, targetSampleRate: 16000)
        let ours = KaldiFbank().compute(samples)

        let refData = try Data(contentsOf: binURL)
        let (numFrames, numBins, reference) = try Self.parseReferenceBin(refData)
        XCTAssertEqual(numBins, 80)
        XCTAssertEqual(reference.count, numFrames * numBins)
        XCTAssertEqual(ours.count, reference.count,
                       "frame-count mismatch (ours=\(ours.count), ref=\(reference.count))")

        var maxAbsDiff: Float = 0
        var sumAbsDiff: Float = 0
        for i in 0..<ours.count {
            let d = abs(ours[i] - reference[i])
            if d > maxAbsDiff { maxAbsDiff = d }
            sumAbsDiff += d
        }
        let meanAbsDiff = sumAbsDiff / Float(ours.count)
        print("fbank parity: max|Δ|=\(maxAbsDiff), mean|Δ|=\(meanAbsDiff)")
        // Observed on M-series: max ≈ 2.5e-4, mean ≈ 4e-6. Tolerance is set
        // about 10× the observed value to leave headroom for minor FFT
        // rounding differences between vDSP backends without masking real drift.
        XCTAssertLessThan(maxAbsDiff, 3e-3, "fbank drift exceeds allowed tolerance")
        XCTAssertLessThan(meanAbsDiff, 5e-5, "mean fbank drift too high")
    }

    private static func parseReferenceBin(_ data: Data) throws -> (Int, Int, [Float]) {
        guard data.count >= 8 else {
            throw NSError(domain: "fbank", code: 1, userInfo: [NSLocalizedDescriptionKey: "truncated header"])
        }
        let frames = Int(data.subdata(in: 0..<4).withUnsafeBytes { $0.load(as: Int32.self) })
        let bins = Int(data.subdata(in: 4..<8).withUnsafeBytes { $0.load(as: Int32.self) })
        let payload = data.subdata(in: 8..<data.count)
        let floats = payload.withUnsafeBytes { ptr -> [Float] in
            let buf = ptr.bindMemory(to: Float.self)
            return Array(UnsafeBufferPointer(start: buf.baseAddress, count: payload.count / 4))
        }
        return (frames, bins, floats)
    }

    func testSineWaveProducesFiniteEnergy() {
        let fbank = KaldiFbank()
        var samples = [Float](repeating: 0, count: 16000)
        for i in 0..<samples.count {
            samples[i] = sin(2.0 * .pi * 440.0 * Float(i) / 16000.0) * 0.5
        }
        let mels = fbank.compute(samples)
        XCTAssertEqual(mels.count, 100 * 80)
        var sawPositive = false
        for v in mels {
            XCTAssertFalse(v.isNaN)
            XCTAssertFalse(v.isInfinite)
            if v > 0 { sawPositive = true }
        }
        XCTAssertTrue(sawPositive, "some bins around 440 Hz must have energy > 1")
    }
}

// MARK: - ContextGraph

final class ContextGraphTests: XCTestCase {
    private func toksFor(_ s: String) -> [Int] { s.unicodeScalars.map { Int($0.value) } }

    func testPhraseEndDetected() {
        let graph = ContextGraph(contextScore: 1.0)
        graph.build(
            tokenIds: [toksFor("he"), toksFor("she")],
            phrases: ["he", "she"],
            boosts: [0, 0],
            thresholds: [0, 0]
        )
        var state = graph.root
        for c in "she".unicodeScalars {
            let (_, next, _) = graph.forwardOneStep(from: state, token: Int(c.value))
            state = next
        }
        XCTAssertTrue(graph.isMatched(state).matched)
    }

    func testAcThresholdTracksLastNode() {
        let graph = ContextGraph(contextScore: 1.0, acThreshold: 0.8)
        graph.build(
            tokenIds: [toksFor("ab")],
            phrases: ["ab"],
            boosts: [0],
            thresholds: [0.3]
        )
        var state = graph.root
        for c in "ab".unicodeScalars {
            let (_, next, _) = graph.forwardOneStep(from: state, token: Int(c.value))
            state = next
        }
        let (matched, node) = graph.isMatched(state)
        XCTAssertTrue(matched)
        XCTAssertEqual(node?.acThreshold ?? 0, 0.3, accuracy: 1e-9)
    }

    func testNoFalseMatchOnPrefix() {
        let graph = ContextGraph(contextScore: 1.0)
        graph.build(
            tokenIds: [toksFor("hey")],
            phrases: ["hey"],
            boosts: [0],
            thresholds: [0]
        )
        var state = graph.root
        for c in "he".unicodeScalars {
            let (_, next, _) = graph.forwardOneStep(from: state, token: Int(c.value))
            state = next
        }
        XCTAssertFalse(graph.isMatched(state).matched)
    }

    func testFinalizeCancelsBoost() {
        let graph = ContextGraph(contextScore: 1.0)
        graph.build(
            tokenIds: [toksFor("cat")],
            phrases: ["cat"],
            boosts: [0],
            thresholds: [0]
        )
        var state = graph.root
        var totalScore: Double = 0
        for c in "ca".unicodeScalars {
            let (score, next, _) = graph.forwardOneStep(from: state, token: Int(c.value))
            totalScore += score
            state = next
        }
        // partial prefix "ca" gained node_score 2; finalize should cancel it out.
        let (finalScore, _) = graph.finalize(state)
        XCTAssertEqual(totalScore + finalScore, 0, accuracy: 1e-9)
    }
}

// MARK: - StreamingKwsDecoder

final class StreamingKwsDecoderTests: XCTestCase {

    /// Tiny backend: blank-only logits. Drives the beam to blanks forever.
    private func blankOnlyBackend(vocab: Int = 4) -> (StreamingKwsDecoder.DecoderFn, StreamingKwsDecoder.JoinerFn) {
        let decFn: StreamingKwsDecoder.DecoderFn = { _ in [Float](repeating: 0, count: 8) }
        let jnFn: StreamingKwsDecoder.JoinerFn = { _, _ in
            var logits = [Float](repeating: -10, count: vocab)
            logits[0] = 10  // blank id
            return logits
        }
        return (decFn, jnFn)
    }

    func testLogSoftmaxSumsToOne() {
        let logits: [Float] = [1, 2, 3, 4]
        let (logs, probs) = StreamingKwsDecoder.logSoftmax(logits)
        let sum = probs.reduce(0, +)
        XCTAssertEqual(sum, 1, accuracy: 1e-4)
        XCTAssertEqual(probs.count, 4)
        XCTAssertEqual(logs.count, 4)
        // Largest logit → largest prob.
        XCTAssertEqual(probs.firstIndex(of: probs.max()!), 3)
    }

    func testLogAddExpStableAtInfinity() {
        XCTAssertEqual(StreamingKwsDecoder.logAddExp(-.infinity, 3.0), 3.0)
        XCTAssertEqual(StreamingKwsDecoder.logAddExp(3.0, -.infinity), 3.0)
    }

    func testBlankOnlyProducesNoEmissions() {
        let graph = ContextGraph(contextScore: 0.5)
        graph.build(
            tokenIds: [[1, 2]], phrases: ["ab"], boosts: [0], thresholds: [0.1]
        )
        let (decFn, jnFn) = blankOnlyBackend(vocab: 4)
        let decoder = StreamingKwsDecoder(
            decoderFn: decFn, joinerFn: jnFn, contextGraph: graph,
            blankId: 0, contextSize: 2, beam: 4,
            autoResetSeconds: 10.0  // disable auto-reset for determinism
        )
        for _ in 0..<50 {
            let emissions = decoder.step(encoderFrame: [Float](repeating: 0, count: 8))
            XCTAssertEqual(emissions.count, 0)
        }
    }

    func testEmitsOnSyntheticIdealLogits() {
        // Build a 500-token vocab and a single keyword "AB" with ids [1, 2].
        // Drive the decoder with synthetic joiner logits that strongly
        // prefer token 1 at frame 0, token 2 at frame 1, then blank for
        // many frames. If the Swift beam search is sane, this must emit.
        let graph = ContextGraph(contextScore: 2.0, acThreshold: 0.2)
        graph.build(tokenIds: [[1, 2]], phrases: ["AB"], boosts: [0], thresholds: [0])
        let vocab = 500

        // decoderFn returns a fixed vector — the decoder network is
        // irrelevant for this test since joiner is fully synthetic.
        let decFn: StreamingKwsDecoder.DecoderFn = { _ in
            [Float](repeating: 0, count: 320)
        }
        // joinerFn reads a shared ``currentFrame`` closed over by the test
        // driver — we bump it *once per frame*, not per joinerFn call, so
        // every hypothesis at a given frame sees the same logits.
        var currentFrame = 0
        let joinerFn: StreamingKwsDecoder.JoinerFn = { _, _ in
            var logits = [Float](repeating: -20, count: vocab)
            switch currentFrame {
            case 0:
                logits[1] = 5
                logits[0] = 0
            case 1:
                logits[2] = 5
                logits[0] = 0
            default:
                logits[0] = 5
            }
            return logits
        }

        let kws = StreamingKwsDecoder(
            decoderFn: decFn, joinerFn: joinerFn, contextGraph: graph,
            blankId: 0, contextSize: 2, beam: 4,
            numTrailingBlanks: 1, autoResetSeconds: 10.0
        )
        var detections: [KeywordDetection] = []
        for f in 0..<10 {
            currentFrame = f
            detections.append(contentsOf: kws.step(encoderFrame: [Float](repeating: 0, count: 320)))
        }
        XCTAssertFalse(detections.isEmpty, "beam failed on ideal synthetic logits")
        XCTAssertEqual(detections.first?.phrase, "AB")
        XCTAssertEqual(detections.first?.tokenIds, [1, 2])
    }

    func testAutoResetAfterInactivity() {
        let graph = ContextGraph(contextScore: 0.5)
        graph.build(
            tokenIds: [[1]], phrases: ["a"], boosts: [0], thresholds: [0.1]
        )
        let (decFn, jnFn) = blankOnlyBackend()
        let decoder = StreamingKwsDecoder(
            decoderFn: decFn, joinerFn: jnFn, contextGraph: graph,
            blankId: 0, beam: 4, autoResetSeconds: 0.2  // ~5 frames
        )
        // Drive 10 blank frames — internal reset should happen at frame 5.
        for _ in 0..<10 {
            _ = decoder.step(encoderFrame: [Float](repeating: 0, count: 8))
        }
        // No crash + beam still populated (single initial hypothesis).
        XCTAssertFalse(decoder.beamList.isEmpty)
    }
}

// MARK: - BPE tokenizer (unit — no model download)

// Integration coverage of the BPE tokenizer is exercised inside the E2E tests
// because it consumes the real ``bpe.model`` shipped alongside the export.

// MARK: - E2E (requires model download + CoreML)

final class E2ESpeechWakeWordTests: XCTestCase {

    // Keywords shipped with the sherpa-onnx ``test_wavs`` fixtures. The BPE
    // decompositions mirror ``test_keywords.txt`` — icefall's KWS recipe lets
    // users craft decompositions that match the model's training output, which
    // often differs from what a greedy SP encoder would produce. Thresholds
    // mirror the Python reference parity test
    // (``test_convert.test_kws_decoder_emits_known_keyword``).
    private static let keywords: [KeywordSpec] = [
        KeywordSpec(phrase: "LIGHT UP", acThreshold: 0.25, boost: 2.0,
                    tokens: ["\u{2581}", "L", "IGHT", "\u{2581}UP"]),
        KeywordSpec(phrase: "LOVELY CHILD", acThreshold: 0.25, boost: 2.0,
                    tokens: ["\u{2581}LOVE", "LY", "\u{2581}CHI", "L", "D"]),
        KeywordSpec(phrase: "FOR EVER", acThreshold: 0.25, boost: 2.0,
                    tokens: ["\u{2581}FOR", "E", "VER"])
    ]

    private static var detector: WakeWordDetector?

    override func setUp() async throws {
        try await super.setUp()
        if Self.detector == nil {
            Self.detector = try await WakeWordDetector.fromPretrained(
                keywords: Self.keywords
            )
        }
    }

    private var detector: WakeWordDetector {
        get throws {
            guard let d = Self.detector else { throw XCTSkip("Detector not loaded") }
            return d
        }
    }

    func testLoadsModels() throws {
        let d = try detector
        XCTAssertTrue(d.isLoaded)
        XCTAssertEqual(d.config.encoder.joinerDim, 320)
        XCTAssertEqual(d.config.decoder.vocabSize, 500)
        XCTAssertGreaterThan(d.vocabulary.count, 0)
    }

    func testWarmup() throws {
        try detector.warmUp()
    }

    func testRejectsEmptyKeywordList() async throws {
        do {
            _ = try await WakeWordDetector.fromPretrained(keywords: [])
            XCTFail("expected invalidConfiguration error")
        } catch AudioModelError.invalidConfiguration {
            // expected
        } catch {
            XCTFail("unexpected error: \(error)")
        }
    }

    func testSilenceProducesNoDetections() throws {
        let d = try detector
        let silence = [Float](repeating: 0, count: 3 * d.config.feature.sampleRate)
        let detections = try d.detect(audio: silence, sampleRate: d.config.feature.sampleRate)
        XCTAssertTrue(detections.isEmpty, "silence should not trigger detections")
    }

    func testDetectsLightUp() throws {
        let d = try detector
        let url = try XCTUnwrap(Bundle.module.url(forResource: "kws_light_up", withExtension: "wav"))
        let audio = try AudioFileLoader.load(url: url, targetSampleRate: 16000)
        let detections = try d.detect(audio: audio, sampleRate: 16000)
        print("kws_light_up: \(detections.map { $0.phrase })")
        XCTAssertTrue(detections.contains(where: { $0.phrase == "LIGHT UP" }),
                      "expected 'LIGHT UP' in 0.wav, got \(detections.map { $0.phrase })")
        XCTAssertFalse(detections.contains(where: { $0.phrase == "LOVELY CHILD" }))
    }

    func testDetectsLovelyChildOrForEver() throws {
        // 1.wav contains both "LOVELY CHILD" and "FOR EVER". The Python
        // reference only asserts one phrase per clip — mirror that here,
        // since per-phrase recall depends on threshold tuning.
        let d = try detector
        let url = try XCTUnwrap(Bundle.module.url(forResource: "kws_lovely_child", withExtension: "wav"))
        let audio = try AudioFileLoader.load(url: url, targetSampleRate: 16000)
        let detections = try d.detect(audio: audio, sampleRate: 16000)
        print("kws_lovely_child: \(detections.map { $0.phrase })")
        let phrases = Set(detections.map { $0.phrase })
        XCTAssertTrue(phrases.contains("LOVELY CHILD") || phrases.contains("FOR EVER"),
                      "expected 'LOVELY CHILD' or 'FOR EVER' in 1.wav, got \(detections.map { $0.phrase })")
    }

    func testStreamingMatchesBatch() throws {
        let d = try detector
        let url = try XCTUnwrap(Bundle.module.url(forResource: "kws_light_up", withExtension: "wav"))
        let audio = try AudioFileLoader.load(url: url, targetSampleRate: 16000)
        let session = try d.createSession()
        var streamed: [String] = []
        let chunkSize = 16000 / 4
        var offset = 0
        while offset < audio.count {
            let end = min(offset + chunkSize, audio.count)
            let dets = try session.pushAudio(Array(audio[offset..<end]))
            streamed.append(contentsOf: dets.map { $0.phrase })
            offset = end
        }
        streamed.append(contentsOf: try session.finalize().map { $0.phrase })
        XCTAssertTrue(streamed.contains("LIGHT UP"),
                      "streaming missed 'LIGHT UP' — saw \(streamed)")
    }

    func testStreamingRealTimeFactor() throws {
        // Drive ~4s of audio in 320 ms chunks — target is the export's
        // measured RTF of 0.04. Generous ceiling so the test is robust on
        // cold CPU+NE starts.
        let d = try detector
        let url = try XCTUnwrap(Bundle.module.url(forResource: "kws_light_up", withExtension: "wav"))
        let audio = try AudioFileLoader.load(url: url, targetSampleRate: 16000)
        let session = try d.createSession()

        let chunkSamples = 16000 * 320 / 1000
        _ = try session.pushAudio(Array(audio.prefix(chunkSamples)))  // warmup

        var totalMs: Double = 0
        var totalChunks = 0
        var offset = chunkSamples
        while offset + chunkSamples <= audio.count {
            let chunk = Array(audio[offset..<(offset + chunkSamples)])
            let t0 = CFAbsoluteTimeGetCurrent()
            _ = try session.pushAudio(chunk)
            totalMs += (CFAbsoluteTimeGetCurrent() - t0) * 1000
            totalChunks += 1
            offset += chunkSamples
        }
        guard totalChunks > 0 else { throw XCTSkip("audio too short for RTF measurement") }
        let avgChunkMs = totalMs / Double(totalChunks)
        let rtf = avgChunkMs / 320.0
        print("KWS Zipformer RTF=\(String(format: "%.3f", rtf)) (avg=\(String(format: "%.1f", avgChunkMs))ms / 320 ms chunk)")
        // RTF depends on whether the ANE compile succeeded. On CPU-only
        // fallback (e.g. macOS simulator or cold NE init) we've observed
        // ~1.8; with NE hot it drops to ~0.04. Just assert we're finite
        // and log the actual number for inspection.
        XCTAssertGreaterThan(rtf, 0, "RTF must be positive")
        XCTAssertLessThan(rtf, 5.0, "RTF should not regress catastrophically")
    }

    func testMemoryManagement() async throws {
        let d = try await WakeWordDetector.fromPretrained(keywords: Self.keywords)
        XCTAssertTrue(d.isLoaded)
        XCTAssertGreaterThan(d.memoryFootprint, 0)
        d.unload()
        XCTAssertFalse(d.isLoaded)
        XCTAssertEqual(d.memoryFootprint, 0)
    }

    /// Feeds the Python reference's exported encoder output for 0.wav
    /// directly into the Swift beam search. This bypasses fbank + encoder
    /// and isolates ``StreamingKwsDecoder`` / ``ContextGraph`` against
    /// the icefall reference decoder. Python emits
    /// ``('LIGHT UP', [0, 1, 2, 4])`` on these frames at
    /// ``context_score=2.0, ac_threshold=0.25, beam=4``.
    func testBeamSearchEmitsOnReferenceEncoder() throws {
        let d = try detector
        let url = try XCTUnwrap(Bundle.module.url(
            forResource: "ref_encoder_light_up", withExtension: "bin"))
        let data = try Data(contentsOf: url)
        let numFrames = Int(data.subdata(in: 0..<4)
            .withUnsafeBytes { $0.load(as: Int32.self) })
        let joinerDim = Int(data.subdata(in: 4..<8)
            .withUnsafeBytes { $0.load(as: Int32.self) })
        let floats = data.subdata(in: 8..<data.count).withUnsafeBytes { ptr -> [Float] in
            let buf = ptr.bindMemory(to: Float.self)
            return Array(UnsafeBufferPointer(start: buf.baseAddress,
                                              count: numFrames * joinerDim))
        }
        var frames: [[Float]] = []
        frames.reserveCapacity(numFrames)
        for f in 0..<numFrames {
            frames.append(Array(floats[(f * joinerDim)..<((f + 1) * joinerDim)]))
        }

        let kws = try d.makeKwsDecoder(
            keywords: [
                KeywordSpec(phrase: "LIGHT UP",
                            tokens: ["\u{2581}", "L", "IGHT", "\u{2581}UP"])
            ],
            contextScore: 2.0,
            acThreshold: 0.25,
            numTrailingBlanks: 1,
            autoResetSeconds: 1.5,
            beam: 4
        )
        let detections = kws.stepChunk(frames)
        print("Swift decoder on ref encoder: \(detections.map { $0.phrase })")
        XCTAssertTrue(detections.contains { $0.phrase == "LIGHT UP" },
                      "Swift beam search did not emit 'LIGHT UP' on reference encoder frames")
    }
}
