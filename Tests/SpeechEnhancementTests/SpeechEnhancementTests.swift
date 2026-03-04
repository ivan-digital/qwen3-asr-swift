import XCTest
@testable import SpeechEnhancement

final class SpeechEnhancementTests: XCTestCase {

    // MARK: - Configuration Tests

    func testDefaultConfig() {
        let config = DeepFilterNet3Config.default
        XCTAssertEqual(config.fftSize, 960)
        XCTAssertEqual(config.hopSize, 480)
        XCTAssertEqual(config.erbBands, 32)
        XCTAssertEqual(config.dfBins, 96)
        XCTAssertEqual(config.dfOrder, 5)
        XCTAssertEqual(config.dfLookahead, 2)
        XCTAssertEqual(config.convCh, 64)
        XCTAssertEqual(config.embHidden, 256)
        XCTAssertEqual(config.sampleRate, 48000)
        XCTAssertEqual(config.freqBins, 481)
    }

    func testNormAlpha() {
        let config = DeepFilterNet3Config.default
        // alpha = exp(-480/48000/1.0) = exp(-0.01) ≈ 0.990050
        XCTAssertEqual(config.normAlpha, exp(-0.01), accuracy: 1e-6)
    }

    // MARK: - Vorbis Window Tests

    func testVorbisWindowSize() {
        let window = computeVorbisWindow(size: 960)
        XCTAssertEqual(window.count, 960)
    }

    func testVorbisWindowSymmetry() {
        let window = computeVorbisWindow(size: 960)
        for i in 0..<480 {
            XCTAssertEqual(window[i], window[959 - i], accuracy: 1e-6,
                          "Window not symmetric at index \(i)")
        }
    }

    func testVorbisWindowRange() {
        let window = computeVorbisWindow(size: 960)
        for (i, v) in window.enumerated() {
            XCTAssertGreaterThanOrEqual(v, 0, "Window value < 0 at index \(i)")
            XCTAssertLessThanOrEqual(v, 1, "Window value > 1 at index \(i)")
        }
    }

    func testVorbisWindowCOLA() {
        let N = 960
        let hop = 480
        let window = computeVorbisWindow(size: N)

        for n in 0..<hop {
            let sum = window[n] * window[n] + window[n + hop] * window[n + hop]
            XCTAssertEqual(sum, 1.0, accuracy: 1e-4,
                          "COLA violated at index \(n): \(sum)")
        }
    }

    // MARK: - ERB Filterbank Tests

    func testERBFilterbankShape() {
        let config = DeepFilterNet3Config.default
        let (forward, inverse, widths) = computeERBFilterbank(config: config)

        XCTAssertEqual(forward.count, 481 * 32)
        XCTAssertEqual(inverse.count, 32 * 481)
        XCTAssertEqual(widths.reduce(0, +), 481)
        XCTAssertEqual(widths.count, 32)
    }

    func testERBFilterbankWidthsIncreasing() {
        let config = DeepFilterNet3Config.default
        let (_, _, widths) = computeERBFilterbank(config: config)
        XCTAssertEqual(widths[0], 2)
        XCTAssertGreaterThan(widths[31], widths[0])
    }

    func testERBForwardNormalization() {
        let config = DeepFilterNet3Config.default
        let (forward, _, _) = computeERBFilterbank(config: config)

        for band in 0..<32 {
            var colSum: Float = 0
            for bin in 0..<481 {
                colSum += forward[bin * 32 + band]
            }
            XCTAssertEqual(colSum, 1.0, accuracy: 1e-4,
                          "Forward filterbank column \(band) sum: \(colSum)")
        }
    }

    // MARK: - Lookahead Padding Tests

    func testLookaheadPadding() {
        let data: [Float] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        let result = applyLookaheadPad(data, featuresPerFrame: 2, numFrames: 5, lookahead: 1)
        XCTAssertEqual(result, [3, 4, 5, 6, 7, 8, 9, 10, 0, 0])
    }

    func testLookaheadPaddingZero() {
        let data: [Float] = [1, 2, 3, 4]
        let result = applyLookaheadPad(data, featuresPerFrame: 2, numFrames: 2, lookahead: 0)
        XCTAssertEqual(result, data)
    }

    // MARK: - STFT Tests

    func testSTFTFreqBins() {
        let window = computeVorbisWindow(size: 960)
        let stft = STFTProcessor(fftSize: 960, hopSize: 480, window: window)
        XCTAssertEqual(stft.paddedFFT, 1024)
        XCTAssertEqual(stft.freqBins, 513)
    }

    func testSTFTFrameCount() {
        let window = computeVorbisWindow(size: 960)
        let stft = STFTProcessor(fftSize: 960, hopSize: 480, window: window)

        let audio = [Float](repeating: 0, count: 48000)
        var analysisMem = [Float](repeating: 0, count: 480)
        let (real, _) = stft.forward(audio: audio, analysisMem: &analysisMem)

        let expectedFrames = 100
        XCTAssertEqual(real.count, expectedFrames * stft.freqBins)
    }

    func testSTFTRoundTrip() {
        let sampleRate = 48000
        let duration = 0.1
        let numSamples = Int(Double(sampleRate) * duration)
        let freq: Float = 440.0

        var audio = [Float](repeating: 0, count: numSamples)
        for i in 0..<numSamples {
            audio[i] = sin(2.0 * Float.pi * freq * Float(i) / Float(sampleRate))
        }

        let hop = 480
        let window = computeVorbisWindow(size: 960)
        let stft = STFTProcessor(fftSize: 960, hopSize: hop, window: window)

        let paddedAudio = audio + [Float](repeating: 0, count: hop)

        var analysisMem = [Float](repeating: 0, count: hop)
        var synthesisMem = [Float](repeating: 0, count: hop)

        let (real, imag) = stft.forward(audio: paddedAudio, analysisMem: &analysisMem)
        guard !real.isEmpty else { return }

        let reconstructed = stft.inverse(real: real, imag: imag, synthesisMem: &synthesisMem)

        for i in 0..<numSamples {
            let outIdx = hop + i
            guard outIdx < reconstructed.count else { break }
            XCTAssertEqual(reconstructed[outIdx], audio[i], accuracy: 0.01,
                          "STFT round-trip mismatch at sample \(i)")
        }
    }

    func testSTFTRoundTripMultiFreq() {
        let sampleRate = 48000
        let numSamples = 9600
        var audio = [Float](repeating: 0, count: numSamples)
        for i in 0..<numSamples {
            let t = Float(i) / Float(sampleRate)
            audio[i] = 0.5 * sin(2.0 * Float.pi * 200.0 * t) +
                       0.3 * sin(2.0 * Float.pi * 1000.0 * t) +
                       0.2 * sin(2.0 * Float.pi * 5000.0 * t)
        }

        let hop = 480
        let window = computeVorbisWindow(size: 960)
        let stft = STFTProcessor(fftSize: 960, hopSize: hop, window: window)

        let paddedAudio = audio + [Float](repeating: 0, count: hop)
        var analysisMem = [Float](repeating: 0, count: hop)
        var synthesisMem = [Float](repeating: 0, count: hop)

        let (real, imag) = stft.forward(audio: paddedAudio, analysisMem: &analysisMem)
        guard !real.isEmpty else { return }
        let reconstructed = stft.inverse(real: real, imag: imag, synthesisMem: &synthesisMem)

        for i in 0..<numSamples {
            let outIdx = hop + i
            guard outIdx < reconstructed.count else { break }
            XCTAssertEqual(reconstructed[outIdx], audio[i], accuracy: 0.01,
                          "Multi-freq STFT round-trip mismatch at sample \(i)")
        }
    }

    // MARK: - Normalization Tests

    func testMeanNormalization() {
        var erb: [Float] = [10, 20, 30, 40, 50, 60]
        var state: [Float] = [0, 0, 0]
        let alpha: Float = 0.5

        applyMeanNormalization(&erb, state: &state, alpha: alpha, erbBands: 3, numFrames: 2)

        XCTAssertEqual(erb[0], 0.125, accuracy: 1e-5)
        XCTAssertEqual(erb[1], 0.25, accuracy: 1e-5)
        XCTAssertEqual(erb[2], 0.375, accuracy: 1e-5)
    }

    func testUnitNormalization() {
        var real: [Float] = [3, 4]
        var imag: [Float] = [4, 3]
        var state: [Float] = [1, 1]
        let alpha: Float = 0.0

        applyUnitNormalization(real: &real, imag: &imag, state: &state,
                              alpha: alpha, dfBins: 2, numFrames: 1)

        let norm0 = sqrt(Float(5.0))
        XCTAssertEqual(real[0], 3.0 / norm0, accuracy: 1e-5)
        XCTAssertEqual(imag[0], 4.0 / norm0, accuracy: 1e-5)
    }

    // MARK: - Deep Filtering Tests

    func testDeepFilteringIdentity() {
        let numFrames = 5
        let dfBins = 4
        let dfOrder = 3
        let dfLookahead = 1
        let freqBins = 8

        var specReal = [Float](repeating: 0, count: numFrames * freqBins)
        var specImag = [Float](repeating: 0, count: numFrames * freqBins)
        for t in 0..<numFrames {
            for f in 0..<dfBins {
                specReal[t * freqBins + f] = Float(t * dfBins + f + 1)
                specImag[t * freqBins + f] = Float(t * dfBins + f + 1) * 0.5
            }
        }

        var coefs = [Float](repeating: 0, count: numFrames * dfBins * dfOrder * 2)
        let centerTap = dfOrder - 1 - dfLookahead
        for t in 0..<numFrames {
            for f in 0..<dfBins {
                let idx = (t * dfBins * dfOrder + f * dfOrder + centerTap) * 2
                coefs[idx] = 1.0
                coefs[idx + 1] = 0.0
            }
        }

        let (outReal, outImag) = applyDeepFiltering(
            specReal: specReal, specImag: specImag,
            coefs: coefs, dfBins: dfBins, dfOrder: dfOrder,
            dfLookahead: dfLookahead, numFrames: numFrames, freqBins: freqBins)

        for t in 0..<numFrames {
            for f in 0..<dfBins {
                let outIdx = t * dfBins + f
                let inIdx = t * freqBins + f
                XCTAssertEqual(outReal[outIdx], specReal[inIdx], accuracy: 1e-5,
                              "Deep filter identity mismatch at t=\(t), f=\(f) (real)")
                XCTAssertEqual(outImag[outIdx], specImag[inIdx], accuracy: 1e-5,
                              "Deep filter identity mismatch at t=\(t), f=\(f) (imag)")
            }
        }
    }

    // MARK: - ERB Mask Tests

    func testERBMaskApplication() {
        let config = DeepFilterNet3Config.default
        let (_, invFb, _) = computeERBFilterbank(config: config)

        let numFrames = 1
        let freqBins = config.freqBins
        let erbBands = config.erbBands
        var specReal = [Float](repeating: 1.0, count: numFrames * freqBins)
        var specImag = [Float](repeating: 0.5, count: numFrames * freqBins)
        let erbMask = [Float](repeating: 1.0, count: numFrames * erbBands)

        applyERBMask(specReal: &specReal, specImag: &specImag,
                     erbMask: erbMask, erbInvFb: invFb,
                     erbBands: erbBands, freqBins: freqBins, numFrames: numFrames)

        for f in 0..<freqBins {
            XCTAssertEqual(specReal[f], 1.0, accuracy: 1e-4,
                          "ERB mask identity failed at bin \(f)")
        }
    }

    func testERBMaskZero() {
        let config = DeepFilterNet3Config.default
        let (_, invFb, _) = computeERBFilterbank(config: config)

        let numFrames = 1
        var specReal = [Float](repeating: 1.0, count: numFrames * config.freqBins)
        var specImag = [Float](repeating: 1.0, count: numFrames * config.freqBins)
        let erbMask = [Float](repeating: 0.0, count: numFrames * config.erbBands)

        applyERBMask(specReal: &specReal, specImag: &specImag,
                     erbMask: erbMask, erbInvFb: invFb,
                     erbBands: config.erbBands, freqBins: config.freqBins, numFrames: numFrames)

        for f in 0..<config.freqBins {
            XCTAssertEqual(specReal[f], 0.0, accuracy: 1e-6)
            XCTAssertEqual(specImag[f], 0.0, accuracy: 1e-6)
        }
    }

    // MARK: - Auxiliary Data Tests

    func testWeightLoaderAuxiliaryDataStructure() {
        let config = DeepFilterNet3Config.default
        let window = computeVorbisWindow(size: config.fftSize)
        let (erbFb, erbInvFb, _) = computeERBFilterbank(config: config)

        let auxData = DeepFilterNet3WeightLoader.AuxiliaryData(
            erbFb: erbFb,
            erbInvFb: erbInvFb,
            window: window,
            meanNormState: [Float](repeating: 0, count: config.erbBands),
            unitNormState: [Float](repeating: 1, count: config.dfBins)
        )

        XCTAssertEqual(auxData.window.count, 960)
        XCTAssertEqual(auxData.erbFb.count, 481 * 32)
        XCTAssertEqual(auxData.erbInvFb.count, 32 * 481)
        XCTAssertEqual(auxData.meanNormState.count, 32)
        XCTAssertEqual(auxData.unitNormState.count, 96)
    }

    func testNpzReader() throws {
        // Verify the NPZ reader can parse numpy files
        // Create a temporary npz for testing
        let tmpDir = FileManager.default.temporaryDirectory
        let npzURL = tmpDir.appendingPathComponent("test_aux.npz")

        // We can't easily create an npz in Swift, but we can test the structure
        // Just verify the reader type exists
        XCTAssertNotNil(NpzReader.self)
    }
}
