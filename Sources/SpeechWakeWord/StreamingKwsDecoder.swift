import Foundation

/// A single keyword emission from the streaming decoder.
public struct KeywordDetection: Sendable, Equatable {
    /// Human-readable phrase (e.g. "hey soniqo").
    public let phrase: String
    /// BPE token ids that matched the phrase.
    public let tokenIds: [Int]
    /// Encoder frame indices the tokens were emitted at (40 ms / frame).
    public let timestamps: [Int]
    /// Encoder frame index at which the emission fired.
    public let frameIndex: Int

    /// Detection time in seconds, given a ``frameShiftSeconds``.
    public func time(frameShiftSeconds: Double) -> Double {
        Double(frameIndex) * frameShiftSeconds
    }
}

/// Port of ``kws_decoder.StreamingKwsDecoder`` — single-stream modified beam
/// search over a stateless transducer, with an Aho-Corasick ``ContextGraph``
/// boosting registered keywords.
///
/// The backend is abstract: the caller supplies ``decoderFn`` and ``joinerFn``
/// closures so the same decoder can drive CoreML, PyTorch reference, or a
/// stubbed backend in tests.
public final class StreamingKwsDecoder {
    public typealias DecoderFn = ([Int]) -> [Float]
    public typealias JoinerFn = ([Float], [Float]) -> [Float]

    public let contextGraph: ContextGraph
    public let blankId: Int
    public let unkId: Int
    public let contextSize: Int
    public let beam: Int
    public let numTrailingBlanks: Int
    public let blankPenalty: Float
    public let frameShiftSeconds: Double
    public let autoResetFrames: Int

    private let decoderFn: DecoderFn
    private let joinerFn: JoinerFn

    private var decCache: [[Int]: [Float]] = [:]
    private(set) var beamList: [Hypothesis] = []
    private var t: Int = 0
    private var framesSinceEmission: Int = 0

    public init(
        decoderFn: @escaping DecoderFn,
        joinerFn: @escaping JoinerFn,
        contextGraph: ContextGraph,
        blankId: Int = 0,
        unkId: Int? = nil,
        contextSize: Int = 2,
        beam: Int = 4,
        numTrailingBlanks: Int = 1,
        blankPenalty: Float = 0,
        frameShiftSeconds: Double = 0.04,
        autoResetSeconds: Double = 1.5
    ) {
        self.decoderFn = decoderFn
        self.joinerFn = joinerFn
        self.contextGraph = contextGraph
        self.blankId = blankId
        self.unkId = unkId ?? blankId
        self.contextSize = contextSize
        self.beam = beam
        self.numTrailingBlanks = numTrailingBlanks
        self.blankPenalty = blankPenalty
        self.frameShiftSeconds = frameShiftSeconds
        self.autoResetFrames = max(1, Int((autoResetSeconds / frameShiftSeconds).rounded()))
        reset()
    }

    // MARK: - Hypothesis

    public struct Hypothesis {
        public var ys: [Int]
        public var logProb: Double
        public var acProbs: [Double]
        public var timestamps: [Int]
        public var contextState: ContextGraph.State
        public var numTailingBlanks: Int

        public var key: String {
            // Reuse the same dict-key scheme as upstream: join ys with `_`.
            return ys.map(String.init).joined(separator: "_")
        }
    }

    // MARK: - state management

    public func reset() {
        t = 0
        framesSinceEmission = 0
        decCache.removeAll(keepingCapacity: true)
        let initYs = Array(repeating: -1, count: max(contextSize - 1, 0)) + [blankId]
        beamList = [
            Hypothesis(
                ys: initYs,
                logProb: 0,
                acProbs: [],
                timestamps: [],
                contextState: contextGraph.root,
                numTailingBlanks: 0
            )
        ]
    }

    /// Advance one encoder output frame (already in joiner space).
    public func step(encoderFrame: [Float]) -> [KeywordDetection] {
        var emissions: [KeywordDetection] = []

        // Expand beam across candidate tokens.
        struct Candidate {
            let totalLogProb: Double
            let hypIndex: Int
            let token: Int
            let tokenProb: Double
        }
        var candidates: [Candidate] = []
        candidates.reserveCapacity(beamList.count * 32)

        for (i, hyp) in beamList.enumerated() {
            let decOut = decoderFor(hyp.ys)
            var logits = joinerFn(encoderFrame, decOut)
            if blankPenalty != 0, blankId < logits.count {
                logits[blankId] -= blankPenalty
            }
            let (logProbs, probs) = Self.logSoftmax(logits)
            for token in 0..<logProbs.count {
                candidates.append(
                    Candidate(
                        totalLogProb: hyp.logProb + Double(logProbs[token]),
                        hypIndex: i,
                        token: token,
                        tokenProb: Double(probs[token])
                    )
                )
            }
        }

        candidates.sort { $0.totalLogProb > $1.totalLogProb }
        let topK = candidates.prefix(beam)

        var nextBeam: [String: Hypothesis] = [:]
        for cand in topK {
            var hyp = beamList[cand.hypIndex]
            hyp.numTailingBlanks += 1

            var contextScore: Double = 0
            if cand.token != blankId && cand.token != unkId {
                hyp.ys.append(cand.token)
                hyp.timestamps.append(t)
                hyp.acProbs.append(cand.tokenProb)
                let (boost, next, _) = contextGraph.forwardOneStep(
                    from: hyp.contextState, token: cand.token
                )
                contextScore = boost
                hyp.contextState = next
                hyp.numTailingBlanks = 0
                if next.token == -1 {
                    // Rewind BPE prefix back to initial when we drop back to root.
                    let tail = hyp.ys.suffix(contextSize)
                    let replacement =
                        Array(repeating: -1, count: max(contextSize - 1, 0)) + [blankId]
                    hyp.ys.removeLast(tail.count)
                    hyp.ys.append(contentsOf: replacement)
                }
            }
            hyp.logProb = cand.totalLogProb + contextScore

            let key = hyp.key
            if var existing = nextBeam[key] {
                existing.logProb = Self.logAddExp(existing.logProb, hyp.logProb)
                nextBeam[key] = existing
            } else {
                nextBeam[key] = hyp
            }
        }
        beamList = Array(nextBeam.values)

        // Check emission on most-probable hypothesis (length-normalized).
        let top = beamList.max { a, b in
            let an = a.logProb / Double(max(a.ys.count, 1))
            let bn = b.logProb / Double(max(b.ys.count, 1))
            return an < bn
        }

        if let top, let matched = contextGraph.isMatched(top.contextState).state {
            let level = matched.level
            if level > 0 && top.acProbs.count >= level {
                let window = top.acProbs.suffix(level)
                let acProb = window.reduce(0, +) / Double(level)
                if top.numTailingBlanks > numTrailingBlanks && acProb >= matched.acThreshold {
                    let tokens = Array(top.ys.suffix(level))
                    let timestamps = Array(top.timestamps.suffix(level))
                    emissions.append(
                        KeywordDetection(
                            phrase: matched.phrase,
                            tokenIds: tokens,
                            timestamps: timestamps,
                            frameIndex: t
                        )
                    )
                    reset()
                    t += 1
                    framesSinceEmission = 0
                    return emissions
                }
            }
        }

        t += 1
        if emissions.isEmpty {
            framesSinceEmission += 1
            if framesSinceEmission >= autoResetFrames {
                reset()
            }
        } else {
            framesSinceEmission = 0
        }
        return emissions
    }

    /// Convenience: iterate across a chunk of encoder frames ``[numFrames][joinerDim]``.
    public func stepChunk(_ frames: [[Float]]) -> [KeywordDetection] {
        var out: [KeywordDetection] = []
        for frame in frames {
            out.append(contentsOf: step(encoderFrame: frame))
        }
        return out
    }

    // MARK: - helpers

    private func decoderFor(_ ys: [Int]) -> [Float] {
        let ctx = Array(ys.suffix(contextSize))
        if let cached = decCache[ctx] { return cached }
        let value = decoderFn(ctx)
        decCache[ctx] = value
        return value
    }

    static func logAddExp(_ a: Double, _ b: Double) -> Double {
        if a == -Double.infinity { return b }
        if b == -Double.infinity { return a }
        let m = max(a, b)
        return m + log1p(exp(-abs(a - b)))
    }

    static func logSoftmax(_ logits: [Float]) -> (log: [Float], prob: [Float]) {
        guard !logits.isEmpty else { return ([], []) }
        let m = logits.max() ?? 0
        var exps = [Float](repeating: 0, count: logits.count)
        var s: Float = 0
        for i in 0..<logits.count {
            let e = Foundation.exp(logits[i] - m)
            exps[i] = e
            s += e
        }
        var logs = [Float](repeating: 0, count: logits.count)
        var probs = [Float](repeating: 0, count: logits.count)
        for i in 0..<logits.count {
            let p = exps[i] / s
            probs[i] = p
            logs[i] = p > 0 ? Foundation.log(p) : -.infinity
        }
        return (logs, probs)
    }
}
