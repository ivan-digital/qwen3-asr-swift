import Foundation

/// Aho-Corasick trie over BPE token ids with keyword-phrase acoustic thresholds.
///
/// Port of icefall's ``context_graph.py`` (Apache-2.0, upstream vendored under
/// ``models/kws-zipformer/export/icefall_kws`` in ``soniqo/speech-models``).
/// Used by :class:`StreamingKwsDecoder` to boost phrases during modified-beam
/// search and to detect end-of-phrase transitions.
public final class ContextGraph {
    public final class State {
        public let id: Int
        public let token: Int
        public var tokenScore: Double
        public var nodeScore: Double
        public var outputScore: Double
        public var isEnd: Bool
        public let level: Int
        public var phrase: String
        public var acThreshold: Double
        public var next: [Int: State] = [:]
        // Reference cycles are fine — graph is owned for the detector's lifetime.
        public var fail: State?
        public var output: State?

        init(
            id: Int,
            token: Int,
            tokenScore: Double,
            nodeScore: Double,
            outputScore: Double,
            isEnd: Bool,
            level: Int,
            phrase: String = "",
            acThreshold: Double = 1.0
        ) {
            self.id = id
            self.token = token
            self.tokenScore = tokenScore
            self.nodeScore = nodeScore
            self.outputScore = outputScore
            self.isEnd = isEnd
            self.level = level
            self.phrase = phrase
            self.acThreshold = acThreshold
        }
    }

    public let contextScore: Double
    public let acThreshold: Double
    public let root: State
    private(set) var numNodes: Int = 0

    public init(contextScore: Double, acThreshold: Double = 1.0) {
        self.contextScore = contextScore
        self.acThreshold = acThreshold
        self.root = State(
            id: 0,
            token: -1,
            tokenScore: 0,
            nodeScore: 0,
            outputScore: 0,
            isEnd: false,
            level: 0
        )
        self.root.fail = self.root
    }

    /// Add phrases with per-phrase BPE-token sequences.
    /// - Parameters:
    ///   - tokenIds: One int list per phrase.
    ///   - phrases: Display strings, same length as ``tokenIds``.
    ///   - boosts: Per-phrase score override; 0 → use ``contextScore``.
    ///   - thresholds: Per-phrase acoustic threshold override; 0 → use ``acThreshold``.
    public func build(
        tokenIds: [[Int]],
        phrases: [String],
        boosts: [Double],
        thresholds: [Double]
    ) {
        precondition(tokenIds.count == phrases.count)
        precondition(boosts.count == tokenIds.count)
        precondition(thresholds.count == tokenIds.count)

        for index in 0..<tokenIds.count {
            let tokens = tokenIds[index]
            guard !tokens.isEmpty else { continue }
            let phrase = phrases[index]
            let score = boosts[index] == 0 ? contextScore : boosts[index]
            let threshold = thresholds[index] == 0 ? acThreshold : thresholds[index]

            var node = root
            for (i, token) in tokens.enumerated() {
                let isEnd = (i == tokens.count - 1)
                if let existing = node.next[token] {
                    existing.tokenScore = max(score, existing.tokenScore)
                    existing.nodeScore = node.nodeScore + existing.tokenScore
                    let combinedIsEnd = existing.isEnd || isEnd
                    existing.outputScore = combinedIsEnd ? existing.nodeScore : 0
                    existing.isEnd = combinedIsEnd
                    if isEnd {
                        existing.phrase = phrase
                        existing.acThreshold = threshold
                    }
                    node = existing
                } else {
                    numNodes += 1
                    let nodeScore = node.nodeScore + score
                    let newNode = State(
                        id: numNodes,
                        token: token,
                        tokenScore: score,
                        nodeScore: nodeScore,
                        outputScore: isEnd ? nodeScore : 0,
                        isEnd: isEnd,
                        level: i + 1,
                        phrase: isEnd ? phrase : "",
                        acThreshold: isEnd ? threshold : 0.0
                    )
                    node.next[token] = newNode
                    node = newNode
                }
            }
        }
        fillFailAndOutput()
    }

    private func fillFailAndOutput() {
        var queue: [State] = []
        for (_, node) in root.next {
            node.fail = root
            queue.append(node)
        }
        var cursor = 0
        while cursor < queue.count {
            let current = queue[cursor]
            cursor += 1
            for (token, node) in current.next {
                var fail = current.fail ?? root
                if let nxt = fail.next[token] {
                    fail = nxt
                } else {
                    fail = fail.fail ?? root
                    while fail.next[token] == nil {
                        guard fail.token != -1 else { break }
                        fail = fail.fail ?? root
                    }
                    if let nxt = fail.next[token] { fail = nxt }
                }
                node.fail = fail

                var output: State? = node.fail
                while let o = output, !o.isEnd {
                    if let nextFail = o.fail, nextFail.token != -1 {
                        output = nextFail
                    } else {
                        output = nil
                        break
                    }
                }
                node.output = output
                node.outputScore += output?.outputScore ?? 0
                queue.append(node)
            }
        }
    }

    /// Advance by one token. Returns (boostScore, nextState, matchedEndState?).
    public func forwardOneStep(
        from state: State,
        token: Int
    ) -> (score: Double, next: State, matched: State?) {
        var node: State
        var score: Double = 0
        if let direct = state.next[token] {
            node = direct
            score = node.tokenScore
        } else {
            var fail = state.fail ?? root
            while fail.next[token] == nil {
                guard fail.token != -1 else { break }
                fail = fail.fail ?? root
            }
            node = fail.next[token] ?? fail
            if fail.next[token] != nil {
                node = fail.next[token]!
            }
            score = node.nodeScore - state.nodeScore
        }

        let matched: State?
        if node.isEnd {
            matched = node
        } else if let out = node.output {
            matched = out
        } else {
            matched = nil
        }
        return (score + node.outputScore, node, matched)
    }

    /// Whether a given state has matched a phrase (either ``isEnd`` itself or
    /// has a non-nil ``output``).
    public func isMatched(_ state: State) -> (matched: Bool, state: State?) {
        if state.isEnd { return (true, state) }
        if let output = state.output { return (true, output) }
        return (false, nil)
    }

    /// Cancel the accumulated boost when leaving a matched branch back to root.
    public func finalize(_ state: State) -> (score: Double, next: State) {
        return (-state.nodeScore, root)
    }
}
