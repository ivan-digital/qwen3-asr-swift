import Foundation
import AudioCommon

// MARK: - RTTM Format

/// RTTM (Rich Transcription Time Marked) segment for standard diarization evaluation.
public struct RTTMSegment: Sendable {
    public let filename: String
    public let startTime: Float
    public let duration: Float
    public let speakerLabel: String

    public init(filename: String, startTime: Float, duration: Float, speakerLabel: String) {
        self.filename = filename
        self.startTime = startTime
        self.duration = duration
        self.speakerLabel = speakerLabel
    }

    /// Format as standard RTTM line: `SPEAKER <file> 1 <start> <dur> <NA> <NA> <speaker> <NA> <NA>`
    public var rttmLine: String {
        let s = String(format: "%.3f", startTime)
        let d = String(format: "%.3f", duration)
        return "SPEAKER \(filename) 1 \(s) \(d) <NA> <NA> \(speakerLabel) <NA> <NA>"
    }
}

/// Convert diarization result to RTTM format.
public func toRTTM(segments: [DiarizedSegment], filename: String) -> [RTTMSegment] {
    segments.map { seg in
        RTTMSegment(
            filename: filename,
            startTime: seg.startTime,
            duration: seg.duration,
            speakerLabel: "speaker_\(seg.speakerId)"
        )
    }
}

/// Write RTTM segments to string.
public func formatRTTM(_ rttmSegments: [RTTMSegment]) -> String {
    rttmSegments.map(\.rttmLine).joined(separator: "\n")
}

// MARK: - DER Computation

/// Diarization Error Rate result.
public struct DERResult: Sendable {
    /// Total scored speech duration in seconds
    public let totalSpeech: Float
    /// False alarm duration (non-speech classified as speech)
    public let falseAlarm: Float
    /// Missed speech duration
    public let missedSpeech: Float
    /// Speaker confusion duration (wrong speaker assigned)
    public let confusion: Float

    /// Diarization Error Rate = (FA + Miss + Confusion) / TotalSpeech
    public var der: Float {
        guard totalSpeech > 0 else { return 0 }
        return (falseAlarm + missedSpeech + confusion) / totalSpeech
    }

    /// Diarization Error Rate as percentage
    public var derPercent: Float { der * 100 }
}

/// Compute Diarization Error Rate between reference and hypothesis.
///
/// Uses frame-level scoring with configurable resolution and collar.
/// Collar applies forgiveness around reference segment boundaries.
///
/// - Parameters:
///   - reference: reference (ground truth) segments
///   - hypothesis: hypothesis (system output) segments
///   - collar: forgiveness collar in seconds around boundaries (default 0.25s)
///   - resolution: scoring resolution in seconds (default 0.01s = 10ms)
/// - Returns: DER breakdown
public func computeDER(
    reference: [DiarizedSegment],
    hypothesis: [DiarizedSegment],
    collar: Float = 0.25,
    resolution: Float = 0.01
) -> DERResult {
    guard !reference.isEmpty else {
        let hTotal = hypothesis.reduce(Float(0)) { $0 + $1.duration }
        return DERResult(totalSpeech: 0, falseAlarm: hTotal, missedSpeech: 0, confusion: 0)
    }

    // Find time range
    let allSegments: [DiarizedSegment] = reference + hypothesis
    let maxTime = allSegments.map(\.endTime).max()!
    let numFrames = Int(ceil(maxTime / resolution))
    guard numFrames > 0 else {
        return DERResult(totalSpeech: 0, falseAlarm: 0, missedSpeech: 0, confusion: 0)
    }

    // Build collar mask: frames near reference boundaries are excluded from scoring
    var collarMask = [Bool](repeating: false, count: numFrames)
    if collar > 0 {
        for seg in reference {
            let startFrame = Int(seg.startTime / resolution)
            let endFrame = Int(seg.endTime / resolution)
            let collarFrames = Int(collar / resolution)

            for f in max(0, startFrame - collarFrames)..<min(numFrames, startFrame + collarFrames) {
                collarMask[f] = true
            }
            for f in max(0, endFrame - collarFrames)..<min(numFrames, endFrame + collarFrames) {
                collarMask[f] = true
            }
        }
    }

    // Build per-frame speaker sets for reference and hypothesis
    // Use sorted arrays of speaker IDs (faster than Set for small N)
    let refSpeakers = buildFrameSpeakers(segments: reference, numFrames: numFrames, resolution: resolution)
    let hypSpeakers = buildFrameSpeakers(segments: hypothesis, numFrames: numFrames, resolution: resolution)

    // Score each frame
    var totalSpeech: Float = 0
    var falseAlarm: Float = 0
    var missedSpeech: Float = 0
    var confusion: Float = 0

    for f in 0..<numFrames {
        if collarMask[f] { continue }

        let refCount = refSpeakers[f].count
        let hypCount = hypSpeakers[f].count

        if refCount == 0 && hypCount == 0 { continue }

        if refCount == 0 {
            // No reference speech, any hypothesis is false alarm
            falseAlarm += Float(hypCount) * resolution
            continue
        }

        // Reference speech exists — count it
        totalSpeech += Float(refCount) * resolution

        if hypCount == 0 {
            // All reference speech is missed
            missedSpeech += Float(refCount) * resolution
            continue
        }

        // Both have speakers — compute exact match by speaker ID
        let matched = countExactMatched(ref: refSpeakers[f], hyp: hypSpeakers[f])
        let unmatchedRef = refCount - matched  // ref speakers with no matching hyp
        let unmatchedHyp = hypCount - matched  // hyp speakers with no matching ref
        // Confusion: min of unmatched ref/hyp (wrong speaker assigned)
        let conf = min(unmatchedRef, unmatchedHyp)
        // Missed: unmatched ref beyond confusion
        let missed = unmatchedRef - conf
        // False alarm: unmatched hyp beyond confusion
        let fa = unmatchedHyp - conf

        missedSpeech += Float(missed) * resolution
        confusion += Float(conf) * resolution
        falseAlarm += Float(fa) * resolution
    }

    return DERResult(
        totalSpeech: totalSpeech,
        falseAlarm: falseAlarm,
        missedSpeech: missedSpeech,
        confusion: confusion
    )
}

// MARK: - Optimal Speaker Mapping

/// Compute DER with optimal 1-to-1 speaker mapping between reference and hypothesis.
///
/// Tries all permutations of hypothesis speaker labels to find the mapping
/// that minimizes DER. For >8 speakers, falls back to greedy matching.
public func computeDERWithOptimalMapping(
    reference: [DiarizedSegment],
    hypothesis: [DiarizedSegment],
    collar: Float = 0.25,
    resolution: Float = 0.01
) -> DERResult {
    let refSpeakers = Set(reference.map(\.speakerId)).sorted()
    let hypSpeakers = Set(hypothesis.map(\.speakerId)).sorted()

    guard !refSpeakers.isEmpty, !hypSpeakers.isEmpty else {
        return computeDER(reference: reference, hypothesis: hypothesis,
                         collar: collar, resolution: resolution)
    }

    // For small speaker counts, try all permutations
    if hypSpeakers.count <= 8 {
        return bruteForceOptimalMapping(
            reference: reference, hypothesis: hypothesis,
            refSpeakers: refSpeakers, hypSpeakers: hypSpeakers,
            collar: collar, resolution: resolution
        )
    }

    // For large speaker counts, use greedy matching
    return greedyOptimalMapping(
        reference: reference, hypothesis: hypothesis,
        refSpeakers: refSpeakers, hypSpeakers: hypSpeakers,
        collar: collar, resolution: resolution
    )
}

// MARK: - RTTM Parsing

/// Parse RTTM file content into DiarizedSegments.
public func parseRTTM(_ content: String) -> [DiarizedSegment] {
    var segments = [DiarizedSegment]()
    var speakerMap = [String: Int]()
    var nextId = 0

    for line in content.split(separator: "\n") {
        let trimmed = line.trimmingCharacters(in: .whitespaces)
        guard !trimmed.isEmpty, trimmed.hasPrefix("SPEAKER") else { continue }

        let parts = trimmed.split(whereSeparator: \.isWhitespace).map(String.init)
        guard parts.count >= 8 else { continue }

        guard let start = Float(parts[3]),
              let dur = Float(parts[4]) else { continue }

        let speaker = parts[7]
        if speakerMap[speaker] == nil {
            speakerMap[speaker] = nextId
            nextId += 1
        }

        segments.append(DiarizedSegment(
            startTime: start,
            endTime: start + dur,
            speakerId: speakerMap[speaker]!
        ))
    }

    return segments.sorted { $0.startTime < $1.startTime }
}

// MARK: - Internals

private func buildFrameSpeakers(
    segments: [DiarizedSegment],
    numFrames: Int,
    resolution: Float
) -> [[Int]] {
    var result = [[Int]](repeating: [], count: numFrames)

    for seg in segments {
        let startFrame = max(0, Int(seg.startTime / resolution))
        let endFrame = min(numFrames, Int(seg.endTime / resolution))

        for f in startFrame..<endFrame {
            if !result[f].contains(seg.speakerId) {
                result[f].append(seg.speakerId)
            }
        }
    }

    return result
}

/// Count speakers present in both ref and hyp by exact ID match.
private func countExactMatched(ref: [Int], hyp: [Int]) -> Int {
    var matched = 0
    for rSpk in ref {
        if hyp.contains(rSpk) {
            matched += 1
        }
    }
    return matched
}

private func bruteForceOptimalMapping(
    reference: [DiarizedSegment],
    hypothesis: [DiarizedSegment],
    refSpeakers: [Int],
    hypSpeakers: [Int],
    collar: Float,
    resolution: Float
) -> DERResult {
    let permutations = generatePermutations(Array(0..<hypSpeakers.count))
    var bestResult: DERResult?

    for perm in permutations {
        // Build mapping: hypSpeakers[i] → refSpeakers[perm[i]] (if in range)
        var mapping = [Int: Int]()
        for (i, p) in perm.enumerated() {
            if p < refSpeakers.count {
                mapping[hypSpeakers[i]] = refSpeakers[p]
            } else {
                mapping[hypSpeakers[i]] = 1000 + i // unmapped → unique ID
            }
        }

        let remapped = hypothesis.map { seg in
            DiarizedSegment(
                startTime: seg.startTime,
                endTime: seg.endTime,
                speakerId: mapping[seg.speakerId] ?? seg.speakerId
            )
        }

        let result = computeDER(reference: reference, hypothesis: remapped,
                               collar: collar, resolution: resolution)

        if bestResult == nil || result.der < bestResult!.der {
            bestResult = result
        }
    }

    return bestResult!
}

private func greedyOptimalMapping(
    reference: [DiarizedSegment],
    hypothesis: [DiarizedSegment],
    refSpeakers: [Int],
    hypSpeakers: [Int],
    collar: Float,
    resolution: Float
) -> DERResult {
    // Compute overlap matrix between ref and hyp speakers
    let allSegs: [DiarizedSegment] = reference + hypothesis
    let maxTime = allSegs.map(\.endTime).max()!
    let numFrames = Int(ceil(maxTime / resolution))

    let refFrames = buildFrameSpeakers(segments: reference, numFrames: numFrames, resolution: resolution)
    let hypFrames = buildFrameSpeakers(segments: hypothesis, numFrames: numFrames, resolution: resolution)

    // Overlap[r][h] = number of frames where ref speaker r and hyp speaker h both active
    var overlap = [[Int]](repeating: [Int](repeating: 0, count: hypSpeakers.count), count: refSpeakers.count)
    let refIndex = Dictionary(uniqueKeysWithValues: refSpeakers.enumerated().map { ($1, $0) })
    let hypIndex = Dictionary(uniqueKeysWithValues: hypSpeakers.enumerated().map { ($1, $0) })

    for f in 0..<numFrames {
        for rSpk in refFrames[f] {
            guard let ri = refIndex[rSpk] else { continue }
            for hSpk in hypFrames[f] {
                guard let hi = hypIndex[hSpk] else { continue }
                overlap[ri][hi] += 1
            }
        }
    }

    // Greedy match: pick highest overlap pair, assign, repeat
    var mapping = [Int: Int]()
    var usedRef = Set<Int>()
    var usedHyp = Set<Int>()

    for _ in 0..<min(refSpeakers.count, hypSpeakers.count) {
        var bestR = -1, bestH = -1, bestOverlap = -1
        for r in 0..<refSpeakers.count where !usedRef.contains(r) {
            for h in 0..<hypSpeakers.count where !usedHyp.contains(h) {
                if overlap[r][h] > bestOverlap {
                    bestOverlap = overlap[r][h]
                    bestR = r
                    bestH = h
                }
            }
        }
        guard bestR >= 0 else { break }
        mapping[hypSpeakers[bestH]] = refSpeakers[bestR]
        usedRef.insert(bestR)
        usedHyp.insert(bestH)
    }

    let remapped = hypothesis.map { seg in
        DiarizedSegment(
            startTime: seg.startTime,
            endTime: seg.endTime,
            speakerId: mapping[seg.speakerId] ?? (1000 + seg.speakerId)
        )
    }

    return computeDER(reference: reference, hypothesis: remapped,
                     collar: collar, resolution: resolution)
}

private func generatePermutations(_ elements: [Int]) -> [[Int]] {
    if elements.count <= 1 { return [elements] }

    var result = [[Int]]()
    // Generate permutations of size elements.count from range 0..<elements.count
    // For speaker mapping, we need arrangements: pick from 0..<max(ref,hyp) count
    let n = elements.count
    func permute(_ current: [Int], _ remaining: [Int]) {
        if current.count == n {
            result.append(current)
            return
        }
        for (i, elem) in remaining.enumerated() {
            var rest = remaining
            rest.remove(at: i)
            permute(current + [elem], rest)
        }
    }

    // Permute indices 0..<n (mapping hyp speakers to ref speaker slots)
    let indices = Array(0..<max(n, n))
    permute([], indices)

    return result
}
