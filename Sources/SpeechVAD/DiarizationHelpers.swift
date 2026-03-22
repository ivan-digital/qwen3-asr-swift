import Foundation
import AudioCommon

/// Shared helpers for diarization post-processing, used by both
/// PyannoteDiarizationPipeline and SortformerDiarizer.
enum DiarizationHelpers {

    /// Merge adjacent segments from the same speaker when the gap is below `minSilence`.
    ///
    /// Segments are grouped per-speaker, merged within each group, then sorted globally.
    static func mergeSegments(
        _ segments: [DiarizedSegment],
        minSilence: Float
    ) -> [DiarizedSegment] {
        guard !segments.isEmpty else { return [] }

        var bySpeaker = [Int: [DiarizedSegment]]()
        for seg in segments {
            bySpeaker[seg.speakerId, default: []].append(seg)
        }

        var merged = [DiarizedSegment]()
        for (spk, spkSegs) in bySpeaker {
            let sorted = spkSegs.sorted { $0.startTime < $1.startTime }
            var current = sorted[0]

            for i in 1..<sorted.count {
                let next = sorted[i]
                if next.startTime - current.endTime < minSilence {
                    current = DiarizedSegment(
                        startTime: current.startTime,
                        endTime: next.endTime,
                        speakerId: spk
                    )
                } else {
                    merged.append(current)
                    current = next
                }
            }
            merged.append(current)
        }

        merged.sort { $0.startTime < $1.startTime }
        return merged
    }

    /// Remap speaker IDs to contiguous 0-based range, preserving order of first appearance.
    static func compactSpeakerIds(_ segments: [DiarizedSegment]) -> [DiarizedSegment] {
        let usedIds = Set(segments.map(\.speakerId)).sorted()
        let idMap = Dictionary(uniqueKeysWithValues: usedIds.enumerated().map { ($1, $0) })
        return segments.map {
            DiarizedSegment(
                startTime: $0.startTime,
                endTime: $0.endTime,
                speakerId: idMap[$0.speakerId] ?? $0.speakerId
            )
        }
    }

    /// Resample audio via AVAudioConverter (delegates to AudioFileLoader).
    static func resample(_ audio: [Float], from sourceSR: Int, to targetSR: Int) -> [Float] {
        AudioFileLoader.resample(audio, from: sourceSR, to: targetSR)
    }

    // MARK: - Constrained Agglomerative Clustering

    /// Item for constrained agglomerative clustering.
    struct ClusterItem {
        let windowIndex: Int
        let localSpeakerId: Int
        let embedding: [Float]
    }

    /// Constrained agglomerative clustering with centroid linkage and cosine distance.
    ///
    /// Items from the same window can never be merged (same-window constraint).
    /// Merges closest unconstrained pair until distance exceeds threshold.
    ///
    /// - Parameters:
    ///   - items: per-window per-speaker embeddings
    ///   - threshold: cosine distance threshold (0–2). Pairs with distance >= threshold are not merged.
    /// - Returns: cluster assignment for each item, and cluster centroids
    static func constrainedAgglomerativeClustering(
        items: [ClusterItem],
        threshold: Float
    ) -> (clusterAssignment: [Int], centroids: [[Float]]) {
        guard !items.isEmpty else { return ([], []) }
        if items.count == 1 {
            return ([0], [items[0].embedding])
        }

        let n = items.count
        let dim = items[0].embedding.count

        // Each item starts as its own cluster
        var clusterOf = Array(0..<n)  // item → cluster ID
        var centroids = items.map { $0.embedding }  // cluster ID → centroid
        var clusterMembers = (0..<n).map { [$0] }  // cluster ID → member items
        // Window indices per cluster (for constraint checking)
        var clusterWindows = items.map { Set([$0.windowIndex]) }
        var active = Set(0..<n)

        while active.count > 1 {
            // Find closest unconstrained pair
            var bestDist: Float = Float.greatestFiniteMagnitude
            var bestI = -1, bestJ = -1

            let activeList = active.sorted()
            for ai in 0..<activeList.count {
                for aj in (ai + 1)..<activeList.count {
                    let ci = activeList[ai], cj = activeList[aj]

                    // Same-window constraint: if clusters share any window, skip
                    if !clusterWindows[ci].isDisjoint(with: clusterWindows[cj]) {
                        continue
                    }

                    let dist = cosineDistance(centroids[ci], centroids[cj])
                    if dist < bestDist {
                        bestDist = dist
                        bestI = ci
                        bestJ = cj
                    }
                }
            }

            guard bestDist < threshold && bestI >= 0 else { break }

            // Merge bestJ into bestI
            let sizeI = clusterMembers[bestI].count
            let sizeJ = clusterMembers[bestJ].count
            let totalSize = Float(sizeI + sizeJ)

            // Weighted average centroid
            var newCentroid = [Float](repeating: 0, count: dim)
            for d in 0..<dim {
                newCentroid[d] = (centroids[bestI][d] * Float(sizeI) + centroids[bestJ][d] * Float(sizeJ)) / totalSize
            }
            centroids[bestI] = newCentroid

            // Transfer members
            for member in clusterMembers[bestJ] {
                clusterOf[member] = bestI
            }
            clusterMembers[bestI].append(contentsOf: clusterMembers[bestJ])

            // Propagate window constraints
            clusterWindows[bestI].formUnion(clusterWindows[bestJ])

            active.remove(bestJ)
        }

        // Build final compact assignment
        let activeSorted = active.sorted()
        var clusterMap = [Int: Int]()  // old cluster ID → new compact ID
        for (newId, oldId) in activeSorted.enumerated() {
            clusterMap[oldId] = newId
        }

        let assignment = (0..<n).map { clusterMap[clusterOf[$0]]! }
        let finalCentroids = activeSorted.map { centroids[$0] }

        return (assignment, finalCentroids)
    }

    /// Cosine distance between two vectors: 1 - cosine_similarity.
    /// Returns value in [0, 2].
    static func cosineDistance(_ a: [Float], _ b: [Float]) -> Float {
        let n = min(a.count, b.count)
        guard n > 0 else { return 2.0 }

        var dot: Float = 0, normA: Float = 0, normB: Float = 0
        for i in 0..<n {
            dot += a[i] * b[i]
            normA += a[i] * a[i]
            normB += b[i] * b[i]
        }

        let denom = sqrt(normA) * sqrt(normB)
        guard denom > 1e-10 else { return 2.0 }
        return 1.0 - dot / denom
    }
}
