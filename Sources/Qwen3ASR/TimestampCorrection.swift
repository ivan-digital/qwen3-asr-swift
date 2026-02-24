import Foundation

/// Monotonicity correction for forced alignment timestamps using LIS
public enum TimestampCorrection {

    /// Enforce monotonically increasing timestamps via LIS + interpolation.
    ///
    /// 1. Find Longest Increasing Subsequence of raw timestamp indices (O(n log n))
    /// 2. For positions not in LIS:
    ///    - Small gaps (<=2): nearest-neighbor correction
    ///    - Larger gaps: linear interpolation between LIS anchors
    ///
    /// - Parameter rawIndices: Raw timestamp class indices from argmax
    /// - Returns: Corrected monotonically increasing indices
    public static func enforceMonotonicity(_ rawIndices: [Int]) -> [Int] {
        guard rawIndices.count > 1 else { return rawIndices }

        // Find LIS positions
        let lisPositions = longestIncreasingSubsequencePositions(rawIndices)
        let lisSet = Set(lisPositions)

        // Build anchor points: (position_in_array, value)
        var anchors: [(pos: Int, val: Int)] = []
        for pos in lisPositions {
            anchors.append((pos, rawIndices[pos]))
        }

        // If LIS covers everything, already monotonic
        if anchors.count == rawIndices.count {
            return rawIndices
        }

        var corrected = rawIndices

        // Fill gaps between anchors
        var anchorIdx = 0
        var i = 0
        while i < corrected.count {
            if lisSet.contains(i) {
                // This position is an anchor, keep it
                anchorIdx = anchors.firstIndex(where: { $0.pos == i }) ?? anchorIdx
                i += 1
                continue
            }

            // Find surrounding anchors
            let prevAnchor: (pos: Int, val: Int)?
            let nextAnchor: (pos: Int, val: Int)?

            if anchorIdx < anchors.count && anchors[anchorIdx].pos < i {
                prevAnchor = anchors[anchorIdx]
            } else if anchorIdx > 0 {
                prevAnchor = anchors[anchorIdx - 1]
            } else {
                prevAnchor = nil
            }

            // Find next anchor after position i
            var nextIdx = anchorIdx
            while nextIdx < anchors.count && anchors[nextIdx].pos <= i {
                nextIdx += 1
            }
            nextAnchor = nextIdx < anchors.count ? anchors[nextIdx] : nil

            // Interpolate
            if let prev = prevAnchor, let next = nextAnchor {
                let gapSize = next.pos - prev.pos
                if gapSize <= 3 {
                    // Small gap: nearest neighbor
                    let distToPrev = i - prev.pos
                    let distToNext = next.pos - i
                    corrected[i] = distToPrev <= distToNext ? prev.val : next.val
                } else {
                    // Linear interpolation
                    let t = Float(i - prev.pos) / Float(next.pos - prev.pos)
                    corrected[i] = prev.val + Int(t * Float(next.val - prev.val))
                }
            } else if let prev = prevAnchor {
                // After last anchor: clamp to last anchor value
                corrected[i] = prev.val
            } else if let next = nextAnchor {
                // Before first anchor: clamp to first anchor value
                corrected[i] = next.val
            }

            i += 1
        }

        // Final pass: ensure strict monotonicity
        for i in 1..<corrected.count {
            if corrected[i] < corrected[i - 1] {
                corrected[i] = corrected[i - 1]
            }
        }

        return corrected
    }

    /// Find positions of the Longest Increasing Subsequence (O(n log n))
    static func longestIncreasingSubsequencePositions(_ arr: [Int]) -> [Int] {
        guard !arr.isEmpty else { return [] }

        let n = arr.count
        // tails[i] = smallest tail element for increasing subsequence of length i+1
        var tails: [Int] = []
        // tailIndices[i] = index in arr where tails[i] comes from
        var tailIndices: [Int] = []
        // parent[i] = index of previous element in LIS ending at arr[i]
        var parent = [Int](repeating: -1, count: n)

        for i in 0..<n {
            // Binary search for position to insert arr[i]
            var lo = 0, hi = tails.count
            while lo < hi {
                let mid = (lo + hi) / 2
                if tails[mid] < arr[i] {
                    lo = mid + 1
                } else {
                    hi = mid
                }
            }

            if lo == tails.count {
                tails.append(arr[i])
                tailIndices.append(i)
            } else {
                tails[lo] = arr[i]
                tailIndices[lo] = i
            }

            parent[i] = lo > 0 ? tailIndices[lo - 1] : -1
        }

        // Reconstruct LIS positions
        var positions: [Int] = []
        var idx = tailIndices[tails.count - 1]
        while idx != -1 {
            positions.append(idx)
            idx = parent[idx]
        }

        positions.reverse()
        return positions
    }
}
