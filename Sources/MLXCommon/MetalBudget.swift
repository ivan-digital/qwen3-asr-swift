import Cmlx
import Foundation
import MLX

/// Metal GPU memory budget utilities.
public enum MetalBudget {

    /// Query real Metal headroom: recommended working set minus active allocations.
    /// Returns nil if Metal device info is unavailable.
    public static var availableBytes: Int? {
        let info = GPU.deviceInfo()
        let maxWorking = Int(info.maxRecommendedWorkingSetSize)
        guard maxWorking > 0 else { return nil }
        let active = Memory.activeMemory
        let overhead = 256 * 1024 * 1024  // 256 MB safety margin
        return max(0, maxWorking - active - overhead)
    }

    /// Total device memory in bytes.
    public static var totalMemory: Int {
        GPU.deviceInfo().memorySize
    }

    /// Maximum recommended working set size in bytes.
    public static var maxRecommendedWorkingSet: Int {
        Int(GPU.deviceInfo().maxRecommendedWorkingSetSize)
    }

    /// Currently active (non-cache) MLX memory in bytes.
    public static var activeMemory: Int {
        Memory.activeMemory
    }

    /// Pin GPU memory to prevent paging under pressure.
    /// Uses 90% of recommended working set by default.
    /// Only effective on macOS 15+ / iOS 18+.
    @discardableResult
    public static func pinMemory(fraction: Double = 0.9) -> Int {
        let limit = Int(Double(maxRecommendedWorkingSet) * fraction)
        var previous: size_t = 0
        mlx_set_wired_limit(&previous, size_t(limit))
        return Int(previous)
    }

    /// Unpin GPU memory (set wired limit to 0).
    @discardableResult
    public static func unpinMemory() -> Int {
        var previous: size_t = 0
        mlx_set_wired_limit(&previous, 0)
        return Int(previous)
    }

    /// Check if a model of the given size (bytes) can fit in available GPU memory.
    public static func canFit(modelBytes: Int) -> Bool {
        guard let available = availableBytes else { return true }
        return modelBytes <= available
    }
}
