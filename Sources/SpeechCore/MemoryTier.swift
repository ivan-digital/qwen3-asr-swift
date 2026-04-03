#if canImport(Darwin)
import Foundation
#if canImport(UIKit)
import UIKit
#endif

/// Memory tier for selecting model configurations on constrained devices.
///
/// Auto-detects available RAM and selects the appropriate model combination.
/// Use this to avoid OOM on iOS where 4 CoreML models can't coexist.
public enum MemoryTier: String, Sendable {
    /// 8GB+ (iPad Pro, M-series Mac): all CoreML models, full quality
    case full
    /// 4-6GB (iPhone Pro with entitlement): lazy CoreML pipeline, INT8 iOS models
    case standard
    /// 2-4GB (iPhone, no entitlement): Apple Speech + small CoreML models
    case constrained
    /// <2GB (old devices): Apple Speech + Apple TTS, no on-device LLM
    case minimal

    /// Detect the appropriate tier for the current device.
    public static func detect() -> MemoryTier {
        let totalGB = Double(ProcessInfo.processInfo.physicalMemory) / (1024 * 1024 * 1024)

        #if os(iOS)
        // On iOS, available memory is much less than physical RAM.
        // os_proc_available_memory() gives a more accurate picture.
        let availableMB = os_proc_available_memory() / (1024 * 1024)
        if availableMB > 5000 { return .full }       // iPad Pro / entitlement
        if availableMB > 3000 { return .standard }    // iPhone Pro
        if availableMB > 1500 { return .constrained } // iPhone
        return .minimal
        #else
        // macOS: plenty of memory, always full
        if totalGB >= 8 { return .full }
        if totalGB >= 4 { return .standard }
        return .constrained
        #endif
    }

    /// Human-readable description for diagnostics.
    public var description: String {
        switch self {
        case .full: return "full (8GB+, all CoreML)"
        case .standard: return "standard (4-6GB, lazy CoreML)"
        case .constrained: return "constrained (2-4GB, Apple Speech fallbacks)"
        case .minimal: return "minimal (<2GB, system TTS only)"
        }
    }

    /// Whether to use CoreML ASR (Parakeet) or fall back to Apple SFSpeechRecognizer.
    public var useCoreMLASR: Bool {
        self == .full || self == .standard
    }

    /// Whether to use CoreML TTS (Kokoro) or fall back to AVSpeechSynthesizer.
    public var useCoreMLTTS: Bool {
        self != .minimal
    }

    /// Whether an on-device LLM is feasible.
    public var useLLM: Bool {
        self != .minimal
    }

    /// Whether to auto-unload models between pipeline phases.
    public var autoUnload: Bool {
        self != .full
    }
}

/// Returns available memory in bytes on iOS, or total physical memory on macOS.
private func os_proc_available_memory() -> UInt64 {
    #if os(iOS)
    // os_proc_available_memory() returns footprint limit - current footprint
    var info = mach_task_basic_info()
    var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4
    let result = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
        }
    }
    guard result == KERN_SUCCESS else {
        return ProcessInfo.processInfo.physicalMemory / 2
    }
    // Rough estimate: total physical - 1.5GB system overhead
    let total = ProcessInfo.processInfo.physicalMemory
    let used = info.resident_size
    let available = total > used + 1_500_000_000 ? total - used - 1_500_000_000 : total / 4
    return available
    #else
    return ProcessInfo.processInfo.physicalMemory
    #endif
}
#endif
