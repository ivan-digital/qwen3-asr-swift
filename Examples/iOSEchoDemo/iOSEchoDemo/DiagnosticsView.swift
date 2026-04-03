import SwiftUI
import Charts
import Darwin.Mach

// MARK: - Diagnostics Data

@Observable
@MainActor
final class DiagnosticsMonitor {
    struct Sample: Identifiable {
        let id = UUID()
        let time: Date
        let value: Float
    }

    private(set) var cpuUsage: Float = 0
    private(set) var memoryMB: Float = 0
    private(set) var vadLevel: Float = 0

    private(set) var cpuHistory: [Sample] = []
    private(set) var memoryHistory: [Sample] = []
    private(set) var vadHistory: [Sample] = []

    private let maxSamples = 120  // ~60s at 2Hz
    private var timer: Timer?

    func start() {
        guard timer == nil else { return }
        timer = Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true) { [weak self] _ in
            Task { @MainActor in self?.tick() }
        }
    }

    func stop() {
        timer?.invalidate()
        timer = nil
    }

    func updateVAD(_ level: Float) {
        vadLevel = level
    }

    private func tick() {
        cpuUsage = Self.getAppCPUUsage()
        memoryMB = Self.getAppMemoryMB()

        let now = Date()
        cpuHistory.append(Sample(time: now, value: cpuUsage))
        memoryHistory.append(Sample(time: now, value: memoryMB))
        vadHistory.append(Sample(time: now, value: vadLevel))

        if cpuHistory.count > maxSamples { cpuHistory.removeFirst() }
        if memoryHistory.count > maxSamples { memoryHistory.removeFirst() }
        if vadHistory.count > maxSamples { vadHistory.removeFirst() }
    }

    // MARK: - System Metrics

    private static func getAppCPUUsage() -> Float {
        var threadList: thread_act_array_t?
        var threadCount: mach_msg_type_number_t = 0
        guard task_threads(mach_task_self_, &threadList, &threadCount) == KERN_SUCCESS,
              let threads = threadList else { return 0 }

        var totalUsage: Float = 0
        for i in 0..<Int(threadCount) {
            var info = thread_basic_info()
            var count = mach_msg_type_number_t(MemoryLayout<thread_basic_info>.size / MemoryLayout<integer_t>.size)
            let kr = withUnsafeMutablePointer(to: &info) { ptr in
                ptr.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { raw in
                    thread_info(threads[i], thread_flavor_t(THREAD_BASIC_INFO), raw, &count)
                }
            }
            if kr == KERN_SUCCESS && (info.flags & TH_FLAGS_IDLE) == 0 {
                totalUsage += Float(info.cpu_usage) / Float(TH_USAGE_SCALE) * 100
            }
        }
        vm_deallocate(mach_task_self_, vm_address_t(bitPattern: threads), vm_size_t(threadCount) * vm_size_t(MemoryLayout<thread_t>.size))
        return totalUsage
    }

    private static func getAppMemoryMB() -> Float {
        // Use phys_footprint — matches Xcode Memory Report and Instruments.
        // resident_size is inflated on simulator (includes shared libs, debugger overhead).
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) { ptr in
            ptr.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { raw in
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), raw, &count)
            }
        }
        guard kr == KERN_SUCCESS else { return 0 }
        return Float(info.phys_footprint) / (1024 * 1024)
    }
}

// MARK: - Diagnostics View

struct DiagnosticsView: View {
    let monitor: DiagnosticsMonitor

    var body: some View {
        VStack(spacing: 8) {
            // Current values
            HStack(spacing: 16) {
                MetricLabel(title: "CPU", value: String(format: "%.0f%%", monitor.cpuUsage), color: cpuColor)
                MetricLabel(title: "MEM", value: String(format: "%.0fMB", monitor.memoryMB), color: memColor)
                MetricLabel(title: "VAD", value: String(format: "%.2f", monitor.vadLevel), color: vadColor)
            }
            .font(.caption2.monospaced())

            // Graphs
            HStack(spacing: 4) {
                MiniChart(data: monitor.vadHistory, color: .red, maxY: 0.5, label: "VAD")
                MiniChart(data: monitor.cpuHistory, color: .blue, maxY: 200, label: "CPU%")
                MiniChart(data: monitor.memoryHistory, color: .purple, maxY: nil, label: "MB")
            }
            .frame(height: 50)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
        .background(.ultraThinMaterial)
        .clipShape(RoundedRectangle(cornerRadius: 10))
        .padding(.horizontal)
    }

    private var cpuColor: Color { monitor.cpuUsage > 150 ? .red : monitor.cpuUsage > 80 ? .orange : .blue }
    private var memColor: Color { monitor.memoryMB > 2000 ? .red : monitor.memoryMB > 1000 ? .orange : .purple }
    private var vadColor: Color { monitor.vadLevel > 0.1 ? .red : monitor.vadLevel > 0.03 ? .orange : .green }
}

// MARK: - Helpers

private struct MetricLabel: View {
    let title: String
    let value: String
    let color: Color

    var body: some View {
        VStack(spacing: 1) {
            Text(title)
                .foregroundStyle(.secondary)
            Text(value)
                .foregroundStyle(color)
                .fontWeight(.medium)
        }
    }
}

private struct MiniChart: View {
    let data: [DiagnosticsMonitor.Sample]
    let color: Color
    let maxY: Float?
    let label: String

    var body: some View {
        Chart {
            ForEach(data) { sample in
                LineMark(
                    x: .value("T", sample.time),
                    y: .value(label, sample.value)
                )
                .foregroundStyle(color.opacity(0.7))
                .lineStyle(StrokeStyle(lineWidth: 1))

                AreaMark(
                    x: .value("T", sample.time),
                    y: .value(label, sample.value)
                )
                .foregroundStyle(color.opacity(0.1))
            }
        }
        .chartXAxis(.hidden)
        .chartYAxis(.hidden)
        .chartYScale(domain: 0...(maxY.map { Double($0) } ?? autoMax))
    }

    private var autoMax: Double {
        let peak = data.map { Double($0.value) }.max() ?? 1
        return max(peak * 1.2, 1)
    }
}
