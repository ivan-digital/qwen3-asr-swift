#if os(macOS)
import SwiftUI

struct EchoView: View {
    @State private var vm = EchoViewModel()

    var body: some View {
        VStack(spacing: 16) {
            // Load models
            if !vm.modelsLoaded && !vm.isLoading {
                Button("Load Models (VAD + ASR + TTS)") {
                    Task { await vm.loadModels() }
                }
                .buttonStyle(.borderedProminent)
            }

            if vm.isLoading {
                ProgressView()
                Text(vm.loadingStatus)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            if vm.modelsLoaded {
                // State indicator + VAD level
                HStack {
                    Circle()
                        .fill(stateColor)
                        .frame(width: 12, height: 12)
                    Text(vm.pipelineState)
                        .font(.headline)
                        .foregroundStyle(stateColor)
                    Spacer()
                    if vm.isRunning {
                        Text(String(format: "%.2f", vm.vadLevel))
                            .font(.system(.caption, design: .monospaced))
                            .foregroundStyle(vm.vadLevel > 0.5 ? .green : .secondary)
                        ProgressView(value: Double(vm.vadLevel), total: 1.0)
                            .frame(width: 80)
                            .tint(vm.vadLevel > 0.5 ? .green : .gray)
                    }
                }

                // Start/Stop
                HStack(spacing: 12) {
                    Button(vm.isRunning ? "Stop" : "Start Echo") {
                        if vm.isRunning {
                            vm.stopPipeline()
                        } else {
                            vm.startPipeline()
                        }
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(vm.isRunning ? .red : .blue)
                }

                // Last transcription (prominent display)
                if !vm.lastTranscription.isEmpty {
                    GroupBox {
                        VStack(alignment: .leading, spacing: 4) {
                            HStack {
                                Text("Last heard")
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                                if !vm.lastLanguage.isEmpty {
                                    Text("[\(vm.lastLanguage)]")
                                        .font(.caption)
                                        .foregroundStyle(.blue)
                                }
                                Spacer()
                            }
                            Text(vm.lastTranscription)
                                .font(.body)
                                .textSelection(.enabled)
                        }
                        .padding(4)
                    }
                }
            }

            // Log
            if !vm.log.isEmpty {
                ScrollViewReader { proxy in
                    ScrollView {
                        LazyVStack(alignment: .leading, spacing: 2) {
                            ForEach(Array(vm.log.enumerated()), id: \.offset) { i, line in
                                Text(line)
                                    .font(.system(.caption, design: .monospaced))
                                    .foregroundStyle(logColor(for: line))
                                    .id(i)
                            }
                        }
                        .padding(8)
                    }
                    .frame(maxHeight: 200)
                    .background(Color.black.opacity(0.05))
                    .cornerRadius(8)
                    .onChange(of: vm.log.count) { _, _ in
                        if let last = vm.log.indices.last {
                            proxy.scrollTo(last, anchor: .bottom)
                        }
                    }
                }
            }

            if let error = vm.errorMessage {
                Text(error)
                    .foregroundStyle(.red)
                    .font(.caption)
            }
        }
        .padding()
    }

    private var stateColor: Color {
        switch vm.pipelineState {
        case "speech detected": return .green
        case "transcribing...": return .orange
        case "synthesizing...": return .purple
        case "speaking...": return .blue
        case "listening": return .green
        case _ where vm.pipelineState.contains("interrupted"): return .yellow
        default: return .gray
        }
    }

    private func logColor(for line: String) -> Color {
        if line.contains("[STT") { return .primary }
        if line.contains("[VAD]") { return .green }
        if line.contains("[TTS") { return .blue }
        if line.contains("[ERROR") { return .red }
        if line.contains("[Interrupted") { return .orange }
        return .secondary
    }
}
#endif
