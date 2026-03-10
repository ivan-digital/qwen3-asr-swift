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
                // State indicator
                HStack {
                    Circle()
                        .fill(vm.isRunning ? .green : .gray)
                        .frame(width: 10, height: 10)
                    Text(vm.pipelineState)
                        .font(.headline)
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
            }

            // Log
            if !vm.log.isEmpty {
                ScrollViewReader { proxy in
                    ScrollView {
                        LazyVStack(alignment: .leading, spacing: 2) {
                            ForEach(Array(vm.log.enumerated()), id: \.offset) { i, line in
                                Text(line)
                                    .font(.system(.caption, design: .monospaced))
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
}
#endif
