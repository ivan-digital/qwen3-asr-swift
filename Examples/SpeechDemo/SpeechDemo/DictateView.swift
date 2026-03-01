import SwiftUI

struct DictateView: View {
    @State private var vm = DictateViewModel()

    var body: some View {
        VStack(spacing: 16) {
            // Engine picker
            Picker("Engine", selection: $vm.selectedEngine) {
                ForEach(ASREngine.allCases) { engine in
                    Text(engine.rawValue).tag(engine)
                }
            }
            .pickerStyle(.segmented)
            .disabled(vm.isRecording || vm.isTranscribing)
            .frame(maxWidth: 300)

            // Language picker (Qwen3-ASR only)
            if vm.selectedEngine == .qwen3 {
                Picker("Language", selection: $vm.selectedLanguage) {
                    Text("Auto-detect").tag("auto")
                    Text("English").tag("en")
                    Text("Chinese").tag("zh")
                    Text("Japanese").tag("ja")
                    Text("Korean").tag("ko")
                    Text("French").tag("fr")
                    Text("Spanish").tag("es")
                    Text("German").tag("de")
                    Text("Russian").tag("ru")
                }
                .frame(maxWidth: 200)
                .disabled(vm.isRecording || vm.isTranscribing)
            }

            // Load model button
            if !vm.modelLoaded && !vm.isLoading {
                Button("Load \(vm.selectedEngine.rawValue)") {
                    Task { await vm.loadModel() }
                }
                .buttonStyle(.borderedProminent)
            }

            // Loading indicator
            if vm.isLoading {
                VStack(spacing: 8) {
                    ProgressView()
                    Text(vm.loadingStatus)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }

            Spacer()

            // Record button
            if vm.modelLoaded {
                recordButton
            }

            // Transcribing indicator
            if vm.isTranscribing {
                VStack(spacing: 8) {
                    ProgressView()
                    Text("Transcribing...")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }

            // Transcription result
            if !vm.transcription.isEmpty {
                GroupBox("Transcription") {
                    VStack(alignment: .leading, spacing: 8) {
                        Text(vm.transcription)
                            .textSelection(.enabled)
                            .frame(maxWidth: .infinity, alignment: .leading)

                        Button("Copy") {
                            vm.copyToClipboard()
                        }
                        .buttonStyle(.bordered)
                    }
                    .padding(4)
                }
            }

            // Error message
            if let error = vm.errorMessage {
                Text(error)
                    .foregroundStyle(.red)
                    .font(.caption)
            }

            Spacer()
        }
        .padding()
    }

    @ViewBuilder
    private var recordButton: some View {
        VStack(spacing: 8) {
            Button {
                if vm.isRecording {
                    Task { await vm.stopAndTranscribe() }
                } else if !vm.isTranscribing {
                    vm.startRecording()
                }
            } label: {
                VStack(spacing: 4) {
                    Image(systemName: vm.isRecording ? "stop.fill" : "mic.fill")
                        .font(.system(size: 32))
                    Text(vm.isRecording ? "Stop" : "Record")
                        .font(.caption)
                }
                .frame(width: 120, height: 80)
            }
            .buttonStyle(.borderedProminent)
            .tint(vm.isRecording ? .red : .accentColor)
            .disabled(vm.isTranscribing)

            // Audio level indicator
            if vm.isRecording {
                audioLevelBar
            }
        }
    }

    private var audioLevelBar: some View {
        GeometryReader { geo in
            RoundedRectangle(cornerRadius: 2)
                .fill(.green)
                .frame(
                    width: max(4, geo.size.width * CGFloat(min(vm.recorder.audioLevel, 1.0))),
                    height: 4
                )
        }
        .frame(height: 4)
        .frame(maxWidth: 200)
        .background(RoundedRectangle(cornerRadius: 2).fill(.gray.opacity(0.2)))
    }

}
