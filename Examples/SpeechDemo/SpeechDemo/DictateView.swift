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
            Button(action: {}) {
                VStack(spacing: 4) {
                    Image(systemName: vm.isRecording ? "mic.fill" : "mic")
                        .font(.system(size: 32))
                    Text(vm.isRecording ? "Recording..." : "Hold to Record")
                        .font(.caption)
                }
                .frame(width: 120, height: 80)
            }
            .buttonStyle(.borderedProminent)
            .tint(vm.isRecording ? .red : .accentColor)
            .simultaneousGesture(
                DragGesture(minimumDistance: 0)
                    .onChanged { _ in
                        if !vm.isRecording && !vm.isTranscribing {
                            vm.startRecording()
                        }
                    }
                    .onEnded { _ in
                        if vm.isRecording {
                            Task { await vm.stopAndTranscribe() }
                        }
                    }
            )
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
