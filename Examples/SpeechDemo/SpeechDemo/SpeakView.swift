import SwiftUI

struct SpeakView: View {
    @State private var vm = SpeakViewModel()

    var body: some View {
        VStack(spacing: 16) {
            // Load model button
            if !vm.modelLoaded && !vm.isLoading {
                Button("Load Qwen3-TTS") {
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

            if vm.modelLoaded {
                // Language picker
                HStack {
                    Text("Language:")
                    Picker("Language", selection: $vm.language) {
                        ForEach(vm.languages, id: \.self) { lang in
                            Text(lang.capitalized).tag(lang)
                        }
                    }
                    .frame(width: 150)
                }

                // Text input
                TextEditor(text: $vm.text)
                    .font(.body)
                    .frame(minHeight: 100)
                    .border(Color.gray.opacity(0.3))
                    .disabled(vm.isSynthesizing)

                // Controls
                HStack(spacing: 12) {
                    Button(vm.isSynthesizing ? "Synthesizing..." : "Synthesize") {
                        Task { await vm.synthesize() }
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(vm.isSynthesizing || vm.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)

                    if vm.isPlaying {
                        Button("Stop") {
                            vm.stopPlayback()
                        }
                        .buttonStyle(.bordered)
                    }
                }

                if vm.isSynthesizing {
                    ProgressView()
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
}
