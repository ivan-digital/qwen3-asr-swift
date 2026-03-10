import SwiftUI

struct SpeakView: View {
    @State private var vm = SpeakViewModel()

    var body: some View {
        VStack(spacing: 16) {
            #if os(macOS)
            // Load model button (macOS uses Qwen3-TTS)
            if !vm.modelLoaded && !vm.isLoading {
                Button("Load Qwen3-TTS (CustomVoice)") {
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
            #else
            Text("System Speech Synthesis")
                .font(.caption)
                .foregroundStyle(.secondary)
            #endif

            if vm.modelLoaded {
                // Language & speaker pickers
                HStack {
                    Text("Language:")
                    Picker("Language", selection: $vm.language) {
                        ForEach(vm.languages, id: \.self) { lang in
                            Text(lang.capitalized).tag(lang)
                        }
                    }
                    #if os(macOS)
                    .frame(width: 150)
                    #endif

                    if !vm.speakers.isEmpty {
                        Text("Speaker:")
                        Picker("Speaker", selection: $vm.speaker) {
                            ForEach(vm.speakers, id: \.self) { s in
                                Text(s.capitalized).tag(s)
                            }
                        }
                        #if os(macOS)
                        .frame(width: 150)
                        #endif
                    }
                }

                // Text input
                TextEditor(text: $vm.text)
                    .font(.body)
                    .frame(minHeight: 100)
                    #if os(macOS)
                    .border(Color.gray.opacity(0.3))
                    #else
                    .overlay(
                        RoundedRectangle(cornerRadius: 8)
                            .stroke(Color.gray.opacity(0.3))
                    )
                    #endif
                    .disabled(vm.isSynthesizing)

                // Controls
                HStack(spacing: 12) {
                    Button(vm.isSynthesizing ? "Synthesizing..." : "Speak") {
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
