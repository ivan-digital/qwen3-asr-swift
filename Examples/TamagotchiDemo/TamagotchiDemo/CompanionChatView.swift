import SwiftUI

struct CompanionChatView: View {
    @State private var vm = CompanionChatViewModel()
    @Environment(\.scenePhase) private var scenePhase

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                if !vm.modelsLoaded {
                    loadingSection
                } else {
                    chatList

                    statusBar

                    if vm.isListening {
                        DiagnosticsView(monitor: vm.diagnostics)

                        VoiceLevelBar(level: vm.audioLevel)
                            .frame(height: 4)
                            .padding(.horizontal)
                    }

                    inputBar
                }
            }
            .navigationTitle("Companion")
            #if os(iOS)
            .navigationBarTitleDisplayMode(.inline)
            #endif
            .onChange(of: scenePhase) { _, newPhase in
                switch newPhase {
                case .background:
                    // Stop pipeline to free audio resources
                    if vm.isListening { vm.stopListening() }
                case .active:
                    // If models were unloaded (OOM kill), reload
                    if !vm.modelsLoaded && !vm.isLoading {
                        Task { await vm.loadModels() }
                    }
                default:
                    break
                }
            }
            .toolbar {
                ToolbarItem(placement: .automatic) {
                    if vm.modelsLoaded {
                        Menu {
                            Button("Clear Chat") { vm.clearChat() }
                            Toggle("Speak Responses", isOn: $vm.speakEnabled)
                            Divider()
                            if vm.isListening {
                                Button("Stop Listening") { vm.stopListening() }
                            } else {
                                Button("Start Listening") { vm.startListening() }
                            }
                        } label: {
                            Image(systemName: "ellipsis.circle")
                        }
                    }
                }
            }
        }
    }

    // MARK: - Loading

    private var loadingSection: some View {
        VStack(spacing: 16) {
            Spacer()

            if vm.isLoading {
                ProgressView(value: vm.loadProgress) {
                    Text(vm.loadingStatus)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                .padding(.horizontal, 40)
            } else {
                Image(systemName: "bubble.left.and.bubble.right.fill")
                    .font(.system(size: 48))
                    .foregroundStyle(.tint)

                Text("On-Device Companion")
                    .font(.title2.bold())

                Text("On-device voice pipeline\nASR + VAD + LLM + TTS")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)

                Button("Load Models") {
                    Task { await vm.loadModels() }
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.large)
            }

            if let error = vm.errorMessage {
                Text(error)
                    .foregroundStyle(.red)
                    .font(.caption)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal)
            }

            Spacer()
        }
    }

    // MARK: - Chat

    private var chatList: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(spacing: 8) {
                    ForEach(vm.messages) { msg in
                        ChatBubble(message: msg)
                            .id(msg.id)
                    }

                    if vm.isGenerating {
                        HStack {
                            TypingIndicator()
                            Spacer()
                        }
                        .padding(.horizontal)
                        .id("typing")
                    }
                }
                .padding(.vertical, 8)
            }
            .onChange(of: vm.messages.count) {
                withAnimation {
                    if let last = vm.messages.last {
                        proxy.scrollTo(last.id, anchor: .bottom)
                    }
                }
            }
            .onChange(of: vm.isGenerating) {
                if vm.isGenerating {
                    withAnimation {
                        proxy.scrollTo("typing", anchor: .bottom)
                    }
                }
            }
        }
    }

    // MARK: - Status Bar

    private var statusBar: some View {
        HStack(spacing: 6) {
            Circle()
                .fill(statusColor)
                .frame(width: 8, height: 8)

            Text(vm.pipelineState)
                .font(.caption2)
                .foregroundStyle(.secondary)

            Spacer()

            if vm.isSpeechDetected {
                Image(systemName: "waveform")
                    .foregroundStyle(.red)
                    .font(.caption)
            }
        }
        .padding(.horizontal)
        .padding(.vertical, 4)
    }

    private var statusColor: Color {
        if vm.isSpeechDetected { return .red }
        if vm.isGenerating { return .orange }
        if vm.isListening { return .green }
        return .gray
    }

    // MARK: - Input (text fallback)

    private var inputBar: some View {
        HStack(spacing: 8) {
            Button {
                if vm.isListening {
                    vm.stopListening()
                } else {
                    vm.startListening()
                }
            } label: {
                Image(systemName: vm.isListening ? "mic.fill" : "mic.slash")
                    .font(.title3)
                    .foregroundStyle(vm.isSpeechDetected ? .red : (vm.isListening ? .green : .gray))
            }

            TextField("Message...", text: $vm.inputText, axis: .vertical)
                .textFieldStyle(.plain)
                .lineLimit(1...4)
                .padding(.horizontal, 12)
                .padding(.vertical, 8)
                .background(Color.gray.opacity(0.15))
                .clipShape(RoundedRectangle(cornerRadius: 20))
                .disabled(vm.isGenerating)
                .onSubmit { sendIfReady() }

            Button {
                sendIfReady()
            } label: {
                Image(systemName: "arrow.up.circle.fill")
                    .font(.title2)
            }
            .disabled(
                vm.inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                || vm.isGenerating
            )
        }
        .padding(.horizontal)
        .padding(.vertical, 8)
        .background(.bar)
    }

    private func sendIfReady() {
        let text = vm.inputText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty, !vm.isGenerating else { return }
        vm.send(text)
    }
}
