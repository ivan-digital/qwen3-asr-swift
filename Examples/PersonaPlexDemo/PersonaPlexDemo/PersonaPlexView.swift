import SwiftUI
import PersonaPlex

struct PersonaPlexView: View {
    @State private var vm = PersonaPlexViewModel()

    var body: some View {
        VStack(spacing: 16) {
            Text("PersonaPlex")
                .font(.largeTitle.bold())

            if !vm.modelLoaded && !vm.isLoading {
                Spacer()
                Button("Load PersonaPlex") {
                    Task { await vm.loadModel() }
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.large)
                Text("~5.5 GB download on first run")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Spacer()
            }

            if vm.isLoading {
                Spacer()
                ProgressView()
                    .controlSize(.large)
                if let status = vm.loadingStatus {
                    Text(status)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .multilineTextAlignment(.center)
                }
                Spacer()
            }

            if vm.modelLoaded {
                modelLoadedContent
            }

            if let error = vm.errorMessage {
                Text(error)
                    .foregroundStyle(.red)
                    .font(.caption)
                    .multilineTextAlignment(.center)
            }
        }
        .padding(24)
        .frame(minWidth: 600, minHeight: 500)
    }

    @ViewBuilder
    private var modelLoadedContent: some View {
        // Settings row
        HStack(spacing: 16) {
            VStack(alignment: .leading, spacing: 4) {
                Text("Voice").font(.caption).foregroundStyle(.secondary)
                Picker("Voice", selection: $vm.selectedVoice) {
                    Section("Natural Female") {
                        ForEach([PersonaPlexVoice.NATF0, .NATF1, .NATF2, .NATF3], id: \.self) { v in
                            Text(v.displayName).tag(v)
                        }
                    }
                    Section("Natural Male") {
                        ForEach([PersonaPlexVoice.NATM0, .NATM1, .NATM2, .NATM3], id: \.self) { v in
                            Text(v.displayName).tag(v)
                        }
                    }
                    Section("Variety Female") {
                        ForEach([PersonaPlexVoice.VARF0, .VARF1, .VARF2, .VARF3, .VARF4], id: \.self) { v in
                            Text(v.displayName).tag(v)
                        }
                    }
                    Section("Variety Male") {
                        ForEach([PersonaPlexVoice.VARM0, .VARM1, .VARM2, .VARM3, .VARM4], id: \.self) { v in
                            Text(v.displayName).tag(v)
                        }
                    }
                }
                .labelsHidden()
                .frame(width: 140)
            }

            VStack(alignment: .leading, spacing: 4) {
                Text("Max Steps").font(.caption).foregroundStyle(.secondary)
                Stepper("\(vm.maxSteps)", value: $vm.maxSteps, in: 50...500, step: 50)
                    .frame(width: 120)
            }
        }
        .disabled(vm.isActive || vm.isBusy)

        Spacer()

        // Main conversation button
        VStack(spacing: 12) {
            conversationButton
            stateLabel
        }

        // Debug info
        if !vm.debugInfo.isEmpty {
            Text(vm.debugInfo)
                .font(.caption.monospaced())
                .foregroundStyle(.orange)
        }

        Spacer()

        // Latency info
        if let info = vm.latencyInfo {
            Text(info)
                .font(.caption.monospaced())
                .foregroundStyle(.secondary)
        }

        // Transcripts
        if !vm.userTranscript.isEmpty || !vm.modelTranscript.isEmpty {
            VStack(spacing: 8) {
                if !vm.userTranscript.isEmpty {
                    GroupBox("You said") {
                        ScrollView {
                            Text(vm.userTranscript)
                                .textSelection(.enabled)
                                .frame(maxWidth: .infinity, alignment: .leading)
                        }
                        .frame(maxHeight: 80)
                    }
                }
                if !vm.modelTranscript.isEmpty {
                    GroupBox("Model response") {
                        ScrollView {
                            Text(vm.modelTranscript)
                                .textSelection(.enabled)
                                .frame(maxWidth: .infinity, alignment: .leading)
                        }
                        .frame(maxHeight: 80)
                    }
                }
            }
        }
    }

    @ViewBuilder
    private var conversationButton: some View {
        let size: CGFloat = 100

        Button {
            vm.toggleConversation()
        } label: {
            ZStack {
                Circle()
                    .fill(buttonColor)
                    .frame(width: size, height: size)
                    .overlay {
                        if vm.conversationState == .listening {
                            Circle()
                                .stroke(Color.white.opacity(0.6), lineWidth: 3 + CGFloat(vm.audioLevel) * 12)
                                .frame(width: size + 10, height: size + 10)
                        }
                    }
                    .overlay {
                        buttonIcon
                    }
                    .shadow(color: buttonShadow, radius: 10)
            }
        }
        .buttonStyle(.plain)
        .disabled(vm.isBusy)
        .animation(.easeInOut(duration: 0.3), value: vm.conversationState)
    }

    private var buttonColor: Color {
        switch vm.conversationState {
        case .inactive: return .accentColor
        case .listening: return .green
        case .processing: return .orange
        case .speaking: return .purple
        }
    }

    private var buttonShadow: Color {
        switch vm.conversationState {
        case .inactive: return .clear
        case .listening: return .green.opacity(0.4)
        case .processing: return .orange.opacity(0.4)
        case .speaking: return .purple.opacity(0.4)
        }
    }

    @ViewBuilder
    private var buttonIcon: some View {
        switch vm.conversationState {
        case .inactive:
            Image(systemName: "mic.fill")
                .font(.system(size: 36))
                .foregroundStyle(.white)
        case .listening:
            Image(systemName: "waveform")
                .font(.system(size: 36))
                .foregroundStyle(.white)
        case .processing:
            ProgressView()
                .controlSize(.large)
                .tint(.white)
        case .speaking:
            Image(systemName: "speaker.wave.2.fill")
                .font(.system(size: 36))
                .foregroundStyle(.white)
        }
    }

    @ViewBuilder
    private var stateLabel: some View {
        switch vm.conversationState {
        case .inactive:
            Text("Tap to start conversation")
                .font(.callout)
                .foregroundStyle(.secondary)
        case .listening:
            Text("Listening... speak now")
                .font(.callout)
                .foregroundStyle(.green)
        case .processing:
            Text("Generating response...")
                .font(.callout)
                .foregroundStyle(.orange)
        case .speaking:
            Text("Speaking... tap to stop")
                .font(.callout)
                .foregroundStyle(.purple)
        }
    }
}
