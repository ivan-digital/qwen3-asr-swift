import SwiftUI

@main
struct DictateDemoApp: App {
    @State private var viewModel = DictateViewModel()

    var body: some Scene {
        Window("Dictate", id: "main") {
            ContentView(viewModel: viewModel)
        }
        .defaultSize(width: 500, height: 400)
    }
}

struct ContentView: View {
    @Bindable var viewModel: DictateViewModel

    var body: some View {
        VStack(spacing: 0) {
            // Toolbar
            HStack {
                if viewModel.isLoading {
                    ProgressView().controlSize(.small)
                    Text(viewModel.loadingStatus).font(.caption)
                } else if !viewModel.modelLoaded {
                    Button("Load Models") { Task { await viewModel.loadModels() } }
                } else {
                    Button {
                        viewModel.toggleRecording()
                    } label: {
                        Image(systemName: viewModel.isRecording ? "stop.circle.fill" : "mic.circle.fill")
                            .font(.title2)
                            .foregroundStyle(viewModel.isRecording ? .red : .accentColor)
                    }
                    .buttonStyle(.plain)

                    if viewModel.isRecording {
                        Circle()
                            .fill(viewModel.isSpeechActive ? .green : .orange)
                            .frame(width: 8, height: 8)
                        Text(viewModel.isSpeechActive ? "Speech" : "Silence")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }

                    Spacer()

                    if !viewModel.fullText.isEmpty {
                        Text("\(viewModel.wordCount) words")
                            .font(.caption).monospacedDigit()
                            .foregroundStyle(.tertiary)
                        Button("Copy") { viewModel.copyToClipboard() }
                            .buttonStyle(.bordered).controlSize(.small)
                        Button("Clear") { viewModel.clearText() }
                            .buttonStyle(.bordered).controlSize(.small)
                    }
                }

                if let error = viewModel.errorMessage {
                    Text(error).font(.caption).foregroundStyle(.red)
                }
            }
            .padding()

            Divider()

            // Transcript
            ScrollViewReader { proxy in
                ScrollView {
                    VStack(alignment: .leading, spacing: 6) {
                        if viewModel.sentences.isEmpty && viewModel.partialText.isEmpty {
                            Text(viewModel.isRecording ? "Speak..." : "Click mic to start")
                                .foregroundStyle(.tertiary)
                                .frame(maxWidth: .infinity, alignment: .center)
                                .padding(.top, 40)
                        }

                        ForEach(Array(viewModel.sentences.enumerated()), id: \.offset) { i, sentence in
                            Text(sentence)
                                .font(.system(.body, design: .rounded))
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .id(i)
                        }

                        if !viewModel.partialText.isEmpty {
                            Text(viewModel.partialText)
                                .font(.system(.body, design: .rounded))
                                .foregroundStyle(.secondary)
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .id("partial")
                        }
                    }
                    .padding()
                }
                .onChange(of: viewModel.sentences.count) {
                    withAnimation { proxy.scrollTo(viewModel.sentences.count - 1, anchor: .bottom) }
                }
                .onChange(of: viewModel.partialText) {
                    withAnimation { proxy.scrollTo("partial", anchor: .bottom) }
                }
            }
        }
    }
}
