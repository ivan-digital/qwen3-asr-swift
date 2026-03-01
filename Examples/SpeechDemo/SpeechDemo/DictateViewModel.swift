import AppKit
import Foundation
import Observation
import ParakeetASR
import Qwen3ASR

enum ASREngine: String, CaseIterable, Identifiable {
    case parakeet = "Parakeet TDT"
    case qwen3 = "Qwen3-ASR"
    var id: String { rawValue }
}

@Observable
@MainActor
final class DictateViewModel {
    var transcription = ""
    var isLoading = false
    var isRecording = false
    var isTranscribing = false
    var loadingStatus = ""
    var errorMessage: String?
    var selectedEngine: ASREngine = .parakeet

    private var parakeetModel: ParakeetASRModel?
    private var qwen3Model: Qwen3ASRModel?
    let recorder = AudioRecorder()

    var modelLoaded: Bool {
        switch selectedEngine {
        case .parakeet: return parakeetModel != nil
        case .qwen3: return qwen3Model != nil
        }
    }

    func loadModel() async {
        isLoading = true
        errorMessage = nil
        loadingStatus = "Downloading model..."

        do {
            switch selectedEngine {
            case .parakeet:
                if parakeetModel == nil {
                    let model = try await ParakeetASRModel.fromPretrained { [weak self] progress, status in
                        Task { @MainActor in
                            self?.loadingStatus = status.isEmpty
                                ? "Downloading... \(Int(progress * 100))%"
                                : status
                        }
                    }
                    loadingStatus = "Compiling CoreML model..."
                    try model.warmUp()
                    parakeetModel = model
                }
            case .qwen3:
                if qwen3Model == nil {
                    let model = try await Qwen3ASRModel.fromPretrained { [weak self] progress, status in
                        Task { @MainActor in
                            self?.loadingStatus = status.isEmpty
                                ? "Downloading... \(Int(progress * 100))%"
                                : status
                        }
                    }
                    qwen3Model = model
                }
            }
            loadingStatus = ""
        } catch {
            errorMessage = "Failed to load model: \(error.localizedDescription)"
            loadingStatus = ""
        }

        isLoading = false
    }

    func startRecording() {
        transcription = ""
        errorMessage = nil
        recorder.startRecording()
        isRecording = true
    }

    func stopAndTranscribe() async {
        let audio = recorder.stopRecording()
        isRecording = false

        guard !audio.isEmpty else {
            errorMessage = "No audio captured."
            return
        }

        isTranscribing = true
        errorMessage = nil

        do {
            let text: String
            switch selectedEngine {
            case .parakeet:
                guard let model = parakeetModel else {
                    errorMessage = "Model not loaded."
                    isTranscribing = false
                    return
                }
                text = try model.transcribeAudio(audio, sampleRate: 16000)
            case .qwen3:
                guard let model = qwen3Model else {
                    errorMessage = "Model not loaded."
                    isTranscribing = false
                    return
                }
                text = model.transcribe(audio: audio, sampleRate: 16000)
            }
            transcription = text
        } catch {
            errorMessage = "Transcription failed: \(error.localizedDescription)"
        }

        isTranscribing = false
    }

    func copyToClipboard() {
        #if canImport(AppKit)
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(transcription, forType: .string)
        #endif
    }
}
