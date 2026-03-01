import Foundation
import Observation
import Qwen3TTS

@Observable
@MainActor
final class SpeakViewModel {
    var text = "The quick brown fox jumps over the lazy dog."
    var language = "english"
    var isLoading = false
    var isSynthesizing = false
    var loadingStatus = ""
    var errorMessage: String?

    let languages = ["english", "chinese", "japanese", "korean", "french", "german", "spanish"]

    private var ttsModel: Qwen3TTSModel?
    private let player = AudioPlayer()

    var isPlaying: Bool { player.isPlaying }
    var modelLoaded: Bool { ttsModel != nil }

    func loadModel() async {
        isLoading = true
        errorMessage = nil
        loadingStatus = "Downloading model..."

        do {
            if ttsModel == nil {
                let model = try await Qwen3TTSModel.fromPretrained { [weak self] progress, status in
                    Task { @MainActor in
                        self?.loadingStatus = status.isEmpty
                            ? "Downloading... \(Int(progress * 100))%"
                            : status
                    }
                }
                loadingStatus = "Warming up..."
                model.warmUp()
                ttsModel = model
            }
            loadingStatus = ""
        } catch {
            errorMessage = "Failed to load model: \(error.localizedDescription)"
            loadingStatus = ""
        }

        isLoading = false
    }

    func synthesize() async {
        guard let model = ttsModel else {
            errorMessage = "Model not loaded."
            return
        }
        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            errorMessage = "Enter some text first."
            return
        }

        isSynthesizing = true
        errorMessage = nil

        let inputText = text
        let inputLang = language

        let samples = model.synthesize(text: inputText, language: inputLang)

        guard !samples.isEmpty else {
            errorMessage = "Synthesis produced no audio."
            isSynthesizing = false
            return
        }

        do {
            try player.play(samples: samples, sampleRate: 24000)
        } catch {
            errorMessage = "Playback failed: \(error.localizedDescription)"
        }

        isSynthesizing = false
    }

    func stopPlayback() {
        player.stop()
    }
}
