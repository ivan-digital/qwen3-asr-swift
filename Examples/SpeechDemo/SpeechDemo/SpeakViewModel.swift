import AVFoundation
import Foundation
import Observation
#if os(macOS)
import Qwen3TTS
#endif

@Observable
@MainActor
final class SpeakViewModel {
    var text = "The quick brown fox jumps over the lazy dog."
    var language = "english"
    var speaker = "vivian"
    var isLoading = false
    var isSynthesizing = false
    var loadingStatus = ""
    var errorMessage: String?

    let languages = ["english", "chinese", "japanese", "korean", "french", "german", "spanish"]
    var speakers: [String] = []

    #if os(macOS)
    private var ttsModel: Qwen3TTSModel?
    private let player = AudioPlayer()
    var isPlaying: Bool { player.isPlaying }
    var modelLoaded: Bool { ttsModel != nil }
    #else
    private let synthesizer = AVSpeechSynthesizer()
    var isPlaying: Bool { synthesizer.isSpeaking }
    var modelLoaded: Bool { true }  // System TTS always available
    #endif

    func loadModel() async {
        #if os(macOS)
        isLoading = true
        errorMessage = nil
        loadingStatus = "Downloading model..."

        do {
            if ttsModel == nil {
                let model = try await Task.detached {
                    try await Qwen3TTSModel.fromPretrained(
                        modelId: TTSModelVariant.customVoice.rawValue
                    ) { (progress: Double, status: String) in
                        DispatchQueue.main.async { [weak self] in
                            self?.loadingStatus = status.isEmpty
                                ? "Downloading... \(Int(progress * 100))%"
                                : "\(status) (\(Int(progress * 100))%)"
                        }
                    }
                }.value
                ttsModel = model
                speakers = model.availableSpeakers
                if !speakers.contains(speaker) { speaker = speakers.first ?? "" }
            }
            loadingStatus = ""
        } catch {
            errorMessage = "Failed to load model: \(error.localizedDescription)"
            loadingStatus = ""
        }

        isLoading = false
        #endif
    }

    func synthesize() async {
        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            errorMessage = "Enter some text first."
            return
        }

        isSynthesizing = true
        errorMessage = nil

        #if os(macOS)
        guard let model = ttsModel else {
            errorMessage = "Model not loaded."
            isSynthesizing = false
            return
        }

        let inputText = text
        let inputLang = language
        let inputSpeaker = speaker
        let samples = model.synthesize(text: inputText, language: inputLang, speaker: inputSpeaker)

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
        #else
        // iOS: use system AVSpeechSynthesizer
        let utterance = AVSpeechUtterance(string: text)
        utterance.voice = avVoice(for: language)
        utterance.rate = AVSpeechUtteranceDefaultSpeechRate
        synthesizer.speak(utterance)
        #endif

        isSynthesizing = false
    }

    func stopPlayback() {
        #if os(macOS)
        player.stop()
        #else
        synthesizer.stopSpeaking(at: .immediate)
        #endif
    }

    #if os(iOS)
    private func avVoice(for language: String) -> AVSpeechSynthesisVoice? {
        let langCode: String
        switch language.lowercased() {
        case "english": langCode = "en-US"
        case "chinese": langCode = "zh-CN"
        case "japanese": langCode = "ja-JP"
        case "korean": langCode = "ko-KR"
        case "french": langCode = "fr-FR"
        case "german": langCode = "de-DE"
        case "spanish": langCode = "es-ES"
        default: langCode = "en-US"
        }
        return AVSpeechSynthesisVoice(language: langCode)
    }
    #endif
}
