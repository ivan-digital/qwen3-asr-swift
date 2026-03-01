import AVFoundation
import AudioCommon
import Observation

@Observable
final class AudioPlayer: NSObject, AVAudioPlayerDelegate {
    private(set) var isPlaying = false
    private var player: AVAudioPlayer?
    private var tempURL: URL?

    func play(samples: [Float], sampleRate: Int = 24000) throws {
        stop()

        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("speech_demo_\(UUID().uuidString).wav")
        try WAVWriter.write(samples: samples, sampleRate: sampleRate, to: url)
        tempURL = url

        let audioPlayer = try AVAudioPlayer(contentsOf: url)
        audioPlayer.delegate = self
        audioPlayer.play()
        player = audioPlayer
        isPlaying = true
    }

    func stop() {
        player?.stop()
        player = nil
        isPlaying = false
        if let url = tempURL {
            try? FileManager.default.removeItem(at: url)
            tempURL = nil
        }
    }

    func audioPlayerDidFinishPlaying(_ player: AVAudioPlayer, successfully _: Bool) {
        DispatchQueue.main.async { [weak self] in
            self?.isPlaying = false
        }
        if let url = tempURL {
            try? FileManager.default.removeItem(at: url)
            tempURL = nil
        }
    }
}
