import Foundation
import PersonaPlex
import Qwen3ASR
import AudioCommon

enum ConversationState: String {
    case inactive
    case listening
    case processing
    case speaking
}

@Observable
@MainActor
final class PersonaPlexViewModel {
    // MARK: - State

    var conversationState: ConversationState = .inactive
    var isLoading = false
    var loadingStatus: String?
    var errorMessage: String?
    var userTranscript = ""
    var modelTranscript = ""
    var latencyInfo: String?
    var debugInfo: String = ""

    var selectedVoice: PersonaPlexVoice = .NATM0
    var maxSteps: Int = 75

    // Combined assistant + focused prompt tokens (SentencePiece):
    // "<system> You are a helpful assistant. Answer questions clearly and concisely.
    //  Listen carefully to what the user says, then respond directly to their question
    //  or request. Stay on topic. Be concise. <system>"
    private let systemPromptTokens: [Int32] = [
        607, 4831, 578, 493, 298, 272, 3850, 5019, 263,
        506, 1292, 2366, 267, 22876, 362, 263,
        17453, 6716, 269, 419, 262, 819, 1182, 261, 409,
        4816, 1312, 269, 347, 560, 307, 2498, 263, 17308,
        291, 3398, 263, 1451, 22876, 263, 607, 4831, 578
    ]

    // MARK: - Private

    private var model: PersonaPlexModel?
    private var asrModel: Qwen3ASRModel?
    private let recorder = AudioRecorder(targetSampleRate: 24000)
    private let player = StreamingAudioPlayer()
    private var spmDecoder: SentencePieceDecoder?
    private var conversationTask: Task<Void, Never>?

    // MARK: - Computed

    var modelLoaded: Bool { model != nil }
    var isActive: Bool { conversationState != .inactive }
    var audioLevel: Float { recorder.audioLevel }
    var isBusy: Bool { isLoading }

    // MARK: - Model Loading

    func loadModel() async {
        isLoading = true
        loadingStatus = "Downloading PersonaPlex (~5.5 GB)..."
        errorMessage = nil

        do {
            let loadStart = CFAbsoluteTimeGetCurrent()

            let m = try await PersonaPlexModel.fromPretrained { [weak self] progress, status in
                DispatchQueue.main.async {
                    self?.loadingStatus = "\(status) (\(Int(progress * 100))%)"
                }
            }
            loadingStatus = "Warming up (compiling Metal kernels)..."
            let warmStart = CFAbsoluteTimeGetCurrent()
            await Task.detached { m.warmUp() }.value
            let warmTime = CFAbsoluteTimeGetCurrent() - warmStart

            let cacheDir = try HuggingFaceDownloader.getCacheDirectory(
                for: "aufklarer/PersonaPlex-7B-MLX-4bit"
            )
            let spmPath = cacheDir.appendingPathComponent("tokenizer_spm_32k_3.model").path
            if FileManager.default.fileExists(atPath: spmPath) {
                spmDecoder = try SentencePieceDecoder(modelPath: spmPath)
            }

            model = m

            loadingStatus = "Loading ASR model (~400 MB)..."
            let asr = try await Qwen3ASRModel.fromPretrained { [weak self] progress, status in
                DispatchQueue.main.async {
                    self?.loadingStatus = "ASR: \(status) (\(Int(progress * 100))%)"
                }
            }
            asrModel = asr

            let loadTime = CFAbsoluteTimeGetCurrent() - loadStart
            latencyInfo = String(format: "Models loaded in %.1fs (warmup %.1fs)", loadTime, warmTime)
            loadingStatus = nil
        } catch {
            errorMessage = "Failed to load model: \(error.localizedDescription)"
            loadingStatus = nil
        }
        isLoading = false
    }

    // MARK: - Conversation Control

    func toggleConversation() {
        if conversationState == .inactive {
            startListening()
        } else {
            stopConversation()
        }
    }

    private func startListening() {
        conversationState = .listening
        errorMessage = nil
        debugInfo = "Listening..."

        recorder.onSilenceAfterSpeech = { [weak self] in
            Task { @MainActor [weak self] in
                self?.onSilenceDetected()
            }
        }
        recorder.startRecording()
    }

    private func onSilenceDetected() {
        guard conversationState == .listening else { return }
        debugInfo = "Processing..."
        conversationTask = Task { await processAndRespond() }
    }

    private func stopConversation() {
        conversationTask?.cancel()
        conversationTask = nil
        recorder.onSilenceAfterSpeech = nil
        if recorder.isRecording {
            _ = recorder.stopRecording()
        }
        player.stop()
        conversationState = .inactive
        debugInfo = ""
    }

    // MARK: - Process & Respond

    private func processAndRespond() async {
        let rawAudio = recorder.stopRecording()

        guard rawAudio.count > 2400 else {
            startListening()
            return
        }

        conversationState = .processing
        userTranscript = ""
        modelTranscript = ""
        latencyInfo = nil

        let audio = Self.trimTrailingSilence(rawAudio, sampleRate: 24000, threshold: 0.01)
        let audioDuration = Double(audio.count) / 24000.0

        guard let model, let asr = asrModel else {
            errorMessage = "Model not loaded."
            stopConversation()
            return
        }

        // Run ASR + inference on a SINGLE detached task to avoid MLX thread-safety crash
        let voice = selectedVoice
        let promptTokens = systemPromptTokens
        let steps = maxSteps
        let spmDec = spmDecoder

        debugInfo = "Transcribing + generating..."
        let overallStart = CFAbsoluteTimeGetCurrent()

        let result = await Task.detached(priority: .userInitiated) {
            () -> (String, Double, [Float], [Int32], String, Double) in

            // ASR
            let audio16k = Self.downsample(audio, from: 24000, to: 16000)
            let asrStart = CFAbsoluteTimeGetCurrent()
            let asrText = asr.transcribe(audio: audio16k, sampleRate: 16000, language: "en")
            let asrTime = CFAbsoluteTimeGetCurrent() - asrStart

            // Inference (same thread, no concurrent MLX access)
            let inferStart = CFAbsoluteTimeGetCurrent()
            let (response, textTokens) = model.respond(
                userAudio: audio,
                voice: voice,
                systemPromptTokens: promptTokens,
                maxSteps: steps
            )
            let inferTime = CFAbsoluteTimeGetCurrent() - inferStart

            // Decode model transcript
            var transcript = ""
            if let dec = spmDec, !textTokens.isEmpty {
                transcript = dec.decode(textTokens)
            } else if !textTokens.isEmpty {
                transcript = "(\(textTokens.count) text tokens)"
            }

            return (asrText, asrTime, response, textTokens, transcript, inferTime)
        }.value

        guard conversationState != .inactive else { return }

        let (asrText, asrTime, response, _, transcript, inferTime) = result
        let overallTime = CFAbsoluteTimeGetCurrent() - overallStart

        userTranscript = asrText
        modelTranscript = transcript

        let responseDuration = Double(response.count) / 24000.0
        let rtf = inferTime / max(responseDuration, 0.01)

        latencyInfo = String(
            format: "ASR: %.1fs | Infer: %.1fs | Total: %.1fs | %.1fs in / %.1fs out | RTF: %.2f",
            asrTime, inferTime, overallTime, audioDuration, responseDuration, rtf
        )

        // Play response
        if !response.isEmpty {
            conversationState = .speaking
            debugInfo = "Speaking..."
            do {
                try player.start(sampleRate: 24000)
                player.scheduleChunk(response)
                await player.waitForCompletion()
                player.stop()
            } catch {
                errorMessage = "Playback error: \(error.localizedDescription)"
            }

            if conversationState == .speaking {
                startListening()
            }
        } else {
            errorMessage = "Model returned empty audio."
            startListening()
        }
    }

    // MARK: - Audio Processing

    nonisolated private static func trimTrailingSilence(
        _ samples: [Float], sampleRate: Int, threshold: Float
    ) -> [Float] {
        let chunkSize = sampleRate / 20
        var lastSpeechEnd = samples.count
        var i = samples.count
        while i > chunkSize {
            i -= chunkSize
            let end = min(i + chunkSize, samples.count)
            var sum: Float = 0
            for j in i..<end { sum += samples[j] * samples[j] }
            let rms = sqrt(sum / Float(end - i))
            if rms > threshold {
                lastSpeechEnd = end + sampleRate / 10
                break
            }
        }
        return Array(samples.prefix(min(lastSpeechEnd, samples.count)))
    }

    nonisolated private static func downsample(
        _ samples: [Float], from srcRate: Int, to dstRate: Int
    ) -> [Float] {
        guard srcRate != dstRate, !samples.isEmpty else { return samples }
        let ratio = Double(srcRate) / Double(dstRate)
        let outCount = Int(Double(samples.count) / ratio)
        var result = [Float](repeating: 0, count: outCount)
        for i in 0..<outCount {
            let srcIdx = Double(i) * ratio
            let idx0 = Int(srcIdx)
            let frac = Float(srcIdx - Double(idx0))
            let s0 = samples[min(idx0, samples.count - 1)]
            let s1 = samples[min(idx0 + 1, samples.count - 1)]
            result[i] = s0 + frac * (s1 - s0)
        }
        return result
    }
}
