import AppKit
import Foundation
import Observation
import ParakeetStreamingASR

/// Handles audio buffering and ASR processing off the main thread.
final class ASRProcessor: Sendable {
    private let session: StreamingSession
    private let lock = NSLock()
    private let _buffer = UnsafeMutablePointer<[Float]>.allocate(capacity: 1)

    init(session: StreamingSession) {
        self.session = session
        _buffer.initialize(to: [])
    }

    deinit {
        _buffer.deinitialize(count: 1)
        _buffer.deallocate()
    }

    /// Called from audio thread — just buffers samples.
    func appendAudio(_ samples: [Float]) {
        lock.lock()
        _buffer.pointee.append(contentsOf: samples)
        lock.unlock()
    }

    /// Check if enough audio is buffered for processing.
    var hasEnoughAudio: Bool {
        lock.lock()
        let count = _buffer.pointee.count
        lock.unlock()
        return count >= 5120
    }

    /// Process buffered audio and return partials. Called from processQueue.
    @discardableResult
    func processBuffered() -> [ParakeetStreamingASRModel.PartialTranscript] {
        lock.lock()
        let chunk = _buffer.pointee
        _buffer.pointee.removeAll(keepingCapacity: true)
        lock.unlock()

        guard !chunk.isEmpty else { return [] }

        do {
            return try session.pushAudio(chunk)
        } catch {
            print("[ASR] Error: \(error)")
            return []
        }
    }

    func finalize() -> [ParakeetStreamingASRModel.PartialTranscript] {
        let remaining = processBuffered()
        do {
            return remaining + (try session.finalize())
        } catch {
            print("[ASR] Finalize error: \(error)")
            return remaining
        }
    }
}

@Observable
@MainActor
final class DictateViewModel {
    var partialText = ""
    var committedText = ""
    var isRecording = false
    var isLoading = false
    var loadingStatus = ""
    var errorMessage: String?

    private var model: ParakeetStreamingASRModel?
    private var processor: ASRProcessor?
    private let recorder = StreamingRecorder()
    private let processQueue = DispatchQueue(label: "dictate.asr", qos: .userInteractive)

    var modelLoaded: Bool { model != nil }
    var audioLevel: Float { recorder.audioLevel }

    var fullText: String {
        if committedText.isEmpty { return partialText }
        if partialText.isEmpty { return committedText }
        return committedText + " " + partialText
    }

    // MARK: - Model Loading

    func loadModel() async {
        guard model == nil else { return }
        isLoading = true
        errorMessage = nil
        loadingStatus = "Downloading model..."

        do {
            let loaded = try await Task.detached {
                try await ParakeetStreamingASRModel.fromPretrained { [weak self] progress, status in
                    DispatchQueue.main.async {
                        self?.loadingStatus = status.isEmpty
                            ? "Downloading... \(Int(progress * 100))%"
                            : "\(status) (\(Int(progress * 100))%)"
                    }
                }
            }.value

            loadingStatus = "Warming up..."
            try loaded.warmUp()
            model = loaded
            loadingStatus = ""
            print("[Dictate] Model loaded and warmed up")
        } catch {
            errorMessage = "Failed to load: \(error.localizedDescription)"
            loadingStatus = ""
        }

        isLoading = false
    }

    // MARK: - Recording

    func toggleRecording() {
        if isRecording {
            stopRecording()
        } else {
            startRecording()
        }
    }

    func startRecording() {
        guard let model else { return }
        errorMessage = nil
        partialText = ""

        do {
            let session = try model.createSession()
            let proc = ASRProcessor(session: session)
            processor = proc
            print("[Dictate] Session created, starting recorder...")

            recorder.start { [weak self, proc] chunk in
                proc.appendAudio(chunk)
                guard proc.hasEnoughAudio else { return }

                self?.processQueue.async { [weak self] in
                    let partials = proc.processBuffered()
                    guard !partials.isEmpty else { return }
                    DispatchQueue.main.async {
                        for partial in partials {
                            print("[Dictate] \(partial.isFinal ? "FINAL" : "partial"): '\(partial.text)'")
                            if partial.isFinal {
                                self?.commitText(partial.text)
                                self?.partialText = ""
                            } else {
                                self?.partialText = partial.text
                            }
                        }
                    }
                }
            }
            isRecording = true
            print("[Dictate] Recording started")
        } catch {
            errorMessage = "Failed to start: \(error.localizedDescription)"
        }
    }

    func stopRecording() {
        recorder.stop()
        isRecording = false

        guard let processor else { return }
        let finals = processor.finalize()
        for partial in finals {
            if partial.isFinal && !partial.text.isEmpty {
                commitText(partial.text)
            }
        }
        self.processor = nil
        partialText = ""
    }

    // MARK: - Private

    private func commitText(_ text: String) {
        if committedText.isEmpty {
            committedText = text
        } else {
            committedText += " " + text
        }
    }

    // MARK: - Actions

    func pasteToFrontApp() {
        let text = fullText
        guard !text.isEmpty else { return }

        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(text, forType: .string)

        DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
            let src = CGEventSource(stateID: .hidSystemState)
            let keyDown = CGEvent(keyboardEventSource: src, virtualKey: 0x09, keyDown: true)
            keyDown?.flags = .maskCommand
            let keyUp = CGEvent(keyboardEventSource: src, virtualKey: 0x09, keyDown: false)
            keyUp?.flags = .maskCommand
            keyDown?.post(tap: .cghidEventTap)
            keyUp?.post(tap: .cghidEventTap)
        }
    }

    func copyToClipboard() {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(fullText, forType: .string)
    }

    func clearText() {
        committedText = ""
        partialText = ""
    }
}
