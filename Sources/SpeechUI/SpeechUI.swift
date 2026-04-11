/// SpeechUI provides minimal SwiftUI building blocks for streaming speech apps.
///
/// The module is intentionally narrow. It ships only what is genuinely
/// speech-domain — display of finals vs in-progress partials and the adapter
/// that lets any streaming ASR backend feed it. Generic audio utilities like
/// waveform rendering and level meters are out of scope; use AVFoundation or a
/// dedicated audio-visualization library for those.
///
/// - ``TranscriptionView`` — scrolling transcript with finals + partial line
/// - ``TranscriptionStore`` — `@Observable` adapter for any streaming ASR
public enum SpeechUI {}
