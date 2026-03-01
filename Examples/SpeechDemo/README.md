# SpeechDemo

A minimal SwiftUI macOS app demonstrating the Qwen3 Speech library: mic recording with ASR transcription and text-to-speech synthesis with playback.

## Requirements

- macOS 14+
- Apple Silicon (M1 or later)
- Xcode 15+

## Build & Run

1. Open `Examples/SpeechDemo/` in Xcode (File > Open, select this folder)
2. Xcode resolves the SPM dependency on the parent `Qwen3Speech` package automatically
3. Select the `SpeechDemo` scheme and press Run

### Entitlements (Xcode project only)

If running from an Xcode project, add these entitlements to the signing config:

- `com.apple.security.app-sandbox` — required for App Sandbox
- `com.apple.security.device.audio-input` — microphone access for recording
- `com.apple.security.network.client` — outbound network for model downloads

The entitlements file is provided at `SpeechDemo/SpeechDemo.entitlements`.

### Command-line build

```bash
cd Examples/SpeechDemo
swift build
```

Note: The command-line build produces an executable, not a sandboxed .app bundle. Microphone access requires running from Xcode with proper entitlements, or granting Terminal microphone permission in System Settings > Privacy & Security > Microphone.

## Usage

The app has two tabs:

### Dictate

1. Select an ASR engine: **Parakeet TDT** (CoreML, ~400 MB) or **Qwen3-ASR** (MLX, ~400 MB)
2. Click **Load** to download the model (first run only — cached in `~/Library/Caches/qwen3-speech/`)
3. Hold the **Record** button, speak, then release
4. The transcription appears below — click **Copy** to copy to clipboard

### Speak

1. Click **Load Qwen3-TTS** to download the model (~1 GB, first run only)
2. Type or paste text into the text field
3. Select a language from the dropdown
4. Click **Synthesize** — audio plays automatically when ready

## Models downloaded on first run

| Model | Size | Cache location |
|-------|------|----------------|
| Parakeet TDT (CoreML INT4) | ~400 MB | `~/Library/Caches/qwen3-speech/aufklarer_Parakeet-TDT-v3-CoreML-INT4/` |
| Qwen3-ASR (MLX 4-bit) | ~400 MB | `~/Library/Caches/qwen3-speech/aufklarer_Qwen3-ASR-0.6B-MLX-4bit/` |
| Qwen3-TTS (MLX 4-bit) | ~1 GB | `~/Library/Caches/qwen3-speech/aufklarer_Qwen3-TTS-12Hz-0.6B-Base-MLX-4bit/` |
| Qwen3-TTS Tokenizer | ~650 MB | `~/Library/Caches/qwen3-speech/Qwen_Qwen3-TTS-Tokenizer-12Hz/` |
