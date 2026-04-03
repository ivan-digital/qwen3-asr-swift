# iOS Echo Demo

ASR → TTS echo pipeline. Speak and hear it back.

- **Device**: Parakeet ASR + Kokoro TTS (CoreML E2E)
- **Simulator**: Parakeet ASR + Apple built-in TTS

## Setup

```bash
cd Examples/iOSEchoDemo
xcodegen generate
open iOSEchoDemo.xcodeproj
```

Set your signing team in Xcode, build and run.

Models (~500 MB) download from HuggingFace on first launch.

## Features

- Voice activity detection (Silero VAD)
- Force-cut at 5s with system message
- Adaptive echo prevention (cooldown based on TTS audio duration)
- Diagnostics view (CPU, memory, VAD level)
