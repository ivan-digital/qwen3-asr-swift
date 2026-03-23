---
name: run-demo
description: Build and launch SpeechDemo macOS app. Use for testing Echo (voice pipeline), Speak (TTS), or Dictate (ASR) tabs.
disable-model-invocation: true
argument-hint: [speech|personaplex]
allowed-tools: Bash
---

# Run Demo

Build and launch a demo app.

## SpeechDemo (default)

Three tabs: Dictate (ASR), Speak (TTS), Echo (full voice pipeline).

```bash
cd Examples/SpeechDemo
swift build -c debug --disable-sandbox 2>&1 | tail -5
# Copy metallib if needed
cp ../../.build/arm64-apple-macosx/debug/mlx.metallib .build/debug/mlx.metallib 2>/dev/null
cp ../../.build/arm64-apple-macosx/debug/mlx.metallib .build/arm64-apple-macosx/debug/mlx.metallib 2>/dev/null
.build/debug/SpeechDemo &
echo "SpeechDemo launched"
```

## Echo Tab State Machine

```
idle → [Load Models] → loading → loaded
loaded → [Start] → listening
listening → [VAD onset] → speech_detected
speech_detected → [VAD offset + silence] → transcribing
transcribing → [STT complete] → synthesizing
synthesizing → [TTS chunk] → speaking
speaking → [all chunks played] → listening
speaking → [user interrupts] → listening (TTS cancelled)
any → [Stop] → idle
```

## Debug files

After stopping Echo, debug files are saved:
- Mic recording: `/tmp/echo_debug_mic.wav` (16kHz mono)
- TTS output: `/tmp/echo_debug_tts.wav` (24kHz mono)
- Event log: `/tmp/echo_debug.log`

## Known limitations

- AEC not perfect — TTS playback can leak into mic, causing false VAD triggers
- STT takes 2-3s (Parakeet CoreML) — phrases queued during inference
- TTS warmup chunks (first 0.2s) are dropped (codec decoder initialization noise)
