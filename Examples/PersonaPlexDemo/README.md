# PersonaPlexDemo

Voice assistant powered by [PersonaPlex](https://huggingface.co/nvidia/personaplex-7b-v1) — a 7B speech-to-speech model based on the [Moshi](https://github.com/kyutai-labs/moshi) architecture by Kyutai, fine-tuned by NVIDIA.

## About PersonaPlex

PersonaPlex is a **full-duplex speech-to-speech model** — it takes raw audio in and produces raw audio out, with no intermediate text ASR/TTS pipeline. Internally it generates text tokens as an "inner monologue" but speech is the primary modality.

### Architecture

- **Temporal Transformer**: 32-layer, 4096-dim, 32 heads — processes interleaved text + audio token streams
- **Depformer**: 6-layer, 1024-dim — predicts 16 audio codebook tokens per step using MultiLinear layers
- **Mimi Codec**: Neural audio codec (12.5Hz frame rate, 16 codebooks, 24kHz output)
- **4-bit quantized**: ~5.5 GB on disk (original FP16: ~14 GB)

### Model Behavior & Expectations

- **English only** — trained on English conversational data
- **No end-of-sequence token** — the model generates audio for exactly `maxSteps` frames. There is no "I'm done talking" signal. Set `maxSteps` to control response length (75 steps = ~6s)
- **Conversational but limited** — responses are contextually aware but the 7B model (4-bit) can produce tangential or repetitive answers. It works best for short exchanges
- **18 voice presets** — Natural Female/Male (NATF/NATM 0-3), Variety Female/Male (VARF/VARM 0-4)
- **System prompts** — pre-tokenized SentencePiece instructions that influence response style. The demo uses a combined "assistant + focused" prompt for concise, on-topic answers
- **Inner monologue** — the model generates text tokens alongside audio. These are decoded with SentencePiece and shown as "Model response" transcript. Quality varies

### Inference Modes

The library provides three inference APIs:

| Method | Returns | Threading | Use Case |
|--------|---------|-----------|----------|
| `respond()` | `(audio, textTokens)` | Synchronous (caller's thread) | Best for multi-turn — no thread-safety issues |
| `respondStream()` | `AsyncThrowingStream<AudioChunk>` | Internal background Task | Low-latency streaming playback |
| `respondDiagnostic()` | `(audio, DiagnosticInfo)` | Synchronous | Debugging — captures hidden state stats |

**This demo uses `respond()`** on a single detached task because MLX's compiler cache is not thread-safe — running ASR and PersonaPlex inference on separate threads crashes. Sequential execution on one thread is reliable for multi-turn conversation.

### Performance (M2 Max, 64 GB)

| Metric | Value |
|--------|-------|
| Model load | ~10s (cached) |
| Warmup (Metal compile) | ~15s |
| RTF (respond, compiled) | ~0.94 |
| ms/step | ~75ms |
| Max response | ~6s (75 steps at 12.5Hz) |
| ASR (Qwen3-ASR) | ~0.2s |

RTF < 1.0 means the model generates audio faster than real-time.

## Streaming & Real-Time

### Current Mode: Turn-Based

The demo records your full utterance, processes it, then plays the response:

```
Record (24kHz) → detect silence → ASR → respond() → play → repeat
```

This adds latency (wait for full recording + inference) but is reliable for multi-turn.

### Streaming Mode: `respondStream()`

`respondStream()` yields audio chunks as they're generated (~2s to first chunk). The demo doesn't use it currently because of an MLX thread-safety crash on multi-turn, but it's available for single-shot use:

```swift
let stream = model.respondStream(
    userAudio: audio,
    voice: .NATM0,
    systemPromptTokens: promptTokens,
    maxSteps: 75
)

for try await chunk in stream {
    player.scheduleChunk(chunk.samples)  // play immediately
    if chunk.isFinal {
        let transcript = spmDecoder.decode(chunk.textTokens)
    }
}
```

### True Real-Time: Full-Duplex (Not Yet Implemented)

PersonaPlex (Moshi architecture) is designed for **simultaneous input/output** — the model can listen and speak at the same time at 12.5Hz. A true real-time implementation would:

1. Continuously encode mic audio with Mimi codec (12.5Hz frames)
2. Feed user audio tokens to the temporal transformer each step
3. Generate agent audio tokens + decode with Mimi each step
4. Play decoded audio in real-time (80ms per frame)
5. No silence detection needed — the model handles turn-taking internally

This requires a `respondRealtime()` API that takes a continuous audio input stream and produces a continuous audio output stream, running the generation loop step-by-step in sync with real-time audio. The main challenges are:

- **MLX thread safety** — all MLX operations must stay on one thread
- **Latency budget** — each step must complete in <80ms to maintain real-time (currently ~75ms on M2 Max — tight but feasible)
- **Audio pipeline** — simultaneous mic capture + speaker playback with echo cancellation

Contributions welcome.

## Build & Run

```bash
cd Examples/PersonaPlexDemo
swift build -c release
```

### As a macOS app (recommended)

```bash
swift build -c release

APP="/tmp/PersonaPlexDemo.app"
mkdir -p "$APP/Contents/MacOS"

# Find the built binary
BINARY=$(find .build -name 'PersonaPlexDemo' -type f | head -1)
cp "$BINARY" "$APP/Contents/MacOS/"

# Copy MLX metallib (required for compiled inference)
METALLIB=$(find .build -name 'mlx.metallib' 2>/dev/null | head -1)
cp "$METALLIB" "$APP/Contents/MacOS/"

cat > "$APP/Contents/Info.plist" << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key><string>PersonaPlexDemo</string>
    <key>CFBundleIdentifier</key><string>com.example.PersonaPlexDemo</string>
    <key>CFBundleExecutable</key><string>PersonaPlexDemo</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>NSMicrophoneUsageDescription</key>
    <string>PersonaPlex needs microphone access for voice conversation.</string>
</dict>
</plist>
EOF

open "$APP"
```

## Usage

1. Click **Load PersonaPlex** — downloads ~5.5 GB + ~400 MB ASR on first run
2. Wait for warmup (~15s, compiles Metal kernels)
3. Tap the circle to start conversation
4. Speak — auto-detects when you stop (1.5s silence)
5. Model generates response, plays audio, shows transcripts
6. Auto-resumes listening after response
7. Tap again to stop

## Configuration

- **Voice**: 18 presets (NATF0-3, NATM0-3, VARF0-4, VARM0-4) — selectable in UI
- **Max Steps**: 75 default (~6s). Each step = 80ms audio at 12.5Hz. Adjustable in UI (50-500)
- **System Prompt**: Combined assistant + focused — "Answer questions clearly and concisely. Listen carefully to what the user says, then respond directly. Stay on topic. Be concise."
- **Input**: 24kHz mono audio (English only)

## Files

| File | Purpose |
|------|---------|
| `PersonaPlexDemoApp.swift` | App entry point |
| `PersonaPlexView.swift` | SwiftUI interface — toggle button, settings, transcripts |
| `PersonaPlexViewModel.swift` | Conversation state machine, ASR + inference orchestration |
| `AudioRecorder.swift` | Mic capture at 24kHz with RMS silence detection |
| `StreamingAudioPlayer.swift` | AVAudioEngine streaming playback |
| `AudioPlayer.swift` | Simple WAV file playback (unused in conversation mode) |
| `SentencePieceDecoder.swift` | Minimal protobuf parser for SPM vocabulary, decodes text tokens |
