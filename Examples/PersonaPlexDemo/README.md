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
- **No end-of-sequence token** — the model generates audio for exactly `maxSteps` frames. There is no "I'm done talking" signal. Set `maxSteps` to control response length (75 steps ≈ 6s)
- **Conversational but limited** — responses are contextually aware but the 7B model (4-bit) can produce tangential or repetitive answers. It works best for short exchanges
- **18 voice presets** — Natural Female/Male (NATF/NATM 0-3), Variety Female/Male (VARF/VARM 0-4)
- **System prompts** — pre-tokenized SentencePiece instructions that influence response style. The demo uses a combined "assistant + focused" prompt for concise, on-topic answers
- **Inner monologue** — the model generates text tokens alongside audio. These are decoded with SentencePiece and shown as "Model response" transcript. Quality varies

### Inference Modes

The library provides four inference APIs:

| Method | Returns | Threading | Use Case |
|--------|---------|-----------|----------|
| `respond()` | `(audio, textTokens)` | Synchronous (caller's thread) | Turn-based multi-turn conversation |
| `respondStream()` | `AsyncThrowingStream<AudioChunk>` | Internal background Task | Low-latency streaming playback |
| `respondRealtime()` | `AsyncThrowingStream<[Float]>` | Internal background Task | Full-duplex simultaneous I/O |
| `respondDiagnostic()` | `(audio, DiagnosticInfo)` | Synchronous | Debugging — captures hidden state stats |

**Turn-based mode** uses `respond()` on a single detached task because MLX's compiler cache is not thread-safe — running ASR and PersonaPlex inference on separate threads crashes. Sequential execution on one thread is reliable.

**Full-duplex mode** uses `respondRealtime()` which runs the generation loop in lock-step with audio I/O: mic samples are encoded via `mimi.encodeStep()` each step and injected into the token cache, while agent audio is decoded and yielded immediately.

### Performance (M2 Max, 64 GB)

| Metric | Value |
|--------|-------|
| Model load | ~10s (cached) |
| Warmup (Metal compile) | ~15s (first run), ~3s (cached) |
| RTF (respond, compiled) | ~0.94 |
| ms/step (full-duplex) | ~80–95ms |
| Max response | ~6s (75 steps at 12.5Hz, turn-based) |
| ASR (Qwen3-ASR) | ~0.2s |

RTF < 1.0 means the model generates audio faster than real-time.

> **Note on full-duplex latency**: Each step must complete in <80ms to sustain gapless audio. M2 Max averages ~85–95ms/step in full-duplex mode (slightly over budget). A 3-frame (~240ms) pre-buffer absorbs jitter and keeps playback smooth in practice.

## Inference Modes in the Demo

### Turn-Based (default)

Records your full utterance, runs ASR + inference, then plays the response:

```
Record (24kHz) → silence detection → ASR (Qwen3-ASR) → respond() → play → repeat
```

Adds latency (full recording + inference before playback) but is reliable for multi-turn conversation.

### Full-Duplex

Simultaneous mic input and audio output — the model listens and speaks at the same time:

```
Mic (continuous) ──→ AudioRingBuffer ──→ respondRealtime() ──→ StreamingAudioPlayer
                                              ↕ (80ms/step)
```

Key implementation details:
- `AudioRingBuffer`: thread-safe ring buffer bridging the mic capture thread and the MLX inference thread
- `mimi.encodeStep()`: encodes one 1920-sample (80ms) mic frame per step into user audio tokens
- Echo suppression: optionally writes zeros to the ring buffer while the agent is speaking (for speaker setups; not needed with headphones)
- `continuation.onTermination`: ensures the inference task is cancelled when the stream consumer stops, preventing multiple concurrent inference tasks from accumulating

## Build & Run

**Important**: The metallib must be built from within the `Examples/PersonaPlexDemo` directory, not from the repo root, to use the correct MLX version.

```bash
cd Examples/PersonaPlexDemo
swift build -c release --disable-sandbox
../../scripts/build_mlx_metallib.sh release
```

### As a macOS app (recommended)

SwiftUI requires an `.app` bundle for microphone permissions and proper window management.

```bash
cd Examples/PersonaPlexDemo
swift build -c release --disable-sandbox
../../scripts/build_mlx_metallib.sh release

APP="/tmp/PersonaPlexDemo.app"
rm -rf "$APP" && mkdir -p "$APP/Contents/MacOS"

cp .build/release/PersonaPlexDemo "$APP/Contents/MacOS/"
cp .build/release/mlx.metallib "$APP/Contents/MacOS/"

cat > "$APP/Contents/Info.plist" << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key><string>PersonaPlexDemo</string>
    <key>CFBundleIdentifier</key><string>com.example.PersonaPlexDemo</string>
    <key>CFBundleName</key><string>PersonaPlexDemo</string>
    <key>CFBundleVersion</key><string>1</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>LSMinimumSystemVersion</key><string>14.0</string>
    <key>NSMicrophoneUsageDescription</key>
    <string>PersonaPlex needs microphone access for voice conversation.</string>
</dict>
</plist>
EOF

open "$APP"
```

To see per-step timing logs, run the binary directly instead of `open`:

```bash
/tmp/PersonaPlexDemo.app/Contents/MacOS/PersonaPlexDemo
```

### Rebuilding after code changes

The metallib only needs to be rebuilt if the MLX dependency version changes. For code-only changes:

```bash
cd Examples/PersonaPlexDemo
swift build -c release --disable-sandbox
# Re-package the .app (same commands as above, starting from rm -rf "$APP")
```

## Usage

### Turn-Based Mode

1. Click **Load PersonaPlex** — downloads ~5.5 GB model + ~400 MB ASR on first run
2. Wait for warmup (~15s first run, ~3s subsequently)
3. Ensure **Mode** is set to **Turn-based**
4. Tap the circle to start listening
5. Speak — VAD auto-detects when you stop (~1s silence)
6. Model generates and plays the response, then resumes listening
7. Tap again to stop

### Full-Duplex Mode

1. Load the model (same as above)
2. Switch **Mode** to **Full-Duplex**
3. **Echo Suppression**: leave OFF for headphones; enable only when using external speakers
4. Tap the button — the model starts listening and speaking simultaneously
5. Speak naturally — no need to wait for the model to finish
6. Tap again to stop

> **Headphone tip**: Echo Suppression mutes the mic while the agent is speaking. With headphones there is no acoustic echo path, so suppression is unnecessary and prevents the model from hearing you during its responses. Keep it off.

## Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| Voice | NATM0 | 18 presets: NATF/NATM 0-3, VARF/VARM 0-4 |
| Max Steps | 75 | Turn-based only. Each step = 80ms at 12.5Hz. Range: 50–500 |
| Mode | Turn-based | Turn-based or Full-Duplex |
| Echo Suppression | Off | Full-duplex only. Enable for speaker output, disable for headphones |

**System Prompt** (pre-tokenized SentencePiece): "You are a helpful assistant. Answer questions clearly and concisely. Listen carefully to what the user says, then respond directly to their question or request. Stay on topic. Be concise."

## Files

| File | Purpose |
|------|---------|
| `PersonaPlexDemoApp.swift` | App entry point |
| `PersonaPlexView.swift` | SwiftUI interface — mode picker, echo suppression toggle, conversation button, transcripts |
| `PersonaPlexViewModel.swift` | State machine orchestrating turn-based and full-duplex conversation modes |
| `AudioRecorder.swift` | Mic capture at 24kHz; `startRecording()` for turn-based, `startContinuous()` for full-duplex |
| `StreamingAudioPlayer.swift` | AVAudioEngine streaming playback (`@unchecked Sendable` for direct background-task access) |
| `SentencePieceDecoder.swift` | Minimal protobuf parser for SPM vocabulary; decodes inner-monologue text tokens |
