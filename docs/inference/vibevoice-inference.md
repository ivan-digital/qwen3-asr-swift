# VibeVoice — Inference Pipeline

TTS in English + Chinese. Two variants with **different APIs** because the
underlying architectures are different (see
[vibevoice.md](../models/vibevoice.md) for the architectural diff):

- `VibeVoiceTTSModel` — 0.5B Realtime, voice-cache flow, low latency
- `VibeVoice15BTTSModel` — 1.5B long-form, dual-encoder single-shot, podcast quality

## Quick start — 0.5B Realtime

```swift
import VibeVoiceTTS

let tts = try await VibeVoiceTTSModel.fromPretrained()
try tts.loadVoice(from: "/path/to/voice_cache/en-Mike_man.safetensors")
let pcm = try await tts.generate(text: "Hello world.")
// pcm: [Float] at 24 kHz mono
```

## Quick start — 1.5B long-form

```swift
import VibeVoiceTTS

let tts = try await VibeVoice15BTTSModel.fromPretrained()
let pcm = try await tts.generate(
    text: "Hello world. This is the long-form variant.",
    referenceAudio: refSamples,            // [Float] mono speech, any rate
    referenceTranscript: "",               // optional, currently unused
    sampleRate: 24000
)
```

Reference audio carries the speaker voice (real human speech needed for a
real clone — sine waves give garbled output). `Configuration` defaults:
`numInferenceSteps=20`, `cfgScale=1.5`, `maxSpeechTokens=4000`, model
`aufklarer/VibeVoice-1.5B-MLX-INT4`, tokenizer `Qwen/Qwen2.5-1.5B`.

Conforms to `SpeechGenerationModel`, so it drops into any code path that takes
a generic TTS model (voice pipelines, audiobook generation, etc.).

## Configuration

```swift
var config = VibeVoiceTTSModel.Configuration(
    modelId: "microsoft/VibeVoice-Realtime-0.5B",
    tokenizerModelId: "Qwen/Qwen2.5-0.5B",
    numInferenceSteps: 20,      // DPM-Solver steps (higher = slower, higher quality)
    cfgScale: 1.3,              // classifier-free guidance
    maxSpeechTokens: 500        // cap per generate()
)

// Or pick the long-form 1.5B preset:
let config = VibeVoiceTTSModel.Configuration.longForm1_5B
//   modelId: microsoft/VibeVoice-1.5B, tokenizer: Qwen/Qwen2.5-1.5B,
//   cfgScale: 1.5, maxSpeechTokens: 4000

let tts = try await VibeVoiceTTSModel.fromPretrained(configuration: config)
```

## Voice cache

Speaker identity comes from a `.safetensors` voice cache — see
[vibevoice.md](../models/vibevoice.md#voice-cloning-via-voice-cache) for the
format. Swap voices cheaply on a loaded model:

```swift
try tts.loadVoice(from: URL(fileURLWithPath: "en-Mike_man.safetensors"))
let a = try await tts.generate(text: "First line.")
try tts.loadVoice(from: URL(fileURLWithPath: "en-Emma_woman.safetensors"))
let b = try await tts.generate(text: "Second line.")
```

### Mint a voice cache from reference audio

```swift
let url = try tts.encodeAndSaveVoice(
    referenceAudio: pcm24k,            // [Float] mono samples
    sampleRate: 24000,
    transcript: "actual words spoken in the audio",
    to: URL(fileURLWithPath: "voices/my-voice.safetensors")
)
// `tts` now has the new voice loaded.
```

Or via the CLI:

```bash
audio vibevoice-encode-voice reference.wav "actual transcript" \
    --output voices/my-voice.safetensors
```

The encoder runs the audio through `acoustic_tokenizer.encode` and the
transcript through both LMs, capturing per-layer KV caches and hidden
states. Encoding is fast (a 17-second clip encodes in ~2 s on M2 Max).

## Streaming

`VibeVoiceTTSModel` inherits the default `SpeechGenerationModel.generateStream`
which yields a single `AudioChunk` containing the full waveform (since VibeVoice
generates as a whole before returning). For play-while-generating with true
chunk streaming, wire `AudioCommon.StreamingAudioPlayer.scheduleChunk(_:)` into
the internal chunk loop — PR welcome.

## CLI

```bash
# 0.5B Realtime (default) — voice-cache flow
audio vibevoice "Hello world." \
  --voice-cache voice_cache/en-Mike_man.safetensors \
  --output hello.wav

# 1.5B long-form — single-shot with reference audio + transcript
audio vibevoice "Long paragraph ..." \
  --long-form \
  --reference-audio reference_speech.wav \
  --reference-transcript "exact transcript of the reference" \
  --max-tokens 500 --steps 20 \
  --output episode.wav

# Mint a 0.5B voice cache from a recording
audio vibevoice-encode-voice reference.wav \
  "exact transcript of the recording" \
  --output voices/my-voice.safetensors
```

Common flags: `--steps` (DPM-Solver steps), `--cfg` (guidance scale),
`--model` / `--tokenizer` to override HuggingFace IDs, `--verbose` for timing.

## Picking among speech-swift TTS modules

| | Kokoro-82M | Qwen3-TTS | CosyVoice3 | VibeVoice Realtime-0.5B | VibeVoice 1.5B |
|---|---|---|---|---|---|
| Params | 82M | 7B (+0.6B ICL) | 7B | 500M | 1.5B |
| Backend | CoreML (ANE) | MLX | MLX | MLX | MLX |
| Languages | 8 | 10+ | 10+ | EN/ZH | EN/ZH |
| Voice cloning | Fixed presets | ICL reference audio | Zero-shot reference | Voice cache | Voice cache |
| Long-form | Short/medium | Streaming | Streaming | Streaming | **Up to 90 min / 4 speakers** |
| Emotion control | Preset-encoded | Via prompt | Reference emotion | None explicit | None explicit |

Use VibeVoice when you need long-form (multi-minute) output or
multi-speaker podcast / audiobook generation. Use Kokoro for short
iOS-native synthesis; Qwen3-TTS / CosyVoice for flexible multilingual
single-utterance synthesis.
