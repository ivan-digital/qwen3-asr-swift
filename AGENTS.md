# Agent Instructions

AI speech models for Apple Silicon (MLX Swift). ASR, TTS, speech-to-speech, VAD, diarization, speech enhancement.

## Build

```bash
# Release build (recommended)
make build

# Debug build
make debug

# Run tests (builds debug first)
make test

# Clean
make clean
```

Or manually:

```bash
swift build -c release --disable-sandbox
./scripts/build_mlx_metallib.sh release
```

The metallib step compiles MLX Metal shaders — without it, inference runs ~5x slower due to JIT shader compilation.

## Project Structure

- `Sources/Qwen3ASR/` — Speech-to-text (Qwen3-ASR)
- `Sources/ParakeetASR/` — Speech-to-text (Parakeet TDT, CoreML)
- `Sources/Qwen3TTS/` — Text-to-speech (Qwen3-TTS)
- `Sources/CosyVoiceTTS/` — Text-to-speech (CosyVoice3, streaming)
- `Sources/PersonaPlex/` — Speech-to-speech (PersonaPlex 7B, full-duplex)
- `Sources/SpeechVAD/` — VAD (Silero + Pyannote), speaker diarization, speaker embedding (WeSpeaker)
- `Sources/SpeechEnhancement/` — Noise suppression (DeepFilterNet3, CoreML)
- `Sources/Qwen3Common/` — Shared model components (KV cache, RoPE, quantization)
- `Sources/AudioCommon/` — Audio I/O, protocols, HuggingFace downloader
- `Sources/AudioCLILib/` — CLI commands
- `Sources/AudioCLI/` — CLI entry point (`audio` binary)
- `Tests/` — Unit and integration tests
- `scripts/` — Model conversion (PyTorch → MLX/CoreML), benchmarking
- `Examples/` — Demo apps (PersonaPlexDemo, SpeechDemo)

## Key Conventions

- Swift 6, macOS 14+, Apple Silicon (M-series)
- MLX for GPU inference (Metal), CoreML for Neural Engine (DeepFilterNet3, Silero VAD optional)
- Models are downloaded from HuggingFace on first use, cached in `~/Library/Caches/qwen3-speech/`
- All audio processing uses Float32 PCM, resampled to model-specific rates internally
- `DiarizedSegment`, `SpeechSegment`, protocol types defined in `Sources/AudioCommon/Protocols.swift`
- Tests that use MLX arrays require the compiled metallib; config/logic-only tests work without it

## Testing

Safe tests (no GPU/model download required):
```bash
make test
```

Full test suite (requires metallib + model downloads):
```bash
make test
```

## CLI

The `audio` binary is the main entry point:

```bash
.build/release/audio transcribe recording.wav          # ASR
.build/release/audio speak "Hello" --output hi.wav     # TTS
.build/release/audio respond --input q.wav             # Speech-to-speech
.build/release/audio diarize meeting.wav               # Speaker diarization
.build/release/audio diarize meeting.wav --rttm        # RTTM output
.build/release/audio vad audio.wav                     # Voice activity detection
.build/release/audio embed-speaker voice.wav           # Speaker embedding
.build/release/audio denoise noisy.wav                 # Speech enhancement
```
