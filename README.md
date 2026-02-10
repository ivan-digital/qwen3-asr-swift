# Qwen3-ASR Swift

A Swift implementation of Qwen3 speech models using [MLX Swift](https://github.com/ml-explore/mlx-swift) for Apple Silicon.

- **Qwen3-ASR** — Speech-to-text (automatic speech recognition)
- **Qwen3-TTS** — Text-to-speech synthesis

Papers: [Qwen3-ASR](https://arxiv.org/abs/2601.21337), [Qwen3-TTS](https://arxiv.org/abs/2601.15621), [Mimi codec](https://arxiv.org/abs/2410.00037) (Kyutai)

## Models

| Model | Task | Download Size | HuggingFace |
|-------|------|--------------|-------------|
| Qwen3-ASR-0.6B (4-bit) | ASR | ~400 MB | `mlx-community/Qwen3-ASR-0.6B-4bit` |
| Qwen3-TTS-0.6B (4-bit) | TTS | ~977 MB + 651 MB codec | `mlx-community/Qwen3-TTS-12Hz-0.6B-Base-4bit` |

## Installation

### Swift Package Manager

Add to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/ivan-digital/qwen3-asr-swift", branch: "main")
]
```

Import the module you need:

```swift
import Qwen3ASR    // Speech recognition
import Qwen3TTS    // Text-to-speech
import Qwen3Common // Shared utilities
```

### Requirements

- Swift 5.9+
- macOS 14+ or iOS 17+
- Apple Silicon (M1/M2/M3/M4)
- Xcode 15+

## ASR Usage

### Basic Transcription

```swift
import Qwen3ASR

let model = try await Qwen3ASRModel.fromPretrained()
// Audio can be any sample rate — automatically resampled to 16kHz internally
let transcription = model.transcribe(audio: audioSamples, sampleRate: 16000)
print(transcription)
```

### ASR CLI

```bash
swift build -c release
.build/release/qwen3-asr-cli audio.wav
```

## TTS Usage

### Basic Synthesis

```swift
import Qwen3TTS
import Qwen3Common  // for WAVWriter

let model = try await Qwen3TTSModel.fromPretrained()
// Downloads ~1.7 GB on first run (model + codec weights)
let audio = model.synthesize(text: "Hello world", language: "english")
// Output is 24kHz mono float samples
try WAVWriter.write(samples: audio, sampleRate: 24000, to: outputURL)
```

### TTS CLI

```bash
swift build -c release
.build/release/qwen3-tts-cli "Hello world" --output output.wav --language english
```

### Sampling Options

```swift
let config = SamplingConfig(temperature: 0.9, topK: 50, repetitionPenalty: 1.05)
let audio = model.synthesize(text: "Hello", language: "english", sampling: config)
```

## Latency (M2 Max, 64 GB)

### ASR

| Model | Framework | RTF | 10s audio processed in |
|-------|-----------|-----|------------------------|
| Qwen3-ASR-0.6B (4-bit) | MLX Swift | ~0.06 | ~0.6s |
| Whisper-large-v3 | whisper.cpp (Q5_0) | ~0.10 | ~1.0s |
| Whisper-small | whisper.cpp (Q5_0) | ~0.04 | ~0.4s |

### TTS

| Model | Framework | Short (1s) | Medium (3s) | Long (6s) |
|-------|-----------|-----------|-------------|------------|
| Qwen3-TTS-0.6B (4-bit) | MLX Swift (release) | 1.6s (RTF 1.2) | 2.3s (RTF 0.7) | 3.9s (RTF 0.7) |
| Apple `AVSpeechSynthesizer` | AVFoundation | 0.08s | 0.08s | 0.17s (RTF 0.02) |

> Qwen3-TTS generates natural, expressive speech with prosody and emotion, running **faster than real-time** (RTF < 1.0). Apple's built-in TTS is ~35x faster but produces robotic, monotone speech.

RTF = Real-Time Factor (lower is better, < 1.0 = faster than real-time).

## Architecture

See [ASR Inference](docs/asr-inference.md), [ASR Model](docs/asr-model.md), [TTS Inference](docs/tts-inference.md), [TTS Model](docs/tts-model.md) for architecture details.

### ASR Pipeline

```
Audio (16kHz) -> Mel Spectrogram -> Audio Encoder (18L) -> Projector -> Text Decoder (28L) -> Text
```

### TTS Pipeline

```
Text -> BPE Tokenize -> Talker (28L, MRoPE) -> Code Predictor (5L) -> Codec Decoder (Mimi) -> Audio (24kHz)
```

## Cache Configuration

Model weights are cached locally. Override the cache location with:

```bash
export QWEN3_CACHE_DIR=/path/to/cache
```

## MLX Metal Library

If you see `Failed to load the default metallib`, build it manually:

```bash
xcodebuild -downloadComponent MetalToolchain
swift build -c release --disable-sandbox
./scripts/build_mlx_metallib.sh release
```

## Testing

Unit tests (config, sampling) run without model downloads:

```bash
swift test --filter "Qwen3TTSConfigTests|SamplingTests"
```

Integration tests require model weights (downloaded automatically on first run):

```bash
# TTS round-trip: synthesize text, save WAV, transcribe back with ASR
swift test --filter TTSASRRoundTripTests

# ASR only: transcribe test audio
swift test --filter Qwen3ASRIntegrationTests
```

> **Note:** MLX Metal library must be built before running tests that use MLX operations.
> See [MLX Metal Library](#mlx-metal-library) for instructions.

## Development Status

### ASR

- [x] Audio encoder + text decoder
- [x] Mel spectrogram preprocessing (Accelerate FFT)
- [x] HuggingFace model download
- [x] Tokenizer + language auto-detection
- [x] Inference optimizations (SDPA, vectorized preprocessing)

### TTS

- [x] Talker transformer (MRoPE, GQA, SDPA)
- [x] Code Predictor (residual codebook prediction)
- [x] Speech Tokenizer Decoder (Mimi codec)
- [x] Sampling (temperature, top-k, top-p, repetition penalty)
- [x] CLI tool with WAV output
- [x] BPE text encoding

## Roadmap

- [ ] TTS streaming inference
- [ ] TTS voice cloning (speaker encoder)
- [ ] TTS voice design
- [x] TTS inference optimizations (chunked decode, batch embeddings)
- [ ] ASR streaming inference
- [ ] iOS app example

## Supported Languages

### ASR (52 Languages)

Chinese, English, Cantonese, Arabic, German, French, Spanish, Portuguese, Indonesian, Italian, Korean, Russian, Thai, Vietnamese, Japanese, Turkish, Hindi, Malay, Dutch, Swedish, Danish, Finnish, Polish, Czech, Filipino, Persian, Greek, Hungarian, Macedonian, Romanian + 22 Chinese dialects

### TTS (4 Languages)

English, Chinese, German, Japanese (more via language ID extension)

## License

Apache 2.0 (same as original Qwen3 models)

