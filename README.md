# Qwen3-ASR Swift

A Swift implementation of Qwen3-ASR speech recognition model using [MLX Swift](https://github.com/ml-explore/mlx-swift) for Apple Silicon.

## Overview

Qwen3-ASR is a state-of-the-art automatic speech recognition model from Alibaba/Qwen that offers:

- **52 languages**: 30 major languages + 22 Chinese dialects
- **Excellent noise robustness**: Outperforms Whisper and GPT-4o in noisy conditions
- **Fast inference**: 92ms TTFT, RTF 0.064 at high concurrency
- **On-device**: Runs locally on Apple Silicon Macs and iPhones

## Latency (Apple Silicon, 10s audio)

| Model | Framework | RTF | 10s audio processed in |
|-------|-----------|-----|------------------------|
| Qwen3-ASR-0.6B (4-bit) | MLX Swift | ~0.06 | ~0.6s |
| Whisper-large-v3 | whisper.cpp (Q5_0) | ~0.10 | ~1.0s |
| Whisper-small | whisper.cpp (Q5_0) | ~0.04 | ~0.4s |
| Whisper-large-v3 | MLX Python | ~0.15 | ~1.5s |

RTF = Real-Time Factor (lower is better, < 1.0 = faster than real-time).

## Models

| Model | Parameters | Use Case |
|-------|-----------|----------|
| Qwen3-ASR-0.6B | 600M | Efficient, on-device |
| Qwen3-ASR-1.7B | 1.7B | Best accuracy |

## Installation

### Swift Package Manager

Add to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/ivan-digital/qwen3-asr-swift", from: "0.1.0")
]
```

### Requirements

- macOS 14+ or iOS 17+
- Apple Silicon (M1/M2/M3/M4)
- Xcode 15+

## Usage

### Basic Transcription

```swift
import Qwen3ASR

// Load model
let model = try await Qwen3ASRModel.fromPretrained(modelId: "Qwen/Qwen3-ASR-0.6B")

// Transcribe audio (24kHz mono float samples)
let transcription = model.transcribe(audio: audioSamples, sampleRate: 24000)
print(transcription)
```

### Streaming Transcription

```swift
await model.streamTranscribe(audio: audioSamples, sampleRate: 24000) { token in
    print(token, terminator: "")
}
```

### CLI Tool

```bash
# Build CLI
swift build -c release

# Transcribe audio file (model is downloaded automatically on first run)
.build/release/qwen3-asr-cli audio.wav
```

### Cache Configuration

Model weights are cached locally. Override the cache location with:

```bash
export QWEN3_ASR_CACHE_DIR=/path/to/cache
```

### MLX Metal Library

If you see `Failed to load the default metallib`, build it manually:

```bash
xcodebuild -downloadComponent MetalToolchain
swift build -c release --disable-sandbox
./scripts/build_mlx_metallib.sh release
```

## Architecture

See [Inference Architecture & Performance](docs/inference-architecture.md) for detailed optimization notes.

```
Audio Input (24kHz)
    │
    ▼
┌─────────────────┐
│  Mel Spectrogram│  (WhisperFeatureExtractor)
│  128 bins, 8ms  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Audio Encoder  │  (Conv2D + Transformer)
│  12/18 layers   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Projector     │  (2-layer MLP)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Text Decoder   │  (Qwen3 LLM)
│  28 layers      │
└────────┬────────┘
         │
         ▼
     Text Output
```

## Performance

### Comparison with Whisper

| | Qwen3-ASR-0.6B (4-bit) | Whisper-large-v3 | Whisper-small |
|---|---|---|---|
| Parameters | 600M | 1.5B | 244M |
| LibriSpeech (clean) WER | 2.11% | 1.51% | 3.43% |
| Noisy conditions WER | 17.88% | 63.17% | - |
| Languages | 52 | 99 | 99 |
| On-device (Apple Silicon) | Yes (MLX) | Via whisper.cpp | Via whisper.cpp |

Qwen3-ASR offers significantly better noise robustness than Whisper while maintaining competitive clean-speech accuracy at a fraction of the model size.

## Supported Languages

### Major Languages (30)
Chinese, English, Cantonese, Arabic, German, French, Spanish, Portuguese, Indonesian, Italian, Korean, Russian, Thai, Vietnamese, Japanese, Turkish, Hindi, Malay, Dutch, Swedish, Danish, Finnish, Polish, Czech, Filipino, Persian, Greek, Hungarian, Macedonian, Romanian

### Chinese Dialects (22)
Anhui, Dongbei, Fujian, Gansu, Guizhou, Hebei, Henan, Hubei, Hunan, Jiangxi, Ningxia, Shandong, Shaanxi, Shanxi, Sichuan, Tianjin, Yunnan, Zhejiang, Cantonese (HK/Guangdong), Wu, Minnan

## Development Status

- [x] Configuration classes
- [x] Audio encoder (Conv2D + Transformer)
- [x] Text decoder (Qwen3)
- [x] Audio preprocessing (Mel spectrogram)
- [x] Weight loading infrastructure
- [x] HuggingFace model download
- [x] Tokenizer integration
- [x] Language auto-detection
- [x] iOS support (iOS 17+)
- [x] Inference optimizations (SDPA, Accelerate FFT, vectorized preprocessing)
- [ ] Streaming inference

## License

Apache 2.0 (same as original Qwen3-ASR)

## Credits

- [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) - Original model by Alibaba/Qwen
- [MLX Swift](https://github.com/ml-explore/mlx-swift) - Apple's ML framework for Swift
- [mlx-audio](https://github.com/ml-explore/mlx-audio) - Reference Python implementation
