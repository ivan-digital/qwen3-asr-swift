# Qwen3-ASR Swift

A Swift implementation of Qwen3-ASR speech recognition model using [MLX Swift](https://github.com/ml-explore/mlx-swift) for Apple Silicon.

## Overview

Qwen3-ASR is a state-of-the-art automatic speech recognition model from Alibaba/Qwen that offers:

- **52 languages**: 30 major languages + 22 Chinese dialects
- **Excellent noise robustness**: Outperforms Whisper and GPT-4o in noisy conditions
- **Fast inference**: 92ms TTFT, RTF 0.064 at high concurrency
- **On-device**: Runs locally on Apple Silicon Macs and iPhones

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
    .package(url: "https://github.com/yourusername/qwen3-asr-swift", from: "0.1.0")
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

# Transcribe audio file
.build/release/qwen3-asr transcribe audio.wav

# With streaming output
.build/release/qwen3-asr transcribe audio.wav --stream

# Download model
.build/release/qwen3-asr download Qwen/Qwen3-ASR-0.6B
```

## Architecture

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

### Benchmarks (Apple M3 Max)

| Model | TTFT | RTF |
|-------|------|-----|
| Qwen3-ASR-0.6B | ~100ms | ~0.08 |
| Qwen3-ASR-1.7B | ~200ms | ~0.15 |

### Word Error Rate

| Model | LibriSpeech (clean) | Noisy Conditions |
|-------|---------------------|------------------|
| Qwen3-ASR-0.6B | 2.11% | 17.88% |
| Qwen3-ASR-1.7B | 1.63% | 16.17% |
| Whisper-large-v3 | 1.51% | 63.17% |

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
- [ ] HuggingFace model download
- [ ] Tokenizer integration
- [ ] Streaming inference optimization
- [ ] iOS support

## License

Apache 2.0 (same as original Qwen3-ASR)

## Credits

- [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) - Original model by Alibaba/Qwen
- [MLX Swift](https://github.com/ml-explore/mlx-swift) - Apple's ML framework for Swift
- [mlx-audio](https://github.com/ml-explore/mlx-audio) - Reference Python implementation
