# Speech Swift

AI speech models for Apple Silicon, powered by [MLX Swift](https://github.com/ml-explore/mlx-swift).

- **Qwen3-ASR** — Speech-to-text (automatic speech recognition)
- **Qwen3-TTS** — Text-to-speech synthesis (highest quality, custom speakers)
- **CosyVoice TTS** — Text-to-speech with streaming (9 languages, DiT flow matching)

Papers: [Qwen3-ASR](https://arxiv.org/abs/2601.21337), [Qwen3-TTS](https://arxiv.org/abs/2601.15621), [CosyVoice 3](https://arxiv.org/abs/2505.17589), [Mimi](https://arxiv.org/abs/2410.00037) (audio codec used by Qwen3-TTS)

## Models

| Model | Task | Streaming | Languages | Download Size |
|-------|------|-----------|-----------|--------------|
| Qwen3-ASR-0.6B (4-bit) | Speech → Text | No | 52 languages | ~400 MB |
| Qwen3-ASR-1.7B (8-bit) | Speech → Text | No | 52 languages | ~2.5 GB |
| Qwen3-TTS-0.6B Base (4-bit) | Text → Speech | No | 10 languages | ~1.7 GB |
| Qwen3-TTS-0.6B CustomVoice (4-bit) | Text → Speech | No | 10 languages | ~1.7 GB |
| CosyVoice3-0.5B (4-bit) | Text → Speech | Yes (~150ms) | 9 languages | ~1.9 GB |

### When to Use Which TTS

- **Qwen3-TTS**: Best quality, 9 built-in speakers, 10 languages, batch synthesis
- **CosyVoice TTS**: Low-latency streaming (~150ms), 9 languages, DiT flow matching + HiFi-GAN vocoder

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
import Qwen3TTS    // Text-to-speech (Qwen3)
import CosyVoiceTTS // Text-to-speech (streaming)
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

// Default: 0.6B model
let model = try await Qwen3ASRModel.fromPretrained()

// Or use the larger 1.7B model for better accuracy
let model = try await Qwen3ASRModel.fromPretrained(
    modelId: "mlx-community/Qwen3-ASR-1.7B-8bit"
)

// Audio can be any sample rate — automatically resampled to 16kHz internally
let transcription = model.transcribe(audio: audioSamples, sampleRate: 16000)
print(transcription)
```

### ASR CLI

```bash
swift build -c release

# Default (0.6B)
.build/release/qwen3-asr-cli audio.wav

# Use 1.7B model
.build/release/qwen3-asr-cli --model 1.7B audio.wav
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

### Custom Voice / Speaker Selection

The **CustomVoice** model variant supports 9 built-in speaker voices. Load it by passing the CustomVoice model ID:

```swift
import Qwen3TTS

// Load the CustomVoice model (downloads ~1.7 GB on first run)
let model = try await Qwen3TTSModel.fromPretrained(
    modelId: TTSModelVariant.customVoice.rawValue
)

// Synthesize with a specific speaker
let audio = model.synthesize(text: "Hello world", language: "english", speaker: "vivian")

// List available speakers
print(model.availableSpeakers)  // ["aiden", "dylan", "eric", ...]
```

CLI:

```bash
# Use CustomVoice model with a speaker
.build/release/qwen3-tts-cli "Hello world" --model customVoice --speaker vivian --output vivian.wav

# List available speakers
.build/release/qwen3-tts-cli --model customVoice --list-speakers

```

### Batch Synthesis

Synthesize multiple texts in a single batched forward pass for higher throughput:

```swift
let texts = ["Good morning everyone.", "The weather is nice today.", "Please open the window."]
let audioList = model.synthesizeBatch(texts: texts, language: "english", maxBatchSize: 4)
// audioList[i] is 24kHz mono float samples for texts[i]
for (i, audio) in audioList.enumerated() {
    try WAVWriter.write(samples: audio, sampleRate: 24000, to: URL(fileURLWithPath: "output_\(i).wav"))
}
```

#### Batch CLI

```bash
# Create a file with one text per line
echo "Hello world.\nGoodbye world." > texts.txt
.build/release/qwen3-tts-cli --batch-file texts.txt --output output.wav --batch-size 4
# Produces output_0.wav, output_1.wav, ...
```

> Batch mode amortizes model weight loads across items. Expect ~1.5-2.5x throughput improvement for B=4 on Apple Silicon. Best results when texts produce similar-length audio.

### Sampling Options

```swift
let config = SamplingConfig(temperature: 0.9, topK: 50, repetitionPenalty: 1.05)
let audio = model.synthesize(text: "Hello", language: "english", sampling: config)
```

## CosyVoice TTS Usage

### Basic Synthesis

```swift
import CosyVoiceTTS
import Qwen3Common  // for WAVWriter

let model = try await CosyVoiceTTSModel.fromPretrained()
// Downloads ~1.9 GB on first run (LLM + DiT + HiFi-GAN weights)

let audio = model.synthesize(text: "Hello, how are you today?", language: "english")
// Output is 24kHz mono float samples
try WAVWriter.write(samples: audio, sampleRate: 24000, to: outputURL)
```

### Streaming Synthesis

```swift
// Streaming: receive audio chunks as they're generated (~150ms to first chunk)
for try await chunk in model.synthesizeStream(text: "Hello, how are you today?", language: "english") {
    // chunk.audio: [Float], chunk.sampleRate: Int
    playAudio(chunk.audio)  // play immediately
}
```

### CosyVoice TTS CLI

```bash
swift build -c release

# Basic synthesis
.build/release/cosyvoice-tts-cli "Hello world" --language english --output output.wav

# Streaming synthesis
.build/release/cosyvoice-tts-cli "Hello world" --language english --stream --output output.wav
```

## Latency (M2 Max, 64 GB)

### ASR

| Model | Framework | RTF | 10s audio processed in |
|-------|-----------|-----|------------------------|
| Qwen3-ASR-0.6B (4-bit) | MLX Swift | ~0.06 | ~0.6s |
| Qwen3-ASR-1.7B (8-bit) | MLX Swift | ~0.11 | ~1.1s |
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

See [ASR Inference](docs/asr-inference.md), [ASR Model](docs/asr-model.md), [Qwen3-TTS Inference](docs/qwen3-tts-inference.md), [TTS Model](docs/tts-model.md), [CosyVoice TTS](docs/cosyvoice-tts.md) for detailed architecture docs.

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
swift test --filter "Qwen3TTSConfigTests|SamplingTests|CosyVoiceTTSConfigTests"
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

## Supported Languages

| Model | Languages |
|-------|-----------|
| Qwen3-ASR | 52 languages (CN, EN, Cantonese, DE, FR, ES, JA, KO, RU, + 22 Chinese dialects, ...) |
| Qwen3-TTS | EN, CN, DE, JA, ES, FR, KO, RU, IT, PT (+ Beijing/Sichuan dialects via CustomVoice) |
| CosyVoice TTS | CN, EN, JA, KO, DE, ES, FR, IT, RU |

## License

Apache 2.0 (same as original Qwen3 models)

