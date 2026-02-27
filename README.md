# Speech Swift

AI speech models for Apple Silicon, powered by [MLX Swift](https://github.com/ml-explore/mlx-swift).

- **Qwen3-ASR** — Speech-to-text (automatic speech recognition)
- **Qwen3-ForcedAligner** — Word-level timestamp alignment (audio + text → timestamps)
- **Qwen3-TTS** — Text-to-speech synthesis (highest quality, custom speakers)
- **CosyVoice TTS** — Text-to-speech with streaming (9 languages, DiT flow matching)
- **PersonaPlex** — Full-duplex speech-to-speech (7B, audio in → audio out)
- **Silero VAD** — Streaming voice activity detection (32ms chunks, ~309K params)
- **Pyannote VAD** — Offline voice activity detection (10s windows, multi-speaker overlap)
- **Speaker Diarization** — Who spoke when (pyannote segmentation + WeSpeaker embedding + clustering)

Papers: [Qwen3-ASR](https://arxiv.org/abs/2601.21337), [Qwen3-TTS](https://arxiv.org/abs/2601.15621), [CosyVoice 3](https://arxiv.org/abs/2505.17589), [PersonaPlex](https://arxiv.org/abs/2602.06053), [Mimi](https://arxiv.org/abs/2410.00037) (audio codec)

## News

- **26 Feb 2026** — [Speaker Diarization and Voice Activity Detection on Apple Silicon — Native Swift with MLX](https://blog.ivan.digital/speaker-diarization-and-voice-activity-detection-on-apple-silicon-native-swift-with-mlx-92ea0c9aca0f)
- **23 Feb 2026** — [NVIDIA PersonaPlex 7B on Apple Silicon — Full-Duplex Speech-to-Speech in Native Swift with MLX](https://blog.ivan.digital/nvidia-personaplex-7b-on-apple-silicon-full-duplex-speech-to-speech-in-native-swift-with-mlx-0aa5276f2e23)
- **12 Feb 2026** — [Qwen3-ASR Swift: On-Device ASR + TTS for Apple Silicon — Architecture and Benchmarks](https://blog.ivan.digital/qwen3-asr-swift-on-device-asr-tts-for-apple-silicon-architecture-and-benchmarks-27cbf1e4463f)

## Models

| Model | Task | Streaming | Languages | Download Size |
|-------|------|-----------|-----------|--------------|
| Qwen3-ASR-0.6B (4-bit) | Speech → Text | No | 52 languages | ~400 MB |
| Qwen3-ASR-1.7B (8-bit) | Speech → Text | No | 52 languages | ~2.5 GB |
| Qwen3-ForcedAligner-0.6B (4-bit) | Audio + Text → Timestamps | No | Multi | ~979 MB |
| Qwen3-TTS-0.6B Base (4-bit) | Text → Speech | Yes (~120ms) | 10 languages | ~1.7 GB |
| Qwen3-TTS-0.6B CustomVoice (4-bit) | Text → Speech | Yes (~120ms) | 10 languages | ~1.7 GB |
| CosyVoice3-0.5B (4-bit) | Text → Speech | Yes (~150ms) | 9 languages | ~1.9 GB |
| PersonaPlex-7B (4-bit) | Speech → Speech | Yes (~2s chunks) | EN | ~5.3 GB |
| Silero-VAD-v5 | Voice Activity Detection | Yes (32ms chunks) | Language-agnostic | ~1.2 MB |
| Pyannote-Segmentation-3.0 | VAD + Speaker Segmentation | No (10s windows) | Language-agnostic | ~5.7 MB |
| WeSpeaker-ResNet34-LM | Speaker Embedding (256-dim) | No | Language-agnostic | ~25 MB |

### When to Use Which TTS

- **Qwen3-TTS**: Best quality, streaming (~120ms), 9 built-in speakers, 10 languages, batch synthesis
- **CosyVoice TTS**: Streaming (~150ms), 9 languages, DiT flow matching + HiFi-GAN vocoder
- **PersonaPlex**: Full-duplex speech-to-speech (audio in → audio out), streaming (~2s chunks), 18 voice presets, based on Moshi architecture

## Installation

### Homebrew

```bash
brew tap ivan-digital/speech https://github.com/ivan-digital/qwen3-asr-swift
brew install speech
```

Then use:

```bash
audio transcribe recording.wav
audio speak "Hello world"
audio speak "Hallo Welt" --engine cosyvoice --language german
audio respond --input question.wav
```

### Swift Package Manager

Add to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/ivan-digital/qwen3-asr-swift", branch: "main")
]
```

Import the module you need:

```swift
import Qwen3ASR     // Speech recognition
import Qwen3TTS     // Text-to-speech (Qwen3)
import CosyVoiceTTS  // Text-to-speech (streaming)
import PersonaPlex   // Speech-to-speech (full-duplex)
import SpeechVAD     // Voice activity detection (pyannote + Silero)
import AudioCommon   // Shared utilities
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
.build/release/audio transcribe audio.wav

# Use 1.7B model
.build/release/audio transcribe audio.wav --model 1.7B
```

## Forced Alignment

### Word-Level Timestamps

```swift
import Qwen3ASR

let aligner = try await Qwen3ForcedAligner.fromPretrained()
// Downloads ~979 MB on first run

let aligned = aligner.align(
    audio: audioSamples,
    text: "Can you guarantee that the replacement part will be shipped tomorrow?",
    sampleRate: 24000
)

for word in aligned {
    print("[\(String(format: "%.2f", word.startTime))s - \(String(format: "%.2f", word.endTime))s] \(word.text)")
}
```

### Forced Alignment CLI

```bash
swift build -c release

# Align with provided text
.build/release/audio align audio.wav --text "Hello world"

# Transcribe first, then align
.build/release/audio align audio.wav
```

Output:
```
[0.12s - 0.45s] Can
[0.45s - 0.72s] you
[0.72s - 1.20s] guarantee
...
```

Non-autoregressive — single forward pass, no sampling loop. See [Forced Aligner](docs/forced-aligner.md) for architecture details.

## TTS Usage

### Basic Synthesis

```swift
import Qwen3TTS
import AudioCommon  // for WAVWriter

let model = try await Qwen3TTSModel.fromPretrained()
// Downloads ~1.7 GB on first run (model + codec weights)
let audio = model.synthesize(text: "Hello world", language: "english")
// Output is 24kHz mono float samples
try WAVWriter.write(samples: audio, sampleRate: 24000, to: outputURL)
```

### TTS CLI

```bash
swift build -c release
.build/release/audio speak "Hello world" --output output.wav --language english
```

### Custom Voice / Speaker Selection

The **CustomVoice** model variant supports 9 built-in speaker voices and natural language instructions for tone/style control. Load it by passing the CustomVoice model ID:

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
.build/release/audio speak "Hello world" --model customVoice --speaker vivian --output vivian.wav

# List available speakers
.build/release/audio speak --model customVoice --list-speakers
```

### Tone / Style Instructions (CustomVoice only)

The CustomVoice model accepts a natural language `instruct` parameter to control speaking style, tone, emotion, and pacing. The instruction is prepended to the model input in ChatML format.

```swift
// Cheerful tone
let audio = model.synthesize(
    text: "Welcome to our store!",
    language: "english",
    speaker: "ryan",
    instruct: "Speak in a cheerful, upbeat tone"
)

// Slow and serious
let audio = model.synthesize(
    text: "We regret to inform you...",
    language: "english",
    speaker: "aiden",
    instruct: "Read this slowly and solemnly"
)

// Whispering
let audio = model.synthesize(
    text: "Can you keep a secret?",
    language: "english",
    speaker: "vivian",
    instruct: "Whisper this softly"
)
```

CLI:

```bash
# With style instruction
.build/release/audio speak "Good morning!" --model customVoice --speaker ryan \
    --instruct "Speak in a cheerful, upbeat tone" --output cheerful.wav

# Default instruct ("Speak naturally.") is applied automatically when using CustomVoice
.build/release/audio speak "Hello world" --model customVoice --speaker ryan --output natural.wav
```

When no `--instruct` is provided with the CustomVoice model, `"Speak naturally."` is applied automatically to prevent rambling output. The Base model does not support instruct.

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
.build/release/audio speak --batch-file texts.txt --output output.wav --batch-size 4
# Produces output_0.wav, output_1.wav, ...
```

> Batch mode amortizes model weight loads across items. Expect ~1.5-2.5x throughput improvement for B=4 on Apple Silicon. Best results when texts produce similar-length audio.

### Sampling Options

```swift
let config = SamplingConfig(temperature: 0.9, topK: 50, repetitionPenalty: 1.05)
let audio = model.synthesize(text: "Hello", language: "english", sampling: config)
```

### Streaming Synthesis

Emit audio chunks incrementally for low first-packet latency:

```swift
let stream = model.synthesizeStream(
    text: "Hello, this is streaming synthesis.",
    language: "english",
    streaming: .lowLatency  // ~120ms to first audio chunk
)

for try await chunk in stream {
    // chunk.samples: [Float] PCM @ 24kHz
    // chunk.isFinal: true on last chunk
    playAudio(chunk.samples)
}
```

CLI:

```bash
# Default streaming (3-frame first chunk, ~225ms latency)
.build/release/audio speak "Hello world" --stream

# Low-latency (1-frame first chunk, ~120ms latency)
.build/release/audio speak "Hello world" --stream --first-chunk-frames 1
```

## PersonaPlex Usage

### Speech-to-Speech

```swift
import PersonaPlex
import AudioCommon  // for WAVWriter, AudioFileLoader

let model = try await PersonaPlexModel.fromPretrained()
// Downloads ~5.5 GB on first run (temporal 4-bit + depformer + Mimi codec + voice presets)

let audio = try AudioFileLoader.load(url: inputURL, targetSampleRate: 24000)
let response = model.respond(userAudio: audio, voice: .NATM0)
// Output is 24kHz mono float samples
try WAVWriter.write(samples: response, sampleRate: 24000, to: outputURL)
```

### Streaming Speech-to-Speech

```swift
// Receive audio chunks as they're generated (~2s per chunk)
let stream = model.respondStream(userAudio: audio, voice: .NATM0)
for try await chunk in stream {
    playAudio(chunk.samples)  // play immediately, 24kHz mono
    if chunk.isFinal { break }
}
```

### Voice Selection

18 voice presets available:
- **Natural Female**: NATF0, NATF1, NATF2, NATF3
- **Natural Male**: NATM0, NATM1, NATM2, NATM3
- **Variety Female**: VARF0, VARF1, VARF2, VARF3, VARF4
- **Variety Male**: VARM0, VARM1, VARM2, VARM3, VARM4

### System Prompts

The system prompt steers the model's conversational behavior. The `focused` default keeps responses on-topic:

```swift
// Use a preset
let response = model.respond(
    userAudio: audio,
    voice: .NATM0,
    systemPromptTokens: SystemPromptPreset.customerService.tokens
)
```

Available presets: `focused` (default), `assistant`, `customerService`, `teacher`.

### PersonaPlex CLI

```bash
swift build -c release

# Basic speech-to-speech
.build/release/audio respond --input question.wav --output response.wav

# Choose a voice and system prompt preset
.build/release/audio respond --input question.wav --voice NATF1 --system-prompt focused --output response.wav

# List available voices and prompts
.build/release/audio respond --list-voices
.build/release/audio respond --list-prompts
```

## CosyVoice TTS Usage

### Basic Synthesis

```swift
import CosyVoiceTTS
import AudioCommon  // for WAVWriter

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
.build/release/audio speak "Hello world" --engine cosyvoice --language english --output output.wav

# Streaming synthesis
.build/release/audio speak "Hello world" --engine cosyvoice --language english --stream --output output.wav
```

## Voice Activity Detection

### Streaming VAD (Silero)

Silero VAD v5 processes 32ms audio chunks with sub-millisecond latency — ideal for real-time speech detection from microphones or streams.

```swift
import SpeechVAD

let vad = try await SileroVADModel.fromPretrained()
// Downloads ~1.2 MB on first run

// Streaming: process 512-sample chunks (32ms @ 16kHz)
let prob = vad.processChunk(samples)  // → 0.0...1.0
vad.resetState()  // call between different audio streams

// Or detect all segments at once
let segments = vad.detectSpeech(audio: audioSamples, sampleRate: 16000)
for seg in segments {
    print("Speech: \(seg.startTime)s - \(seg.endTime)s")
}
```

### Event-Driven Streaming

```swift
let processor = StreamingVADProcessor(model: vad)

// Feed audio of any length — events emitted as speech is confirmed
let events = processor.process(samples: audioBuffer)
for event in events {
    switch event {
    case .speechStarted(let time):
        print("Speech started at \(time)s")
    case .speechEnded(let segment):
        print("Speech: \(segment.startTime)s - \(segment.endTime)s")
    }
}

// Flush at end of stream
let final = processor.flush()
```

### VAD CLI

```bash
swift build -c release

# Streaming Silero VAD (32ms chunks)
.build/release/audio vad-stream audio.wav

# With custom thresholds
.build/release/audio vad-stream audio.wav --onset 0.6 --offset 0.4

# JSON output
.build/release/audio vad-stream audio.wav --json

# Batch pyannote VAD (10s sliding windows)
.build/release/audio vad audio.wav
```

## Speaker Diarization

### Diarization Pipeline

```swift
import SpeechVAD

let pipeline = try await DiarizationPipeline.fromPretrained()
// Downloads pyannote (~5.7 MB) + WeSpeaker (~25 MB) on first run

let result = pipeline.diarize(audio: samples, sampleRate: 16000)
for seg in result.segments {
    print("Speaker \(seg.speakerId): [\(seg.startTime)s - \(seg.endTime)s]")
}
print("\(result.numSpeakers) speakers detected")
```

### Speaker Embedding

```swift
let model = try await WeSpeakerModel.fromPretrained()
let embedding = model.embed(audio: samples, sampleRate: 16000)
// embedding: [Float] of length 256, L2-normalized

// Compare speakers
let similarity = WeSpeakerModel.cosineSimilarity(embeddingA, embeddingB)
```

### Speaker Extraction

Extract only a specific speaker's segments using a reference recording:

```swift
let pipeline = try await DiarizationPipeline.fromPretrained()
let targetEmb = pipeline.embeddingModel.embed(audio: enrollmentAudio, sampleRate: 16000)
let segments = pipeline.extractSpeaker(
    audio: meetingAudio, sampleRate: 16000,
    targetEmbedding: targetEmb
)
```

### Diarization CLI

```bash
swift build -c release

# Speaker diarization
.build/release/audio diarize meeting.wav

# With options
.build/release/audio diarize meeting.wav --threshold 0.6 --max-speakers 3 --json

# Extract a specific speaker
.build/release/audio diarize meeting.wav --target-speaker enrollment.wav

# Speaker embedding
.build/release/audio embed-speaker enrollment.wav --json
```

See [Speaker Diarization](docs/speaker-diarization.md) for architecture details.

## Latency (M2 Max, 64 GB)

### ASR

| Model | Framework | RTF | 10s audio processed in |
|-------|-----------|-----|------------------------|
| Qwen3-ASR-0.6B (4-bit) | MLX Swift | ~0.06 | ~0.6s |
| Qwen3-ASR-1.7B (8-bit) | MLX Swift | ~0.11 | ~1.1s |
| Whisper-large-v3 | whisper.cpp (Q5_0) | ~0.10 | ~1.0s |
| Whisper-small | whisper.cpp (Q5_0) | ~0.04 | ~0.4s |

### Forced Alignment

| Model | Framework | 20s audio | RTF |
|-------|-----------|-----------|-----|
| Qwen3-ForcedAligner-0.6B (4-bit) | MLX Swift (debug) | ~365ms | ~0.018 |

> Single non-autoregressive forward pass — no sampling loop. Audio encoder dominates (~328ms), decoder single-pass is ~37ms. **55x faster than real-time.**

### TTS

| Model | Framework | Short (1s) | Medium (3s) | Long (6s) | Streaming First-Packet |
|-------|-----------|-----------|-------------|------------|----------------------|
| Qwen3-TTS-0.6B (4-bit) | MLX Swift (release) | 1.6s (RTF 1.2) | 2.3s (RTF 0.7) | 3.9s (RTF 0.7) | ~120ms (1-frame) |
| Apple `AVSpeechSynthesizer` | AVFoundation | 0.08s | 0.08s | 0.17s (RTF 0.02) | N/A |

> Qwen3-TTS generates natural, expressive speech with prosody and emotion, running **faster than real-time** (RTF < 1.0). Streaming synthesis delivers the first audio chunk in ~120ms. Apple's built-in TTS is ~35x faster but produces robotic, monotone speech.

### PersonaPlex (Speech-to-Speech)

| Model | Framework | ms/step | RTF | Notes |
|-------|-----------|---------|-----|-------|
| PersonaPlex-7B (4-bit) | MLX Swift (release) | ~68ms | ~0.87 | 20s input → 36s output in ~31s |

> PersonaPlex runs at ~68ms/step — well under the 80ms real-time threshold at 12.5 Hz, achieving **faster-than-real-time** inference (RTF < 1.0). Both temporal transformer and depformer are 4-bit quantized.

### VAD

| Model | Framework | Chunk Size | 20s audio processed in | Throughput |
|-------|-----------|-----------|------------------------|------------|
| Silero-VAD-v5 | MLX Swift (release) | 32ms (512 samples) | ~0.87s | 23x real-time |

> Silero VAD processes each 32ms chunk in ~40μs after model load. The streaming `StreamingVADProcessor` applies hysteresis thresholding with configurable onset/offset thresholds and minimum duration filtering.

RTF = Real-Time Factor (lower is better, < 1.0 = faster than real-time).

## Architecture

See [ASR Inference](docs/asr-inference.md), [ASR Model](docs/asr-model.md), [Forced Aligner](docs/forced-aligner.md), [Qwen3-TTS Inference](docs/qwen3-tts-inference.md), [TTS Model](docs/tts-model.md), [CosyVoice TTS](docs/cosyvoice-tts.md), [PersonaPlex](docs/personaplex.md), [Silero VAD](docs/silero-vad.md), [Speaker Diarization](docs/speaker-diarization.md), [Shared Protocols](docs/shared-protocols.md) for detailed architecture docs.

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

Unit tests (config, sampling, text preprocessing, timestamp correction) run without model downloads:

```bash
swift test --filter "Qwen3TTSConfigTests|SamplingTests|CosyVoiceTTSConfigTests|PersonaPlexTests|ForcedAlignerTests/testText|ForcedAlignerTests/testTimestamp|ForcedAlignerTests/testLIS|SileroVADTests/testSilero|SileroVADTests/testReflection|SileroVADTests/testProcess|SileroVADTests/testReset|SileroVADTests/testDetect|SileroVADTests/testStreaming|SileroVADTests/testVADEvent"
```

Integration tests require model weights (downloaded automatically on first run):

```bash
# TTS round-trip: synthesize text, save WAV, transcribe back with ASR
swift test --filter TTSASRRoundTripTests

# ASR only: transcribe test audio
swift test --filter Qwen3ASRIntegrationTests

# Forced Aligner E2E: word-level timestamps (~979 MB download)
swift test --filter ForcedAlignerTests/testForcedAlignerE2E

# PersonaPlex E2E: speech-to-speech pipeline (~5.5 GB download)
PERSONAPLEX_E2E=1 swift test --filter PersonaPlexE2ETests
```

> **Note:** MLX Metal library must be built before running tests that use MLX operations.
> See [MLX Metal Library](#mlx-metal-library) for instructions.

## Supported Languages

| Model | Languages |
|-------|-----------|
| Qwen3-ASR | 52 languages (CN, EN, Cantonese, DE, FR, ES, JA, KO, RU, + 22 Chinese dialects, ...) |
| Qwen3-TTS | EN, CN, DE, JA, ES, FR, KO, RU, IT, PT (+ Beijing/Sichuan dialects via CustomVoice) |
| CosyVoice TTS | CN, EN, JA, KO, DE, ES, FR, IT, RU |
| PersonaPlex | EN |

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=ivan-digital/qwen3-asr-swift&type=date&legend=top-left)](https://www.star-history.com/#ivan-digital/qwen3-asr-swift&type=date&legend=top-left)

## License

Apache 2.0 (same as original Qwen3 models)

