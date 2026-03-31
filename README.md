# Speech Swift

AI speech models for Apple Silicon, powered by MLX Swift and CoreML.

📖 Read in: [English](README.md) · [中文](README_zh.md) · [日本語](README_ja.md) · [한국어](README_ko.md) · [Español](README_es.md) · [Deutsch](README_de.md) · [Français](README_fr.md) · [हिन्दी](README_hi.md) · [Português](README_pt.md) · [Русский](README_ru.md)

On-device speech recognition, synthesis, and understanding for Mac and iOS. Runs locally on Apple Silicon — no cloud, no API keys, no data leaves your device.

[Install via Homebrew](#homebrew) or add as a Swift Package dependency.

**[Documentation](https://soniqo.audio)** · **[HuggingFace Models](https://huggingface.co/aufklarer)** · **[Blog](https://blog.ivan.digital)**

- **Qwen3-ASR** — Speech-to-text / speech recognition (automatic speech recognition, 52 languages)
- **Parakeet TDT** — Speech-to-text via CoreML (Neural Engine, NVIDIA FastConformer + TDT decoder, 25 languages)
- **Qwen3-ForcedAligner** — Word-level timestamp alignment (audio + text → timestamps)
- **Qwen3-TTS** — Text-to-speech synthesis (highest quality, streaming, custom speakers, 10 languages)
- **CosyVoice TTS** — Text-to-speech with streaming, voice cloning, multi-speaker dialogue, and emotion tags (9 languages, DiT flow matching, CAM++ speaker encoder)
- **Kokoro TTS** — On-device text-to-speech (82M params, CoreML/Neural Engine, 50 voices, iOS-ready, 10 languages)
- **Qwen3.5-Chat** — On-device LLM chat (0.8B, MLX INT4 + CoreML INT8, DeltaNet hybrid, streaming tokens)
- **PersonaPlex** — Full-duplex speech-to-speech conversation (7B, audio in → audio out, 18 voice presets)
- **DeepFilterNet3** — Speech enhancement / noise suppression (2.1M params, real-time 48kHz)
- **FireRedVAD** — Offline voice activity detection (DFSMN, CoreML, 100+ languages, 97.6% F1)
- **Silero VAD** — Streaming voice activity detection (32ms chunks, sub-millisecond latency)
- **Pyannote VAD** — Offline voice activity detection (10s windows, multi-speaker overlap)
- **Speaker Diarization** — Who spoke when (Pyannote segmentation + activity-based speaker chaining, or end-to-end Sortformer on Neural Engine)
- **Speaker Embeddings** — Speaker verification and identification (WeSpeaker ResNet34, 256-dim vectors)

Papers: [Qwen3-ASR](https://arxiv.org/abs/2601.21337) (Alibaba), [Qwen3-TTS](https://arxiv.org/abs/2601.15621) (Alibaba), [Qwen3](https://arxiv.org/abs/2505.09388) (Alibaba), [Parakeet TDT](https://arxiv.org/abs/2304.06795) (NVIDIA), [CosyVoice 3](https://arxiv.org/abs/2505.17589) (Alibaba), [Kokoro](https://arxiv.org/abs/2301.01695) (StyleTTS 2), [PersonaPlex](https://arxiv.org/abs/2602.06053) (NVIDIA), [Mimi](https://arxiv.org/abs/2410.00037) (Kyutai), [Sortformer](https://arxiv.org/abs/2409.06656) (NVIDIA)

## Roadmap

See [Roadmap discussion](https://github.com/soniqo/speech-swift/discussions/81) for what's planned — comments and suggestions welcome!

## News

- **20 Mar 2026** — [We Beat Whisper Large v3 with a 600M Model Running Entirely on Your Mac](https://blog.ivan.digital/we-beat-whisper-large-v3-with-a-600m-model-running-entirely-on-your-mac-20e6ce191174)
- **26 Feb 2026** — [Speaker Diarization and Voice Activity Detection on Apple Silicon — Native Swift with MLX](https://blog.ivan.digital/speaker-diarization-and-voice-activity-detection-on-apple-silicon-native-swift-with-mlx-92ea0c9aca0f)
- **23 Feb 2026** — [NVIDIA PersonaPlex 7B on Apple Silicon — Full-Duplex Speech-to-Speech in Native Swift with MLX](https://blog.ivan.digital/nvidia-personaplex-7b-on-apple-silicon-full-duplex-speech-to-speech-in-native-swift-with-mlx-0aa5276f2e23)
- **12 Feb 2026** — [Qwen3-ASR Swift: On-Device ASR + TTS for Apple Silicon — Architecture and Benchmarks](https://blog.ivan.digital/qwen3-asr-swift-on-device-asr-tts-for-apple-silicon-architecture-and-benchmarks-27cbf1e4463f)

## Models

| Model | Task | Streaming | Languages | Sizes |
|-------|------|-----------|-----------|-------|
| Qwen3-ASR-0.6B | Speech → Text | No | 52 languages | [4-bit](https://huggingface.co/aufklarer/Qwen3-ASR-0.6B-MLX-4bit) 680 MB · [8-bit](https://huggingface.co/aufklarer/Qwen3-ASR-0.6B-MLX-8bit) 1.0 GB · [CoreML](https://huggingface.co/aufklarer/Qwen3-ASR-CoreML) 180 MB |
| Qwen3-ASR-1.7B | Speech → Text | No | 52 languages | [4-bit](https://huggingface.co/aufklarer/Qwen3-ASR-1.7B-MLX-4bit) 2.1 GB · [8-bit](https://huggingface.co/aufklarer/Qwen3-ASR-1.7B-MLX-8bit) 3.2 GB |
| Parakeet-TDT-0.6B | Speech → Text | No | 25 European languages | [CoreML INT4](https://huggingface.co/aufklarer/Parakeet-TDT-v3-CoreML-INT4) 315 MB · [CoreML INT8](https://huggingface.co/aufklarer/Parakeet-TDT-v3-CoreML-INT8) 500 MB |
| Qwen3-ForcedAligner-0.6B | Audio + Text → Timestamps | No | Multi | [4-bit](https://huggingface.co/aufklarer/Qwen3-ForcedAligner-0.6B-4bit) 979 MB · [8-bit](https://huggingface.co/aufklarer/Qwen3-ForcedAligner-0.6B-8bit) 1.4 GB · [CoreML INT4](https://huggingface.co/aufklarer/Qwen3-ForcedAligner-0.6B-CoreML-INT4) 630 MB · [CoreML INT8](https://huggingface.co/aufklarer/Qwen3-ForcedAligner-0.6B-CoreML-INT8) 1.0 GB |
| Qwen3-TTS-0.6B Base | Text → Speech | Yes (~120ms) | 10 languages | [4-bit](https://huggingface.co/aufklarer/Qwen3-TTS-12Hz-0.6B-Base-MLX-4bit) 1.7 GB · [8-bit](https://huggingface.co/aufklarer/Qwen3-TTS-12Hz-0.6B-Base-MLX-8bit) 2.4 GB |
| Qwen3-TTS-0.6B CustomVoice | Text → Speech | Yes (~120ms) | 10 languages | [4-bit](https://huggingface.co/aufklarer/Qwen3-TTS-12Hz-0.6B-CustomVoice-MLX-4bit) 1.7 GB |
| Qwen3-TTS-1.7B Base | Text → Speech | Yes (~120ms) | 10 languages | [4-bit](https://huggingface.co/aufklarer/Qwen3-TTS-12Hz-1.7B-Base-MLX-4bit) 3.2 GB · [8-bit](https://huggingface.co/aufklarer/Qwen3-TTS-12Hz-1.7B-Base-MLX-8bit) 4.8 GB |
| CosyVoice3-0.5B | Text → Speech | Yes (~150ms) | 9 languages | [4-bit](https://huggingface.co/aufklarer/CosyVoice3-0.5B-MLX-4bit) 1.2 GB |
| Kokoro-82M | Text → Speech | No | 10 languages | [CoreML](https://huggingface.co/aufklarer/Kokoro-82M-CoreML) ~170 MB |
| Qwen3.5-0.8B Chat | Text → Text (LLM) | Yes (streaming) | Multi | [MLX INT4](https://huggingface.co/aufklarer/Qwen3.5-0.8B-Chat-MLX) 418 MB · [CoreML INT8](https://huggingface.co/aufklarer/Qwen3.5-0.8B-Chat-CoreML) 981 MB |
| PersonaPlex-7B | Speech → Speech | Yes (~2s chunks) | EN | [4-bit](https://huggingface.co/aufklarer/PersonaPlex-7B-MLX-4bit) 4.9 GB · [8-bit](https://huggingface.co/aufklarer/PersonaPlex-7B-MLX-8bit) 9.1 GB |
| FireRedVAD | Voice Activity Detection | No (offline) | 100+ languages | [CoreML](https://huggingface.co/aufklarer/FireRedVAD-CoreML) ~1.2 MB |
| Silero-VAD-v5 | Voice Activity Detection | Yes (32ms chunks) | Language-agnostic | [MLX](https://huggingface.co/aufklarer/Silero-VAD-v5-MLX) · [CoreML](https://huggingface.co/aufklarer/Silero-VAD-v5-CoreML) ~1.2 MB |
| Pyannote-Segmentation-3.0 | VAD + Speaker Segmentation | No (10s windows) | Language-agnostic | [MLX](https://huggingface.co/aufklarer/Pyannote-Segmentation-MLX) ~5.7 MB |
| DeepFilterNet3 | Speech Enhancement | Yes (10ms frames) | Language-agnostic | [CoreML FP16](https://huggingface.co/aufklarer/DeepFilterNet3-CoreML) ~4.2 MB |
| WeSpeaker-ResNet34-LM | Speaker Embedding (256-dim) | No | Language-agnostic | [MLX](https://huggingface.co/aufklarer/WeSpeaker-ResNet34-LM-MLX) · [CoreML](https://huggingface.co/aufklarer/WeSpeaker-ResNet34-LM-CoreML) ~25 MB |
| CAM++ | Speaker Embedding (192-dim) | No | Language-agnostic | [CoreML](https://huggingface.co/aufklarer/CamPlusPlus-Speaker-CoreML) ~14 MB |
| Sortformer | Speaker Diarization (end-to-end) | Yes (chunked) | Language-agnostic | [CoreML](https://huggingface.co/aufklarer/Sortformer-Diarization-CoreML) ~240 MB |

### Memory Requirements

Weight memory is the GPU (MLX) or ANE (CoreML) memory consumed by model parameters. Peak inference includes KV caches, activations, and intermediate tensors.

| Model | Weight Memory | Peak Inference |
|-------|--------------|----------------|
| Qwen3-ASR-0.6B (4-bit, MLX) | 675 MB | ~2.2 GB |
| Qwen3-ASR-0.6B (INT8, CoreML) | 180 MB | ~400 MB |
| Qwen3-ASR-1.7B (8-bit, MLX) | 2,349 MB | ~4 GB |
| Parakeet-TDT-0.6B (CoreML) | 315 MB | ~400 MB |
| Qwen3-ForcedAligner-0.6B (4-bit, MLX) | 933 MB | ~1.5 GB |
| Qwen3-TTS-1.7B (4-bit, MLX) | 2,300 MB | ~4–6 GB |
| Qwen3-TTS-0.6B (4-bit, MLX) | 977 MB | ~2 GB |
| CosyVoice3-0.5B (4-bit, MLX) | 732 MB | ~2.5 GB |
| Kokoro-82M (CoreML) | 170 MB | ~200 MB |
| Qwen3.5-Chat-0.8B (INT4, MLX) | 418 MB | ~700 MB |
| Qwen3.5-Chat-0.8B (INT8, CoreML) | 981 MB | ~1.2 GB |
| PersonaPlex-7B (8-bit, MLX) | 9,100 MB | ~11 GB |
| PersonaPlex-7B (4-bit, MLX) | 4,900 MB | ~6.5 GB |
| Silero-VAD-v5 (MLX) | 1.2 MB | ~5 MB |
| Silero-VAD-v5 (CoreML) | 0.7 MB | ~3 MB |
| Pyannote-Segmentation-3.0 (MLX) | 6 MB | ~20 MB |
| DeepFilterNet3 (CoreML FP16) | 4.2 MB | ~10 MB |
| WeSpeaker-ResNet34-LM (MLX) | 25 MB | ~50 MB |

### When to Use Which TTS

- **Qwen3-TTS**: Best quality, streaming (~120ms), 9 built-in speakers, 10 languages, batch synthesis
- **CosyVoice TTS**: Streaming (~150ms), 9 languages, voice cloning (CAM++ speaker encoder), multi-speaker dialogue (`[S1] ... [S2] ...`), inline emotion/style tags (`(happy)`, `(whispers)`), DiT flow matching + HiFi-GAN vocoder
- **Kokoro TTS**: Lightweight iOS-ready TTS (82M params), CoreML/Neural Engine, 54 voices, 10 languages, non-autoregressive 3-stage pipeline
- **PersonaPlex**: Full-duplex speech-to-speech (audio in → audio out), streaming (~2s chunks), 18 voice presets, based on Moshi architecture

## Installation

### Homebrew

Requires native ARM Homebrew (`/opt/homebrew`). Rosetta/x86_64 Homebrew is not supported.

```bash
brew tap soniqo/speech https://github.com/soniqo/speech-swift
brew install speech
```

Then use:

```bash
audio transcribe recording.wav
audio speak "Hello world"
audio speak "Hallo Welt" --engine cosyvoice --language german
audio respond --input question.wav --transcript
```

> For interactive voice conversation with microphone input, see **[PersonaPlexDemo](Examples/PersonaPlexDemo/)**.

### Swift Package Manager

Add to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/soniqo/speech-swift", branch: "main")
]
```

Import the module you need:

```swift
import Qwen3ASR      // Speech recognition (MLX)
import ParakeetASR   // Speech recognition (CoreML)
import Qwen3TTS      // Text-to-speech (Qwen3)
import CosyVoiceTTS  // Text-to-speech (streaming)
import KokoroTTS     // Text-to-speech (CoreML, iOS-ready)
import Qwen3Chat     // On-device LLM chat (CoreML)
import PersonaPlex   // Speech-to-speech (full-duplex)
import SpeechVAD          // Voice activity detection (pyannote + Silero)
import SpeechEnhancement  // Noise suppression (DeepFilterNet3)
import AudioCommon        // Shared utilities
```

### Requirements

- Swift 5.9+
- macOS 14+ or iOS 17+
- Apple Silicon (M1/M2/M3/M4)
- Xcode 15+ (with Metal Toolchain — run `xcodebuild -downloadComponent MetalToolchain` if missing)

### Build from Source

```bash
git clone https://github.com/soniqo/speech-swift
cd speech-swift
make build
```

This compiles the Swift package **and** the MLX Metal shader library in one step. The Metal library (`mlx.metallib`) is required for GPU inference — without it you'll get `Failed to load the default metallib` at runtime.

For debug builds: `make debug`. To run unit tests: `make test`.

## Try the Voice Assistant

**[PersonaPlexDemo](Examples/PersonaPlexDemo/)** is a ready-to-run macOS voice assistant — tap to talk, get spoken responses in real-time. Uses microphone input with Silero VAD for automatic speech detection, Qwen3-ASR for transcription, and PersonaPlex 7B for speech-to-speech generation. Multi-turn conversation with 18 voice presets and inner monologue transcript display.

```bash
make build  # from repo root — builds everything including MLX metallib
cd Examples/PersonaPlexDemo
# See Examples/PersonaPlexDemo/README.md for .app bundle instructions
```

> RTF ~0.94 on M2 Max (faster than real-time). Models download automatically on first run (~5.5 GB PersonaPlex + ~400 MB ASR).

## Demo Apps

- **[PersonaPlexDemo](Examples/PersonaPlexDemo/)** — Conversational voice assistant (mic input, VAD, multi-turn). See above.
- **[SpeechDemo](Examples/SpeechDemo/)** — Dictation (Parakeet TDT / Qwen3-ASR with language auto-detect) and text-to-speech synthesis (Qwen3-TTS) in a tabbed interface.

Build and run as a macOS `.app` bundle — see each demo's README for instructions.

## Speech-to-Text (ASR) — Transcribe Audio in Swift

### Basic Transcription

```swift
import Qwen3ASR

// Default: 0.6B model
let model = try await Qwen3ASRModel.fromPretrained()

// Or use the larger 1.7B model for better accuracy
let model = try await Qwen3ASRModel.fromPretrained(
    modelId: "aufklarer/Qwen3-ASR-1.7B-MLX-8bit"
)

// Audio can be any sample rate — automatically resampled to 16kHz internally
let transcription = model.transcribe(audio: audioSamples, sampleRate: 16000)
print(transcription)
```

### CoreML Encoder (Neural Engine)

Hybrid mode: CoreML encoder on Neural Engine + MLX text decoder on GPU. Lower power, frees GPU for the encoder pass.

```swift
import Qwen3ASR

let encoder = try await CoreMLASREncoder.fromPretrained()
let model = try await Qwen3ASRModel.fromPretrained()
let text = try model.transcribe(audio: audioSamples, sampleRate: 16000, coremlEncoder: encoder)
```

INT8 (180 MB, default) and INT4 (90 MB) variants available. INT8 recommended (cosine similarity > 0.999 vs FP32).

### Parakeet TDT (CoreML)

```swift
import ParakeetASR

let model = try await ParakeetASRModel.fromPretrained()
let transcription = model.transcribe(audio: audioSamples, sampleRate: 16000)
```

Runs on Neural Engine via CoreML — frees the GPU for concurrent workloads. 25 European languages, ~315 MB.

### ASR CLI

```bash
make build  # or: swift build -c release && ./scripts/build_mlx_metallib.sh release

# Default (Qwen3-ASR 0.6B, MLX)
.build/release/audio transcribe audio.wav

# Use 1.7B model
.build/release/audio transcribe audio.wav --model 1.7B

# CoreML encoder (Neural Engine + MLX decoder)
.build/release/audio transcribe --engine qwen3-coreml audio.wav

# Parakeet TDT (CoreML, Neural Engine)
.build/release/audio transcribe --engine parakeet audio.wav
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

Non-autoregressive — single forward pass, no sampling loop. See [Forced Aligner](docs/inference/forced-aligner.md) for architecture details.

## Text-to-Speech (TTS) — Generate Speech in Swift

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
make build
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

### Voice Cloning (Base model)

Clone a speaker's voice from a reference audio file. Two modes:

**ICL mode** (recommended) — encodes reference audio into codec tokens with transcript. Higher quality, reliable EOS:

```swift
let (model, encoder) = try await Qwen3TTSModel.fromPretrainedWithEncoder()
let refAudio = try AudioFileLoader.load(url: referenceURL, targetSampleRate: 24000)
let audio = model.synthesizeWithVoiceCloneICL(
    text: "Hello world",
    referenceAudio: refAudio,
    referenceSampleRate: 24000,
    referenceText: "Exact transcript of reference audio.",
    language: "english",
    codecEncoder: encoder
)
```

**X-vector mode** — speaker embedding only, no transcript needed but lower quality:

```swift
let audio = model.synthesizeWithVoiceClone(
    text: "Hello world",
    referenceAudio: refAudio,
    referenceSampleRate: 24000,
    language: "english"
)
```

CLI:

```bash
.build/release/audio speak "Hello world" --voice-sample reference.wav --output cloned.wav
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

## Speech-to-Speech — Full-Duplex Voice Conversation

> For an interactive voice assistant with microphone input, see **[PersonaPlexDemo](Examples/PersonaPlexDemo/)** — tap to talk, multi-turn conversation with automatic speech detection.

### Speech-to-Speech

```swift
import PersonaPlex
import AudioCommon  // for WAVWriter, AudioFileLoader

let model = try await PersonaPlexModel.fromPretrained()
// Downloads ~5.5 GB on first run (temporal 4-bit + depformer + Mimi codec + voice presets)

let audio = try AudioFileLoader.load(url: inputURL, targetSampleRate: 24000)
let (response, textTokens) = model.respond(userAudio: audio, voice: .NATM0)
// response: 24kHz mono float samples
// textTokens: model's inner monologue (SentencePiece token IDs)
try WAVWriter.write(samples: response.audio, sampleRate: 24000, to: outputURL)
```

### Inner Monologue (Text Output)

PersonaPlex generates text tokens alongside audio — the model's internal reasoning. Decode them with the built-in SentencePiece decoder:

```swift
let decoder = try SentencePieceDecoder(modelPath: "tokenizer_spm_32k_3.model")
let transcript = decoder.decode(textTokens)
print(transcript)  // e.g. "Sure, I can help you with that..."
```

### Streaming Speech-to-Speech

```swift
// Receive audio chunks as they're generated (~2s per chunk)
let stream = model.respondStream(userAudio: audio, voice: .NATM0)
for try await chunk in stream {
    playAudio(chunk.samples)  // play immediately, 24kHz mono
    // chunk.textTokens has this chunk's text; final chunk has all tokens
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

The system prompt steers the model's conversational behavior. Pass any custom prompt as a plain string:

```swift
// Custom system prompt (tokenized automatically)
let response = model.respond(
    userAudio: audio,
    voice: .NATM0,
    systemPrompt: "You enjoy having a good conversation."
)

// Or use a preset
let response = model.respond(
    userAudio: audio,
    voice: .NATM0,
    systemPromptTokens: SystemPromptPreset.customerService.tokens
)
```

Available presets: `focused` (default), `assistant`, `customerService`, `teacher`.

### PersonaPlex CLI

```bash
make build

# Basic speech-to-speech
.build/release/audio respond --input question.wav --output response.wav

# With transcript (decodes inner monologue text)
.build/release/audio respond --input question.wav --transcript

# JSON output (audio path, transcript, latency metrics)
.build/release/audio respond --input question.wav --json

# Custom system prompt text
.build/release/audio respond --input question.wav --system-prompt-text "You enjoy having a good conversation."

# Choose a voice and system prompt preset
.build/release/audio respond --input question.wav --voice NATF1 --system-prompt focused

# Tune sampling parameters
.build/release/audio respond --input question.wav --audio-temp 0.6 --repetition-penalty 1.5

# Enable text entropy early stopping (stops if text collapses)
.build/release/audio respond --input question.wav --entropy-threshold 1.0 --entropy-window 5

# List available voices and prompts
.build/release/audio respond --list-voices
.build/release/audio respond --list-prompts
```

## CosyVoice TTS — Streaming Text-to-Speech with Voice Cloning

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

### Voice Cloning (CosyVoice)

Clone a speaker's voice using the CAM++ speaker encoder (192-dim, CoreML Neural Engine):

```swift
import CosyVoiceTTS
import AudioCommon

let model = try await CosyVoiceTTSModel.fromPretrained()
let speaker = try await CamPlusPlusSpeaker.fromPretrained()
// Downloads ~14 MB CAM++ CoreML model on first use

let refAudio = try AudioFileLoader.load(url: referenceURL, targetSampleRate: 16000)
let embedding = try speaker.embed(audio: refAudio, sampleRate: 16000)
// embedding: [Float] of length 192

let audio = model.synthesize(
    text: "Hello in a cloned voice!",
    language: "english",
    speakerEmbedding: embedding
)
```

### CosyVoice TTS CLI

```bash
make build

# Basic synthesis
.build/release/audio speak "Hello world" --engine cosyvoice --language english --output output.wav

# Voice cloning (downloads CAM++ speaker encoder on first use)
.build/release/audio speak "Hello world" --engine cosyvoice --voice-sample reference.wav --output cloned.wav

# Multi-speaker dialogue with voice cloning
.build/release/audio speak "[S1] Hello there! [S2] Hey, how are you?" \
    --engine cosyvoice --speakers s1=alice.wav,s2=bob.wav -o dialogue.wav

# Inline emotion/style tags
.build/release/audio speak "(excited) Wow, amazing! (sad) But I have to go..." \
    --engine cosyvoice -o emotion.wav

# Combined: dialogue + emotions + voice cloning
.build/release/audio speak "[S1] (happy) Great news! [S2] (surprised) Really?" \
    --engine cosyvoice --speakers s1=alice.wav,s2=bob.wav -o combined.wav

# Custom style instruction
.build/release/audio speak "Hello world" --engine cosyvoice --cosy-instruct "Speak cheerfully" -o cheerful.wav

# Streaming synthesis
.build/release/audio speak "Hello world" --engine cosyvoice --language english --stream --output output.wav
```

## Kokoro TTS — Lightweight On-Device Text-to-Speech (iOS + macOS)

### Basic Synthesis

```swift
import KokoroTTS
import AudioCommon  // for WAVWriter

let tts = try await KokoroTTSModel.fromPretrained()
// Downloads ~170 MB on first run (CoreML models + voice embeddings + dictionaries)

let audio = try tts.synthesize(text: "Hello world", voice: "af_heart")
// Output is 24kHz mono float samples
try WAVWriter.write(samples: audio, sampleRate: 24000, to: outputURL)
```

54 preset voices across 10 languages. Non-autoregressive 3-stage CoreML pipeline, no sampling loop. Runs on Neural Engine, frees the GPU entirely.

### Kokoro TTS CLI

```bash
make build

# Basic synthesis
.build/release/audio kokoro "Hello world" --voice af_heart --output hello.wav

# Choose language
.build/release/audio kokoro "Bonjour le monde" --voice ff_siwis --language fr --output bonjour.wav

# List available voices
.build/release/audio kokoro --list-voices
```

## Qwen3 Chat (On-Device LLM)

```swift
import Qwen3Chat

let chat = try await Qwen3ChatModel.fromPretrained()
// Downloads ~318 MB on first run (INT4 CoreML model + tokenizer)

// Single response
let response = try chat.generate("What is Swift?", systemPrompt: "Answer briefly.")
print(response)

// Streaming tokens
let stream = chat.chatStream("Tell me a joke", systemPrompt: "Be funny.")
for try await token in stream {
    print(token, terminator: "")
}
```

Qwen3-0.6B INT4 quantized for CoreML. Runs on Neural Engine with ~2 tok/s on iPhone, ~15 tok/s on M-series. Supports multi-turn conversation with KV cache, thinking mode (`<think>` tokens), and configurable sampling (temperature, top-k, top-p, repetition penalty).

## Voice Activity Detection (VAD) — Detect Speech in Audio

### Streaming VAD (Silero)

Silero VAD v5 processes 32ms audio chunks with sub-millisecond latency — ideal for real-time speech detection from microphones or streams.

```swift
import SpeechVAD

let vad = try await SileroVADModel.fromPretrained()
// Or use CoreML (Neural Engine, lower power):
// let vad = try await SileroVADModel.fromPretrained(engine: .coreml)

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
make build

# Streaming Silero VAD (32ms chunks)
.build/release/audio vad-stream audio.wav

# CoreML backend (Neural Engine)
.build/release/audio vad-stream audio.wav --engine coreml

# With custom thresholds
.build/release/audio vad-stream audio.wav --onset 0.6 --offset 0.4

# JSON output
.build/release/audio vad-stream audio.wav --json

# Batch pyannote VAD (10s sliding windows)
.build/release/audio vad audio.wav
```

## Speaker Diarization — Who Spoke When

### Diarization Pipeline

```swift
import SpeechVAD

let pipeline = try await DiarizationPipeline.fromPretrained()
// Or use CoreML embeddings (Neural Engine, frees GPU):
// let pipeline = try await DiarizationPipeline.fromPretrained(embeddingEngine: .coreml)

let result = pipeline.diarize(audio: samples, sampleRate: 16000)
for seg in result.segments {
    print("Speaker \(seg.speakerId): [\(seg.startTime)s - \(seg.endTime)s]")
}
print("\(result.numSpeakers) speakers detected")
```

### Speaker Embedding

```swift
let model = try await WeSpeakerModel.fromPretrained()
// Or: let model = try await WeSpeakerModel.fromPretrained(engine: .coreml)
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

### Sortformer Diarization (End-to-End, CoreML)

NVIDIA Sortformer predicts per-frame speaker activity for up to 4 speakers directly — no embedding or clustering needed. Runs on Neural Engine.

```swift
let diarizer = try await SortformerDiarizer.fromPretrained()
let result = diarizer.diarize(audio: samples, sampleRate: 16000, config: .default)
for seg in result.segments {
    print("Speaker \(seg.speakerId): [\(seg.startTime)s - \(seg.endTime)s]")
}
```

### Diarization CLI

```bash
make build

# Pyannote diarization (default)
.build/release/audio diarize meeting.wav

# Sortformer diarization (CoreML, Neural Engine)
.build/release/audio diarize meeting.wav --engine sortformer

# CoreML embeddings (Neural Engine, pyannote only)
.build/release/audio diarize meeting.wav --embedding-engine coreml

# JSON output
.build/release/audio diarize meeting.wav --json

# Extract a specific speaker (pyannote only)
.build/release/audio diarize meeting.wav --target-speaker enrollment.wav

# Speaker embedding
.build/release/audio embed-speaker enrollment.wav --json
.build/release/audio embed-speaker enrollment.wav --engine coreml
```

See [Speaker Diarization](docs/inference/speaker-diarization.md) for architecture details.

## Speech Enhancement — Noise Suppression and Audio Cleanup

### Noise Suppression

```swift
import SpeechEnhancement
import AudioCommon  // for WAVWriter

let enhancer = try await SpeechEnhancer.fromPretrained()
// Downloads ~4.3 MB on first run (Core ML FP16 model + auxiliary data)

let cleanAudio = try enhancer.enhance(audio: noisyAudio, sampleRate: 48000)
try WAVWriter.write(samples: cleanAudio, sampleRate: 48000, to: outputURL)
```

### Denoise CLI

```bash
make build

# Basic noise removal
.build/release/audio denoise noisy.wav

# Custom output path
.build/release/audio denoise noisy.wav --output clean.wav
```

See [Speech Enhancement](docs/inference/speech-enhancement.md) for architecture details.

## Pipelines — Compose Multiple Models

All models conform to shared protocols (`SpeechRecognitionModel`, `SpeechGenerationModel`, `SpeechEnhancementModel`, etc.) and can be composed into pipelines:

### Noisy Speech Recognition (DeepFilterNet + ASR)

```swift
import SpeechEnhancement
import Qwen3ASR

let enhancer = try await SpeechEnhancer.fromPretrained()
let asr = try await Qwen3ASRModel.fromPretrained()

// Enhance at 48kHz, then transcribe at 16kHz
let clean = try enhancer.enhance(audio: noisyAudio, sampleRate: 48000)
let clean16k = AudioResampler.resample(clean, from: 48000, to: 16000)
let text = asr.transcribe(audio: clean16k, sampleRate: 16000)
```

### Voice-to-Voice Relay (VAD + ASR + TTS)

```swift
import SpeechVAD
import Qwen3ASR
import Qwen3TTS

let vad = try await SileroVADModel.fromPretrained()
let asr = try await Qwen3ASRModel.fromPretrained()
let tts = try await Qwen3TTSModel.fromPretrained()

// Detect speech segments, transcribe, re-synthesize
let segments = vad.detectSpeech(audio: audio, sampleRate: 16000)
for seg in segments {
    let chunk = Array(audio[Int(seg.startTime * 16000)..<Int(seg.endTime * 16000)])
    let text = asr.transcribe(audio: chunk, sampleRate: 16000)
    let speech = tts.synthesize(text: text, language: "english")
    // speech: 24kHz mono float samples
}
```

### Meeting Transcription (Diarization + ASR)

```swift
import SpeechVAD
import Qwen3ASR

let pipeline = try await DiarizationPipeline.fromPretrained()
let asr = try await Qwen3ASRModel.fromPretrained()

let result = pipeline.diarize(audio: meetingAudio, sampleRate: 16000)
for seg in result.segments {
    let chunk = Array(meetingAudio[Int(seg.startTime * 16000)..<Int(seg.endTime * 16000)])
    let text = asr.transcribe(audio: chunk, sampleRate: 16000)
    print("Speaker \(seg.speakerId) [\(seg.startTime)s-\(seg.endTime)s]: \(text)")
}
```

See [Shared Protocols](docs/shared-protocols.md) for the full protocol reference.

## HTTP API Server

A standalone HTTP server exposes all models via REST and WebSocket endpoints. Models are loaded lazily on first request.

```bash
swift build -c release
.build/release/audio-server --port 8080

# Transcribe audio
curl -X POST http://localhost:8080/transcribe --data-binary @audio.wav -H "Content-Type: audio/wav"

# Text-to-speech
curl -X POST http://localhost:8080/speak -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "engine": "cosyvoice"}' -o output.wav

# Speech-to-speech (PersonaPlex)
curl -X POST http://localhost:8080/respond --data-binary @question.wav -o response.wav

# Speech enhancement
curl -X POST http://localhost:8080/enhance --data-binary @noisy.wav -o clean.wav

# Preload all models on startup
.build/release/audio-server --preload --port 8080
```

### WebSocket Streaming

#### OpenAI Realtime API (`/v1/realtime`)

The primary WebSocket endpoint implements the [OpenAI Realtime API](https://platform.openai.com/docs/api-reference/realtime) protocol — all messages are JSON with a `type` field, audio is base64-encoded PCM16 24kHz mono.

**Client → Server events:**

| Event | Description |
|-------|-------------|
| `session.update` | Configure engine, language, audio format |
| `input_audio_buffer.append` | Send base64 PCM16 audio chunk |
| `input_audio_buffer.commit` | Transcribe accumulated audio (ASR) |
| `input_audio_buffer.clear` | Clear audio buffer |
| `response.create` | Request TTS synthesis |

**Server → Client events:**

| Event | Description |
|-------|-------------|
| `session.created` | Session initialized |
| `session.updated` | Configuration confirmed |
| `input_audio_buffer.committed` | Audio committed for transcription |
| `conversation.item.input_audio_transcription.completed` | ASR result |
| `response.audio.delta` | Base64 PCM16 audio chunk (TTS) |
| `response.audio.done` | Audio streaming complete |
| `response.done` | Response complete with metadata |
| `error` | Error with type and message |

```javascript
const ws = new WebSocket('ws://localhost:8080/v1/realtime');

// ASR: send audio, get transcription
ws.send(JSON.stringify({ type: 'input_audio_buffer.append', audio: base64PCM16 }));
ws.send(JSON.stringify({ type: 'input_audio_buffer.commit' }));
// → receives: conversation.item.input_audio_transcription.completed

// TTS: send text, get streamed audio
ws.send(JSON.stringify({
  type: 'response.create',
  response: { modalities: ['audio', 'text'], instructions: 'Hello world' }
}));
// → receives: response.audio.delta (base64 chunks), response.audio.done, response.done
```

An example HTML client is at `Examples/websocket-client.html` — open it in a browser while the server is running.

The server is a separate `AudioServer` module and `audio-server` executable — it does not add Hummingbird/WebSocket to the main `audio` CLI.

## Latency (M2 Max, 64 GB)

### ASR

| Model | Backend | RTF | 10s audio processed in |
|-------|---------|-----|------------------------|
| Qwen3-ASR-0.6B (4-bit) | MLX | ~0.06 | ~0.6s |
| Qwen3-ASR-0.6B (INT8) | CoreML + MLX | ~0.09 | ~0.9s |
| Qwen3-ASR-1.7B (8-bit) | MLX | ~0.11 | ~1.1s |
| Parakeet-TDT-0.6B (INT4) | CoreML (Neural Engine) | ~0.12 cold, ~0.03 warm | ~1.2s / ~0.3s |
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
| Kokoro-82M | CoreML (Neural Engine) | ~1.4s (RTFx 0.7) | ~4.3s (RTFx 0.7) | ~8.6s (RTFx 0.7) | N/A (non-autoregressive) |
| Apple `AVSpeechSynthesizer` | AVFoundation | 0.08s | 0.08s | 0.17s (RTF 0.02) | N/A |

> Qwen3-TTS generates natural, expressive speech with prosody and emotion, running **faster than real-time** (RTF < 1.0). Streaming synthesis delivers the first audio chunk in ~120ms. Kokoro-82M runs entirely on Neural Engine with a 3-stage pipeline (RTFx ~0.7), ideal for iOS. Apple's built-in TTS is faster but produces robotic, monotone speech.

### PersonaPlex (Speech-to-Speech)

| Model | Framework | ms/step | RTF | Notes |
|-------|-----------|---------|-----|-------|
| PersonaPlex-7B (8-bit) | MLX Swift (release) | ~112ms | ~1.4 | Recommended — coherent responses, 30% faster than 4-bit |
| PersonaPlex-7B (4-bit) | MLX Swift (release) | ~158ms | ~1.97 | Not recommended — garbled output quality |

> **Use 8-bit.** INT8 is both faster (112 ms/step vs 158 ms/step) and produces coherent full-duplex responses. INT4 quantization degrades generation quality, producing incoherent speech ("I go tea my coffee brewing..."). INT8 runs at ~112ms/step on M2 Max — above the 80ms real-time threshold but close to usable for streaming, and the output quality difference is decisive.

### VAD & Speaker Embedding

| Model | Backend | Per-call Latency | RTF | Notes |
|-------|---------|-----------------|-----|-------|
| Silero-VAD-v5 | MLX | ~2.1ms / chunk | 0.065 | GPU (Metal) |
| Silero-VAD-v5 | CoreML | ~0.27ms / chunk | 0.008 | Neural Engine, **7.7x faster** |
| WeSpeaker ResNet34-LM | MLX | ~310ms / 20s audio | 0.016 | GPU (Metal) |
| WeSpeaker ResNet34-LM | CoreML | ~430ms / 20s audio | 0.021 | Neural Engine, frees GPU |

> Silero VAD CoreML runs on the Neural Engine at 7.7x the speed of MLX, making it ideal for always-on microphone input. WeSpeaker MLX is faster on GPU, but CoreML frees the GPU for concurrent workloads (TTS, ASR). Both backends produce equivalent results.

### Speech Enhancement

| Model | Backend | Duration | Latency | RTF |
|-------|---------|----------|---------|-----|
| DeepFilterNet3 (FP16) | CoreML | 5s | 0.65s | 0.13 |
| DeepFilterNet3 (FP16) | CoreML | 10s | 1.2s | 0.12 |
| DeepFilterNet3 (FP16) | CoreML | 20s | 4.8s | 0.24 |

RTF = Real-Time Factor (lower is better, < 1.0 = faster than real-time). GRU cost scales ~O(n²).

### MLX vs CoreML

Both backends produce equivalent results. Choose based on your workload:

| | MLX | CoreML |
|---|---|---|
| **Hardware** | GPU (Metal shaders) | Neural Engine + CPU |
| **Best for** | Maximum throughput, single-model workloads | Multi-model pipelines, background tasks |
| **Power** | Higher GPU utilization | Lower power, frees GPU |
| **Latency** | Faster for large models (WeSpeaker) | Faster for small models (Silero VAD) |

**Desktop inference**: MLX is the default — fastest single-model performance on Apple Silicon. Switch to CoreML when running multiple models concurrently (e.g., VAD + ASR + TTS) to avoid GPU contention, or for battery-sensitive workloads on laptops.

CoreML models are available for Qwen3-ASR encoder, Silero VAD, and WeSpeaker. For Qwen3-ASR, use `--engine qwen3-coreml` (hybrid: CoreML encoder on ANE + MLX text decoder on GPU). For VAD/embeddings, pass `engine: .coreml` at construction time — inference API is identical.

## Accuracy Benchmarks

### ASR — Word Error Rate ([details](docs/benchmarks/asr-wer.md))

| Model | WER% (LibriSpeech test-clean) | RTF |
|-------|-------------------------------|-----|
| Qwen3-ASR 1.7B 8-bit | **2.35** | 0.090 |
| Qwen3-ASR 1.7B 4-bit | 2.57 | 0.045 |
| Parakeet TDT INT8 | 2.74 | 0.089 |
| Qwen3-ASR 0.6B 8-bit | 2.80 | 0.025 |

Qwen3-ASR 1.7B 8-bit beats Whisper Large v3 Turbo (2.5%) at comparable size. Multilingual: 10 languages benchmarked on FLEURS.

### TTS — Round-Trip Intelligibility ([details](docs/benchmarks/tts-roundtrip.md))

| Engine | WER% | RTF |
|--------|------|-----|
| CosyVoice3 | **3.25** | 0.59 |
| Qwen3-TTS 1.7B | 3.47 | 0.79 |
| Kokoro-82M | 3.90 | 0.17 |

### VAD — Speech Detection ([details](docs/benchmarks/vad-detection.md))

| Engine | F1% (FLEURS) | RTF |
|--------|-------------|-----|
| FireRedVAD | **99.12** | 0.007 |
| Silero CoreML | 95.13 | 0.022 |
| Pyannote MLX | 94.86 | 0.358 |

## Architecture

**Models:** [ASR Model](docs/models/asr-model.md), [TTS Model](docs/models/tts-model.md), [CosyVoice TTS](docs/models/cosyvoice-tts.md), [Kokoro TTS](docs/models/kokoro-tts.md), [Parakeet TDT](docs/models/parakeet-asr.md), [PersonaPlex](docs/models/personaplex.md), [FireRedVAD](docs/models/fireredvad.md)

**Inference:** [Qwen3-ASR](docs/inference/qwen3-asr-inference.md), [Parakeet TDT](docs/inference/parakeet-asr-inference.md), [TTS](docs/inference/qwen3-tts-inference.md), [Forced Aligner](docs/inference/forced-aligner.md), [FireRedVAD](docs/inference/fireredvad.md), [Silero VAD](docs/inference/silero-vad.md), [Speaker Diarization](docs/inference/speaker-diarization.md), [Speech Enhancement](docs/inference/speech-enhancement.md)

**Audio:** [Streaming Playback](docs/audio/playback.md), [Voice Pipeline](docs/audio/voice-pipeline.md)

**Benchmarks:** [ASR WER](docs/benchmarks/asr-wer.md), [TTS Round-Trip](docs/benchmarks/tts-roundtrip.md), [VAD Detection](docs/benchmarks/vad-detection.md)

**Reference:** [Shared Protocols](docs/shared-protocols.md)

## Cache Configuration

Model weights are cached locally. Override the cache location with:

```bash
export QWEN3_CACHE_DIR=/path/to/cache
```

## MLX Metal Library

If you see `Failed to load the default metallib` at runtime, the Metal shader library is missing. Run `make build` (or `./scripts/build_mlx_metallib.sh release` after a manual `swift build`) to compile it. If the Metal Toolchain is missing, install it first:

```bash
xcodebuild -downloadComponent MetalToolchain
```

## Testing

Unit tests (config, sampling, text preprocessing, timestamp correction) run without model downloads:

```bash
swift test --filter "Qwen3TTSConfigTests|SamplingTests|CosyVoiceTTSConfigTests|CamPlusPlusMelExtractorTests|PersonaPlexTests|ForcedAlignerTests/testText|ForcedAlignerTests/testTimestamp|ForcedAlignerTests/testLIS|SileroVADTests/testSilero|SileroVADTests/testReflection|SileroVADTests/testProcess|SileroVADTests/testReset|SileroVADTests/testDetect|SileroVADTests/testStreaming|SileroVADTests/testVADEvent|KokoroTTSTests"
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
| Parakeet TDT | 25 European languages (BG, CS, DA, DE, EL, EN, ES, ET, FI, FR, HR, HU, IT, LT, LV, MT, NL, PL, PT, RO, RU, SK, SL, SV, UK) |
| Qwen3-TTS | EN, CN, DE, JA, ES, FR, KO, RU, IT, PT (+ Beijing/Sichuan dialects via CustomVoice) |
| CosyVoice TTS | CN, EN, JA, KO, DE, ES, FR, IT, RU |
| Kokoro TTS | EN (US/UK), ES, FR, HI, IT, JA, PT, CN, KO, DE |
| PersonaPlex | EN |

## How It Compares

### Speech-to-Text (ASR): speech-swift vs Alternatives

| | **speech-swift (Qwen3-ASR)** | **whisper.cpp** | **Apple SFSpeechRecognizer** | **Google Cloud Speech** |
|---|---|---|---|---|
| **Runtime** | On-device (MLX/CoreML) | On-device (CPU/GPU) | On-device or cloud | Cloud only |
| **Languages** | 52 | 100+ | ~70 (on-device: limited) | 125+ |
| **RTF (10s audio, M2 Max)** | 0.06 (17x real-time) | 0.10 (Whisper-large-v3) | N/A | N/A |
| **Streaming** | No (batch) | No (batch) | Yes | Yes |
| **Custom models** | Yes (swap HuggingFace weights) | Yes (GGML models) | No | No |
| **Swift API** | Native async/await | C++ with Swift bridge | Native | REST/gRPC |
| **Privacy** | Fully on-device | Fully on-device | Depends on config | Data sent to cloud |
| **Word timestamps** | Yes (Forced Aligner) | Yes | Limited | Yes |
| **Cost** | Free (Apache 2.0) | Free (MIT) | Free (on-device) | Pay per minute |

### Text-to-Speech (TTS): speech-swift vs Alternatives

| | **speech-swift (Qwen3-TTS)** | **speech-swift (Kokoro)** | **Apple AVSpeechSynthesizer** | **ElevenLabs / Cloud TTS** |
|---|---|---|---|---|
| **Quality** | Neural, expressive | Neural, natural | Robotic, monotone | Neural, highest quality |
| **Runtime** | On-device (MLX) | On-device (CoreML) | On-device | Cloud only |
| **Streaming** | Yes (~120ms first chunk) | No (single pass, ~45ms) | No | Yes |
| **Voice cloning** | Yes | No | No | Yes |
| **Voices** | 9 built-in + clone any | 50 preset voices | ~50 system voices | 1000+ |
| **Languages** | 10 | 10 | 60+ | 30+ |
| **iOS support** | macOS only | iOS + macOS | iOS + macOS | Any (API) |
| **Cost** | Free (Apache 2.0) | Free (Apache 2.0) | Free | Pay per character |

### When to Use speech-swift

- **Privacy-critical apps** — medical, legal, enterprise where audio cannot leave the device
- **Offline use** — no internet connection needed after initial model download
- **Cost-sensitive** — no per-minute or per-character API charges
- **Apple Silicon optimization** — built specifically for M-series GPU (Metal) and Neural Engine
- **Full pipeline** — combine ASR + TTS + VAD + diarization + enhancement in a single Swift package

## FAQ

**Does speech-swift work on iOS?**
Kokoro TTS, Qwen3.5-Chat (CoreML), Silero VAD, Parakeet ASR, DeepFilterNet3, and WeSpeaker all run on iOS 17+ via CoreML on the Neural Engine. MLX-based models (Qwen3-ASR, Qwen3-TTS, Qwen3.5-Chat MLX, PersonaPlex) require macOS 14+ on Apple Silicon.

**Does it require an internet connection?**
Only for the initial model download from HuggingFace (automatic, cached in `~/Library/Caches/qwen3-speech/`). After that, all inference runs fully offline with no network access.

**How does speech-swift compare to Whisper?**
Qwen3-ASR-0.6B achieves RTF 0.06 on M2 Max — 40% faster than Whisper-large-v3 via whisper.cpp (RTF 0.10) — with comparable accuracy across 52 languages. speech-swift provides a native Swift async/await API, while whisper.cpp requires a C++ bridge.

**Can I use it in a commercial app?**
Yes. speech-swift is licensed under Apache 2.0. The underlying model weights have their own licenses (check each model's HuggingFace page).

**What Apple Silicon chips are supported?**
All M-series chips: M1, M2, M3, M4 and their Pro/Max/Ultra variants. Requires macOS 14+ (Sonoma) or iOS 17+.

**How much memory does it need?**
From ~3 MB (Silero VAD) to ~6.5 GB (PersonaPlex 7B). Kokoro TTS uses ~500 MB, Qwen3-ASR ~2.2 GB. See the [Memory Requirements](#memory-requirements) table for full details.

**Can I run multiple models simultaneously?**
Yes. Use CoreML models on the Neural Engine alongside MLX models on the GPU to avoid contention — for example, Silero VAD (CoreML) + Qwen3-ASR (MLX) + Qwen3-TTS (MLX).

**Is there a REST API?**
Yes. The `audio-server` binary exposes all models via HTTP REST and WebSocket endpoints, including an OpenAI Realtime API-compatible WebSocket at `/v1/realtime`.

## Contributing

We welcome contributions! Whether it's a bug fix, new model integration, or documentation improvement — PRs are appreciated.

**To get started:**
1. Fork the repo and create a feature branch
2. `make build` to compile (requires Xcode + Metal Toolchain)
3. `make test` to run the test suite
4. Open a PR against `main`

## License

Apache 2.0

