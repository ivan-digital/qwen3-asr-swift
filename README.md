# Speech Swift

AI speech models for Apple Silicon, powered by MLX Swift and CoreML.

📖 Read in: [English](README.md) · [中文](README_zh.md) · [日本語](README_ja.md) · [한국어](README_ko.md) · [Español](README_es.md) · [Deutsch](README_de.md) · [Français](README_fr.md) · [हिन्दी](README_hi.md) · [Português](README_pt.md) · [Русский](README_ru.md)

On-device speech recognition, synthesis, and understanding for Mac and iOS. Runs locally on Apple Silicon — no cloud, no API keys, no data leaves your device.

**[📚 Full Documentation →](https://soniqo.audio)** · **[🤗 HuggingFace Models](https://huggingface.co/aufklarer)** · **[📝 Blog](https://blog.ivan.digital)**

- **[Qwen3-ASR](https://soniqo.audio/guides/transcribe)** — Speech-to-text (automatic speech recognition, 52 languages, MLX + CoreML)
- **[Parakeet TDT](https://soniqo.audio/guides/parakeet)** — Speech-to-text via CoreML (Neural Engine, NVIDIA FastConformer + TDT decoder, 25 languages)
- **[Omnilingual ASR](https://soniqo.audio/guides/omnilingual)** — Speech-to-text (Meta wav2vec2 + CTC, **1,672 languages** across 32 scripts, CoreML 300M + MLX 300M/1B/3B/7B)
- **[Streaming Dictation](https://soniqo.audio/guides/dictate)** — Real-time dictation with partials and end-of-utterance detection (Parakeet-EOU-120M)
- **[Nemotron Streaming](https://soniqo.audio/guides/nemotron)** — Low-latency streaming ASR with native punctuation and capitalization (NVIDIA Nemotron-Speech-Streaming-0.6B, CoreML, English)
- **[Qwen3-ForcedAligner](https://soniqo.audio/guides/align)** — Word-level timestamp alignment (audio + text → timestamps)
- **[Qwen3-TTS](https://soniqo.audio/guides/speak)** — Text-to-speech (highest quality, streaming, custom speakers, 10 languages)
- **[CosyVoice TTS](https://soniqo.audio/guides/cosyvoice)** — Streaming TTS with voice cloning, multi-speaker dialogue, emotion tags (9 languages)
- **[Kokoro TTS](https://soniqo.audio/guides/kokoro)** — On-device TTS (82M, CoreML/Neural Engine, 54 voices, iOS-ready, 10 languages)
- **[Qwen3.5-Chat](https://soniqo.audio/guides/chat)** — On-device LLM chat (0.8B, MLX INT4 + CoreML INT8, DeltaNet hybrid, streaming tokens)
- **[PersonaPlex](https://soniqo.audio/guides/respond)** — Full-duplex speech-to-speech (7B, audio in → audio out, 18 voice presets)
- **[DeepFilterNet3](https://soniqo.audio/guides/denoise)** — Real-time noise suppression (2.1M params, 48 kHz)
- **[Source Separation](https://soniqo.audio/guides/separate)** — Music source separation via Open-Unmix (UMX-HQ / UMX-L, 4 stems: vocals/drums/bass/other, 44.1 kHz stereo)
- **[Wake-word](https://soniqo.audio/guides/wake-word)** — On-device keyword spotting (KWS Zipformer 3M, CoreML, 26× real-time, configurable keyword list)
- **[VAD](https://soniqo.audio/guides/vad)** — Voice activity detection (Silero streaming, Pyannote offline, FireRedVAD 100+ languages)
- **[Speaker Diarization](https://soniqo.audio/guides/diarize)** — Who spoke when (Pyannote pipeline, Sortformer end-to-end on Neural Engine)
- **[Speaker Embeddings](https://soniqo.audio/guides/embed-speaker)** — WeSpeaker ResNet34 (256-dim), CAM++ (192-dim)

Papers: [Qwen3-ASR](https://arxiv.org/abs/2601.21337) (Alibaba) · [Qwen3-TTS](https://arxiv.org/abs/2601.15621) (Alibaba) · [Omnilingual ASR](https://arxiv.org/abs/2511.09690) (Meta) · [Parakeet TDT](https://arxiv.org/abs/2304.06795) (NVIDIA) · [CosyVoice 3](https://arxiv.org/abs/2505.17589) (Alibaba) · [Kokoro](https://arxiv.org/abs/2301.01695) (StyleTTS 2) · [PersonaPlex](https://arxiv.org/abs/2602.06053) (NVIDIA) · [Mimi](https://arxiv.org/abs/2410.00037) (Kyutai) · [Sortformer](https://arxiv.org/abs/2409.06656) (NVIDIA)

## News

- **19 Apr 2026** — [MLX vs CoreML on Apple Silicon — A Practical Guide to Picking the Right Backend](https://blog.ivan.digital/mlx-vs-coreml-on-apple-silicon-a-practical-guide-to-picking-the-right-backend-and-why-you-should-f77ddea7b27a)
- **20 Mar 2026** — [We Beat Whisper Large v3 with a 600M Model Running Entirely on Your Mac](https://blog.ivan.digital/we-beat-whisper-large-v3-with-a-600m-model-running-entirely-on-your-mac-20e6ce191174)
- **26 Feb 2026** — [Speaker Diarization and Voice Activity Detection on Apple Silicon — Native Swift with MLX](https://blog.ivan.digital/speaker-diarization-and-voice-activity-detection-on-apple-silicon-native-swift-with-mlx-92ea0c9aca0f)
- **23 Feb 2026** — [NVIDIA PersonaPlex 7B on Apple Silicon — Full-Duplex Speech-to-Speech in Native Swift with MLX](https://blog.ivan.digital/nvidia-personaplex-7b-on-apple-silicon-full-duplex-speech-to-speech-in-native-swift-with-mlx-0aa5276f2e23)
- **12 Feb 2026** — [Qwen3-ASR Swift: On-Device ASR + TTS for Apple Silicon — Architecture and Benchmarks](https://blog.ivan.digital/qwen3-asr-swift-on-device-asr-tts-for-apple-silicon-architecture-and-benchmarks-27cbf1e4463f)

## Quick start

Add the package to your `Package.swift`:

```swift
.package(url: "https://github.com/soniqo/speech-swift", branch: "main")
```

Import only the modules you need — every model is its own SPM library, so you don't pay for what you don't use:

```swift
.product(name: "ParakeetStreamingASR", package: "speech-swift"),
.product(name: "SpeechUI",             package: "speech-swift"),  // optional SwiftUI views
```

**Transcribe an audio buffer in 3 lines:**

```swift
import ParakeetStreamingASR

let model = try await ParakeetStreamingASRModel.fromPretrained()
let text = try model.transcribeAudio(audioSamples, sampleRate: 16000)
```

**Live streaming with partials:**

```swift
for await partial in model.transcribeStream(audio: samples, sampleRate: 16000) {
    print(partial.isFinal ? "FINAL: \(partial.text)" : "... \(partial.text)")
}
```

**SwiftUI dictation view in ~10 lines:**

```swift
import SwiftUI
import ParakeetStreamingASR
import SpeechUI

@MainActor
struct DictateView: View {
    @State private var store = TranscriptionStore()

    var body: some View {
        TranscriptionView(finals: store.finalLines, currentPartial: store.currentPartial)
            .task {
                let model = try? await ParakeetStreamingASRModel.fromPretrained()
                guard let model else { return }
                for await p in model.transcribeStream(audio: samples, sampleRate: 16000) {
                    store.apply(text: p.text, isFinal: p.isFinal)
                }
            }
    }
}
```

`SpeechUI` ships only `TranscriptionView` (finals + partials) and `TranscriptionStore` (streaming ASR adapter). Use AVFoundation for audio visualization and playback.

Available SPM products: `Qwen3ASR`, `Qwen3TTS`, `Qwen3TTSCoreML`, `ParakeetASR`, `ParakeetStreamingASR`, `NemotronStreamingASR`, `OmnilingualASR`, `KokoroTTS`, `CosyVoiceTTS`, `PersonaPlex`, `SpeechVAD`, `SpeechEnhancement`, `SourceSeparation`, `Qwen3Chat`, `SpeechCore`, `SpeechUI`, `AudioCommon`.

## Models

Compact view below. **[Full model catalogue with sizes, quantisations, download URLs, and memory tables → soniqo.audio/architecture](https://soniqo.audio/architecture)**.

| Model | Task | Backends | Sizes | Languages |
|-------|------|----------|-------|-----------|
| [Qwen3-ASR](https://soniqo.audio/guides/transcribe) | Speech → Text | MLX, CoreML (hybrid) | 0.6B, 1.7B | 52 |
| [Parakeet TDT](https://soniqo.audio/guides/parakeet) | Speech → Text | CoreML (ANE) | 0.6B | 25 European |
| [Parakeet EOU](https://soniqo.audio/guides/dictate) | Speech → Text (streaming) | CoreML (ANE) | 120M | 25 European |
| [Nemotron Streaming](https://soniqo.audio/guides/nemotron) | Speech → Text (streaming, punctuated) | CoreML (ANE) | 0.6B | EN |
| [Omnilingual ASR](https://soniqo.audio/guides/omnilingual) | Speech → Text | CoreML (ANE), MLX | 300M / 1B / 3B / 7B | **[1,672](https://github.com/facebookresearch/omnilingual-asr/blob/main/src/omnilingual_asr/models/wav2vec2_llama/lang_ids.py)** |
| [Qwen3-ForcedAligner](https://soniqo.audio/guides/align) | Audio + Text → Timestamps | MLX, CoreML | 0.6B | Multi |
| [Qwen3-TTS](https://soniqo.audio/guides/speak) | Text → Speech | MLX, CoreML | 0.6B, 1.7B | 10 |
| [CosyVoice3](https://soniqo.audio/guides/cosyvoice) | Text → Speech | MLX | 0.5B | 9 |
| [Kokoro-82M](https://soniqo.audio/guides/kokoro) | Text → Speech | CoreML (ANE) | 82M | 10 |
| [Qwen3.5-Chat](https://soniqo.audio/guides/chat) | Text → Text (LLM) | MLX, CoreML | 0.8B | Multi |
| [PersonaPlex](https://soniqo.audio/guides/respond) | Speech → Speech | MLX | 7B | EN |
| [Silero VAD](https://soniqo.audio/guides/vad) | Voice Activity Detection | MLX, CoreML | 309K | Agnostic |
| [Pyannote](https://soniqo.audio/guides/diarize) | VAD + Diarization | MLX | 1.5M | Agnostic |
| [Sortformer](https://soniqo.audio/guides/diarize) | Diarization (E2E) | CoreML (ANE) | — | Agnostic |
| [DeepFilterNet3](https://soniqo.audio/guides/denoise) | Speech Enhancement | CoreML | 2.1M | Agnostic |
| [Open-Unmix](https://soniqo.audio/guides/separate) | Source Separation | MLX | 8.6M | Agnostic |
| [WeSpeaker](https://soniqo.audio/guides/embed-speaker) | Speaker Embedding | MLX, CoreML | 6.6M | Agnostic |

## Installation

### Homebrew

Requires native ARM Homebrew (`/opt/homebrew`). Rosetta/x86_64 Homebrew is not supported.

```bash
brew tap soniqo/speech https://github.com/soniqo/speech-swift
brew install speech
```

Then:

```bash
audio transcribe recording.wav
audio speak "Hello world"
audio respond --input question.wav --transcript
audio-server --port 8080            # local HTTP / WebSocket server (OpenAI-compatible /v1/realtime)
```

**[Full CLI reference →](https://soniqo.audio/cli)**

### Swift Package Manager

```swift
dependencies: [
    .package(url: "https://github.com/soniqo/speech-swift", branch: "main")
]
```

Import only what you need — every model is its own SPM target:

```swift
import Qwen3ASR             // Speech recognition (MLX)
import ParakeetASR          // Speech recognition (CoreML, batch)
import ParakeetStreamingASR // Streaming dictation with partials + EOU
import NemotronStreamingASR // English streaming ASR with native punctuation (0.6B)
import OmnilingualASR       // 1,672 languages (CoreML + MLX)
import Qwen3TTS             // Text-to-speech
import CosyVoiceTTS         // Text-to-speech with voice cloning
import KokoroTTS            // Text-to-speech (iOS-ready)
import Qwen3Chat            // On-device LLM chat
import PersonaPlex          // Full-duplex speech-to-speech
import SpeechVAD            // VAD + speaker diarization + embeddings
import SpeechEnhancement    // Noise suppression
import SourceSeparation     // Music source separation (Open-Unmix, 4 stems)
import SpeechUI             // SwiftUI components for streaming transcripts
import AudioCommon          // Shared protocols and utilities
```

### Requirements

- Swift 6+, Xcode 16+ (with Metal Toolchain)
- macOS 15+ (Sequoia) or iOS 18+, Apple Silicon (M1/M2/M3/M4)

The macOS 15 / iOS 18 minimum comes from [MLState](https://developer.apple.com/documentation/coreml/mlstate) — Apple's persistent ANE state API used by the CoreML pipelines (Qwen3-ASR, Qwen3-Chat, Qwen3-TTS) to keep KV caches resident on the Neural Engine across token steps.

### Build from source

```bash
git clone https://github.com/soniqo/speech-swift
cd speech-swift
make build
```

`make build` compiles the Swift package **and** the MLX Metal shader library. The Metal library is required for GPU inference — without it you'll see `Failed to load the default metallib` at runtime. `make debug` for debug builds, `make test` for the test suite.

**[Full build and install guide →](https://soniqo.audio/getting-started)**

## Demo apps

- **[DictateDemo](Examples/DictateDemo/)** ([docs](https://soniqo.audio/guides/dictate)) — macOS menu-bar streaming dictation with live partials, VAD-driven end-of-utterance detection, and one-click copy. Runs as a background agent (Parakeet-EOU-120M + Silero VAD).
- **[iOSEchoDemo](Examples/iOSEchoDemo/)** — iOS echo demo (Parakeet ASR + Kokoro TTS). Device and simulator.
- **[PersonaPlexDemo](Examples/PersonaPlexDemo/)** — Conversational voice assistant with mic input, VAD, and multi-turn context. macOS. RTF ~0.94 on M2 Max (faster than real-time).
- **[SpeechDemo](Examples/SpeechDemo/)** — Dictation and TTS synthesis in a tabbed interface. macOS.

Each demo's README has build instructions.

## Code examples

The snippets below show the minimal path for each domain. Every section links to a full guide on [soniqo.audio](https://soniqo.audio) with configuration options, multiple backends, streaming patterns, and CLI recipes.

### Speech-to-Text — [full guide →](https://soniqo.audio/guides/transcribe)

```swift
import Qwen3ASR

let model = try await Qwen3ASRModel.fromPretrained()
let text = model.transcribe(audio: audioSamples, sampleRate: 16000)
```

Alternative backends: [Parakeet TDT](https://soniqo.audio/guides/parakeet) (CoreML, 32× realtime), [Omnilingual ASR](https://soniqo.audio/guides/omnilingual) (1,672 languages, CoreML or MLX), [Streaming dictation](https://soniqo.audio/guides/dictate) (live partials).

### Forced Alignment — [full guide →](https://soniqo.audio/guides/align)

```swift
import Qwen3ASR

let aligner = try await Qwen3ForcedAligner.fromPretrained()
let aligned = aligner.align(
    audio: audioSamples,
    text: "Can you guarantee that the replacement part will be shipped tomorrow?",
    sampleRate: 24000
)
for word in aligned {
    print("[\(word.startTime)s - \(word.endTime)s] \(word.text)")
}
```

### Text-to-Speech — [full guide →](https://soniqo.audio/guides/speak)

```swift
import Qwen3TTS
import AudioCommon

let model = try await Qwen3TTSModel.fromPretrained()
let audio = model.synthesize(text: "Hello world", language: "english")
try WAVWriter.write(samples: audio, sampleRate: 24000, to: outputURL)
```

Alternative TTS engines: [CosyVoice3](https://soniqo.audio/guides/cosyvoice) (streaming + voice cloning + emotion tags), [Kokoro-82M](https://soniqo.audio/guides/kokoro) (iOS-ready, 54 voices), [Voice cloning](https://soniqo.audio/guides/voice-cloning).

### Speech-to-Speech — [full guide →](https://soniqo.audio/guides/respond)

```swift
import PersonaPlex

let model = try await PersonaPlexModel.fromPretrained()
let responseAudio = model.respond(userAudio: userSamples)
// 24 kHz mono Float32 output ready for playback
```

### LLM Chat — [full guide →](https://soniqo.audio/guides/chat)

```swift
import Qwen3Chat

let chat = try await Qwen35MLXChat.fromPretrained()
chat.chat(messages: [(.user, "Explain MLX in one sentence")]) { token, isFinal in
    print(token, terminator: "")
}
```

### Voice Activity Detection — [full guide →](https://soniqo.audio/guides/vad)

```swift
import SpeechVAD

let vad = try await SileroVADModel.fromPretrained()
let segments = vad.detectSpeech(audio: samples, sampleRate: 16000)
for s in segments { print("\(s.startTime)s → \(s.endTime)s") }
```

### Speaker Diarization — [full guide →](https://soniqo.audio/guides/diarize)

```swift
import SpeechVAD

let diarizer = try await DiarizationPipeline.fromPretrained()
let segments = diarizer.diarize(audio: samples, sampleRate: 16000)
for s in segments { print("Speaker \(s.speakerId): \(s.startTime)s - \(s.endTime)s") }
```

### Speech Enhancement — [full guide →](https://soniqo.audio/guides/denoise)

```swift
import SpeechEnhancement

let denoiser = try await DeepFilterNet3Model.fromPretrained()
let clean = try denoiser.enhance(audio: noisySamples, sampleRate: 48000)
```

### Voice Pipeline (ASR → LLM → TTS) — [full guide →](https://soniqo.audio/api)

```swift
import SpeechCore

let pipeline = VoicePipeline(
    stt: parakeetASR,
    tts: qwen3TTS,
    vad: sileroVAD,
    config: .init(mode: .voicePipeline),
    onEvent: { event in print(event) }
)
pipeline.start()
pipeline.pushAudio(micSamples)
```

`VoicePipeline` is the real-time voice-agent state machine (powered by [speech-core](https://github.com/soniqo/speech-core)) with VAD-driven turn detection, interruption handling, and eager STT. It connects any `SpeechRecognitionModel` + `SpeechGenerationModel` + `StreamingVADProvider`.

### HTTP API server

```bash
audio-server --port 8080
```

Exposes every model via HTTP REST + WebSocket endpoints, including an OpenAI Realtime API-compatible WebSocket at `/v1/realtime`. See [`Sources/AudioServer/`](Sources/AudioServer/).

## Architecture

speech-swift is split into one SPM target per model so consumers only pay for what they import. Shared infrastructure lives in `AudioCommon` (protocols, audio I/O, HuggingFace downloader, `SentencePieceModel`) and `MLXCommon` (weight loading, `QuantizedLinear` helpers, `SDPA` multi-head attention helper).

**[Full architecture diagram with backends, memory tables, and module map → soniqo.audio/architecture](https://soniqo.audio/architecture)** · **[API reference → soniqo.audio/api](https://soniqo.audio/api)** · **[Benchmarks → soniqo.audio/benchmarks](https://soniqo.audio/benchmarks)**

Local docs (repo):
- **Models:** [Qwen3-ASR](docs/models/asr-model.md) · [Qwen3-TTS](docs/models/tts-model.md) · [CosyVoice](docs/models/cosyvoice-tts.md) · [Kokoro](docs/models/kokoro-tts.md) · [Parakeet TDT](docs/models/parakeet-asr.md) · [Parakeet Streaming](docs/models/parakeet-streaming-asr.md) · [Nemotron Streaming](docs/models/nemotron-streaming.md) · [Omnilingual ASR](docs/models/omnilingual-asr.md) · [PersonaPlex](docs/models/personaplex.md) · [FireRedVAD](docs/models/fireredvad.md) · [Source Separation](docs/models/source-separation.md)
- **Inference:** [Qwen3-ASR](docs/inference/qwen3-asr-inference.md) · [Parakeet TDT](docs/inference/parakeet-asr-inference.md) · [Parakeet Streaming](docs/inference/parakeet-streaming-asr-inference.md) · [Nemotron Streaming](docs/inference/nemotron-streaming-inference.md) · [Omnilingual ASR](docs/inference/omnilingual-asr-inference.md) · [TTS](docs/inference/qwen3-tts-inference.md) · [Forced Aligner](docs/inference/forced-aligner.md) · [Silero VAD](docs/inference/silero-vad.md) · [Speaker Diarization](docs/inference/speaker-diarization.md) · [Speech Enhancement](docs/inference/speech-enhancement.md)
- **Reference:** [Shared Protocols](docs/shared-protocols.md)

## Cache configuration

Model weights download from HuggingFace on first use and cache to `~/Library/Caches/qwen3-speech/`. Override with `QWEN3_CACHE_DIR` (CLI) or `cacheDir:` (Swift API). All `fromPretrained()` entry points also accept `offlineMode: true` to skip network when weights are already cached.

See [`docs/inference/cache-and-offline.md`](docs/inference/cache-and-offline.md) for full details including sandboxed iOS container paths.

## MLX Metal library

If you see `Failed to load the default metallib` at runtime, the Metal shader library is missing. Run `make build` or `./scripts/build_mlx_metallib.sh release` after a manual `swift build`. If the Metal Toolchain is missing, install it first:

```bash
xcodebuild -downloadComponent MetalToolchain
```

## Testing

```bash
make test                            # full suite (unit + E2E with model downloads)
swift test --skip E2E                # unit only (CI-safe, no downloads)
swift test --filter Qwen3ASRTests    # specific module
```

E2E test classes use the `E2E` prefix so CI can filter them out with `--skip E2E`. See [CLAUDE.md](CLAUDE.md#testing) for the full testing convention.

## Contributing

PRs welcome — bug fixes, new model integrations, documentation. Fork, create a feature branch, `make build && make test`, open a PR against `main`.

## License

Apache 2.0
