# Speech Swift

KI-Sprachmodelle für Apple Silicon, basierend auf MLX Swift und CoreML.

📖 Read in: [English](README.md) · [中文](README_zh.md) · [日本語](README_ja.md) · [한국어](README_ko.md) · [Español](README_es.md) · [Deutsch](README_de.md) · [Français](README_fr.md) · [हिन्दी](README_hi.md) · [Português](README_pt.md) · [Русский](README_ru.md)

Spracherkennung, -synthese und -verständnis auf dem Gerät für Mac und iOS. Läuft vollständig lokal auf Apple Silicon — keine Cloud, keine API-Schlüssel, keine Daten verlassen das Gerät.

**[📚 Vollständige Dokumentation →](https://soniqo.audio/de)** · **[🤗 HuggingFace-Modelle](https://huggingface.co/aufklarer)** · **[📝 Blog](https://blog.ivan.digital)**

- **[Qwen3-ASR](https://soniqo.audio/de/guides/transcribe)** — Sprache-zu-Text (automatische Spracherkennung, 52 Sprachen, MLX + CoreML)
- **[Parakeet TDT](https://soniqo.audio/de/guides/parakeet)** — Sprache-zu-Text über CoreML (Neural Engine, NVIDIA FastConformer + TDT-Decoder, 25 Sprachen)
- **[Omnilingual ASR](https://soniqo.audio/de/guides/omnilingual)** — Sprache-zu-Text (Meta wav2vec2 + CTC, **1.672 Sprachen** in 32 Schriften, CoreML 300M + MLX 300M/1B/3B/7B)
- **[Streaming-Diktat](https://soniqo.audio/de/guides/dictate)** — Echtzeit-Diktat mit Teilergebnissen und Äußerungsende-Erkennung (Parakeet-EOU-120M)
- **[Qwen3-ForcedAligner](https://soniqo.audio/de/guides/align)** — Wortgenaue Zeitstempel-Zuordnung (Audio + Text → Zeitstempel)
- **[Qwen3-TTS](https://soniqo.audio/de/guides/speak)** — Sprachsynthese (höchste Qualität, Streaming, benutzerdefinierte Sprecher, 10 Sprachen)
- **[CosyVoice TTS](https://soniqo.audio/de/guides/cosyvoice)** — Streaming-TTS mit Stimmklonen, Mehrsprecherdialog, Emotions-Tags (9 Sprachen)
- **[Kokoro TTS](https://soniqo.audio/de/guides/kokoro)** — TTS auf dem Gerät (82M, CoreML/Neural Engine, 54 Stimmen, iOS-tauglich, 10 Sprachen)
- **[Qwen3.5-Chat](https://soniqo.audio/de/guides/chat)** — LLM-Chat auf dem Gerät (0.8B, MLX INT4 + CoreML INT8, DeltaNet-Hybrid, Token-Streaming)
- **[PersonaPlex](https://soniqo.audio/de/guides/respond)** — Vollduplex-Sprache-zu-Sprache (7B, Audio rein → Audio raus, 18 Stimmvoreinstellungen)
- **[DeepFilterNet3](https://soniqo.audio/de/guides/denoise)** — Echtzeit-Rauschunterdrückung (2,1M Parameter, 48 kHz)
- **[Wake-Word](https://soniqo.audio/de/guides/wake-word)** — Schlüsselworterkennung auf dem Gerät (KWS Zipformer 3M, CoreML, 26× Echtzeit, konfigurierbare Stichwortliste)
- **[VAD](https://soniqo.audio/de/guides/vad)** — Sprachaktivitätserkennung (Silero Streaming, Pyannote Offline, FireRedVAD 100+ Sprachen)
- **[Sprecherdiarisierung](https://soniqo.audio/de/guides/diarize)** — Wer hat wann gesprochen (Pyannote-Pipeline, durchgängiger Sortformer auf der Neural Engine)
- **[Sprechereinbettungen](https://soniqo.audio/de/guides/embed-speaker)** — WeSpeaker ResNet34 (256-dim), CAM++ (192-dim)

Paper: [Qwen3-ASR](https://arxiv.org/abs/2601.21337) (Alibaba) · [Qwen3-TTS](https://arxiv.org/abs/2601.15621) (Alibaba) · [Omnilingual ASR](https://arxiv.org/abs/2511.09690) (Meta) · [Parakeet TDT](https://arxiv.org/abs/2304.06795) (NVIDIA) · [CosyVoice 3](https://arxiv.org/abs/2505.17589) (Alibaba) · [Kokoro](https://arxiv.org/abs/2301.01695) (StyleTTS 2) · [PersonaPlex](https://arxiv.org/abs/2602.06053) (NVIDIA) · [Mimi](https://arxiv.org/abs/2410.00037) (Kyutai) · [Sortformer](https://arxiv.org/abs/2409.06656) (NVIDIA)

## Neuigkeiten

- **20. März 2026** — [Wir schlagen Whisper Large v3 mit einem 600M-Modell, das vollständig auf deinem Mac läuft](https://blog.ivan.digital/we-beat-whisper-large-v3-with-a-600m-model-running-entirely-on-your-mac-20e6ce191174)
- **26. Feb. 2026** — [Sprecherdiarisierung und Sprachaktivitätserkennung auf Apple Silicon — natives Swift mit MLX](https://blog.ivan.digital/speaker-diarization-and-voice-activity-detection-on-apple-silicon-native-swift-with-mlx-92ea0c9aca0f)
- **23. Feb. 2026** — [NVIDIA PersonaPlex 7B auf Apple Silicon — Vollduplex-Sprache-zu-Sprache in nativem Swift mit MLX](https://blog.ivan.digital/nvidia-personaplex-7b-on-apple-silicon-full-duplex-speech-to-speech-in-native-swift-with-mlx-0aa5276f2e23)
- **12. Feb. 2026** — [Qwen3-ASR Swift: ASR + TTS auf dem Gerät für Apple Silicon — Architektur und Benchmarks](https://blog.ivan.digital/qwen3-asr-swift-on-device-asr-tts-for-apple-silicon-architecture-and-benchmarks-27cbf1e4463f)

## Schnellstart

Füge das Paket zu deiner `Package.swift` hinzu:

```swift
.package(url: "https://github.com/soniqo/speech-swift", from: "0.0.9")
```

Importiere nur die Module, die du benötigst — jedes Modell ist eine eigene SPM-Bibliothek, du zahlst nicht für das, was du nicht nutzt:

```swift
.product(name: "ParakeetStreamingASR", package: "speech-swift"),
.product(name: "SpeechUI",             package: "speech-swift"),  // optionale SwiftUI-Views
```

**Audio-Puffer in 3 Zeilen transkribieren:**

```swift
import ParakeetStreamingASR

let model = try await ParakeetStreamingASRModel.fromPretrained()
let text = try model.transcribeAudio(audioSamples, sampleRate: 16000)
```

**Live-Streaming mit Teilergebnissen:**

```swift
for await partial in model.transcribeStream(audio: samples, sampleRate: 16000) {
    print(partial.isFinal ? "FINAL: \(partial.text)" : "... \(partial.text)")
}
```

**SwiftUI-Diktat-View in ~10 Zeilen:**

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

`SpeechUI` liefert nur `TranscriptionView` (finale + partielle Ergebnisse) und `TranscriptionStore` (Streaming-ASR-Adapter). Verwende AVFoundation für Audio-Visualisierung und Wiedergabe.

Verfügbare SPM-Produkte: `Qwen3ASR`, `Qwen3TTS`, `Qwen3TTSCoreML`, `ParakeetASR`, `ParakeetStreamingASR`, `OmnilingualASR`, `KokoroTTS`, `CosyVoiceTTS`, `PersonaPlex`, `SpeechVAD`, `SpeechEnhancement`, `Qwen3Chat`, `SpeechCore`, `SpeechUI`, `AudioCommon`.

## Modelle

Kompakte Übersicht unten. **[Vollständiger Modellkatalog mit Größen, Quantisierungen, Download-URLs und Speichertabellen → soniqo.audio/architecture](https://soniqo.audio/de/architecture)**.

| Modell | Aufgabe | Backends | Größen | Sprachen |
|-------|------|----------|-------|-----------|
| [Qwen3-ASR](https://soniqo.audio/de/guides/transcribe) | Sprache → Text | MLX, CoreML (hybrid) | 0.6B, 1.7B | 52 |
| [Parakeet TDT](https://soniqo.audio/de/guides/parakeet) | Sprache → Text | CoreML (ANE) | 0.6B | 25 europäisch |
| [Parakeet EOU](https://soniqo.audio/de/guides/dictate) | Sprache → Text (Streaming) | CoreML (ANE) | 120M | 25 europäisch |
| [Omnilingual ASR](https://soniqo.audio/de/guides/omnilingual) | Sprache → Text | CoreML (ANE), MLX | 300M / 1B / 3B / 7B | **[1.672](https://github.com/facebookresearch/omnilingual-asr/blob/main/src/omnilingual_asr/models/wav2vec2_llama/lang_ids.py)** |
| [Qwen3-ForcedAligner](https://soniqo.audio/de/guides/align) | Audio + Text → Zeitstempel | MLX, CoreML | 0.6B | Multi |
| [Qwen3-TTS](https://soniqo.audio/de/guides/speak) | Text → Sprache | MLX, CoreML | 0.6B, 1.7B | 10 |
| [CosyVoice3](https://soniqo.audio/de/guides/cosyvoice) | Text → Sprache | MLX | 0.5B | 9 |
| [Kokoro-82M](https://soniqo.audio/de/guides/kokoro) | Text → Sprache | CoreML (ANE) | 82M | 10 |
| [Qwen3.5-Chat](https://soniqo.audio/de/guides/chat) | Text → Text (LLM) | MLX, CoreML | 0.8B | Multi |
| [PersonaPlex](https://soniqo.audio/de/guides/respond) | Sprache → Sprache | MLX | 7B | EN |
| [Silero VAD](https://soniqo.audio/de/guides/vad) | Sprachaktivitätserkennung | MLX, CoreML | 309K | Sprachunabhängig |
| [Pyannote](https://soniqo.audio/de/guides/diarize) | VAD + Diarisierung | MLX | 1.5M | Sprachunabhängig |
| [Sortformer](https://soniqo.audio/de/guides/diarize) | Diarisierung (E2E) | CoreML (ANE) | — | Sprachunabhängig |
| [DeepFilterNet3](https://soniqo.audio/de/guides/denoise) | Sprachverbesserung | CoreML | 2.1M | Sprachunabhängig |
| [WeSpeaker](https://soniqo.audio/de/guides/embed-speaker) | Sprechereinbettung | MLX, CoreML | 6.6M | Sprachunabhängig |

## Installation

### Homebrew

Erfordert natives ARM-Homebrew (`/opt/homebrew`). Rosetta/x86_64-Homebrew wird nicht unterstützt.

```bash
brew tap soniqo/speech https://github.com/soniqo/speech-swift
brew install speech
```

Dann:

```bash
audio transcribe recording.wav
audio speak "Hello world"
audio respond --input question.wav --transcript
```

**[Vollständige CLI-Referenz →](https://soniqo.audio/de/cli)**

### Swift Package Manager

```swift
dependencies: [
    .package(url: "https://github.com/soniqo/speech-swift", from: "0.0.9")
]
```

Importiere nur, was du brauchst — jedes Modell hat sein eigenes SPM-Target:

```swift
import Qwen3ASR             // Spracherkennung (MLX)
import ParakeetASR          // Spracherkennung (CoreML, Batch)
import ParakeetStreamingASR // Streaming-Diktat mit Teilergebnissen + EOU
import OmnilingualASR       // 1.672 Sprachen (CoreML + MLX)
import Qwen3TTS             // Sprachsynthese
import CosyVoiceTTS         // Sprachsynthese mit Stimmklonen
import KokoroTTS            // Sprachsynthese (iOS-tauglich)
import Qwen3Chat            // LLM-Chat auf dem Gerät
import PersonaPlex          // Vollduplex-Sprache-zu-Sprache
import SpeechVAD            // VAD + Sprecherdiarisierung + Einbettungen
import SpeechEnhancement    // Rauschunterdrückung
import SpeechUI             // SwiftUI-Komponenten für Streaming-Transkripte
import AudioCommon          // Geteilte Protokolle und Utilities
```

### Voraussetzungen

- Swift 5.9+, Xcode 15+ (mit Metal Toolchain)
- macOS 14+ oder iOS 17+, Apple Silicon (M1/M2/M3/M4)

### Aus dem Quellcode bauen

```bash
git clone https://github.com/soniqo/speech-swift
cd speech-swift
make build
```

`make build` kompiliert das Swift-Paket **und** die MLX-Metal-Shader-Bibliothek. Die Metal-Bibliothek ist für GPU-Inferenz erforderlich — ohne sie siehst du zur Laufzeit `Failed to load the default metallib`. `make debug` für Debug-Builds, `make test` für die Test-Suite.

**[Vollständige Build- und Installationsanleitung →](https://soniqo.audio/de/getting-started)**

## Demo-Apps

- **[DictateDemo](Examples/DictateDemo/)** ([Docs](https://soniqo.audio/de/guides/dictate)) — macOS-Menüleisten-Streaming-Diktat mit Live-Teilergebnissen, VAD-basierter Äußerungsende-Erkennung und Ein-Klick-Kopieren. Läuft als Hintergrund-agent (Parakeet-EOU-120M + Silero VAD).
- **[iOSEchoDemo](Examples/iOSEchoDemo/)** — iOS-Echo-Demo (Parakeet ASR + Kokoro TTS). Gerät und Simulator.
- **[PersonaPlexDemo](Examples/PersonaPlexDemo/)** — Konversationeller Sprachassistent mit Mikrofoneingang, VAD und Multi-Turn-Kontext. macOS. RTF ~0.94 auf M2 Max (schneller als Echtzeit).
- **[SpeechDemo](Examples/SpeechDemo/)** — Diktat und TTS-Synthese in einer Tab-Oberfläche. macOS.

Die README jedes Demos enthält Bauanleitungen.

## Codebeispiele

Die folgenden Snippets zeigen den minimalen Pfad für jede Domäne. Jeder Abschnitt verlinkt auf eine vollständige Anleitung auf [soniqo.audio](https://soniqo.audio/de) mit Konfigurationsoptionen, mehreren Backends, Streaming-Mustern und CLI-Rezepten.

### Sprache-zu-Text — [vollständige Anleitung →](https://soniqo.audio/de/guides/transcribe)

```swift
import Qwen3ASR

let model = try await Qwen3ASRModel.fromPretrained()
let text = model.transcribe(audio: audioSamples, sampleRate: 16000)
```

Alternative Backends: [Parakeet TDT](https://soniqo.audio/de/guides/parakeet) (CoreML, 32× Echtzeit), [Omnilingual ASR](https://soniqo.audio/de/guides/omnilingual) (1.672 Sprachen, CoreML oder MLX), [Streaming-Diktat](https://soniqo.audio/de/guides/dictate) (Live-Teilergebnisse).

### Forced Alignment — [vollständige Anleitung →](https://soniqo.audio/de/guides/align)

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

### Text-zu-Sprache — [vollständige Anleitung →](https://soniqo.audio/de/guides/speak)

```swift
import Qwen3TTS
import AudioCommon

let model = try await Qwen3TTSModel.fromPretrained()
let audio = model.synthesize(text: "Hello world", language: "english")
try WAVWriter.write(samples: audio, sampleRate: 24000, to: outputURL)
```

Alternative TTS-Engines: [CosyVoice3](https://soniqo.audio/de/guides/cosyvoice) (Streaming + Stimmklonen + Emotions-Tags), [Kokoro-82M](https://soniqo.audio/de/guides/kokoro) (iOS-tauglich, 54 Stimmen), [Stimmklonen](https://soniqo.audio/de/guides/voice-cloning).

### Sprache-zu-Sprache — [vollständige Anleitung →](https://soniqo.audio/de/guides/respond)

```swift
import PersonaPlex

let model = try await PersonaPlexModel.fromPretrained()
let responseAudio = model.respond(userAudio: userSamples)
// 24 kHz Mono Float32-Ausgabe, bereit zur Wiedergabe
```

### LLM-Chat — [vollständige Anleitung →](https://soniqo.audio/de/guides/chat)

```swift
import Qwen3Chat

let chat = try await Qwen35MLXChat.fromPretrained()
chat.chat(messages: [(.user, "Explain MLX in one sentence")]) { token, isFinal in
    print(token, terminator: "")
}
```

### Sprachaktivitätserkennung — [vollständige Anleitung →](https://soniqo.audio/de/guides/vad)

```swift
import SpeechVAD

let vad = try await SileroVADModel.fromPretrained()
let segments = vad.detectSpeech(audio: samples, sampleRate: 16000)
for s in segments { print("\(s.startTime)s → \(s.endTime)s") }
```

### Sprecherdiarisierung — [vollständige Anleitung →](https://soniqo.audio/de/guides/diarize)

```swift
import SpeechVAD

let diarizer = try await DiarizationPipeline.fromPretrained()
let segments = diarizer.diarize(audio: samples, sampleRate: 16000)
for s in segments { print("Speaker \(s.speakerId): \(s.startTime)s - \(s.endTime)s") }
```

### Sprachverbesserung — [vollständige Anleitung →](https://soniqo.audio/de/guides/denoise)

```swift
import SpeechEnhancement

let denoiser = try await DeepFilterNet3Model.fromPretrained()
let clean = try denoiser.enhance(audio: noisySamples, sampleRate: 48000)
```

### Voice Pipeline (ASR → LLM → TTS) — [vollständige Anleitung →](https://soniqo.audio/de/api)

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

`VoicePipeline` ist die Echtzeit-Voice-agent-Zustandsmaschine (angetrieben von [speech-core](https://github.com/soniqo/speech-core)) mit VAD-basierter Sprecherwechsel-Erkennung, Unterbrechungsbehandlung und eager STT. Sie verbindet beliebige `SpeechRecognitionModel` + `SpeechGenerationModel` + `StreamingVADProvider`.

### HTTP-API-Server

```bash
audio-server --port 8080
```

Stellt jedes Modell über HTTP-REST- + WebSocket-Endpunkte bereit, einschließlich eines mit OpenAI Realtime API kompatiblen WebSocket unter `/v1/realtime`. Siehe [`Sources/AudioServer/`](Sources/AudioServer/).

## Architektur

speech-swift ist in ein SPM-Target pro Modell aufgeteilt, sodass Konsumenten nur für das bezahlen, was sie importieren. Geteilte Infrastruktur lebt in `AudioCommon` (Protokolle, Audio-I/O, HuggingFace-Downloader, `SentencePieceModel`) und `MLXCommon` (Gewichtsladen, `QuantizedLinear`-Helfer, `SDPA`-Multi-Head-Attention-Helfer).

**[Vollständiges Architekturdiagramm mit Backends, Speichertabellen und Modulkarte → soniqo.audio/architecture](https://soniqo.audio/de/architecture)** · **[API-Referenz → soniqo.audio/api](https://soniqo.audio/de/api)** · **[Benchmarks → soniqo.audio/benchmarks](https://soniqo.audio/de/benchmarks)**

Lokale Docs (Repo):
- **Modelle:** [Qwen3-ASR](docs/models/asr-model.md) · [Qwen3-TTS](docs/models/tts-model.md) · [CosyVoice](docs/models/cosyvoice-tts.md) · [Kokoro](docs/models/kokoro-tts.md) · [Parakeet TDT](docs/models/parakeet-asr.md) · [Parakeet Streaming](docs/models/parakeet-streaming-asr.md) · [Omnilingual ASR](docs/models/omnilingual-asr.md) · [PersonaPlex](docs/models/personaplex.md) · [FireRedVAD](docs/models/fireredvad.md)
- **Inferenz:** [Qwen3-ASR](docs/inference/qwen3-asr-inference.md) · [Parakeet TDT](docs/inference/parakeet-asr-inference.md) · [Parakeet Streaming](docs/inference/parakeet-streaming-asr-inference.md) · [Omnilingual ASR](docs/inference/omnilingual-asr-inference.md) · [TTS](docs/inference/qwen3-tts-inference.md) · [Forced Aligner](docs/inference/forced-aligner.md) · [Silero VAD](docs/inference/silero-vad.md) · [Sprecherdiarisierung](docs/inference/speaker-diarization.md) · [Sprachverbesserung](docs/inference/speech-enhancement.md)
- **Referenz:** [Geteilte Protokolle](docs/shared-protocols.md)

## Cache-Konfiguration

Modellgewichte werden beim ersten Gebrauch von HuggingFace heruntergeladen und in `~/Library/Caches/qwen3-speech/` zwischengespeichert. Überschreibe mit `QWEN3_CACHE_DIR` (CLI) oder `cacheDir:` (Swift-API). Alle `fromPretrained()`-Einstiegspunkte akzeptieren `offlineMode: true`, um das Netzwerk zu überspringen, wenn die Gewichte bereits im Cache sind.

Siehe [`docs/inference/cache-and-offline.md`](docs/inference/cache-and-offline.md) für vollständige Details einschließlich sandboxed iOS-Container-Pfade.

## MLX-Metal-Bibliothek

Wenn du zur Laufzeit `Failed to load the default metallib` siehst, fehlt die Metal-Shader-Bibliothek. Führe nach einem manuellen `swift build` `make build` oder `./scripts/build_mlx_metallib.sh release` aus. Falls das Metal Toolchain fehlt, installiere es zuerst:

```bash
xcodebuild -downloadComponent MetalToolchain
```

## Tests

```bash
make test                            # Vollständige Suite (Unit + E2E mit Modell-Downloads)
swift test --skip E2E                # Nur Unit (CI-sicher, keine Downloads)
swift test --filter Qwen3ASRTests    # Bestimmtes Modul
```

E2E-Testklassen verwenden das Präfix `E2E`, damit CI sie mit `--skip E2E` ausfiltern kann. Siehe [CLAUDE.md](CLAUDE.md#testing) für die vollständige Testkonvention.

## Mitwirken

PRs willkommen — Bugfixes, neue Modellintegrationen, Dokumentation. Fork, Feature-Branch anlegen, `make build && make test`, PR gegen `main` eröffnen.

## Lizenz

Apache 2.0
