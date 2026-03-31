# Speech Swift

KI-Sprachmodelle für Apple Silicon, basierend auf MLX Swift und CoreML.

📖 Read in: [English](README.md) · [中文](README_zh.md) · [日本語](README_ja.md) · [한국어](README_ko.md) · [Español](README_es.md) · [Deutsch](README_de.md) · [Français](README_fr.md) · [हिन्दी](README_hi.md) · [Português](README_pt.md) · [Русский](README_ru.md)

Spracherkennung, -synthese und -verständnis auf dem Gerät für Mac und iOS. Läuft vollständig lokal auf Apple Silicon — keine Cloud, keine API-Schlüssel, keine Daten verlassen das Gerät.

[Installation über Homebrew](#homebrew) oder als Swift-Package-Abhängigkeit.

**[Dokumentation](https://soniqo.audio)** · **[HuggingFace-Modelle](https://huggingface.co/aufklarer)** · **[Blog](https://blog.ivan.digital)**

- **Qwen3-ASR** — Sprache-zu-Text / Spracherkennung (automatische Spracherkennung, 52 Sprachen)
- **Parakeet TDT** — Sprache-zu-Text über CoreML (Neural Engine, NVIDIA FastConformer + TDT-Decoder, 25 Sprachen)
- **Qwen3-ForcedAligner** — Wortgenaue Zeitstempel-Zuordnung (Audio + Text → Zeitstempel)
- **Qwen3-TTS** — Sprachsynthese (höchste Qualität, Streaming, benutzerdefinierte Sprecher, 10 Sprachen)
- **CosyVoice TTS** — Sprachsynthese mit Streaming, Stimmklonen, Mehrsprecherdialog und Emotions-Tags (9 Sprachen, DiT Flow Matching, CAM++ Sprecherencoder)
- **Kokoro TTS** — Sprachsynthese auf dem Gerät (82M Parameter, CoreML/Neural Engine, 54 Stimmen, iOS-tauglich, 10 Sprachen)
- **Qwen3.5-Chat** — LLM-Chat auf dem Gerät (0.8B, MLX + CoreML, INT4 + CoreML INT8, DeltaNet-Hybrid, Token-Streaming)
- **PersonaPlex** — Vollduplex-Sprache-zu-Sprache-Konversation (7B, Audio rein → Audio raus, 18 Stimmvoreinstellungen)
- **DeepFilterNet3** — Sprachverbesserung / Rauschunterdrückung (2,1M Parameter, Echtzeit 48kHz)
- **FireRedVAD** — Offline-Sprachaktivitätserkennung (DFSMN, CoreML, 100+ Sprachen, 97,6% F1)
- **Silero VAD** — Streaming-Sprachaktivitätserkennung (32ms-Blöcke, Latenz unter einer Millisekunde)
- **Pyannote VAD** — Offline-Sprachaktivitätserkennung (10s-Fenster, Mehrsprecher-Überlappung)
- **Sprecherdiarisierung** — Wer hat wann gesprochen (Pyannote-Segmentierung + aktivitätsbasierte Sprecherzuordnung, oder durchgängiger Sortformer auf der Neural Engine)
- **Sprechereinbettungen** — Sprecherverifizierung und -identifikation (WeSpeaker ResNet34, 256-dimensionale Vektoren)

Paper: [Qwen3-ASR](https://arxiv.org/abs/2601.21337) (Alibaba), [Qwen3-TTS](https://arxiv.org/abs/2601.15621) (Alibaba), [Qwen3](https://arxiv.org/abs/2505.09388) (Alibaba), [Parakeet TDT](https://arxiv.org/abs/2304.06795) (NVIDIA), [CosyVoice 3](https://arxiv.org/abs/2505.17589) (Alibaba), [Kokoro](https://arxiv.org/abs/2301.01695) (StyleTTS 2), [PersonaPlex](https://arxiv.org/abs/2602.06053) (NVIDIA), [Mimi](https://arxiv.org/abs/2410.00037) (Kyutai), [Sortformer](https://arxiv.org/abs/2409.06656) (NVIDIA)

## Roadmap

Siehe [Roadmap-Diskussion](https://github.com/soniqo/speech-swift/discussions/81) für geplante Funktionen — Kommentare und Vorschläge sind willkommen!

## Neuigkeiten

- **20. März 2026** — [We Beat Whisper Large v3 with a 600M Model Running Entirely on Your Mac](https://blog.ivan.digital/we-beat-whisper-large-v3-with-a-600m-model-running-entirely-on-your-mac-20e6ce191174)
- **26. Feb. 2026** — [Speaker Diarization and Voice Activity Detection on Apple Silicon — Native Swift with MLX](https://blog.ivan.digital/speaker-diarization-and-voice-activity-detection-on-apple-silicon-native-swift-with-mlx-92ea0c9aca0f)
- **23. Feb. 2026** — [NVIDIA PersonaPlex 7B on Apple Silicon — Full-Duplex Speech-to-Speech in Native Swift with MLX](https://blog.ivan.digital/nvidia-personaplex-7b-on-apple-silicon-full-duplex-speech-to-speech-in-native-swift-with-mlx-0aa5276f2e23)
- **12. Feb. 2026** — [Qwen3-ASR Swift: On-Device ASR + TTS for Apple Silicon — Architecture and Benchmarks](https://blog.ivan.digital/qwen3-asr-swift-on-device-asr-tts-for-apple-silicon-architecture-and-benchmarks-27cbf1e4463f)

## Modelle

| Modell | Aufgabe | Streaming | Sprachen | Größen |
|--------|---------|-----------|----------|--------|
| Qwen3-ASR-0.6B | Sprache → Text | Nein | 52 Sprachen | [4-bit](https://huggingface.co/aufklarer/Qwen3-ASR-0.6B-MLX-4bit) 680 MB · [8-bit](https://huggingface.co/aufklarer/Qwen3-ASR-0.6B-MLX-8bit) 1.0 GB · [CoreML](https://huggingface.co/aufklarer/Qwen3-ASR-CoreML) 180 MB |
| Qwen3-ASR-1.7B | Sprache → Text | Nein | 52 Sprachen | [4-bit](https://huggingface.co/aufklarer/Qwen3-ASR-1.7B-MLX-4bit) 2.1 GB · [8-bit](https://huggingface.co/aufklarer/Qwen3-ASR-1.7B-MLX-8bit) 3.2 GB |
| Parakeet-TDT-0.6B | Sprache → Text | Nein | 25 europäische Sprachen | [CoreML INT4](https://huggingface.co/aufklarer/Parakeet-TDT-v3-CoreML-INT4) 315 MB · [CoreML INT8](https://huggingface.co/aufklarer/Parakeet-TDT-v3-CoreML-INT8) 500 MB |
| Qwen3-ForcedAligner-0.6B | Audio + Text → Zeitstempel | Nein | Mehrsprachig | [4-bit](https://huggingface.co/aufklarer/Qwen3-ForcedAligner-0.6B-4bit) 979 MB · [8-bit](https://huggingface.co/aufklarer/Qwen3-ForcedAligner-0.6B-8bit) 1.4 GB · [CoreML INT4](https://huggingface.co/aufklarer/Qwen3-ForcedAligner-0.6B-CoreML-INT4) 630 MB · [CoreML INT8](https://huggingface.co/aufklarer/Qwen3-ForcedAligner-0.6B-CoreML-INT8) 1.0 GB |
| Qwen3-TTS-0.6B Base | Text → Sprache | Ja (~120ms) | 10 Sprachen | [4-bit](https://huggingface.co/aufklarer/Qwen3-TTS-12Hz-0.6B-Base-MLX-4bit) 1.7 GB · [8-bit](https://huggingface.co/aufklarer/Qwen3-TTS-12Hz-0.6B-Base-MLX-8bit) 2.4 GB |
| Qwen3-TTS-0.6B CustomVoice | Text → Sprache | Ja (~120ms) | 10 Sprachen | [4-bit](https://huggingface.co/aufklarer/Qwen3-TTS-12Hz-0.6B-CustomVoice-MLX-4bit) 1.7 GB |
| Qwen3-TTS-1.7B Base | Text → Sprache | Ja (~120ms) | 10 Sprachen | [4-bit](https://huggingface.co/aufklarer/Qwen3-TTS-12Hz-1.7B-Base-MLX-4bit) 3.2 GB · [8-bit](https://huggingface.co/aufklarer/Qwen3-TTS-12Hz-1.7B-Base-MLX-8bit) 4.8 GB |
| CosyVoice3-0.5B | Text → Sprache | Ja (~150ms) | 9 Sprachen | [4-bit](https://huggingface.co/aufklarer/CosyVoice3-0.5B-MLX-4bit) 1.2 GB |
| Kokoro-82M | Text → Sprache | Nein | 10 Sprachen | [CoreML](https://huggingface.co/aufklarer/Kokoro-82M-CoreML) ~170 MB |
| Qwen3.5-0.8B Chat | Text → Text (LLM) | Yes (streaming) | Multi | [MLX INT4](https://huggingface.co/aufklarer/Qwen3.5-0.8B-Chat-MLX) 418 MB · [CoreML INT8](https://huggingface.co/aufklarer/Qwen3.5-0.8B-Chat-CoreML) 981 MB |
| PersonaPlex-7B | Sprache → Sprache | Ja (~2s-Blöcke) | EN | [4-bit](https://huggingface.co/aufklarer/PersonaPlex-7B-MLX-4bit) 4.9 GB · [8-bit](https://huggingface.co/aufklarer/PersonaPlex-7B-MLX-8bit) 9.1 GB |
| FireRedVAD | Sprachaktivitätserkennung | Nein (offline) | 100+ Sprachen | [CoreML](https://huggingface.co/aufklarer/FireRedVAD-CoreML) ~1.2 MB |
| Silero-VAD-v5 | Sprachaktivitätserkennung | Ja (32ms-Blöcke) | Sprachunabhängig | [MLX](https://huggingface.co/aufklarer/Silero-VAD-v5-MLX) · [CoreML](https://huggingface.co/aufklarer/Silero-VAD-v5-CoreML) ~1.2 MB |
| Pyannote-Segmentation-3.0 | VAD + Sprechersegmentierung | Nein (10s-Fenster) | Sprachunabhängig | [MLX](https://huggingface.co/aufklarer/Pyannote-Segmentation-MLX) ~5.7 MB |
| DeepFilterNet3 | Sprachverbesserung | Ja (10ms-Frames) | Sprachunabhängig | [CoreML FP16](https://huggingface.co/aufklarer/DeepFilterNet3-CoreML) ~4.2 MB |
| WeSpeaker-ResNet34-LM | Sprechereinbettung (256-dim) | Nein | Sprachunabhängig | [MLX](https://huggingface.co/aufklarer/WeSpeaker-ResNet34-LM-MLX) · [CoreML](https://huggingface.co/aufklarer/WeSpeaker-ResNet34-LM-CoreML) ~25 MB |
| CAM++ | Sprechereinbettung (192-dim) | Nein | Sprachunabhängig | [CoreML](https://huggingface.co/aufklarer/CamPlusPlus-Speaker-CoreML) ~14 MB |
| Sortformer | Sprecherdiarisierung (durchgängig) | Ja (blockweise) | Sprachunabhängig | [CoreML](https://huggingface.co/aufklarer/Sortformer-Diarization-CoreML) ~240 MB |

### Speicheranforderungen

Gewichtsspeicher ist der GPU- (MLX) oder ANE-Speicher (CoreML), der von Modellparametern belegt wird. Spitzenverbrauch bei Inferenz umfasst KV-Caches, Aktivierungen und Zwischentensoren.

| Modell | Gewichtsspeicher | Spitze bei Inferenz |
|--------|-----------------|---------------------|
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

### Welches TTS-Modell für welchen Einsatz

- **Qwen3-TTS**: Beste Qualität, Streaming (~120ms), 9 eingebaute Sprecher, 10 Sprachen, Stapelverarbeitung
- **CosyVoice TTS**: Streaming (~150ms), 9 Sprachen, Stimmklonen (CAM++ Sprecherencoder), Mehrsprecherdialog (`[S1] ... [S2] ...`), Inline-Emotions-/Stil-Tags (`(happy)`, `(whispers)`), DiT Flow Matching + HiFi-GAN Vocoder
- **Kokoro TTS**: Leichtgewichtige iOS-taugliche TTS (82M Parameter), CoreML/Neural Engine, 54 Stimmen, 10 Sprachen, nicht-autoregressiv (3-stufige Pipeline)
- **PersonaPlex**: Vollduplex-Sprache-zu-Sprache (Audio rein → Audio raus), Streaming (~2s-Blöcke), 18 Stimmvoreinstellungen, basiert auf der Moshi-Architektur

## Installation

### Homebrew

Erfordert natives ARM Homebrew (`/opt/homebrew`). Rosetta/x86_64 Homebrew wird nicht unterstützt.

```bash
brew tap soniqo/speech https://github.com/soniqo/speech-swift
brew install speech
```

Anschließend verwenden:

```bash
audio transcribe recording.wav
audio speak "Hello world"
audio speak "Hallo Welt" --engine cosyvoice --language german
audio respond --input question.wav --transcript
```

> Für interaktive Sprachkonversation mit Mikrofoneingabe siehe **[PersonaPlexDemo](Examples/PersonaPlexDemo/)**.

### Swift Package Manager

Zu `Package.swift` hinzufügen:

```swift
dependencies: [
    .package(url: "https://github.com/soniqo/speech-swift", branch: "main")
]
```

Das benötigte Modul importieren:

```swift
import Qwen3ASR      // Spracherkennung (MLX)
import ParakeetASR   // Spracherkennung (CoreML)
import Qwen3TTS      // Sprachsynthese (Qwen3)
import CosyVoiceTTS  // Sprachsynthese (Streaming)
import KokoroTTS     // Sprachsynthese (CoreML, iOS-tauglich)
import Qwen3Chat     // LLM-Chat auf dem Gerät (CoreML)
import PersonaPlex   // Sprache-zu-Sprache (Vollduplex)
import SpeechVAD          // Sprachaktivitätserkennung (Pyannote + Silero)
import SpeechEnhancement  // Rauschunterdrückung (DeepFilterNet3)
import AudioCommon        // Gemeinsame Hilfsfunktionen
```

### Voraussetzungen

- Swift 5.9+
- macOS 14+ oder iOS 17+
- Apple Silicon (M1/M2/M3/M4)
- Xcode 15+ (mit Metal Toolchain — bei Bedarf `xcodebuild -downloadComponent MetalToolchain` ausführen)

### Aus Quellcode kompilieren

```bash
git clone https://github.com/soniqo/speech-swift
cd speech-swift
make build
```

Dies kompiliert das Swift-Paket **und** die MLX-Metal-Shader-Bibliothek in einem Schritt. Die Metal-Bibliothek (`mlx.metallib`) wird für GPU-Inferenz benötigt — ohne sie erscheint zur Laufzeit der Fehler `Failed to load the default metallib`.

Für Debug-Builds: `make debug`. Um Unit-Tests auszuführen: `make test`.

## Sprachassistent ausprobieren

**[PersonaPlexDemo](Examples/PersonaPlexDemo/)** ist ein sofort lauffähiger macOS-Sprachassistent — tippen zum Sprechen, Antworten in Echtzeit. Nutzt Mikrofoneingabe mit Silero VAD zur automatischen Spracherkennung, Qwen3-ASR für Transkription und PersonaPlex 7B für Sprache-zu-Sprache-Generierung. Mehrrundenkonversation mit 18 Stimmvoreinstellungen und Anzeige des inneren Monologs.

```bash
make build  # im Repo-Stammverzeichnis — kompiliert alles einschließlich MLX metallib
cd Examples/PersonaPlexDemo
# Siehe Examples/PersonaPlexDemo/README.md für Anweisungen zum .app-Bundle
```

> RTF ~0,94 auf M2 Max (schneller als Echtzeit). Modelle werden beim ersten Start automatisch heruntergeladen (~5,5 GB PersonaPlex + ~400 MB ASR).

## Demo-Apps

- **[PersonaPlexDemo](Examples/PersonaPlexDemo/)** — Konversationeller Sprachassistent (Mikrofoneingabe, VAD, Mehrrunden). Siehe oben.
- **[SpeechDemo](Examples/SpeechDemo/)** — Diktat (Parakeet TDT / Qwen3-ASR mit automatischer Spracherkennung) und Sprachsynthese (Qwen3-TTS) in einer Tab-Oberfläche.

Als macOS-`.app`-Bundle kompilieren und ausführen — siehe die README jeder Demo für Anleitungen.

## Sprache-zu-Text (ASR) — Audio in Swift transkribieren

### Einfache Transkription

```swift
import Qwen3ASR

// Standard: 0.6B-Modell
let model = try await Qwen3ASRModel.fromPretrained()

// Oder das größere 1.7B-Modell für bessere Genauigkeit
let model = try await Qwen3ASRModel.fromPretrained(
    modelId: "aufklarer/Qwen3-ASR-1.7B-MLX-8bit"
)

// Audio kann jede Abtastrate haben — wird intern automatisch auf 16kHz resampled
let transcription = model.transcribe(audio: audioSamples, sampleRate: 16000)
print(transcription)
```

### CoreML-Encoder (Neural Engine)

Hybridmodus: CoreML-Encoder auf der Neural Engine + MLX-Textdecoder auf der GPU. Geringerer Stromverbrauch, entlastet die GPU beim Encoder-Durchlauf.

```swift
import Qwen3ASR

let encoder = try await CoreMLASREncoder.fromPretrained()
let model = try await Qwen3ASRModel.fromPretrained()
let text = try model.transcribe(audio: audioSamples, sampleRate: 16000, coremlEncoder: encoder)
```

INT8- (180 MB, Standard) und INT4-Varianten (90 MB) verfügbar. INT8 empfohlen (Kosinusähnlichkeit > 0,999 gegenüber FP32).

### Parakeet TDT (CoreML)

```swift
import ParakeetASR

let model = try await ParakeetASRModel.fromPretrained()
let transcription = model.transcribe(audio: audioSamples, sampleRate: 16000)
```

Läuft auf der Neural Engine über CoreML — hält die GPU für gleichzeitige Aufgaben frei. 25 europäische Sprachen, ~315 MB.

### ASR CLI

```bash
make build  # oder: swift build -c release && ./scripts/build_mlx_metallib.sh release

# Standard (Qwen3-ASR 0.6B, MLX)
.build/release/audio transcribe audio.wav

# 1.7B-Modell verwenden
.build/release/audio transcribe audio.wav --model 1.7B

# CoreML-Encoder (Neural Engine + MLX-Decoder)
.build/release/audio transcribe --engine qwen3-coreml audio.wav

# Parakeet TDT (CoreML, Neural Engine)
.build/release/audio transcribe --engine parakeet audio.wav
```

## Gezwungene Ausrichtung (Forced Alignment)

### Wortgenaue Zeitstempel

```swift
import Qwen3ASR

let aligner = try await Qwen3ForcedAligner.fromPretrained()
// Lädt beim ersten Start ~979 MB herunter

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

# Mit vorgegebenem Text ausrichten
.build/release/audio align audio.wav --text "Hello world"

# Erst transkribieren, dann ausrichten
.build/release/audio align audio.wav
```

Ausgabe:
```
[0.12s - 0.45s] Can
[0.45s - 0.72s] you
[0.72s - 1.20s] guarantee
...
```

Nicht-autoregressiv — 3-stufige Pipeline, keine Sampling-Schleife. Siehe [Forced Aligner](docs/inference/forced-aligner.md) für Architekturdetails.

## Text-zu-Sprache (TTS) — Sprachsynthese in Swift

### Einfache Synthese

```swift
import Qwen3TTS
import AudioCommon  // für WAVWriter

let model = try await Qwen3TTSModel.fromPretrained()
// Lädt beim ersten Start ~1,7 GB herunter (Modell- + Codec-Gewichte)
let audio = model.synthesize(text: "Hello world", language: "english")
// Ausgabe: 24kHz Mono-Float-Samples
try WAVWriter.write(samples: audio, sampleRate: 24000, to: outputURL)
```

### TTS CLI

```bash
make build
.build/release/audio speak "Hello world" --output output.wav --language english
```

### Benutzerdefinierte Stimme / Sprecherauswahl

Die **CustomVoice**-Modellvariante unterstützt 9 eingebaute Sprecherstimmen und natürlichsprachliche Anweisungen zur Steuerung von Ton und Stil. Laden Sie sie durch Angabe der CustomVoice-Modell-ID:

```swift
import Qwen3TTS

// CustomVoice-Modell laden (lädt beim ersten Start ~1,7 GB herunter)
let model = try await Qwen3TTSModel.fromPretrained(
    modelId: TTSModelVariant.customVoice.rawValue
)

// Mit einem bestimmten Sprecher synthetisieren
let audio = model.synthesize(text: "Hello world", language: "english", speaker: "vivian")

// Verfügbare Sprecher auflisten
print(model.availableSpeakers)  // ["aiden", "dylan", "eric", ...]
```

CLI:

```bash
# CustomVoice-Modell mit einem Sprecher verwenden
.build/release/audio speak "Hello world" --model customVoice --speaker vivian --output vivian.wav

# Verfügbare Sprecher auflisten
.build/release/audio speak --model customVoice --list-speakers
```

### Stimmklonen (Base-Modell)

Klonen Sie die Stimme eines Sprechers aus einer Referenzaudiodatei. Zwei Modi:

**ICL-Modus** (empfohlen) — kodiert Referenzaudio mit Transkript in Codec-Tokens. Höhere Qualität, zuverlässiges EOS:

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

**X-Vektor-Modus** — nur Sprechereinbettung, kein Transkript nötig, aber geringere Qualität:

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

### Ton- / Stilanweisungen (nur CustomVoice)

Das CustomVoice-Modell akzeptiert einen natürlichsprachlichen `instruct`-Parameter zur Steuerung von Sprechstil, Ton, Emotion und Tempo. Die Anweisung wird dem Modellinput im ChatML-Format vorangestellt.

```swift
// Fröhlicher Ton
let audio = model.synthesize(
    text: "Welcome to our store!",
    language: "english",
    speaker: "ryan",
    instruct: "Speak in a cheerful, upbeat tone"
)

// Langsam und ernst
let audio = model.synthesize(
    text: "We regret to inform you...",
    language: "english",
    speaker: "aiden",
    instruct: "Read this slowly and solemnly"
)

// Flüstern
let audio = model.synthesize(
    text: "Can you keep a secret?",
    language: "english",
    speaker: "vivian",
    instruct: "Whisper this softly"
)
```

CLI:

```bash
# Mit Stilanweisung
.build/release/audio speak "Good morning!" --model customVoice --speaker ryan \
    --instruct "Speak in a cheerful, upbeat tone" --output cheerful.wav

# Standard-Instruct ("Speak naturally.") wird automatisch angewendet bei CustomVoice
.build/release/audio speak "Hello world" --model customVoice --speaker ryan --output natural.wav
```

Wenn bei dem CustomVoice-Modell kein `--instruct` angegeben wird, wird automatisch `"Speak naturally."` angewendet, um abschweifende Ausgabe zu vermeiden. Das Base-Modell unterstützt kein Instruct.

### Stapelverarbeitung (Batch Synthesis)

Mehrere Texte in einem einzelnen gebündelten Vorwärtsdurchlauf für höheren Durchsatz synthetisieren:

```swift
let texts = ["Good morning everyone.", "The weather is nice today.", "Please open the window."]
let audioList = model.synthesizeBatch(texts: texts, language: "english", maxBatchSize: 4)
// audioList[i] enthält 24kHz-Mono-Float-Samples für texts[i]
for (i, audio) in audioList.enumerated() {
    try WAVWriter.write(samples: audio, sampleRate: 24000, to: URL(fileURLWithPath: "output_\(i).wav"))
}
```

#### Batch CLI

```bash
# Datei mit einem Text pro Zeile erstellen
echo "Hello world.\nGoodbye world." > texts.txt
.build/release/audio speak --batch-file texts.txt --output output.wav --batch-size 4
# Erzeugt output_0.wav, output_1.wav, ...
```

> Der Stapelmodus amortisiert das Laden der Modellgewichte über die Elemente. Erwarten Sie eine ~1,5-2,5-fache Durchsatzverbesserung bei B=4 auf Apple Silicon. Beste Ergebnisse bei Texten mit ähnlicher Audiolänge.

### Sampling-Optionen

```swift
let config = SamplingConfig(temperature: 0.9, topK: 50, repetitionPenalty: 1.05)
let audio = model.synthesize(text: "Hello", language: "english", sampling: config)
```

### Streaming-Synthese

Audioblöcke inkrementell für niedrige First-Packet-Latenz ausgeben:

```swift
let stream = model.synthesizeStream(
    text: "Hello, this is streaming synthesis.",
    language: "english",
    streaming: .lowLatency  // ~120ms bis zum ersten Audioblock
)

for try await chunk in stream {
    // chunk.samples: [Float] PCM @ 24kHz
    // chunk.isFinal: true beim letzten Block
    playAudio(chunk.samples)
}
```

CLI:

```bash
# Standard-Streaming (3-Frame erster Block, ~225ms Latenz)
.build/release/audio speak "Hello world" --stream

# Niedrige Latenz (1-Frame erster Block, ~120ms Latenz)
.build/release/audio speak "Hello world" --stream --first-chunk-frames 1
```

## Sprache-zu-Sprache — Vollduplex-Sprachkonversation

> Für einen interaktiven Sprachassistenten mit Mikrofoneingabe siehe **[PersonaPlexDemo](Examples/PersonaPlexDemo/)** — tippen zum Sprechen, Mehrrundenkonversation mit automatischer Spracherkennung.

### Sprache-zu-Sprache

```swift
import PersonaPlex
import AudioCommon  // für WAVWriter, AudioFileLoader

let model = try await PersonaPlexModel.fromPretrained()
// Lädt beim ersten Start ~5,5 GB herunter (Temporal 4-bit + Depformer + Mimi-Codec + Stimmvoreinstellungen)

let audio = try AudioFileLoader.load(url: inputURL, targetSampleRate: 24000)
let (response, textTokens) = model.respond(userAudio: audio, voice: .NATM0)
// response: 24kHz-Mono-Float-Samples
// textTokens: innerer Monolog des Modells (SentencePiece Token-IDs)
try WAVWriter.write(samples: response.audio, sampleRate: 24000, to: outputURL)
```

### Innerer Monolog (Textausgabe)

PersonaPlex erzeugt parallel zum Audio auch Text-Tokens — das interne Denken des Modells. Dekodieren Sie diese mit dem integrierten SentencePiece-Decoder:

```swift
let decoder = try SentencePieceDecoder(modelPath: "tokenizer_spm_32k_3.model")
let transcript = decoder.decode(textTokens)
print(transcript)  // z.B. "Sure, I can help you with that..."
```

### Streaming Sprache-zu-Sprache

```swift
// Audioblöcke empfangen, während sie erzeugt werden (~2s pro Block)
let stream = model.respondStream(userAudio: audio, voice: .NATM0)
for try await chunk in stream {
    playAudio(chunk.samples)  // sofort abspielen, 24kHz Mono
    // chunk.textTokens enthält den Text dieses Blocks; letzter Block enthält alle Tokens
    if chunk.isFinal { break }
}
```

### Stimmauswahl

18 Stimmvoreinstellungen verfügbar:
- **Natürlich weiblich**: NATF0, NATF1, NATF2, NATF3
- **Natürlich männlich**: NATM0, NATM1, NATM2, NATM3
- **Vielfalt weiblich**: VARF0, VARF1, VARF2, VARF3, VARF4
- **Vielfalt männlich**: VARM0, VARM1, VARM2, VARM3, VARM4

### Systemaufforderungen

Die Systemaufforderung steuert das Gesprächsverhalten des Modells. Sie können eine beliebige benutzerdefinierte Aufforderung als Zeichenkette übergeben:

```swift
// Benutzerdefinierte Systemaufforderung (automatisch tokenisiert)
let response = model.respond(
    userAudio: audio,
    voice: .NATM0,
    systemPrompt: "You enjoy having a good conversation."
)

// Oder eine Voreinstellung verwenden
let response = model.respond(
    userAudio: audio,
    voice: .NATM0,
    systemPromptTokens: SystemPromptPreset.customerService.tokens
)
```

Verfügbare Voreinstellungen: `focused` (Standard), `assistant`, `customerService`, `teacher`.

### PersonaPlex CLI

```bash
make build

# Einfache Sprache-zu-Sprache
.build/release/audio respond --input question.wav --output response.wav

# Mit Transkript (dekodiert inneren Monolog)
.build/release/audio respond --input question.wav --transcript

# JSON-Ausgabe (Audiopfad, Transkript, Latenzmetriken)
.build/release/audio respond --input question.wav --json

# Benutzerdefinierter Systemaufforderungstext
.build/release/audio respond --input question.wav --system-prompt-text "You enjoy having a good conversation."

# Stimme und Systemaufforderungs-Voreinstellung wählen
.build/release/audio respond --input question.wav --voice NATF1 --system-prompt focused

# Sampling-Parameter anpassen
.build/release/audio respond --input question.wav --audio-temp 0.6 --repetition-penalty 1.5

# Text-Entropie-Frühstopp aktivieren (stoppt bei Textkollaps)
.build/release/audio respond --input question.wav --entropy-threshold 1.0 --entropy-window 5

# Verfügbare Stimmen und Aufforderungen auflisten
.build/release/audio respond --list-voices
.build/release/audio respond --list-prompts
```

## CosyVoice TTS — Streaming-Sprachsynthese mit Stimmklonen

### Einfache Synthese

```swift
import CosyVoiceTTS
import AudioCommon  // für WAVWriter

let model = try await CosyVoiceTTSModel.fromPretrained()
// Lädt beim ersten Start ~1,9 GB herunter (LLM + DiT + HiFi-GAN-Gewichte)

let audio = model.synthesize(text: "Hello, how are you today?", language: "english")
// Ausgabe: 24kHz-Mono-Float-Samples
try WAVWriter.write(samples: audio, sampleRate: 24000, to: outputURL)
```

### Streaming-Synthese

```swift
// Streaming: Audioblöcke empfangen, während sie erzeugt werden (~150ms bis zum ersten Block)
for try await chunk in model.synthesizeStream(text: "Hello, how are you today?", language: "english") {
    // chunk.audio: [Float], chunk.sampleRate: Int
    playAudio(chunk.audio)  // sofort abspielen
}
```

### Stimmklonen (CosyVoice)

Klonen Sie eine Sprecherstimme mit dem CAM++ Sprecherencoder (192-dim, CoreML Neural Engine):

```swift
import CosyVoiceTTS
import AudioCommon

let model = try await CosyVoiceTTSModel.fromPretrained()
let speaker = try await CamPlusPlusSpeaker.fromPretrained()
// Lädt beim ersten Gebrauch das ~14 MB CAM++ CoreML-Modell herunter

let refAudio = try AudioFileLoader.load(url: referenceURL, targetSampleRate: 16000)
let embedding = try speaker.embed(audio: refAudio, sampleRate: 16000)
// embedding: [Float] der Länge 192

let audio = model.synthesize(
    text: "Hello in a cloned voice!",
    language: "english",
    speakerEmbedding: embedding
)
```

### CosyVoice TTS CLI

```bash
make build

# Einfache Synthese
.build/release/audio speak "Hello world" --engine cosyvoice --language english --output output.wav

# Stimmklonen (lädt CAM++ Sprecherencoder beim ersten Gebrauch herunter)
.build/release/audio speak "Hello world" --engine cosyvoice --voice-sample reference.wav --output cloned.wav

# Mehrsprecherdialog mit Stimmklonen
.build/release/audio speak "[S1] Hello there! [S2] Hey, how are you?" \
    --engine cosyvoice --speakers s1=alice.wav,s2=bob.wav -o dialogue.wav

# Inline-Emotions-/Stil-Tags
.build/release/audio speak "(excited) Wow, amazing! (sad) But I have to go..." \
    --engine cosyvoice -o emotion.wav

# Kombiniert: Dialog + Emotionen + Stimmklonen
.build/release/audio speak "[S1] (happy) Great news! [S2] (surprised) Really?" \
    --engine cosyvoice --speakers s1=alice.wav,s2=bob.wav -o combined.wav

# Benutzerdefinierte Stilanweisung
.build/release/audio speak "Hello world" --engine cosyvoice --cosy-instruct "Speak cheerfully" -o cheerful.wav

# Streaming-Synthese
.build/release/audio speak "Hello world" --engine cosyvoice --language english --stream --output output.wav
```

## Kokoro TTS — Leichtgewichtige Sprachsynthese auf dem Gerät (iOS + macOS)

### Einfache Synthese

```swift
import KokoroTTS
import AudioCommon  // für WAVWriter

let tts = try await KokoroTTSModel.fromPretrained()
// Lädt beim ersten Start ~170 MB herunter (CoreML-Modelle + Stimmeinbettungen + Wörterbücher)

let audio = try tts.synthesize(text: "Hello world", voice: "af_heart")
// Ausgabe: 24kHz-Mono-Float-Samples
try WAVWriter.write(samples: audio, sampleRate: 24000, to: outputURL)
```

54 voreingestellte Stimmen in 10 Sprachen. Nicht-autoregressiv — 3-stufige CoreML-Pipeline, keine Sampling-Schleife. Läuft auf der Neural Engine, entlastet die GPU vollständig.

### Kokoro TTS CLI

```bash
make build

# Einfache Synthese
.build/release/audio kokoro "Hello world" --voice af_heart --output hello.wav

# Sprache wählen
.build/release/audio kokoro "Bonjour le monde" --voice ff_siwis --language fr --output bonjour.wav

# Verfügbare Stimmen auflisten
.build/release/audio kokoro --list-voices
```

## Qwen3 Chat (LLM auf dem Gerät)

```swift
import Qwen3Chat

let chat = try await Qwen3ChatModel.fromPretrained()
// Lädt beim ersten Start ~318 MB herunter (INT4 CoreML-Modell + Tokenizer)

// Einzelne Antwort
let response = try chat.generate("What is Swift?", systemPrompt: "Answer briefly.")
print(response)

// Token-Streaming
let stream = chat.chatStream("Tell me a joke", systemPrompt: "Be funny.")
for try await token in stream {
    print(token, terminator: "")
}
```

Qwen3-0.6B INT4-quantisiert für CoreML. Läuft auf der Neural Engine mit ~2 Tok/s auf dem iPhone, ~15 Tok/s auf M-Serie. Unterstützt Mehrrunden-Konversation mit KV-Cache, Denkmodus (`<think>`-Tokens) und konfigurierbare Sampling-Parameter (Temperatur, Top-k, Top-p, Wiederholungsstrafe).

## Sprachaktivitätserkennung (VAD) — Sprache in Audio erkennen

### Streaming-VAD (Silero)

Silero VAD v5 verarbeitet 32ms-Audioblöcke mit Latenz unter einer Millisekunde — ideal für Echtzeit-Spracherkennung von Mikrofonen oder Streams.

```swift
import SpeechVAD

let vad = try await SileroVADModel.fromPretrained()
// Oder CoreML verwenden (Neural Engine, geringerer Stromverbrauch):
// let vad = try await SileroVADModel.fromPretrained(engine: .coreml)

// Streaming: 512-Sample-Blöcke verarbeiten (32ms @ 16kHz)
let prob = vad.processChunk(samples)  // → 0.0...1.0
vad.resetState()  // zwischen verschiedenen Audiostreams aufrufen

// Oder alle Segmente auf einmal erkennen
let segments = vad.detectSpeech(audio: audioSamples, sampleRate: 16000)
for seg in segments {
    print("Sprache: \(seg.startTime)s - \(seg.endTime)s")
}
```

### Ereignisgesteuertes Streaming

```swift
let processor = StreamingVADProcessor(model: vad)

// Audio beliebiger Länge einspeisen — Ereignisse werden ausgelöst, wenn Sprache bestätigt wird
let events = processor.process(samples: audioBuffer)
for event in events {
    switch event {
    case .speechStarted(let time):
        print("Sprache begonnen um \(time)s")
    case .speechEnded(let segment):
        print("Sprache: \(segment.startTime)s - \(segment.endTime)s")
    }
}

// Am Ende des Streams leeren
let final = processor.flush()
```

### VAD CLI

```bash
make build

# Streaming Silero VAD (32ms-Blöcke)
.build/release/audio vad-stream audio.wav

# CoreML-Backend (Neural Engine)
.build/release/audio vad-stream audio.wav --engine coreml

# Mit benutzerdefinierten Schwellenwerten
.build/release/audio vad-stream audio.wav --onset 0.6 --offset 0.4

# JSON-Ausgabe
.build/release/audio vad-stream audio.wav --json

# Batch-Pyannote-VAD (10s gleitende Fenster)
.build/release/audio vad audio.wav
```

## Sprecherdiarisierung — Wer hat wann gesprochen

### Diarisierungs-Pipeline

```swift
import SpeechVAD

let pipeline = try await DiarizationPipeline.fromPretrained()
// Oder CoreML-Einbettungen verwenden (Neural Engine, entlastet die GPU):
// let pipeline = try await DiarizationPipeline.fromPretrained(embeddingEngine: .coreml)

let result = pipeline.diarize(audio: samples, sampleRate: 16000)
for seg in result.segments {
    print("Sprecher \(seg.speakerId): [\(seg.startTime)s - \(seg.endTime)s]")
}
print("\(result.numSpeakers) Sprecher erkannt")
```

### Sprechereinbettung

```swift
let model = try await WeSpeakerModel.fromPretrained()
// Oder: let model = try await WeSpeakerModel.fromPretrained(engine: .coreml)
let embedding = model.embed(audio: samples, sampleRate: 16000)
// embedding: [Float] der Länge 256, L2-normalisiert

// Sprecher vergleichen
let similarity = WeSpeakerModel.cosineSimilarity(embeddingA, embeddingB)
```

### Sprecherextraktion

Nur die Segmente eines bestimmten Sprechers anhand einer Referenzaufnahme extrahieren:

```swift
let pipeline = try await DiarizationPipeline.fromPretrained()
let targetEmb = pipeline.embeddingModel.embed(audio: enrollmentAudio, sampleRate: 16000)
let segments = pipeline.extractSpeaker(
    audio: meetingAudio, sampleRate: 16000,
    targetEmbedding: targetEmb
)
```

### Sortformer-Diarisierung (durchgängig, CoreML)

NVIDIA Sortformer sagt die Sprecheraktivität pro Frame für bis zu 4 Sprecher direkt vorher — ohne Einbettung oder Clustering. Läuft auf der Neural Engine.

```swift
let diarizer = try await SortformerDiarizer.fromPretrained()
let result = diarizer.diarize(audio: samples, sampleRate: 16000, config: .default)
for seg in result.segments {
    print("Sprecher \(seg.speakerId): [\(seg.startTime)s - \(seg.endTime)s]")
}
```

### Diarisierungs-CLI

```bash
make build

# Pyannote-Diarisierung (Standard)
.build/release/audio diarize meeting.wav

# Sortformer-Diarisierung (CoreML, Neural Engine)
.build/release/audio diarize meeting.wav --engine sortformer

# CoreML-Einbettungen (Neural Engine, nur Pyannote)
.build/release/audio diarize meeting.wav --embedding-engine coreml

# JSON-Ausgabe
.build/release/audio diarize meeting.wav --json

# Bestimmten Sprecher extrahieren (nur Pyannote)
.build/release/audio diarize meeting.wav --target-speaker enrollment.wav

# Sprechereinbettung
.build/release/audio embed-speaker enrollment.wav --json
.build/release/audio embed-speaker enrollment.wav --engine coreml
```

Siehe [Sprecherdiarisierung](docs/inference/speaker-diarization.md) für Architekturdetails.

## Sprachverbesserung — Rauschunterdrückung und Audiobereinigung

### Rauschunterdrückung

```swift
import SpeechEnhancement
import AudioCommon  // für WAVWriter

let enhancer = try await SpeechEnhancer.fromPretrained()
// Lädt beim ersten Start ~4,3 MB herunter (CoreML FP16-Modell + Hilfsdaten)

let cleanAudio = try enhancer.enhance(audio: noisyAudio, sampleRate: 48000)
try WAVWriter.write(samples: cleanAudio, sampleRate: 48000, to: outputURL)
```

### Denoise CLI

```bash
make build

# Einfache Rauschentfernung
.build/release/audio denoise noisy.wav

# Benutzerdefinierter Ausgabepfad
.build/release/audio denoise noisy.wav --output clean.wav
```

Siehe [Sprachverbesserung](docs/inference/speech-enhancement.md) für Architekturdetails.

## Pipelines — Mehrere Modelle kombinieren

Alle Modelle entsprechen gemeinsamen Protokollen (`SpeechRecognitionModel`, `SpeechGenerationModel`, `SpeechEnhancementModel` usw.) und können zu Pipelines kombiniert werden:

### Verrauschte Spracherkennung (DeepFilterNet + ASR)

```swift
import SpeechEnhancement
import Qwen3ASR

let enhancer = try await SpeechEnhancer.fromPretrained()
let asr = try await Qwen3ASRModel.fromPretrained()

// Bei 48kHz verbessern, dann bei 16kHz transkribieren
let clean = try enhancer.enhance(audio: noisyAudio, sampleRate: 48000)
let clean16k = AudioResampler.resample(clean, from: 48000, to: 16000)
let text = asr.transcribe(audio: clean16k, sampleRate: 16000)
```

### Stimme-zu-Stimme-Weiterleitung (VAD + ASR + TTS)

```swift
import SpeechVAD
import Qwen3ASR
import Qwen3TTS

let vad = try await SileroVADModel.fromPretrained()
let asr = try await Qwen3ASRModel.fromPretrained()
let tts = try await Qwen3TTSModel.fromPretrained()

// Sprachsegmente erkennen, transkribieren, neu synthetisieren
let segments = vad.detectSpeech(audio: audio, sampleRate: 16000)
for seg in segments {
    let chunk = Array(audio[Int(seg.startTime * 16000)..<Int(seg.endTime * 16000)])
    let text = asr.transcribe(audio: chunk, sampleRate: 16000)
    let speech = tts.synthesize(text: text, language: "english")
    // speech: 24kHz-Mono-Float-Samples
}
```

### Meeting-Transkription (Diarisierung + ASR)

```swift
import SpeechVAD
import Qwen3ASR

let pipeline = try await DiarizationPipeline.fromPretrained()
let asr = try await Qwen3ASRModel.fromPretrained()

let result = pipeline.diarize(audio: meetingAudio, sampleRate: 16000)
for seg in result.segments {
    let chunk = Array(meetingAudio[Int(seg.startTime * 16000)..<Int(seg.endTime * 16000)])
    let text = asr.transcribe(audio: chunk, sampleRate: 16000)
    print("Sprecher \(seg.speakerId) [\(seg.startTime)s-\(seg.endTime)s]: \(text)")
}
```

Siehe [Gemeinsame Protokolle](docs/shared-protocols.md) für die vollständige Protokollreferenz.

## HTTP-API-Server

Ein eigenständiger HTTP-Server stellt alle Modelle über REST- und WebSocket-Endpunkte bereit. Modelle werden beim ersten Zugriff verzögert geladen.

```bash
swift build -c release
.build/release/audio-server --port 8080

# Audio transkribieren
curl -X POST http://localhost:8080/transcribe --data-binary @audio.wav -H "Content-Type: audio/wav"

# Text-zu-Sprache
curl -X POST http://localhost:8080/speak -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "engine": "cosyvoice"}' -o output.wav

# Sprache-zu-Sprache (PersonaPlex)
curl -X POST http://localhost:8080/respond --data-binary @question.wav -o response.wav

# Sprachverbesserung
curl -X POST http://localhost:8080/enhance --data-binary @noisy.wav -o clean.wav

# Alle Modelle beim Start vorladen
.build/release/audio-server --preload --port 8080
```

### WebSocket-Streaming

#### OpenAI Realtime API (`/v1/realtime`)

Der primäre WebSocket-Endpunkt implementiert das [OpenAI Realtime API](https://platform.openai.com/docs/api-reference/realtime)-Protokoll — alle Nachrichten sind JSON mit einem `type`-Feld, Audio ist base64-kodiertes PCM16 24kHz Mono.

**Client → Server-Ereignisse:**

| Ereignis | Beschreibung |
|----------|-------------|
| `session.update` | Engine, Sprache, Audioformat konfigurieren |
| `input_audio_buffer.append` | Base64-PCM16-Audioblock senden |
| `input_audio_buffer.commit` | Angesammeltes Audio transkribieren (ASR) |
| `input_audio_buffer.clear` | Audiopuffer leeren |
| `response.create` | TTS-Synthese anfordern |

**Server → Client-Ereignisse:**

| Ereignis | Beschreibung |
|----------|-------------|
| `session.created` | Sitzung initialisiert |
| `session.updated` | Konfiguration bestätigt |
| `input_audio_buffer.committed` | Audio zur Transkription übermittelt |
| `conversation.item.input_audio_transcription.completed` | ASR-Ergebnis |
| `response.audio.delta` | Base64-PCM16-Audioblock (TTS) |
| `response.audio.done` | Audio-Streaming abgeschlossen |
| `response.done` | Antwort mit Metadaten abgeschlossen |
| `error` | Fehler mit Typ und Nachricht |

```javascript
const ws = new WebSocket('ws://localhost:8080/v1/realtime');

// ASR: Audio senden, Transkription erhalten
ws.send(JSON.stringify({ type: 'input_audio_buffer.append', audio: base64PCM16 }));
ws.send(JSON.stringify({ type: 'input_audio_buffer.commit' }));
// → empfängt: conversation.item.input_audio_transcription.completed

// TTS: Text senden, gestreamtes Audio erhalten
ws.send(JSON.stringify({
  type: 'response.create',
  response: { modalities: ['audio', 'text'], instructions: 'Hello world' }
}));
// → empfängt: response.audio.delta (Base64-Blöcke), response.audio.done, response.done
```

Ein Beispiel-HTML-Client befindet sich unter `Examples/websocket-client.html` — öffnen Sie ihn im Browser, während der Server läuft.

Der Server ist ein separates `AudioServer`-Modul und ein eigenständiges `audio-server`-Executable — er fügt dem Haupt-CLI `audio` kein Hummingbird/WebSocket hinzu.

## Latenz (M2 Max, 64 GB)

### ASR

| Modell | Backend | RTF | 10s Audio verarbeitet in |
|--------|---------|-----|--------------------------|
| Qwen3-ASR-0.6B (4-bit) | MLX | ~0.06 | ~0.6s |
| Qwen3-ASR-0.6B (INT8) | CoreML + MLX | ~0.09 | ~0.9s |
| Qwen3-ASR-1.7B (8-bit) | MLX | ~0.11 | ~1.1s |
| Parakeet-TDT-0.6B (INT4) | CoreML (Neural Engine) | ~0.12 kalt, ~0.03 warm | ~1.2s / ~0.3s |
| Whisper-large-v3 | whisper.cpp (Q5_0) | ~0.10 | ~1.0s |
| Whisper-small | whisper.cpp (Q5_0) | ~0.04 | ~0.4s |

### Gezwungene Ausrichtung (Forced Alignment)

| Modell | Framework | 20s Audio | RTF |
|--------|-----------|-----------|-----|
| Qwen3-ForcedAligner-0.6B (4-bit) | MLX Swift (debug) | ~365ms | ~0.018 |

> Einzelner nicht-autoregressiver Vorwärtsdurchlauf — keine Sampling-Schleife. Der Audio-Encoder dominiert (~328ms), der Decoder-Einzeldurchlauf benötigt ~37ms. **55x schneller als Echtzeit.**

### TTS

| Modell | Framework | Kurz (1s) | Mittel (3s) | Lang (6s) | Streaming First-Packet |
|--------|-----------|-----------|-------------|-----------|------------------------|
| Qwen3-TTS-0.6B (4-bit) | MLX Swift (release) | 1.6s (RTF 1.2) | 2.3s (RTF 0.7) | 3.9s (RTF 0.7) | ~120ms (1-Frame) |
| Kokoro-82M | CoreML (Neural Engine) | ~1.4s (RTFx 0.7) | ~4.3s (RTFx 0.7) | ~8.6s (RTFx 0.7) | N/A (nicht-autoregressiv) |
| Apple `AVSpeechSynthesizer` | AVFoundation | 0.08s | 0.08s | 0.17s (RTF 0.02) | N/A |

> Qwen3-TTS erzeugt natürliche, ausdrucksstarke Sprache mit Prosodie und Emotion und läuft **schneller als Echtzeit** (RTF < 1.0). Streaming-Synthese liefert den ersten Audioblock in ~120ms. Kokoro-82M läuft vollständig auf der Neural Engine mit einer 3-stufigen Pipeline (RTFx ~0.7), ideal für iOS. Apples eingebaute TTS ist schneller, erzeugt aber roboterhafte, monotone Sprache.

### PersonaPlex (Sprache-zu-Sprache)

| Modell | Framework | ms/Schritt | RTF | Anmerkungen |
|--------|-----------|------------|-----|-------------|
| PersonaPlex-7B (8-bit) | MLX Swift (release) | ~112ms | ~1.4 | Empfohlen — kohärente Antworten, 30% schneller als 4-bit |
| PersonaPlex-7B (4-bit) | MLX Swift (release) | ~158ms | ~1.97 | Nicht empfohlen — beeinträchtigte Ausgabequalität |

> **8-bit verwenden.** INT8 ist sowohl schneller (112 ms/Schritt vs. 158 ms/Schritt) als auch erzeugt kohärente Vollduplex-Antworten. INT4-Quantisierung verschlechtert die Generierungsqualität und produziert unverständliche Sprache. INT8 läuft mit ~112ms/Schritt auf M2 Max.

### VAD & Sprechereinbettung

| Modell | Backend | Latenz pro Aufruf | RTF | Anmerkungen |
|--------|---------|-------------------|-----|-------------|
| Silero-VAD-v5 | MLX | ~2.1ms / Block | 0.065 | GPU (Metal) |
| Silero-VAD-v5 | CoreML | ~0.27ms / Block | 0.008 | Neural Engine, **7,7x schneller** |
| WeSpeaker ResNet34-LM | MLX | ~310ms / 20s Audio | 0.016 | GPU (Metal) |
| WeSpeaker ResNet34-LM | CoreML | ~430ms / 20s Audio | 0.021 | Neural Engine, entlastet GPU |

> Silero VAD CoreML läuft auf der Neural Engine mit 7,7-facher Geschwindigkeit gegenüber MLX und eignet sich damit ideal für Dauerüberwachung per Mikrofon. WeSpeaker MLX ist auf der GPU schneller, aber CoreML entlastet die GPU für gleichzeitige Aufgaben (TTS, ASR). Beide Backends liefern gleichwertige Ergebnisse.

### Sprachverbesserung

| Modell | Backend | Dauer | Latenz | RTF |
|--------|---------|-------|--------|-----|
| DeepFilterNet3 (FP16) | CoreML | 5s | 0.65s | 0.13 |
| DeepFilterNet3 (FP16) | CoreML | 10s | 1.2s | 0.12 |
| DeepFilterNet3 (FP16) | CoreML | 20s | 4.8s | 0.24 |

RTF = Echtzeitfaktor (niedriger ist besser, < 1.0 = schneller als Echtzeit). GRU-Kosten skalieren ~O(n^2).

### MLX vs CoreML

Beide Backends liefern gleichwertige Ergebnisse. Wählen Sie basierend auf Ihrem Anwendungsfall:

| | MLX | CoreML |
|---|---|---|
| **Hardware** | GPU (Metal-Shader) | Neural Engine + CPU |
| **Geeignet für** | Maximaler Durchsatz, Einzelmodell-Aufgaben | Mehrmodell-Pipelines, Hintergrundaufgaben |
| **Stromverbrauch** | Höhere GPU-Auslastung | Geringerer Verbrauch, entlastet GPU |
| **Latenz** | Schneller bei großen Modellen (WeSpeaker) | Schneller bei kleinen Modellen (Silero VAD) |

**Desktop-Inferenz**: MLX ist der Standard — schnellste Einzelmodell-Leistung auf Apple Silicon. Wechseln Sie zu CoreML, wenn mehrere Modelle gleichzeitig laufen (z.B. VAD + ASR + TTS), um GPU-Konflikte zu vermeiden, oder für batteriesensible Aufgaben auf Laptops.

CoreML-Modelle sind für den Qwen3-ASR-Encoder, Silero VAD und WeSpeaker verfügbar. Für Qwen3-ASR verwenden Sie `--engine qwen3-coreml` (Hybrid: CoreML-Encoder auf ANE + MLX-Textdecoder auf GPU). Für VAD/Einbettungen übergeben Sie `engine: .coreml` bei der Erstellung — die Inferenz-API ist identisch.

## Genauigkeits-Benchmarks

### ASR — Word Error Rate ([Details](docs/benchmarks/asr-wer.md))

| Modell | WER% (LibriSpeech test-clean) | RTF |
|--------|-------------------------------|-----|
| Qwen3-ASR 1.7B 8-bit | **2.35** | 0.090 |
| Qwen3-ASR 1.7B 4-bit | 2.57 | 0.045 |
| Parakeet TDT INT8 | 2.74 | 0.089 |
| Qwen3-ASR 0.6B 8-bit | 2.80 | 0.025 |

Qwen3-ASR 1.7B 8-bit übertrifft Whisper Large v3 Turbo (2,5%) bei vergleichbarer Größe. Mehrsprachig: 10 Sprachen auf FLEURS gemessen.

### TTS — Round-Trip-Verständlichkeit ([Details](docs/benchmarks/tts-roundtrip.md))

| Engine | WER% | RTF |
|--------|------|-----|
| CosyVoice3 | **3.25** | 0.59 |
| Qwen3-TTS 1.7B | 3.47 | 0.79 |
| Kokoro-82M | 3.90 | 0.17 |

### VAD — Spracherkennung ([Details](docs/benchmarks/vad-detection.md))

| Engine | F1% (FLEURS) | RTF |
|--------|-------------|-----|
| FireRedVAD | **99.12** | 0.007 |
| Silero CoreML | 95.13 | 0.022 |
| Pyannote MLX | 94.86 | 0.358 |

## Architektur

**Modelle:** [ASR-Modell](docs/models/asr-model.md), [TTS-Modell](docs/models/tts-model.md), [CosyVoice TTS](docs/models/cosyvoice-tts.md), [Kokoro TTS](docs/models/kokoro-tts.md), [Parakeet TDT](docs/models/parakeet-asr.md), [PersonaPlex](docs/models/personaplex.md), [FireRedVAD](docs/models/fireredvad.md)

**Inferenz:** [ASR-Inferenz](docs/inference/qwen3-asr-inference.md), [TTS-Inferenz](docs/inference/qwen3-tts-inference.md), [Forced Aligner](docs/inference/forced-aligner.md), [FireRedVAD](docs/inference/fireredvad.md), [Silero VAD](docs/inference/silero-vad.md), [Sprecherdiarisierung](docs/inference/speaker-diarization.md), [Sprachverbesserung](docs/inference/speech-enhancement.md), [Audio-Wiedergabe](docs/audio/playback.md)

**Benchmarks:** [ASR WER](docs/benchmarks/asr-wer.md), [TTS Round-Trip](docs/benchmarks/tts-roundtrip.md), [VAD-Erkennung](docs/benchmarks/vad-detection.md)

**Referenz:** [Gemeinsame Protokolle](docs/shared-protocols.md)

## Cache-Konfiguration

Modellgewichte werden lokal zwischengespeichert. Cache-Verzeichnis anpassen mit:

```bash
export QWEN3_CACHE_DIR=/path/to/cache
```

## MLX Metal-Bibliothek

Falls zur Laufzeit `Failed to load the default metallib` erscheint, fehlt die Metal-Shader-Bibliothek. Führen Sie `make build` aus (oder `./scripts/build_mlx_metallib.sh release` nach einem manuellen `swift build`), um sie zu kompilieren. Falls die Metal Toolchain fehlt, installieren Sie diese zuerst:

```bash
xcodebuild -downloadComponent MetalToolchain
```

## Tests

Unit-Tests (Konfiguration, Sampling, Textvorverarbeitung, Zeitstempelkorrektur) laufen ohne Modell-Downloads:

```bash
swift test --filter "Qwen3TTSConfigTests|SamplingTests|CosyVoiceTTSConfigTests|CamPlusPlusMelExtractorTests|PersonaPlexTests|ForcedAlignerTests/testText|ForcedAlignerTests/testTimestamp|ForcedAlignerTests/testLIS|SileroVADTests/testSilero|SileroVADTests/testReflection|SileroVADTests/testProcess|SileroVADTests/testReset|SileroVADTests/testDetect|SileroVADTests/testStreaming|SileroVADTests/testVADEvent|KokoroTTSTests"
```

Integrationstests erfordern Modellgewichte (werden beim ersten Start automatisch heruntergeladen):

```bash
# TTS Round-Trip: Text synthetisieren, WAV speichern, mit ASR zurück transkribieren
swift test --filter TTSASRRoundTripTests

# Nur ASR: Test-Audio transkribieren
swift test --filter Qwen3ASRIntegrationTests

# Forced Aligner E2E: Wortgenaue Zeitstempel (~979 MB Download)
swift test --filter ForcedAlignerTests/testForcedAlignerE2E

# PersonaPlex E2E: Sprache-zu-Sprache-Pipeline (~5,5 GB Download)
PERSONAPLEX_E2E=1 swift test --filter PersonaPlexE2ETests
```

> **Hinweis:** Die MLX-Metal-Bibliothek muss vor der Ausführung von Tests, die MLX-Operationen verwenden, kompiliert sein.
> Siehe [MLX Metal-Bibliothek](#mlx-metal-bibliothek) für Anweisungen.

## Unterstützte Sprachen

| Modell | Sprachen |
|--------|----------|
| Qwen3-ASR | 52 Sprachen (CN, EN, Kantonesisch, DE, FR, ES, JA, KO, RU, + 22 chinesische Dialekte, ...) |
| Parakeet TDT | 25 europäische Sprachen (BG, CS, DA, DE, EL, EN, ES, ET, FI, FR, HR, HU, IT, LT, LV, MT, NL, PL, PT, RO, RU, SK, SL, SV, UK) |
| Qwen3-TTS | EN, CN, DE, JA, ES, FR, KO, RU, IT, PT (+ Peking-/Sichuan-Dialekte über CustomVoice) |
| CosyVoice TTS | CN, EN, JA, KO, DE, ES, FR, IT, RU |
| Kokoro TTS | EN (US/UK), ES, FR, HI, IT, JA, PT, CN, KO, DE |
| PersonaPlex | EN |

## Im Vergleich

### Sprache-zu-Text (ASR): speech-swift vs. Alternativen

| | **speech-swift (Qwen3-ASR)** | **whisper.cpp** | **Apple SFSpeechRecognizer** | **Google Cloud Speech** |
|---|---|---|---|---|
| **Laufzeitumgebung** | Auf dem Gerät (MLX/CoreML) | Auf dem Gerät (CPU/GPU) | Auf dem Gerät oder Cloud | Nur Cloud |
| **Sprachen** | 52 | 100+ | ~70 (auf dem Gerät: eingeschränkt) | 125+ |
| **RTF (10s Audio, M2 Max)** | 0.06 (17x Echtzeit) | 0.10 (Whisper-large-v3) | N/A | N/A |
| **Streaming** | Nein (Batch) | Nein (Batch) | Ja | Ja |
| **Eigene Modelle** | Ja (HuggingFace-Gewichte austauschen) | Ja (GGML-Modelle) | Nein | Nein |
| **Swift-API** | Natives async/await | C++ mit Swift-Bridge | Nativ | REST/gRPC |
| **Datenschutz** | Vollständig auf dem Gerät | Vollständig auf dem Gerät | Abhängig von Konfiguration | Daten werden an Cloud gesendet |
| **Wortzeitstempel** | Ja (Forced Aligner) | Ja | Eingeschränkt | Ja |
| **Kosten** | Kostenlos (Apache 2.0) | Kostenlos (MIT) | Kostenlos (auf dem Gerät) | Bezahlung pro Minute |

### Text-zu-Sprache (TTS): speech-swift vs. Alternativen

| | **speech-swift (Qwen3-TTS)** | **speech-swift (Kokoro)** | **Apple AVSpeechSynthesizer** | **ElevenLabs / Cloud TTS** |
|---|---|---|---|---|
| **Qualität** | Neural, ausdrucksstark | Neural, natürlich | Roboterhaft, monoton | Neural, höchste Qualität |
| **Laufzeitumgebung** | Auf dem Gerät (MLX) | Auf dem Gerät (CoreML) | Auf dem Gerät | Nur Cloud |
| **Streaming** | Ja (~120ms erster Block) | Nein (3-stufige Pipeline) | Nein | Ja |
| **Stimmklonen** | Ja | Nein | Nein | Ja |
| **Stimmen** | 9 eingebaut + beliebige klonen | 54 voreingestellte Stimmen | ~50 Systemstimmen | 1000+ |
| **Sprachen** | 10 | 10 | 60+ | 30+ |
| **iOS-Unterstützung** | Nur macOS | iOS + macOS | iOS + macOS | Beliebig (API) |
| **Kosten** | Kostenlos (Apache 2.0) | Kostenlos (Apache 2.0) | Kostenlos | Bezahlung pro Zeichen |

### Wann speech-swift verwenden

- **Datenschutzkritische Apps** — Medizin, Recht, Unternehmen, wo Audio das Gerät nicht verlassen darf
- **Offline-Nutzung** — nach dem erstmaligen Modell-Download keine Internetverbindung erforderlich
- **Kostensensibel** — keine Gebühren pro Minute oder pro Zeichen
- **Apple-Silicon-Optimierung** — speziell für M-Serie-GPU (Metal) und Neural Engine entwickelt
- **Vollständige Pipeline** — ASR + TTS + VAD + Diarisierung + Verbesserung in einem einzigen Swift-Paket kombinieren

## FAQ

**Funktioniert speech-swift auf iOS?**
Kokoro TTS, Qwen3.5-Chat (CoreML), Silero VAD, Parakeet ASR, DeepFilterNet3 und WeSpeaker laufen alle auf iOS 17+ über CoreML auf der Neural Engine. MLX-basierte Modelle (Qwen3-ASR, Qwen3-TTS, Qwen3.5-Chat MLX, PersonaPlex) erfordern macOS 14+ auf Apple Silicon.

**Wird eine Internetverbindung benötigt?**
Nur für den erstmaligen Modell-Download von HuggingFace (automatisch, zwischengespeichert in `~/Library/Caches/qwen3-speech/`). Danach läuft alle Inferenz vollständig offline ohne Netzwerkzugriff.

**Wie schneidet speech-swift im Vergleich zu Whisper ab?**
Qwen3-ASR-0.6B erreicht RTF 0,06 auf M2 Max — 40% schneller als Whisper-large-v3 über whisper.cpp (RTF 0,10) — bei vergleichbarer Genauigkeit in 52 Sprachen. speech-swift bietet eine native Swift-async/await-API, während whisper.cpp eine C++-Bridge erfordert.

**Kann ich es in einer kommerziellen App verwenden?**
Ja. speech-swift ist unter Apache 2.0 lizenziert. Die zugrunde liegenden Modellgewichte haben eigene Lizenzen (siehe die HuggingFace-Seite jedes Modells).

**Welche Apple-Silicon-Chips werden unterstützt?**
Alle M-Serie-Chips: M1, M2, M3, M4 und deren Pro/Max/Ultra-Varianten. Erfordert macOS 14+ (Sonoma) oder iOS 17+.

**Wie viel Speicher wird benötigt?**
Von ~3 MB (Silero VAD) bis ~6,5 GB (PersonaPlex 7B). Kokoro TTS benötigt ~500 MB, Qwen3-ASR ~2,2 GB. Siehe die Tabelle [Speicheranforderungen](#speicheranforderungen) für alle Details.

**Können mehrere Modelle gleichzeitig laufen?**
Ja. Verwenden Sie CoreML-Modelle auf der Neural Engine zusammen mit MLX-Modellen auf der GPU, um Konflikte zu vermeiden — zum Beispiel Silero VAD (CoreML) + Qwen3-ASR (MLX) + Qwen3-TTS (MLX).

**Gibt es eine REST-API?**
Ja. Das `audio-server`-Binary stellt alle Modelle über HTTP-REST- und WebSocket-Endpunkte bereit, einschließlich eines OpenAI-Realtime-API-kompatiblen WebSocket unter `/v1/realtime`.

## Mitwirken

Wir freuen uns über Beiträge! Ob Fehlerbehebung, neue Modellintegration oder Verbesserung der Dokumentation — Pull Requests sind willkommen.

**So starten Sie:**
1. Forken Sie das Repository und erstellen Sie einen Feature-Branch
2. `make build` zum Kompilieren (erfordert Xcode + Metal Toolchain)
3. `make test` zum Ausführen der Testsuite
4. Öffnen Sie einen PR gegen `main`

## Lizenz

Apache 2.0
