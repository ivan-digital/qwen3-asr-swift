# Speech Swift

Modelos de IA de voz para Apple Silicon, impulsados por MLX Swift y CoreML.

📖 Read in: [English](README.md) · [中文](README_zh.md) · [日本語](README_ja.md) · [한국어](README_ko.md) · [Español](README_es.md) · [Deutsch](README_de.md) · [Français](README_fr.md) · [हिन्दी](README_hi.md) · [Português](README_pt.md) · [Русский](README_ru.md)

Reconocimiento, síntesis y comprensión de voz en el dispositivo para Mac e iOS. Se ejecuta localmente en Apple Silicon — sin nube, sin claves de API, ningún dato sale del dispositivo.

**[📚 Documentación completa →](https://soniqo.audio/es)** · **[🤗 Modelos en HuggingFace](https://huggingface.co/aufklarer)** · **[📝 Blog](https://blog.ivan.digital)**

- **[Qwen3-ASR](https://soniqo.audio/es/guides/transcribe)** — Voz a texto (reconocimiento automático del habla, 52 idiomas, MLX + CoreML)
- **[Parakeet TDT](https://soniqo.audio/es/guides/parakeet)** — Voz a texto vía CoreML (Neural Engine, NVIDIA FastConformer + decodificador TDT, 25 idiomas)
- **[Omnilingual ASR](https://soniqo.audio/es/guides/omnilingual)** — Voz a texto (Meta wav2vec2 + CTC, **1.672 idiomas** en 32 escrituras, CoreML 300M + MLX 300M/1B/3B/7B)
- **[Dictado en streaming](https://soniqo.audio/es/guides/dictate)** — Dictado en tiempo real con resultados parciales y detección de fin de enunciado (Parakeet-EOU-120M)
- **[Nemotron Streaming](https://soniqo.audio/es/guides/nemotron)** — ASR en streaming de baja latencia con puntuación y mayúsculas nativas (NVIDIA Nemotron-Speech-Streaming-0.6B, CoreML, inglés)
- **[Qwen3-ForcedAligner](https://soniqo.audio/es/guides/align)** — Alineación de marcas temporales a nivel de palabra (audio + texto → marcas temporales)
- **[Qwen3-TTS](https://soniqo.audio/es/guides/speak)** — Síntesis de texto a voz (máxima calidad, streaming, hablantes personalizados, 10 idiomas)
- **[CosyVoice TTS](https://soniqo.audio/es/guides/cosyvoice)** — TTS con streaming, clonación de voz, diálogo multi-hablante y etiquetas de emoción (9 idiomas)
- **[Kokoro TTS](https://soniqo.audio/es/guides/kokoro)** — TTS en el dispositivo (82M, CoreML/Neural Engine, 54 voces, listo para iOS, 10 idiomas)
- **[Qwen3.5-Chat](https://soniqo.audio/es/guides/chat)** — Chat LLM en el dispositivo (0.8B, MLX INT4 + CoreML INT8, DeltaNet híbrido, tokens en streaming)
- **[PersonaPlex](https://soniqo.audio/es/guides/respond)** — Voz a voz full-duplex (7B, audio de entrada → audio de salida, 18 presets de voz)
- **[DeepFilterNet3](https://soniqo.audio/es/guides/denoise)** — Supresión de ruido en tiempo real (2.1M parámetros, 48 kHz)
- **[Palabra de activación](https://soniqo.audio/es/guides/wake-word)** — Detección de palabras clave en el dispositivo (KWS Zipformer 3M, CoreML, 26× tiempo real, lista de palabras clave configurable)
- **[VAD](https://soniqo.audio/es/guides/vad)** — Detección de actividad vocal (Silero streaming, Pyannote offline, FireRedVAD 100+ idiomas)
- **[Diarización de hablantes](https://soniqo.audio/es/guides/diarize)** — Quién habló cuándo (pipeline Pyannote, Sortformer de extremo a extremo en Neural Engine)
- **[Embeddings de hablante](https://soniqo.audio/es/guides/embed-speaker)** — WeSpeaker ResNet34 (256 dim), CAM++ (192 dim)

Papers: [Qwen3-ASR](https://arxiv.org/abs/2601.21337) (Alibaba) · [Qwen3-TTS](https://arxiv.org/abs/2601.15621) (Alibaba) · [Omnilingual ASR](https://arxiv.org/abs/2511.09690) (Meta) · [Parakeet TDT](https://arxiv.org/abs/2304.06795) (NVIDIA) · [CosyVoice 3](https://arxiv.org/abs/2505.17589) (Alibaba) · [Kokoro](https://arxiv.org/abs/2301.01695) (StyleTTS 2) · [PersonaPlex](https://arxiv.org/abs/2602.06053) (NVIDIA) · [Mimi](https://arxiv.org/abs/2410.00037) (Kyutai) · [Sortformer](https://arxiv.org/abs/2409.06656) (NVIDIA)

## Novedades

- **19 abr 2026** — [MLX frente a CoreML en Apple Silicon — guía práctica para elegir el backend correcto](https://blog.ivan.digital/mlx-vs-coreml-on-apple-silicon-a-practical-guide-to-picking-the-right-backend-and-why-you-should-f77ddea7b27a)
- **20 mar 2026** — [Superamos a Whisper Large v3 con un modelo de 600M ejecutándose completamente en tu Mac](https://blog.ivan.digital/we-beat-whisper-large-v3-with-a-600m-model-running-entirely-on-your-mac-20e6ce191174)
- **26 feb 2026** — [Diarización de hablantes y detección de actividad vocal en Apple Silicon — Swift nativo con MLX](https://blog.ivan.digital/speaker-diarization-and-voice-activity-detection-on-apple-silicon-native-swift-with-mlx-92ea0c9aca0f)
- **23 feb 2026** — [NVIDIA PersonaPlex 7B en Apple Silicon — Voz a voz full-duplex en Swift nativo con MLX](https://blog.ivan.digital/nvidia-personaplex-7b-on-apple-silicon-full-duplex-speech-to-speech-in-native-swift-with-mlx-0aa5276f2e23)
- **12 feb 2026** — [Qwen3-ASR Swift: ASR + TTS en el dispositivo para Apple Silicon — Arquitectura y benchmarks](https://blog.ivan.digital/qwen3-asr-swift-on-device-asr-tts-for-apple-silicon-architecture-and-benchmarks-27cbf1e4463f)

## Inicio rápido

Añade el paquete a tu `Package.swift`:

```swift
.package(url: "https://github.com/soniqo/speech-swift", branch: "main")
```

Importa solo los módulos que necesites — cada modelo es una librería SPM independiente, así no pagas por lo que no uses:

```swift
.product(name: "ParakeetStreamingASR", package: "speech-swift"),
.product(name: "SpeechUI",             package: "speech-swift"),  // vistas SwiftUI opcionales
```

**Transcribe un buffer de audio en 3 líneas:**

```swift
import ParakeetStreamingASR

let model = try await ParakeetStreamingASRModel.fromPretrained()
let text = try model.transcribeAudio(audioSamples, sampleRate: 16000)
```

**Streaming en vivo con resultados parciales:**

```swift
for await partial in model.transcribeStream(audio: samples, sampleRate: 16000) {
    print(partial.isFinal ? "FINAL: \(partial.text)" : "... \(partial.text)")
}
```

**Vista de dictado SwiftUI en ~10 líneas:**

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

`SpeechUI` solo incluye `TranscriptionView` (finales + parciales) y `TranscriptionStore` (adaptador de ASR en streaming). Usa AVFoundation para la visualización y reproducción de audio.

Productos SPM disponibles: `Qwen3ASR`, `Qwen3TTS`, `Qwen3TTSCoreML`, `ParakeetASR`, `ParakeetStreamingASR`, `NemotronStreamingASR`, `OmnilingualASR`, `KokoroTTS`, `CosyVoiceTTS`, `PersonaPlex`, `SpeechVAD`, `SpeechEnhancement`, `Qwen3Chat`, `SpeechCore`, `SpeechUI`, `AudioCommon`.

## Modelos

Vista compacta a continuación. **[Catálogo completo de modelos con tamaños, cuantizaciones, URLs de descarga y tablas de memoria → soniqo.audio/architecture](https://soniqo.audio/es/architecture)**.

| Modelo | Tarea | Backends | Tamaños | Idiomas |
|-------|------|----------|-------|-----------|
| [Qwen3-ASR](https://soniqo.audio/es/guides/transcribe) | Voz → Texto | MLX, CoreML (híbrido) | 0.6B, 1.7B | 52 |
| [Parakeet TDT](https://soniqo.audio/es/guides/parakeet) | Voz → Texto | CoreML (ANE) | 0.6B | 25 europeos |
| [Parakeet EOU](https://soniqo.audio/es/guides/dictate) | Voz → Texto (streaming) | CoreML (ANE) | 120M | 25 europeos |
| [Nemotron Streaming](https://soniqo.audio/es/guides/nemotron) | Voz → Texto (streaming, con puntuación) | CoreML (ANE) | 0.6B | Inglés |
| [Omnilingual ASR](https://soniqo.audio/es/guides/omnilingual) | Voz → Texto | CoreML (ANE), MLX | 300M / 1B / 3B / 7B | **[1.672](https://github.com/facebookresearch/omnilingual-asr/blob/main/src/omnilingual_asr/models/wav2vec2_llama/lang_ids.py)** |
| [Qwen3-ForcedAligner](https://soniqo.audio/es/guides/align) | Audio + Texto → Marcas temp. | MLX, CoreML | 0.6B | Multi |
| [Qwen3-TTS](https://soniqo.audio/es/guides/speak) | Texto → Voz | MLX, CoreML | 0.6B, 1.7B | 10 |
| [CosyVoice3](https://soniqo.audio/es/guides/cosyvoice) | Texto → Voz | MLX | 0.5B | 9 |
| [Kokoro-82M](https://soniqo.audio/es/guides/kokoro) | Texto → Voz | CoreML (ANE) | 82M | 10 |
| [Qwen3.5-Chat](https://soniqo.audio/es/guides/chat) | Texto → Texto (LLM) | MLX, CoreML | 0.8B | Multi |
| [PersonaPlex](https://soniqo.audio/es/guides/respond) | Voz → Voz | MLX | 7B | EN |
| [Silero VAD](https://soniqo.audio/es/guides/vad) | Detección de actividad vocal | MLX, CoreML | 309K | Agnóstico |
| [Pyannote](https://soniqo.audio/es/guides/diarize) | VAD + Diarización | MLX | 1.5M | Agnóstico |
| [Sortformer](https://soniqo.audio/es/guides/diarize) | Diarización (E2E) | CoreML (ANE) | — | Agnóstico |
| [DeepFilterNet3](https://soniqo.audio/es/guides/denoise) | Mejora de voz | CoreML | 2.1M | Agnóstico |
| [WeSpeaker](https://soniqo.audio/es/guides/embed-speaker) | Embedding de hablante | MLX, CoreML | 6.6M | Agnóstico |

## Instalación

### Homebrew

Requiere Homebrew ARM nativo (`/opt/homebrew`). Homebrew Rosetta/x86_64 no está soportado.

```bash
brew tap soniqo/speech https://github.com/soniqo/speech-swift
brew install speech
```

Luego:

```bash
audio transcribe recording.wav
audio speak "Hello world"
audio respond --input question.wav --transcript
audio-server --port 8080            # servidor HTTP / WebSocket local (OpenAI-compatible /v1/realtime)
```

**[Referencia completa de la CLI →](https://soniqo.audio/es/cli)**

### Swift Package Manager

```swift
dependencies: [
    .package(url: "https://github.com/soniqo/speech-swift", branch: "main")
]
```

Importa solo lo que necesites — cada modelo es su propio target SPM:

```swift
import Qwen3ASR             // Reconocimiento de voz (MLX)
import ParakeetASR          // Reconocimiento de voz (CoreML, batch)
import ParakeetStreamingASR // Dictado en streaming con parciales + EOU
import NemotronStreamingASR // ASR streaming en inglés con puntuación nativa (0.6B)
import OmnilingualASR       // 1.672 idiomas (CoreML + MLX)
import Qwen3TTS             // Síntesis de voz
import CosyVoiceTTS         // Síntesis de voz con clonación
import KokoroTTS            // Síntesis de voz (listo para iOS)
import Qwen3Chat            // Chat LLM en el dispositivo
import PersonaPlex          // Voz a voz full-duplex
import SpeechVAD            // VAD + diarización + embeddings
import SpeechEnhancement    // Supresión de ruido
import SpeechUI             // Componentes SwiftUI para transcripciones en streaming
import AudioCommon          // Protocolos y utilidades compartidas
```

### Requisitos

- Swift 6+, Xcode 16+ (con Metal Toolchain)
- macOS 15+ (Sequoia) o iOS 18+, Apple Silicon (M1/M2/M3/M4)

El mínimo de macOS 15 / iOS 18 proviene de [MLState](https://developer.apple.com/documentation/coreml/mlstate) —— la API de estado persistente de ANE de Apple —— que los pipelines CoreML (Qwen3-ASR, Qwen3-Chat, Qwen3-TTS) usan para mantener las cachés KV residentes en el Neural Engine entre pasos de token.

### Compilar desde el código fuente

```bash
git clone https://github.com/soniqo/speech-swift
cd speech-swift
make build
```

`make build` compila el paquete Swift **y** la librería de shaders MLX Metal. La librería Metal es necesaria para la inferencia en GPU — sin ella verás `Failed to load the default metallib` en tiempo de ejecución. `make debug` para builds de depuración, `make test` para la suite de pruebas.

**[Guía completa de compilación e instalación →](https://soniqo.audio/es/getting-started)**

## Aplicaciones de demostración

- **[DictateDemo](Examples/DictateDemo/)** ([documentación](https://soniqo.audio/es/guides/dictate)) — Dictado en streaming en la barra de menús de macOS con parciales en vivo, detección de fin de enunciado basada en VAD y copia con un clic. Se ejecuta como agent en segundo plano (Parakeet-EOU-120M + Silero VAD).
- **[iOSEchoDemo](Examples/iOSEchoDemo/)** — Demo de eco iOS (Parakeet ASR + Kokoro TTS). Dispositivo y simulador.
- **[PersonaPlexDemo](Examples/PersonaPlexDemo/)** — Asistente de voz conversacional con entrada de micrófono, VAD y contexto multi-turno. macOS. RTF ~0.94 en M2 Max (más rápido que tiempo real).
- **[SpeechDemo](Examples/SpeechDemo/)** — Dictado y síntesis TTS en una interfaz de pestañas. macOS.

El README de cada demo tiene instrucciones de compilación.

## Ejemplos de código

Los fragmentos siguientes muestran el camino mínimo para cada dominio. Cada sección enlaza a una guía completa en [soniqo.audio](https://soniqo.audio/es) con opciones de configuración, múltiples backends, patrones de streaming y recetas de CLI.

### Voz a texto — [guía completa →](https://soniqo.audio/es/guides/transcribe)

```swift
import Qwen3ASR

let model = try await Qwen3ASRModel.fromPretrained()
let text = model.transcribe(audio: audioSamples, sampleRate: 16000)
```

Backends alternativos: [Parakeet TDT](https://soniqo.audio/es/guides/parakeet) (CoreML, 32× tiempo real), [Omnilingual ASR](https://soniqo.audio/es/guides/omnilingual) (1.672 idiomas, CoreML o MLX), [Dictado en streaming](https://soniqo.audio/es/guides/dictate) (parciales en vivo).

### Alineación forzada — [guía completa →](https://soniqo.audio/es/guides/align)

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

### Texto a voz — [guía completa →](https://soniqo.audio/es/guides/speak)

```swift
import Qwen3TTS
import AudioCommon

let model = try await Qwen3TTSModel.fromPretrained()
let audio = model.synthesize(text: "Hello world", language: "english")
try WAVWriter.write(samples: audio, sampleRate: 24000, to: outputURL)
```

Motores TTS alternativos: [CosyVoice3](https://soniqo.audio/es/guides/cosyvoice) (streaming + clonación + etiquetas de emoción), [Kokoro-82M](https://soniqo.audio/es/guides/kokoro) (listo para iOS, 54 voces), [Clonación de voz](https://soniqo.audio/es/guides/voice-cloning).

### Voz a voz — [guía completa →](https://soniqo.audio/es/guides/respond)

```swift
import PersonaPlex

let model = try await PersonaPlexModel.fromPretrained()
let responseAudio = model.respond(userAudio: userSamples)
// Salida Float32 mono 24 kHz lista para reproducir
```

### Chat LLM — [guía completa →](https://soniqo.audio/es/guides/chat)

```swift
import Qwen3Chat

let chat = try await Qwen35MLXChat.fromPretrained()
chat.chat(messages: [(.user, "Explain MLX in one sentence")]) { token, isFinal in
    print(token, terminator: "")
}
```

### Detección de actividad vocal — [guía completa →](https://soniqo.audio/es/guides/vad)

```swift
import SpeechVAD

let vad = try await SileroVADModel.fromPretrained()
let segments = vad.detectSpeech(audio: samples, sampleRate: 16000)
for s in segments { print("\(s.startTime)s → \(s.endTime)s") }
```

### Diarización de hablantes — [guía completa →](https://soniqo.audio/es/guides/diarize)

```swift
import SpeechVAD

let diarizer = try await DiarizationPipeline.fromPretrained()
let segments = diarizer.diarize(audio: samples, sampleRate: 16000)
for s in segments { print("Speaker \(s.speakerId): \(s.startTime)s - \(s.endTime)s") }
```

### Mejora de voz — [guía completa →](https://soniqo.audio/es/guides/denoise)

```swift
import SpeechEnhancement

let denoiser = try await DeepFilterNet3Model.fromPretrained()
let clean = try denoiser.enhance(audio: noisySamples, sampleRate: 48000)
```

### Pipeline de voz (ASR → LLM → TTS) — [guía completa →](https://soniqo.audio/es/api)

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

`VoicePipeline` es la máquina de estados de agent de voz en tiempo real (impulsada por [speech-core](https://github.com/soniqo/speech-core)) con detección de turnos basada en VAD, manejo de interrupciones y STT eager. Conecta cualquier `SpeechRecognitionModel` + `SpeechGenerationModel` + `StreamingVADProvider`.

### Servidor API HTTP

```bash
audio-server --port 8080
```

Expone cada modelo a través de endpoints HTTP REST + WebSocket, incluyendo un WebSocket compatible con OpenAI Realtime API en `/v1/realtime`. Ver [`Sources/AudioServer/`](Sources/AudioServer/).

## Arquitectura

speech-swift está dividido en un target SPM por modelo para que los consumidores solo paguen por lo que importan. La infraestructura compartida vive en `AudioCommon` (protocolos, E/S de audio, descargador de HuggingFace, `SentencePieceModel`) y `MLXCommon` (carga de pesos, helpers `QuantizedLinear`, helper de atención multi-head `SDPA`).

**[Diagrama completo de arquitectura con backends, tablas de memoria y mapa de módulos → soniqo.audio/architecture](https://soniqo.audio/es/architecture)** · **[Referencia de API → soniqo.audio/api](https://soniqo.audio/es/api)** · **[Benchmarks → soniqo.audio/benchmarks](https://soniqo.audio/es/benchmarks)**

Docs locales (repositorio):
- **Modelos:** [Qwen3-ASR](docs/models/asr-model.md) · [Qwen3-TTS](docs/models/tts-model.md) · [CosyVoice](docs/models/cosyvoice-tts.md) · [Kokoro](docs/models/kokoro-tts.md) · [Parakeet TDT](docs/models/parakeet-asr.md) · [Parakeet Streaming](docs/models/parakeet-streaming-asr.md) · [Nemotron Streaming](docs/models/nemotron-streaming.md) · [Omnilingual ASR](docs/models/omnilingual-asr.md) · [PersonaPlex](docs/models/personaplex.md) · [FireRedVAD](docs/models/fireredvad.md)
- **Inferencia:** [Qwen3-ASR](docs/inference/qwen3-asr-inference.md) · [Parakeet TDT](docs/inference/parakeet-asr-inference.md) · [Parakeet Streaming](docs/inference/parakeet-streaming-asr-inference.md) · [Nemotron Streaming](docs/inference/nemotron-streaming-inference.md) · [Omnilingual ASR](docs/inference/omnilingual-asr-inference.md) · [TTS](docs/inference/qwen3-tts-inference.md) · [Forced Aligner](docs/inference/forced-aligner.md) · [Silero VAD](docs/inference/silero-vad.md) · [Diarización](docs/inference/speaker-diarization.md) · [Mejora de voz](docs/inference/speech-enhancement.md)
- **Referencia:** [Protocolos compartidos](docs/shared-protocols.md)

## Configuración de caché

Los pesos del modelo se descargan desde HuggingFace en el primer uso y se almacenan en `~/Library/Caches/qwen3-speech/`. Puedes sobrescribir con `QWEN3_CACHE_DIR` (CLI) o `cacheDir:` (API Swift). Todos los puntos de entrada `fromPretrained()` aceptan `offlineMode: true` para omitir la red cuando los pesos ya están en caché.

Consulta [`docs/inference/cache-and-offline.md`](docs/inference/cache-and-offline.md) para los detalles completos, incluyendo rutas de contenedor iOS sandboxed.

## Librería MLX Metal

Si ves `Failed to load the default metallib` en tiempo de ejecución, falta la librería de shaders Metal. Ejecuta `make build` o `./scripts/build_mlx_metallib.sh release` después de un `swift build` manual. Si falta el Metal Toolchain, instálalo primero:

```bash
xcodebuild -downloadComponent MetalToolchain
```

## Pruebas

```bash
make test                            # suite completa (unidad + E2E con descargas de modelos)
swift test --skip E2E                # solo unidad (seguro para CI, sin descargas)
swift test --filter Qwen3ASRTests    # módulo específico
```

Las clases de test E2E usan el prefijo `E2E` para que CI pueda filtrarlas con `--skip E2E`. Consulta [CLAUDE.md](CLAUDE.md#testing) para la convención completa de pruebas.

## Contribuir

PRs bienvenidos — correcciones de bugs, integraciones de nuevos modelos, documentación. Fork, crea una rama de feature, `make build && make test`, abre un PR contra `main`.

## Licencia

Apache 2.0
