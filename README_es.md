# Speech Swift

Modelos de IA de voz para Apple Silicon, impulsados por MLX Swift y CoreML.

📖 Read in: [English](README.md) · [中文](README_zh.md) · [日本語](README_ja.md) · [한국어](README_ko.md) · [Español](README_es.md) · [Deutsch](README_de.md) · [Français](README_fr.md) · [हिन्दी](README_hi.md) · [Português](README_pt.md) · [Русский](README_ru.md)

Reconocimiento, síntesis y comprensión de voz en el dispositivo para Mac e iOS. Se ejecuta localmente en Apple Silicon — sin nube, sin claves de API, ningún dato sale del dispositivo.

[Instala mediante Homebrew](#homebrew) o añádelo como dependencia de Swift Package.

**[Documentación](https://soniqo.audio)** · **[Modelos en HuggingFace](https://huggingface.co/aufklarer)** · **[Blog](https://blog.ivan.digital)**

- **Qwen3-ASR** — Voz a texto / reconocimiento de voz (reconocimiento automático del habla, 52 idiomas)
- **Parakeet TDT** — Voz a texto vía CoreML (Neural Engine, NVIDIA FastConformer + decodificador TDT, 25 idiomas)
- **Qwen3-ForcedAligner** — Alineación de marcas temporales a nivel de palabra (audio + texto → marcas temporales)
- **Qwen3-TTS** — Síntesis de texto a voz (máxima calidad, streaming, hablantes personalizados, 10 idiomas)
- **CosyVoice TTS** — Texto a voz con streaming, clonación de voz, diálogo multi-hablante y etiquetas de emoción (9 idiomas, DiT flow matching, codificador de hablante CAM++)
- **Kokoro TTS** — Texto a voz en el dispositivo (82M parámetros, CoreML/Neural Engine, 54 voces, listo para iOS, 10 idiomas)
- **Qwen3.5-Chat** — Chat LLM en el dispositivo (0.8B, MLX + CoreML, INT4 + CoreML INT8, DeltaNet hibrido, tokens en streaming)
- **PersonaPlex** — Conversación de voz a voz en full-duplex (7B, audio de entrada → audio de salida, 18 presets de voz)
- **DeepFilterNet3** — Mejora de voz / supresión de ruido (2.1M parámetros, tiempo real 48kHz)
- **FireRedVAD** — Detección de actividad vocal offline (DFSMN, CoreML, más de 100 idiomas, 97.6% F1)
- **Silero VAD** — Detección de actividad vocal en streaming (fragmentos de 32ms, latencia sub-milisegundo)
- **Pyannote VAD** — Detección de actividad vocal offline (ventanas de 10s, superposición de múltiples hablantes)
- **Speaker Diarization** — Quién habló cuándo (segmentación Pyannote + encadenamiento de hablantes basado en actividad, o Sortformer de extremo a extremo en Neural Engine)
- **Speaker Embeddings** — Verificación e identificación de hablantes (WeSpeaker ResNet34, vectores de 256 dimensiones)

Papers: [Qwen3-ASR](https://arxiv.org/abs/2601.21337) (Alibaba), [Qwen3-TTS](https://arxiv.org/abs/2601.15621) (Alibaba), [Qwen3](https://arxiv.org/abs/2505.09388) (Alibaba), [Parakeet TDT](https://arxiv.org/abs/2304.06795) (NVIDIA), [CosyVoice 3](https://arxiv.org/abs/2505.17589) (Alibaba), [Kokoro](https://arxiv.org/abs/2301.01695) (StyleTTS 2), [PersonaPlex](https://arxiv.org/abs/2602.06053) (NVIDIA), [Mimi](https://arxiv.org/abs/2410.00037) (Kyutai), [Sortformer](https://arxiv.org/abs/2409.06656) (NVIDIA)

## Hoja de ruta

Consulta la [discusión sobre la hoja de ruta](https://github.com/soniqo/speech-swift/discussions/81) para ver lo planificado — ¡comentarios y sugerencias son bienvenidos!

## Novedades

- **20 Mar 2026** — [We Beat Whisper Large v3 with a 600M Model Running Entirely on Your Mac](https://blog.ivan.digital/we-beat-whisper-large-v3-with-a-600m-model-running-entirely-on-your-mac-20e6ce191174)
- **26 Feb 2026** — [Speaker Diarization and Voice Activity Detection on Apple Silicon — Native Swift with MLX](https://blog.ivan.digital/speaker-diarization-and-voice-activity-detection-on-apple-silicon-native-swift-with-mlx-92ea0c9aca0f)
- **23 Feb 2026** — [NVIDIA PersonaPlex 7B on Apple Silicon — Full-Duplex Speech-to-Speech in Native Swift with MLX](https://blog.ivan.digital/nvidia-personaplex-7b-on-apple-silicon-full-duplex-speech-to-speech-in-native-swift-with-mlx-0aa5276f2e23)
- **12 Feb 2026** — [Qwen3-ASR Swift: On-Device ASR + TTS for Apple Silicon — Architecture and Benchmarks](https://blog.ivan.digital/qwen3-asr-swift-on-device-asr-tts-for-apple-silicon-architecture-and-benchmarks-27cbf1e4463f)

## Modelos

| Modelo | Tarea | Streaming | Idiomas | Tamaños |
|--------|-------|-----------|---------|---------|
| Qwen3-ASR-0.6B | Voz → Texto | No | 52 idiomas | [4-bit](https://huggingface.co/aufklarer/Qwen3-ASR-0.6B-MLX-4bit) 680 MB · [8-bit](https://huggingface.co/aufklarer/Qwen3-ASR-0.6B-MLX-8bit) 1.0 GB · [CoreML](https://huggingface.co/aufklarer/Qwen3-ASR-CoreML) 180 MB |
| Qwen3-ASR-1.7B | Voz → Texto | No | 52 idiomas | [4-bit](https://huggingface.co/aufklarer/Qwen3-ASR-1.7B-MLX-4bit) 2.1 GB · [8-bit](https://huggingface.co/aufklarer/Qwen3-ASR-1.7B-MLX-8bit) 3.2 GB |
| Parakeet-TDT-0.6B | Voz → Texto | No | 25 idiomas europeos | [CoreML INT4](https://huggingface.co/aufklarer/Parakeet-TDT-v3-CoreML-INT4) 315 MB · [CoreML INT8](https://huggingface.co/aufklarer/Parakeet-TDT-v3-CoreML-INT8) 500 MB |
| Qwen3-ForcedAligner-0.6B | Audio + Texto → Marcas temporales | No | Multi | [4-bit](https://huggingface.co/aufklarer/Qwen3-ForcedAligner-0.6B-4bit) 979 MB · [8-bit](https://huggingface.co/aufklarer/Qwen3-ForcedAligner-0.6B-8bit) 1.4 GB · [CoreML INT4](https://huggingface.co/aufklarer/Qwen3-ForcedAligner-0.6B-CoreML-INT4) 630 MB · [CoreML INT8](https://huggingface.co/aufklarer/Qwen3-ForcedAligner-0.6B-CoreML-INT8) 1.0 GB |
| Qwen3-TTS-0.6B Base | Texto → Voz | Sí (~120ms) | 10 idiomas | [4-bit](https://huggingface.co/aufklarer/Qwen3-TTS-12Hz-0.6B-Base-MLX-4bit) 1.7 GB · [8-bit](https://huggingface.co/aufklarer/Qwen3-TTS-12Hz-0.6B-Base-MLX-8bit) 2.4 GB |
| Qwen3-TTS-0.6B CustomVoice | Texto → Voz | Sí (~120ms) | 10 idiomas | [4-bit](https://huggingface.co/aufklarer/Qwen3-TTS-12Hz-0.6B-CustomVoice-MLX-4bit) 1.7 GB |
| Qwen3-TTS-1.7B Base | Texto → Voz | Sí (~120ms) | 10 idiomas | [4-bit](https://huggingface.co/aufklarer/Qwen3-TTS-12Hz-1.7B-Base-MLX-4bit) 3.2 GB · [8-bit](https://huggingface.co/aufklarer/Qwen3-TTS-12Hz-1.7B-Base-MLX-8bit) 4.8 GB |
| CosyVoice3-0.5B | Texto → Voz | Sí (~150ms) | 9 idiomas | [4-bit](https://huggingface.co/aufklarer/CosyVoice3-0.5B-MLX-4bit) 1.2 GB |
| Kokoro-82M | Texto → Voz | No | 10 idiomas | [CoreML](https://huggingface.co/aufklarer/Kokoro-82M-CoreML) ~170 MB |
| Qwen3.5-0.8B Chat | Text → Text (LLM) | Yes (streaming) | Multi | [MLX INT4](https://huggingface.co/aufklarer/Qwen3.5-0.8B-Chat-MLX) 418 MB · [CoreML INT8](https://huggingface.co/aufklarer/Qwen3.5-0.8B-Chat-CoreML) 981 MB |
| PersonaPlex-7B | Voz → Voz | Sí (~2s fragmentos) | EN | [4-bit](https://huggingface.co/aufklarer/PersonaPlex-7B-MLX-4bit) 4.9 GB · [8-bit](https://huggingface.co/aufklarer/PersonaPlex-7B-MLX-8bit) 9.1 GB |
| FireRedVAD | Detección de actividad vocal | No (offline) | Más de 100 idiomas | [CoreML](https://huggingface.co/aufklarer/FireRedVAD-CoreML) ~1.2 MB |
| Silero-VAD-v5 | Detección de actividad vocal | Sí (fragmentos de 32ms) | Independiente del idioma | [MLX](https://huggingface.co/aufklarer/Silero-VAD-v5-MLX) · [CoreML](https://huggingface.co/aufklarer/Silero-VAD-v5-CoreML) ~1.2 MB |
| Pyannote-Segmentation-3.0 | VAD + Segmentación de hablantes | No (ventanas de 10s) | Independiente del idioma | [MLX](https://huggingface.co/aufklarer/Pyannote-Segmentation-MLX) ~5.7 MB |
| DeepFilterNet3 | Mejora de voz | Sí (tramas de 10ms) | Independiente del idioma | [CoreML FP16](https://huggingface.co/aufklarer/DeepFilterNet3-CoreML) ~4.2 MB |
| WeSpeaker-ResNet34-LM | Embedding de hablante (256-dim) | No | Independiente del idioma | [MLX](https://huggingface.co/aufklarer/WeSpeaker-ResNet34-LM-MLX) · [CoreML](https://huggingface.co/aufklarer/WeSpeaker-ResNet34-LM-CoreML) ~25 MB |
| CAM++ | Embedding de hablante (192-dim) | No | Independiente del idioma | [CoreML](https://huggingface.co/aufklarer/CamPlusPlus-Speaker-CoreML) ~14 MB |
| Sortformer | Diarización de hablantes (extremo a extremo) | Sí (fragmentado) | Independiente del idioma | [CoreML](https://huggingface.co/aufklarer/Sortformer-Diarization-CoreML) ~240 MB |

### Requisitos de memoria

La memoria de pesos es la memoria de GPU (MLX) o ANE (CoreML) consumida por los parámetros del modelo. La inferencia pico incluye cachés KV, activaciones y tensores intermedios.

| Modelo | Memoria de pesos | Inferencia pico |
|--------|-----------------|-----------------|
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

### Cuándo usar cada TTS

- **Qwen3-TTS**: Mejor calidad, streaming (~120ms), 9 hablantes integrados, 10 idiomas, síntesis por lotes
- **CosyVoice TTS**: Streaming (~150ms), 9 idiomas, clonación de voz (codificador de hablante CAM++), diálogo multi-hablante (`[S1] ... [S2] ...`), etiquetas de emoción/estilo en línea (`(happy)`, `(whispers)`), DiT flow matching + vocoder HiFi-GAN
- **Kokoro TTS**: TTS ligero listo para iOS (82M parámetros), CoreML/Neural Engine, 54 voces, 10 idiomas, no autorregresivo (pipeline de 3 etapas)
- **PersonaPlex**: Voz a voz en full-duplex (audio de entrada → audio de salida), streaming (~2s fragmentos), 18 presets de voz, basado en la arquitectura Moshi

## Instalación

### Homebrew

Requiere Homebrew nativo ARM (`/opt/homebrew`). Homebrew bajo Rosetta/x86_64 no es compatible.

```bash
brew tap soniqo/speech https://github.com/soniqo/speech-swift
brew install speech
```

Luego usa:

```bash
audio transcribe recording.wav
audio speak "Hello world"
audio speak "Hallo Welt" --engine cosyvoice --language german
audio respond --input question.wav --transcript
```

> Para conversación de voz interactiva con entrada de micrófono, consulta **[PersonaPlexDemo](Examples/PersonaPlexDemo/)**.

### Swift Package Manager

Añádelo a tu `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/soniqo/speech-swift", branch: "main")
]
```

Importa el módulo que necesites:

```swift
import Qwen3ASR      // Reconocimiento de voz (MLX)
import ParakeetASR   // Reconocimiento de voz (CoreML)
import Qwen3TTS      // Texto a voz (Qwen3)
import CosyVoiceTTS  // Texto a voz (streaming)
import KokoroTTS     // Texto a voz (CoreML, listo para iOS)
import Qwen3Chat     // Chat LLM en el dispositivo (CoreML)
import PersonaPlex   // Voz a voz (full-duplex)
import SpeechVAD          // Detección de actividad vocal (pyannote + Silero)
import SpeechEnhancement  // Supresión de ruido (DeepFilterNet3)
import AudioCommon        // Utilidades compartidas
```

### Requisitos

- Swift 5.9+
- macOS 14+ o iOS 17+
- Apple Silicon (M1/M2/M3/M4)
- Xcode 15+ (con Metal Toolchain — ejecuta `xcodebuild -downloadComponent MetalToolchain` si falta)

### Compilar desde el código fuente

```bash
git clone https://github.com/soniqo/speech-swift
cd speech-swift
make build
```

Esto compila el paquete Swift **y** la biblioteca de shaders Metal de MLX en un solo paso. La biblioteca Metal (`mlx.metallib`) es necesaria para la inferencia en GPU — sin ella obtendrás `Failed to load the default metallib` en tiempo de ejecución.

Para compilaciones de depuración: `make debug`. Para ejecutar pruebas unitarias: `make test`.

## Prueba el asistente de voz

**[PersonaPlexDemo](Examples/PersonaPlexDemo/)** es un asistente de voz para macOS listo para ejecutar — toca para hablar y obtén respuestas habladas en tiempo real. Utiliza entrada de micrófono con Silero VAD para detección automática del habla, Qwen3-ASR para transcripción y PersonaPlex 7B para generación de voz a voz. Conversación multi-turno con 18 presets de voz y visualización de la transcripción del monólogo interno.

```bash
make build  # desde la raíz del repositorio — compila todo incluyendo el metallib de MLX
cd Examples/PersonaPlexDemo
# Consulta Examples/PersonaPlexDemo/README.md para instrucciones del bundle .app
```

> RTF ~0.94 en M2 Max (más rápido que tiempo real). Los modelos se descargan automáticamente en la primera ejecución (~5.5 GB PersonaPlex + ~400 MB ASR).

## Aplicaciones de demostración

- **[PersonaPlexDemo](Examples/PersonaPlexDemo/)** — Asistente de voz conversacional (entrada de micrófono, VAD, multi-turno). Ver arriba.
- **[SpeechDemo](Examples/SpeechDemo/)** — Dictado (Parakeet TDT / Qwen3-ASR con detección automática de idioma) y síntesis de texto a voz (Qwen3-TTS) en una interfaz con pestañas.

Compila y ejecuta como bundle `.app` de macOS — consulta el README de cada demo para instrucciones.

## Voz a texto (ASR) — Transcribir audio en Swift

### Transcripción básica

```swift
import Qwen3ASR

// Por defecto: modelo 0.6B
let model = try await Qwen3ASRModel.fromPretrained()

// O usa el modelo más grande 1.7B para mejor precisión
let model = try await Qwen3ASRModel.fromPretrained(
    modelId: "aufklarer/Qwen3-ASR-1.7B-MLX-8bit"
)

// El audio puede tener cualquier tasa de muestreo — se remuestrea automáticamente a 16kHz internamente
let transcription = model.transcribe(audio: audioSamples, sampleRate: 16000)
print(transcription)
```

### Codificador CoreML (Neural Engine)

Modo híbrido: codificador CoreML en Neural Engine + decodificador de texto MLX en GPU. Menor consumo energético, libera la GPU para el paso del codificador.

```swift
import Qwen3ASR

let encoder = try await CoreMLASREncoder.fromPretrained()
let model = try await Qwen3ASRModel.fromPretrained()
let text = try model.transcribe(audio: audioSamples, sampleRate: 16000, coremlEncoder: encoder)
```

Variantes INT8 (180 MB, por defecto) e INT4 (90 MB) disponibles. Se recomienda INT8 (similitud coseno > 0.999 respecto a FP32).

### Parakeet TDT (CoreML)

```swift
import ParakeetASR

let model = try await ParakeetASRModel.fromPretrained()
let transcription = model.transcribe(audio: audioSamples, sampleRate: 16000)
```

Se ejecuta en el Neural Engine vía CoreML — libera la GPU para cargas de trabajo simultáneas. 25 idiomas europeos, ~315 MB.

### CLI de ASR

```bash
make build  # o: swift build -c release && ./scripts/build_mlx_metallib.sh release

# Por defecto (Qwen3-ASR 0.6B, MLX)
.build/release/audio transcribe audio.wav

# Usar el modelo 1.7B
.build/release/audio transcribe audio.wav --model 1.7B

# Codificador CoreML (Neural Engine + decodificador MLX)
.build/release/audio transcribe --engine qwen3-coreml audio.wav

# Parakeet TDT (CoreML, Neural Engine)
.build/release/audio transcribe --engine parakeet audio.wav
```

## Alineación forzada

### Marcas temporales a nivel de palabra

```swift
import Qwen3ASR

let aligner = try await Qwen3ForcedAligner.fromPretrained()
// Descarga ~979 MB en la primera ejecución

let aligned = aligner.align(
    audio: audioSamples,
    text: "Can you guarantee that the replacement part will be shipped tomorrow?",
    sampleRate: 24000
)

for word in aligned {
    print("[\(String(format: "%.2f", word.startTime))s - \(String(format: "%.2f", word.endTime))s] \(word.text)")
}
```

### CLI de alineación forzada

```bash
swift build -c release

# Alinear con texto proporcionado
.build/release/audio align audio.wav --text "Hello world"

# Transcribir primero, luego alinear
.build/release/audio align audio.wav
```

Salida:
```
[0.12s - 0.45s] Can
[0.45s - 0.72s] you
[0.72s - 1.20s] guarantee
...
```

No autorregresivo — pipeline de 3 etapas, sin bucle de muestreo. Consulta [Forced Aligner](docs/inference/forced-aligner.md) para detalles de la arquitectura.

## Texto a voz (TTS) — Generar voz en Swift

### Síntesis básica

```swift
import Qwen3TTS
import AudioCommon  // para WAVWriter

let model = try await Qwen3TTSModel.fromPretrained()
// Descarga ~1.7 GB en la primera ejecución (modelo + pesos del códec)
let audio = model.synthesize(text: "Hello world", language: "english")
// La salida son muestras float mono a 24kHz
try WAVWriter.write(samples: audio, sampleRate: 24000, to: outputURL)
```

### CLI de TTS

```bash
make build
.build/release/audio speak "Hello world" --output output.wav --language english
```

### Voz personalizada / Selección de hablante

La variante del modelo **CustomVoice** admite 9 voces de hablantes integradas e instrucciones en lenguaje natural para controlar tono y estilo. Cárgalo pasando el ID del modelo CustomVoice:

```swift
import Qwen3TTS

// Cargar el modelo CustomVoice (descarga ~1.7 GB en la primera ejecución)
let model = try await Qwen3TTSModel.fromPretrained(
    modelId: TTSModelVariant.customVoice.rawValue
)

// Sintetizar con un hablante específico
let audio = model.synthesize(text: "Hello world", language: "english", speaker: "vivian")

// Listar hablantes disponibles
print(model.availableSpeakers)  // ["aiden", "dylan", "eric", ...]
```

CLI:

```bash
# Usar el modelo CustomVoice con un hablante
.build/release/audio speak "Hello world" --model customVoice --speaker vivian --output vivian.wav

# Listar hablantes disponibles
.build/release/audio speak --model customVoice --list-speakers
```

### Clonación de voz (modelo Base)

Clona la voz de un hablante a partir de un archivo de audio de referencia. Dos modos:

**Modo ICL** (recomendado) — codifica el audio de referencia en tokens de códec con transcripción. Mayor calidad, EOS fiable:

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

**Modo X-vector** — solo embedding del hablante, no necesita transcripción pero menor calidad:

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

### Instrucciones de tono / estilo (solo CustomVoice)

El modelo CustomVoice acepta un parámetro `instruct` en lenguaje natural para controlar el estilo de habla, tono, emoción y ritmo. La instrucción se antepone a la entrada del modelo en formato ChatML.

```swift
// Tono alegre
let audio = model.synthesize(
    text: "Welcome to our store!",
    language: "english",
    speaker: "ryan",
    instruct: "Speak in a cheerful, upbeat tone"
)

// Lento y serio
let audio = model.synthesize(
    text: "We regret to inform you...",
    language: "english",
    speaker: "aiden",
    instruct: "Read this slowly and solemnly"
)

// Susurrando
let audio = model.synthesize(
    text: "Can you keep a secret?",
    language: "english",
    speaker: "vivian",
    instruct: "Whisper this softly"
)
```

CLI:

```bash
# Con instrucción de estilo
.build/release/audio speak "Good morning!" --model customVoice --speaker ryan \
    --instruct "Speak in a cheerful, upbeat tone" --output cheerful.wav

# La instrucción por defecto ("Speak naturally.") se aplica automáticamente al usar CustomVoice
.build/release/audio speak "Hello world" --model customVoice --speaker ryan --output natural.wav
```

Cuando no se proporciona `--instruct` con el modelo CustomVoice, se aplica `"Speak naturally."` automáticamente para evitar salida divagante. El modelo Base no admite instruct.

### Síntesis por lotes

Sintetiza múltiples textos en un solo paso forward por lotes para mayor rendimiento:

```swift
let texts = ["Good morning everyone.", "The weather is nice today.", "Please open the window."]
let audioList = model.synthesizeBatch(texts: texts, language: "english", maxBatchSize: 4)
// audioList[i] son muestras float mono a 24kHz para texts[i]
for (i, audio) in audioList.enumerated() {
    try WAVWriter.write(samples: audio, sampleRate: 24000, to: URL(fileURLWithPath: "output_\(i).wav"))
}
```

#### CLI por lotes

```bash
# Crea un archivo con un texto por línea
echo "Hello world.\nGoodbye world." > texts.txt
.build/release/audio speak --batch-file texts.txt --output output.wav --batch-size 4
# Produce output_0.wav, output_1.wav, ...
```

> El modo por lotes amortiza las cargas de pesos del modelo entre elementos. Se espera una mejora de rendimiento de ~1.5-2.5x para B=4 en Apple Silicon. Mejores resultados cuando los textos producen audio de duración similar.

### Opciones de muestreo

```swift
let config = SamplingConfig(temperature: 0.9, topK: 50, repetitionPenalty: 1.05)
let audio = model.synthesize(text: "Hello", language: "english", sampling: config)
```

### Síntesis en streaming

Emite fragmentos de audio incrementalmente para baja latencia del primer paquete:

```swift
let stream = model.synthesizeStream(
    text: "Hello, this is streaming synthesis.",
    language: "english",
    streaming: .lowLatency  // ~120ms hasta el primer fragmento de audio
)

for try await chunk in stream {
    // chunk.samples: [Float] PCM @ 24kHz
    // chunk.isFinal: true en el último fragmento
    playAudio(chunk.samples)
}
```

CLI:

```bash
# Streaming por defecto (primer fragmento de 3 tramas, ~225ms de latencia)
.build/release/audio speak "Hello world" --stream

# Baja latencia (primer fragmento de 1 trama, ~120ms de latencia)
.build/release/audio speak "Hello world" --stream --first-chunk-frames 1
```

## Voz a voz — Conversación de voz en full-duplex

> Para un asistente de voz interactivo con entrada de micrófono, consulta **[PersonaPlexDemo](Examples/PersonaPlexDemo/)** — toca para hablar, conversación multi-turno con detección automática del habla.

### Voz a voz

```swift
import PersonaPlex
import AudioCommon  // para WAVWriter, AudioFileLoader

let model = try await PersonaPlexModel.fromPretrained()
// Descarga ~5.5 GB en la primera ejecución (temporal 4-bit + depformer + códec Mimi + presets de voz)

let audio = try AudioFileLoader.load(url: inputURL, targetSampleRate: 24000)
let (response, textTokens) = model.respond(userAudio: audio, voice: .NATM0)
// response: muestras float mono a 24kHz
// textTokens: monólogo interno del modelo (IDs de tokens SentencePiece)
try WAVWriter.write(samples: response.audio, sampleRate: 24000, to: outputURL)
```

### Monólogo interno (salida de texto)

PersonaPlex genera tokens de texto junto con el audio — el razonamiento interno del modelo. Decodifícalos con el decodificador SentencePiece integrado:

```swift
let decoder = try SentencePieceDecoder(modelPath: "tokenizer_spm_32k_3.model")
let transcript = decoder.decode(textTokens)
print(transcript)  // ej. "Sure, I can help you with that..."
```

### Voz a voz en streaming

```swift
// Recibe fragmentos de audio conforme se generan (~2s por fragmento)
let stream = model.respondStream(userAudio: audio, voice: .NATM0)
for try await chunk in stream {
    playAudio(chunk.samples)  // reproduce inmediatamente, mono 24kHz
    // chunk.textTokens tiene el texto de este fragmento; el fragmento final tiene todos los tokens
    if chunk.isFinal { break }
}
```

### Selección de voz

18 presets de voz disponibles:
- **Natural Femenina**: NATF0, NATF1, NATF2, NATF3
- **Natural Masculina**: NATM0, NATM1, NATM2, NATM3
- **Variedad Femenina**: VARF0, VARF1, VARF2, VARF3, VARF4
- **Variedad Masculina**: VARM0, VARM1, VARM2, VARM3, VARM4

### Prompts de sistema

El prompt de sistema guía el comportamiento conversacional del modelo. Puedes pasar cualquier prompt personalizado como una cadena de texto:

```swift
// Prompt de sistema personalizado (tokenizado automáticamente)
let response = model.respond(
    userAudio: audio,
    voice: .NATM0,
    systemPrompt: "You enjoy having a good conversation."
)

// O usar un preset
let response = model.respond(
    userAudio: audio,
    voice: .NATM0,
    systemPromptTokens: SystemPromptPreset.customerService.tokens
)
```

Presets disponibles: `focused` (por defecto), `assistant`, `customerService`, `teacher`.

### CLI de PersonaPlex

```bash
make build

# Voz a voz básico
.build/release/audio respond --input question.wav --output response.wav

# Con transcripción (decodifica el texto del monólogo interno)
.build/release/audio respond --input question.wav --transcript

# Salida JSON (ruta del audio, transcripción, métricas de latencia)
.build/release/audio respond --input question.wav --json

# Texto de prompt de sistema personalizado
.build/release/audio respond --input question.wav --system-prompt-text "You enjoy having a good conversation."

# Elegir una voz y preset de prompt de sistema
.build/release/audio respond --input question.wav --voice NATF1 --system-prompt focused

# Ajustar parámetros de muestreo
.build/release/audio respond --input question.wav --audio-temp 0.6 --repetition-penalty 1.5

# Activar parada temprana por entropía de texto (detiene si el texto colapsa)
.build/release/audio respond --input question.wav --entropy-threshold 1.0 --entropy-window 5

# Listar voces y prompts disponibles
.build/release/audio respond --list-voices
.build/release/audio respond --list-prompts
```

## CosyVoice TTS — Texto a voz en streaming con clonación de voz

### Síntesis básica

```swift
import CosyVoiceTTS
import AudioCommon  // para WAVWriter

let model = try await CosyVoiceTTSModel.fromPretrained()
// Descarga ~1.9 GB en la primera ejecución (pesos LLM + DiT + HiFi-GAN)

let audio = model.synthesize(text: "Hello, how are you today?", language: "english")
// La salida son muestras float mono a 24kHz
try WAVWriter.write(samples: audio, sampleRate: 24000, to: outputURL)
```

### Síntesis en streaming

```swift
// Streaming: recibe fragmentos de audio conforme se generan (~150ms hasta el primer fragmento)
for try await chunk in model.synthesizeStream(text: "Hello, how are you today?", language: "english") {
    // chunk.audio: [Float], chunk.sampleRate: Int
    playAudio(chunk.audio)  // reproduce inmediatamente
}
```

### Clonación de voz (CosyVoice)

Clona la voz de un hablante usando el codificador de hablante CAM++ (192-dim, CoreML Neural Engine):

```swift
import CosyVoiceTTS
import AudioCommon

let model = try await CosyVoiceTTSModel.fromPretrained()
let speaker = try await CamPlusPlusSpeaker.fromPretrained()
// Descarga ~14 MB del modelo CoreML CAM++ en el primer uso

let refAudio = try AudioFileLoader.load(url: referenceURL, targetSampleRate: 16000)
let embedding = try speaker.embed(audio: refAudio, sampleRate: 16000)
// embedding: [Float] de longitud 192

let audio = model.synthesize(
    text: "Hello in a cloned voice!",
    language: "english",
    speakerEmbedding: embedding
)
```

### CLI de CosyVoice TTS

```bash
make build

# Síntesis básica
.build/release/audio speak "Hello world" --engine cosyvoice --language english --output output.wav

# Clonación de voz (descarga el codificador de hablante CAM++ en el primer uso)
.build/release/audio speak "Hello world" --engine cosyvoice --voice-sample reference.wav --output cloned.wav

# Diálogo multi-hablante con clonación de voz
.build/release/audio speak "[S1] Hello there! [S2] Hey, how are you?" \
    --engine cosyvoice --speakers s1=alice.wav,s2=bob.wav -o dialogue.wav

# Etiquetas de emoción/estilo en línea
.build/release/audio speak "(excited) Wow, amazing! (sad) But I have to go..." \
    --engine cosyvoice -o emotion.wav

# Combinado: diálogo + emociones + clonación de voz
.build/release/audio speak "[S1] (happy) Great news! [S2] (surprised) Really?" \
    --engine cosyvoice --speakers s1=alice.wav,s2=bob.wav -o combined.wav

# Instrucción de estilo personalizada
.build/release/audio speak "Hello world" --engine cosyvoice --cosy-instruct "Speak cheerfully" -o cheerful.wav

# Síntesis en streaming
.build/release/audio speak "Hello world" --engine cosyvoice --language english --stream --output output.wav
```

## Kokoro TTS — Texto a voz ligero en el dispositivo (iOS + macOS)

### Síntesis básica

```swift
import KokoroTTS
import AudioCommon  // para WAVWriter

let tts = try await KokoroTTSModel.fromPretrained()
// Descarga ~170 MB en la primera ejecución (modelos CoreML + embeddings de voz + diccionarios)

let audio = try tts.synthesize(text: "Hello world", voice: "af_heart")
// La salida son muestras float mono a 24kHz
try WAVWriter.write(samples: audio, sampleRate: 24000, to: outputURL)
```

54 voces preconfiguradas en 10 idiomas. No autorregresivo — pipeline de 3 etapas en CoreML, sin bucle de muestreo. Se ejecuta en el Neural Engine, libera completamente la GPU.

### CLI de Kokoro TTS

```bash
make build

# Síntesis básica
.build/release/audio kokoro "Hello world" --voice af_heart --output hello.wav

# Elegir idioma
.build/release/audio kokoro "Bonjour le monde" --voice ff_siwis --language fr --output bonjour.wav

# Listar voces disponibles
.build/release/audio kokoro --list-voices
```

## Qwen3 Chat (LLM en el dispositivo)

```swift
import Qwen3Chat

let chat = try await Qwen3ChatModel.fromPretrained()
// Descarga ~318 MB en la primera ejecución (modelo CoreML INT4 + tokenizador)

// Respuesta única
let response = try chat.generate("What is Swift?", systemPrompt: "Answer briefly.")
print(response)

// Tokens en streaming
let stream = chat.chatStream("Tell me a joke", systemPrompt: "Be funny.")
for try await token in stream {
    print(token, terminator: "")
}
```

Qwen3-0.6B cuantizado a INT4 para CoreML. Se ejecuta en el Neural Engine con ~2 tok/s en iPhone, ~15 tok/s en chips M-series. Admite conversación multi-turno con caché KV, modo de razonamiento (tokens `<think>`), y muestreo configurable (temperature, top-k, top-p, repetition penalty).

## Detección de actividad vocal (VAD) — Detectar voz en audio

### VAD en streaming (Silero)

Silero VAD v5 procesa fragmentos de audio de 32ms con latencia sub-milisegundo — ideal para detección de voz en tiempo real desde micrófonos o flujos de audio.

```swift
import SpeechVAD

let vad = try await SileroVADModel.fromPretrained()
// O usa CoreML (Neural Engine, menor consumo):
// let vad = try await SileroVADModel.fromPretrained(engine: .coreml)

// Streaming: procesa fragmentos de 512 muestras (32ms @ 16kHz)
let prob = vad.processChunk(samples)  // → 0.0...1.0
vad.resetState()  // llamar entre diferentes flujos de audio

// O detecta todos los segmentos de una vez
let segments = vad.detectSpeech(audio: audioSamples, sampleRate: 16000)
for seg in segments {
    print("Speech: \(seg.startTime)s - \(seg.endTime)s")
}
```

### Streaming basado en eventos

```swift
let processor = StreamingVADProcessor(model: vad)

// Alimenta audio de cualquier longitud — los eventos se emiten cuando se confirma el habla
let events = processor.process(samples: audioBuffer)
for event in events {
    switch event {
    case .speechStarted(let time):
        print("Speech started at \(time)s")
    case .speechEnded(let segment):
        print("Speech: \(segment.startTime)s - \(segment.endTime)s")
    }
}

// Vaciar al final del flujo
let final = processor.flush()
```

### CLI de VAD

```bash
make build

# Silero VAD en streaming (fragmentos de 32ms)
.build/release/audio vad-stream audio.wav

# Backend CoreML (Neural Engine)
.build/release/audio vad-stream audio.wav --engine coreml

# Con umbrales personalizados
.build/release/audio vad-stream audio.wav --onset 0.6 --offset 0.4

# Salida JSON
.build/release/audio vad-stream audio.wav --json

# VAD por lotes con Pyannote (ventanas deslizantes de 10s)
.build/release/audio vad audio.wav
```

## Diarización de hablantes — Quién habló cuándo

### Pipeline de diarización

```swift
import SpeechVAD

let pipeline = try await DiarizationPipeline.fromPretrained()
// O usa embeddings CoreML (Neural Engine, libera la GPU):
// let pipeline = try await DiarizationPipeline.fromPretrained(embeddingEngine: .coreml)

let result = pipeline.diarize(audio: samples, sampleRate: 16000)
for seg in result.segments {
    print("Speaker \(seg.speakerId): [\(seg.startTime)s - \(seg.endTime)s]")
}
print("\(result.numSpeakers) speakers detected")
```

### Embedding de hablante

```swift
let model = try await WeSpeakerModel.fromPretrained()
// O: let model = try await WeSpeakerModel.fromPretrained(engine: .coreml)
let embedding = model.embed(audio: samples, sampleRate: 16000)
// embedding: [Float] de longitud 256, normalizado L2

// Comparar hablantes
let similarity = WeSpeakerModel.cosineSimilarity(embeddingA, embeddingB)
```

### Extracción de hablante

Extrae solo los segmentos de un hablante específico usando una grabación de referencia:

```swift
let pipeline = try await DiarizationPipeline.fromPretrained()
let targetEmb = pipeline.embeddingModel.embed(audio: enrollmentAudio, sampleRate: 16000)
let segments = pipeline.extractSpeaker(
    audio: meetingAudio, sampleRate: 16000,
    targetEmbedding: targetEmb
)
```

### Diarización con Sortformer (extremo a extremo, CoreML)

NVIDIA Sortformer predice la actividad de hablante por trama para hasta 4 hablantes directamente — sin necesidad de embedding ni clustering. Se ejecuta en el Neural Engine.

```swift
let diarizer = try await SortformerDiarizer.fromPretrained()
let result = diarizer.diarize(audio: samples, sampleRate: 16000, config: .default)
for seg in result.segments {
    print("Speaker \(seg.speakerId): [\(seg.startTime)s - \(seg.endTime)s]")
}
```

### CLI de diarización

```bash
make build

# Diarización con Pyannote (por defecto)
.build/release/audio diarize meeting.wav

# Diarización con Sortformer (CoreML, Neural Engine)
.build/release/audio diarize meeting.wav --engine sortformer

# Embeddings CoreML (Neural Engine, solo pyannote)
.build/release/audio diarize meeting.wav --embedding-engine coreml

# Salida JSON
.build/release/audio diarize meeting.wav --json

# Extraer un hablante específico (solo pyannote)
.build/release/audio diarize meeting.wav --target-speaker enrollment.wav

# Embedding de hablante
.build/release/audio embed-speaker enrollment.wav --json
.build/release/audio embed-speaker enrollment.wav --engine coreml
```

Consulta [Speaker Diarization](docs/inference/speaker-diarization.md) para detalles de la arquitectura.

## Mejora de voz — Supresión de ruido y limpieza de audio

### Supresión de ruido

```swift
import SpeechEnhancement
import AudioCommon  // para WAVWriter

let enhancer = try await SpeechEnhancer.fromPretrained()
// Descarga ~4.3 MB en la primera ejecución (modelo Core ML FP16 + datos auxiliares)

let cleanAudio = try enhancer.enhance(audio: noisyAudio, sampleRate: 48000)
try WAVWriter.write(samples: cleanAudio, sampleRate: 48000, to: outputURL)
```

### CLI de eliminación de ruido

```bash
make build

# Eliminación básica de ruido
.build/release/audio denoise noisy.wav

# Ruta de salida personalizada
.build/release/audio denoise noisy.wav --output clean.wav
```

Consulta [Speech Enhancement](docs/inference/speech-enhancement.md) para detalles de la arquitectura.

## Pipelines — Componer múltiples modelos

Todos los modelos se ajustan a protocolos compartidos (`SpeechRecognitionModel`, `SpeechGenerationModel`, `SpeechEnhancementModel`, etc.) y pueden componerse en pipelines:

### Reconocimiento de voz ruidosa (DeepFilterNet + ASR)

```swift
import SpeechEnhancement
import Qwen3ASR

let enhancer = try await SpeechEnhancer.fromPretrained()
let asr = try await Qwen3ASRModel.fromPretrained()

// Mejorar a 48kHz, luego transcribir a 16kHz
let clean = try enhancer.enhance(audio: noisyAudio, sampleRate: 48000)
let clean16k = AudioResampler.resample(clean, from: 48000, to: 16000)
let text = asr.transcribe(audio: clean16k, sampleRate: 16000)
```

### Relé de voz a voz (VAD + ASR + TTS)

```swift
import SpeechVAD
import Qwen3ASR
import Qwen3TTS

let vad = try await SileroVADModel.fromPretrained()
let asr = try await Qwen3ASRModel.fromPretrained()
let tts = try await Qwen3TTSModel.fromPretrained()

// Detectar segmentos de voz, transcribir, re-sintetizar
let segments = vad.detectSpeech(audio: audio, sampleRate: 16000)
for seg in segments {
    let chunk = Array(audio[Int(seg.startTime * 16000)..<Int(seg.endTime * 16000)])
    let text = asr.transcribe(audio: chunk, sampleRate: 16000)
    let speech = tts.synthesize(text: text, language: "english")
    // speech: muestras float mono a 24kHz
}
```

### Transcripción de reuniones (Diarización + ASR)

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

Consulta [Shared Protocols](docs/shared-protocols.md) para la referencia completa de protocolos.

## Servidor HTTP API

Un servidor HTTP independiente expone todos los modelos mediante endpoints REST y WebSocket. Los modelos se cargan bajo demanda en la primera solicitud.

```bash
swift build -c release
.build/release/audio-server --port 8080

# Transcribir audio
curl -X POST http://localhost:8080/transcribe --data-binary @audio.wav -H "Content-Type: audio/wav"

# Texto a voz
curl -X POST http://localhost:8080/speak -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "engine": "cosyvoice"}' -o output.wav

# Voz a voz (PersonaPlex)
curl -X POST http://localhost:8080/respond --data-binary @question.wav -o response.wav

# Mejora de voz
curl -X POST http://localhost:8080/enhance --data-binary @noisy.wav -o clean.wav

# Precargar todos los modelos al inicio
.build/release/audio-server --preload --port 8080
```

### Streaming por WebSocket

#### OpenAI Realtime API (`/v1/realtime`)

El endpoint principal de WebSocket implementa el protocolo [OpenAI Realtime API](https://platform.openai.com/docs/api-reference/realtime) — todos los mensajes son JSON con un campo `type`, el audio es PCM16 24kHz mono codificado en base64.

**Eventos del cliente al servidor:**

| Evento | Descripción |
|--------|-------------|
| `session.update` | Configurar motor, idioma, formato de audio |
| `input_audio_buffer.append` | Enviar fragmento de audio PCM16 en base64 |
| `input_audio_buffer.commit` | Transcribir audio acumulado (ASR) |
| `input_audio_buffer.clear` | Limpiar búfer de audio |
| `response.create` | Solicitar síntesis TTS |

**Eventos del servidor al cliente:**

| Evento | Descripción |
|--------|-------------|
| `session.created` | Sesión inicializada |
| `session.updated` | Configuración confirmada |
| `input_audio_buffer.committed` | Audio enviado para transcripción |
| `conversation.item.input_audio_transcription.completed` | Resultado de ASR |
| `response.audio.delta` | Fragmento de audio PCM16 en base64 (TTS) |
| `response.audio.done` | Streaming de audio completado |
| `response.done` | Respuesta completa con metadatos |
| `error` | Error con tipo y mensaje |

```javascript
const ws = new WebSocket('ws://localhost:8080/v1/realtime');

// ASR: enviar audio, obtener transcripción
ws.send(JSON.stringify({ type: 'input_audio_buffer.append', audio: base64PCM16 }));
ws.send(JSON.stringify({ type: 'input_audio_buffer.commit' }));
// → recibe: conversation.item.input_audio_transcription.completed

// TTS: enviar texto, obtener audio en streaming
ws.send(JSON.stringify({
  type: 'response.create',
  response: { modalities: ['audio', 'text'], instructions: 'Hello world' }
}));
// → recibe: response.audio.delta (fragmentos base64), response.audio.done, response.done
```

Un cliente HTML de ejemplo se encuentra en `Examples/websocket-client.html` — ábrelo en un navegador mientras el servidor está en ejecución.

El servidor es un módulo separado `AudioServer` y un ejecutable `audio-server` — no añade Hummingbird/WebSocket al CLI principal `audio`.

## Latencia (M2 Max, 64 GB)

### ASR

| Modelo | Backend | RTF | 10s de audio procesados en |
|--------|---------|-----|---------------------------|
| Qwen3-ASR-0.6B (4-bit) | MLX | ~0.06 | ~0.6s |
| Qwen3-ASR-0.6B (INT8) | CoreML + MLX | ~0.09 | ~0.9s |
| Qwen3-ASR-1.7B (8-bit) | MLX | ~0.11 | ~1.1s |
| Parakeet-TDT-0.6B (INT4) | CoreML (Neural Engine) | ~0.12 frío, ~0.03 caliente | ~1.2s / ~0.3s |
| Whisper-large-v3 | whisper.cpp (Q5_0) | ~0.10 | ~1.0s |
| Whisper-small | whisper.cpp (Q5_0) | ~0.04 | ~0.4s |

### Alineación forzada

| Modelo | Framework | 20s de audio | RTF |
|--------|-----------|-------------|-----|
| Qwen3-ForcedAligner-0.6B (4-bit) | MLX Swift (debug) | ~365ms | ~0.018 |

> Un solo paso forward no autorregresivo — sin bucle de muestreo. El codificador de audio domina (~328ms), el paso único del decodificador es ~37ms. **55x más rápido que tiempo real.**

### TTS

| Modelo | Framework | Corto (1s) | Medio (3s) | Largo (6s) | Primer paquete en streaming |
|--------|-----------|-----------|------------|-----------|---------------------------|
| Qwen3-TTS-0.6B (4-bit) | MLX Swift (release) | 1.6s (RTF 1.2) | 2.3s (RTF 0.7) | 3.9s (RTF 0.7) | ~120ms (1-frame) |
| Kokoro-82M | CoreML (Neural Engine) | ~1.4s (RTFx 0.7) | ~4.3s (RTFx 0.7) | ~8.6s (RTFx 0.7) | N/A (no autorregresivo) |
| Apple `AVSpeechSynthesizer` | AVFoundation | 0.08s | 0.08s | 0.17s (RTF 0.02) | N/A |

> Qwen3-TTS genera voz natural y expresiva con prosodia y emoción, ejecutándose **más rápido que tiempo real** (RTF < 1.0). La síntesis en streaming entrega el primer fragmento de audio en ~120ms. Kokoro-82M se ejecuta completamente en el Neural Engine con un pipeline de 3 etapas (RTFx ~0.7), ideal para iOS. El TTS integrado de Apple es más rápido pero produce voz robótica y monótona.

### PersonaPlex (voz a voz)

| Modelo | Framework | ms/paso | RTF | Notas |
|--------|-----------|---------|-----|-------|
| PersonaPlex-7B (8-bit) | MLX Swift (release) | ~112ms | ~1.4 | Recomendado — respuestas coherentes, 30% mas rapido que 4-bit |
| PersonaPlex-7B (4-bit) | MLX Swift (release) | ~158ms | ~1.97 | No recomendado — calidad de salida degradada |

> **Use 8-bit.** INT8 es mas rapido (112 ms/paso vs. 158 ms/paso) y produce respuestas full-duplex coherentes. La cuantizacion INT4 degrada la calidad de generacion, produciendo habla incoherente. INT8 se ejecuta a ~112ms/paso en M2 Max.

### VAD y embedding de hablante

| Modelo | Backend | Latencia por llamada | RTF | Notas |
|--------|---------|---------------------|-----|-------|
| Silero-VAD-v5 | MLX | ~2.1ms / fragmento | 0.065 | GPU (Metal) |
| Silero-VAD-v5 | CoreML | ~0.27ms / fragmento | 0.008 | Neural Engine, **7.7x más rápido** |
| WeSpeaker ResNet34-LM | MLX | ~310ms / 20s de audio | 0.016 | GPU (Metal) |
| WeSpeaker ResNet34-LM | CoreML | ~430ms / 20s de audio | 0.021 | Neural Engine, libera la GPU |

> Silero VAD CoreML se ejecuta en el Neural Engine a 7.7x la velocidad de MLX, lo que lo hace ideal para entrada de micrófono siempre activa. WeSpeaker MLX es más rápido en GPU, pero CoreML libera la GPU para cargas de trabajo simultáneas (TTS, ASR). Ambos backends producen resultados equivalentes.

### Mejora de voz

| Modelo | Backend | Duración | Latencia | RTF |
|--------|---------|----------|----------|-----|
| DeepFilterNet3 (FP16) | CoreML | 5s | 0.65s | 0.13 |
| DeepFilterNet3 (FP16) | CoreML | 10s | 1.2s | 0.12 |
| DeepFilterNet3 (FP16) | CoreML | 20s | 4.8s | 0.24 |

RTF = Factor de Tiempo Real (menor es mejor, < 1.0 = más rápido que tiempo real). El coste de GRU escala ~O(n^2).

### MLX vs CoreML

Ambos backends producen resultados equivalentes. Elige según tu carga de trabajo:

| | MLX | CoreML |
|---|---|---|
| **Hardware** | GPU (shaders Metal) | Neural Engine + CPU |
| **Ideal para** | Máximo rendimiento, cargas de un solo modelo | Pipelines multi-modelo, tareas en segundo plano |
| **Consumo** | Mayor uso de GPU | Menor consumo, libera la GPU |
| **Latencia** | Más rápido para modelos grandes (WeSpeaker) | Más rápido para modelos pequeños (Silero VAD) |

**Inferencia en escritorio**: MLX es el predeterminado — el rendimiento más rápido para un solo modelo en Apple Silicon. Cambia a CoreML cuando ejecutes múltiples modelos simultáneamente (ej., VAD + ASR + TTS) para evitar contención de GPU, o para cargas de trabajo sensibles a la batería en portátiles.

Los modelos CoreML están disponibles para el codificador Qwen3-ASR, Silero VAD y WeSpeaker. Para Qwen3-ASR, usa `--engine qwen3-coreml` (híbrido: codificador CoreML en ANE + decodificador de texto MLX en GPU). Para VAD/embeddings, pasa `engine: .coreml` al construir — la API de inferencia es idéntica.

## Benchmarks de precisión

### ASR — Tasa de error de palabras ([detalles](docs/benchmarks/asr-wer.md))

| Modelo | WER% (LibriSpeech test-clean) | RTF |
|--------|-------------------------------|-----|
| Qwen3-ASR 1.7B 8-bit | **2.35** | 0.090 |
| Qwen3-ASR 1.7B 4-bit | 2.57 | 0.045 |
| Parakeet TDT INT8 | 2.74 | 0.089 |
| Qwen3-ASR 0.6B 8-bit | 2.80 | 0.025 |

Qwen3-ASR 1.7B 8-bit supera a Whisper Large v3 Turbo (2.5%) con un tamaño comparable. Multilingüe: 10 idiomas evaluados en FLEURS.

### TTS — Inteligibilidad de ida y vuelta ([detalles](docs/benchmarks/tts-roundtrip.md))

| Motor | WER% | RTF |
|-------|------|-----|
| CosyVoice3 | **3.25** | 0.59 |
| Qwen3-TTS 1.7B | 3.47 | 0.79 |
| Kokoro-82M | 3.90 | 0.17 |

### VAD — Detección de voz ([detalles](docs/benchmarks/vad-detection.md))

| Motor | F1% (FLEURS) | RTF |
|-------|-------------|-----|
| FireRedVAD | **99.12** | 0.007 |
| Silero CoreML | 95.13 | 0.022 |
| Pyannote MLX | 94.86 | 0.358 |

## Arquitectura

**Modelos:** [ASR Model](docs/models/asr-model.md), [TTS Model](docs/models/tts-model.md), [CosyVoice TTS](docs/models/cosyvoice-tts.md), [Kokoro TTS](docs/models/kokoro-tts.md), [Parakeet TDT](docs/models/parakeet-asr.md), [PersonaPlex](docs/models/personaplex.md), [FireRedVAD](docs/models/fireredvad.md)

**Inferencia:** [Qwen3-ASR](docs/inference/qwen3-asr-inference.md), [Parakeet TDT](docs/inference/parakeet-asr-inference.md), [TTS Inference](docs/inference/qwen3-tts-inference.md), [Forced Aligner](docs/inference/forced-aligner.md), [FireRedVAD](docs/inference/fireredvad.md), [Silero VAD](docs/inference/silero-vad.md), [Speaker Diarization](docs/inference/speaker-diarization.md), [Speech Enhancement](docs/inference/speech-enhancement.md), [Reproducción de Audio](docs/audio/playback.md)

**Benchmarks:** [ASR WER](docs/benchmarks/asr-wer.md), [TTS Round-Trip](docs/benchmarks/tts-roundtrip.md), [VAD Detection](docs/benchmarks/vad-detection.md)

**Referencia:** [Shared Protocols](docs/shared-protocols.md)

## Configuración de caché

Los pesos de los modelos se almacenan en caché localmente. Puedes cambiar la ubicación de la caché con:

```bash
export QWEN3_CACHE_DIR=/path/to/cache
```

## Biblioteca Metal de MLX

Si ves `Failed to load the default metallib` en tiempo de ejecución, la biblioteca de shaders Metal falta. Ejecuta `make build` (o `./scripts/build_mlx_metallib.sh release` después de un `swift build` manual) para compilarla. Si falta el Metal Toolchain, instálalo primero:

```bash
xcodebuild -downloadComponent MetalToolchain
```

## Pruebas

Las pruebas unitarias (configuración, muestreo, preprocesamiento de texto, corrección de marcas temporales) se ejecutan sin necesidad de descargar modelos:

```bash
swift test --filter "Qwen3TTSConfigTests|SamplingTests|CosyVoiceTTSConfigTests|CamPlusPlusMelExtractorTests|PersonaPlexTests|ForcedAlignerTests/testText|ForcedAlignerTests/testTimestamp|ForcedAlignerTests/testLIS|SileroVADTests/testSilero|SileroVADTests/testReflection|SileroVADTests/testProcess|SileroVADTests/testReset|SileroVADTests/testDetect|SileroVADTests/testStreaming|SileroVADTests/testVADEvent|KokoroTTSTests"
```

Las pruebas de integración requieren pesos del modelo (se descargan automáticamente en la primera ejecución):

```bash
# Ida y vuelta TTS: sintetizar texto, guardar WAV, transcribir de vuelta con ASR
swift test --filter TTSASRRoundTripTests

# Solo ASR: transcribir audio de prueba
swift test --filter Qwen3ASRIntegrationTests

# Forced Aligner E2E: marcas temporales a nivel de palabra (~979 MB de descarga)
swift test --filter ForcedAlignerTests/testForcedAlignerE2E

# PersonaPlex E2E: pipeline de voz a voz (~5.5 GB de descarga)
PERSONAPLEX_E2E=1 swift test --filter PersonaPlexE2ETests
```

> **Nota:** La biblioteca Metal de MLX debe compilarse antes de ejecutar pruebas que usen operaciones MLX.
> Consulta [Biblioteca Metal de MLX](#biblioteca-metal-de-mlx) para instrucciones.

## Idiomas soportados

| Modelo | Idiomas |
|--------|---------|
| Qwen3-ASR | 52 idiomas (CN, EN, Cantonés, DE, FR, ES, JA, KO, RU, + 22 dialectos chinos, ...) |
| Parakeet TDT | 25 idiomas europeos (BG, CS, DA, DE, EL, EN, ES, ET, FI, FR, HR, HU, IT, LT, LV, MT, NL, PL, PT, RO, RU, SK, SL, SV, UK) |
| Qwen3-TTS | EN, CN, DE, JA, ES, FR, KO, RU, IT, PT (+ dialectos de Beijing/Sichuan vía CustomVoice) |
| CosyVoice TTS | CN, EN, JA, KO, DE, ES, FR, IT, RU |
| Kokoro TTS | EN (US/UK), ES, FR, HI, IT, JA, PT, CN, KO, DE |
| PersonaPlex | EN |

## Comparativa

### Voz a texto (ASR): speech-swift vs alternativas

| | **speech-swift (Qwen3-ASR)** | **whisper.cpp** | **Apple SFSpeechRecognizer** | **Google Cloud Speech** |
|---|---|---|---|---|
| **Ejecución** | En el dispositivo (MLX/CoreML) | En el dispositivo (CPU/GPU) | En el dispositivo o nube | Solo en la nube |
| **Idiomas** | 52 | 100+ | ~70 (en dispositivo: limitado) | 125+ |
| **RTF (10s audio, M2 Max)** | 0.06 (17x tiempo real) | 0.10 (Whisper-large-v3) | N/A | N/A |
| **Streaming** | No (por lotes) | No (por lotes) | Sí | Sí |
| **Modelos personalizados** | Sí (intercambiar pesos de HuggingFace) | Sí (modelos GGML) | No | No |
| **API Swift** | Nativa async/await | C++ con puente Swift | Nativa | REST/gRPC |
| **Privacidad** | Totalmente en el dispositivo | Totalmente en el dispositivo | Depende de la configuración | Datos enviados a la nube |
| **Marcas temporales de palabras** | Sí (Forced Aligner) | Sí | Limitado | Sí |
| **Coste** | Gratuito (Apache 2.0) | Gratuito (MIT) | Gratuito (en dispositivo) | Pago por minuto |

### Texto a voz (TTS): speech-swift vs alternativas

| | **speech-swift (Qwen3-TTS)** | **speech-swift (Kokoro)** | **Apple AVSpeechSynthesizer** | **ElevenLabs / Cloud TTS** |
|---|---|---|---|---|
| **Calidad** | Neural, expresiva | Neural, natural | Robótica, monótona | Neural, máxima calidad |
| **Ejecución** | En el dispositivo (MLX) | En el dispositivo (CoreML) | En el dispositivo | Solo en la nube |
| **Streaming** | Sí (~120ms primer fragmento) | No (pipeline de 3 etapas) | No | Sí |
| **Clonación de voz** | Sí | No | No | Sí |
| **Voces** | 9 integradas + clonar cualquiera | 54 voces preconfiguradas | ~50 voces del sistema | 1000+ |
| **Idiomas** | 10 | 10 | 60+ | 30+ |
| **Soporte iOS** | Solo macOS | iOS + macOS | iOS + macOS | Cualquiera (API) |
| **Coste** | Gratuito (Apache 2.0) | Gratuito (Apache 2.0) | Gratuito | Pago por carácter |

### Cuándo usar speech-swift

- **Aplicaciones con requisitos de privacidad** — médicas, legales, empresariales donde el audio no puede salir del dispositivo
- **Uso sin conexión** — no se necesita conexión a internet después de la descarga inicial del modelo
- **Sensible al coste** — sin cargos por minuto o por carácter de API
- **Optimización para Apple Silicon** — desarrollado específicamente para GPU M-series (Metal) y Neural Engine
- **Pipeline completo** — combina ASR + TTS + VAD + diarización + mejora de voz en un solo paquete Swift

## Preguntas frecuentes

**¿Funciona speech-swift en iOS?**
Kokoro TTS, Qwen3.5-Chat (CoreML), Silero VAD, Parakeet ASR, DeepFilterNet3 y WeSpeaker funcionan en iOS 17+ vía CoreML en el Neural Engine. Los modelos basados en MLX (Qwen3-ASR, Qwen3-TTS, Qwen3.5-Chat MLX, PersonaPlex) requieren macOS 14+ en Apple Silicon.

**¿Requiere conexión a internet?**
Solo para la descarga inicial del modelo desde HuggingFace (automática, almacenada en caché en `~/Library/Caches/qwen3-speech/`). Después de eso, toda la inferencia se ejecuta completamente sin conexión y sin acceso a la red.

**¿Cómo se compara speech-swift con Whisper?**
Qwen3-ASR-0.6B alcanza un RTF de 0.06 en M2 Max — un 40% más rápido que Whisper-large-v3 mediante whisper.cpp (RTF 0.10) — con precisión comparable en 52 idiomas. speech-swift proporciona una API nativa Swift con async/await, mientras que whisper.cpp requiere un puente C++.

**¿Puedo usarlo en una aplicación comercial?**
Sí. speech-swift tiene licencia Apache 2.0. Los pesos de los modelos subyacentes tienen sus propias licencias (consulta la página de HuggingFace de cada modelo).

**¿Qué chips Apple Silicon son compatibles?**
Todos los chips de la serie M: M1, M2, M3, M4 y sus variantes Pro/Max/Ultra. Requiere macOS 14+ (Sonoma) o iOS 17+.

**¿Cuánta memoria necesita?**
Desde ~3 MB (Silero VAD) hasta ~6.5 GB (PersonaPlex 7B). Kokoro TTS usa ~500 MB, Qwen3-ASR ~2.2 GB. Consulta la tabla de [Requisitos de memoria](#requisitos-de-memoria) para todos los detalles.

**¿Puedo ejecutar múltiples modelos simultáneamente?**
Sí. Usa modelos CoreML en el Neural Engine junto con modelos MLX en la GPU para evitar contención — por ejemplo, Silero VAD (CoreML) + Qwen3-ASR (MLX) + Qwen3-TTS (MLX).

**¿Hay una API REST?**
Sí. El binario `audio-server` expone todos los modelos mediante endpoints HTTP REST y WebSocket, incluyendo un WebSocket compatible con la OpenAI Realtime API en `/v1/realtime`.

## Contribuciones

¡Las contribuciones son bienvenidas! Ya sea una corrección de errores, integración de un nuevo modelo o mejora de la documentación — los PRs son apreciados.

**Para empezar:**
1. Haz un fork del repositorio y crea una rama de funcionalidad
2. `make build` para compilar (requiere Xcode + Metal Toolchain)
3. `make test` para ejecutar la suite de pruebas
4. Abre un PR contra `main`

## Licencia

Apache 2.0
