# Speech Swift

Модели ИИ для обработки речи на Apple Silicon, на базе MLX Swift и CoreML.

📖 Read in: [English](README.md) · [中文](README_zh.md) · [日本語](README_ja.md) · [한국어](README_ko.md) · [Español](README_es.md) · [Deutsch](README_de.md) · [Français](README_fr.md) · [हिन्दी](README_hi.md) · [Português](README_pt.md) · [Русский](README_ru.md)

Распознавание, синтез и понимание речи на устройстве для Mac и iOS. Работает полностью локально на Apple Silicon — без облака, без API-ключей, данные не покидают устройство.

[Установите через Homebrew](#homebrew) или добавьте как зависимость Swift Package.

**[Документация](https://soniqo.audio)** · **[Модели на HuggingFace](https://huggingface.co/aufklarer)** · **[Блог](https://blog.ivan.digital)**

- **Qwen3-ASR** — Распознавание речи (автоматическое распознавание речи, 52 языка)
- **Parakeet TDT** — Распознавание речи через CoreML (Neural Engine, NVIDIA FastConformer + TDT-декодер, 25 языков)
- **Qwen3-ForcedAligner** — Выравнивание временных меток на уровне слов (аудио + текст → временные метки)
- **Qwen3-TTS** — Синтез речи из текста (наивысшее качество, потоковый режим, пользовательские голоса, 10 языков)
- **CosyVoice TTS** — Синтез речи с потоковой генерацией, клонированием голоса, многоголосым диалогом и тегами эмоций (9 языков, DiT flow matching, CAM++ speaker encoder)
- **Kokoro TTS** — Синтез речи на устройстве (82M параметров, CoreML/Neural Engine, 54 голоса, готов для iOS, 10 языков)
- **Qwen3.5-Chat** — Локальный чат на базе LLM (0.8B, MLX + CoreML, INT4/INT8, гибридная архитектура DeltaNet, потоковая генерация токенов)
- **PersonaPlex** — Полнодуплексная генерация речи из речи (7B, аудио на входе → аудио на выходе, 18 голосовых пресетов)
- **DeepFilterNet3** — Улучшение речи / подавление шума (2.1M параметров, реальное время 48kHz)
- **FireRedVAD** — Офлайн-обнаружение голосовой активности (DFSMN, CoreML, 100+ языков, 97.6% F1)
- **Silero VAD** — Потоковое обнаружение голосовой активности (фрагменты по 32мс, субмиллисекундная задержка)
- **Pyannote VAD** — Офлайн-обнаружение голосовой активности (окна по 10с, перекрытие нескольких спикеров)
- **Speaker Diarization** — Кто говорил и когда (сегментация Pyannote + цепочки спикеров на основе активности, или сквозная модель Sortformer на Neural Engine)
- **Speaker Embeddings** — Верификация и идентификация спикеров (WeSpeaker ResNet34, 256-мерные векторы)

Статьи: [Qwen3-ASR](https://arxiv.org/abs/2601.21337) (Alibaba), [Qwen3-TTS](https://arxiv.org/abs/2601.15621) (Alibaba), [Qwen3](https://arxiv.org/abs/2505.09388) (Alibaba), [Parakeet TDT](https://arxiv.org/abs/2304.06795) (NVIDIA), [CosyVoice 3](https://arxiv.org/abs/2505.17589) (Alibaba), [Kokoro](https://arxiv.org/abs/2301.01695) (StyleTTS 2), [PersonaPlex](https://arxiv.org/abs/2602.06053) (NVIDIA), [Mimi](https://arxiv.org/abs/2410.00037) (Kyutai), [Sortformer](https://arxiv.org/abs/2409.06656) (NVIDIA)

## Дорожная карта

Смотрите [обсуждение дорожной карты](https://github.com/soniqo/speech-swift/discussions/81) — комментарии и предложения приветствуются!

## Новости

- **20 марта 2026** — [We Beat Whisper Large v3 with a 600M Model Running Entirely on Your Mac](https://blog.ivan.digital/we-beat-whisper-large-v3-with-a-600m-model-running-entirely-on-your-mac-20e6ce191174)
- **26 февраля 2026** — [Speaker Diarization and Voice Activity Detection on Apple Silicon — Native Swift with MLX](https://blog.ivan.digital/speaker-diarization-and-voice-activity-detection-on-apple-silicon-native-swift-with-mlx-92ea0c9aca0f)
- **23 февраля 2026** — [NVIDIA PersonaPlex 7B on Apple Silicon — Full-Duplex Speech-to-Speech in Native Swift with MLX](https://blog.ivan.digital/nvidia-personaplex-7b-on-apple-silicon-full-duplex-speech-to-speech-in-native-swift-with-mlx-0aa5276f2e23)
- **12 февраля 2026** — [Qwen3-ASR Swift: On-Device ASR + TTS for Apple Silicon — Architecture and Benchmarks](https://blog.ivan.digital/qwen3-asr-swift-on-device-asr-tts-for-apple-silicon-architecture-and-benchmarks-27cbf1e4463f)

## Модели

| Модель | Задача | Потоковый режим | Языки | Размеры |
|--------|--------|-----------------|-------|---------|
| Qwen3-ASR-0.6B | Речь → Текст | Нет | 52 языка | [4-bit](https://huggingface.co/aufklarer/Qwen3-ASR-0.6B-MLX-4bit) 680 MB · [8-bit](https://huggingface.co/aufklarer/Qwen3-ASR-0.6B-MLX-8bit) 1.0 GB · [CoreML](https://huggingface.co/aufklarer/Qwen3-ASR-CoreML) 180 MB |
| Qwen3-ASR-1.7B | Речь → Текст | Нет | 52 языка | [4-bit](https://huggingface.co/aufklarer/Qwen3-ASR-1.7B-MLX-4bit) 2.1 GB · [8-bit](https://huggingface.co/aufklarer/Qwen3-ASR-1.7B-MLX-8bit) 3.2 GB |
| Parakeet-TDT-0.6B | Речь → Текст | Нет | 25 европейских языков | [CoreML INT4](https://huggingface.co/aufklarer/Parakeet-TDT-v3-CoreML-INT4) 315 MB · [CoreML INT8](https://huggingface.co/aufklarer/Parakeet-TDT-v3-CoreML-INT8) 500 MB |
| Qwen3-ForcedAligner-0.6B | Аудио + Текст → Временные метки | Нет | Мульти | [4-bit](https://huggingface.co/aufklarer/Qwen3-ForcedAligner-0.6B-4bit) 979 MB · [8-bit](https://huggingface.co/aufklarer/Qwen3-ForcedAligner-0.6B-8bit) 1.4 GB · [CoreML INT4](https://huggingface.co/aufklarer/Qwen3-ForcedAligner-0.6B-CoreML-INT4) 630 MB · [CoreML INT8](https://huggingface.co/aufklarer/Qwen3-ForcedAligner-0.6B-CoreML-INT8) 1.0 GB |
| Qwen3-TTS-0.6B Base | Текст → Речь | Да (~120мс) | 10 языков | [4-bit](https://huggingface.co/aufklarer/Qwen3-TTS-12Hz-0.6B-Base-MLX-4bit) 1.7 GB · [8-bit](https://huggingface.co/aufklarer/Qwen3-TTS-12Hz-0.6B-Base-MLX-8bit) 2.4 GB |
| Qwen3-TTS-0.6B CustomVoice | Текст → Речь | Да (~120мс) | 10 языков | [4-bit](https://huggingface.co/aufklarer/Qwen3-TTS-12Hz-0.6B-CustomVoice-MLX-4bit) 1.7 GB |
| Qwen3-TTS-1.7B Base | Текст → Речь | Да (~120мс) | 10 языков | [4-bit](https://huggingface.co/aufklarer/Qwen3-TTS-12Hz-1.7B-Base-MLX-4bit) 3.2 GB · [8-bit](https://huggingface.co/aufklarer/Qwen3-TTS-12Hz-1.7B-Base-MLX-8bit) 4.8 GB |
| CosyVoice3-0.5B | Текст → Речь | Да (~150мс) | 9 языков | [4-bit](https://huggingface.co/aufklarer/CosyVoice3-0.5B-MLX-4bit) 1.2 GB |
| Kokoro-82M | Текст → Речь | Нет | 10 языков | [CoreML](https://huggingface.co/aufklarer/Kokoro-82M-CoreML) ~170 MB |
| Qwen3.5-0.8B Chat | Text → Text (LLM) | Yes (streaming) | Multi | [MLX INT4](https://huggingface.co/aufklarer/Qwen3.5-0.8B-Chat-MLX) 418 MB · [CoreML INT8](https://huggingface.co/aufklarer/Qwen3.5-0.8B-Chat-CoreML) 981 MB |
| PersonaPlex-7B | Речь → Речь | Да (~2с фрагменты) | EN | [4-bit](https://huggingface.co/aufklarer/PersonaPlex-7B-MLX-4bit) 4.9 GB · [8-bit](https://huggingface.co/aufklarer/PersonaPlex-7B-MLX-8bit) 9.1 GB |
| FireRedVAD | Обнаружение голосовой активности | Нет (офлайн) | 100+ языков | [CoreML](https://huggingface.co/aufklarer/FireRedVAD-CoreML) ~1.2 MB |
| Silero-VAD-v5 | Обнаружение голосовой активности | Да (фрагменты по 32мс) | Языконезависимый | [MLX](https://huggingface.co/aufklarer/Silero-VAD-v5-MLX) · [CoreML](https://huggingface.co/aufklarer/Silero-VAD-v5-CoreML) ~1.2 MB |
| Pyannote-Segmentation-3.0 | VAD + сегментация спикеров | Нет (окна по 10с) | Языконезависимый | [MLX](https://huggingface.co/aufklarer/Pyannote-Segmentation-MLX) ~5.7 MB |
| DeepFilterNet3 | Улучшение речи | Да (кадры по 10мс) | Языконезависимый | [CoreML FP16](https://huggingface.co/aufklarer/DeepFilterNet3-CoreML) ~4.2 MB |
| WeSpeaker-ResNet34-LM | Голосовые эмбеддинги (256-мерные) | Нет | Языконезависимый | [MLX](https://huggingface.co/aufklarer/WeSpeaker-ResNet34-LM-MLX) · [CoreML](https://huggingface.co/aufklarer/WeSpeaker-ResNet34-LM-CoreML) ~25 MB |
| CAM++ | Голосовые эмбеддинги (192-мерные) | Нет | Языконезависимый | [CoreML](https://huggingface.co/aufklarer/CamPlusPlus-Speaker-CoreML) ~14 MB |
| Sortformer | Диаризация спикеров (сквозная) | Да (фрагментами) | Языконезависимый | [CoreML](https://huggingface.co/aufklarer/Sortformer-Diarization-CoreML) ~240 MB |

### Требования к памяти

Память для весов — это память GPU (MLX) или ANE (CoreML), занятая параметрами модели. Пиковое потребление при инференсе включает KV-кеши, активации и промежуточные тензоры.

| Модель | Память для весов | Пиковое потребление |
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

### Какой TTS выбрать

- **Qwen3-TTS**: Лучшее качество, потоковый режим (~120мс), 9 встроенных голосов, 10 языков, пакетный синтез
- **CosyVoice TTS**: Потоковый режим (~150мс), 9 языков, клонирование голоса (CAM++ speaker encoder), многоголосый диалог (`[S1] ... [S2] ...`), встроенные теги эмоций/стиля (`(happy)`, `(whispers)`), DiT flow matching + HiFi-GAN вокодер
- **Kokoro TTS**: Лёгкий TTS для iOS (82M параметров), CoreML/Neural Engine, 54 голоса, 10 языков, неавторегрессионный (3-этапный пайплайн)
- **PersonaPlex**: Полнодуплексная генерация речи из речи (аудио на входе → аудио на выходе), потоковый режим (~2с фрагменты), 18 голосовых пресетов, на базе архитектуры Moshi

## Установка

### Homebrew

Требуется нативный ARM Homebrew (`/opt/homebrew`). Homebrew через Rosetta/x86_64 не поддерживается.

```bash
brew tap soniqo/speech https://github.com/soniqo/speech-swift
brew install speech
```

Затем используйте:

```bash
audio transcribe recording.wav
audio speak "Hello world"
audio speak "Hallo Welt" --engine cosyvoice --language german
audio respond --input question.wav --transcript
```

> Для интерактивного голосового диалога с микрофоном смотрите **[PersonaPlexDemo](Examples/PersonaPlexDemo/)**.

### Swift Package Manager

Добавьте в ваш `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/soniqo/speech-swift", branch: "main")
]
```

Импортируйте нужный модуль:

```swift
import Qwen3ASR      // Распознавание речи (MLX)
import ParakeetASR   // Распознавание речи (CoreML)
import Qwen3TTS      // Синтез речи (Qwen3)
import CosyVoiceTTS  // Синтез речи (потоковый)
import KokoroTTS     // Синтез речи (CoreML, для iOS)
import Qwen3Chat     // Локальный чат на базе LLM (CoreML)
import PersonaPlex   // Генерация речи из речи (полный дуплекс)
import SpeechVAD          // Обнаружение голосовой активности (pyannote + Silero)
import SpeechEnhancement  // Подавление шума (DeepFilterNet3)
import AudioCommon        // Общие утилиты
```

### Системные требования

- Swift 5.9+
- macOS 14+ или iOS 17+
- Apple Silicon (M1/M2/M3/M4)
- Xcode 15+ (с Metal Toolchain — выполните `xcodebuild -downloadComponent MetalToolchain`, если отсутствует)

### Сборка из исходного кода

```bash
git clone https://github.com/soniqo/speech-swift
cd speech-swift
make build
```

Эта команда компилирует Swift-пакет **и** библиотеку Metal-шейдеров MLX за один шаг. Библиотека Metal (`mlx.metallib`) необходима для GPU-инференса — без неё при запуске возникнет ошибка `Failed to load the default metallib`.

Отладочная сборка: `make debug`. Запуск юнит-тестов: `make test`.

## Попробуйте голосового ассистента

**[PersonaPlexDemo](Examples/PersonaPlexDemo/)** — это готовое к запуску macOS-приложение голосового ассистента: нажмите, чтобы говорить, и получайте голосовые ответы в реальном времени. Использует микрофонный ввод с Silero VAD для автоматического обнаружения речи, Qwen3-ASR для транскрипции и PersonaPlex 7B для генерации речи из речи. Многоходовый диалог с 18 голосовыми пресетами и отображением внутреннего монолога.

```bash
make build  # из корня репозитория — собирает всё, включая MLX metallib
cd Examples/PersonaPlexDemo
# См. Examples/PersonaPlexDemo/README.md для инструкций по сборке .app
```

> RTF ~0.94 на M2 Max (быстрее реального времени). Модели загружаются автоматически при первом запуске (~5.5 GB PersonaPlex + ~400 MB ASR).

## Демо-приложения

- **[PersonaPlexDemo](Examples/PersonaPlexDemo/)** — Голосовой ассистент для диалога (микрофонный ввод, VAD, многоходовый диалог). См. выше.
- **[SpeechDemo](Examples/SpeechDemo/)** — Диктовка (Parakeet TDT / Qwen3-ASR с автоопределением языка) и синтез речи (Qwen3-TTS) в интерфейсе с вкладками.

Соберите и запустите как macOS-приложение `.app` — инструкции в README каждого демо.

## Распознавание речи (ASR) — транскрибирование аудио на Swift

### Базовая транскрипция

```swift
import Qwen3ASR

// По умолчанию: модель 0.6B
let model = try await Qwen3ASRModel.fromPretrained()

// Или используйте более крупную модель 1.7B для лучшей точности
let model = try await Qwen3ASRModel.fromPretrained(
    modelId: "aufklarer/Qwen3-ASR-1.7B-MLX-8bit"
)

// Аудио может быть любой частоты дискретизации — автоматический ресемплинг до 16kHz
let transcription = model.transcribe(audio: audioSamples, sampleRate: 16000)
print(transcription)
```

### CoreML-энкодер (Neural Engine)

Гибридный режим: CoreML-энкодер на Neural Engine + MLX текстовый декодер на GPU. Меньшее энергопотребление, освобождает GPU на этапе работы энкодера.

```swift
import Qwen3ASR

let encoder = try await CoreMLASREncoder.fromPretrained()
let model = try await Qwen3ASRModel.fromPretrained()
let text = try model.transcribe(audio: audioSamples, sampleRate: 16000, coremlEncoder: encoder)
```

Доступны варианты INT8 (180 MB, по умолчанию) и INT4 (90 MB). Рекомендуется INT8 (косинусное сходство > 0.999 относительно FP32).

### Parakeet TDT (CoreML)

```swift
import ParakeetASR

let model = try await ParakeetASRModel.fromPretrained()
let transcription = model.transcribe(audio: audioSamples, sampleRate: 16000)
```

Работает на Neural Engine через CoreML — освобождает GPU для параллельных задач. 25 европейских языков, ~315 MB.

### CLI для ASR

```bash
make build  # или: swift build -c release && ./scripts/build_mlx_metallib.sh release

# По умолчанию (Qwen3-ASR 0.6B, MLX)
.build/release/audio transcribe audio.wav

# Модель 1.7B
.build/release/audio transcribe audio.wav --model 1.7B

# CoreML-энкодер (Neural Engine + MLX-декодер)
.build/release/audio transcribe --engine qwen3-coreml audio.wav

# Parakeet TDT (CoreML, Neural Engine)
.build/release/audio transcribe --engine parakeet audio.wav
```

## Принудительное выравнивание

### Временные метки на уровне слов

```swift
import Qwen3ASR

let aligner = try await Qwen3ForcedAligner.fromPretrained()
// Загружает ~979 MB при первом запуске

let aligned = aligner.align(
    audio: audioSamples,
    text: "Can you guarantee that the replacement part will be shipped tomorrow?",
    sampleRate: 24000
)

for word in aligned {
    print("[\(String(format: "%.2f", word.startTime))s - \(String(format: "%.2f", word.endTime))s] \(word.text)")
}
```

### CLI для принудительного выравнивания

```bash
swift build -c release

# Выравнивание с указанным текстом
.build/release/audio align audio.wav --text "Hello world"

# Сначала транскрипция, затем выравнивание
.build/release/audio align audio.wav
```

Вывод:
```
[0.12s - 0.45s] Can
[0.45s - 0.72s] you
[0.72s - 1.20s] guarantee
...
```

Неавторегрессионный — 3-этапный пайплайн, без цикла сэмплирования. Подробнее об архитектуре см. [Forced Aligner](docs/inference/forced-aligner.md).

## Синтез речи (TTS) — генерация речи на Swift

### Базовый синтез

```swift
import Qwen3TTS
import AudioCommon  // для WAVWriter

let model = try await Qwen3TTSModel.fromPretrained()
// Загружает ~1.7 GB при первом запуске (модель + веса кодека)
let audio = model.synthesize(text: "Hello world", language: "english")
// Выход: моно-сэмплы float 24kHz
try WAVWriter.write(samples: audio, sampleRate: 24000, to: outputURL)
```

### CLI для TTS

```bash
make build
.build/release/audio speak "Hello world" --output output.wav --language english
```

### Пользовательский голос / Выбор диктора

Вариант модели **CustomVoice** поддерживает 9 встроенных голосов и инструкции на естественном языке для управления тоном и стилем. Загрузите его, указав ID модели CustomVoice:

```swift
import Qwen3TTS

// Загрузка модели CustomVoice (загружает ~1.7 GB при первом запуске)
let model = try await Qwen3TTSModel.fromPretrained(
    modelId: TTSModelVariant.customVoice.rawValue
)

// Синтез с конкретным диктором
let audio = model.synthesize(text: "Hello world", language: "english", speaker: "vivian")

// Список доступных дикторов
print(model.availableSpeakers)  // ["aiden", "dylan", "eric", ...]
```

CLI:

```bash
# Использование модели CustomVoice с диктором
.build/release/audio speak "Hello world" --model customVoice --speaker vivian --output vivian.wav

# Список доступных дикторов
.build/release/audio speak --model customVoice --list-speakers
```

### Клонирование голоса (базовая модель)

Клонирование голоса диктора из эталонного аудиофайла. Два режима:

**Режим ICL** (рекомендуется) — кодирует эталонное аудио в токены кодека с транскриптом. Более высокое качество, надёжный EOS:

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

**Режим X-vector** — только голосовой эмбеддинг, транскрипт не нужен, но качество ниже:

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

### Управление тоном и стилем (только CustomVoice)

Модель CustomVoice принимает параметр `instruct` на естественном языке для управления стилем речи, тоном, эмоциями и темпом. Инструкция добавляется к входным данным модели в формате ChatML.

```swift
// Радостный тон
let audio = model.synthesize(
    text: "Welcome to our store!",
    language: "english",
    speaker: "ryan",
    instruct: "Speak in a cheerful, upbeat tone"
)

// Медленно и серьёзно
let audio = model.synthesize(
    text: "We regret to inform you...",
    language: "english",
    speaker: "aiden",
    instruct: "Read this slowly and solemnly"
)

// Шёпотом
let audio = model.synthesize(
    text: "Can you keep a secret?",
    language: "english",
    speaker: "vivian",
    instruct: "Whisper this softly"
)
```

CLI:

```bash
# С инструкцией стиля
.build/release/audio speak "Good morning!" --model customVoice --speaker ryan \
    --instruct "Speak in a cheerful, upbeat tone" --output cheerful.wav

# Инструкция по умолчанию ("Speak naturally.") применяется автоматически при использовании CustomVoice
.build/release/audio speak "Hello world" --model customVoice --speaker ryan --output natural.wav
```

Если `--instruct` не указан для модели CustomVoice, автоматически применяется `"Speak naturally."`, чтобы предотвратить бессвязную генерацию. Базовая модель не поддерживает instruct.

### Пакетный синтез

Синтез нескольких текстов за один пакетный прямой проход для более высокой пропускной способности:

```swift
let texts = ["Good morning everyone.", "The weather is nice today.", "Please open the window."]
let audioList = model.synthesizeBatch(texts: texts, language: "english", maxBatchSize: 4)
// audioList[i] — моно-сэмплы float 24kHz для texts[i]
for (i, audio) in audioList.enumerated() {
    try WAVWriter.write(samples: audio, sampleRate: 24000, to: URL(fileURLWithPath: "output_\(i).wav"))
}
```

#### CLI для пакетного синтеза

```bash
# Создайте файл с одним текстом на строку
echo "Hello world.\nGoodbye world." > texts.txt
.build/release/audio speak --batch-file texts.txt --output output.wav --batch-size 4
# Создаёт output_0.wav, output_1.wav, ...
```

> Пакетный режим амортизирует загрузку весов модели между элементами. Ожидайте прирост пропускной способности в ~1.5-2.5 раза при B=4 на Apple Silicon. Лучшие результаты достигаются, когда тексты генерируют аудио примерно одинаковой длины.

### Параметры сэмплирования

```swift
let config = SamplingConfig(temperature: 0.9, topK: 50, repetitionPenalty: 1.05)
let audio = model.synthesize(text: "Hello", language: "english", sampling: config)
```

### Потоковый синтез

Генерация аудиофрагментов по мере готовности для минимальной задержки до первого пакета:

```swift
let stream = model.synthesizeStream(
    text: "Hello, this is streaming synthesis.",
    language: "english",
    streaming: .lowLatency  // ~120мс до первого аудиофрагмента
)

for try await chunk in stream {
    // chunk.samples: [Float] PCM @ 24kHz
    // chunk.isFinal: true для последнего фрагмента
    playAudio(chunk.samples)
}
```

CLI:

```bash
# Потоковый режим по умолчанию (первый фрагмент из 3 кадров, задержка ~225мс)
.build/release/audio speak "Hello world" --stream

# Минимальная задержка (первый фрагмент из 1 кадра, задержка ~120мс)
.build/release/audio speak "Hello world" --stream --first-chunk-frames 1
```

## Генерация речи из речи — полнодуплексный голосовой диалог

> Для интерактивного голосового ассистента с микрофонным вводом см. **[PersonaPlexDemo](Examples/PersonaPlexDemo/)** — нажмите, чтобы говорить, многоходовый диалог с автоматическим обнаружением речи.

### Генерация речи из речи

```swift
import PersonaPlex
import AudioCommon  // для WAVWriter, AudioFileLoader

let model = try await PersonaPlexModel.fromPretrained()
// Загружает ~5.5 GB при первом запуске (temporal 4-bit + depformer + Mimi кодек + голосовые пресеты)

let audio = try AudioFileLoader.load(url: inputURL, targetSampleRate: 24000)
let (response, textTokens) = model.respond(userAudio: audio, voice: .NATM0)
// response: моно-сэмплы float 24kHz
// textTokens: внутренний монолог модели (ID токенов SentencePiece)
try WAVWriter.write(samples: response.audio, sampleRate: 24000, to: outputURL)
```

### Внутренний монолог (текстовый вывод)

PersonaPlex генерирует текстовые токены параллельно с аудио — это внутренние рассуждения модели. Декодируйте их встроенным декодером SentencePiece:

```swift
let decoder = try SentencePieceDecoder(modelPath: "tokenizer_spm_32k_3.model")
let transcript = decoder.decode(textTokens)
print(transcript)  // например, "Sure, I can help you with that..."
```

### Потоковая генерация речи из речи

```swift
// Получайте аудиофрагменты по мере генерации (~2с на фрагмент)
let stream = model.respondStream(userAudio: audio, voice: .NATM0)
for try await chunk in stream {
    playAudio(chunk.samples)  // воспроизводите сразу, 24kHz моно
    // chunk.textTokens содержит текст этого фрагмента; последний фрагмент содержит все токены
    if chunk.isFinal { break }
}
```

### Выбор голоса

Доступно 18 голосовых пресетов:
- **Естественные женские**: NATF0, NATF1, NATF2, NATF3
- **Естественные мужские**: NATM0, NATM1, NATM2, NATM3
- **Разнообразные женские**: VARF0, VARF1, VARF2, VARF3, VARF4
- **Разнообразные мужские**: VARM0, VARM1, VARM2, VARM3, VARM4

### Системные промпты

Системный промпт определяет поведение модели в диалоге. Можно передать любой пользовательский промпт в виде обычной строки:

```swift
// Пользовательский системный промпт (токенизируется автоматически)
let response = model.respond(
    userAudio: audio,
    voice: .NATM0,
    systemPrompt: "You enjoy having a good conversation."
)

// Или использование пресета
let response = model.respond(
    userAudio: audio,
    voice: .NATM0,
    systemPromptTokens: SystemPromptPreset.customerService.tokens
)
```

Доступные пресеты: `focused` (по умолчанию), `assistant`, `customerService`, `teacher`.

### CLI для PersonaPlex

```bash
make build

# Базовая генерация речи из речи
.build/release/audio respond --input question.wav --output response.wav

# С транскрипцией (декодирует текст внутреннего монолога)
.build/release/audio respond --input question.wav --transcript

# JSON-вывод (путь к аудио, транскрипт, метрики задержки)
.build/release/audio respond --input question.wav --json

# Пользовательский текст системного промпта
.build/release/audio respond --input question.wav --system-prompt-text "You enjoy having a good conversation."

# Выбор голоса и пресета системного промпта
.build/release/audio respond --input question.wav --voice NATF1 --system-prompt focused

# Настройка параметров сэмплирования
.build/release/audio respond --input question.wav --audio-temp 0.6 --repetition-penalty 1.5

# Ранняя остановка по энтропии текста (останавливается при схлопывании текста)
.build/release/audio respond --input question.wav --entropy-threshold 1.0 --entropy-window 5

# Список доступных голосов и промптов
.build/release/audio respond --list-voices
.build/release/audio respond --list-prompts
```

## CosyVoice TTS — потоковый синтез речи с клонированием голоса

### Базовый синтез

```swift
import CosyVoiceTTS
import AudioCommon  // для WAVWriter

let model = try await CosyVoiceTTSModel.fromPretrained()
// Загружает ~1.9 GB при первом запуске (LLM + DiT + веса HiFi-GAN)

let audio = model.synthesize(text: "Hello, how are you today?", language: "english")
// Выход: моно-сэмплы float 24kHz
try WAVWriter.write(samples: audio, sampleRate: 24000, to: outputURL)
```

### Потоковый синтез

```swift
// Потоковый режим: получайте аудиофрагменты по мере генерации (~150мс до первого фрагмента)
for try await chunk in model.synthesizeStream(text: "Hello, how are you today?", language: "english") {
    // chunk.audio: [Float], chunk.sampleRate: Int
    playAudio(chunk.audio)  // воспроизводите сразу
}
```

### Клонирование голоса (CosyVoice)

Клонирование голоса диктора с помощью CAM++ speaker encoder (192-мерный, CoreML Neural Engine):

```swift
import CosyVoiceTTS
import AudioCommon

let model = try await CosyVoiceTTSModel.fromPretrained()
let speaker = try await CamPlusPlusSpeaker.fromPretrained()
// Загружает ~14 MB модели CAM++ CoreML при первом использовании

let refAudio = try AudioFileLoader.load(url: referenceURL, targetSampleRate: 16000)
let embedding = try speaker.embed(audio: refAudio, sampleRate: 16000)
// embedding: [Float] длиной 192

let audio = model.synthesize(
    text: "Hello in a cloned voice!",
    language: "english",
    speakerEmbedding: embedding
)
```

### CLI для CosyVoice TTS

```bash
make build

# Базовый синтез
.build/release/audio speak "Hello world" --engine cosyvoice --language english --output output.wav

# Клонирование голоса (загружает CAM++ speaker encoder при первом использовании)
.build/release/audio speak "Hello world" --engine cosyvoice --voice-sample reference.wav --output cloned.wav

# Многоголосый диалог с клонированием голоса
.build/release/audio speak "[S1] Hello there! [S2] Hey, how are you?" \
    --engine cosyvoice --speakers s1=alice.wav,s2=bob.wav -o dialogue.wav

# Встроенные теги эмоций/стиля
.build/release/audio speak "(excited) Wow, amazing! (sad) But I have to go..." \
    --engine cosyvoice -o emotion.wav

# Комбинация: диалог + эмоции + клонирование голоса
.build/release/audio speak "[S1] (happy) Great news! [S2] (surprised) Really?" \
    --engine cosyvoice --speakers s1=alice.wav,s2=bob.wav -o combined.wav

# Пользовательская инструкция стиля
.build/release/audio speak "Hello world" --engine cosyvoice --cosy-instruct "Speak cheerfully" -o cheerful.wav

# Потоковый синтез
.build/release/audio speak "Hello world" --engine cosyvoice --language english --stream --output output.wav
```

## Kokoro TTS — лёгкий синтез речи на устройстве (iOS + macOS)

### Базовый синтез

```swift
import KokoroTTS
import AudioCommon  // для WAVWriter

let tts = try await KokoroTTSModel.fromPretrained()
// Загружает ~170 MB при первом запуске (CoreML-модели + голосовые эмбеддинги + словари)

let audio = try tts.synthesize(text: "Hello world", voice: "af_heart")
// Выход: моно-сэмплы float 24kHz
try WAVWriter.write(samples: audio, sampleRate: 24000, to: outputURL)
```

54 предустановленных голоса на 10 языках. Неавторегрессионный — 3-этапный пайплайн CoreML, без цикла сэмплирования. Работает на Neural Engine, полностью освобождая GPU.

### CLI для Kokoro TTS

```bash
make build

# Базовый синтез
.build/release/audio kokoro "Hello world" --voice af_heart --output hello.wav

# Выбор языка
.build/release/audio kokoro "Bonjour le monde" --voice ff_siwis --language fr --output bonjour.wav

# Список доступных голосов
.build/release/audio kokoro --list-voices
```

## Qwen3 Chat (локальный LLM)

```swift
import Qwen3Chat

let chat = try await Qwen3ChatModel.fromPretrained()
// Загружает ~318 MB при первом запуске (INT4 CoreML-модель + токенизатор)

// Одиночный ответ
let response = try chat.generate("What is Swift?", systemPrompt: "Answer briefly.")
print(response)

// Потоковая генерация токенов
let stream = chat.chatStream("Tell me a joke", systemPrompt: "Be funny.")
for try await token in stream {
    print(token, terminator: "")
}
```

Qwen3-0.6B с INT4-квантованием для CoreML. Работает на Neural Engine со скоростью ~2 ток/с на iPhone, ~15 ток/с на M-серии. Поддерживает многоходовый диалог с KV-кешем, режим размышления (токены `<think>`), настраиваемое сэмплирование (temperature, top-k, top-p, repetition penalty).

## Обнаружение голосовой активности (VAD) — детекция речи в аудио

### Потоковый VAD (Silero)

Silero VAD v5 обрабатывает аудиофрагменты по 32мс с субмиллисекундной задержкой — идеально для обнаружения речи в реальном времени с микрофона или потока.

```swift
import SpeechVAD

let vad = try await SileroVADModel.fromPretrained()
// Или используйте CoreML (Neural Engine, меньше энергопотребление):
// let vad = try await SileroVADModel.fromPretrained(engine: .coreml)

// Потоковый режим: обработка фрагментов по 512 сэмплов (32мс @ 16kHz)
let prob = vad.processChunk(samples)  // → 0.0...1.0
vad.resetState()  // вызывайте между разными аудиопотоками

// Или определите все сегменты сразу
let segments = vad.detectSpeech(audio: audioSamples, sampleRate: 16000)
for seg in segments {
    print("Речь: \(seg.startTime)s - \(seg.endTime)s")
}
```

### Событийный потоковый режим

```swift
let processor = StreamingVADProcessor(model: vad)

// Подавайте аудио любой длины — события генерируются при подтверждении речи
let events = processor.process(samples: audioBuffer)
for event in events {
    switch event {
    case .speechStarted(let time):
        print("Речь началась в \(time)с")
    case .speechEnded(let segment):
        print("Речь: \(segment.startTime)s - \(segment.endTime)s")
    }
}

// Завершите обработку в конце потока
let final = processor.flush()
```

### CLI для VAD

```bash
make build

# Потоковый Silero VAD (фрагменты по 32мс)
.build/release/audio vad-stream audio.wav

# CoreML-бэкенд (Neural Engine)
.build/release/audio vad-stream audio.wav --engine coreml

# С пользовательскими порогами
.build/release/audio vad-stream audio.wav --onset 0.6 --offset 0.4

# JSON-вывод
.build/release/audio vad-stream audio.wav --json

# Пакетный pyannote VAD (скользящие окна по 10с)
.build/release/audio vad audio.wav
```

## Диаризация спикеров — кто говорил и когда

### Конвейер диаризации

```swift
import SpeechVAD

let pipeline = try await DiarizationPipeline.fromPretrained()
// Или используйте CoreML-эмбеддинги (Neural Engine, освобождает GPU):
// let pipeline = try await DiarizationPipeline.fromPretrained(embeddingEngine: .coreml)

let result = pipeline.diarize(audio: samples, sampleRate: 16000)
for seg in result.segments {
    print("Спикер \(seg.speakerId): [\(seg.startTime)s - \(seg.endTime)s]")
}
print("Обнаружено \(result.numSpeakers) спикеров")
```

### Голосовые эмбеддинги

```swift
let model = try await WeSpeakerModel.fromPretrained()
// Или: let model = try await WeSpeakerModel.fromPretrained(engine: .coreml)
let embedding = model.embed(audio: samples, sampleRate: 16000)
// embedding: [Float] длиной 256, L2-нормализованный

// Сравнение спикеров
let similarity = WeSpeakerModel.cosineSimilarity(embeddingA, embeddingB)
```

### Извлечение спикера

Извлечение сегментов конкретного спикера по эталонной записи:

```swift
let pipeline = try await DiarizationPipeline.fromPretrained()
let targetEmb = pipeline.embeddingModel.embed(audio: enrollmentAudio, sampleRate: 16000)
let segments = pipeline.extractSpeaker(
    audio: meetingAudio, sampleRate: 16000,
    targetEmbedding: targetEmb
)
```

### Диаризация Sortformer (сквозная, CoreML)

NVIDIA Sortformer предсказывает покадровую активность до 4 спикеров напрямую — без эмбеддингов и кластеризации. Работает на Neural Engine.

```swift
let diarizer = try await SortformerDiarizer.fromPretrained()
let result = diarizer.diarize(audio: samples, sampleRate: 16000, config: .default)
for seg in result.segments {
    print("Спикер \(seg.speakerId): [\(seg.startTime)s - \(seg.endTime)s]")
}
```

### CLI для диаризации

```bash
make build

# Диаризация Pyannote (по умолчанию)
.build/release/audio diarize meeting.wav

# Диаризация Sortformer (CoreML, Neural Engine)
.build/release/audio diarize meeting.wav --engine sortformer

# CoreML-эмбеддинги (Neural Engine, только pyannote)
.build/release/audio diarize meeting.wav --embedding-engine coreml

# JSON-вывод
.build/release/audio diarize meeting.wav --json

# Извлечение конкретного спикера (только pyannote)
.build/release/audio diarize meeting.wav --target-speaker enrollment.wav

# Голосовой эмбеддинг
.build/release/audio embed-speaker enrollment.wav --json
.build/release/audio embed-speaker enrollment.wav --engine coreml
```

Подробнее об архитектуре см. [Speaker Diarization](docs/inference/speaker-diarization.md).

## Улучшение речи — подавление шума и очистка аудио

### Подавление шума

```swift
import SpeechEnhancement
import AudioCommon  // для WAVWriter

let enhancer = try await SpeechEnhancer.fromPretrained()
// Загружает ~4.3 MB при первом запуске (Core ML FP16-модель + вспомогательные данные)

let cleanAudio = try enhancer.enhance(audio: noisyAudio, sampleRate: 48000)
try WAVWriter.write(samples: cleanAudio, sampleRate: 48000, to: outputURL)
```

### CLI для шумоподавления

```bash
make build

# Базовое подавление шума
.build/release/audio denoise noisy.wav

# Пользовательский путь вывода
.build/release/audio denoise noisy.wav --output clean.wav
```

Подробнее об архитектуре см. [Speech Enhancement](docs/inference/speech-enhancement.md).

## Конвейеры — композиция нескольких моделей

Все модели реализуют общие протоколы (`SpeechRecognitionModel`, `SpeechGenerationModel`, `SpeechEnhancementModel` и др.) и могут объединяться в конвейеры:

### Распознавание зашумлённой речи (DeepFilterNet + ASR)

```swift
import SpeechEnhancement
import Qwen3ASR

let enhancer = try await SpeechEnhancer.fromPretrained()
let asr = try await Qwen3ASRModel.fromPretrained()

// Улучшение на 48kHz, затем распознавание на 16kHz
let clean = try enhancer.enhance(audio: noisyAudio, sampleRate: 48000)
let clean16k = AudioResampler.resample(clean, from: 48000, to: 16000)
let text = asr.transcribe(audio: clean16k, sampleRate: 16000)
```

### Голосовая ретрансляция (VAD + ASR + TTS)

```swift
import SpeechVAD
import Qwen3ASR
import Qwen3TTS

let vad = try await SileroVADModel.fromPretrained()
let asr = try await Qwen3ASRModel.fromPretrained()
let tts = try await Qwen3TTSModel.fromPretrained()

// Обнаружение речевых сегментов, распознавание, повторный синтез
let segments = vad.detectSpeech(audio: audio, sampleRate: 16000)
for seg in segments {
    let chunk = Array(audio[Int(seg.startTime * 16000)..<Int(seg.endTime * 16000)])
    let text = asr.transcribe(audio: chunk, sampleRate: 16000)
    let speech = tts.synthesize(text: text, language: "english")
    // speech: моно-сэмплы float 24kHz
}
```

### Транскрипция совещаний (диаризация + ASR)

```swift
import SpeechVAD
import Qwen3ASR

let pipeline = try await DiarizationPipeline.fromPretrained()
let asr = try await Qwen3ASRModel.fromPretrained()

let result = pipeline.diarize(audio: meetingAudio, sampleRate: 16000)
for seg in result.segments {
    let chunk = Array(meetingAudio[Int(seg.startTime * 16000)..<Int(seg.endTime * 16000)])
    let text = asr.transcribe(audio: chunk, sampleRate: 16000)
    print("Спикер \(seg.speakerId) [\(seg.startTime)s-\(seg.endTime)s]: \(text)")
}
```

Полный справочник протоколов см. в [Shared Protocols](docs/shared-protocols.md).

## HTTP API-сервер

Автономный HTTP-сервер предоставляет доступ ко всем моделям через REST и WebSocket. Модели загружаются лениво при первом запросе.

```bash
swift build -c release
.build/release/audio-server --port 8080

# Распознавание речи
curl -X POST http://localhost:8080/transcribe --data-binary @audio.wav -H "Content-Type: audio/wav"

# Синтез речи
curl -X POST http://localhost:8080/speak -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "engine": "cosyvoice"}' -o output.wav

# Генерация речи из речи (PersonaPlex)
curl -X POST http://localhost:8080/respond --data-binary @question.wav -o response.wav

# Улучшение речи
curl -X POST http://localhost:8080/enhance --data-binary @noisy.wav -o clean.wav

# Предзагрузка всех моделей при запуске
.build/release/audio-server --preload --port 8080
```

### Потоковая передача по WebSocket

#### OpenAI Realtime API (`/v1/realtime`)

Основной WebSocket-эндпоинт реализует протокол [OpenAI Realtime API](https://platform.openai.com/docs/api-reference/realtime) — все сообщения в формате JSON с полем `type`, аудио в формате base64 PCM16 24kHz моно.

**События от клиента к серверу:**

| Событие | Описание |
|---------|----------|
| `session.update` | Настройка движка, языка, аудиоформата |
| `input_audio_buffer.append` | Отправка аудиофрагмента в base64 PCM16 |
| `input_audio_buffer.commit` | Запуск распознавания накопленного аудио (ASR) |
| `input_audio_buffer.clear` | Очистка аудиобуфера |
| `response.create` | Запрос синтеза речи (TTS) |

**События от сервера к клиенту:**

| Событие | Описание |
|---------|----------|
| `session.created` | Сессия инициализирована |
| `session.updated` | Конфигурация подтверждена |
| `input_audio_buffer.committed` | Аудио принято для распознавания |
| `conversation.item.input_audio_transcription.completed` | Результат ASR |
| `response.audio.delta` | Аудиофрагмент в base64 PCM16 (TTS) |
| `response.audio.done` | Потоковая передача аудио завершена |
| `response.done` | Ответ завершён с метаданными |
| `error` | Ошибка с типом и сообщением |

```javascript
const ws = new WebSocket('ws://localhost:8080/v1/realtime');

// ASR: отправка аудио, получение транскрипции
ws.send(JSON.stringify({ type: 'input_audio_buffer.append', audio: base64PCM16 }));
ws.send(JSON.stringify({ type: 'input_audio_buffer.commit' }));
// → получает: conversation.item.input_audio_transcription.completed

// TTS: отправка текста, получение потокового аудио
ws.send(JSON.stringify({
  type: 'response.create',
  response: { modalities: ['audio', 'text'], instructions: 'Hello world' }
}));
// → получает: response.audio.delta (base64-фрагменты), response.audio.done, response.done
```

Пример HTML-клиента находится в `Examples/websocket-client.html` — откройте его в браузере при запущенном сервере.

Сервер является отдельным модулем `AudioServer` и исполняемым файлом `audio-server` — он не добавляет Hummingbird/WebSocket к основному CLI `audio`.

## Задержка (M2 Max, 64 GB)

### ASR

| Модель | Бэкенд | RTF | 10с аудио обработано за |
|--------|--------|-----|-------------------------|
| Qwen3-ASR-0.6B (4-bit) | MLX | ~0.06 | ~0.6с |
| Qwen3-ASR-0.6B (INT8) | CoreML + MLX | ~0.09 | ~0.9с |
| Qwen3-ASR-1.7B (8-bit) | MLX | ~0.11 | ~1.1с |
| Parakeet-TDT-0.6B (INT4) | CoreML (Neural Engine) | ~0.12 холодный, ~0.03 прогретый | ~1.2с / ~0.3с |
| Whisper-large-v3 | whisper.cpp (Q5_0) | ~0.10 | ~1.0с |
| Whisper-small | whisper.cpp (Q5_0) | ~0.04 | ~0.4с |

### Принудительное выравнивание

| Модель | Фреймворк | 20с аудио | RTF |
|--------|-----------|-----------|-----|
| Qwen3-ForcedAligner-0.6B (4-bit) | MLX Swift (debug) | ~365мс | ~0.018 |

> Один неавторегрессионный прямой проход — без цикла сэмплирования. Аудио-энкодер доминирует (~328мс), один проход декодера ~37мс. **В 55 раз быстрее реального времени.**

### TTS

| Модель | Фреймворк | Короткий (1с) | Средний (3с) | Длинный (6с) | Потоковый первый пакет |
|--------|-----------|---------------|--------------|--------------|----------------------|
| Qwen3-TTS-0.6B (4-bit) | MLX Swift (release) | 1.6с (RTF 1.2) | 2.3с (RTF 0.7) | 3.9с (RTF 0.7) | ~120мс (1 кадр) |
| Kokoro-82M | CoreML (Neural Engine) | ~1.4с (RTFx 0.7) | ~4.3с (RTFx 0.7) | ~8.6с (RTFx 0.7) | Н/Д (неавторегрессионный) |
| Apple `AVSpeechSynthesizer` | AVFoundation | 0.08с | 0.08с | 0.17с (RTF 0.02) | Н/Д |

> Qwen3-TTS генерирует естественную, выразительную речь с просодией и эмоциями, работая **быстрее реального времени** (RTF < 1.0). Потоковый синтез выдаёт первый аудиофрагмент за ~120мс. Kokoro-82M работает полностью на Neural Engine с 3-этапным пайплайном (RTFx ~0.7), идеально для iOS. Встроенный TTS от Apple быстрее, но производит роботизированную, монотонную речь.

### PersonaPlex (генерация речи из речи)

| Модель | Фреймворк | мс/шаг | RTF | Примечания |
|--------|-----------|--------|-----|------------|
| PersonaPlex-7B (8-bit) | MLX Swift (release) | ~112мс | ~1.4 | Рекомендуется — связные ответы, на 30% быстрее 4-bit |
| PersonaPlex-7B (4-bit) | MLX Swift (release) | ~158мс | ~1.97 | Не рекомендуется — ухудшенное качество |

> **Используйте 8-bit.** INT8 быстрее (112 мс/шаг vs. 158 мс/шаг) и генерирует связные полнодуплексные ответы. Квантование INT4 ухудшает качество генерации, создавая бессвязную речь. INT8 работает со скоростью ~112мс/шаг на M2 Max.

### VAD и голосовые эмбеддинги

| Модель | Бэкенд | Задержка вызова | RTF | Примечания |
|--------|--------|----------------|-----|------------|
| Silero-VAD-v5 | MLX | ~2.1мс / фрагмент | 0.065 | GPU (Metal) |
| Silero-VAD-v5 | CoreML | ~0.27мс / фрагмент | 0.008 | Neural Engine, **в 7.7 раз быстрее** |
| WeSpeaker ResNet34-LM | MLX | ~310мс / 20с аудио | 0.016 | GPU (Metal) |
| WeSpeaker ResNet34-LM | CoreML | ~430мс / 20с аудио | 0.021 | Neural Engine, освобождает GPU |

> Silero VAD CoreML работает на Neural Engine в 7.7 раз быстрее MLX, что делает его идеальным для постоянного мониторинга микрофонного ввода. WeSpeaker MLX быстрее на GPU, но CoreML освобождает GPU для параллельных задач (TTS, ASR). Оба бэкенда дают эквивалентные результаты.

### Улучшение речи

| Модель | Бэкенд | Длительность | Задержка | RTF |
|--------|--------|-------------|---------|-----|
| DeepFilterNet3 (FP16) | CoreML | 5с | 0.65с | 0.13 |
| DeepFilterNet3 (FP16) | CoreML | 10с | 1.2с | 0.12 |
| DeepFilterNet3 (FP16) | CoreML | 20с | 4.8с | 0.24 |

RTF = Real-Time Factor (чем меньше, тем лучше, < 1.0 = быстрее реального времени). Стоимость GRU растёт как ~O(n^2).

### MLX и CoreML

Оба бэкенда дают эквивалентные результаты. Выбирайте в зависимости от вашей задачи:

| | MLX | CoreML |
|---|---|---|
| **Аппаратура** | GPU (Metal-шейдеры) | Neural Engine + CPU |
| **Лучше для** | Максимальная пропускная способность, работа с одной моделью | Мультимодельные конвейеры, фоновые задачи |
| **Энергопотребление** | Высокая загрузка GPU | Меньшее потребление, освобождает GPU |
| **Задержка** | Быстрее для крупных моделей (WeSpeaker) | Быстрее для мелких моделей (Silero VAD) |

**Настольный инференс**: MLX используется по умолчанию — максимальная производительность для одной модели на Apple Silicon. Переключайтесь на CoreML при одновременном запуске нескольких моделей (например, VAD + ASR + TTS), чтобы избежать конкуренции за GPU, или для энергозависимых задач на ноутбуках.

CoreML-модели доступны для энкодера Qwen3-ASR, Silero VAD и WeSpeaker. Для Qwen3-ASR используйте `--engine qwen3-coreml` (гибрид: CoreML-энкодер на ANE + MLX текстовый декодер на GPU). Для VAD/эмбеддингов передайте `engine: .coreml` при создании — API инференса идентичен.

## Бенчмарки точности

### ASR — Word Error Rate ([подробности](docs/benchmarks/asr-wer.md))

| Модель | WER% (LibriSpeech test-clean) | RTF |
|--------|-------------------------------|-----|
| Qwen3-ASR 1.7B 8-bit | **2.35** | 0.090 |
| Qwen3-ASR 1.7B 4-bit | 2.57 | 0.045 |
| Parakeet TDT INT8 | 2.74 | 0.089 |
| Qwen3-ASR 0.6B 8-bit | 2.80 | 0.025 |

Qwen3-ASR 1.7B 8-bit превосходит Whisper Large v3 Turbo (2.5%) при сопоставимом размере. Многоязычность: 10 языков протестировано на FLEURS.

### TTS — оценка через обратное распознавание ([подробности](docs/benchmarks/tts-roundtrip.md))

| Движок | WER% | RTF |
|--------|------|-----|
| CosyVoice3 | **3.25** | 0.59 |
| Qwen3-TTS 1.7B | 3.47 | 0.79 |
| Kokoro-82M | 3.90 | 0.17 |

### VAD — обнаружение речи ([подробности](docs/benchmarks/vad-detection.md))

| Движок | F1% (FLEURS) | RTF |
|--------|-------------|-----|
| FireRedVAD | **99.12** | 0.007 |
| Silero CoreML | 95.13 | 0.022 |
| Pyannote MLX | 94.86 | 0.358 |

## Архитектура

**Модели:** [ASR Model](docs/models/asr-model.md), [TTS Model](docs/models/tts-model.md), [CosyVoice TTS](docs/models/cosyvoice-tts.md), [Kokoro TTS](docs/models/kokoro-tts.md), [Parakeet TDT](docs/models/parakeet-asr.md), [PersonaPlex](docs/models/personaplex.md), [FireRedVAD](docs/models/fireredvad.md)

**Инференс:** [Qwen3-ASR](docs/inference/qwen3-asr-inference.md), [Parakeet TDT](docs/inference/parakeet-asr-inference.md), [TTS Inference](docs/inference/qwen3-tts-inference.md), [Forced Aligner](docs/inference/forced-aligner.md), [FireRedVAD](docs/inference/fireredvad.md), [Silero VAD](docs/inference/silero-vad.md), [Speaker Diarization](docs/inference/speaker-diarization.md), [Speech Enhancement](docs/inference/speech-enhancement.md), [Воспроизведение аудио](docs/audio/playback.md)

**Бенчмарки:** [ASR WER](docs/benchmarks/asr-wer.md), [TTS Round-Trip](docs/benchmarks/tts-roundtrip.md), [VAD Detection](docs/benchmarks/vad-detection.md)

**Справочник:** [Shared Protocols](docs/shared-protocols.md)

## Настройка кеша

Веса моделей кешируются локально. Изменить путь к кешу можно так:

```bash
export QWEN3_CACHE_DIR=/path/to/cache
```

## Библиотека MLX Metal

Если при запуске появляется ошибка `Failed to load the default metallib`, библиотека Metal-шейдеров отсутствует. Выполните `make build` (или `./scripts/build_mlx_metallib.sh release` после ручной `swift build`) для её компиляции. Если Metal Toolchain отсутствует, сначала установите его:

```bash
xcodebuild -downloadComponent MetalToolchain
```

## Тестирование

Юнит-тесты (конфигурация, сэмплирование, предобработка текста, коррекция временных меток) выполняются без загрузки моделей:

```bash
swift test --filter "Qwen3TTSConfigTests|SamplingTests|CosyVoiceTTSConfigTests|CamPlusPlusMelExtractorTests|PersonaPlexTests|ForcedAlignerTests/testText|ForcedAlignerTests/testTimestamp|ForcedAlignerTests/testLIS|SileroVADTests/testSilero|SileroVADTests/testReflection|SileroVADTests/testProcess|SileroVADTests/testReset|SileroVADTests/testDetect|SileroVADTests/testStreaming|SileroVADTests/testVADEvent|KokoroTTSTests"
```

Интеграционные тесты требуют весов моделей (загружаются автоматически при первом запуске):

```bash
# Круговая проверка TTS: синтез текста, сохранение WAV, обратное распознавание через ASR
swift test --filter TTSASRRoundTripTests

# Только ASR: распознавание тестового аудио
swift test --filter Qwen3ASRIntegrationTests

# E2E принудительного выравнивания: временные метки на уровне слов (~979 MB загрузка)
swift test --filter ForcedAlignerTests/testForcedAlignerE2E

# E2E PersonaPlex: конвейер генерации речи из речи (~5.5 GB загрузка)
PERSONAPLEX_E2E=1 swift test --filter PersonaPlexE2ETests
```

> **Примечание:** Библиотека MLX Metal должна быть скомпилирована перед запуском тестов, использующих MLX-операции.
> Инструкции см. в [MLX Metal Library](#библиотека-mlx-metal).

## Поддерживаемые языки

| Модель | Языки |
|--------|-------|
| Qwen3-ASR | 52 языка (CN, EN, кантонский, DE, FR, ES, JA, KO, RU, + 22 китайских диалекта, ...) |
| Parakeet TDT | 25 европейских языков (BG, CS, DA, DE, EL, EN, ES, ET, FI, FR, HR, HU, IT, LT, LV, MT, NL, PL, PT, RO, RU, SK, SL, SV, UK) |
| Qwen3-TTS | EN, CN, DE, JA, ES, FR, KO, RU, IT, PT (+ пекинский/сычуаньский диалекты через CustomVoice) |
| CosyVoice TTS | CN, EN, JA, KO, DE, ES, FR, IT, RU |
| Kokoro TTS | EN (US/UK), ES, FR, HI, IT, JA, PT, CN, KO, DE |
| PersonaPlex | EN |

## Сравнение

### Распознавание речи (ASR): speech-swift и альтернативы

| | **speech-swift (Qwen3-ASR)** | **whisper.cpp** | **Apple SFSpeechRecognizer** | **Google Cloud Speech** |
|---|---|---|---|---|
| **Среда выполнения** | На устройстве (MLX/CoreML) | На устройстве (CPU/GPU) | На устройстве или облако | Только облако |
| **Языки** | 52 | 100+ | ~70 (на устройстве: ограничено) | 125+ |
| **RTF (10с аудио, M2 Max)** | 0.06 (в 17 раз быстрее реального) | 0.10 (Whisper-large-v3) | Н/Д | Н/Д |
| **Потоковый режим** | Нет (пакетный) | Нет (пакетный) | Да | Да |
| **Свои модели** | Да (замена весов HuggingFace) | Да (GGML-модели) | Нет | Нет |
| **Swift API** | Нативный async/await | C++ с мостом в Swift | Нативный | REST/gRPC |
| **Приватность** | Полностью на устройстве | Полностью на устройстве | Зависит от настроек | Данные отправляются в облако |
| **Пословные метки** | Да (Forced Aligner) | Да | Ограничено | Да |
| **Стоимость** | Бесплатно (Apache 2.0) | Бесплатно (MIT) | Бесплатно (на устройстве) | Оплата за минуту |

### Синтез речи (TTS): speech-swift и альтернативы

| | **speech-swift (Qwen3-TTS)** | **speech-swift (Kokoro)** | **Apple AVSpeechSynthesizer** | **ElevenLabs / Cloud TTS** |
|---|---|---|---|---|
| **Качество** | Нейросетевое, выразительное | Нейросетевое, естественное | Роботизированное, монотонное | Нейросетевое, наивысшее качество |
| **Среда выполнения** | На устройстве (MLX) | На устройстве (CoreML) | На устройстве | Только облако |
| **Потоковый режим** | Да (~120мс первый фрагмент) | Нет (3-этапный пайплайн) | Нет | Да |
| **Клонирование голоса** | Да | Нет | Нет | Да |
| **Голоса** | 9 встроенных + клонирование | 54 предустановленных | ~50 системных | 1000+ |
| **Языки** | 10 | 10 | 60+ | 30+ |
| **Поддержка iOS** | Только macOS | iOS + macOS | iOS + macOS | Любая платформа (API) |
| **Стоимость** | Бесплатно (Apache 2.0) | Бесплатно (Apache 2.0) | Бесплатно | Оплата за символ |

### Когда использовать speech-swift

- **Критичная приватность** — медицина, юриспруденция, корпоративные задачи, где аудио не должно покидать устройство
- **Офлайн-использование** — интернет не нужен после первоначальной загрузки модели
- **Экономия** — без поминутной или посимвольной оплаты API
- **Оптимизация для Apple Silicon** — разработан специально для GPU M-серии (Metal) и Neural Engine
- **Полный конвейер** — ASR + TTS + VAD + диаризация + улучшение речи в одном Swift-пакете

## Частые вопросы

**Работает ли speech-swift на iOS?**
Kokoro TTS, Qwen3.5-Chat (CoreML), Silero VAD, Parakeet ASR, DeepFilterNet3 и WeSpeaker работают на iOS 17+ через CoreML на Neural Engine. Модели на базе MLX (Qwen3-ASR, Qwen3-TTS, Qwen3.5-Chat MLX, PersonaPlex) требуют macOS 14+ на Apple Silicon.

**Нужен ли интернет?**
Только для первоначальной загрузки модели с HuggingFace (автоматически, кешируется в `~/Library/Caches/qwen3-speech/`). После этого весь инференс работает полностью офлайн без сетевого доступа.

**Как speech-swift соотносится с Whisper?**
Qwen3-ASR-0.6B достигает RTF 0.06 на M2 Max — на 40% быстрее, чем Whisper-large-v3 через whisper.cpp (RTF 0.10) — с сопоставимой точностью на 52 языках. speech-swift предоставляет нативный Swift API с async/await, тогда как whisper.cpp требует моста C++.

**Можно ли использовать в коммерческом приложении?**
Да. speech-swift распространяется под лицензией Apache 2.0. У весов моделей свои лицензии (проверяйте на странице HuggingFace каждой модели).

**Какие чипы Apple Silicon поддерживаются?**
Все чипы M-серии: M1, M2, M3, M4 и их варианты Pro/Max/Ultra. Требуется macOS 14+ (Sonoma) или iOS 17+.

**Сколько памяти нужно?**
От ~3 MB (Silero VAD) до ~6.5 GB (PersonaPlex 7B). Kokoro TTS использует ~500 MB, Qwen3-ASR ~2.2 GB. Полные данные см. в таблице [Требования к памяти](#требования-к-памяти).

**Можно ли запускать несколько моделей одновременно?**
Да. Используйте CoreML-модели на Neural Engine вместе с MLX-моделями на GPU, чтобы избежать конкуренции — например, Silero VAD (CoreML) + Qwen3-ASR (MLX) + Qwen3-TTS (MLX).

**Есть ли REST API?**
Да. Исполняемый файл `audio-server` предоставляет доступ ко всем моделям через HTTP REST и WebSocket, включая WebSocket-эндпоинт, совместимый с OpenAI Realtime API, по адресу `/v1/realtime`.

## Участие в проекте

Мы приветствуем вклад в проект! Будь то исправление ошибки, интеграция новой модели или улучшение документации — пулл-реквесты приветствуются.

**Как начать:**
1. Сделайте форк репозитория и создайте ветку для новой функциональности
2. `make build` для компиляции (требуется Xcode + Metal Toolchain)
3. `make test` для запуска тестов
4. Откройте пулл-реквест в ветку `main`

## Лицензия

Apache 2.0
