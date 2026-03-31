# Speech Swift

Modelos de IA para fala em Apple Silicon, com tecnologia MLX Swift e CoreML.

📖 Read in: [English](README.md) · [中文](README_zh.md) · [日本語](README_ja.md) · [한국어](README_ko.md) · [Español](README_es.md) · [Deutsch](README_de.md) · [Français](README_fr.md) · [हिन्दी](README_hi.md) · [Português](README_pt.md) · [Русский](README_ru.md)

Reconhecimento, sintese e compreensao de fala no dispositivo para Mac e iOS. Executa localmente no Apple Silicon — sem nuvem, sem chaves de API, nenhum dado sai do dispositivo.

[Instale via Homebrew](#homebrew) ou adicione como dependencia do Swift Package.

**[Documentacao](https://soniqo.audio)** · **[Modelos no HuggingFace](https://huggingface.co/aufklarer)** · **[Blog](https://blog.ivan.digital)**

- **Qwen3-ASR** — Fala para texto / reconhecimento de fala (reconhecimento automatico de fala, 52 idiomas)
- **Parakeet TDT** — Fala para texto via CoreML (Neural Engine, NVIDIA FastConformer + decodificador TDT, 25 idiomas)
- **Qwen3-ForcedAligner** — Alinhamento de timestamps por palavra (audio + texto → timestamps)
- **Qwen3-TTS** — Sintese de texto para fala (mais alta qualidade, streaming, locutores personalizados, 10 idiomas)
- **CosyVoice TTS** — Texto para fala com streaming, clonagem de voz, dialogo multi-locutor e tags de emocao (9 idiomas, DiT flow matching, codificador de locutor CAM++)
- **Kokoro TTS** — Texto para fala no dispositivo (82M parametros, CoreML/Neural Engine, 54 vozes, pronto para iOS, 10 idiomas)
- **Qwen3.5-Chat** — Chat com LLM no dispositivo (0.8B, MLX + CoreML, INT4 + CoreML INT8, DeltaNet hibrido, tokens em streaming)
- **PersonaPlex** — Conversa fala-a-fala full-duplex (7B, audio de entrada → audio de saida, 18 presets de voz)
- **DeepFilterNet3** — Aprimoramento de fala / supressao de ruido (2.1M parametros, tempo real a 48kHz)
- **FireRedVAD** — Deteccao offline de atividade de voz (DFSMN, CoreML, 100+ idiomas, 97.6% F1)
- **Silero VAD** — Deteccao de atividade de voz em streaming (blocos de 32ms, latencia sub-milissegundo)
- **Pyannote VAD** — Deteccao offline de atividade de voz (janelas de 10s, sobreposicao multi-locutor)
- **Speaker Diarization** — Quem falou quando (segmentacao Pyannote + encadeamento de falantes por atividade, ou Sortformer ponta-a-ponta no Neural Engine)
- **Speaker Embeddings** — Verificacao e identificacao de falantes (WeSpeaker ResNet34, vetores de 256 dimensoes)

Papers: [Qwen3-ASR](https://arxiv.org/abs/2601.21337) (Alibaba), [Qwen3-TTS](https://arxiv.org/abs/2601.15621) (Alibaba), [Qwen3](https://arxiv.org/abs/2505.09388) (Alibaba), [Parakeet TDT](https://arxiv.org/abs/2304.06795) (NVIDIA), [CosyVoice 3](https://arxiv.org/abs/2505.17589) (Alibaba), [Kokoro](https://arxiv.org/abs/2301.01695) (StyleTTS 2), [PersonaPlex](https://arxiv.org/abs/2602.06053) (NVIDIA), [Mimi](https://arxiv.org/abs/2410.00037) (Kyutai), [Sortformer](https://arxiv.org/abs/2409.06656) (NVIDIA)

## Roadmap

Veja a [discussao do Roadmap](https://github.com/soniqo/speech-swift/discussions/81) para o que esta planejado — comentarios e sugestoes sao bem-vindos!

## Novidades

- **20 Mar 2026** — [We Beat Whisper Large v3 with a 600M Model Running Entirely on Your Mac](https://blog.ivan.digital/we-beat-whisper-large-v3-with-a-600m-model-running-entirely-on-your-mac-20e6ce191174)
- **26 Feb 2026** — [Speaker Diarization and Voice Activity Detection on Apple Silicon — Native Swift with MLX](https://blog.ivan.digital/speaker-diarization-and-voice-activity-detection-on-apple-silicon-native-swift-with-mlx-92ea0c9aca0f)
- **23 Feb 2026** — [NVIDIA PersonaPlex 7B on Apple Silicon — Full-Duplex Speech-to-Speech in Native Swift with MLX](https://blog.ivan.digital/nvidia-personaplex-7b-on-apple-silicon-full-duplex-speech-to-speech-in-native-swift-with-mlx-0aa5276f2e23)
- **12 Feb 2026** — [Qwen3-ASR Swift: On-Device ASR + TTS for Apple Silicon — Architecture and Benchmarks](https://blog.ivan.digital/qwen3-asr-swift-on-device-asr-tts-for-apple-silicon-architecture-and-benchmarks-27cbf1e4463f)

## Modelos

| Modelo | Tarefa | Streaming | Idiomas | Tamanhos |
|--------|--------|-----------|---------|----------|
| Qwen3-ASR-0.6B | Fala → Texto | Nao | 52 idiomas | [4-bit](https://huggingface.co/aufklarer/Qwen3-ASR-0.6B-MLX-4bit) 680 MB · [8-bit](https://huggingface.co/aufklarer/Qwen3-ASR-0.6B-MLX-8bit) 1.0 GB · [CoreML](https://huggingface.co/aufklarer/Qwen3-ASR-CoreML) 180 MB |
| Qwen3-ASR-1.7B | Fala → Texto | Nao | 52 idiomas | [4-bit](https://huggingface.co/aufklarer/Qwen3-ASR-1.7B-MLX-4bit) 2.1 GB · [8-bit](https://huggingface.co/aufklarer/Qwen3-ASR-1.7B-MLX-8bit) 3.2 GB |
| Parakeet-TDT-0.6B | Fala → Texto | Nao | 25 idiomas europeus | [CoreML INT4](https://huggingface.co/aufklarer/Parakeet-TDT-v3-CoreML-INT4) 315 MB · [CoreML INT8](https://huggingface.co/aufklarer/Parakeet-TDT-v3-CoreML-INT8) 500 MB |
| Qwen3-ForcedAligner-0.6B | Audio + Texto → Timestamps | Nao | Multi | [4-bit](https://huggingface.co/aufklarer/Qwen3-ForcedAligner-0.6B-4bit) 979 MB · [8-bit](https://huggingface.co/aufklarer/Qwen3-ForcedAligner-0.6B-8bit) 1.4 GB · [CoreML INT4](https://huggingface.co/aufklarer/Qwen3-ForcedAligner-0.6B-CoreML-INT4) 630 MB · [CoreML INT8](https://huggingface.co/aufklarer/Qwen3-ForcedAligner-0.6B-CoreML-INT8) 1.0 GB |
| Qwen3-TTS-0.6B Base | Texto → Fala | Sim (~120ms) | 10 idiomas | [4-bit](https://huggingface.co/aufklarer/Qwen3-TTS-12Hz-0.6B-Base-MLX-4bit) 1.7 GB · [8-bit](https://huggingface.co/aufklarer/Qwen3-TTS-12Hz-0.6B-Base-MLX-8bit) 2.4 GB |
| Qwen3-TTS-0.6B CustomVoice | Texto → Fala | Sim (~120ms) | 10 idiomas | [4-bit](https://huggingface.co/aufklarer/Qwen3-TTS-12Hz-0.6B-CustomVoice-MLX-4bit) 1.7 GB |
| Qwen3-TTS-1.7B Base | Texto → Fala | Sim (~120ms) | 10 idiomas | [4-bit](https://huggingface.co/aufklarer/Qwen3-TTS-12Hz-1.7B-Base-MLX-4bit) 3.2 GB · [8-bit](https://huggingface.co/aufklarer/Qwen3-TTS-12Hz-1.7B-Base-MLX-8bit) 4.8 GB |
| CosyVoice3-0.5B | Texto → Fala | Sim (~150ms) | 9 idiomas | [4-bit](https://huggingface.co/aufklarer/CosyVoice3-0.5B-MLX-4bit) 1.2 GB |
| Kokoro-82M | Texto → Fala | Nao | 10 idiomas | [CoreML](https://huggingface.co/aufklarer/Kokoro-82M-CoreML) ~170 MB |
| Qwen3.5-0.8B Chat | Text → Text (LLM) | Yes (streaming) | Multi | [MLX INT4](https://huggingface.co/aufklarer/Qwen3.5-0.8B-Chat-MLX) 418 MB · [CoreML INT8](https://huggingface.co/aufklarer/Qwen3.5-0.8B-Chat-CoreML) 981 MB |
| PersonaPlex-7B | Fala → Fala | Sim (~2s por bloco) | EN | [4-bit](https://huggingface.co/aufklarer/PersonaPlex-7B-MLX-4bit) 4.9 GB · [8-bit](https://huggingface.co/aufklarer/PersonaPlex-7B-MLX-8bit) 9.1 GB |
| FireRedVAD | Deteccao de Atividade de Voz | Nao (offline) | 100+ idiomas | [CoreML](https://huggingface.co/aufklarer/FireRedVAD-CoreML) ~1.2 MB |
| Silero-VAD-v5 | Deteccao de Atividade de Voz | Sim (blocos de 32ms) | Independente de idioma | [MLX](https://huggingface.co/aufklarer/Silero-VAD-v5-MLX) · [CoreML](https://huggingface.co/aufklarer/Silero-VAD-v5-CoreML) ~1.2 MB |
| Pyannote-Segmentation-3.0 | VAD + Segmentacao de Falantes | Nao (janelas de 10s) | Independente de idioma | [MLX](https://huggingface.co/aufklarer/Pyannote-Segmentation-MLX) ~5.7 MB |
| DeepFilterNet3 | Aprimoramento de Fala | Sim (quadros de 10ms) | Independente de idioma | [CoreML FP16](https://huggingface.co/aufklarer/DeepFilterNet3-CoreML) ~4.2 MB |
| WeSpeaker-ResNet34-LM | Embedding de Falante (256-dim) | Nao | Independente de idioma | [MLX](https://huggingface.co/aufklarer/WeSpeaker-ResNet34-LM-MLX) · [CoreML](https://huggingface.co/aufklarer/WeSpeaker-ResNet34-LM-CoreML) ~25 MB |
| CAM++ | Embedding de Falante (192-dim) | Nao | Independente de idioma | [CoreML](https://huggingface.co/aufklarer/CamPlusPlus-Speaker-CoreML) ~14 MB |
| Sortformer | Diarizacao de Falantes (ponta-a-ponta) | Sim (em blocos) | Independente de idioma | [CoreML](https://huggingface.co/aufklarer/Sortformer-Diarization-CoreML) ~240 MB |

### Requisitos de Memoria

Memoria de pesos e a memoria da GPU (MLX) ou ANE (CoreML) consumida pelos parametros do modelo. Pico de inferencia inclui caches KV, ativacoes e tensores intermediarios.

| Modelo | Memoria de Pesos | Pico de Inferencia |
|--------|-----------------|-------------------|
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

### Quando Usar Qual TTS

- **Qwen3-TTS**: Melhor qualidade, streaming (~120ms), 9 locutores integrados, 10 idiomas, sintese em lote
- **CosyVoice TTS**: Streaming (~150ms), 9 idiomas, clonagem de voz (codificador de locutor CAM++), dialogo multi-locutor (`[S1] ... [S2] ...`), tags de emocao/estilo em linha (`(happy)`, `(whispers)`), DiT flow matching + vocoder HiFi-GAN
- **Kokoro TTS**: TTS leve pronto para iOS (82M parametros), CoreML/Neural Engine, 54 vozes, 10 idiomas, nao-autorregressivo (pipeline de 3 estagios)
- **PersonaPlex**: Fala-a-fala full-duplex (audio de entrada → audio de saida), streaming (~2s por bloco), 18 presets de voz, baseado na arquitetura Moshi

## Instalacao

### Homebrew

Requer Homebrew nativo ARM (`/opt/homebrew`). Homebrew via Rosetta/x86_64 nao e suportado.

```bash
brew tap soniqo/speech https://github.com/soniqo/speech-swift
brew install speech
```

Depois use:

```bash
audio transcribe recording.wav
audio speak "Hello world"
audio speak "Hallo Welt" --engine cosyvoice --language german
audio respond --input question.wav --transcript
```

> Para conversa interativa por voz com entrada de microfone, veja **[PersonaPlexDemo](Examples/PersonaPlexDemo/)**.

### Swift Package Manager

Adicione ao seu `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/soniqo/speech-swift", branch: "main")
]
```

Importe o modulo que voce precisa:

```swift
import Qwen3ASR      // Reconhecimento de fala (MLX)
import ParakeetASR   // Reconhecimento de fala (CoreML)
import Qwen3TTS      // Texto para fala (Qwen3)
import CosyVoiceTTS  // Texto para fala (streaming)
import KokoroTTS     // Texto para fala (CoreML, pronto para iOS)
import Qwen3Chat     // Chat com LLM no dispositivo (CoreML)
import PersonaPlex   // Fala-a-fala (full-duplex)
import SpeechVAD          // Deteccao de atividade de voz (pyannote + Silero)
import SpeechEnhancement  // Supressao de ruido (DeepFilterNet3)
import AudioCommon        // Utilitarios compartilhados
```

### Requisitos

- Swift 5.9+
- macOS 14+ ou iOS 17+
- Apple Silicon (M1/M2/M3/M4)
- Xcode 15+ (com Metal Toolchain — execute `xcodebuild -downloadComponent MetalToolchain` se estiver faltando)

### Compilar a Partir do Codigo-Fonte

```bash
git clone https://github.com/soniqo/speech-swift
cd speech-swift
make build
```

Isso compila o pacote Swift **e** a biblioteca de shaders Metal do MLX em um unico passo. A biblioteca Metal (`mlx.metallib`) e necessaria para inferencia via GPU — sem ela voce vera `Failed to load the default metallib` em tempo de execucao.

Para builds de depuracao: `make debug`. Para executar testes unitarios: `make test`.

## Experimente o Assistente de Voz

**[PersonaPlexDemo](Examples/PersonaPlexDemo/)** e um assistente de voz para macOS pronto para usar — toque para falar e receba respostas faladas em tempo real. Usa entrada de microfone com Silero VAD para deteccao automatica de fala, Qwen3-ASR para transcricao e PersonaPlex 7B para geracao fala-a-fala. Conversa multi-turno com 18 presets de voz e exibicao de transcricao do monologo interno.

```bash
make build  # a partir da raiz do repositorio — compila tudo incluindo o metallib do MLX
cd Examples/PersonaPlexDemo
# Veja Examples/PersonaPlexDemo/README.md para instrucoes do pacote .app
```

> RTF ~0.94 no M2 Max (mais rapido que tempo real). Os modelos sao baixados automaticamente na primeira execucao (~5.5 GB PersonaPlex + ~400 MB ASR).

## Apps de Demonstracao

- **[PersonaPlexDemo](Examples/PersonaPlexDemo/)** — Assistente de voz conversacional (entrada por microfone, VAD, multi-turno). Veja acima.
- **[SpeechDemo](Examples/SpeechDemo/)** — Ditado (Parakeet TDT / Qwen3-ASR com deteccao automatica de idioma) e sintese de texto para fala (Qwen3-TTS) em uma interface com abas.

Compile e execute como pacote `.app` para macOS — veja o README de cada demo para instrucoes.

## Fala para Texto (ASR) — Transcrever Audio em Swift

### Transcricao Basica

```swift
import Qwen3ASR

// Padrao: modelo 0.6B
let model = try await Qwen3ASRModel.fromPretrained()

// Ou use o modelo maior 1.7B para melhor precisao
let model = try await Qwen3ASRModel.fromPretrained(
    modelId: "aufklarer/Qwen3-ASR-1.7B-MLX-8bit"
)

// O audio pode ter qualquer taxa de amostragem — reamostrado automaticamente para 16kHz internamente
let transcription = model.transcribe(audio: audioSamples, sampleRate: 16000)
print(transcription)
```

### Encoder CoreML (Neural Engine)

Modo hibrido: encoder CoreML no Neural Engine + decodificador de texto MLX na GPU. Menor consumo de energia, libera a GPU durante a passagem do encoder.

```swift
import Qwen3ASR

let encoder = try await CoreMLASREncoder.fromPretrained()
let model = try await Qwen3ASRModel.fromPretrained()
let text = try model.transcribe(audio: audioSamples, sampleRate: 16000, coremlEncoder: encoder)
```

Variantes INT8 (180 MB, padrao) e INT4 (90 MB) disponiveis. INT8 recomendado (similaridade cosseno > 0.999 vs FP32).

### Parakeet TDT (CoreML)

```swift
import ParakeetASR

let model = try await ParakeetASRModel.fromPretrained()
let transcription = model.transcribe(audio: audioSamples, sampleRate: 16000)
```

Executa no Neural Engine via CoreML — libera a GPU para cargas de trabalho concorrentes. 25 idiomas europeus, ~315 MB.

### CLI de ASR

```bash
make build  # ou: swift build -c release && ./scripts/build_mlx_metallib.sh release

# Padrao (Qwen3-ASR 0.6B, MLX)
.build/release/audio transcribe audio.wav

# Usar modelo 1.7B
.build/release/audio transcribe audio.wav --model 1.7B

# Encoder CoreML (Neural Engine + decodificador MLX)
.build/release/audio transcribe --engine qwen3-coreml audio.wav

# Parakeet TDT (CoreML, Neural Engine)
.build/release/audio transcribe --engine parakeet audio.wav
```

## Alinhamento Forcado

### Timestamps por Palavra

```swift
import Qwen3ASR

let aligner = try await Qwen3ForcedAligner.fromPretrained()
// Baixa ~979 MB na primeira execucao

let aligned = aligner.align(
    audio: audioSamples,
    text: "Can you guarantee that the replacement part will be shipped tomorrow?",
    sampleRate: 24000
)

for word in aligned {
    print("[\(String(format: "%.2f", word.startTime))s - \(String(format: "%.2f", word.endTime))s] \(word.text)")
}
```

### CLI de Alinhamento Forcado

```bash
swift build -c release

# Alinhar com texto fornecido
.build/release/audio align audio.wav --text "Hello world"

# Transcrever primeiro, depois alinhar
.build/release/audio align audio.wav
```

Saida:
```
[0.12s - 0.45s] Can
[0.45s - 0.72s] you
[0.72s - 1.20s] guarantee
...
```

Nao-autorregressivo — pipeline de 3 estagios, sem loop de amostragem. Veja [Forced Aligner](docs/inference/forced-aligner.md) para detalhes da arquitetura.

## Texto para Fala (TTS) — Gerar Fala em Swift

### Sintese Basica

```swift
import Qwen3TTS
import AudioCommon  // para WAVWriter

let model = try await Qwen3TTSModel.fromPretrained()
// Baixa ~1.7 GB na primeira execucao (modelo + pesos do codec)
let audio = model.synthesize(text: "Hello world", language: "english")
// Saida: amostras float mono a 24kHz
try WAVWriter.write(samples: audio, sampleRate: 24000, to: outputURL)
```

### CLI de TTS

```bash
make build
.build/release/audio speak "Hello world" --output output.wav --language english
```

### Voz Personalizada / Selecao de Locutor

A variante **CustomVoice** do modelo suporta 9 vozes de locutores integradas e instrucoes em linguagem natural para controle de tom/estilo. Carregue-a passando o ID do modelo CustomVoice:

```swift
import Qwen3TTS

// Carregar o modelo CustomVoice (baixa ~1.7 GB na primeira execucao)
let model = try await Qwen3TTSModel.fromPretrained(
    modelId: TTSModelVariant.customVoice.rawValue
)

// Sintetizar com um locutor especifico
let audio = model.synthesize(text: "Hello world", language: "english", speaker: "vivian")

// Listar locutores disponiveis
print(model.availableSpeakers)  // ["aiden", "dylan", "eric", ...]
```

CLI:

```bash
# Usar modelo CustomVoice com um locutor
.build/release/audio speak "Hello world" --model customVoice --speaker vivian --output vivian.wav

# Listar locutores disponiveis
.build/release/audio speak --model customVoice --list-speakers
```

### Clonagem de Voz (modelo Base)

Clone a voz de um locutor a partir de um arquivo de audio de referencia. Dois modos:

**Modo ICL** (recomendado) — codifica o audio de referencia em tokens de codec com transcricao. Maior qualidade, EOS confiavel:

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

**Modo X-vector** — apenas embedding do locutor, sem necessidade de transcricao, porem menor qualidade:

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

### Instrucoes de Tom / Estilo (apenas CustomVoice)

O modelo CustomVoice aceita um parametro `instruct` em linguagem natural para controlar estilo de fala, tom, emocao e ritmo. A instrucao e adicionada a entrada do modelo no formato ChatML.

```swift
// Tom alegre
let audio = model.synthesize(
    text: "Welcome to our store!",
    language: "english",
    speaker: "ryan",
    instruct: "Speak in a cheerful, upbeat tone"
)

// Lento e serio
let audio = model.synthesize(
    text: "We regret to inform you...",
    language: "english",
    speaker: "aiden",
    instruct: "Read this slowly and solemnly"
)

// Sussurrando
let audio = model.synthesize(
    text: "Can you keep a secret?",
    language: "english",
    speaker: "vivian",
    instruct: "Whisper this softly"
)
```

CLI:

```bash
# Com instrucao de estilo
.build/release/audio speak "Good morning!" --model customVoice --speaker ryan \
    --instruct "Speak in a cheerful, upbeat tone" --output cheerful.wav

# O instruct padrao ("Speak naturally.") e aplicado automaticamente ao usar CustomVoice
.build/release/audio speak "Hello world" --model customVoice --speaker ryan --output natural.wav
```

Quando nenhum `--instruct` e fornecido com o modelo CustomVoice, `"Speak naturally."` e aplicado automaticamente para evitar saida prolixa. O modelo Base nao suporta instruct.

### Sintese em Lote

Sintetize multiplos textos em uma unica passagem em lote para maior vazao:

```swift
let texts = ["Good morning everyone.", "The weather is nice today.", "Please open the window."]
let audioList = model.synthesizeBatch(texts: texts, language: "english", maxBatchSize: 4)
// audioList[i] sao amostras float mono a 24kHz para texts[i]
for (i, audio) in audioList.enumerated() {
    try WAVWriter.write(samples: audio, sampleRate: 24000, to: URL(fileURLWithPath: "output_\(i).wav"))
}
```

#### CLI em Lote

```bash
# Crie um arquivo com um texto por linha
echo "Hello world.\nGoodbye world." > texts.txt
.build/release/audio speak --batch-file texts.txt --output output.wav --batch-size 4
# Produz output_0.wav, output_1.wav, ...
```

> O modo em lote amortiza o carregamento dos pesos do modelo entre os itens. Espere ~1.5-2.5x de melhoria na vazao para B=4 em Apple Silicon. Melhores resultados quando os textos produzem audio de comprimento similar.

### Opcoes de Amostragem

```swift
let config = SamplingConfig(temperature: 0.9, topK: 50, repetitionPenalty: 1.05)
let audio = model.synthesize(text: "Hello", language: "english", sampling: config)
```

### Sintese em Streaming

Emita blocos de audio incrementalmente para baixa latencia do primeiro pacote:

```swift
let stream = model.synthesizeStream(
    text: "Hello, this is streaming synthesis.",
    language: "english",
    streaming: .lowLatency  // ~120ms ate o primeiro bloco de audio
)

for try await chunk in stream {
    // chunk.samples: [Float] PCM @ 24kHz
    // chunk.isFinal: true no ultimo bloco
    playAudio(chunk.samples)
}
```

CLI:

```bash
# Streaming padrao (primeiro bloco de 3 quadros, ~225ms de latencia)
.build/release/audio speak "Hello world" --stream

# Baixa latencia (primeiro bloco de 1 quadro, ~120ms de latencia)
.build/release/audio speak "Hello world" --stream --first-chunk-frames 1
```

## Fala-a-Fala — Conversa Full-Duplex por Voz

> Para um assistente de voz interativo com entrada de microfone, veja **[PersonaPlexDemo](Examples/PersonaPlexDemo/)** — toque para falar, conversa multi-turno com deteccao automatica de fala.

### Fala-a-Fala

```swift
import PersonaPlex
import AudioCommon  // para WAVWriter, AudioFileLoader

let model = try await PersonaPlexModel.fromPretrained()
// Baixa ~5.5 GB na primeira execucao (temporal 4-bit + depformer + codec Mimi + presets de voz)

let audio = try AudioFileLoader.load(url: inputURL, targetSampleRate: 24000)
let (response, textTokens) = model.respond(userAudio: audio, voice: .NATM0)
// response: amostras float mono a 24kHz
// textTokens: monologo interno do modelo (IDs de tokens SentencePiece)
try WAVWriter.write(samples: response.audio, sampleRate: 24000, to: outputURL)
```

### Monologo Interno (Saida de Texto)

PersonaPlex gera tokens de texto junto com o audio — o raciocinio interno do modelo. Decodifique-os com o decodificador SentencePiece integrado:

```swift
let decoder = try SentencePieceDecoder(modelPath: "tokenizer_spm_32k_3.model")
let transcript = decoder.decode(textTokens)
print(transcript)  // ex.: "Sure, I can help you with that..."
```

### Fala-a-Fala em Streaming

```swift
// Receba blocos de audio conforme sao gerados (~2s por bloco)
let stream = model.respondStream(userAudio: audio, voice: .NATM0)
for try await chunk in stream {
    playAudio(chunk.samples)  // reproduza imediatamente, 24kHz mono
    // chunk.textTokens contem o texto deste bloco; o bloco final tem todos os tokens
    if chunk.isFinal { break }
}
```

### Selecao de Voz

18 presets de voz disponiveis:
- **Natural Feminino**: NATF0, NATF1, NATF2, NATF3
- **Natural Masculino**: NATM0, NATM1, NATM2, NATM3
- **Variedade Feminino**: VARF0, VARF1, VARF2, VARF3, VARF4
- **Variedade Masculino**: VARM0, VARM1, VARM2, VARM3, VARM4

### Prompts de Sistema

O prompt de sistema direciona o comportamento conversacional do modelo. Passe qualquer prompt personalizado como uma string simples:

```swift
// Prompt de sistema personalizado (tokenizado automaticamente)
let response = model.respond(
    userAudio: audio,
    voice: .NATM0,
    systemPrompt: "You enjoy having a good conversation."
)

// Ou usar um preset
let response = model.respond(
    userAudio: audio,
    voice: .NATM0,
    systemPromptTokens: SystemPromptPreset.customerService.tokens
)
```

Presets disponiveis: `focused` (padrao), `assistant`, `customerService`, `teacher`.

### CLI PersonaPlex

```bash
make build

# Fala-a-fala basico
.build/release/audio respond --input question.wav --output response.wav

# Com transcricao (decodifica texto do monologo interno)
.build/release/audio respond --input question.wav --transcript

# Saida JSON (caminho do audio, transcricao, metricas de latencia)
.build/release/audio respond --input question.wav --json

# Texto de prompt de sistema personalizado
.build/release/audio respond --input question.wav --system-prompt-text "You enjoy having a good conversation."

# Escolher uma voz e preset de prompt de sistema
.build/release/audio respond --input question.wav --voice NATF1 --system-prompt focused

# Ajustar parametros de amostragem
.build/release/audio respond --input question.wav --audio-temp 0.6 --repetition-penalty 1.5

# Ativar parada antecipada por entropia de texto (para se o texto colapsar)
.build/release/audio respond --input question.wav --entropy-threshold 1.0 --entropy-window 5

# Listar vozes e prompts disponiveis
.build/release/audio respond --list-voices
.build/release/audio respond --list-prompts
```

## CosyVoice TTS — Texto para Fala em Streaming com Clonagem de Voz

### Sintese Basica

```swift
import CosyVoiceTTS
import AudioCommon  // para WAVWriter

let model = try await CosyVoiceTTSModel.fromPretrained()
// Baixa ~1.9 GB na primeira execucao (pesos LLM + DiT + HiFi-GAN)

let audio = model.synthesize(text: "Hello, how are you today?", language: "english")
// Saida: amostras float mono a 24kHz
try WAVWriter.write(samples: audio, sampleRate: 24000, to: outputURL)
```

### Sintese em Streaming

```swift
// Streaming: receba blocos de audio conforme sao gerados (~150ms ate o primeiro bloco)
for try await chunk in model.synthesizeStream(text: "Hello, how are you today?", language: "english") {
    // chunk.audio: [Float], chunk.sampleRate: Int
    playAudio(chunk.audio)  // reproduza imediatamente
}
```

### Clonagem de Voz (CosyVoice)

Clone a voz de um locutor usando o codificador de locutor CAM++ (192-dim, CoreML Neural Engine):

```swift
import CosyVoiceTTS
import AudioCommon

let model = try await CosyVoiceTTSModel.fromPretrained()
let speaker = try await CamPlusPlusSpeaker.fromPretrained()
// Baixa ~14 MB do modelo CoreML CAM++ no primeiro uso

let refAudio = try AudioFileLoader.load(url: referenceURL, targetSampleRate: 16000)
let embedding = try speaker.embed(audio: refAudio, sampleRate: 16000)
// embedding: [Float] de comprimento 192

let audio = model.synthesize(
    text: "Hello in a cloned voice!",
    language: "english",
    speakerEmbedding: embedding
)
```

### CLI CosyVoice TTS

```bash
make build

# Sintese basica
.build/release/audio speak "Hello world" --engine cosyvoice --language english --output output.wav

# Clonagem de voz (baixa o codificador de locutor CAM++ no primeiro uso)
.build/release/audio speak "Hello world" --engine cosyvoice --voice-sample reference.wav --output cloned.wav

# Dialogo multi-locutor com clonagem de voz
.build/release/audio speak "[S1] Hello there! [S2] Hey, how are you?" \
    --engine cosyvoice --speakers s1=alice.wav,s2=bob.wav -o dialogue.wav

# Tags de emocao/estilo em linha
.build/release/audio speak "(excited) Wow, amazing! (sad) But I have to go..." \
    --engine cosyvoice -o emotion.wav

# Combinado: dialogo + emocoes + clonagem de voz
.build/release/audio speak "[S1] (happy) Great news! [S2] (surprised) Really?" \
    --engine cosyvoice --speakers s1=alice.wav,s2=bob.wav -o combined.wav

# Instrucao de estilo personalizada
.build/release/audio speak "Hello world" --engine cosyvoice --cosy-instruct "Speak cheerfully" -o cheerful.wav

# Sintese em streaming
.build/release/audio speak "Hello world" --engine cosyvoice --language english --stream --output output.wav
```

## Kokoro TTS — Texto para Fala Leve no Dispositivo (iOS + macOS)

### Sintese Basica

```swift
import KokoroTTS
import AudioCommon  // para WAVWriter

let tts = try await KokoroTTSModel.fromPretrained()
// Baixa ~170 MB na primeira execucao (modelos CoreML + embeddings de voz + dicionarios)

let audio = try tts.synthesize(text: "Hello world", voice: "af_heart")
// Saida: amostras float mono a 24kHz
try WAVWriter.write(samples: audio, sampleRate: 24000, to: outputURL)
```

54 vozes predefinidas em 10 idiomas. Nao-autorregressivo — pipeline de 3 estagios no CoreML, sem loop de amostragem. Executa no Neural Engine, liberando a GPU completamente.

### CLI Kokoro TTS

```bash
make build

# Sintese basica
.build/release/audio kokoro "Hello world" --voice af_heart --output hello.wav

# Escolher idioma
.build/release/audio kokoro "Bonjour le monde" --voice ff_siwis --language fr --output bonjour.wav

# Listar vozes disponiveis
.build/release/audio kokoro --list-voices
```

## Qwen3 Chat (LLM no Dispositivo)

```swift
import Qwen3Chat

let chat = try await Qwen3ChatModel.fromPretrained()
// Baixa ~318 MB na primeira execucao (modelo CoreML INT4 + tokenizer)

// Resposta unica
let response = try chat.generate("What is Swift?", systemPrompt: "Answer briefly.")
print(response)

// Tokens em streaming
let stream = chat.chatStream("Tell me a joke", systemPrompt: "Be funny.")
for try await token in stream {
    print(token, terminator: "")
}
```

Qwen3-0.6B quantizado em INT4 para CoreML. Executa no Neural Engine com ~2 tok/s no iPhone, ~15 tok/s em chips M-series. Suporta conversa multi-turno com cache KV, modo de raciocinio (tokens `<think>`), e amostragem configuravel (temperature, top-k, top-p, penalidade de repeticao).

## Deteccao de Atividade de Voz (VAD) — Detectar Fala em Audio

### VAD em Streaming (Silero)

Silero VAD v5 processa blocos de audio de 32ms com latencia sub-milissegundo — ideal para deteccao de fala em tempo real a partir de microfones ou streams.

```swift
import SpeechVAD

let vad = try await SileroVADModel.fromPretrained()
// Ou use CoreML (Neural Engine, menor consumo de energia):
// let vad = try await SileroVADModel.fromPretrained(engine: .coreml)

// Streaming: processar blocos de 512 amostras (32ms @ 16kHz)
let prob = vad.processChunk(samples)  // → 0.0...1.0
vad.resetState()  // chame entre diferentes streams de audio

// Ou detecte todos os segmentos de uma vez
let segments = vad.detectSpeech(audio: audioSamples, sampleRate: 16000)
for seg in segments {
    print("Speech: \(seg.startTime)s - \(seg.endTime)s")
}
```

### Streaming Orientado a Eventos

```swift
let processor = StreamingVADProcessor(model: vad)

// Alimente audio de qualquer comprimento — eventos sao emitidos conforme a fala e confirmada
let events = processor.process(samples: audioBuffer)
for event in events {
    switch event {
    case .speechStarted(let time):
        print("Speech started at \(time)s")
    case .speechEnded(let segment):
        print("Speech: \(segment.startTime)s - \(segment.endTime)s")
    }
}

// Finalize ao fim do stream
let final = processor.flush()
```

### CLI de VAD

```bash
make build

# Silero VAD em streaming (blocos de 32ms)
.build/release/audio vad-stream audio.wav

# Backend CoreML (Neural Engine)
.build/release/audio vad-stream audio.wav --engine coreml

# Com limiares personalizados
.build/release/audio vad-stream audio.wav --onset 0.6 --offset 0.4

# Saida JSON
.build/release/audio vad-stream audio.wav --json

# VAD em lote com pyannote (janelas deslizantes de 10s)
.build/release/audio vad audio.wav
```

## Diarizacao de Falantes — Quem Falou Quando

### Pipeline de Diarizacao

```swift
import SpeechVAD

let pipeline = try await DiarizationPipeline.fromPretrained()
// Ou use embeddings CoreML (Neural Engine, libera a GPU):
// let pipeline = try await DiarizationPipeline.fromPretrained(embeddingEngine: .coreml)

let result = pipeline.diarize(audio: samples, sampleRate: 16000)
for seg in result.segments {
    print("Speaker \(seg.speakerId): [\(seg.startTime)s - \(seg.endTime)s]")
}
print("\(result.numSpeakers) speakers detected")
```

### Embedding de Falante

```swift
let model = try await WeSpeakerModel.fromPretrained()
// Ou: let model = try await WeSpeakerModel.fromPretrained(engine: .coreml)
let embedding = model.embed(audio: samples, sampleRate: 16000)
// embedding: [Float] de comprimento 256, normalizado L2

// Comparar falantes
let similarity = WeSpeakerModel.cosineSimilarity(embeddingA, embeddingB)
```

### Extracao de Falante

Extraia apenas os segmentos de um falante especifico usando uma gravacao de referencia:

```swift
let pipeline = try await DiarizationPipeline.fromPretrained()
let targetEmb = pipeline.embeddingModel.embed(audio: enrollmentAudio, sampleRate: 16000)
let segments = pipeline.extractSpeaker(
    audio: meetingAudio, sampleRate: 16000,
    targetEmbedding: targetEmb
)
```

### Diarizacao Sortformer (Ponta-a-Ponta, CoreML)

NVIDIA Sortformer prediz atividade por quadro para ate 4 falantes diretamente — sem necessidade de embedding ou clusterizacao. Executa no Neural Engine.

```swift
let diarizer = try await SortformerDiarizer.fromPretrained()
let result = diarizer.diarize(audio: samples, sampleRate: 16000, config: .default)
for seg in result.segments {
    print("Speaker \(seg.speakerId): [\(seg.startTime)s - \(seg.endTime)s]")
}
```

### CLI de Diarizacao

```bash
make build

# Diarizacao Pyannote (padrao)
.build/release/audio diarize meeting.wav

# Diarizacao Sortformer (CoreML, Neural Engine)
.build/release/audio diarize meeting.wav --engine sortformer

# Embeddings CoreML (Neural Engine, apenas pyannote)
.build/release/audio diarize meeting.wav --embedding-engine coreml

# Saida JSON
.build/release/audio diarize meeting.wav --json

# Extrair um falante especifico (apenas pyannote)
.build/release/audio diarize meeting.wav --target-speaker enrollment.wav

# Embedding de falante
.build/release/audio embed-speaker enrollment.wav --json
.build/release/audio embed-speaker enrollment.wav --engine coreml
```

Veja [Speaker Diarization](docs/inference/speaker-diarization.md) para detalhes da arquitetura.

## Aprimoramento de Fala — Supressao de Ruido e Limpeza de Audio

### Supressao de Ruido

```swift
import SpeechEnhancement
import AudioCommon  // para WAVWriter

let enhancer = try await SpeechEnhancer.fromPretrained()
// Baixa ~4.3 MB na primeira execucao (modelo Core ML FP16 + dados auxiliares)

let cleanAudio = try enhancer.enhance(audio: noisyAudio, sampleRate: 48000)
try WAVWriter.write(samples: cleanAudio, sampleRate: 48000, to: outputURL)
```

### CLI de Reducao de Ruido

```bash
make build

# Remocao basica de ruido
.build/release/audio denoise noisy.wav

# Caminho de saida personalizado
.build/release/audio denoise noisy.wav --output clean.wav
```

Veja [Speech Enhancement](docs/inference/speech-enhancement.md) para detalhes da arquitetura.

## Pipelines — Compor Multiplos Modelos

Todos os modelos seguem protocolos compartilhados (`SpeechRecognitionModel`, `SpeechGenerationModel`, `SpeechEnhancementModel`, etc.) e podem ser compostos em pipelines:

### Reconhecimento de Fala com Ruido (DeepFilterNet + ASR)

```swift
import SpeechEnhancement
import Qwen3ASR

let enhancer = try await SpeechEnhancer.fromPretrained()
let asr = try await Qwen3ASRModel.fromPretrained()

// Aprimorar a 48kHz, depois transcrever a 16kHz
let clean = try enhancer.enhance(audio: noisyAudio, sampleRate: 48000)
let clean16k = AudioResampler.resample(clean, from: 48000, to: 16000)
let text = asr.transcribe(audio: clean16k, sampleRate: 16000)
```

### Retransmissao Voz-a-Voz (VAD + ASR + TTS)

```swift
import SpeechVAD
import Qwen3ASR
import Qwen3TTS

let vad = try await SileroVADModel.fromPretrained()
let asr = try await Qwen3ASRModel.fromPretrained()
let tts = try await Qwen3TTSModel.fromPretrained()

// Detectar segmentos de fala, transcrever, re-sintetizar
let segments = vad.detectSpeech(audio: audio, sampleRate: 16000)
for seg in segments {
    let chunk = Array(audio[Int(seg.startTime * 16000)..<Int(seg.endTime * 16000)])
    let text = asr.transcribe(audio: chunk, sampleRate: 16000)
    let speech = tts.synthesize(text: text, language: "english")
    // speech: amostras float mono a 24kHz
}
```

### Transcricao de Reuniao (Diarizacao + ASR)

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

Veja [Shared Protocols](docs/shared-protocols.md) para a referencia completa de protocolos.

## Servidor de API HTTP

Um servidor HTTP independente expoe todos os modelos via endpoints REST e WebSocket. Os modelos sao carregados sob demanda na primeira requisicao.

```bash
swift build -c release
.build/release/audio-server --port 8080

# Transcrever audio
curl -X POST http://localhost:8080/transcribe --data-binary @audio.wav -H "Content-Type: audio/wav"

# Texto para fala
curl -X POST http://localhost:8080/speak -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "engine": "cosyvoice"}' -o output.wav

# Fala-a-fala (PersonaPlex)
curl -X POST http://localhost:8080/respond --data-binary @question.wav -o response.wav

# Aprimoramento de fala
curl -X POST http://localhost:8080/enhance --data-binary @noisy.wav -o clean.wav

# Pre-carregar todos os modelos na inicializacao
.build/release/audio-server --preload --port 8080
```

### Streaming via WebSocket

#### API Realtime OpenAI (`/v1/realtime`)

O endpoint WebSocket principal implementa o protocolo da [API Realtime da OpenAI](https://platform.openai.com/docs/api-reference/realtime) — todas as mensagens sao JSON com um campo `type`, audio e PCM16 codificado em base64 a 24kHz mono.

**Eventos Cliente → Servidor:**

| Evento | Descricao |
|--------|-----------|
| `session.update` | Configurar motor, idioma, formato de audio |
| `input_audio_buffer.append` | Enviar bloco de audio PCM16 em base64 |
| `input_audio_buffer.commit` | Transcrever audio acumulado (ASR) |
| `input_audio_buffer.clear` | Limpar buffer de audio |
| `response.create` | Solicitar sintese TTS |

**Eventos Servidor → Cliente:**

| Evento | Descricao |
|--------|-----------|
| `session.created` | Sessao inicializada |
| `session.updated` | Configuracao confirmada |
| `input_audio_buffer.committed` | Audio enviado para transcricao |
| `conversation.item.input_audio_transcription.completed` | Resultado do ASR |
| `response.audio.delta` | Bloco de audio PCM16 em base64 (TTS) |
| `response.audio.done` | Streaming de audio concluido |
| `response.done` | Resposta completa com metadados |
| `error` | Erro com tipo e mensagem |

```javascript
const ws = new WebSocket('ws://localhost:8080/v1/realtime');

// ASR: enviar audio, receber transcricao
ws.send(JSON.stringify({ type: 'input_audio_buffer.append', audio: base64PCM16 }));
ws.send(JSON.stringify({ type: 'input_audio_buffer.commit' }));
// → recebe: conversation.item.input_audio_transcription.completed

// TTS: enviar texto, receber audio em streaming
ws.send(JSON.stringify({
  type: 'response.create',
  response: { modalities: ['audio', 'text'], instructions: 'Hello world' }
}));
// → recebe: response.audio.delta (blocos base64), response.audio.done, response.done
```

Um exemplo de cliente HTML esta em `Examples/websocket-client.html` — abra em um navegador enquanto o servidor estiver rodando.

O servidor e um modulo `AudioServer` separado e um executavel `audio-server` — ele nao adiciona Hummingbird/WebSocket ao CLI principal `audio`.

## Latencia (M2 Max, 64 GB)

### ASR

| Modelo | Backend | RTF | Audio de 10s processado em |
|--------|---------|-----|---------------------------|
| Qwen3-ASR-0.6B (4-bit) | MLX | ~0.06 | ~0.6s |
| Qwen3-ASR-0.6B (INT8) | CoreML + MLX | ~0.09 | ~0.9s |
| Qwen3-ASR-1.7B (8-bit) | MLX | ~0.11 | ~1.1s |
| Parakeet-TDT-0.6B (INT4) | CoreML (Neural Engine) | ~0.12 frio, ~0.03 quente | ~1.2s / ~0.3s |
| Whisper-large-v3 | whisper.cpp (Q5_0) | ~0.10 | ~1.0s |
| Whisper-small | whisper.cpp (Q5_0) | ~0.04 | ~0.4s |

### Alinhamento Forcado

| Modelo | Framework | Audio de 20s | RTF |
|--------|-----------|-------------|-----|
| Qwen3-ForcedAligner-0.6B (4-bit) | MLX Swift (debug) | ~365ms | ~0.018 |

> Passagem unica nao-autorregressiva — sem loop de amostragem. O encoder de audio domina (~328ms), decodificador de passagem unica e ~37ms. **55x mais rapido que tempo real.**

### TTS

| Modelo | Framework | Curto (1s) | Medio (3s) | Longo (6s) | Primeiro Pacote em Streaming |
|--------|-----------|-----------|------------|-----------|---------------------------|
| Qwen3-TTS-0.6B (4-bit) | MLX Swift (release) | 1.6s (RTF 1.2) | 2.3s (RTF 0.7) | 3.9s (RTF 0.7) | ~120ms (1 quadro) |
| Kokoro-82M | CoreML (Neural Engine) | ~1.4s (RTFx 0.7) | ~4.3s (RTFx 0.7) | ~8.6s (RTFx 0.7) | N/A (nao-autorregressivo) |
| Apple `AVSpeechSynthesizer` | AVFoundation | 0.08s | 0.08s | 0.17s (RTF 0.02) | N/A |

> Qwen3-TTS gera fala natural e expressiva com prosodia e emocao, executando **mais rapido que tempo real** (RTF < 1.0). A sintese em streaming entrega o primeiro bloco de audio em ~120ms. Kokoro-82M executa inteiramente no Neural Engine com um pipeline de 3 estagios (RTFx ~0.7), ideal para iOS. O TTS integrado da Apple e mais rapido, mas produz fala robotica e monotona.

### PersonaPlex (Fala-a-Fala)

| Modelo | Framework | ms/passo | RTF | Observacoes |
|--------|-----------|----------|-----|-------------|
| PersonaPlex-7B (8-bit) | MLX Swift (release) | ~112ms | ~1.4 | Recomendado — respostas coerentes, 30% mais rapido que 4-bit |
| PersonaPlex-7B (4-bit) | MLX Swift (release) | ~158ms | ~1.97 | Nao recomendado — qualidade de saida degradada |

> **Use 8-bit.** INT8 e mais rapido (112 ms/passo vs. 158 ms/passo) e produz respostas full-duplex coerentes. A quantizacao INT4 degrada a qualidade de geracao, produzindo fala incoerente. INT8 executa a ~112ms/passo no M2 Max.

### VAD e Embedding de Falante

| Modelo | Backend | Latencia por Chamada | RTF | Observacoes |
|--------|---------|---------------------|-----|-------------|
| Silero-VAD-v5 | MLX | ~2.1ms / bloco | 0.065 | GPU (Metal) |
| Silero-VAD-v5 | CoreML | ~0.27ms / bloco | 0.008 | Neural Engine, **7.7x mais rapido** |
| WeSpeaker ResNet34-LM | MLX | ~310ms / audio de 20s | 0.016 | GPU (Metal) |
| WeSpeaker ResNet34-LM | CoreML | ~430ms / audio de 20s | 0.021 | Neural Engine, libera a GPU |

> Silero VAD CoreML executa no Neural Engine a 7.7x a velocidade do MLX, sendo ideal para entrada continua de microfone. WeSpeaker MLX e mais rapido na GPU, mas CoreML libera a GPU para cargas de trabalho concorrentes (TTS, ASR). Ambos os backends produzem resultados equivalentes.

### Aprimoramento de Fala

| Modelo | Backend | Duracao | Latencia | RTF |
|--------|---------|---------|----------|-----|
| DeepFilterNet3 (FP16) | CoreML | 5s | 0.65s | 0.13 |
| DeepFilterNet3 (FP16) | CoreML | 10s | 1.2s | 0.12 |
| DeepFilterNet3 (FP16) | CoreML | 20s | 4.8s | 0.24 |

RTF = Fator de Tempo Real (menor e melhor, < 1.0 = mais rapido que tempo real). O custo da GRU escala ~O(n^2).

### MLX vs CoreML

Ambos os backends produzem resultados equivalentes. Escolha de acordo com sua carga de trabalho:

| | MLX | CoreML |
|---|---|---|
| **Hardware** | GPU (shaders Metal) | Neural Engine + CPU |
| **Melhor para** | Vazao maxima, cargas de modelo unico | Pipelines multi-modelo, tarefas em segundo plano |
| **Energia** | Maior utilizacao de GPU | Menor consumo, libera a GPU |
| **Latencia** | Mais rapido para modelos grandes (WeSpeaker) | Mais rapido para modelos pequenos (Silero VAD) |

**Inferencia em desktop**: MLX e o padrao — melhor desempenho de modelo unico em Apple Silicon. Mude para CoreML ao executar multiplos modelos simultaneamente (ex.: VAD + ASR + TTS) para evitar contencao de GPU, ou para cargas de trabalho sensiveis a bateria em laptops.

Modelos CoreML estao disponiveis para o encoder Qwen3-ASR, Silero VAD e WeSpeaker. Para Qwen3-ASR, use `--engine qwen3-coreml` (hibrido: encoder CoreML no ANE + decodificador de texto MLX na GPU). Para VAD/embeddings, passe `engine: .coreml` no momento da construcao — a API de inferencia e identica.

## Benchmarks de Precisao

### ASR — Taxa de Erro de Palavras ([detalhes](docs/benchmarks/asr-wer.md))

| Modelo | WER% (LibriSpeech test-clean) | RTF |
|--------|-------------------------------|-----|
| Qwen3-ASR 1.7B 8-bit | **2.35** | 0.090 |
| Qwen3-ASR 1.7B 4-bit | 2.57 | 0.045 |
| Parakeet TDT INT8 | 2.74 | 0.089 |
| Qwen3-ASR 0.6B 8-bit | 2.80 | 0.025 |

Qwen3-ASR 1.7B 8-bit supera Whisper Large v3 Turbo (2.5%) com tamanho comparavel. Multilingual: 10 idiomas avaliados no FLEURS.

### TTS — Inteligibilidade Round-Trip ([detalhes](docs/benchmarks/tts-roundtrip.md))

| Motor | WER% | RTF |
|-------|------|-----|
| CosyVoice3 | **3.25** | 0.59 |
| Qwen3-TTS 1.7B | 3.47 | 0.79 |
| Kokoro-82M | 3.90 | 0.17 |

### VAD — Deteccao de Fala ([detalhes](docs/benchmarks/vad-detection.md))

| Motor | F1% (FLEURS) | RTF |
|-------|-------------|-----|
| FireRedVAD | **99.12** | 0.007 |
| Silero CoreML | 95.13 | 0.022 |
| Pyannote MLX | 94.86 | 0.358 |

## Arquitetura

**Modelos:** [ASR Model](docs/models/asr-model.md), [TTS Model](docs/models/tts-model.md), [CosyVoice TTS](docs/models/cosyvoice-tts.md), [Kokoro TTS](docs/models/kokoro-tts.md), [Parakeet TDT](docs/models/parakeet-asr.md), [PersonaPlex](docs/models/personaplex.md), [FireRedVAD](docs/models/fireredvad.md)

**Inferencia:** [Qwen3-ASR](docs/inference/qwen3-asr-inference.md), [Parakeet TDT](docs/inference/parakeet-asr-inference.md), [TTS Inference](docs/inference/qwen3-tts-inference.md), [Forced Aligner](docs/inference/forced-aligner.md), [FireRedVAD](docs/inference/fireredvad.md), [Silero VAD](docs/inference/silero-vad.md), [Speaker Diarization](docs/inference/speaker-diarization.md), [Speech Enhancement](docs/inference/speech-enhancement.md), [Reprodução de Áudio](docs/audio/playback.md)

**Benchmarks:** [ASR WER](docs/benchmarks/asr-wer.md), [TTS Round-Trip](docs/benchmarks/tts-roundtrip.md), [VAD Detection](docs/benchmarks/vad-detection.md)

**Referencia:** [Shared Protocols](docs/shared-protocols.md)

## Configuracao de Cache

Os pesos dos modelos sao armazenados em cache localmente. Altere o local do cache com:

```bash
export QWEN3_CACHE_DIR=/path/to/cache
```

## Biblioteca Metal MLX

Se voce vir `Failed to load the default metallib` em tempo de execucao, a biblioteca de shaders Metal esta faltando. Execute `make build` (ou `./scripts/build_mlx_metallib.sh release` apos um `swift build` manual) para compila-la. Se o Metal Toolchain estiver faltando, instale-o primeiro:

```bash
xcodebuild -downloadComponent MetalToolchain
```

## Testes

Testes unitarios (configuracao, amostragem, pre-processamento de texto, correcao de timestamps) executam sem download de modelos:

```bash
swift test --filter "Qwen3TTSConfigTests|SamplingTests|CosyVoiceTTSConfigTests|CamPlusPlusMelExtractorTests|PersonaPlexTests|ForcedAlignerTests/testText|ForcedAlignerTests/testTimestamp|ForcedAlignerTests/testLIS|SileroVADTests/testSilero|SileroVADTests/testReflection|SileroVADTests/testProcess|SileroVADTests/testReset|SileroVADTests/testDetect|SileroVADTests/testStreaming|SileroVADTests/testVADEvent|KokoroTTSTests"
```

Testes de integracao requerem pesos dos modelos (baixados automaticamente na primeira execucao):

```bash
# Round-trip TTS: sintetizar texto, salvar WAV, transcrever de volta com ASR
swift test --filter TTSASRRoundTripTests

# Apenas ASR: transcrever audio de teste
swift test --filter Qwen3ASRIntegrationTests

# Alinhamento Forcado E2E: timestamps por palavra (~979 MB de download)
swift test --filter ForcedAlignerTests/testForcedAlignerE2E

# PersonaPlex E2E: pipeline fala-a-fala (~5.5 GB de download)
PERSONAPLEX_E2E=1 swift test --filter PersonaPlexE2ETests
```

> **Nota:** A biblioteca Metal MLX deve ser compilada antes de executar testes que usam operacoes MLX.
> Veja [Biblioteca Metal MLX](#biblioteca-metal-mlx) para instrucoes.

## Idiomas Suportados

| Modelo | Idiomas |
|--------|---------|
| Qwen3-ASR | 52 idiomas (CN, EN, Cantones, DE, FR, ES, JA, KO, RU, + 22 dialetos chineses, ...) |
| Parakeet TDT | 25 idiomas europeus (BG, CS, DA, DE, EL, EN, ES, ET, FI, FR, HR, HU, IT, LT, LV, MT, NL, PL, PT, RO, RU, SK, SL, SV, UK) |
| Qwen3-TTS | EN, CN, DE, JA, ES, FR, KO, RU, IT, PT (+ dialetos de Pequim/Sichuan via CustomVoice) |
| CosyVoice TTS | CN, EN, JA, KO, DE, ES, FR, IT, RU |
| Kokoro TTS | EN (US/UK), ES, FR, HI, IT, JA, PT, CN, KO, DE |
| PersonaPlex | EN |

## Comparacao

### Fala para Texto (ASR): speech-swift vs Alternativas

| | **speech-swift (Qwen3-ASR)** | **whisper.cpp** | **Apple SFSpeechRecognizer** | **Google Cloud Speech** |
|---|---|---|---|---|
| **Execucao** | No dispositivo (MLX/CoreML) | No dispositivo (CPU/GPU) | No dispositivo ou nuvem | Somente nuvem |
| **Idiomas** | 52 | 100+ | ~70 (no dispositivo: limitado) | 125+ |
| **RTF (audio de 10s, M2 Max)** | 0.06 (17x tempo real) | 0.10 (Whisper-large-v3) | N/A | N/A |
| **Streaming** | Nao (em lote) | Nao (em lote) | Sim | Sim |
| **Modelos personalizados** | Sim (trocar pesos do HuggingFace) | Sim (modelos GGML) | Nao | Nao |
| **API Swift** | Nativa async/await | C++ com ponte Swift | Nativa | REST/gRPC |
| **Privacidade** | Totalmente no dispositivo | Totalmente no dispositivo | Depende da configuracao | Dados enviados a nuvem |
| **Timestamps por palavra** | Sim (Forced Aligner) | Sim | Limitado | Sim |
| **Custo** | Gratuito (Apache 2.0) | Gratuito (MIT) | Gratuito (no dispositivo) | Pago por minuto |

### Texto para Fala (TTS): speech-swift vs Alternativas

| | **speech-swift (Qwen3-TTS)** | **speech-swift (Kokoro)** | **Apple AVSpeechSynthesizer** | **ElevenLabs / TTS na Nuvem** |
|---|---|---|---|---|
| **Qualidade** | Neural, expressiva | Neural, natural | Robotica, monotona | Neural, mais alta qualidade |
| **Execucao** | No dispositivo (MLX) | No dispositivo (CoreML) | No dispositivo | Somente nuvem |
| **Streaming** | Sim (~120ms primeiro bloco) | Nao (pipeline de 3 estagios) | Nao | Sim |
| **Clonagem de voz** | Sim | Nao | Nao | Sim |
| **Vozes** | 9 integradas + clonar qualquer | 54 vozes predefinidas | ~50 vozes do sistema | 1000+ |
| **Idiomas** | 10 | 10 | 60+ | 30+ |
| **Suporte iOS** | Somente macOS | iOS + macOS | iOS + macOS | Qualquer (API) |
| **Custo** | Gratuito (Apache 2.0) | Gratuito (Apache 2.0) | Gratuito | Pago por caractere |

### Quando Usar speech-swift

- **Apps com privacidade critica** — saude, juridico, empresarial onde o audio nao pode sair do dispositivo
- **Uso offline** — nao precisa de conexao com internet apos o download inicial do modelo
- **Sensivel a custos** — sem cobranca por minuto ou por caractere de API
- **Otimizado para Apple Silicon** — construido especificamente para GPU M-series (Metal) e Neural Engine
- **Pipeline completo** — combine ASR + TTS + VAD + diarizacao + aprimoramento em um unico pacote Swift

## Perguntas Frequentes

**O speech-swift funciona no iOS?**
Kokoro TTS, Qwen3.5-Chat (CoreML), Silero VAD, Parakeet ASR, DeepFilterNet3 e WeSpeaker funcionam no iOS 17+ via CoreML no Neural Engine. Modelos baseados em MLX (Qwen3-ASR, Qwen3-TTS, Qwen3.5-Chat MLX, PersonaPlex) requerem macOS 14+ em Apple Silicon.

**E necessaria conexao com a internet?**
Apenas para o download inicial do modelo a partir do HuggingFace (automatico, armazenado em cache em `~/Library/Caches/qwen3-speech/`). Depois disso, toda a inferencia executa totalmente offline sem acesso a rede.

**Como o speech-swift se compara ao Whisper?**
Qwen3-ASR-0.6B alcanca RTF 0.06 no M2 Max — 40% mais rapido que Whisper-large-v3 via whisper.cpp (RTF 0.10) — com precisao comparavel em 52 idiomas. speech-swift oferece uma API Swift nativa com async/await, enquanto whisper.cpp requer uma ponte C++.

**Posso usar em um app comercial?**
Sim. speech-swift esta licenciado sob Apache 2.0. Os pesos dos modelos subjacentes possuem suas proprias licencas (verifique a pagina do HuggingFace de cada modelo).

**Quais chips Apple Silicon sao suportados?**
Todos os chips da serie M: M1, M2, M3, M4 e suas variantes Pro/Max/Ultra. Requer macOS 14+ (Sonoma) ou iOS 17+.

**Quanta memoria e necessaria?**
De ~3 MB (Silero VAD) ate ~6.5 GB (PersonaPlex 7B). Kokoro TTS usa ~500 MB, Qwen3-ASR ~2.2 GB. Veja a tabela de [Requisitos de Memoria](#requisitos-de-memoria) para detalhes completos.

**Posso executar multiplos modelos simultaneamente?**
Sim. Use modelos CoreML no Neural Engine junto com modelos MLX na GPU para evitar contencao — por exemplo, Silero VAD (CoreML) + Qwen3-ASR (MLX) + Qwen3-TTS (MLX).

**Existe uma API REST?**
Sim. O binario `audio-server` expoe todos os modelos via endpoints HTTP REST e WebSocket, incluindo um WebSocket compativel com a API Realtime da OpenAI em `/v1/realtime`.

## Contribuindo

Contribuicoes sao bem-vindas! Seja uma correcao de bug, integracao de um novo modelo ou melhoria na documentacao — PRs sao apreciados.

**Para comecar:**
1. Faca um fork do repositorio e crie uma branch de feature
2. `make build` para compilar (requer Xcode + Metal Toolchain)
3. `make test` para executar a suite de testes
4. Abra um PR contra `main`

## Licenca

Apache 2.0
