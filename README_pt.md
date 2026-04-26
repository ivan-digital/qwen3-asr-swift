# Speech Swift

Modelos de IA para fala em Apple Silicon, com tecnologia MLX Swift e CoreML.

📖 Read in: [English](README.md) · [中文](README_zh.md) · [日本語](README_ja.md) · [한국어](README_ko.md) · [Español](README_es.md) · [Deutsch](README_de.md) · [Français](README_fr.md) · [हिन्दी](README_hi.md) · [Português](README_pt.md) · [Русский](README_ru.md)

Reconhecimento, sintese e compreensao de fala no dispositivo para Mac e iOS. Executa localmente no Apple Silicon — sem nuvem, sem chaves de API, nenhum dado sai do dispositivo.

**[📚 Documentacao completa →](https://soniqo.audio/pt)** · **[🤗 Modelos no HuggingFace](https://huggingface.co/aufklarer)** · **[📝 Blog](https://blog.ivan.digital)** · **[💬 Discord](https://discord.gg/TnCryqEMgu)**

- **[Qwen3-ASR](https://soniqo.audio/pt/guides/transcribe)** — Fala para texto (reconhecimento automatico de fala, 52 idiomas, MLX + CoreML)
- **[Parakeet TDT](https://soniqo.audio/pt/guides/parakeet)** — Fala para texto via CoreML (Neural Engine, NVIDIA FastConformer + decodificador TDT, 25 idiomas)
- **[Omnilingual ASR](https://soniqo.audio/pt/guides/omnilingual)** — Fala para texto (Meta wav2vec2 + CTC, **1.672 idiomas** em 32 escritas, CoreML 300M + MLX 300M/1B/3B/7B)
- **[Ditado em streaming](https://soniqo.audio/pt/guides/dictate)** — Ditado em tempo real com resultados parciais e deteccao de fim de enunciado (Parakeet-EOU-120M)
- **[Nemotron Streaming](https://soniqo.audio/pt/guides/nemotron)** — ASR de streaming de baixa latência com pontuação e capitalização nativas (NVIDIA Nemotron-Speech-Streaming-0.6B, CoreML, inglês)
- **[Qwen3-ForcedAligner](https://soniqo.audio/pt/guides/align)** — Alinhamento de timestamps por palavra (audio + texto → timestamps)
- **[Qwen3-TTS](https://soniqo.audio/pt/guides/speak)** — Sintese de texto para fala (mais alta qualidade, streaming, locutores personalizados, 10 idiomas)
- **[CosyVoice TTS](https://soniqo.audio/pt/guides/cosyvoice)** — TTS em streaming com clonagem de voz, dialogo multi-locutor, tags de emocao (9 idiomas)
- **[Kokoro TTS](https://soniqo.audio/pt/guides/kokoro)** — TTS no dispositivo (82M, CoreML/Neural Engine, 54 vozes, pronto para iOS, 10 idiomas)
- **[VibeVoice TTS](https://soniqo.audio/pt/guides/vibevoice)** — TTS de formato longo / multi-alto-falante (Microsoft VibeVoice Realtime-0.5B + 1.5B, MLX, sintese de podcast/audiolivro de ate 90 min, EN/ZH)
- **[Qwen3.5-Chat](https://soniqo.audio/pt/guides/chat)** — Chat LLM no dispositivo (0.8B, MLX INT4 + CoreML INT8, DeltaNet hibrido, tokens em streaming)
- **[MADLAD-400](https://soniqo.audio/pt/guides/translate)** — Tradução multidirecional entre 400+ idiomas (3B, MLX INT4 + INT8, T5 v1.1, Apache 2.0)
- **[PersonaPlex](https://soniqo.audio/pt/guides/respond)** — Fala-a-fala full-duplex (7B, audio de entrada → audio de saida, 18 presets de voz)
- **[DeepFilterNet3](https://soniqo.audio/pt/guides/denoise)** — Supressao de ruido em tempo real (2.1M parametros, 48 kHz)
- **[Separação de fontes](https://soniqo.audio/pt/guides/separate)** — Separação de fontes musicais com Open-Unmix (UMX-HQ / UMX-L, 4 stems: vocais/bateria/baixo/outros, 44,1 kHz estéreo)
- **[Palavra de ativacao](https://soniqo.audio/pt/guides/wake-word)** — Deteccao de palavras-chave no dispositivo (KWS Zipformer 3M, CoreML, 26x tempo real, lista de palavras-chave configuravel)
- **[VAD](https://soniqo.audio/pt/guides/vad)** — Deteccao de atividade de voz (Silero streaming, Pyannote offline, FireRedVAD 100+ idiomas)
- **[Diarizacao de falantes](https://soniqo.audio/pt/guides/diarize)** — Quem falou quando (pipeline Pyannote, Sortformer ponta-a-ponta no Neural Engine)
- **[Embeddings de falante](https://soniqo.audio/pt/guides/embed-speaker)** — WeSpeaker ResNet34 (256 dim), CAM++ (192 dim)

Papers: [Qwen3-ASR](https://arxiv.org/abs/2601.21337) (Alibaba) · [Qwen3-TTS](https://arxiv.org/abs/2601.15621) (Alibaba) · [Omnilingual ASR](https://arxiv.org/abs/2511.09690) (Meta) · [Parakeet TDT](https://arxiv.org/abs/2304.06795) (NVIDIA) · [CosyVoice 3](https://arxiv.org/abs/2505.17589) (Alibaba) · [Kokoro](https://arxiv.org/abs/2301.01695) (StyleTTS 2) · [PersonaPlex](https://arxiv.org/abs/2602.06053) (NVIDIA) · [Mimi](https://arxiv.org/abs/2410.00037) (Kyutai) · [Sortformer](https://arxiv.org/abs/2409.06656) (NVIDIA)

## Novidades

- **19 Abr 2026** — [MLX vs CoreML no Apple Silicon — guia prático para escolher o backend certo](https://blog.ivan.digital/mlx-vs-coreml-on-apple-silicon-a-practical-guide-to-picking-the-right-backend-and-why-you-should-f77ddea7b27a)
- **20 Mar 2026** — [Superamos o Whisper Large v3 com um modelo de 600M rodando inteiramente no seu Mac](https://blog.ivan.digital/we-beat-whisper-large-v3-with-a-600m-model-running-entirely-on-your-mac-20e6ce191174)
- **26 Fev 2026** — [Diarizacao de falantes e deteccao de atividade de voz em Apple Silicon — Swift nativo com MLX](https://blog.ivan.digital/speaker-diarization-and-voice-activity-detection-on-apple-silicon-native-swift-with-mlx-92ea0c9aca0f)
- **23 Fev 2026** — [NVIDIA PersonaPlex 7B em Apple Silicon — fala-a-fala full-duplex em Swift nativo com MLX](https://blog.ivan.digital/nvidia-personaplex-7b-on-apple-silicon-full-duplex-speech-to-speech-in-native-swift-with-mlx-0aa5276f2e23)
- **12 Fev 2026** — [Qwen3-ASR Swift: ASR + TTS no dispositivo para Apple Silicon — arquitetura e benchmarks](https://blog.ivan.digital/qwen3-asr-swift-on-device-asr-tts-for-apple-silicon-architecture-and-benchmarks-27cbf1e4463f)

## Inicio rapido

Adicione o pacote ao seu `Package.swift`:

```swift
.package(url: "https://github.com/soniqo/speech-swift", branch: "main")
```

Importe apenas os modulos que voce precisa — cada modelo e uma biblioteca SPM independente, entao voce nao paga pelo que nao usa:

```swift
.product(name: "ParakeetStreamingASR", package: "speech-swift"),
.product(name: "SpeechUI",             package: "speech-swift"),  // views SwiftUI opcionais
```

**Transcrever um buffer de audio em 3 linhas:**

```swift
import ParakeetStreamingASR

let model = try await ParakeetStreamingASRModel.fromPretrained()
let text = try model.transcribeAudio(audioSamples, sampleRate: 16000)
```

**Streaming ao vivo com resultados parciais:**

```swift
for await partial in model.transcribeStream(audio: samples, sampleRate: 16000) {
    print(partial.isFinal ? "FINAL: \(partial.text)" : "... \(partial.text)")
}
```

**View de ditado SwiftUI em ~10 linhas:**

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

`SpeechUI` inclui apenas `TranscriptionView` (finais + parciais) e `TranscriptionStore` (adaptador de ASR em streaming). Use AVFoundation para visualizacao e reproducao de audio.

Produtos SPM disponiveis: `Qwen3ASR`, `Qwen3TTS`, `Qwen3TTSCoreML`, `ParakeetASR`, `ParakeetStreamingASR`, `NemotronStreamingASR`, `OmnilingualASR`, `KokoroTTS`, `VibeVoiceTTS`, `CosyVoiceTTS`, `PersonaPlex`, `SpeechVAD`, `SpeechEnhancement`, `SourceSeparation`, `Qwen3Chat`, `SpeechCore`, `SpeechUI`, `AudioCommon`.

## Modelos

Vista compacta abaixo. **[Catalogo completo de modelos com tamanhos, quantizacoes, URLs de download e tabelas de memoria → soniqo.audio/architecture](https://soniqo.audio/pt/architecture)**.

| Modelo | Tarefa | Backends | Tamanhos | Idiomas |
|-------|------|----------|-------|-----------|
| [Qwen3-ASR](https://soniqo.audio/pt/guides/transcribe) | Fala → Texto | MLX, CoreML (hibrido) | 0.6B, 1.7B | 52 |
| [Parakeet TDT](https://soniqo.audio/pt/guides/parakeet) | Fala → Texto | CoreML (ANE) | 0.6B | 25 europeus |
| [Parakeet EOU](https://soniqo.audio/pt/guides/dictate) | Fala → Texto (streaming) | CoreML (ANE) | 120M | 25 europeus |
| [Nemotron Streaming](https://soniqo.audio/pt/guides/nemotron) | Fala → Texto (streaming, com pontuação) | CoreML (ANE) | 0.6B | Inglês |
| [Omnilingual ASR](https://soniqo.audio/pt/guides/omnilingual) | Fala → Texto | CoreML (ANE), MLX | 300M / 1B / 3B / 7B | **[1.672](https://github.com/facebookresearch/omnilingual-asr/blob/main/src/omnilingual_asr/models/wav2vec2_llama/lang_ids.py)** |
| [Qwen3-ForcedAligner](https://soniqo.audio/pt/guides/align) | Audio + Texto → Timestamps | MLX, CoreML | 0.6B | Multi |
| [Qwen3-TTS](https://soniqo.audio/pt/guides/speak) | Texto → Fala | MLX, CoreML | 0.6B, 1.7B | 10 |
| [CosyVoice3](https://soniqo.audio/pt/guides/cosyvoice) | Texto → Fala | MLX | 0.5B | 9 |
| [Kokoro-82M](https://soniqo.audio/pt/guides/kokoro) | Texto → Fala | CoreML (ANE) | 82M | 10 |
| [VibeVoice Realtime-0.5B](https://soniqo.audio/pt/guides/vibevoice) | Texto → Fala (formato longo, multi-alto-falante) | MLX | 0.5B | EN/ZH |
| [VibeVoice 1.5B](https://soniqo.audio/pt/guides/vibevoice) | Texto → Fala (podcast de ate 90 min) | MLX | 1.5B | EN/ZH |
| [Qwen3.5-Chat](https://soniqo.audio/pt/guides/chat) | Texto → Texto (LLM) | MLX, CoreML | 0.8B | Multi |
| [MADLAD-400](https://soniqo.audio/pt/guides/translate) | Texto → Texto (Tradução) | MLX | 3B | **400+** |
| [PersonaPlex](https://soniqo.audio/pt/guides/respond) | Fala → Fala | MLX | 7B | EN |
| [Silero VAD](https://soniqo.audio/pt/guides/vad) | Deteccao de atividade de voz | MLX, CoreML | 309K | Agnostico |
| [Pyannote](https://soniqo.audio/pt/guides/diarize) | VAD + Diarizacao | MLX | 1.5M | Agnostico |
| [Sortformer](https://soniqo.audio/pt/guides/diarize) | Diarizacao (E2E) | CoreML (ANE) | — | Agnostico |
| [DeepFilterNet3](https://soniqo.audio/pt/guides/denoise) | Aprimoramento de fala | CoreML | 2.1M | Agnostico |
| [Open-Unmix](https://soniqo.audio/pt/guides/separate) | Separação de fontes | MLX | 8.6M | Agnostic |
| [WeSpeaker](https://soniqo.audio/pt/guides/embed-speaker) | Embedding de falante | MLX, CoreML | 6.6M | Agnostico |

## Instalacao

### Homebrew

Requer Homebrew ARM nativo (`/opt/homebrew`). Homebrew Rosetta/x86_64 nao e suportado.

```bash
brew tap soniqo/speech https://github.com/soniqo/speech-swift
brew install speech
```

Depois:

```bash
audio transcribe recording.wav
audio speak "Hello world"
audio translate "Hello, how are you?" --to es
audio respond --input question.wav --transcript
audio-server --port 8080            # servidor HTTP / WebSocket local (OpenAI-compatible /v1/realtime)
```

**[Referencia completa do CLI →](https://soniqo.audio/pt/cli)**

### Swift Package Manager

```swift
dependencies: [
    .package(url: "https://github.com/soniqo/speech-swift", branch: "main")
]
```

Importe apenas o que voce precisa — cada modelo e o seu proprio target SPM:

```swift
import Qwen3ASR             // Reconhecimento de fala (MLX)
import ParakeetASR          // Reconhecimento de fala (CoreML, batch)
import ParakeetStreamingASR // Ditado em streaming com parciais + EOU
import NemotronStreamingASR // ASR streaming em inglês com pontuação nativa (0.6B)
import OmnilingualASR       // 1.672 idiomas (CoreML + MLX)
import Qwen3TTS             // Sintese de fala
import CosyVoiceTTS         // Sintese de fala com clonagem
import KokoroTTS            // Sintese de fala (pronto para iOS)
import VibeVoiceTTS         // TTS de formato longo / multi-alto-falante (EN/ZH)
import Qwen3Chat            // Chat LLM no dispositivo
import MADLADTranslation    // Tradução multidirecional entre 400+ idiomas
import PersonaPlex          // Fala-a-fala full-duplex
import SpeechVAD            // VAD + diarizacao + embeddings
import SpeechEnhancement    // Supressao de ruido
import SourceSeparation     // Separação de fontes musicais (Open-Unmix, 4 stems)
import SpeechUI             // Componentes SwiftUI para transcricoes em streaming
import AudioCommon          // Protocolos e utilitarios compartilhados
```

### Requisitos

- Swift 6+, Xcode 16+ (com Metal Toolchain)
- macOS 15+ (Sequoia) ou iOS 18+, Apple Silicon (M1/M2/M3/M4)

O mínimo de macOS 15 / iOS 18 vem do [MLState](https://developer.apple.com/documentation/coreml/mlstate) —— a API de estado persistente do ANE da Apple —— que os pipelines CoreML (Qwen3-ASR, Qwen3-Chat, Qwen3-TTS) usam para manter caches KV residentes no Neural Engine entre passos de token.

### Compilar a partir do codigo-fonte

```bash
git clone https://github.com/soniqo/speech-swift
cd speech-swift
make build
```

`make build` compila o pacote Swift **e** a biblioteca de shaders MLX Metal. A biblioteca Metal e necessaria para inferencia em GPU — sem ela voce vera `Failed to load the default metallib` em tempo de execucao. `make debug` para builds de debug, `make test` para a suite de testes.

**[Guia completo de build e instalacao →](https://soniqo.audio/pt/getting-started)**

## Aplicativos de demonstracao

- **[DictateDemo](Examples/DictateDemo/)** ([docs](https://soniqo.audio/pt/guides/dictate)) — Ditado em streaming na barra de menus do macOS com parciais ao vivo, deteccao de fim de enunciado baseada em VAD e copia com um clique. Roda como agent em segundo plano (Parakeet-EOU-120M + Silero VAD).
- **[iOSEchoDemo](Examples/iOSEchoDemo/)** — Demo de eco iOS (Parakeet ASR + Kokoro TTS). Dispositivo e simulador.
- **[PersonaPlexDemo](Examples/PersonaPlexDemo/)** — Assistente de voz conversacional com entrada de microfone, VAD e contexto multi-turno. macOS. RTF ~0.94 em M2 Max (mais rapido que tempo real).
- **[SpeechDemo](Examples/SpeechDemo/)** — Ditado e sintese TTS em uma interface com abas. macOS.

O README de cada demo tem instrucoes de build.

## Exemplos de codigo

Os snippets abaixo mostram o caminho minimo para cada dominio. Cada secao tem link para um guia completo em [soniqo.audio](https://soniqo.audio/pt) com opcoes de configuracao, multiplos backends, padroes de streaming e receitas de CLI.

### Fala para texto — [guia completo →](https://soniqo.audio/pt/guides/transcribe)

```swift
import Qwen3ASR

let model = try await Qwen3ASRModel.fromPretrained()
let text = model.transcribe(audio: audioSamples, sampleRate: 16000)
```

Backends alternativos: [Parakeet TDT](https://soniqo.audio/pt/guides/parakeet) (CoreML, 32× tempo real), [Omnilingual ASR](https://soniqo.audio/pt/guides/omnilingual) (1.672 idiomas, CoreML ou MLX), [Ditado em streaming](https://soniqo.audio/pt/guides/dictate) (parciais ao vivo).

### Alinhamento forcado — [guia completo →](https://soniqo.audio/pt/guides/align)

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

### Texto para fala — [guia completo →](https://soniqo.audio/pt/guides/speak)

```swift
import Qwen3TTS
import AudioCommon

let model = try await Qwen3TTSModel.fromPretrained()
let audio = model.synthesize(text: "Hello world", language: "english")
try WAVWriter.write(samples: audio, sampleRate: 24000, to: outputURL)
```

Engines TTS alternativas: [CosyVoice3](https://soniqo.audio/pt/guides/cosyvoice) (streaming + clonagem + tags de emocao), [Kokoro-82M](https://soniqo.audio/pt/guides/kokoro) (pronto para iOS, 54 vozes), [VibeVoice](https://soniqo.audio/pt/guides/vibevoice) (podcast de formato longo / multi-alto-falante, EN/ZH), [Clonagem de voz](https://soniqo.audio/pt/guides/voice-cloning).

### Fala para fala — [guia completo →](https://soniqo.audio/pt/guides/respond)

```swift
import PersonaPlex

let model = try await PersonaPlexModel.fromPretrained()
let responseAudio = model.respond(userAudio: userSamples)
// Saida mono Float32 a 24 kHz pronta para reproducao
```

### Chat LLM — [guia completo →](https://soniqo.audio/pt/guides/chat)

```swift
import Qwen3Chat

let chat = try await Qwen35MLXChat.fromPretrained()
chat.chat(messages: [(.user, "Explain MLX in one sentence")]) { token, isFinal in
    print(token, terminator: "")
}
```

### Tradução — [guia completo →](https://soniqo.audio/pt/guides/translate)

```swift
import MADLADTranslation

let translator = try await MADLADTranslator.fromPretrained()
let es = try translator.translate("Hello, how are you?", to: "es")
// → "Hola, ¿cómo estás?"
```

### Deteccao de atividade de voz — [guia completo →](https://soniqo.audio/pt/guides/vad)

```swift
import SpeechVAD

let vad = try await SileroVADModel.fromPretrained()
let segments = vad.detectSpeech(audio: samples, sampleRate: 16000)
for s in segments { print("\(s.startTime)s → \(s.endTime)s") }
```

### Diarizacao de falantes — [guia completo →](https://soniqo.audio/pt/guides/diarize)

```swift
import SpeechVAD

let diarizer = try await DiarizationPipeline.fromPretrained()
let segments = diarizer.diarize(audio: samples, sampleRate: 16000)
for s in segments { print("Speaker \(s.speakerId): \(s.startTime)s - \(s.endTime)s") }
```

### Aprimoramento de fala — [guia completo →](https://soniqo.audio/pt/guides/denoise)

```swift
import SpeechEnhancement

let denoiser = try await DeepFilterNet3Model.fromPretrained()
let clean = try denoiser.enhance(audio: noisySamples, sampleRate: 48000)
```

### Voice Pipeline (ASR → LLM → TTS) — [guia completo →](https://soniqo.audio/pt/api)

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

`VoicePipeline` e a maquina de estados de agent de voz em tempo real (movida por [speech-core](https://github.com/soniqo/speech-core)) com deteccao de turnos baseada em VAD, tratamento de interrupcoes e STT eager. Conecta qualquer `SpeechRecognitionModel` + `SpeechGenerationModel` + `StreamingVADProvider`.

### Servidor HTTP API

```bash
audio-server --port 8080
```

Expoe cada modelo via endpoints HTTP REST + WebSocket, incluindo um WebSocket compativel com OpenAI Realtime API em `/v1/realtime`. Veja [`Sources/AudioServer/`](Sources/AudioServer/).

## Arquitetura

speech-swift e dividido em um target SPM por modelo para que os consumidores paguem apenas pelo que importarem. A infraestrutura compartilhada fica em `AudioCommon` (protocolos, I/O de audio, downloader do HuggingFace, `SentencePieceModel`) e `MLXCommon` (carregamento de pesos, helpers `QuantizedLinear`, helper de atencao multi-head `SDPA`).

**[Diagrama completo de arquitetura com backends, tabelas de memoria e mapa de modulos → soniqo.audio/architecture](https://soniqo.audio/pt/architecture)** · **[Referencia de API → soniqo.audio/api](https://soniqo.audio/pt/api)** · **[Benchmarks → soniqo.audio/benchmarks](https://soniqo.audio/pt/benchmarks)**

Docs locais (repositorio):
- **Modelos:** [Qwen3-ASR](docs/models/asr-model.md) · [Qwen3-TTS](docs/models/tts-model.md) · [CosyVoice](docs/models/cosyvoice-tts.md) · [Kokoro](docs/models/kokoro-tts.md) · [VibeVoice](docs/models/vibevoice.md) · [Parakeet TDT](docs/models/parakeet-asr.md) · [Parakeet Streaming](docs/models/parakeet-streaming-asr.md) · [Nemotron Streaming](docs/models/nemotron-streaming.md) · [Omnilingual ASR](docs/models/omnilingual-asr.md) · [PersonaPlex](docs/models/personaplex.md) · [FireRedVAD](docs/models/fireredvad.md) · [Source Separation](docs/models/source-separation.md)
- **Inferencia:** [Qwen3-ASR](docs/inference/qwen3-asr-inference.md) · [Parakeet TDT](docs/inference/parakeet-asr-inference.md) · [Parakeet Streaming](docs/inference/parakeet-streaming-asr-inference.md) · [Nemotron Streaming](docs/inference/nemotron-streaming-inference.md) · [Omnilingual ASR](docs/inference/omnilingual-asr-inference.md) · [TTS](docs/inference/qwen3-tts-inference.md) · [VibeVoice](docs/inference/vibevoice-inference.md) · [Forced Aligner](docs/inference/forced-aligner.md) · [Silero VAD](docs/inference/silero-vad.md) · [Diarizacao](docs/inference/speaker-diarization.md) · [Aprimoramento de fala](docs/inference/speech-enhancement.md)
- **Referencia:** [Protocolos compartilhados](docs/shared-protocols.md)

## Configuracao de cache

Os pesos dos modelos sao baixados do HuggingFace no primeiro uso e armazenados em cache em `~/Library/Caches/qwen3-speech/`. Sobrescreva com `QWEN3_CACHE_DIR` (CLI) ou `cacheDir:` (API Swift). Todos os pontos de entrada `fromPretrained()` aceitam `offlineMode: true` para pular a rede quando os pesos ja estao em cache.

Veja [`docs/inference/cache-and-offline.md`](docs/inference/cache-and-offline.md) para detalhes completos, incluindo caminhos de container iOS sandboxed.

## Biblioteca MLX Metal

Se voce ver `Failed to load the default metallib` em tempo de execucao, a biblioteca de shaders Metal esta faltando. Execute `make build` ou `./scripts/build_mlx_metallib.sh release` apos um `swift build` manual. Se o Metal Toolchain estiver faltando, instale-o primeiro:

```bash
xcodebuild -downloadComponent MetalToolchain
```

## Testes

```bash
make test                            # suite completa (unidade + E2E com downloads de modelos)
swift test --skip E2E                # somente unidade (seguro para CI, sem downloads)
swift test --filter Qwen3ASRTests    # modulo especifico
```

Classes de teste E2E usam o prefixo `E2E` para que a CI possa filtra-las com `--skip E2E`. Veja [CLAUDE.md](CLAUDE.md#testing) para a convencao completa de testes.

## Contribuindo

PRs bem-vindos — correcoes de bugs, integracoes de novos modelos, documentacao. Fork, crie uma branch de feature, `make build && make test`, abra um PR contra `main`.

## Licenca

Apache 2.0
