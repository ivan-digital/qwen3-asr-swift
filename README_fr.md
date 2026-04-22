# Speech Swift

Modeles IA de parole pour Apple Silicon, propulses par MLX Swift et CoreML.

📖 Read in: [English](README.md) · [中文](README_zh.md) · [日本語](README_ja.md) · [한국어](README_ko.md) · [Español](README_es.md) · [Deutsch](README_de.md) · [Français](README_fr.md) · [हिन्दी](README_hi.md) · [Português](README_pt.md) · [Русский](README_ru.md)

Reconnaissance, synthese et comprehension vocale embarquees pour Mac et iOS. S'execute entierement en local sur Apple Silicon -- sans cloud, sans cle API, aucune donnee ne quitte l'appareil.

**[📚 Documentation complete →](https://soniqo.audio/fr)** · **[🤗 Modeles HuggingFace](https://huggingface.co/aufklarer)** · **[📝 Blog](https://blog.ivan.digital)**

- **[Qwen3-ASR](https://soniqo.audio/fr/guides/transcribe)** -- Reconnaissance vocale (reconnaissance automatique de la parole, 52 langues, MLX + CoreML)
- **[Parakeet TDT](https://soniqo.audio/fr/guides/parakeet)** -- Reconnaissance vocale via CoreML (Neural Engine, NVIDIA FastConformer + decodeur TDT, 25 langues)
- **[Omnilingual ASR](https://soniqo.audio/fr/guides/omnilingual)** -- Reconnaissance vocale (Meta wav2vec2 + CTC, **1 672 langues** reparties dans 32 ecritures, CoreML 300M + MLX 300M/1B/3B/7B)
- **[Dictee en streaming](https://soniqo.audio/fr/guides/dictate)** -- Dictee en temps reel avec resultats partiels et detection de fin d'enonce (Parakeet-EOU-120M)
- **[Nemotron Streaming](https://soniqo.audio/fr/guides/nemotron)** — ASR en streaming à faible latence avec ponctuation et majuscules natives (NVIDIA Nemotron-Speech-Streaming-0.6B, CoreML, anglais)
- **[Qwen3-ForcedAligner](https://soniqo.audio/fr/guides/align)** -- Alignement temporel au niveau du mot (audio + texte → horodatages)
- **[Qwen3-TTS](https://soniqo.audio/fr/guides/speak)** -- Synthese vocale (qualite maximale, streaming, locuteurs personnalises, 10 langues)
- **[CosyVoice TTS](https://soniqo.audio/fr/guides/cosyvoice)** -- TTS en streaming avec clonage vocal, dialogue multi-locuteurs, balises d'emotion (9 langues)
- **[Kokoro TTS](https://soniqo.audio/fr/guides/kokoro)** -- TTS embarque (82M, CoreML/Neural Engine, 54 voix, compatible iOS, 10 langues)
- **[Qwen3.5-Chat](https://soniqo.audio/fr/guides/chat)** -- Chat LLM embarque (0.8B, MLX INT4 + CoreML INT8, DeltaNet hybride, tokens en streaming)
- **[PersonaPlex](https://soniqo.audio/fr/guides/respond)** -- Parole-a-parole en full-duplex (7B, audio entrant → audio sortant, 18 preselections de voix)
- **[DeepFilterNet3](https://soniqo.audio/fr/guides/denoise)** -- Suppression de bruit en temps reel (2,1M parametres, 48 kHz)
- **[Mot de reveil](https://soniqo.audio/fr/guides/wake-word)** -- Detection de mots-cles sur appareil (KWS Zipformer 3M, CoreML, 26x temps reel, liste de mots-cles configurable)
- **[VAD](https://soniqo.audio/fr/guides/vad)** -- Detection d'activite vocale (Silero streaming, Pyannote hors ligne, FireRedVAD 100+ langues)
- **[Diarisation de locuteurs](https://soniqo.audio/fr/guides/diarize)** -- Qui a parle quand (pipeline Pyannote, Sortformer de bout en bout sur Neural Engine)
- **[Empreintes de locuteur](https://soniqo.audio/fr/guides/embed-speaker)** -- WeSpeaker ResNet34 (256 dim), CAM++ (192 dim)

Articles : [Qwen3-ASR](https://arxiv.org/abs/2601.21337) (Alibaba) · [Qwen3-TTS](https://arxiv.org/abs/2601.15621) (Alibaba) · [Omnilingual ASR](https://arxiv.org/abs/2511.09690) (Meta) · [Parakeet TDT](https://arxiv.org/abs/2304.06795) (NVIDIA) · [CosyVoice 3](https://arxiv.org/abs/2505.17589) (Alibaba) · [Kokoro](https://arxiv.org/abs/2301.01695) (StyleTTS 2) · [PersonaPlex](https://arxiv.org/abs/2602.06053) (NVIDIA) · [Mimi](https://arxiv.org/abs/2410.00037) (Kyutai) · [Sortformer](https://arxiv.org/abs/2409.06656) (NVIDIA)

## Actualites

- **19 avr. 2026** -- [MLX vs CoreML sur Apple Silicon -- guide pratique pour choisir le bon backend](https://blog.ivan.digital/mlx-vs-coreml-on-apple-silicon-a-practical-guide-to-picking-the-right-backend-and-why-you-should-f77ddea7b27a)
- **20 mars 2026** -- [Nous battons Whisper Large v3 avec un modele de 600M tournant entierement sur votre Mac](https://blog.ivan.digital/we-beat-whisper-large-v3-with-a-600m-model-running-entirely-on-your-mac-20e6ce191174)
- **26 fev. 2026** -- [Diarisation de locuteurs et detection d'activite vocale sur Apple Silicon -- Swift natif avec MLX](https://blog.ivan.digital/speaker-diarization-and-voice-activity-detection-on-apple-silicon-native-swift-with-mlx-92ea0c9aca0f)
- **23 fev. 2026** -- [NVIDIA PersonaPlex 7B sur Apple Silicon -- parole-a-parole full-duplex en Swift natif avec MLX](https://blog.ivan.digital/nvidia-personaplex-7b-on-apple-silicon-full-duplex-speech-to-speech-in-native-swift-with-mlx-0aa5276f2e23)
- **12 fev. 2026** -- [Qwen3-ASR Swift : ASR + TTS embarques pour Apple Silicon -- architecture et benchmarks](https://blog.ivan.digital/qwen3-asr-swift-on-device-asr-tts-for-apple-silicon-architecture-and-benchmarks-27cbf1e4463f)

## Demarrage rapide

Ajoutez le package a votre `Package.swift` :

```swift
.package(url: "https://github.com/soniqo/speech-swift", branch: "main")
```

N'importez que les modules dont vous avez besoin -- chaque modele est une bibliotheque SPM independante, vous ne payez donc pas pour ce que vous n'utilisez pas :

```swift
.product(name: "ParakeetStreamingASR", package: "speech-swift"),
.product(name: "SpeechUI",             package: "speech-swift"),  // vues SwiftUI optionnelles
```

**Transcrire un tampon audio en 3 lignes :**

```swift
import ParakeetStreamingASR

let model = try await ParakeetStreamingASRModel.fromPretrained()
let text = try model.transcribeAudio(audioSamples, sampleRate: 16000)
```

**Streaming en direct avec resultats partiels :**

```swift
for await partial in model.transcribeStream(audio: samples, sampleRate: 16000) {
    print(partial.isFinal ? "FINAL: \(partial.text)" : "... \(partial.text)")
}
```

**Vue de dictee SwiftUI en ~10 lignes :**

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

`SpeechUI` ne fournit que `TranscriptionView` (finaux + partiels) et `TranscriptionStore` (adaptateur ASR en streaming). Utilisez AVFoundation pour la visualisation et la lecture audio.

Produits SPM disponibles : `Qwen3ASR`, `Qwen3TTS`, `Qwen3TTSCoreML`, `ParakeetASR`, `ParakeetStreamingASR`, `NemotronStreamingASR`, `OmnilingualASR`, `KokoroTTS`, `CosyVoiceTTS`, `PersonaPlex`, `SpeechVAD`, `SpeechEnhancement`, `Qwen3Chat`, `SpeechCore`, `SpeechUI`, `AudioCommon`.

## Modeles

Vue compacte ci-dessous. **[Catalogue complet des modeles avec tailles, quantifications, URLs de telechargement et tableaux de memoire → soniqo.audio/architecture](https://soniqo.audio/fr/architecture)**.

| Modele | Tache | Backends | Tailles | Langues |
|-------|------|----------|-------|-----------|
| [Qwen3-ASR](https://soniqo.audio/fr/guides/transcribe) | Parole → Texte | MLX, CoreML (hybride) | 0.6B, 1.7B | 52 |
| [Parakeet TDT](https://soniqo.audio/fr/guides/parakeet) | Parole → Texte | CoreML (ANE) | 0.6B | 25 europeennes |
| [Parakeet EOU](https://soniqo.audio/fr/guides/dictate) | Parole → Texte (streaming) | CoreML (ANE) | 120M | 25 europeennes |
| [Nemotron Streaming](https://soniqo.audio/fr/guides/nemotron) | Voix → Texte (streaming, ponctué) | CoreML (ANE) | 0.6B | Anglais |
| [Omnilingual ASR](https://soniqo.audio/fr/guides/omnilingual) | Parole → Texte | CoreML (ANE), MLX | 300M / 1B / 3B / 7B | **[1 672](https://github.com/facebookresearch/omnilingual-asr/blob/main/src/omnilingual_asr/models/wav2vec2_llama/lang_ids.py)** |
| [Qwen3-ForcedAligner](https://soniqo.audio/fr/guides/align) | Audio + Texte → Horodatages | MLX, CoreML | 0.6B | Multi |
| [Qwen3-TTS](https://soniqo.audio/fr/guides/speak) | Texte → Parole | MLX, CoreML | 0.6B, 1.7B | 10 |
| [CosyVoice3](https://soniqo.audio/fr/guides/cosyvoice) | Texte → Parole | MLX | 0.5B | 9 |
| [Kokoro-82M](https://soniqo.audio/fr/guides/kokoro) | Texte → Parole | CoreML (ANE) | 82M | 10 |
| [Qwen3.5-Chat](https://soniqo.audio/fr/guides/chat) | Texte → Texte (LLM) | MLX, CoreML | 0.8B | Multi |
| [PersonaPlex](https://soniqo.audio/fr/guides/respond) | Parole → Parole | MLX | 7B | EN |
| [Silero VAD](https://soniqo.audio/fr/guides/vad) | Detection d'activite vocale | MLX, CoreML | 309K | Agnostique |
| [Pyannote](https://soniqo.audio/fr/guides/diarize) | VAD + Diarisation | MLX | 1.5M | Agnostique |
| [Sortformer](https://soniqo.audio/fr/guides/diarize) | Diarisation (E2E) | CoreML (ANE) | — | Agnostique |
| [DeepFilterNet3](https://soniqo.audio/fr/guides/denoise) | Amelioration de la parole | CoreML | 2.1M | Agnostique |
| [WeSpeaker](https://soniqo.audio/fr/guides/embed-speaker) | Empreinte de locuteur | MLX, CoreML | 6.6M | Agnostique |

## Installation

### Homebrew

Necessite un Homebrew ARM natif (`/opt/homebrew`). Homebrew Rosetta/x86_64 n'est pas supporte.

```bash
brew tap soniqo/speech https://github.com/soniqo/speech-swift
brew install speech
```

Ensuite :

```bash
audio transcribe recording.wav
audio speak "Hello world"
audio respond --input question.wav --transcript
audio-server --port 8080            # serveur HTTP / WebSocket local (OpenAI-compatible /v1/realtime)
```

**[Reference CLI complete →](https://soniqo.audio/fr/cli)**

### Swift Package Manager

```swift
dependencies: [
    .package(url: "https://github.com/soniqo/speech-swift", branch: "main")
]
```

N'importez que ce dont vous avez besoin -- chaque modele est sa propre cible SPM :

```swift
import Qwen3ASR             // Reconnaissance vocale (MLX)
import ParakeetASR          // Reconnaissance vocale (CoreML, batch)
import ParakeetStreamingASR // Dictee en streaming avec partiels + EOU
import NemotronStreamingASR // ASR streaming en anglais avec ponctuation native (0.6B)
import OmnilingualASR       // 1 672 langues (CoreML + MLX)
import Qwen3TTS             // Synthese vocale
import CosyVoiceTTS         // Synthese vocale avec clonage
import KokoroTTS            // Synthese vocale (compatible iOS)
import Qwen3Chat            // Chat LLM embarque
import PersonaPlex          // Parole-a-parole full-duplex
import SpeechVAD            // VAD + diarisation + empreintes
import SpeechEnhancement    // Suppression de bruit
import SpeechUI             // Composants SwiftUI pour transcriptions en streaming
import AudioCommon          // Protocoles et utilitaires partages
```

### Prerequis

- Swift 6+, Xcode 16+ (avec Metal Toolchain)
- macOS 15+ (Sequoia) ou iOS 18+, Apple Silicon (M1/M2/M3/M4)

Le minimum macOS 15 / iOS 18 vient de [MLState](https://developer.apple.com/documentation/coreml/mlstate) —— l'API d'état persistant ANE d'Apple —— que les pipelines CoreML (Qwen3-ASR, Qwen3-Chat, Qwen3-TTS) utilisent pour garder les caches KV résidents sur le Neural Engine entre les pas de token.

### Compiler depuis les sources

```bash
git clone https://github.com/soniqo/speech-swift
cd speech-swift
make build
```

`make build` compile le package Swift **et** la bibliotheque de shaders MLX Metal. La bibliotheque Metal est necessaire pour l'inference GPU -- sans elle, vous verrez `Failed to load the default metallib` a l'execution. `make debug` pour les builds de debug, `make test` pour la suite de tests.

**[Guide complet de compilation et installation →](https://soniqo.audio/fr/getting-started)**

## Applications de demonstration

- **[DictateDemo](Examples/DictateDemo/)** ([docs](https://soniqo.audio/fr/guides/dictate)) -- Dictee en streaming dans la barre de menus macOS avec partiels en direct, detection de fin d'enonce basee sur VAD et copie en un clic. S'execute comme agent en arriere-plan (Parakeet-EOU-120M + Silero VAD).
- **[iOSEchoDemo](Examples/iOSEchoDemo/)** -- Demo d'echo iOS (Parakeet ASR + Kokoro TTS). Appareil et simulateur.
- **[PersonaPlexDemo](Examples/PersonaPlexDemo/)** -- Assistant vocal conversationnel avec entree micro, VAD et contexte multi-tours. macOS. RTF ~0,94 sur M2 Max (plus rapide que le temps reel).
- **[SpeechDemo](Examples/SpeechDemo/)** -- Dictee et synthese TTS dans une interface a onglets. macOS.

Le README de chaque demo contient les instructions de compilation.

## Exemples de code

Les extraits ci-dessous montrent le chemin minimal pour chaque domaine. Chaque section renvoie vers un guide complet sur [soniqo.audio](https://soniqo.audio/fr) avec les options de configuration, plusieurs backends, les patrons de streaming et les recettes CLI.

### Reconnaissance vocale -- [guide complet →](https://soniqo.audio/fr/guides/transcribe)

```swift
import Qwen3ASR

let model = try await Qwen3ASRModel.fromPretrained()
let text = model.transcribe(audio: audioSamples, sampleRate: 16000)
```

Backends alternatifs : [Parakeet TDT](https://soniqo.audio/fr/guides/parakeet) (CoreML, 32× temps reel), [Omnilingual ASR](https://soniqo.audio/fr/guides/omnilingual) (1 672 langues, CoreML ou MLX), [Dictee en streaming](https://soniqo.audio/fr/guides/dictate) (partiels en direct).

### Alignement force -- [guide complet →](https://soniqo.audio/fr/guides/align)

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

### Synthese vocale -- [guide complet →](https://soniqo.audio/fr/guides/speak)

```swift
import Qwen3TTS
import AudioCommon

let model = try await Qwen3TTSModel.fromPretrained()
let audio = model.synthesize(text: "Hello world", language: "english")
try WAVWriter.write(samples: audio, sampleRate: 24000, to: outputURL)
```

Moteurs TTS alternatifs : [CosyVoice3](https://soniqo.audio/fr/guides/cosyvoice) (streaming + clonage + balises d'emotion), [Kokoro-82M](https://soniqo.audio/fr/guides/kokoro) (compatible iOS, 54 voix), [Clonage vocal](https://soniqo.audio/fr/guides/voice-cloning).

### Parole-a-parole -- [guide complet →](https://soniqo.audio/fr/guides/respond)

```swift
import PersonaPlex

let model = try await PersonaPlexModel.fromPretrained()
let responseAudio = model.respond(userAudio: userSamples)
// Sortie Float32 mono 24 kHz prete pour la lecture
```

### Chat LLM -- [guide complet →](https://soniqo.audio/fr/guides/chat)

```swift
import Qwen3Chat

let chat = try await Qwen35MLXChat.fromPretrained()
chat.chat(messages: [(.user, "Explain MLX in one sentence")]) { token, isFinal in
    print(token, terminator: "")
}
```

### Detection d'activite vocale -- [guide complet →](https://soniqo.audio/fr/guides/vad)

```swift
import SpeechVAD

let vad = try await SileroVADModel.fromPretrained()
let segments = vad.detectSpeech(audio: samples, sampleRate: 16000)
for s in segments { print("\(s.startTime)s → \(s.endTime)s") }
```

### Diarisation de locuteurs -- [guide complet →](https://soniqo.audio/fr/guides/diarize)

```swift
import SpeechVAD

let diarizer = try await DiarizationPipeline.fromPretrained()
let segments = diarizer.diarize(audio: samples, sampleRate: 16000)
for s in segments { print("Speaker \(s.speakerId): \(s.startTime)s - \(s.endTime)s") }
```

### Amelioration de la parole -- [guide complet →](https://soniqo.audio/fr/guides/denoise)

```swift
import SpeechEnhancement

let denoiser = try await DeepFilterNet3Model.fromPretrained()
let clean = try denoiser.enhance(audio: noisySamples, sampleRate: 48000)
```

### Voice Pipeline (ASR → LLM → TTS) -- [guide complet →](https://soniqo.audio/fr/api)

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

`VoicePipeline` est la machine a etats agent vocal temps reel (propulsee par [speech-core](https://github.com/soniqo/speech-core)) avec detection de tours basee sur VAD, gestion des interruptions et STT eager. Elle connecte n'importe quel `SpeechRecognitionModel` + `SpeechGenerationModel` + `StreamingVADProvider`.

### Serveur API HTTP

```bash
audio-server --port 8080
```

Expose chaque modele via des endpoints HTTP REST + WebSocket, y compris un WebSocket compatible OpenAI Realtime API sur `/v1/realtime`. Voir [`Sources/AudioServer/`](Sources/AudioServer/).

## Architecture

speech-swift est decoupe en une cible SPM par modele, de sorte que les consommateurs ne paient que ce qu'ils importent. L'infrastructure partagee reside dans `AudioCommon` (protocoles, I/O audio, telechargeur HuggingFace, `SentencePieceModel`) et `MLXCommon` (chargement de poids, aides `QuantizedLinear`, aide d'attention multi-tete `SDPA`).

**[Diagramme d'architecture complet avec backends, tableaux de memoire et carte des modules → soniqo.audio/architecture](https://soniqo.audio/fr/architecture)** · **[Reference d'API → soniqo.audio/api](https://soniqo.audio/fr/api)** · **[Benchmarks → soniqo.audio/benchmarks](https://soniqo.audio/fr/benchmarks)**

Docs locales (depot) :
- **Modeles :** [Qwen3-ASR](docs/models/asr-model.md) · [Qwen3-TTS](docs/models/tts-model.md) · [CosyVoice](docs/models/cosyvoice-tts.md) · [Kokoro](docs/models/kokoro-tts.md) · [Parakeet TDT](docs/models/parakeet-asr.md) · [Parakeet Streaming](docs/models/parakeet-streaming-asr.md) · [Nemotron Streaming](docs/models/nemotron-streaming.md) · [Omnilingual ASR](docs/models/omnilingual-asr.md) · [PersonaPlex](docs/models/personaplex.md) · [FireRedVAD](docs/models/fireredvad.md)
- **Inference :** [Qwen3-ASR](docs/inference/qwen3-asr-inference.md) · [Parakeet TDT](docs/inference/parakeet-asr-inference.md) · [Parakeet Streaming](docs/inference/parakeet-streaming-asr-inference.md) · [Nemotron Streaming](docs/inference/nemotron-streaming-inference.md) · [Omnilingual ASR](docs/inference/omnilingual-asr-inference.md) · [TTS](docs/inference/qwen3-tts-inference.md) · [Forced Aligner](docs/inference/forced-aligner.md) · [Silero VAD](docs/inference/silero-vad.md) · [Diarisation](docs/inference/speaker-diarization.md) · [Amelioration de la parole](docs/inference/speech-enhancement.md)
- **Reference :** [Protocoles partages](docs/shared-protocols.md)

## Configuration du cache

Les poids de modele sont telecharges depuis HuggingFace lors de la premiere utilisation et mis en cache dans `~/Library/Caches/qwen3-speech/`. Surchargez avec `QWEN3_CACHE_DIR` (CLI) ou `cacheDir:` (API Swift). Tous les points d'entree `fromPretrained()` acceptent `offlineMode: true` pour sauter le reseau lorsque les poids sont deja en cache.

Voir [`docs/inference/cache-and-offline.md`](docs/inference/cache-and-offline.md) pour les details complets incluant les chemins de container iOS sandboxes.

## Bibliotheque MLX Metal

Si vous voyez `Failed to load the default metallib` a l'execution, la bibliotheque de shaders Metal est manquante. Executez `make build` ou `./scripts/build_mlx_metallib.sh release` apres un `swift build` manuel. Si le Metal Toolchain est absent, installez-le d'abord :

```bash
xcodebuild -downloadComponent MetalToolchain
```

## Tests

```bash
make test                            # suite complete (unite + E2E avec telechargements de modeles)
swift test --skip E2E                # unite uniquement (CI-safe, sans telechargements)
swift test --filter Qwen3ASRTests    # module specifique
```

Les classes de test E2E utilisent le prefixe `E2E` pour que la CI puisse les filtrer avec `--skip E2E`. Voir [CLAUDE.md](CLAUDE.md#testing) pour la convention complete de tests.

## Contribuer

PR bienvenues -- corrections de bugs, nouvelles integrations de modeles, documentation. Fork, creez une branche de fonctionnalite, `make build && make test`, ouvrez une PR vers `main`.

## Licence

Apache 2.0
