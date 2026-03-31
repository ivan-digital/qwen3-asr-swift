# Speech Swift

Modeles IA de parole pour Apple Silicon, propulses par MLX Swift et CoreML.

📖 Read in: [English](README.md) · [中文](README_zh.md) · [日本語](README_ja.md) · [한국어](README_ko.md) · [Español](README_es.md) · [Deutsch](README_de.md) · [Français](README_fr.md) · [हिन्दी](README_hi.md) · [Português](README_pt.md) · [Русский](README_ru.md)

Reconnaissance, synthese et comprehension vocale embarquees pour Mac et iOS. S'execute entierement en local sur Apple Silicon — sans cloud, sans cle API, aucune donnee ne quitte l'appareil.

[Installation via Homebrew](#homebrew) ou ajout en tant que dependance Swift Package.

**[Documentation](https://soniqo.audio)** · **[Modeles HuggingFace](https://huggingface.co/aufklarer)** · **[Blog](https://blog.ivan.digital)**

- **Qwen3-ASR** -- Reconnaissance vocale (reconnaissance automatique de la parole, 52 langues)
- **Parakeet TDT** -- Reconnaissance vocale via CoreML (Neural Engine, NVIDIA FastConformer + decodeur TDT, 25 langues)
- **Qwen3-ForcedAligner** -- Alignement temporel au niveau du mot (audio + texte → horodatages)
- **Qwen3-TTS** -- Synthese vocale (qualite maximale, streaming, locuteurs personnalises, 10 langues)
- **CosyVoice TTS** -- Synthese vocale avec streaming, clonage vocal, dialogue multi-locuteurs et balises d'emotion (9 langues, DiT flow matching, encodeur de locuteur CAM++)
- **Kokoro TTS** -- Synthese vocale embarquee (82M parametres, CoreML/Neural Engine, 54 voix, compatible iOS, 10 langues)
- **Qwen3.5-Chat** -- Chat LLM embarque (0.8B, MLX + CoreML, INT4 + CoreML INT8, DeltaNet hybride, tokens en streaming)
- **PersonaPlex** -- Conversation parole-a-parole en full-duplex (7B, audio entrant → audio sortant, 18 preselections de voix)
- **DeepFilterNet3** -- Amelioration de la parole / suppression du bruit (2.1M parametres, temps reel 48kHz)
- **FireRedVAD** -- Detection d'activite vocale hors ligne (DFSMN, CoreML, 100+ langues, 97.6% F1)
- **Silero VAD** -- Detection d'activite vocale en streaming (blocs de 32ms, latence inferieure a la milliseconde)
- **Pyannote VAD** -- Detection d'activite vocale hors ligne (fenetres de 10s, chevauchement multi-locuteurs)
- **Diarisation de locuteurs** -- Qui a parle quand (segmentation Pyannote + chainage de locuteurs base sur l'activite, ou Sortformer de bout en bout sur Neural Engine)
- **Empreintes vocales** -- Verification et identification de locuteurs (WeSpeaker ResNet34, vecteurs 256 dimensions)

Articles : [Qwen3-ASR](https://arxiv.org/abs/2601.21337) (Alibaba), [Qwen3-TTS](https://arxiv.org/abs/2601.15621) (Alibaba), [Qwen3](https://arxiv.org/abs/2505.09388) (Alibaba), [Parakeet TDT](https://arxiv.org/abs/2304.06795) (NVIDIA), [CosyVoice 3](https://arxiv.org/abs/2505.17589) (Alibaba), [Kokoro](https://arxiv.org/abs/2301.01695) (StyleTTS 2), [PersonaPlex](https://arxiv.org/abs/2602.06053) (NVIDIA), [Mimi](https://arxiv.org/abs/2410.00037) (Kyutai), [Sortformer](https://arxiv.org/abs/2409.06656) (NVIDIA)

## Feuille de route

Consultez la [discussion Feuille de route](https://github.com/soniqo/speech-swift/discussions/81) pour voir ce qui est prevu -- commentaires et suggestions bienvenus !

## Actualites

- **20 mars 2026** -- [We Beat Whisper Large v3 with a 600M Model Running Entirely on Your Mac](https://blog.ivan.digital/we-beat-whisper-large-v3-with-a-600m-model-running-entirely-on-your-mac-20e6ce191174)
- **26 fev. 2026** -- [Speaker Diarization and Voice Activity Detection on Apple Silicon — Native Swift with MLX](https://blog.ivan.digital/speaker-diarization-and-voice-activity-detection-on-apple-silicon-native-swift-with-mlx-92ea0c9aca0f)
- **23 fev. 2026** -- [NVIDIA PersonaPlex 7B on Apple Silicon — Full-Duplex Speech-to-Speech in Native Swift with MLX](https://blog.ivan.digital/nvidia-personaplex-7b-on-apple-silicon-full-duplex-speech-to-speech-in-native-swift-with-mlx-0aa5276f2e23)
- **12 fev. 2026** -- [Qwen3-ASR Swift: On-Device ASR + TTS for Apple Silicon — Architecture and Benchmarks](https://blog.ivan.digital/qwen3-asr-swift-on-device-asr-tts-for-apple-silicon-architecture-and-benchmarks-27cbf1e4463f)

## Modeles

| Modele | Tache | Streaming | Langues | Tailles |
|--------|-------|-----------|---------|---------|
| Qwen3-ASR-0.6B | Parole → Texte | Non | 52 langues | [4-bit](https://huggingface.co/aufklarer/Qwen3-ASR-0.6B-MLX-4bit) 680 MB · [8-bit](https://huggingface.co/aufklarer/Qwen3-ASR-0.6B-MLX-8bit) 1.0 GB · [CoreML](https://huggingface.co/aufklarer/Qwen3-ASR-CoreML) 180 MB |
| Qwen3-ASR-1.7B | Parole → Texte | Non | 52 langues | [4-bit](https://huggingface.co/aufklarer/Qwen3-ASR-1.7B-MLX-4bit) 2.1 GB · [8-bit](https://huggingface.co/aufklarer/Qwen3-ASR-1.7B-MLX-8bit) 3.2 GB |
| Parakeet-TDT-0.6B | Parole → Texte | Non | 25 langues europeennes | [CoreML INT4](https://huggingface.co/aufklarer/Parakeet-TDT-v3-CoreML-INT4) 315 MB · [CoreML INT8](https://huggingface.co/aufklarer/Parakeet-TDT-v3-CoreML-INT8) 500 MB |
| Qwen3-ForcedAligner-0.6B | Audio + Texte → Horodatages | Non | Multi | [4-bit](https://huggingface.co/aufklarer/Qwen3-ForcedAligner-0.6B-4bit) 979 MB · [8-bit](https://huggingface.co/aufklarer/Qwen3-ForcedAligner-0.6B-8bit) 1.4 GB · [CoreML INT4](https://huggingface.co/aufklarer/Qwen3-ForcedAligner-0.6B-CoreML-INT4) 630 MB · [CoreML INT8](https://huggingface.co/aufklarer/Qwen3-ForcedAligner-0.6B-CoreML-INT8) 1.0 GB |
| Qwen3-TTS-0.6B Base | Texte → Parole | Oui (~120ms) | 10 langues | [4-bit](https://huggingface.co/aufklarer/Qwen3-TTS-12Hz-0.6B-Base-MLX-4bit) 1.7 GB · [8-bit](https://huggingface.co/aufklarer/Qwen3-TTS-12Hz-0.6B-Base-MLX-8bit) 2.4 GB |
| Qwen3-TTS-0.6B CustomVoice | Texte → Parole | Oui (~120ms) | 10 langues | [4-bit](https://huggingface.co/aufklarer/Qwen3-TTS-12Hz-0.6B-CustomVoice-MLX-4bit) 1.7 GB |
| Qwen3-TTS-1.7B Base | Texte → Parole | Oui (~120ms) | 10 langues | [4-bit](https://huggingface.co/aufklarer/Qwen3-TTS-12Hz-1.7B-Base-MLX-4bit) 3.2 GB · [8-bit](https://huggingface.co/aufklarer/Qwen3-TTS-12Hz-1.7B-Base-MLX-8bit) 4.8 GB |
| CosyVoice3-0.5B | Texte → Parole | Oui (~150ms) | 9 langues | [4-bit](https://huggingface.co/aufklarer/CosyVoice3-0.5B-MLX-4bit) 1.2 GB |
| Kokoro-82M | Texte → Parole | Non | 10 langues | [CoreML](https://huggingface.co/aufklarer/Kokoro-82M-CoreML) ~170 MB |
| Qwen3.5-0.8B Chat | Text → Text (LLM) | Yes (streaming) | Multi | [MLX INT4](https://huggingface.co/aufklarer/Qwen3.5-0.8B-Chat-MLX) 418 MB · [CoreML INT8](https://huggingface.co/aufklarer/Qwen3.5-0.8B-Chat-CoreML) 981 MB |
| PersonaPlex-7B | Parole → Parole | Oui (~2s par bloc) | EN | [4-bit](https://huggingface.co/aufklarer/PersonaPlex-7B-MLX-4bit) 4.9 GB · [8-bit](https://huggingface.co/aufklarer/PersonaPlex-7B-MLX-8bit) 9.1 GB |
| FireRedVAD | Detection d'activite vocale | Non (hors ligne) | 100+ langues | [CoreML](https://huggingface.co/aufklarer/FireRedVAD-CoreML) ~1.2 MB |
| Silero-VAD-v5 | Detection d'activite vocale | Oui (blocs de 32ms) | Independant de la langue | [MLX](https://huggingface.co/aufklarer/Silero-VAD-v5-MLX) · [CoreML](https://huggingface.co/aufklarer/Silero-VAD-v5-CoreML) ~1.2 MB |
| Pyannote-Segmentation-3.0 | VAD + Segmentation de locuteurs | Non (fenetres de 10s) | Independant de la langue | [MLX](https://huggingface.co/aufklarer/Pyannote-Segmentation-MLX) ~5.7 MB |
| DeepFilterNet3 | Amelioration de la parole | Oui (trames de 10ms) | Independant de la langue | [CoreML FP16](https://huggingface.co/aufklarer/DeepFilterNet3-CoreML) ~4.2 MB |
| WeSpeaker-ResNet34-LM | Empreinte vocale (256 dim.) | Non | Independant de la langue | [MLX](https://huggingface.co/aufklarer/WeSpeaker-ResNet34-LM-MLX) · [CoreML](https://huggingface.co/aufklarer/WeSpeaker-ResNet34-LM-CoreML) ~25 MB |
| CAM++ | Empreinte vocale (192 dim.) | Non | Independant de la langue | [CoreML](https://huggingface.co/aufklarer/CamPlusPlus-Speaker-CoreML) ~14 MB |
| Sortformer | Diarisation de locuteurs (de bout en bout) | Oui (par blocs) | Independant de la langue | [CoreML](https://huggingface.co/aufklarer/Sortformer-Diarization-CoreML) ~240 MB |

### Memoire requise

La memoire des poids correspond a la memoire GPU (MLX) ou ANE (CoreML) consommee par les parametres du modele. Le pic d'inference inclut les caches KV, les activations et les tenseurs intermediaires.

| Modele | Memoire des poids | Pic d'inference |
|--------|-------------------|-----------------|
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

### Quel TTS choisir

- **Qwen3-TTS** : Meilleure qualite, streaming (~120ms), 9 locuteurs integres, 10 langues, synthese par lots
- **CosyVoice TTS** : Streaming (~150ms), 9 langues, clonage vocal (encodeur de locuteur CAM++), dialogue multi-locuteurs (`[S1] ... [S2] ...`), balises d'emotion/style en ligne (`(happy)`, `(whispers)`), DiT flow matching + vocodeur HiFi-GAN
- **Kokoro TTS** : TTS leger pour iOS (82M parametres), CoreML/Neural Engine, 54 voix, 10 langues, non autoregressif (pipeline en 3 etapes)
- **PersonaPlex** : Conversation parole-a-parole en full-duplex (audio entrant → audio sortant), streaming (~2s par bloc), 18 preselections de voix, base sur l'architecture Moshi

## Installation

### Homebrew

Necessite Homebrew natif ARM (`/opt/homebrew`). Homebrew Rosetta/x86_64 n'est pas supporte.

```bash
brew tap soniqo/speech https://github.com/soniqo/speech-swift
brew install speech
```

Utilisation :

```bash
audio transcribe recording.wav
audio speak "Hello world"
audio speak "Hallo Welt" --engine cosyvoice --language german
audio respond --input question.wav --transcript
```

> Pour une conversation vocale interactive avec entree micro, voir **[PersonaPlexDemo](Examples/PersonaPlexDemo/)**.

### Swift Package Manager

Ajoutez a votre `Package.swift` :

```swift
dependencies: [
    .package(url: "https://github.com/soniqo/speech-swift", branch: "main")
]
```

Importez le module dont vous avez besoin :

```swift
import Qwen3ASR      // Reconnaissance vocale (MLX)
import ParakeetASR   // Reconnaissance vocale (CoreML)
import Qwen3TTS      // Synthese vocale (Qwen3)
import CosyVoiceTTS  // Synthese vocale (streaming)
import KokoroTTS     // Synthese vocale (CoreML, compatible iOS)
import Qwen3Chat     // Chat LLM embarque (CoreML)
import PersonaPlex   // Parole-a-parole (full-duplex)
import SpeechVAD          // Detection d'activite vocale (pyannote + Silero)
import SpeechEnhancement  // Suppression du bruit (DeepFilterNet3)
import AudioCommon        // Utilitaires partages
```

### Pre-requis

- Swift 5.9+
- macOS 14+ ou iOS 17+
- Apple Silicon (M1/M2/M3/M4)
- Xcode 15+ (avec Metal Toolchain -- executez `xcodebuild -downloadComponent MetalToolchain` si absent)

### Compilation depuis les sources

```bash
git clone https://github.com/soniqo/speech-swift
cd speech-swift
make build
```

Ceci compile le package Swift **et** la bibliotheque Metal MLX en une seule etape. La bibliotheque Metal (`mlx.metallib`) est necessaire pour l'inference GPU -- sans elle, vous obtiendrez `Failed to load the default metallib` a l'execution.

Pour une compilation debug : `make debug`. Pour lancer les tests unitaires : `make test`.

## Essayez l'assistant vocal

**[PersonaPlexDemo](Examples/PersonaPlexDemo/)** est un assistant vocal macOS pret a l'emploi -- appuyez pour parler, obtenez des reponses vocales en temps reel. Utilise l'entree microphone avec Silero VAD pour la detection automatique de la parole, Qwen3-ASR pour la transcription et PersonaPlex 7B pour la generation parole-a-parole. Conversation multi-tours avec 18 preselections de voix et affichage de la transcription du monologue interieur.

```bash
make build  # depuis la racine du depot -- compile tout, y compris la metallib MLX
cd Examples/PersonaPlexDemo
# Voir Examples/PersonaPlexDemo/README.md pour les instructions de creation du bundle .app
```

> RTF ~0.94 sur M2 Max (plus rapide que le temps reel). Les modeles se telechargent automatiquement au premier lancement (~5.5 Go PersonaPlex + ~400 Mo ASR).

## Applications de demonstration

- **[PersonaPlexDemo](Examples/PersonaPlexDemo/)** -- Assistant vocal conversationnel (entree micro, VAD, multi-tours). Voir ci-dessus.
- **[SpeechDemo](Examples/SpeechDemo/)** -- Dictee (Parakeet TDT / Qwen3-ASR avec detection automatique de la langue) et synthese vocale (Qwen3-TTS) dans une interface a onglets.

Compilez et lancez en tant que bundle macOS `.app` -- consultez le README de chaque demo pour les instructions.

## Reconnaissance vocale (ASR) -- Transcrire de l'audio en Swift

### Transcription de base

```swift
import Qwen3ASR

// Par defaut : modele 0.6B
let model = try await Qwen3ASRModel.fromPretrained()

// Ou utilisez le modele 1.7B plus grand pour une meilleure precision
let model = try await Qwen3ASRModel.fromPretrained(
    modelId: "aufklarer/Qwen3-ASR-1.7B-MLX-8bit"
)

// L'audio peut etre a n'importe quelle frequence d'echantillonnage -- reechantillonnage automatique a 16kHz en interne
let transcription = model.transcribe(audio: audioSamples, sampleRate: 16000)
print(transcription)
```

### Encodeur CoreML (Neural Engine)

Mode hybride : encodeur CoreML sur le Neural Engine + decodeur texte MLX sur le GPU. Consommation reduite, libere le GPU pour la passe d'encodage.

```swift
import Qwen3ASR

let encoder = try await CoreMLASREncoder.fromPretrained()
let model = try await Qwen3ASRModel.fromPretrained()
let text = try model.transcribe(audio: audioSamples, sampleRate: 16000, coremlEncoder: encoder)
```

Variantes INT8 (180 Mo, par defaut) et INT4 (90 Mo) disponibles. INT8 recommande (similarite cosinus > 0.999 par rapport au FP32).

### Parakeet TDT (CoreML)

```swift
import ParakeetASR

let model = try await ParakeetASRModel.fromPretrained()
let transcription = model.transcribe(audio: audioSamples, sampleRate: 16000)
```

S'execute sur le Neural Engine via CoreML -- libere le GPU pour d'autres taches simultanees. 25 langues europeennes, ~315 Mo.

### CLI ASR

```bash
make build  # ou : swift build -c release && ./scripts/build_mlx_metallib.sh release

# Par defaut (Qwen3-ASR 0.6B, MLX)
.build/release/audio transcribe audio.wav

# Utiliser le modele 1.7B
.build/release/audio transcribe audio.wav --model 1.7B

# Encodeur CoreML (Neural Engine + decodeur MLX)
.build/release/audio transcribe --engine qwen3-coreml audio.wav

# Parakeet TDT (CoreML, Neural Engine)
.build/release/audio transcribe --engine parakeet audio.wav
```

## Alignement force

### Horodatages au niveau du mot

```swift
import Qwen3ASR

let aligner = try await Qwen3ForcedAligner.fromPretrained()
// Telecharge ~979 Mo au premier lancement

let aligned = aligner.align(
    audio: audioSamples,
    text: "Can you guarantee that the replacement part will be shipped tomorrow?",
    sampleRate: 24000
)

for word in aligned {
    print("[\(String(format: "%.2f", word.startTime))s - \(String(format: "%.2f", word.endTime))s] \(word.text)")
}
```

### CLI Alignement force

```bash
swift build -c release

# Aligner avec un texte fourni
.build/release/audio align audio.wav --text "Hello world"

# Transcrire d'abord, puis aligner
.build/release/audio align audio.wav
```

Sortie :
```
[0.12s - 0.45s] Can
[0.45s - 0.72s] you
[0.72s - 1.20s] guarantee
...
```

Non autoregressif -- pipeline en 3 etapes, pas de boucle d'echantillonnage. Voir [Forced Aligner](docs/inference/forced-aligner.md) pour les details d'architecture.

## Synthese vocale (TTS) -- Generer de la parole en Swift

### Synthese de base

```swift
import Qwen3TTS
import AudioCommon  // pour WAVWriter

let model = try await Qwen3TTSModel.fromPretrained()
// Telecharge ~1.7 Go au premier lancement (poids du modele + codec)
let audio = model.synthesize(text: "Hello world", language: "english")
// Sortie : echantillons float mono 24kHz
try WAVWriter.write(samples: audio, sampleRate: 24000, to: outputURL)
```

### CLI TTS

```bash
make build
.build/release/audio speak "Hello world" --output output.wav --language english
```

### Voix personnalisee / Selection de locuteur

La variante **CustomVoice** prend en charge 9 voix de locuteurs integrees et des instructions en langage naturel pour controler le ton et le style. Chargez-la en passant l'identifiant du modele CustomVoice :

```swift
import Qwen3TTS

// Charger le modele CustomVoice (telecharge ~1.7 Go au premier lancement)
let model = try await Qwen3TTSModel.fromPretrained(
    modelId: TTSModelVariant.customVoice.rawValue
)

// Synthetiser avec un locuteur specifique
let audio = model.synthesize(text: "Hello world", language: "english", speaker: "vivian")

// Lister les locuteurs disponibles
print(model.availableSpeakers)  // ["aiden", "dylan", "eric", ...]
```

CLI :

```bash
# Utiliser le modele CustomVoice avec un locuteur
.build/release/audio speak "Hello world" --model customVoice --speaker vivian --output vivian.wav

# Lister les locuteurs disponibles
.build/release/audio speak --model customVoice --list-speakers
```

### Clonage vocal (modele Base)

Clonez la voix d'un locuteur a partir d'un fichier audio de reference. Deux modes :

**Mode ICL** (recommande) -- encode l'audio de reference en tokens codec avec la transcription. Meilleure qualite, EOS fiable :

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

**Mode X-vector** -- empreinte vocale uniquement, pas de transcription necessaire mais qualite moindre :

```swift
let audio = model.synthesizeWithVoiceClone(
    text: "Hello world",
    referenceAudio: refAudio,
    referenceSampleRate: 24000,
    language: "english"
)
```

CLI :

```bash
.build/release/audio speak "Hello world" --voice-sample reference.wav --output cloned.wav
```

### Instructions de ton / style (CustomVoice uniquement)

Le modele CustomVoice accepte un parametre `instruct` en langage naturel pour controler le style de parole, le ton, l'emotion et le rythme. L'instruction est ajoutee en prefixe a l'entree du modele au format ChatML.

```swift
// Ton joyeux
let audio = model.synthesize(
    text: "Welcome to our store!",
    language: "english",
    speaker: "ryan",
    instruct: "Speak in a cheerful, upbeat tone"
)

// Lent et serieux
let audio = model.synthesize(
    text: "We regret to inform you...",
    language: "english",
    speaker: "aiden",
    instruct: "Read this slowly and solemnly"
)

// Chuchotement
let audio = model.synthesize(
    text: "Can you keep a secret?",
    language: "english",
    speaker: "vivian",
    instruct: "Whisper this softly"
)
```

CLI :

```bash
# Avec instruction de style
.build/release/audio speak "Good morning!" --model customVoice --speaker ryan \
    --instruct "Speak in a cheerful, upbeat tone" --output cheerful.wav

# L'instruction par defaut ("Speak naturally.") est appliquee automatiquement avec CustomVoice
.build/release/audio speak "Hello world" --model customVoice --speaker ryan --output natural.wav
```

Lorsqu'aucun `--instruct` n'est fourni avec le modele CustomVoice, `"Speak naturally."` est applique automatiquement pour eviter une sortie decousue. Le modele Base ne prend pas en charge instruct.

### Synthese par lots

Synthetisez plusieurs textes en une seule passe groupee pour un debit plus eleve :

```swift
let texts = ["Good morning everyone.", "The weather is nice today.", "Please open the window."]
let audioList = model.synthesizeBatch(texts: texts, language: "english", maxBatchSize: 4)
// audioList[i] contient les echantillons float mono 24kHz pour texts[i]
for (i, audio) in audioList.enumerated() {
    try WAVWriter.write(samples: audio, sampleRate: 24000, to: URL(fileURLWithPath: "output_\(i).wav"))
}
```

#### CLI par lots

```bash
# Creer un fichier avec un texte par ligne
echo "Hello world.\nGoodbye world." > texts.txt
.build/release/audio speak --batch-file texts.txt --output output.wav --batch-size 4
# Produit output_0.wav, output_1.wav, ...
```

> Le mode par lots amortit le chargement des poids du modele sur tous les elements. Attendez une amelioration du debit d'environ 1.5-2.5x pour B=4 sur Apple Silicon. Meilleurs resultats lorsque les textes produisent des audios de longueur similaire.

### Options d'echantillonnage

```swift
let config = SamplingConfig(temperature: 0.9, topK: 50, repetitionPenalty: 1.05)
let audio = model.synthesize(text: "Hello", language: "english", sampling: config)
```

### Synthese en streaming

Emettez des blocs audio de maniere incrementale pour une faible latence du premier paquet :

```swift
let stream = model.synthesizeStream(
    text: "Hello, this is streaming synthesis.",
    language: "english",
    streaming: .lowLatency  // ~120ms jusqu'au premier bloc audio
)

for try await chunk in stream {
    // chunk.samples: [Float] PCM @ 24kHz
    // chunk.isFinal: true sur le dernier bloc
    playAudio(chunk.samples)
}
```

CLI :

```bash
# Streaming par defaut (premier bloc de 3 trames, ~225ms de latence)
.build/release/audio speak "Hello world" --stream

# Faible latence (premier bloc de 1 trame, ~120ms de latence)
.build/release/audio speak "Hello world" --stream --first-chunk-frames 1
```

## Parole-a-parole -- Conversation vocale full-duplex

> Pour un assistant vocal interactif avec entree micro, voir **[PersonaPlexDemo](Examples/PersonaPlexDemo/)** -- appuyez pour parler, conversation multi-tours avec detection automatique de la parole.

### Parole-a-parole

```swift
import PersonaPlex
import AudioCommon  // pour WAVWriter, AudioFileLoader

let model = try await PersonaPlexModel.fromPretrained()
// Telecharge ~5.5 Go au premier lancement (temporal 4-bit + depformer + codec Mimi + preselections de voix)

let audio = try AudioFileLoader.load(url: inputURL, targetSampleRate: 24000)
let (response, textTokens) = model.respond(userAudio: audio, voice: .NATM0)
// response : echantillons float mono 24kHz
// textTokens : monologue interieur du modele (IDs de tokens SentencePiece)
try WAVWriter.write(samples: response.audio, sampleRate: 24000, to: outputURL)
```

### Monologue interieur (sortie texte)

PersonaPlex genere des tokens texte en parallele de l'audio -- le raisonnement interne du modele. Decodez-les avec le decodeur SentencePiece integre :

```swift
let decoder = try SentencePieceDecoder(modelPath: "tokenizer_spm_32k_3.model")
let transcript = decoder.decode(textTokens)
print(transcript)  // ex. : "Sure, I can help you with that..."
```

### Streaming parole-a-parole

```swift
// Recevoir les blocs audio au fur et a mesure de leur generation (~2s par bloc)
let stream = model.respondStream(userAudio: audio, voice: .NATM0)
for try await chunk in stream {
    playAudio(chunk.samples)  // lecture immediate, mono 24kHz
    // chunk.textTokens contient le texte de ce bloc ; le dernier bloc contient tous les tokens
    if chunk.isFinal { break }
}
```

### Selection de voix

18 preselections de voix disponibles :
- **Femme naturelle** : NATF0, NATF1, NATF2, NATF3
- **Homme naturel** : NATM0, NATM1, NATM2, NATM3
- **Femme variee** : VARF0, VARF1, VARF2, VARF3, VARF4
- **Homme varie** : VARM0, VARM1, VARM2, VARM3, VARM4

### Prompts systeme

Le prompt systeme oriente le comportement conversationnel du modele. Vous pouvez passer n'importe quel prompt personnalise sous forme de chaine de caracteres :

```swift
// Prompt systeme personnalise (tokenise automatiquement)
let response = model.respond(
    userAudio: audio,
    voice: .NATM0,
    systemPrompt: "You enjoy having a good conversation."
)

// Ou utiliser un preset
let response = model.respond(
    userAudio: audio,
    voice: .NATM0,
    systemPromptTokens: SystemPromptPreset.customerService.tokens
)
```

Presets disponibles : `focused` (par defaut), `assistant`, `customerService`, `teacher`.

### CLI PersonaPlex

```bash
make build

# Parole-a-parole de base
.build/release/audio respond --input question.wav --output response.wav

# Avec transcription (decode le texte du monologue interieur)
.build/release/audio respond --input question.wav --transcript

# Sortie JSON (chemin audio, transcription, metriques de latence)
.build/release/audio respond --input question.wav --json

# Texte de prompt systeme personnalise
.build/release/audio respond --input question.wav --system-prompt-text "You enjoy having a good conversation."

# Choisir une voix et un preset de prompt systeme
.build/release/audio respond --input question.wav --voice NATF1 --system-prompt focused

# Ajuster les parametres d'echantillonnage
.build/release/audio respond --input question.wav --audio-temp 0.6 --repetition-penalty 1.5

# Activer l'arret anticipe par entropie textuelle (arrete si le texte s'effondre)
.build/release/audio respond --input question.wav --entropy-threshold 1.0 --entropy-window 5

# Lister les voix et prompts disponibles
.build/release/audio respond --list-voices
.build/release/audio respond --list-prompts
```

## CosyVoice TTS -- Synthese vocale en streaming avec clonage vocal

### Synthese de base

```swift
import CosyVoiceTTS
import AudioCommon  // pour WAVWriter

let model = try await CosyVoiceTTSModel.fromPretrained()
// Telecharge ~1.9 Go au premier lancement (poids LLM + DiT + HiFi-GAN)

let audio = model.synthesize(text: "Hello, how are you today?", language: "english")
// Sortie : echantillons float mono 24kHz
try WAVWriter.write(samples: audio, sampleRate: 24000, to: outputURL)
```

### Synthese en streaming

```swift
// Streaming : recevez les blocs audio au fur et a mesure de leur generation (~150ms jusqu'au premier bloc)
for try await chunk in model.synthesizeStream(text: "Hello, how are you today?", language: "english") {
    // chunk.audio: [Float], chunk.sampleRate: Int
    playAudio(chunk.audio)  // lecture immediate
}
```

### Clonage vocal (CosyVoice)

Clonez la voix d'un locuteur a l'aide de l'encodeur de locuteur CAM++ (192 dim., CoreML Neural Engine) :

```swift
import CosyVoiceTTS
import AudioCommon

let model = try await CosyVoiceTTSModel.fromPretrained()
let speaker = try await CamPlusPlusSpeaker.fromPretrained()
// Telecharge ~14 Mo du modele CoreML CAM++ a la premiere utilisation

let refAudio = try AudioFileLoader.load(url: referenceURL, targetSampleRate: 16000)
let embedding = try speaker.embed(audio: refAudio, sampleRate: 16000)
// embedding: [Float] de longueur 192

let audio = model.synthesize(
    text: "Hello in a cloned voice!",
    language: "english",
    speakerEmbedding: embedding
)
```

### CLI CosyVoice TTS

```bash
make build

# Synthese de base
.build/release/audio speak "Hello world" --engine cosyvoice --language english --output output.wav

# Clonage vocal (telecharge l'encodeur de locuteur CAM++ a la premiere utilisation)
.build/release/audio speak "Hello world" --engine cosyvoice --voice-sample reference.wav --output cloned.wav

# Dialogue multi-locuteurs avec clonage vocal
.build/release/audio speak "[S1] Hello there! [S2] Hey, how are you?" \
    --engine cosyvoice --speakers s1=alice.wav,s2=bob.wav -o dialogue.wav

# Balises d'emotion/style en ligne
.build/release/audio speak "(excited) Wow, amazing! (sad) But I have to go..." \
    --engine cosyvoice -o emotion.wav

# Combine : dialogue + emotions + clonage vocal
.build/release/audio speak "[S1] (happy) Great news! [S2] (surprised) Really?" \
    --engine cosyvoice --speakers s1=alice.wav,s2=bob.wav -o combined.wav

# Instruction de style personnalisee
.build/release/audio speak "Hello world" --engine cosyvoice --cosy-instruct "Speak cheerfully" -o cheerful.wav

# Synthese en streaming
.build/release/audio speak "Hello world" --engine cosyvoice --language english --stream --output output.wav
```

## Kokoro TTS -- Synthese vocale legere embarquee (iOS + macOS)

### Synthese de base

```swift
import KokoroTTS
import AudioCommon  // pour WAVWriter

let tts = try await KokoroTTSModel.fromPretrained()
// Telecharge ~170 Mo au premier lancement (modeles CoreML + empreintes vocales + dictionnaires)

let audio = try tts.synthesize(text: "Hello world", voice: "af_heart")
// Sortie : echantillons float mono 24kHz
try WAVWriter.write(samples: audio, sampleRate: 24000, to: outputURL)
```

54 voix predefinies dans 10 langues. Non autoregressif -- pipeline CoreML en 3 etapes, pas de boucle d'echantillonnage. S'execute sur le Neural Engine, libere entierement le GPU.

### CLI Kokoro TTS

```bash
make build

# Synthese de base
.build/release/audio kokoro "Hello world" --voice af_heart --output hello.wav

# Choisir la langue
.build/release/audio kokoro "Bonjour le monde" --voice ff_siwis --language fr --output bonjour.wav

# Lister les voix disponibles
.build/release/audio kokoro --list-voices
```

## Qwen3 Chat (LLM embarque)

```swift
import Qwen3Chat

let chat = try await Qwen3ChatModel.fromPretrained()
// Telecharge ~318 Mo au premier lancement (modele CoreML INT4 + tokenizer)

// Reponse unique
let response = try chat.generate("What is Swift?", systemPrompt: "Answer briefly.")
print(response)

// Tokens en streaming
let stream = chat.chatStream("Tell me a joke", systemPrompt: "Be funny.")
for try await token in stream {
    print(token, terminator: "")
}
```

Qwen3-0.6B quantifie INT4 pour CoreML. S'execute sur le Neural Engine avec ~2 tok/s sur iPhone, ~15 tok/s sur les puces M. Prend en charge la conversation multi-tours avec cache KV, le mode reflexion (tokens `<think>`), et l'echantillonnage configurable (temperature, top-k, top-p, penalite de repetition).

## Detection d'activite vocale (VAD) -- Detecter la parole dans l'audio

### VAD en streaming (Silero)

Silero VAD v5 traite des blocs audio de 32ms avec une latence inferieure a la milliseconde -- ideal pour la detection de parole en temps reel depuis des microphones ou des flux.

```swift
import SpeechVAD

let vad = try await SileroVADModel.fromPretrained()
// Ou utiliser CoreML (Neural Engine, consommation reduite) :
// let vad = try await SileroVADModel.fromPretrained(engine: .coreml)

// Streaming : traiter des blocs de 512 echantillons (32ms @ 16kHz)
let prob = vad.processChunk(samples)  // → 0.0...1.0
vad.resetState()  // a appeler entre differents flux audio

// Ou detecter tous les segments d'un coup
let segments = vad.detectSpeech(audio: audioSamples, sampleRate: 16000)
for seg in segments {
    print("Parole : \(seg.startTime)s - \(seg.endTime)s")
}
```

### Streaming evenementiel

```swift
let processor = StreamingVADProcessor(model: vad)

// Alimentez l'audio de n'importe quelle longueur -- les evenements sont emis lorsque la parole est confirmee
let events = processor.process(samples: audioBuffer)
for event in events {
    switch event {
    case .speechStarted(let time):
        print("Parole commencee a \(time)s")
    case .speechEnded(let segment):
        print("Parole : \(segment.startTime)s - \(segment.endTime)s")
    }
}

// Vider le tampon en fin de flux
let final = processor.flush()
```

### CLI VAD

```bash
make build

# Silero VAD en streaming (blocs de 32ms)
.build/release/audio vad-stream audio.wav

# Backend CoreML (Neural Engine)
.build/release/audio vad-stream audio.wav --engine coreml

# Avec des seuils personnalises
.build/release/audio vad-stream audio.wav --onset 0.6 --offset 0.4

# Sortie JSON
.build/release/audio vad-stream audio.wav --json

# VAD pyannote par lots (fenetres glissantes de 10s)
.build/release/audio vad audio.wav
```

## Diarisation de locuteurs -- Qui a parle quand

### Pipeline de diarisation

```swift
import SpeechVAD

let pipeline = try await DiarizationPipeline.fromPretrained()
// Ou utiliser les empreintes CoreML (Neural Engine, libere le GPU) :
// let pipeline = try await DiarizationPipeline.fromPretrained(embeddingEngine: .coreml)

let result = pipeline.diarize(audio: samples, sampleRate: 16000)
for seg in result.segments {
    print("Locuteur \(seg.speakerId) : [\(seg.startTime)s - \(seg.endTime)s]")
}
print("\(result.numSpeakers) locuteurs detectes")
```

### Empreinte vocale

```swift
let model = try await WeSpeakerModel.fromPretrained()
// Ou : let model = try await WeSpeakerModel.fromPretrained(engine: .coreml)
let embedding = model.embed(audio: samples, sampleRate: 16000)
// embedding: [Float] de longueur 256, normalise L2

// Comparer des locuteurs
let similarity = WeSpeakerModel.cosineSimilarity(embeddingA, embeddingB)
```

### Extraction de locuteur

Extrayez uniquement les segments d'un locuteur specifique a l'aide d'un enregistrement de reference :

```swift
let pipeline = try await DiarizationPipeline.fromPretrained()
let targetEmb = pipeline.embeddingModel.embed(audio: enrollmentAudio, sampleRate: 16000)
let segments = pipeline.extractSpeaker(
    audio: meetingAudio, sampleRate: 16000,
    targetEmbedding: targetEmb
)
```

### Diarisation Sortformer (de bout en bout, CoreML)

NVIDIA Sortformer predit l'activite de locuteur par trame pour jusqu'a 4 locuteurs directement -- aucune empreinte ni regroupement necessaires. S'execute sur le Neural Engine.

```swift
let diarizer = try await SortformerDiarizer.fromPretrained()
let result = diarizer.diarize(audio: samples, sampleRate: 16000, config: .default)
for seg in result.segments {
    print("Locuteur \(seg.speakerId) : [\(seg.startTime)s - \(seg.endTime)s]")
}
```

### CLI Diarisation

```bash
make build

# Diarisation pyannote (par defaut)
.build/release/audio diarize meeting.wav

# Diarisation Sortformer (CoreML, Neural Engine)
.build/release/audio diarize meeting.wav --engine sortformer

# Empreintes CoreML (Neural Engine, pyannote uniquement)
.build/release/audio diarize meeting.wav --embedding-engine coreml

# Sortie JSON
.build/release/audio diarize meeting.wav --json

# Extraire un locuteur specifique (pyannote uniquement)
.build/release/audio diarize meeting.wav --target-speaker enrollment.wav

# Empreinte vocale
.build/release/audio embed-speaker enrollment.wav --json
.build/release/audio embed-speaker enrollment.wav --engine coreml
```

Voir [Speaker Diarization](docs/inference/speaker-diarization.md) pour les details d'architecture.

## Amelioration de la parole -- Suppression du bruit et nettoyage audio

### Suppression du bruit

```swift
import SpeechEnhancement
import AudioCommon  // pour WAVWriter

let enhancer = try await SpeechEnhancer.fromPretrained()
// Telecharge ~4.3 Mo au premier lancement (modele Core ML FP16 + donnees auxiliaires)

let cleanAudio = try enhancer.enhance(audio: noisyAudio, sampleRate: 48000)
try WAVWriter.write(samples: cleanAudio, sampleRate: 48000, to: outputURL)
```

### CLI Debruitage

```bash
make build

# Suppression du bruit de base
.build/release/audio denoise noisy.wav

# Chemin de sortie personnalise
.build/release/audio denoise noisy.wav --output clean.wav
```

Voir [Speech Enhancement](docs/inference/speech-enhancement.md) pour les details d'architecture.

## Pipelines -- Composer plusieurs modeles

Tous les modeles sont conformes a des protocoles partages (`SpeechRecognitionModel`, `SpeechGenerationModel`, `SpeechEnhancementModel`, etc.) et peuvent etre composes en pipelines :

### Reconnaissance vocale sur audio bruite (DeepFilterNet + ASR)

```swift
import SpeechEnhancement
import Qwen3ASR

let enhancer = try await SpeechEnhancer.fromPretrained()
let asr = try await Qwen3ASRModel.fromPretrained()

// Ameliorer a 48kHz, puis transcrire a 16kHz
let clean = try enhancer.enhance(audio: noisyAudio, sampleRate: 48000)
let clean16k = AudioResampler.resample(clean, from: 48000, to: 16000)
let text = asr.transcribe(audio: clean16k, sampleRate: 16000)
```

### Relais voix-a-voix (VAD + ASR + TTS)

```swift
import SpeechVAD
import Qwen3ASR
import Qwen3TTS

let vad = try await SileroVADModel.fromPretrained()
let asr = try await Qwen3ASRModel.fromPretrained()
let tts = try await Qwen3TTSModel.fromPretrained()

// Detecter les segments de parole, transcrire, resynthetiser
let segments = vad.detectSpeech(audio: audio, sampleRate: 16000)
for seg in segments {
    let chunk = Array(audio[Int(seg.startTime * 16000)..<Int(seg.endTime * 16000)])
    let text = asr.transcribe(audio: chunk, sampleRate: 16000)
    let speech = tts.synthesize(text: text, language: "english")
    // speech : echantillons float mono 24kHz
}
```

### Transcription de reunion (Diarisation + ASR)

```swift
import SpeechVAD
import Qwen3ASR

let pipeline = try await DiarizationPipeline.fromPretrained()
let asr = try await Qwen3ASRModel.fromPretrained()

let result = pipeline.diarize(audio: meetingAudio, sampleRate: 16000)
for seg in result.segments {
    let chunk = Array(meetingAudio[Int(seg.startTime * 16000)..<Int(seg.endTime * 16000)])
    let text = asr.transcribe(audio: chunk, sampleRate: 16000)
    print("Locuteur \(seg.speakerId) [\(seg.startTime)s-\(seg.endTime)s] : \(text)")
}
```

Voir [Shared Protocols](docs/shared-protocols.md) pour la reference complete des protocoles.

## Serveur API HTTP

Un serveur HTTP autonome expose tous les modeles via des endpoints REST et WebSocket. Les modeles sont charges a la demande lors de la premiere requete.

```bash
swift build -c release
.build/release/audio-server --port 8080

# Transcrire de l'audio
curl -X POST http://localhost:8080/transcribe --data-binary @audio.wav -H "Content-Type: audio/wav"

# Synthese vocale
curl -X POST http://localhost:8080/speak -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "engine": "cosyvoice"}' -o output.wav

# Parole-a-parole (PersonaPlex)
curl -X POST http://localhost:8080/respond --data-binary @question.wav -o response.wav

# Amelioration de la parole
curl -X POST http://localhost:8080/enhance --data-binary @noisy.wav -o clean.wav

# Precharger tous les modeles au demarrage
.build/release/audio-server --preload --port 8080
```

### Streaming WebSocket

#### API Realtime OpenAI (`/v1/realtime`)

L'endpoint WebSocket principal implemente le protocole [OpenAI Realtime API](https://platform.openai.com/docs/api-reference/realtime) -- tous les messages sont en JSON avec un champ `type`, l'audio est encode en PCM16 base64 mono 24kHz.

**Evenements Client → Serveur :**

| Evenement | Description |
|-----------|-------------|
| `session.update` | Configurer le moteur, la langue, le format audio |
| `input_audio_buffer.append` | Envoyer un bloc audio PCM16 en base64 |
| `input_audio_buffer.commit` | Transcrire l'audio accumule (ASR) |
| `input_audio_buffer.clear` | Vider le tampon audio |
| `response.create` | Demander une synthese TTS |

**Evenements Serveur → Client :**

| Evenement | Description |
|-----------|-------------|
| `session.created` | Session initialisee |
| `session.updated` | Configuration confirmee |
| `input_audio_buffer.committed` | Audio soumis pour transcription |
| `conversation.item.input_audio_transcription.completed` | Resultat ASR |
| `response.audio.delta` | Bloc audio PCM16 en base64 (TTS) |
| `response.audio.done` | Streaming audio termine |
| `response.done` | Reponse terminee avec metadonnees |
| `error` | Erreur avec type et message |

```javascript
const ws = new WebSocket('ws://localhost:8080/v1/realtime');

// ASR : envoyer l'audio, obtenir la transcription
ws.send(JSON.stringify({ type: 'input_audio_buffer.append', audio: base64PCM16 }));
ws.send(JSON.stringify({ type: 'input_audio_buffer.commit' }));
// → recoit : conversation.item.input_audio_transcription.completed

// TTS : envoyer du texte, recevoir l'audio en streaming
ws.send(JSON.stringify({
  type: 'response.create',
  response: { modalities: ['audio', 'text'], instructions: 'Hello world' }
}));
// → recoit : response.audio.delta (blocs base64), response.audio.done, response.done
```

Un exemple de client HTML se trouve dans `Examples/websocket-client.html` -- ouvrez-le dans un navigateur pendant que le serveur est en marche.

Le serveur est un module `AudioServer` separe et un executable `audio-server` -- il n'ajoute pas Hummingbird/WebSocket au CLI principal `audio`.

## Latence (M2 Max, 64 Go)

### ASR

| Modele | Backend | RTF | Audio de 10s traite en |
|--------|---------|-----|------------------------|
| Qwen3-ASR-0.6B (4-bit) | MLX | ~0.06 | ~0.6s |
| Qwen3-ASR-0.6B (INT8) | CoreML + MLX | ~0.09 | ~0.9s |
| Qwen3-ASR-1.7B (8-bit) | MLX | ~0.11 | ~1.1s |
| Parakeet-TDT-0.6B (INT4) | CoreML (Neural Engine) | ~0.12 a froid, ~0.03 a chaud | ~1.2s / ~0.3s |
| Whisper-large-v3 | whisper.cpp (Q5_0) | ~0.10 | ~1.0s |
| Whisper-small | whisper.cpp (Q5_0) | ~0.04 | ~0.4s |

### Alignement force

| Modele | Framework | Audio de 20s | RTF |
|--------|-----------|--------------|-----|
| Qwen3-ForcedAligner-0.6B (4-bit) | MLX Swift (debug) | ~365ms | ~0.018 |

> Passe unique non autoregressive -- pas de boucle d'echantillonnage. L'encodeur audio domine (~328ms), le decodeur en passe unique fait ~37ms. **55x plus rapide que le temps reel.**

### TTS

| Modele | Framework | Court (1s) | Moyen (3s) | Long (6s) | Premier paquet en streaming |
|--------|-----------|-----------|------------|-----------|---------------------------|
| Qwen3-TTS-0.6B (4-bit) | MLX Swift (release) | 1.6s (RTF 1.2) | 2.3s (RTF 0.7) | 3.9s (RTF 0.7) | ~120ms (1 trame) |
| Kokoro-82M | CoreML (Neural Engine) | ~1.4s (RTFx 0.7) | ~4.3s (RTFx 0.7) | ~8.6s (RTFx 0.7) | N/A (non autoregressif) |
| Apple `AVSpeechSynthesizer` | AVFoundation | 0.08s | 0.08s | 0.17s (RTF 0.02) | N/A |

> Qwen3-TTS genere une parole naturelle et expressive avec prosodie et emotion, fonctionnant **plus vite que le temps reel** (RTF < 1.0). La synthese en streaming delivre le premier bloc audio en ~120ms. Kokoro-82M s'execute entierement sur le Neural Engine avec un pipeline en 3 etapes (RTFx ~0.7), ideal pour iOS. Le TTS integre d'Apple est plus rapide mais produit une parole robotique et monotone.

### PersonaPlex (Parole-a-parole)

| Modele | Framework | ms/etape | RTF | Notes |
|--------|-----------|----------|-----|-------|
| PersonaPlex-7B (8-bit) | MLX Swift (release) | ~112ms | ~1.4 | Recommande — reponses coherentes, 30% plus rapide que 4-bit |
| PersonaPlex-7B (4-bit) | MLX Swift (release) | ~158ms | ~1.97 | Non recommande — qualite de sortie degradee |

> **Utilisez 8-bit.** INT8 est plus rapide (112 ms/etape vs. 158 ms/etape) et produit des reponses full-duplex coherentes. La quantification INT4 degrade la qualite de generation, produisant un discours incoherent. INT8 fonctionne a ~112ms/etape sur M2 Max.

### VAD et empreinte vocale

| Modele | Backend | Latence par appel | RTF | Notes |
|--------|---------|-------------------|-----|-------|
| Silero-VAD-v5 | MLX | ~2.1ms / bloc | 0.065 | GPU (Metal) |
| Silero-VAD-v5 | CoreML | ~0.27ms / bloc | 0.008 | Neural Engine, **7.7x plus rapide** |
| WeSpeaker ResNet34-LM | MLX | ~310ms / 20s audio | 0.016 | GPU (Metal) |
| WeSpeaker ResNet34-LM | CoreML | ~430ms / 20s audio | 0.021 | Neural Engine, libere le GPU |

> Silero VAD CoreML s'execute sur le Neural Engine a 7.7x la vitesse de MLX, ce qui le rend ideal pour une ecoute micro permanente. WeSpeaker MLX est plus rapide sur GPU, mais CoreML libere le GPU pour des taches simultanees (TTS, ASR). Les deux backends produisent des resultats equivalents.

### Amelioration de la parole

| Modele | Backend | Duree | Latence | RTF |
|--------|---------|-------|---------|-----|
| DeepFilterNet3 (FP16) | CoreML | 5s | 0.65s | 0.13 |
| DeepFilterNet3 (FP16) | CoreML | 10s | 1.2s | 0.12 |
| DeepFilterNet3 (FP16) | CoreML | 20s | 4.8s | 0.24 |

RTF = Real-Time Factor (plus bas est mieux, < 1.0 = plus rapide que le temps reel). Le cout du GRU evolue en ~O(n^2).

### MLX vs CoreML

Les deux backends produisent des resultats equivalents. Choisissez en fonction de votre charge de travail :

| | MLX | CoreML |
|---|---|---|
| **Materiel** | GPU (shaders Metal) | Neural Engine + CPU |
| **Ideal pour** | Debit maximum, taches mono-modele | Pipelines multi-modeles, taches en arriere-plan |
| **Consommation** | Utilisation GPU elevee | Consommation reduite, libere le GPU |
| **Latence** | Plus rapide pour les gros modeles (WeSpeaker) | Plus rapide pour les petits modeles (Silero VAD) |

**Inference sur ordinateur** : MLX est le backend par defaut -- performance mono-modele la plus rapide sur Apple Silicon. Passez a CoreML lorsque vous executez plusieurs modeles simultanement (ex. : VAD + ASR + TTS) pour eviter la contention GPU, ou pour les charges sensibles a la batterie sur les portables.

Des modeles CoreML sont disponibles pour l'encodeur Qwen3-ASR, Silero VAD et WeSpeaker. Pour Qwen3-ASR, utilisez `--engine qwen3-coreml` (hybride : encodeur CoreML sur ANE + decodeur texte MLX sur GPU). Pour VAD/empreintes, passez `engine: .coreml` a la construction -- l'API d'inference est identique.

## Benchmarks de precision

### ASR -- Taux d'erreur de mots ([details](docs/benchmarks/asr-wer.md))

| Modele | WER% (LibriSpeech test-clean) | RTF |
|--------|-------------------------------|-----|
| Qwen3-ASR 1.7B 8-bit | **2.35** | 0.090 |
| Qwen3-ASR 1.7B 4-bit | 2.57 | 0.045 |
| Parakeet TDT INT8 | 2.74 | 0.089 |
| Qwen3-ASR 0.6B 8-bit | 2.80 | 0.025 |

Qwen3-ASR 1.7B 8-bit depasse Whisper Large v3 Turbo (2.5%) a taille comparable. Multilingue : 10 langues evaluees sur FLEURS.

### TTS -- Intelligibilite aller-retour ([details](docs/benchmarks/tts-roundtrip.md))

| Moteur | WER% | RTF |
|--------|------|-----|
| CosyVoice3 | **3.25** | 0.59 |
| Qwen3-TTS 1.7B | 3.47 | 0.79 |
| Kokoro-82M | 3.90 | 0.17 |

### VAD -- Detection de la parole ([details](docs/benchmarks/vad-detection.md))

| Moteur | F1% (FLEURS) | RTF |
|--------|-------------|-----|
| FireRedVAD | **99.12** | 0.007 |
| Silero CoreML | 95.13 | 0.022 |
| Pyannote MLX | 94.86 | 0.358 |

## Architecture

**Modeles :** [ASR Model](docs/models/asr-model.md), [TTS Model](docs/models/tts-model.md), [CosyVoice TTS](docs/models/cosyvoice-tts.md), [Kokoro TTS](docs/models/kokoro-tts.md), [Parakeet TDT](docs/models/parakeet-asr.md), [PersonaPlex](docs/models/personaplex.md), [FireRedVAD](docs/models/fireredvad.md)

**Inference :** [Qwen3-ASR](docs/inference/qwen3-asr-inference.md), [Parakeet TDT](docs/inference/parakeet-asr-inference.md), [TTS Inference](docs/inference/qwen3-tts-inference.md), [Forced Aligner](docs/inference/forced-aligner.md), [FireRedVAD](docs/inference/fireredvad.md), [Silero VAD](docs/inference/silero-vad.md), [Speaker Diarization](docs/inference/speaker-diarization.md), [Speech Enhancement](docs/inference/speech-enhancement.md), [Lecture Audio](docs/audio/playback.md)

**Benchmarks :** [ASR WER](docs/benchmarks/asr-wer.md), [TTS Round-Trip](docs/benchmarks/tts-roundtrip.md), [VAD Detection](docs/benchmarks/vad-detection.md)

**Reference :** [Shared Protocols](docs/shared-protocols.md)

## Configuration du cache

Les poids des modeles sont caches localement. Modifiez l'emplacement du cache avec :

```bash
export QWEN3_CACHE_DIR=/path/to/cache
```

## Bibliotheque Metal MLX

Si vous voyez `Failed to load the default metallib` a l'execution, la bibliotheque de shaders Metal est manquante. Executez `make build` (ou `./scripts/build_mlx_metallib.sh release` apres un `swift build` manuel) pour la compiler. Si le Metal Toolchain est manquant, installez-le d'abord :

```bash
xcodebuild -downloadComponent MetalToolchain
```

## Tests

Les tests unitaires (configuration, echantillonnage, pretraitement de texte, correction d'horodatages) s'executent sans telechargement de modele :

```bash
swift test --filter "Qwen3TTSConfigTests|SamplingTests|CosyVoiceTTSConfigTests|CamPlusPlusMelExtractorTests|PersonaPlexTests|ForcedAlignerTests/testText|ForcedAlignerTests/testTimestamp|ForcedAlignerTests/testLIS|SileroVADTests/testSilero|SileroVADTests/testReflection|SileroVADTests/testProcess|SileroVADTests/testReset|SileroVADTests/testDetect|SileroVADTests/testStreaming|SileroVADTests/testVADEvent|KokoroTTSTests"
```

Les tests d'integration necessitent les poids des modeles (telecharges automatiquement au premier lancement) :

```bash
# Aller-retour TTS : synthetiser du texte, sauvegarder en WAV, retranscrire avec l'ASR
swift test --filter TTSASRRoundTripTests

# ASR uniquement : transcrire un audio de test
swift test --filter Qwen3ASRIntegrationTests

# Alignement force E2E : horodatages au niveau du mot (~979 Mo de telechargement)
swift test --filter ForcedAlignerTests/testForcedAlignerE2E

# PersonaPlex E2E : pipeline parole-a-parole (~5.5 Go de telechargement)
PERSONAPLEX_E2E=1 swift test --filter PersonaPlexE2ETests
```

> **Remarque :** La bibliotheque Metal MLX doit etre compilee avant d'executer les tests qui utilisent des operations MLX.
> Voir [Bibliotheque Metal MLX](#bibliotheque-metal-mlx) pour les instructions.

## Langues prises en charge

| Modele | Langues |
|--------|---------|
| Qwen3-ASR | 52 langues (CN, EN, cantonais, DE, FR, ES, JA, KO, RU, + 22 dialectes chinois, ...) |
| Parakeet TDT | 25 langues europeennes (BG, CS, DA, DE, EL, EN, ES, ET, FI, FR, HR, HU, IT, LT, LV, MT, NL, PL, PT, RO, RU, SK, SL, SV, UK) |
| Qwen3-TTS | EN, CN, DE, JA, ES, FR, KO, RU, IT, PT (+ dialectes de Pekin/Sichuan via CustomVoice) |
| CosyVoice TTS | CN, EN, JA, KO, DE, ES, FR, IT, RU |
| Kokoro TTS | EN (US/UK), ES, FR, HI, IT, JA, PT, CN, KO, DE |
| PersonaPlex | EN |

## Comparatif

### Reconnaissance vocale (ASR) : speech-swift vs alternatives

| | **speech-swift (Qwen3-ASR)** | **whisper.cpp** | **Apple SFSpeechRecognizer** | **Google Cloud Speech** |
|---|---|---|---|---|
| **Execution** | Sur l'appareil (MLX/CoreML) | Sur l'appareil (CPU/GPU) | Sur l'appareil ou cloud | Cloud uniquement |
| **Langues** | 52 | 100+ | ~70 (sur l'appareil : limite) | 125+ |
| **RTF (audio 10s, M2 Max)** | 0.06 (17x temps reel) | 0.10 (Whisper-large-v3) | N/A | N/A |
| **Streaming** | Non (par lots) | Non (par lots) | Oui | Oui |
| **Modeles personnalises** | Oui (poids HuggingFace interchangeables) | Oui (modeles GGML) | Non | Non |
| **API Swift** | async/await natif | C++ avec bridge Swift | Natif | REST/gRPC |
| **Confidentialite** | Entierement sur l'appareil | Entierement sur l'appareil | Selon la config | Donnees envoyees au cloud |
| **Horodatages des mots** | Oui (Forced Aligner) | Oui | Limite | Oui |
| **Cout** | Gratuit (Apache 2.0) | Gratuit (MIT) | Gratuit (sur l'appareil) | Facturation a la minute |

### Synthese vocale (TTS) : speech-swift vs alternatives

| | **speech-swift (Qwen3-TTS)** | **speech-swift (Kokoro)** | **Apple AVSpeechSynthesizer** | **ElevenLabs / Cloud TTS** |
|---|---|---|---|---|
| **Qualite** | Neurale, expressive | Neurale, naturelle | Robotique, monotone | Neurale, qualite maximale |
| **Execution** | Sur l'appareil (MLX) | Sur l'appareil (CoreML) | Sur l'appareil | Cloud uniquement |
| **Streaming** | Oui (~120ms premier bloc) | Non (pipeline en 3 etapes) | Non | Oui |
| **Clonage vocal** | Oui | Non | Non | Oui |
| **Voix** | 9 integrees + clonage | 54 voix predefinies | ~50 voix systeme | 1000+ |
| **Langues** | 10 | 10 | 60+ | 30+ |
| **Support iOS** | macOS uniquement | iOS + macOS | iOS + macOS | Tous (API) |
| **Cout** | Gratuit (Apache 2.0) | Gratuit (Apache 2.0) | Gratuit | Facturation au caractere |

### Quand utiliser speech-swift

- **Applications sensibles a la confidentialite** -- medical, juridique, entreprise ou l'audio ne peut pas quitter l'appareil
- **Usage hors ligne** -- aucune connexion internet necessaire apres le telechargement initial du modele
- **Budget limite** -- pas de frais a la minute ou au caractere
- **Optimisation Apple Silicon** -- concu specifiquement pour le GPU M-series (Metal) et le Neural Engine
- **Pipeline complet** -- combinez ASR + TTS + VAD + diarisation + amelioration dans un seul package Swift

## FAQ

**Est-ce que speech-swift fonctionne sur iOS ?**
Kokoro TTS, Qwen3.5-Chat (CoreML), Silero VAD, Parakeet ASR, DeepFilterNet3 et WeSpeaker fonctionnent tous sur iOS 17+ via CoreML sur le Neural Engine. Les modeles bases sur MLX (Qwen3-ASR, Qwen3-TTS, Qwen3.5-Chat MLX, PersonaPlex) necessitent macOS 14+ sur Apple Silicon.

**Une connexion internet est-elle necessaire ?**
Uniquement pour le telechargement initial du modele depuis HuggingFace (automatique, cache dans `~/Library/Caches/qwen3-speech/`). Ensuite, toute l'inference s'execute entierement hors ligne sans acces reseau.

**Comment speech-swift se compare-t-il a Whisper ?**
Qwen3-ASR-0.6B atteint un RTF de 0.06 sur M2 Max -- 40% plus rapide que Whisper-large-v3 via whisper.cpp (RTF 0.10) -- avec une precision comparable sur 52 langues. speech-swift fournit une API Swift native async/await, alors que whisper.cpp necessite un bridge C++.

**Puis-je l'utiliser dans une application commerciale ?**
Oui. speech-swift est sous licence Apache 2.0. Les poids des modeles sous-jacents ont leurs propres licences (verifiez la page HuggingFace de chaque modele).

**Quelles puces Apple Silicon sont prises en charge ?**
Toutes les puces de la serie M : M1, M2, M3, M4 et leurs variantes Pro/Max/Ultra. Necessite macOS 14+ (Sonoma) ou iOS 17+.

**Quelle quantite de memoire est necessaire ?**
De ~3 Mo (Silero VAD) a ~6.5 Go (PersonaPlex 7B). Kokoro TTS utilise ~500 Mo, Qwen3-ASR ~2.2 Go. Voir le tableau [Memoire requise](#memoire-requise) pour tous les details.

**Peut-on executer plusieurs modeles simultanement ?**
Oui. Utilisez les modeles CoreML sur le Neural Engine conjointement avec les modeles MLX sur le GPU pour eviter la contention -- par exemple, Silero VAD (CoreML) + Qwen3-ASR (MLX) + Qwen3-TTS (MLX).

**Y a-t-il une API REST ?**
Oui. L'executable `audio-server` expose tous les modeles via des endpoints HTTP REST et WebSocket, y compris un WebSocket compatible avec l'API OpenAI Realtime a `/v1/realtime`.

## Contribuer

Les contributions sont les bienvenues ! Qu'il s'agisse d'une correction de bug, de l'integration d'un nouveau modele ou d'une amelioration de la documentation -- les PR sont appreciees.

**Pour commencer :**
1. Forkez le depot et creez une branche de fonctionnalite
2. `make build` pour compiler (necessite Xcode + Metal Toolchain)
3. `make test` pour lancer la suite de tests
4. Ouvrez une PR contre `main`

## Licence

Apache 2.0
