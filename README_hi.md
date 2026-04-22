# Speech Swift

Apple Silicon के लिए AI स्पीच मॉडल, MLX Swift और CoreML द्वारा संचालित।

📖 Read in: [English](README.md) · [中文](README_zh.md) · [日本語](README_ja.md) · [한국어](README_ko.md) · [Español](README_es.md) · [Deutsch](README_de.md) · [Français](README_fr.md) · [हिन्दी](README_hi.md) · [Português](README_pt.md) · [Русский](README_ru.md)

Mac और iOS के लिए ऑन-डिवाइस स्पीच रिकग्निशन, सिंथेसिस और समझ। Apple Silicon पर पूरी तरह लोकली चलता है — कोई क्लाउड नहीं, कोई API key नहीं, कोई डेटा डिवाइस से बाहर नहीं जाता।

**[📚 पूर्ण डॉक्यूमेंटेशन →](https://soniqo.audio/hi)** · **[🤗 HuggingFace मॉडल](https://huggingface.co/aufklarer)** · **[📝 ब्लॉग](https://blog.ivan.digital)**

- **[Qwen3-ASR](https://soniqo.audio/hi/guides/transcribe)** — स्पीच-टू-टेक्स्ट (ऑटोमैटिक स्पीच रिकग्निशन, 52 भाषाएँ, MLX + CoreML)
- **[Parakeet TDT](https://soniqo.audio/hi/guides/parakeet)** — CoreML के माध्यम से स्पीच-टू-टेक्स्ट (Neural Engine, NVIDIA FastConformer + TDT decoder, 25 भाषाएँ)
- **[Omnilingual ASR](https://soniqo.audio/hi/guides/omnilingual)** — स्पीच-टू-टेक्स्ट (Meta wav2vec2 + CTC, **1,672 भाषाएँ** 32 लिपियों में, CoreML 300M + MLX 300M/1B/3B/7B)
- **[Streaming Dictation](https://soniqo.audio/hi/guides/dictate)** — पार्शियल्स और एंड-ऑफ-अटरन्स डिटेक्शन के साथ रियल-टाइम डिक्टेशन (Parakeet-EOU-120M)
- **[Nemotron Streaming](https://soniqo.audio/hi/guides/nemotron)** — नेटिव विराम चिह्न और कैपिटलाइज़ेशन के साथ लो-लेटेंसी स्ट्रीमिंग ASR (NVIDIA Nemotron-Speech-Streaming-0.6B, CoreML, अंग्रेज़ी)
- **[Qwen3-ForcedAligner](https://soniqo.audio/hi/guides/align)** — शब्द-स्तरीय टाइमस्टैम्प अलाइनमेंट (ऑडियो + टेक्स्ट → टाइमस्टैम्प)
- **[Qwen3-TTS](https://soniqo.audio/hi/guides/speak)** — टेक्स्ट-टू-स्पीच (सर्वोच्च गुणवत्ता, स्ट्रीमिंग, कस्टम स्पीकर, 10 भाषाएँ)
- **[CosyVoice TTS](https://soniqo.audio/hi/guides/cosyvoice)** — वॉयस क्लोनिंग, मल्टी-स्पीकर डायलॉग, इमोशन टैग के साथ स्ट्रीमिंग TTS (9 भाषाएँ)
- **[Kokoro TTS](https://soniqo.audio/hi/guides/kokoro)** — ऑन-डिवाइस TTS (82M, CoreML/Neural Engine, 54 वॉयस, iOS-ready, 10 भाषाएँ)
- **[Qwen3.5-Chat](https://soniqo.audio/hi/guides/chat)** — ऑन-डिवाइस LLM चैट (0.8B, MLX INT4 + CoreML INT8, DeltaNet हाइब्रिड, स्ट्रीमिंग टोकन)
- **[PersonaPlex](https://soniqo.audio/hi/guides/respond)** — फुल-डुप्लेक्स स्पीच-टू-स्पीच (7B, ऑडियो इन → ऑडियो आउट, 18 वॉयस प्रीसेट)
- **[DeepFilterNet3](https://soniqo.audio/hi/guides/denoise)** — रियल-टाइम नॉइज़ सप्रेशन (2.1M params, 48 kHz)
- **[वेक-वर्ड](https://soniqo.audio/hi/guides/wake-word)** — ऑन-डिवाइस कीवर्ड स्पॉटिंग (KWS Zipformer 3M, CoreML, 26× रियल-टाइम, कॉन्फ़िगरेबल कीवर्ड सूची)
- **[VAD](https://soniqo.audio/hi/guides/vad)** — वॉयस एक्टिविटी डिटेक्शन (Silero स्ट्रीमिंग, Pyannote ऑफ़लाइन, FireRedVAD 100+ भाषाएँ)
- **[Speaker Diarization](https://soniqo.audio/hi/guides/diarize)** — कौन कब बोला (Pyannote पाइपलाइन, Neural Engine पर एंड-टू-एंड Sortformer)
- **[Speaker Embeddings](https://soniqo.audio/hi/guides/embed-speaker)** — WeSpeaker ResNet34 (256-dim), CAM++ (192-dim)

पेपर: [Qwen3-ASR](https://arxiv.org/abs/2601.21337) (Alibaba) · [Qwen3-TTS](https://arxiv.org/abs/2601.15621) (Alibaba) · [Omnilingual ASR](https://arxiv.org/abs/2511.09690) (Meta) · [Parakeet TDT](https://arxiv.org/abs/2304.06795) (NVIDIA) · [CosyVoice 3](https://arxiv.org/abs/2505.17589) (Alibaba) · [Kokoro](https://arxiv.org/abs/2301.01695) (StyleTTS 2) · [PersonaPlex](https://arxiv.org/abs/2602.06053) (NVIDIA) · [Mimi](https://arxiv.org/abs/2410.00037) (Kyutai) · [Sortformer](https://arxiv.org/abs/2409.06656) (NVIDIA)

## समाचार

- **19 Apr 2026** — [MLX vs CoreML on Apple Silicon — A Practical Guide to Picking the Right Backend](https://blog.ivan.digital/mlx-vs-coreml-on-apple-silicon-a-practical-guide-to-picking-the-right-backend-and-why-you-should-f77ddea7b27a)
- **20 Mar 2026** — [We Beat Whisper Large v3 with a 600M Model Running Entirely on Your Mac](https://blog.ivan.digital/we-beat-whisper-large-v3-with-a-600m-model-running-entirely-on-your-mac-20e6ce191174)
- **26 Feb 2026** — [Speaker Diarization and Voice Activity Detection on Apple Silicon — Native Swift with MLX](https://blog.ivan.digital/speaker-diarization-and-voice-activity-detection-on-apple-silicon-native-swift-with-mlx-92ea0c9aca0f)
- **23 Feb 2026** — [NVIDIA PersonaPlex 7B on Apple Silicon — Full-Duplex Speech-to-Speech in Native Swift with MLX](https://blog.ivan.digital/nvidia-personaplex-7b-on-apple-silicon-full-duplex-speech-to-speech-in-native-swift-with-mlx-0aa5276f2e23)
- **12 Feb 2026** — [Qwen3-ASR Swift: On-Device ASR + TTS for Apple Silicon — Architecture and Benchmarks](https://blog.ivan.digital/qwen3-asr-swift-on-device-asr-tts-for-apple-silicon-architecture-and-benchmarks-27cbf1e4463f)

## त्वरित प्रारंभ

अपने `Package.swift` में पैकेज जोड़ें:

```swift
.package(url: "https://github.com/soniqo/speech-swift", branch: "main")
```

केवल वही मॉड्यूल इम्पोर्ट करें जिनकी आपको ज़रूरत है — प्रत्येक मॉडल अपनी अलग SPM लाइब्रेरी है, इसलिए आप केवल उसी के लिए भुगतान करते हैं जो आप उपयोग करते हैं:

```swift
.product(name: "ParakeetStreamingASR", package: "speech-swift"),
.product(name: "SpeechUI",             package: "speech-swift"),  // वैकल्पिक SwiftUI व्यू
```

**3 लाइनों में ऑडियो बफ़र ट्रांसक्राइब करें:**

```swift
import ParakeetStreamingASR

let model = try await ParakeetStreamingASRModel.fromPretrained()
let text = try model.transcribeAudio(audioSamples, sampleRate: 16000)
```

**पार्शियल्स के साथ लाइव स्ट्रीमिंग:**

```swift
for await partial in model.transcribeStream(audio: samples, sampleRate: 16000) {
    print(partial.isFinal ? "FINAL: \(partial.text)" : "... \(partial.text)")
}
```

**~10 लाइनों में SwiftUI डिक्टेशन व्यू:**

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

`SpeechUI` केवल `TranscriptionView` (finals + partials) और `TranscriptionStore` (स्ट्रीमिंग ASR एडाप्टर) प्रदान करता है। ऑडियो विज़ुअलाइज़ेशन और प्लेबैक के लिए AVFoundation का उपयोग करें।

उपलब्ध SPM उत्पाद: `Qwen3ASR`, `Qwen3TTS`, `Qwen3TTSCoreML`, `ParakeetASR`, `ParakeetStreamingASR`, `NemotronStreamingASR`, `OmnilingualASR`, `KokoroTTS`, `CosyVoiceTTS`, `PersonaPlex`, `SpeechVAD`, `SpeechEnhancement`, `Qwen3Chat`, `SpeechCore`, `SpeechUI`, `AudioCommon`.

## मॉडल

नीचे संक्षिप्त दृश्य। **[पूर्ण मॉडल कैटलॉग (आकार, क्वांटिज़ेशन, डाउनलोड URL, मेमोरी टेबल्स) → soniqo.audio/architecture](https://soniqo.audio/hi/architecture)**.

| मॉडल | कार्य | बैकएंड | आकार | भाषाएँ |
|------|------|--------|------|--------|
| [Qwen3-ASR](https://soniqo.audio/hi/guides/transcribe) | स्पीच → टेक्स्ट | MLX, CoreML (हाइब्रिड) | 0.6B, 1.7B | 52 |
| [Parakeet TDT](https://soniqo.audio/hi/guides/parakeet) | स्पीच → टेक्स्ट | CoreML (ANE) | 0.6B | 25 यूरोपीय |
| [Parakeet EOU](https://soniqo.audio/hi/guides/dictate) | स्पीच → टेक्स्ट (स्ट्रीमिंग) | CoreML (ANE) | 120M | 25 यूरोपीय |
| [Nemotron Streaming](https://soniqo.audio/hi/guides/nemotron) | भाषण → टेक्स्ट (स्ट्रीमिंग, विराम चिह्नों के साथ) | CoreML (ANE) | 0.6B | अंग्रेज़ी |
| [Omnilingual ASR](https://soniqo.audio/hi/guides/omnilingual) | स्पीच → टेक्स्ट | CoreML (ANE), MLX | 300M / 1B / 3B / 7B | **[1,672](https://github.com/facebookresearch/omnilingual-asr/blob/main/src/omnilingual_asr/models/wav2vec2_llama/lang_ids.py)** |
| [Qwen3-ForcedAligner](https://soniqo.audio/hi/guides/align) | ऑडियो + टेक्स्ट → टाइमस्टैम्प | MLX, CoreML | 0.6B | बहुभाषी |
| [Qwen3-TTS](https://soniqo.audio/hi/guides/speak) | टेक्स्ट → स्पीच | MLX, CoreML | 0.6B, 1.7B | 10 |
| [CosyVoice3](https://soniqo.audio/hi/guides/cosyvoice) | टेक्स्ट → स्पीच | MLX | 0.5B | 9 |
| [Kokoro-82M](https://soniqo.audio/hi/guides/kokoro) | टेक्स्ट → स्पीच | CoreML (ANE) | 82M | 10 |
| [Qwen3.5-Chat](https://soniqo.audio/hi/guides/chat) | टेक्स्ट → टेक्स्ट (LLM) | MLX, CoreML | 0.8B | बहुभाषी |
| [PersonaPlex](https://soniqo.audio/hi/guides/respond) | स्पीच → स्पीच | MLX | 7B | EN |
| [Silero VAD](https://soniqo.audio/hi/guides/vad) | वॉयस एक्टिविटी डिटेक्शन | MLX, CoreML | 309K | भाषा-तटस्थ |
| [Pyannote](https://soniqo.audio/hi/guides/diarize) | VAD + Diarization | MLX | 1.5M | भाषा-तटस्थ |
| [Sortformer](https://soniqo.audio/hi/guides/diarize) | Diarization (E2E) | CoreML (ANE) | — | भाषा-तटस्थ |
| [DeepFilterNet3](https://soniqo.audio/hi/guides/denoise) | स्पीच एन्हांसमेंट | CoreML | 2.1M | भाषा-तटस्थ |
| [WeSpeaker](https://soniqo.audio/hi/guides/embed-speaker) | स्पीकर एम्बेडिंग | MLX, CoreML | 6.6M | भाषा-तटस्थ |

## इंस्टॉलेशन

### Homebrew

नेटिव ARM Homebrew (`/opt/homebrew`) आवश्यक है। Rosetta/x86_64 Homebrew समर्थित नहीं है।

```bash
brew tap soniqo/speech https://github.com/soniqo/speech-swift
brew install speech
```

फिर:

```bash
audio transcribe recording.wav
audio speak "Hello world"
audio respond --input question.wav --transcript
audio-server --port 8080            # स्थानीय HTTP / WebSocket सर्वर (OpenAI-compatible /v1/realtime)
```

**[पूर्ण CLI संदर्भ →](https://soniqo.audio/hi/cli)**

### Swift Package Manager

```swift
dependencies: [
    .package(url: "https://github.com/soniqo/speech-swift", branch: "main")
]
```

केवल वही इम्पोर्ट करें जो आपको चाहिए — प्रत्येक मॉडल अपना SPM target है:

```swift
import Qwen3ASR             // स्पीच रिकग्निशन (MLX)
import ParakeetASR          // स्पीच रिकग्निशन (CoreML, बैच)
import ParakeetStreamingASR // पार्शियल्स + EOU के साथ स्ट्रीमिंग डिक्टेशन
import NemotronStreamingASR // अंग्रेज़ी स्ट्रीमिंग ASR नेटिव विराम चिह्न के साथ (0.6B)
import OmnilingualASR       // 1,672 भाषाएँ (CoreML + MLX)
import Qwen3TTS             // टेक्स्ट-टू-स्पीच
import CosyVoiceTTS         // वॉयस क्लोनिंग के साथ TTS
import KokoroTTS            // टेक्स्ट-टू-स्पीच (iOS-ready)
import Qwen3Chat            // ऑन-डिवाइस LLM चैट
import PersonaPlex          // फुल-डुप्लेक्स स्पीच-टू-स्पीच
import SpeechVAD            // VAD + स्पीकर डायराइज़ेशन + एम्बेडिंग
import SpeechEnhancement    // नॉइज़ सप्रेशन
import SpeechUI             // स्ट्रीमिंग ट्रांसक्रिप्ट के लिए SwiftUI कॉम्पोनेंट
import AudioCommon          // शेयर्ड प्रोटोकॉल और यूटिलिटीज़
```

### आवश्यकताएँ

- Swift 6+, Xcode 16+ (Metal Toolchain के साथ)
- macOS 15+ (Sequoia) या iOS 18+, Apple Silicon (M1/M2/M3/M4)

macOS 15 / iOS 18 न्यूनतम आवश्यकता [MLState](https://developer.apple.com/documentation/coreml/mlstate) से आती है —— Apple की परसिस्टेंट ANE स्टेट API —— जिसका उपयोग CoreML पाइपलाइन (Qwen3-ASR, Qwen3-Chat, Qwen3-TTS) टोकन स्टेप्स के बीच KV कैश को Neural Engine पर रखने के लिए करती हैं।

### सोर्स से बिल्ड

```bash
git clone https://github.com/soniqo/speech-swift
cd speech-swift
make build
```

`make build` Swift पैकेज **और** MLX Metal shader library दोनों को कंपाइल करता है। GPU इन्फ़रेंस के लिए Metal library आवश्यक है — इसके बिना आपको रनटाइम पर `Failed to load the default metallib` दिखेगा। `make debug` डीबग बिल्ड के लिए, `make test` टेस्ट सूट के लिए।

**[पूर्ण बिल्ड और इंस्टॉल गाइड →](https://soniqo.audio/hi/getting-started)**

## डेमो ऐप्स

- **[DictateDemo](Examples/DictateDemo/)** ([डॉक्स](https://soniqo.audio/hi/guides/dictate)) — macOS मेनू-बार स्ट्रीमिंग डिक्टेशन, लाइव पार्शियल्स, VAD-संचालित एंड-ऑफ-अटरन्स डिटेक्शन, और वन-क्लिक कॉपी। बैकग्राउंड एजेंट के रूप में चलता है (Parakeet-EOU-120M + Silero VAD)।
- **[iOSEchoDemo](Examples/iOSEchoDemo/)** — iOS इको डेमो (Parakeet ASR + Kokoro TTS)। डिवाइस और सिम्युलेटर।
- **[PersonaPlexDemo](Examples/PersonaPlexDemo/)** — माइक इनपुट, VAD, और मल्टी-टर्न संदर्भ के साथ संवादात्मक वॉयस असिस्टेंट। macOS। M2 Max पर RTF ~0.94 (रियल-टाइम से तेज़)।
- **[SpeechDemo](Examples/SpeechDemo/)** — टैब्ड इंटरफ़ेस में डिक्टेशन और TTS सिंथेसिस। macOS।

प्रत्येक डेमो के README में बिल्ड निर्देश हैं।

## कोड उदाहरण

नीचे दिए गए स्निपेट्स प्रत्येक डोमेन के लिए न्यूनतम पथ दिखाते हैं। प्रत्येक अनुभाग [soniqo.audio](https://soniqo.audio/hi) पर पूर्ण गाइड से लिंक होता है जिसमें कॉन्फ़िगरेशन विकल्प, कई बैकएंड, स्ट्रीमिंग पैटर्न, और CLI रेसिपी शामिल हैं।

### स्पीच-टू-टेक्स्ट — [पूर्ण गाइड →](https://soniqo.audio/hi/guides/transcribe)

```swift
import Qwen3ASR

let model = try await Qwen3ASRModel.fromPretrained()
let text = model.transcribe(audio: audioSamples, sampleRate: 16000)
```

वैकल्पिक बैकएंड: [Parakeet TDT](https://soniqo.audio/hi/guides/parakeet) (CoreML, 32× रियल-टाइम), [Omnilingual ASR](https://soniqo.audio/hi/guides/omnilingual) (1,672 भाषाएँ, CoreML या MLX), [स्ट्रीमिंग डिक्टेशन](https://soniqo.audio/hi/guides/dictate) (लाइव पार्शियल्स)।

### फ़ोर्स्ड अलाइनमेंट — [पूर्ण गाइड →](https://soniqo.audio/hi/guides/align)

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

### टेक्स्ट-टू-स्पीच — [पूर्ण गाइड →](https://soniqo.audio/hi/guides/speak)

```swift
import Qwen3TTS
import AudioCommon

let model = try await Qwen3TTSModel.fromPretrained()
let audio = model.synthesize(text: "Hello world", language: "english")
try WAVWriter.write(samples: audio, sampleRate: 24000, to: outputURL)
```

वैकल्पिक TTS इंजन: [CosyVoice3](https://soniqo.audio/hi/guides/cosyvoice) (स्ट्रीमिंग + वॉयस क्लोनिंग + इमोशन टैग), [Kokoro-82M](https://soniqo.audio/hi/guides/kokoro) (iOS-ready, 54 वॉयस), [वॉयस क्लोनिंग](https://soniqo.audio/hi/guides/voice-cloning)।

### स्पीच-टू-स्पीच — [पूर्ण गाइड →](https://soniqo.audio/hi/guides/respond)

```swift
import PersonaPlex

let model = try await PersonaPlexModel.fromPretrained()
let responseAudio = model.respond(userAudio: userSamples)
// 24 kHz मोनो Float32 आउटपुट — प्लेबैक के लिए तैयार
```

### LLM चैट — [पूर्ण गाइड →](https://soniqo.audio/hi/guides/chat)

```swift
import Qwen3Chat

let chat = try await Qwen35MLXChat.fromPretrained()
chat.chat(messages: [(.user, "Explain MLX in one sentence")]) { token, isFinal in
    print(token, terminator: "")
}
```

### वॉयस एक्टिविटी डिटेक्शन — [पूर्ण गाइड →](https://soniqo.audio/hi/guides/vad)

```swift
import SpeechVAD

let vad = try await SileroVADModel.fromPretrained()
let segments = vad.detectSpeech(audio: samples, sampleRate: 16000)
for s in segments { print("\(s.startTime)s → \(s.endTime)s") }
```

### स्पीकर डायराइज़ेशन — [पूर्ण गाइड →](https://soniqo.audio/hi/guides/diarize)

```swift
import SpeechVAD

let diarizer = try await DiarizationPipeline.fromPretrained()
let segments = diarizer.diarize(audio: samples, sampleRate: 16000)
for s in segments { print("Speaker \(s.speakerId): \(s.startTime)s - \(s.endTime)s") }
```

### स्पीच एन्हांसमेंट — [पूर्ण गाइड →](https://soniqo.audio/hi/guides/denoise)

```swift
import SpeechEnhancement

let denoiser = try await DeepFilterNet3Model.fromPretrained()
let clean = try denoiser.enhance(audio: noisySamples, sampleRate: 48000)
```

### वॉयस पाइपलाइन (ASR → LLM → TTS) — [पूर्ण गाइड →](https://soniqo.audio/hi/api)

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

`VoicePipeline` रियल-टाइम वॉयस-एजेंट स्टेट मशीन है ([speech-core](https://github.com/soniqo/speech-core) द्वारा संचालित) जो VAD-संचालित टर्न डिटेक्शन, इंटरप्शन हैंडलिंग, और eager STT के साथ आता है। यह किसी भी `SpeechRecognitionModel` + `SpeechGenerationModel` + `StreamingVADProvider` को कनेक्ट करता है।

### HTTP API सर्वर

```bash
audio-server --port 8080
```

सभी मॉडलों को HTTP REST + WebSocket endpoints के माध्यम से एक्सपोज़ करता है, जिसमें `/v1/realtime` पर OpenAI Realtime API-संगत WebSocket शामिल है। देखें [`Sources/AudioServer/`](Sources/AudioServer/)।

## आर्किटेक्चर

speech-swift प्रति मॉडल एक SPM टारगेट में विभाजित है ताकि उपभोक्ता केवल उसी के लिए भुगतान करें जो वे इम्पोर्ट करते हैं। साझा इन्फ़्रास्ट्रक्चर `AudioCommon` (प्रोटोकॉल, ऑडियो I/O, HuggingFace डाउनलोडर, `SentencePieceModel`) और `MLXCommon` (वेट लोडिंग, `QuantizedLinear` हेल्पर्स, multi-head attention के लिए `SDPA` हेल्पर) में रहता है।

**[बैकएंड, मेमोरी टेबल्स, और मॉड्यूल मैप के साथ पूर्ण आर्किटेक्चर डायग्राम → soniqo.audio/architecture](https://soniqo.audio/hi/architecture)** · **[API संदर्भ → soniqo.audio/api](https://soniqo.audio/hi/api)** · **[बेंचमार्क → soniqo.audio/benchmarks](https://soniqo.audio/hi/benchmarks)**

स्थानीय डॉक्स (रिपॉज़िटरी):
- **मॉडल:** [Qwen3-ASR](docs/models/asr-model.md) · [Qwen3-TTS](docs/models/tts-model.md) · [CosyVoice](docs/models/cosyvoice-tts.md) · [Kokoro](docs/models/kokoro-tts.md) · [Parakeet TDT](docs/models/parakeet-asr.md) · [Parakeet Streaming](docs/models/parakeet-streaming-asr.md) · [Nemotron Streaming](docs/models/nemotron-streaming.md) · [Omnilingual ASR](docs/models/omnilingual-asr.md) · [PersonaPlex](docs/models/personaplex.md) · [FireRedVAD](docs/models/fireredvad.md)
- **इन्फ़रेंस:** [Qwen3-ASR](docs/inference/qwen3-asr-inference.md) · [Parakeet TDT](docs/inference/parakeet-asr-inference.md) · [Parakeet Streaming](docs/inference/parakeet-streaming-asr-inference.md) · [Nemotron Streaming](docs/inference/nemotron-streaming-inference.md) · [Omnilingual ASR](docs/inference/omnilingual-asr-inference.md) · [TTS](docs/inference/qwen3-tts-inference.md) · [Forced Aligner](docs/inference/forced-aligner.md) · [Silero VAD](docs/inference/silero-vad.md) · [स्पीकर डायराइज़ेशन](docs/inference/speaker-diarization.md) · [स्पीच एन्हांसमेंट](docs/inference/speech-enhancement.md)
- **संदर्भ:** [शेयर्ड प्रोटोकॉल](docs/shared-protocols.md)

## कैश कॉन्फ़िगरेशन

मॉडल वेट पहले उपयोग पर HuggingFace से डाउनलोड होते हैं और `~/Library/Caches/qwen3-speech/` में कैश होते हैं। `QWEN3_CACHE_DIR` (CLI) या `cacheDir:` (Swift API) से ओवरराइड करें। सभी `fromPretrained()` एंट्री पॉइंट `offlineMode: true` भी स्वीकार करते हैं ताकि वेट कैश होने पर नेटवर्क स्किप किया जा सके।

सैंडबॉक्स्ड iOS कंटेनर पाथ सहित पूर्ण विवरण के लिए [`docs/inference/cache-and-offline.md`](docs/inference/cache-and-offline.md) देखें।

## MLX Metal library

यदि आपको रनटाइम पर `Failed to load the default metallib` दिखता है, तो Metal shader library गुम है। मैनुअल `swift build` के बाद `make build` या `./scripts/build_mlx_metallib.sh release` चलाएँ। यदि Metal Toolchain गुम है, तो पहले इसे इंस्टॉल करें:

```bash
xcodebuild -downloadComponent MetalToolchain
```

## टेस्टिंग

```bash
make test                            # पूर्ण सुइट (यूनिट + मॉडल डाउनलोड के साथ E2E)
swift test --skip E2E                # केवल यूनिट (CI-सुरक्षित, कोई डाउनलोड नहीं)
swift test --filter Qwen3ASRTests    # विशिष्ट मॉड्यूल
```

E2E टेस्ट क्लासेस `E2E` उपसर्ग का उपयोग करती हैं ताकि CI उन्हें `--skip E2E` से फ़िल्टर कर सके। पूर्ण टेस्टिंग नियम के लिए [CLAUDE.md](CLAUDE.md#testing) देखें।

## योगदान

PRs का स्वागत है — बग फ़िक्स, नए मॉडल इंटीग्रेशन, डॉक्यूमेंटेशन। फ़ॉर्क करें, feature ब्रांच बनाएँ, `make build && make test`, `main` के विरुद्ध PR खोलें।

## लाइसेंस

Apache 2.0
