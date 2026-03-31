# Speech Swift

Apple Silicon के लिए AI स्पीच मॉडल, MLX Swift और CoreML द्वारा संचालित।

📖 Read in: [English](README.md) · [中文](README_zh.md) · [日本語](README_ja.md) · [한국어](README_ko.md) · [Español](README_es.md) · [Deutsch](README_de.md) · [Français](README_fr.md) · [हिन्दी](README_hi.md) · [Português](README_pt.md) · [Русский](README_ru.md)

Mac और iOS के लिए ऑन-डिवाइस स्पीच रिकग्निशन, सिंथेसिस और समझ। Apple Silicon पर पूरी तरह लोकली चलता है — कोई क्लाउड नहीं, कोई API key नहीं, कोई डेटा डिवाइस से बाहर नहीं जाता।

[Homebrew से इंस्टॉल करें](#homebrew) या Swift Package डिपेंडेंसी के रूप में जोड़ें।

**[डॉक्यूमेंटेशन](https://soniqo.audio)** · **[HuggingFace मॉडल](https://huggingface.co/aufklarer)** · **[ब्लॉग](https://blog.ivan.digital)**

- **Qwen3-ASR** — स्पीच-टू-टेक्स्ट / स्पीच रिकग्निशन (ऑटोमैटिक स्पीच रिकग्निशन, 52 भाषाएँ)
- **Parakeet TDT** — CoreML के माध्यम से स्पीच-टू-टेक्स्ट (Neural Engine, NVIDIA FastConformer + TDT decoder, 25 भाषाएँ)
- **Qwen3-ForcedAligner** — शब्द-स्तरीय टाइमस्टैम्प अलाइनमेंट (ऑडियो + टेक्स्ट → टाइमस्टैम्प)
- **Qwen3-TTS** — टेक्स्ट-टू-स्पीच सिंथेसिस (सर्वोच्च गुणवत्ता, स्ट्रीमिंग, कस्टम स्पीकर, 10 भाषाएँ)
- **CosyVoice TTS** — स्ट्रीमिंग, वॉयस क्लोनिंग, मल्टी-स्पीकर डायलॉग, और इमोशन टैग के साथ टेक्स्ट-टू-स्पीच (9 भाषाएँ, DiT flow matching, CAM++ speaker encoder)
- **Kokoro TTS** — ऑन-डिवाइस टेक्स्ट-टू-स्पीच (82M params, CoreML/Neural Engine, 54 वॉयस, iOS-ready, 10 भाषाएँ)
- **Qwen3.5-Chat** — ऑन-डिवाइस LLM चैट (0.8B, MLX + CoreML, INT4 + CoreML INT8, DeltaNet हाइब्रिड, स्ट्रीमिंग टोकन)
- **PersonaPlex** — फुल-डुप्लेक्स स्पीच-टू-स्पीच वार्तालाप (7B, ऑडियो इन → ऑडियो आउट, 18 वॉयस प्रीसेट)
- **DeepFilterNet3** — स्पीच एन्हांसमेंट / नॉइज़ सप्रेशन (2.1M params, रियल-टाइम 48kHz)
- **FireRedVAD** — ऑफ़लाइन वॉयस एक्टिविटी डिटेक्शन (DFSMN, CoreML, 100+ भाषाएँ, 97.6% F1)
- **Silero VAD** — स्ट्रीमिंग वॉयस एक्टिविटी डिटेक्शन (32ms chunks, सब-मिलीसेकंड लेटेंसी)
- **Pyannote VAD** — ऑफ़लाइन वॉयस एक्टिविटी डिटेक्शन (10s विंडो, मल्टी-स्पीकर ओवरलैप)
- **Speaker Diarization** — कौन कब बोला (Pyannote सेगमेंटेशन + एक्टिविटी-बेस्ड स्पीकर चेनिंग, या एंड-टू-एंड Sortformer Neural Engine पर)
- **Speaker Embeddings** — स्पीकर वेरिफिकेशन और आइडेंटिफिकेशन (WeSpeaker ResNet34, 256-dim वेक्टर)

पेपर: [Qwen3-ASR](https://arxiv.org/abs/2601.21337) (Alibaba), [Qwen3-TTS](https://arxiv.org/abs/2601.15621) (Alibaba), [Qwen3](https://arxiv.org/abs/2505.09388) (Alibaba), [Parakeet TDT](https://arxiv.org/abs/2304.06795) (NVIDIA), [CosyVoice 3](https://arxiv.org/abs/2505.17589) (Alibaba), [Kokoro](https://arxiv.org/abs/2301.01695) (StyleTTS 2), [PersonaPlex](https://arxiv.org/abs/2602.06053) (NVIDIA), [Mimi](https://arxiv.org/abs/2410.00037) (Kyutai), [Sortformer](https://arxiv.org/abs/2409.06656) (NVIDIA)

## रोडमैप

आगे की योजनाओं के लिए [रोडमैप चर्चा](https://github.com/soniqo/speech-swift/discussions/81) देखें — टिप्पणियाँ और सुझावों का स्वागत है!

## समाचार

- **20 Mar 2026** — [We Beat Whisper Large v3 with a 600M Model Running Entirely on Your Mac](https://blog.ivan.digital/we-beat-whisper-large-v3-with-a-600m-model-running-entirely-on-your-mac-20e6ce191174)
- **26 Feb 2026** — [Speaker Diarization and Voice Activity Detection on Apple Silicon — Native Swift with MLX](https://blog.ivan.digital/speaker-diarization-and-voice-activity-detection-on-apple-silicon-native-swift-with-mlx-92ea0c9aca0f)
- **23 Feb 2026** — [NVIDIA PersonaPlex 7B on Apple Silicon — Full-Duplex Speech-to-Speech in Native Swift with MLX](https://blog.ivan.digital/nvidia-personaplex-7b-on-apple-silicon-full-duplex-speech-to-speech-in-native-swift-with-mlx-0aa5276f2e23)
- **12 Feb 2026** — [Qwen3-ASR Swift: On-Device ASR + TTS for Apple Silicon — Architecture and Benchmarks](https://blog.ivan.digital/qwen3-asr-swift-on-device-asr-tts-for-apple-silicon-architecture-and-benchmarks-27cbf1e4463f)

## मॉडल

| Model | Task | Streaming | Languages | Sizes |
|-------|------|-----------|-----------|-------|
| Qwen3-ASR-0.6B | Speech → Text | No | 52 languages | [4-bit](https://huggingface.co/aufklarer/Qwen3-ASR-0.6B-MLX-4bit) 680 MB · [8-bit](https://huggingface.co/aufklarer/Qwen3-ASR-0.6B-MLX-8bit) 1.0 GB · [CoreML](https://huggingface.co/aufklarer/Qwen3-ASR-CoreML) 180 MB |
| Qwen3-ASR-1.7B | Speech → Text | No | 52 languages | [4-bit](https://huggingface.co/aufklarer/Qwen3-ASR-1.7B-MLX-4bit) 2.1 GB · [8-bit](https://huggingface.co/aufklarer/Qwen3-ASR-1.7B-MLX-8bit) 3.2 GB |
| Parakeet-TDT-0.6B | Speech → Text | No | 25 European languages | [CoreML INT4](https://huggingface.co/aufklarer/Parakeet-TDT-v3-CoreML-INT4) 315 MB · [CoreML INT8](https://huggingface.co/aufklarer/Parakeet-TDT-v3-CoreML-INT8) 500 MB |
| Qwen3-ForcedAligner-0.6B | Audio + Text → Timestamps | No | Multi | [4-bit](https://huggingface.co/aufklarer/Qwen3-ForcedAligner-0.6B-4bit) 979 MB · [8-bit](https://huggingface.co/aufklarer/Qwen3-ForcedAligner-0.6B-8bit) 1.4 GB · [CoreML INT4](https://huggingface.co/aufklarer/Qwen3-ForcedAligner-0.6B-CoreML-INT4) 630 MB · [CoreML INT8](https://huggingface.co/aufklarer/Qwen3-ForcedAligner-0.6B-CoreML-INT8) 1.0 GB |
| Qwen3-TTS-0.6B Base | Text → Speech | Yes (~120ms) | 10 languages | [4-bit](https://huggingface.co/aufklarer/Qwen3-TTS-12Hz-0.6B-Base-MLX-4bit) 1.7 GB · [8-bit](https://huggingface.co/aufklarer/Qwen3-TTS-12Hz-0.6B-Base-MLX-8bit) 2.4 GB |
| Qwen3-TTS-0.6B CustomVoice | Text → Speech | Yes (~120ms) | 10 languages | [4-bit](https://huggingface.co/aufklarer/Qwen3-TTS-12Hz-0.6B-CustomVoice-MLX-4bit) 1.7 GB |
| Qwen3-TTS-1.7B Base | Text → Speech | Yes (~120ms) | 10 languages | [4-bit](https://huggingface.co/aufklarer/Qwen3-TTS-12Hz-1.7B-Base-MLX-4bit) 3.2 GB · [8-bit](https://huggingface.co/aufklarer/Qwen3-TTS-12Hz-1.7B-Base-MLX-8bit) 4.8 GB |
| CosyVoice3-0.5B | Text → Speech | Yes (~150ms) | 9 languages | [4-bit](https://huggingface.co/aufklarer/CosyVoice3-0.5B-MLX-4bit) 1.2 GB |
| Kokoro-82M | Text → Speech | No | 10 languages | [CoreML](https://huggingface.co/aufklarer/Kokoro-82M-CoreML) ~170 MB |
| Qwen3.5-0.8B Chat | Text → Text (LLM) | Yes (streaming) | Multi | [MLX INT4](https://huggingface.co/aufklarer/Qwen3.5-0.8B-Chat-MLX) 418 MB · [CoreML INT8](https://huggingface.co/aufklarer/Qwen3.5-0.8B-Chat-CoreML) 981 MB |
| PersonaPlex-7B | Speech → Speech | Yes (~2s chunks) | EN | [4-bit](https://huggingface.co/aufklarer/PersonaPlex-7B-MLX-4bit) 4.9 GB · [8-bit](https://huggingface.co/aufklarer/PersonaPlex-7B-MLX-8bit) 9.1 GB |
| FireRedVAD | Voice Activity Detection | No (offline) | 100+ languages | [CoreML](https://huggingface.co/aufklarer/FireRedVAD-CoreML) ~1.2 MB |
| Silero-VAD-v5 | Voice Activity Detection | Yes (32ms chunks) | Language-agnostic | [MLX](https://huggingface.co/aufklarer/Silero-VAD-v5-MLX) · [CoreML](https://huggingface.co/aufklarer/Silero-VAD-v5-CoreML) ~1.2 MB |
| Pyannote-Segmentation-3.0 | VAD + Speaker Segmentation | No (10s windows) | Language-agnostic | [MLX](https://huggingface.co/aufklarer/Pyannote-Segmentation-MLX) ~5.7 MB |
| DeepFilterNet3 | Speech Enhancement | Yes (10ms frames) | Language-agnostic | [CoreML FP16](https://huggingface.co/aufklarer/DeepFilterNet3-CoreML) ~4.2 MB |
| WeSpeaker-ResNet34-LM | Speaker Embedding (256-dim) | No | Language-agnostic | [MLX](https://huggingface.co/aufklarer/WeSpeaker-ResNet34-LM-MLX) · [CoreML](https://huggingface.co/aufklarer/WeSpeaker-ResNet34-LM-CoreML) ~25 MB |
| CAM++ | Speaker Embedding (192-dim) | No | Language-agnostic | [CoreML](https://huggingface.co/aufklarer/CamPlusPlus-Speaker-CoreML) ~14 MB |
| Sortformer | Speaker Diarization (end-to-end) | Yes (chunked) | Language-agnostic | [CoreML](https://huggingface.co/aufklarer/Sortformer-Diarization-CoreML) ~240 MB |

### मेमोरी आवश्यकताएँ

वेट मेमोरी वह GPU (MLX) या ANE (CoreML) मेमोरी है जो मॉडल पैरामीटर द्वारा उपयोग की जाती है। पीक इनफ़रेंस में KV कैश, एक्टिवेशन, और इंटरमीडिएट टेंसर शामिल हैं।

| Model | Weight Memory | Peak Inference |
|-------|--------------|----------------|
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

### कौन सा TTS कब उपयोग करें

- **Qwen3-TTS**: सर्वोत्तम गुणवत्ता, स्ट्रीमिंग (~120ms), 9 बिल्ट-इन स्पीकर, 10 भाषाएँ, बैच सिंथेसिस
- **CosyVoice TTS**: स्ट्रीमिंग (~150ms), 9 भाषाएँ, वॉयस क्लोनिंग (CAM++ speaker encoder), मल्टी-स्पीकर डायलॉग (`[S1] ... [S2] ...`), इनलाइन इमोशन/स्टाइल टैग (`(happy)`, `(whispers)`), DiT flow matching + HiFi-GAN vocoder
- **Kokoro TTS**: हल्का iOS-ready TTS (82M params), CoreML/Neural Engine, 54 वॉयस, 10 भाषाएँ, नॉन-ऑटोरिग्रेसिव (3-स्टेज पाइपलाइन)
- **PersonaPlex**: फुल-डुप्लेक्स स्पीच-टू-स्पीच (ऑडियो इन → ऑडियो आउट), स्ट्रीमिंग (~2s chunks), 18 वॉयस प्रीसेट, Moshi आर्किटेक्चर पर आधारित

## इंस्टॉलेशन

### Homebrew

नेटिव ARM Homebrew (`/opt/homebrew`) आवश्यक है। Rosetta/x86_64 Homebrew समर्थित नहीं है।

```bash
brew tap soniqo/speech https://github.com/soniqo/speech-swift
brew install speech
```

फिर उपयोग करें:

```bash
audio transcribe recording.wav
audio speak "Hello world"
audio speak "Hallo Welt" --engine cosyvoice --language german
audio respond --input question.wav --transcript
```

> माइक्रोफ़ोन इनपुट के साथ इंटरैक्टिव वॉयस वार्तालाप के लिए, **[PersonaPlexDemo](Examples/PersonaPlexDemo/)** देखें।

### Swift Package Manager

अपने `Package.swift` में जोड़ें:

```swift
dependencies: [
    .package(url: "https://github.com/soniqo/speech-swift", branch: "main")
]
```

आवश्यक मॉड्यूल इम्पोर्ट करें:

```swift
import Qwen3ASR      // Speech recognition (MLX)
import ParakeetASR   // Speech recognition (CoreML)
import Qwen3TTS      // Text-to-speech (Qwen3)
import CosyVoiceTTS  // Text-to-speech (streaming)
import KokoroTTS     // Text-to-speech (CoreML, iOS-ready)
import Qwen3Chat     // On-device LLM chat (CoreML)
import PersonaPlex   // Speech-to-speech (full-duplex)
import SpeechVAD          // Voice activity detection (pyannote + Silero)
import SpeechEnhancement  // Noise suppression (DeepFilterNet3)
import AudioCommon        // Shared utilities
```

### आवश्यकताएँ

- Swift 5.9+
- macOS 14+ या iOS 17+
- Apple Silicon (M1/M2/M3/M4)
- Xcode 15+ (Metal Toolchain के साथ — यदि अनुपलब्ध हो तो `xcodebuild -downloadComponent MetalToolchain` चलाएँ)

### सोर्स से बिल्ड करें

```bash
git clone https://github.com/soniqo/speech-swift
cd speech-swift
make build
```

यह Swift पैकेज **और** MLX Metal shader लाइब्रेरी को एक साथ कंपाइल करता है। Metal लाइब्रेरी (`mlx.metallib`) GPU इनफ़रेंस के लिए आवश्यक है — इसके बिना रनटाइम पर `Failed to load the default metallib` एरर आएगी।

डीबग बिल्ड के लिए: `make debug`। यूनिट टेस्ट चलाने के लिए: `make test`।

## वॉयस असिस्टेंट आज़माएँ

**[PersonaPlexDemo](Examples/PersonaPlexDemo/)** एक रेडी-टू-रन macOS वॉयस असिस्टेंट है — बोलने के लिए टैप करें, रियल-टाइम में बोले गए जवाब पाएँ। माइक्रोफ़ोन इनपुट के साथ Silero VAD का उपयोग करता है ऑटोमैटिक स्पीच डिटेक्शन के लिए, ट्रांसक्रिप्शन के लिए Qwen3-ASR, और स्पीच-टू-स्पीच जनरेशन के लिए PersonaPlex 7B। 18 वॉयस प्रीसेट और इनर मोनोलॉग ट्रांसक्रिप्ट डिस्प्ले के साथ मल्टी-टर्न वार्तालाप।

```bash
make build  # repo root से — MLX metallib सहित सब कुछ बिल्ड करता है
cd Examples/PersonaPlexDemo
# .app बंडल के निर्देशों के लिए Examples/PersonaPlexDemo/README.md देखें
```

> M2 Max पर RTF ~0.94 (रियल-टाइम से तेज़)। पहली बार चलाने पर मॉडल स्वतः डाउनलोड होते हैं (~5.5 GB PersonaPlex + ~400 MB ASR)।

## डेमो ऐप्स

- **[PersonaPlexDemo](Examples/PersonaPlexDemo/)** — वार्तालाप वॉयस असिस्टेंट (माइक इनपुट, VAD, मल्टी-टर्न)। ऊपर देखें।
- **[SpeechDemo](Examples/SpeechDemo/)** — डिक्टेशन (Parakeet TDT / Qwen3-ASR भाषा ऑटो-डिटेक्ट के साथ) और टेक्स्ट-टू-स्पीच सिंथेसिस (Qwen3-TTS) टैब्ड इंटरफ़ेस में।

macOS `.app` बंडल के रूप में बिल्ड और रन करें — निर्देशों के लिए प्रत्येक डेमो का README देखें।

## स्पीच-टू-टेक्स्ट (ASR) — Swift में ऑडियो ट्रांसक्राइब करें

### बेसिक ट्रांसक्रिप्शन

```swift
import Qwen3ASR

// डिफ़ॉल्ट: 0.6B मॉडल
let model = try await Qwen3ASRModel.fromPretrained()

// या बेहतर सटीकता के लिए बड़ा 1.7B मॉडल उपयोग करें
let model = try await Qwen3ASRModel.fromPretrained(
    modelId: "aufklarer/Qwen3-ASR-1.7B-MLX-8bit"
)

// ऑडियो किसी भी सैंपल रेट पर हो सकता है — स्वतः 16kHz में रीसैंपल किया जाता है
let transcription = model.transcribe(audio: audioSamples, sampleRate: 16000)
print(transcription)
```

### CoreML Encoder (Neural Engine)

हाइब्रिड मोड: Neural Engine पर CoreML encoder + GPU पर MLX text decoder। कम पावर, encoder पास के लिए GPU फ़्री करता है।

```swift
import Qwen3ASR

let encoder = try await CoreMLASREncoder.fromPretrained()
let model = try await Qwen3ASRModel.fromPretrained()
let text = try model.transcribe(audio: audioSamples, sampleRate: 16000, coremlEncoder: encoder)
```

INT8 (180 MB, डिफ़ॉल्ट) और INT4 (90 MB) वेरिएंट उपलब्ध हैं। INT8 अनुशंसित है (FP32 के मुकाबले cosine similarity > 0.999)।

### Parakeet TDT (CoreML)

```swift
import ParakeetASR

let model = try await ParakeetASRModel.fromPretrained()
let transcription = model.transcribe(audio: audioSamples, sampleRate: 16000)
```

CoreML के माध्यम से Neural Engine पर चलता है — समवर्ती कार्यभार के लिए GPU फ़्री करता है। 25 यूरोपीय भाषाएँ, ~315 MB।

### ASR CLI

```bash
make build  # या: swift build -c release && ./scripts/build_mlx_metallib.sh release

# डिफ़ॉल्ट (Qwen3-ASR 0.6B, MLX)
.build/release/audio transcribe audio.wav

# 1.7B मॉडल उपयोग करें
.build/release/audio transcribe audio.wav --model 1.7B

# CoreML encoder (Neural Engine + MLX decoder)
.build/release/audio transcribe --engine qwen3-coreml audio.wav

# Parakeet TDT (CoreML, Neural Engine)
.build/release/audio transcribe --engine parakeet audio.wav
```

## फ़ोर्स्ड अलाइनमेंट

### शब्द-स्तरीय टाइमस्टैम्प

```swift
import Qwen3ASR

let aligner = try await Qwen3ForcedAligner.fromPretrained()
// पहली बार चलाने पर ~979 MB डाउनलोड होता है

let aligned = aligner.align(
    audio: audioSamples,
    text: "Can you guarantee that the replacement part will be shipped tomorrow?",
    sampleRate: 24000
)

for word in aligned {
    print("[\(String(format: "%.2f", word.startTime))s - \(String(format: "%.2f", word.endTime))s] \(word.text)")
}
```

### फ़ोर्स्ड अलाइनमेंट CLI

```bash
swift build -c release

# दिए गए टेक्स्ट के साथ अलाइन करें
.build/release/audio align audio.wav --text "Hello world"

# पहले ट्रांसक्राइब करें, फिर अलाइन करें
.build/release/audio align audio.wav
```

आउटपुट:
```
[0.12s - 0.45s] Can
[0.45s - 0.72s] you
[0.72s - 1.20s] guarantee
...
```

नॉन-ऑटोरिग्रेसिव — 3-स्टेज पाइपलाइन, कोई सैंपलिंग लूप नहीं। आर्किटेक्चर विवरण के लिए [Forced Aligner](docs/inference/forced-aligner.md) देखें।

## टेक्स्ट-टू-स्पीच (TTS) — Swift में स्पीच जनरेट करें

### बेसिक सिंथेसिस

```swift
import Qwen3TTS
import AudioCommon  // WAVWriter के लिए

let model = try await Qwen3TTSModel.fromPretrained()
// पहली बार चलाने पर ~1.7 GB डाउनलोड होता है (मॉडल + codec वेट)
let audio = model.synthesize(text: "Hello world", language: "english")
// आउटपुट 24kHz मोनो float सैंपल है
try WAVWriter.write(samples: audio, sampleRate: 24000, to: outputURL)
```

### TTS CLI

```bash
make build
.build/release/audio speak "Hello world" --output output.wav --language english
```

### कस्टम वॉयस / स्पीकर चयन

**CustomVoice** मॉडल वेरिएंट 9 बिल्ट-इन स्पीकर वॉयस और टोन/स्टाइल कंट्रोल के लिए नेचुरल लैंग्वेज निर्देशों का समर्थन करता है। CustomVoice मॉडल ID पास करके इसे लोड करें:

```swift
import Qwen3TTS

// CustomVoice मॉडल लोड करें (पहली बार चलाने पर ~1.7 GB डाउनलोड होता है)
let model = try await Qwen3TTSModel.fromPretrained(
    modelId: TTSModelVariant.customVoice.rawValue
)

// किसी विशिष्ट स्पीकर के साथ सिंथेसाइज़ करें
let audio = model.synthesize(text: "Hello world", language: "english", speaker: "vivian")

// उपलब्ध स्पीकर देखें
print(model.availableSpeakers)  // ["aiden", "dylan", "eric", ...]
```

CLI:

```bash
# स्पीकर के साथ CustomVoice मॉडल उपयोग करें
.build/release/audio speak "Hello world" --model customVoice --speaker vivian --output vivian.wav

# उपलब्ध स्पीकर की सूची देखें
.build/release/audio speak --model customVoice --list-speakers
```

### वॉयस क्लोनिंग (Base मॉडल)

रेफ़रेंस ऑडियो फ़ाइल से स्पीकर की आवाज़ क्लोन करें। दो मोड:

**ICL मोड** (अनुशंसित) — ट्रांसक्रिप्ट के साथ रेफ़रेंस ऑडियो को codec टोकन में एनकोड करता है। उच्च गुणवत्ता, विश्वसनीय EOS:

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

**X-vector मोड** — केवल स्पीकर एम्बेडिंग, ट्रांसक्रिप्ट की आवश्यकता नहीं लेकिन गुणवत्ता कम:

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

### टोन / स्टाइल निर्देश (केवल CustomVoice)

CustomVoice मॉडल बोलने की शैली, टोन, भावना, और गति को नियंत्रित करने के लिए नेचुरल लैंग्वेज `instruct` पैरामीटर स्वीकार करता है। निर्देश ChatML प्रारूप में मॉडल इनपुट से पहले जोड़ा जाता है।

```swift
// खुशमिज़ाज़ टोन
let audio = model.synthesize(
    text: "Welcome to our store!",
    language: "english",
    speaker: "ryan",
    instruct: "Speak in a cheerful, upbeat tone"
)

// धीमा और गंभीर
let audio = model.synthesize(
    text: "We regret to inform you...",
    language: "english",
    speaker: "aiden",
    instruct: "Read this slowly and solemnly"
)

// फुसफुसाहट
let audio = model.synthesize(
    text: "Can you keep a secret?",
    language: "english",
    speaker: "vivian",
    instruct: "Whisper this softly"
)
```

CLI:

```bash
# स्टाइल निर्देश के साथ
.build/release/audio speak "Good morning!" --model customVoice --speaker ryan \
    --instruct "Speak in a cheerful, upbeat tone" --output cheerful.wav

# CustomVoice उपयोग करते समय डिफ़ॉल्ट instruct ("Speak naturally.") स्वतः लागू होता है
.build/release/audio speak "Hello world" --model customVoice --speaker ryan --output natural.wav
```

जब CustomVoice मॉडल के साथ कोई `--instruct` नहीं दिया जाता, तो बेकाबू आउटपुट रोकने के लिए `"Speak naturally."` स्वतः लागू होता है। Base मॉडल instruct को सपोर्ट नहीं करता।

### बैच सिंथेसिस

अधिक थ्रूपुट के लिए एक ही बैच्ड फ़ॉरवर्ड पास में कई टेक्स्ट सिंथेसाइज़ करें:

```swift
let texts = ["Good morning everyone.", "The weather is nice today.", "Please open the window."]
let audioList = model.synthesizeBatch(texts: texts, language: "english", maxBatchSize: 4)
// audioList[i] texts[i] के लिए 24kHz मोनो float सैंपल है
for (i, audio) in audioList.enumerated() {
    try WAVWriter.write(samples: audio, sampleRate: 24000, to: URL(fileURLWithPath: "output_\(i).wav"))
}
```

#### बैच CLI

```bash
# प्रति पंक्ति एक टेक्स्ट वाली फ़ाइल बनाएँ
echo "Hello world.\nGoodbye world." > texts.txt
.build/release/audio speak --batch-file texts.txt --output output.wav --batch-size 4
# output_0.wav, output_1.wav, ... बनता है
```

> बैच मोड आइटम्स में मॉडल वेट लोड को वितरित करता है। Apple Silicon पर B=4 के लिए ~1.5-2.5x थ्रूपुट सुधार की उम्मीद करें। सर्वोत्तम परिणाम तब मिलते हैं जब टेक्स्ट समान-लंबाई का ऑडियो उत्पन्न करते हैं।

### सैंपलिंग विकल्प

```swift
let config = SamplingConfig(temperature: 0.9, topK: 50, repetitionPenalty: 1.05)
let audio = model.synthesize(text: "Hello", language: "english", sampling: config)
```

### स्ट्रीमिंग सिंथेसिस

कम फ़र्स्ट-पैकेट लेटेंसी के लिए ऑडियो चंक्स क्रमिक रूप से भेजें:

```swift
let stream = model.synthesizeStream(
    text: "Hello, this is streaming synthesis.",
    language: "english",
    streaming: .lowLatency  // पहले ऑडियो चंक तक ~120ms
)

for try await chunk in stream {
    // chunk.samples: [Float] PCM @ 24kHz
    // chunk.isFinal: अंतिम चंक पर true
    playAudio(chunk.samples)
}
```

CLI:

```bash
# डिफ़ॉल्ट स्ट्रीमिंग (3-frame पहला चंक, ~225ms लेटेंसी)
.build/release/audio speak "Hello world" --stream

# लो-लेटेंसी (1-frame पहला चंक, ~120ms लेटेंसी)
.build/release/audio speak "Hello world" --stream --first-chunk-frames 1
```

## स्पीच-टू-स्पीच — फुल-डुप्लेक्स वॉयस वार्तालाप

> माइक्रोफ़ोन इनपुट के साथ इंटरैक्टिव वॉयस असिस्टेंट के लिए, **[PersonaPlexDemo](Examples/PersonaPlexDemo/)** देखें — बोलने के लिए टैप करें, ऑटोमैटिक स्पीच डिटेक्शन के साथ मल्टी-टर्न वार्तालाप।

### स्पीच-टू-स्पीच

```swift
import PersonaPlex
import AudioCommon  // WAVWriter, AudioFileLoader के लिए

let model = try await PersonaPlexModel.fromPretrained()
// पहली बार चलाने पर ~5.5 GB डाउनलोड होता है (temporal 4-bit + depformer + Mimi codec + voice presets)

let audio = try AudioFileLoader.load(url: inputURL, targetSampleRate: 24000)
let (response, textTokens) = model.respond(userAudio: audio, voice: .NATM0)
// response: 24kHz मोनो float सैंपल
// textTokens: मॉडल का इनर मोनोलॉग (SentencePiece token IDs)
try WAVWriter.write(samples: response.audio, sampleRate: 24000, to: outputURL)
```

### इनर मोनोलॉग (टेक्स्ट आउटपुट)

PersonaPlex ऑडियो के साथ-साथ टेक्स्ट टोकन जनरेट करता है — मॉडल की आंतरिक तर्क प्रक्रिया। बिल्ट-इन SentencePiece डीकोडर से इन्हें डीकोड करें:

```swift
let decoder = try SentencePieceDecoder(modelPath: "tokenizer_spm_32k_3.model")
let transcript = decoder.decode(textTokens)
print(transcript)  // जैसे "Sure, I can help you with that..."
```

### स्ट्रीमिंग स्पीच-टू-स्पीच

```swift
// जनरेट होते ही ऑडियो चंक प्राप्त करें (~2s प्रति चंक)
let stream = model.respondStream(userAudio: audio, voice: .NATM0)
for try await chunk in stream {
    playAudio(chunk.samples)  // तुरंत चलाएँ, 24kHz मोनो
    // chunk.textTokens में इस चंक का टेक्स्ट है; अंतिम चंक में सभी टोकन हैं
    if chunk.isFinal { break }
}
```

### वॉयस चयन

18 वॉयस प्रीसेट उपलब्ध हैं:
- **Natural Female**: NATF0, NATF1, NATF2, NATF3
- **Natural Male**: NATM0, NATM1, NATM2, NATM3
- **Variety Female**: VARF0, VARF1, VARF2, VARF3, VARF4
- **Variety Male**: VARM0, VARM1, VARM2, VARM3, VARM4

### सिस्टम प्रॉम्प्ट

सिस्टम प्रॉम्प्ट मॉडल के वार्तालाप व्यवहार को निर्देशित करता है। कोई भी कस्टम प्रॉम्प्ट सादे स्ट्रिंग के रूप में पास करें:

```swift
// कस्टम सिस्टम प्रॉम्प्ट (स्वचालित रूप से टोकनाइज़ होता है)
let response = model.respond(
    userAudio: audio,
    voice: .NATM0,
    systemPrompt: "You enjoy having a good conversation."
)

// या प्रीसेट का उपयोग करें
let response = model.respond(
    userAudio: audio,
    voice: .NATM0,
    systemPromptTokens: SystemPromptPreset.customerService.tokens
)
```

उपलब्ध प्रीसेट: `focused` (डिफ़ॉल्ट), `assistant`, `customerService`, `teacher`।

### PersonaPlex CLI

```bash
make build

# बेसिक स्पीच-टू-स्पीच
.build/release/audio respond --input question.wav --output response.wav

# ट्रांसक्रिप्ट के साथ (इनर मोनोलॉग टेक्स्ट डीकोड करता है)
.build/release/audio respond --input question.wav --transcript

# JSON आउटपुट (ऑडियो पाथ, ट्रांसक्रिप्ट, लेटेंसी मेट्रिक्स)
.build/release/audio respond --input question.wav --json

# कस्टम सिस्टम प्रॉम्प्ट टेक्स्ट
.build/release/audio respond --input question.wav --system-prompt-text "You enjoy having a good conversation."

# वॉयस और सिस्टम प्रॉम्प्ट प्रीसेट चुनें
.build/release/audio respond --input question.wav --voice NATF1 --system-prompt focused

# सैंपलिंग पैरामीटर ट्यून करें
.build/release/audio respond --input question.wav --audio-temp 0.6 --repetition-penalty 1.5

# टेक्स्ट एन्ट्रॉपी अर्ली स्टॉपिंग सक्षम करें (टेक्स्ट कोलैप्स होने पर रुकता है)
.build/release/audio respond --input question.wav --entropy-threshold 1.0 --entropy-window 5

# उपलब्ध वॉयस और प्रॉम्प्ट की सूची देखें
.build/release/audio respond --list-voices
.build/release/audio respond --list-prompts
```

## CosyVoice TTS — वॉयस क्लोनिंग के साथ स्ट्रीमिंग टेक्स्ट-टू-स्पीच

### बेसिक सिंथेसिस

```swift
import CosyVoiceTTS
import AudioCommon  // WAVWriter के लिए

let model = try await CosyVoiceTTSModel.fromPretrained()
// पहली बार चलाने पर ~1.9 GB डाउनलोड होता है (LLM + DiT + HiFi-GAN वेट)

let audio = model.synthesize(text: "Hello, how are you today?", language: "english")
// आउटपुट 24kHz मोनो float सैंपल है
try WAVWriter.write(samples: audio, sampleRate: 24000, to: outputURL)
```

### स्ट्रीमिंग सिंथेसिस

```swift
// स्ट्रीमिंग: जनरेट होते ही ऑडियो चंक प्राप्त करें (पहले चंक तक ~150ms)
for try await chunk in model.synthesizeStream(text: "Hello, how are you today?", language: "english") {
    // chunk.audio: [Float], chunk.sampleRate: Int
    playAudio(chunk.audio)  // तुरंत चलाएँ
}
```

### वॉयस क्लोनिंग (CosyVoice)

CAM++ speaker encoder (192-dim, CoreML Neural Engine) का उपयोग करके स्पीकर की आवाज़ क्लोन करें:

```swift
import CosyVoiceTTS
import AudioCommon

let model = try await CosyVoiceTTSModel.fromPretrained()
let speaker = try await CamPlusPlusSpeaker.fromPretrained()
// पहले उपयोग पर ~14 MB CAM++ CoreML मॉडल डाउनलोड होता है

let refAudio = try AudioFileLoader.load(url: referenceURL, targetSampleRate: 16000)
let embedding = try speaker.embed(audio: refAudio, sampleRate: 16000)
// embedding: [Float] लंबाई 192

let audio = model.synthesize(
    text: "Hello in a cloned voice!",
    language: "english",
    speakerEmbedding: embedding
)
```

### CosyVoice TTS CLI

```bash
make build

# बेसिक सिंथेसिस
.build/release/audio speak "Hello world" --engine cosyvoice --language english --output output.wav

# वॉयस क्लोनिंग (पहले उपयोग पर CAM++ speaker encoder डाउनलोड होता है)
.build/release/audio speak "Hello world" --engine cosyvoice --voice-sample reference.wav --output cloned.wav

# वॉयस क्लोनिंग के साथ मल्टी-स्पीकर डायलॉग
.build/release/audio speak "[S1] Hello there! [S2] Hey, how are you?" \
    --engine cosyvoice --speakers s1=alice.wav,s2=bob.wav -o dialogue.wav

# इनलाइन इमोशन/स्टाइल टैग
.build/release/audio speak "(excited) Wow, amazing! (sad) But I have to go..." \
    --engine cosyvoice -o emotion.wav

# संयुक्त: डायलॉग + इमोशन + वॉयस क्लोनिंग
.build/release/audio speak "[S1] (happy) Great news! [S2] (surprised) Really?" \
    --engine cosyvoice --speakers s1=alice.wav,s2=bob.wav -o combined.wav

# कस्टम स्टाइल निर्देश
.build/release/audio speak "Hello world" --engine cosyvoice --cosy-instruct "Speak cheerfully" -o cheerful.wav

# स्ट्रीमिंग सिंथेसिस
.build/release/audio speak "Hello world" --engine cosyvoice --language english --stream --output output.wav
```

## Kokoro TTS — हल्का ऑन-डिवाइस टेक्स्ट-टू-स्पीच (iOS + macOS)

### बेसिक सिंथेसिस

```swift
import KokoroTTS
import AudioCommon  // WAVWriter के लिए

let tts = try await KokoroTTSModel.fromPretrained()
// पहली बार चलाने पर ~170 MB डाउनलोड होता है (CoreML मॉडल + voice embeddings + dictionaries)

let audio = try tts.synthesize(text: "Hello world", voice: "af_heart")
// आउटपुट 24kHz मोनो float सैंपल है
try WAVWriter.write(samples: audio, sampleRate: 24000, to: outputURL)
```

10 भाषाओं में 54 प्रीसेट वॉयस। नॉन-ऑटोरिग्रेसिव — 3-स्टेज CoreML पाइपलाइन, कोई सैंपलिंग लूप नहीं। Neural Engine पर चलता है, GPU पूरी तरह फ़्री रहता है।

### Kokoro TTS CLI

```bash
make build

# बेसिक सिंथेसिस
.build/release/audio kokoro "Hello world" --voice af_heart --output hello.wav

# भाषा चुनें
.build/release/audio kokoro "Bonjour le monde" --voice ff_siwis --language fr --output bonjour.wav

# उपलब्ध वॉयस की सूची देखें
.build/release/audio kokoro --list-voices
```

## Qwen3 Chat (ऑन-डिवाइस LLM)

```swift
import Qwen3Chat

let chat = try await Qwen3ChatModel.fromPretrained()
// पहली बार चलाने पर ~318 MB डाउनलोड होता है (INT4 CoreML मॉडल + tokenizer)

// सिंगल रिस्पॉन्स
let response = try chat.generate("What is Swift?", systemPrompt: "Answer briefly.")
print(response)

// स्ट्रीमिंग टोकन
let stream = chat.chatStream("Tell me a joke", systemPrompt: "Be funny.")
for try await token in stream {
    print(token, terminator: "")
}
```

Qwen3-0.6B CoreML के लिए INT4 क्वांटाइज़्ड। Neural Engine पर ~2 tok/s iPhone पर, ~15 tok/s M-series पर चलता है। KV cache के साथ मल्टी-टर्न वार्तालाप, थिंकिंग मोड (`<think>` टोकन), और कॉन्फ़िगर करने योग्य सैंपलिंग (temperature, top-k, top-p, repetition penalty) का समर्थन करता है।

## वॉयस एक्टिविटी डिटेक्शन (VAD) — ऑडियो में स्पीच डिटेक्ट करें

### स्ट्रीमिंग VAD (Silero)

Silero VAD v5 सब-मिलीसेकंड लेटेंसी के साथ 32ms ऑडियो चंक प्रोसेस करता है — माइक्रोफ़ोन या स्ट्रीम से रियल-टाइम स्पीच डिटेक्शन के लिए आदर्श।

```swift
import SpeechVAD

let vad = try await SileroVADModel.fromPretrained()
// या CoreML उपयोग करें (Neural Engine, कम पावर):
// let vad = try await SileroVADModel.fromPretrained(engine: .coreml)

// स्ट्रीमिंग: 512-सैंपल चंक प्रोसेस करें (16kHz पर 32ms)
let prob = vad.processChunk(samples)  // → 0.0...1.0
vad.resetState()  // अलग-अलग ऑडियो स्ट्रीम के बीच कॉल करें

// या सभी सेगमेंट एक बार में डिटेक्ट करें
let segments = vad.detectSpeech(audio: audioSamples, sampleRate: 16000)
for seg in segments {
    print("Speech: \(seg.startTime)s - \(seg.endTime)s")
}
```

### इवेंट-ड्रिवन स्ट्रीमिंग

```swift
let processor = StreamingVADProcessor(model: vad)

// किसी भी लंबाई का ऑडियो फ़ीड करें — स्पीच कन्फ़र्म होते ही इवेंट भेजे जाते हैं
let events = processor.process(samples: audioBuffer)
for event in events {
    switch event {
    case .speechStarted(let time):
        print("Speech started at \(time)s")
    case .speechEnded(let segment):
        print("Speech: \(segment.startTime)s - \(segment.endTime)s")
    }
}

// स्ट्रीम के अंत में फ़्लश करें
let final = processor.flush()
```

### VAD CLI

```bash
make build

# स्ट्रीमिंग Silero VAD (32ms chunks)
.build/release/audio vad-stream audio.wav

# CoreML बैकएंड (Neural Engine)
.build/release/audio vad-stream audio.wav --engine coreml

# कस्टम थ्रेशोल्ड के साथ
.build/release/audio vad-stream audio.wav --onset 0.6 --offset 0.4

# JSON आउटपुट
.build/release/audio vad-stream audio.wav --json

# बैच pyannote VAD (10s स्लाइडिंग विंडो)
.build/release/audio vad audio.wav
```

## स्पीकर डायराइज़ेशन — कौन कब बोला

### डायराइज़ेशन पाइपलाइन

```swift
import SpeechVAD

let pipeline = try await DiarizationPipeline.fromPretrained()
// या CoreML embeddings (Neural Engine, GPU फ़्री करता है) उपयोग करें:
// let pipeline = try await DiarizationPipeline.fromPretrained(embeddingEngine: .coreml)

let result = pipeline.diarize(audio: samples, sampleRate: 16000)
for seg in result.segments {
    print("Speaker \(seg.speakerId): [\(seg.startTime)s - \(seg.endTime)s]")
}
print("\(result.numSpeakers) speakers detected")
```

### स्पीकर एम्बेडिंग

```swift
let model = try await WeSpeakerModel.fromPretrained()
// या: let model = try await WeSpeakerModel.fromPretrained(engine: .coreml)
let embedding = model.embed(audio: samples, sampleRate: 16000)
// embedding: [Float] लंबाई 256, L2-normalized

// स्पीकर की तुलना करें
let similarity = WeSpeakerModel.cosineSimilarity(embeddingA, embeddingB)
```

### स्पीकर एक्सट्रैक्शन

रेफ़रेंस रिकॉर्डिंग का उपयोग करके केवल किसी विशिष्ट स्पीकर के सेगमेंट निकालें:

```swift
let pipeline = try await DiarizationPipeline.fromPretrained()
let targetEmb = pipeline.embeddingModel.embed(audio: enrollmentAudio, sampleRate: 16000)
let segments = pipeline.extractSpeaker(
    audio: meetingAudio, sampleRate: 16000,
    targetEmbedding: targetEmb
)
```

### Sortformer डायराइज़ेशन (एंड-टू-एंड, CoreML)

NVIDIA Sortformer अधिकतम 4 स्पीकर के लिए प्रति-फ़्रेम स्पीकर एक्टिविटी सीधे प्रेडिक्ट करता है — कोई एम्बेडिंग या क्लस्टरिंग आवश्यक नहीं। Neural Engine पर चलता है।

```swift
let diarizer = try await SortformerDiarizer.fromPretrained()
let result = diarizer.diarize(audio: samples, sampleRate: 16000, config: .default)
for seg in result.segments {
    print("Speaker \(seg.speakerId): [\(seg.startTime)s - \(seg.endTime)s]")
}
```

### डायराइज़ेशन CLI

```bash
make build

# Pyannote डायराइज़ेशन (डिफ़ॉल्ट)
.build/release/audio diarize meeting.wav

# Sortformer डायराइज़ेशन (CoreML, Neural Engine)
.build/release/audio diarize meeting.wav --engine sortformer

# CoreML embeddings (Neural Engine, केवल pyannote)
.build/release/audio diarize meeting.wav --embedding-engine coreml

# JSON आउटपुट
.build/release/audio diarize meeting.wav --json

# किसी विशिष्ट स्पीकर को एक्सट्रैक्ट करें (केवल pyannote)
.build/release/audio diarize meeting.wav --target-speaker enrollment.wav

# स्पीकर एम्बेडिंग
.build/release/audio embed-speaker enrollment.wav --json
.build/release/audio embed-speaker enrollment.wav --engine coreml
```

आर्किटेक्चर विवरण के लिए [Speaker Diarization](docs/inference/speaker-diarization.md) देखें।

## स्पीच एन्हांसमेंट — नॉइज़ सप्रेशन और ऑडियो क्लीनअप

### नॉइज़ सप्रेशन

```swift
import SpeechEnhancement
import AudioCommon  // WAVWriter के लिए

let enhancer = try await SpeechEnhancer.fromPretrained()
// पहली बार चलाने पर ~4.3 MB डाउनलोड होता है (Core ML FP16 मॉडल + सहायक डेटा)

let cleanAudio = try enhancer.enhance(audio: noisyAudio, sampleRate: 48000)
try WAVWriter.write(samples: cleanAudio, sampleRate: 48000, to: outputURL)
```

### Denoise CLI

```bash
make build

# बेसिक नॉइज़ रिमूवल
.build/release/audio denoise noisy.wav

# कस्टम आउटपुट पाथ
.build/release/audio denoise noisy.wav --output clean.wav
```

आर्किटेक्चर विवरण के लिए [Speech Enhancement](docs/inference/speech-enhancement.md) देखें।

## पाइपलाइन — कई मॉडल को मिलाएँ

सभी मॉडल साझा प्रोटोकॉल (`SpeechRecognitionModel`, `SpeechGenerationModel`, `SpeechEnhancementModel`, आदि) का पालन करते हैं और पाइपलाइन में जोड़े जा सकते हैं:

### शोरयुक्त स्पीच रिकग्निशन (DeepFilterNet + ASR)

```swift
import SpeechEnhancement
import Qwen3ASR

let enhancer = try await SpeechEnhancer.fromPretrained()
let asr = try await Qwen3ASRModel.fromPretrained()

// 48kHz पर एन्हांस करें, फिर 16kHz पर ट्रांसक्राइब करें
let clean = try enhancer.enhance(audio: noisyAudio, sampleRate: 48000)
let clean16k = AudioResampler.resample(clean, from: 48000, to: 16000)
let text = asr.transcribe(audio: clean16k, sampleRate: 16000)
```

### वॉयस-टू-वॉयस रिले (VAD + ASR + TTS)

```swift
import SpeechVAD
import Qwen3ASR
import Qwen3TTS

let vad = try await SileroVADModel.fromPretrained()
let asr = try await Qwen3ASRModel.fromPretrained()
let tts = try await Qwen3TTSModel.fromPretrained()

// स्पीच सेगमेंट डिटेक्ट करें, ट्रांसक्राइब करें, पुनः सिंथेसाइज़ करें
let segments = vad.detectSpeech(audio: audio, sampleRate: 16000)
for seg in segments {
    let chunk = Array(audio[Int(seg.startTime * 16000)..<Int(seg.endTime * 16000)])
    let text = asr.transcribe(audio: chunk, sampleRate: 16000)
    let speech = tts.synthesize(text: text, language: "english")
    // speech: 24kHz मोनो float सैंपल
}
```

### मीटिंग ट्रांसक्रिप्शन (डायराइज़ेशन + ASR)

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

पूर्ण प्रोटोकॉल संदर्भ के लिए [Shared Protocols](docs/shared-protocols.md) देखें।

## HTTP API सर्वर

एक स्टैंडअलोन HTTP सर्वर सभी मॉडल को REST और WebSocket एंडपॉइंट के माध्यम से उपलब्ध कराता है। मॉडल पहली रिक्वेस्ट पर लेज़ी लोड होते हैं।

```bash
swift build -c release
.build/release/audio-server --port 8080

# ऑडियो ट्रांसक्राइब करें
curl -X POST http://localhost:8080/transcribe --data-binary @audio.wav -H "Content-Type: audio/wav"

# टेक्स्ट-टू-स्पीच
curl -X POST http://localhost:8080/speak -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "engine": "cosyvoice"}' -o output.wav

# स्पीच-टू-स्पीच (PersonaPlex)
curl -X POST http://localhost:8080/respond --data-binary @question.wav -o response.wav

# स्पीच एन्हांसमेंट
curl -X POST http://localhost:8080/enhance --data-binary @noisy.wav -o clean.wav

# स्टार्टअप पर सभी मॉडल प्रीलोड करें
.build/release/audio-server --preload --port 8080
```

### WebSocket स्ट्रीमिंग

#### OpenAI Realtime API (`/v1/realtime`)

प्राथमिक WebSocket एंडपॉइंट [OpenAI Realtime API](https://platform.openai.com/docs/api-reference/realtime) प्रोटोकॉल को इम्प्लीमेंट करता है — सभी संदेश `type` फ़ील्ड के साथ JSON हैं, ऑडियो base64-encoded PCM16 24kHz mono है।

**Client → Server इवेंट:**

| Event | विवरण |
|-------|-------------|
| `session.update` | इंजन, भाषा, ऑडियो फ़ॉर्मेट कॉन्फ़िगर करें |
| `input_audio_buffer.append` | base64 PCM16 ऑडियो चंक भेजें |
| `input_audio_buffer.commit` | संचित ऑडियो ट्रांसक्राइब करें (ASR) |
| `input_audio_buffer.clear` | ऑडियो बफ़र साफ़ करें |
| `response.create` | TTS सिंथेसिस का अनुरोध करें |

**Server → Client इवेंट:**

| Event | विवरण |
|-------|-------------|
| `session.created` | सेशन इनिशियलाइज़ हुआ |
| `session.updated` | कॉन्फ़िगरेशन पुष्टि |
| `input_audio_buffer.committed` | ट्रांसक्रिप्शन के लिए ऑडियो कमिट हुआ |
| `conversation.item.input_audio_transcription.completed` | ASR परिणाम |
| `response.audio.delta` | Base64 PCM16 ऑडियो चंक (TTS) |
| `response.audio.done` | ऑडियो स्ट्रीमिंग पूर्ण |
| `response.done` | मेटाडेटा के साथ रिस्पॉन्स पूर्ण |
| `error` | प्रकार और संदेश के साथ एरर |

```javascript
const ws = new WebSocket('ws://localhost:8080/v1/realtime');

// ASR: ऑडियो भेजें, ट्रांसक्रिप्शन प्राप्त करें
ws.send(JSON.stringify({ type: 'input_audio_buffer.append', audio: base64PCM16 }));
ws.send(JSON.stringify({ type: 'input_audio_buffer.commit' }));
// → प्राप्त होता है: conversation.item.input_audio_transcription.completed

// TTS: टेक्स्ट भेजें, स्ट्रीम्ड ऑडियो प्राप्त करें
ws.send(JSON.stringify({
  type: 'response.create',
  response: { modalities: ['audio', 'text'], instructions: 'Hello world' }
}));
// → प्राप्त होता है: response.audio.delta (base64 chunks), response.audio.done, response.done
```

एक उदाहरण HTML क्लाइंट `Examples/websocket-client.html` पर है — सर्वर चलते समय इसे ब्राउज़र में खोलें।

सर्वर एक अलग `AudioServer` मॉड्यूल और `audio-server` एक्ज़ीक्यूटेबल है — यह मुख्य `audio` CLI में Hummingbird/WebSocket नहीं जोड़ता।

## लेटेंसी (M2 Max, 64 GB)

### ASR

| Model | Backend | RTF | 10s ऑडियो प्रोसेसिंग समय |
|-------|---------|-----|------------------------|
| Qwen3-ASR-0.6B (4-bit) | MLX | ~0.06 | ~0.6s |
| Qwen3-ASR-0.6B (INT8) | CoreML + MLX | ~0.09 | ~0.9s |
| Qwen3-ASR-1.7B (8-bit) | MLX | ~0.11 | ~1.1s |
| Parakeet-TDT-0.6B (INT4) | CoreML (Neural Engine) | ~0.12 cold, ~0.03 warm | ~1.2s / ~0.3s |
| Whisper-large-v3 | whisper.cpp (Q5_0) | ~0.10 | ~1.0s |
| Whisper-small | whisper.cpp (Q5_0) | ~0.04 | ~0.4s |

### फ़ोर्स्ड अलाइनमेंट

| Model | Framework | 20s ऑडियो | RTF |
|-------|-----------|-----------|-----|
| Qwen3-ForcedAligner-0.6B (4-bit) | MLX Swift (debug) | ~365ms | ~0.018 |

> सिंगल नॉन-ऑटोरिग्रेसिव फ़ॉरवर्ड पास — कोई सैंपलिंग लूप नहीं। ऑडियो encoder प्रमुख है (~328ms), decoder सिंगल-पास ~37ms। **रियल-टाइम से 55x तेज़।**

### TTS

| Model | Framework | छोटा (1s) | मध्यम (3s) | लंबा (6s) | स्ट्रीमिंग फ़र्स्ट-पैकेट |
|-------|-----------|-----------|-------------|------------|----------------------|
| Qwen3-TTS-0.6B (4-bit) | MLX Swift (release) | 1.6s (RTF 1.2) | 2.3s (RTF 0.7) | 3.9s (RTF 0.7) | ~120ms (1-frame) |
| Kokoro-82M | CoreML (Neural Engine) | ~1.4s (RTFx 0.7) | ~4.3s (RTFx 0.7) | ~8.6s (RTFx 0.7) | N/A (non-autoregressive) |
| Apple `AVSpeechSynthesizer` | AVFoundation | 0.08s | 0.08s | 0.17s (RTF 0.02) | N/A |

> Qwen3-TTS प्रोसोडी और इमोशन के साथ प्राकृतिक, अभिव्यक्तिपूर्ण स्पीच जनरेट करता है, **रियल-टाइम से तेज़** (RTF < 1.0) चलता है। स्ट्रीमिंग सिंथेसिस पहला ऑडियो चंक ~120ms में देता है। Kokoro-82M पूरी तरह Neural Engine पर 3-स्टेज पाइपलाइन से चलता है (RTFx ~0.7), iOS के लिए आदर्श। Apple का बिल्ट-इन TTS तेज़ है लेकिन रोबोटिक, एकस्वरीय स्पीच उत्पन्न करता है।

### PersonaPlex (स्पीच-टू-स्पीच)

| Model | Framework | ms/step | RTF | नोट्स |
|-------|-----------|---------|-----|-------|
| PersonaPlex-7B (8-bit) | MLX Swift (release) | ~112ms | ~1.4 | अनुशंसित — सुसंगत प्रतिक्रियाएँ, 4-bit से 30% तेज़ |
| PersonaPlex-7B (4-bit) | MLX Swift (release) | ~158ms | ~1.97 | अनुशंसित नहीं — आउटपुट गुणवत्ता में गिरावट |

> **8-bit का उपयोग करें।** INT8 तेज़ (112 ms/step बनाम 158 ms/step) और सुसंगत फुल-डुप्लेक्स प्रतिक्रियाएँ उत्पन्न करता है। INT4 क्वांटाइज़ेशन जनरेशन गुणवत्ता को ख़राब करता है और अस्पष्ट भाषण उत्पन्न करता है। INT8 M2 Max पर ~112ms/step पर चलता है।

### VAD और स्पीकर एम्बेडिंग

| Model | Backend | प्रति-कॉल लेटेंसी | RTF | नोट्स |
|-------|---------|-----------------|-----|-------|
| Silero-VAD-v5 | MLX | ~2.1ms / chunk | 0.065 | GPU (Metal) |
| Silero-VAD-v5 | CoreML | ~0.27ms / chunk | 0.008 | Neural Engine, **7.7x तेज़** |
| WeSpeaker ResNet34-LM | MLX | ~310ms / 20s audio | 0.016 | GPU (Metal) |
| WeSpeaker ResNet34-LM | CoreML | ~430ms / 20s audio | 0.021 | Neural Engine, GPU फ़्री करता है |

> Silero VAD CoreML Neural Engine पर MLX से 7.7x तेज़ चलता है, जो हमेशा-चालू माइक्रोफ़ोन इनपुट के लिए आदर्श है। WeSpeaker MLX GPU पर तेज़ है, लेकिन CoreML समवर्ती कार्यभार (TTS, ASR) के लिए GPU फ़्री करता है। दोनों बैकएंड समतुल्य परिणाम देते हैं।

### स्पीच एन्हांसमेंट

| Model | Backend | अवधि | लेटेंसी | RTF |
|-------|---------|----------|---------|-----|
| DeepFilterNet3 (FP16) | CoreML | 5s | 0.65s | 0.13 |
| DeepFilterNet3 (FP16) | CoreML | 10s | 1.2s | 0.12 |
| DeepFilterNet3 (FP16) | CoreML | 20s | 4.8s | 0.24 |

RTF = Real-Time Factor (कम बेहतर है, < 1.0 = रियल-टाइम से तेज़)। GRU लागत ~O(n²) से बढ़ती है।

### MLX vs CoreML

दोनों बैकएंड समतुल्य परिणाम देते हैं। अपने कार्यभार के आधार पर चुनें:

| | MLX | CoreML |
|---|---|---|
| **हार्डवेयर** | GPU (Metal shaders) | Neural Engine + CPU |
| **सर्वोत्तम** | अधिकतम थ्रूपुट, सिंगल-मॉडल कार्यभार | मल्टी-मॉडल पाइपलाइन, बैकग्राउंड कार्य |
| **पावर** | अधिक GPU उपयोग | कम पावर, GPU फ़्री करता है |
| **लेटेंसी** | बड़े मॉडल (WeSpeaker) के लिए तेज़ | छोटे मॉडल (Silero VAD) के लिए तेज़ |

**डेस्कटॉप इनफ़रेंस**: MLX डिफ़ॉल्ट है — Apple Silicon पर सबसे तेज़ सिंगल-मॉडल प्रदर्शन। GPU contention से बचने के लिए कई मॉडल एक साथ चलाते समय (जैसे, VAD + ASR + TTS) CoreML पर स्विच करें, या लैपटॉप पर बैटरी-संवेदनशील कार्यभार के लिए।

CoreML मॉडल Qwen3-ASR encoder, Silero VAD, और WeSpeaker के लिए उपलब्ध हैं। Qwen3-ASR के लिए, `--engine qwen3-coreml` उपयोग करें (हाइब्रिड: ANE पर CoreML encoder + GPU पर MLX text decoder)। VAD/embeddings के लिए, निर्माण समय पर `engine: .coreml` पास करें — इनफ़रेंस API समान है।

## सटीकता बेंचमार्क

### ASR — Word Error Rate ([विवरण](docs/benchmarks/asr-wer.md))

| Model | WER% (LibriSpeech test-clean) | RTF |
|-------|-------------------------------|-----|
| Qwen3-ASR 1.7B 8-bit | **2.35** | 0.090 |
| Qwen3-ASR 1.7B 4-bit | 2.57 | 0.045 |
| Parakeet TDT INT8 | 2.74 | 0.089 |
| Qwen3-ASR 0.6B 8-bit | 2.80 | 0.025 |

Qwen3-ASR 1.7B 8-bit तुलनीय आकार में Whisper Large v3 Turbo (2.5%) को पछाड़ता है। बहुभाषी: FLEURS पर 10 भाषाओं का बेंचमार्क।

### TTS — राउंड-ट्रिप इंटेलिजिबिलिटी ([विवरण](docs/benchmarks/tts-roundtrip.md))

| Engine | WER% | RTF |
|--------|------|-----|
| CosyVoice3 | **3.25** | 0.59 |
| Qwen3-TTS 1.7B | 3.47 | 0.79 |
| Kokoro-82M | 3.90 | 0.17 |

### VAD — स्पीच डिटेक्शन ([विवरण](docs/benchmarks/vad-detection.md))

| Engine | F1% (FLEURS) | RTF |
|--------|-------------|-----|
| FireRedVAD | **99.12** | 0.007 |
| Silero CoreML | 95.13 | 0.022 |
| Pyannote MLX | 94.86 | 0.358 |

## आर्किटेक्चर

**मॉडल:** [ASR Model](docs/models/asr-model.md), [TTS Model](docs/models/tts-model.md), [CosyVoice TTS](docs/models/cosyvoice-tts.md), [Kokoro TTS](docs/models/kokoro-tts.md), [Parakeet TDT](docs/models/parakeet-asr.md), [PersonaPlex](docs/models/personaplex.md), [FireRedVAD](docs/models/fireredvad.md)

**इनफ़रेंस:** [Qwen3-ASR](docs/inference/qwen3-asr-inference.md), [Parakeet TDT](docs/inference/parakeet-asr-inference.md), [TTS Inference](docs/inference/qwen3-tts-inference.md), [Forced Aligner](docs/inference/forced-aligner.md), [FireRedVAD](docs/inference/fireredvad.md), [Silero VAD](docs/inference/silero-vad.md), [Speaker Diarization](docs/inference/speaker-diarization.md), [Speech Enhancement](docs/inference/speech-enhancement.md), [ऑडियो प्लेबैक](docs/audio/playback.md)

**बेंचमार्क:** [ASR WER](docs/benchmarks/asr-wer.md), [TTS Round-Trip](docs/benchmarks/tts-roundtrip.md), [VAD Detection](docs/benchmarks/vad-detection.md)

**संदर्भ:** [Shared Protocols](docs/shared-protocols.md)

## कैश कॉन्फ़िगरेशन

मॉडल वेट लोकली कैश होते हैं। कैश स्थान बदलने के लिए:

```bash
export QWEN3_CACHE_DIR=/path/to/cache
```

## MLX Metal लाइब्रेरी

यदि रनटाइम पर `Failed to load the default metallib` एरर दिखता है, तो Metal shader लाइब्रेरी अनुपलब्ध है। `make build` (या मैन्युअल `swift build` के बाद `./scripts/build_mlx_metallib.sh release`) चलाएँ। यदि Metal Toolchain अनुपलब्ध है, तो पहले इसे इंस्टॉल करें:

```bash
xcodebuild -downloadComponent MetalToolchain
```

## टेस्टिंग

यूनिट टेस्ट (config, sampling, text preprocessing, timestamp correction) मॉडल डाउनलोड के बिना चलते हैं:

```bash
swift test --filter "Qwen3TTSConfigTests|SamplingTests|CosyVoiceTTSConfigTests|CamPlusPlusMelExtractorTests|PersonaPlexTests|ForcedAlignerTests/testText|ForcedAlignerTests/testTimestamp|ForcedAlignerTests/testLIS|SileroVADTests/testSilero|SileroVADTests/testReflection|SileroVADTests/testProcess|SileroVADTests/testReset|SileroVADTests/testDetect|SileroVADTests/testStreaming|SileroVADTests/testVADEvent|KokoroTTSTests"
```

इंटीग्रेशन टेस्ट के लिए मॉडल वेट आवश्यक हैं (पहली बार चलाने पर स्वतः डाउनलोड होते हैं):

```bash
# TTS राउंड-ट्रिप: टेक्स्ट सिंथेसाइज़ करें, WAV सेव करें, ASR से वापस ट्रांसक्राइब करें
swift test --filter TTSASRRoundTripTests

# केवल ASR: टेस्ट ऑडियो ट्रांसक्राइब करें
swift test --filter Qwen3ASRIntegrationTests

# Forced Aligner E2E: शब्द-स्तरीय टाइमस्टैम्प (~979 MB डाउनलोड)
swift test --filter ForcedAlignerTests/testForcedAlignerE2E

# PersonaPlex E2E: स्पीच-टू-स्पीच पाइपलाइन (~5.5 GB डाउनलोड)
PERSONAPLEX_E2E=1 swift test --filter PersonaPlexE2ETests
```

> **नोट:** MLX ऑपरेशन उपयोग करने वाले टेस्ट चलाने से पहले MLX Metal लाइब्रेरी बिल्ड होनी चाहिए।
> निर्देशों के लिए [MLX Metal लाइब्रेरी](#mlx-metal-लाइब्रेरी) देखें।

## समर्थित भाषाएँ

| Model | भाषाएँ |
|-------|-----------|
| Qwen3-ASR | 52 भाषाएँ (CN, EN, Cantonese, DE, FR, ES, JA, KO, RU, + 22 चीनी बोलियाँ, ...) |
| Parakeet TDT | 25 यूरोपीय भाषाएँ (BG, CS, DA, DE, EL, EN, ES, ET, FI, FR, HR, HU, IT, LT, LV, MT, NL, PL, PT, RO, RU, SK, SL, SV, UK) |
| Qwen3-TTS | EN, CN, DE, JA, ES, FR, KO, RU, IT, PT (+ CustomVoice द्वारा Beijing/Sichuan बोलियाँ) |
| CosyVoice TTS | CN, EN, JA, KO, DE, ES, FR, IT, RU |
| Kokoro TTS | EN (US/UK), ES, FR, HI, IT, JA, PT, CN, KO, DE |
| PersonaPlex | EN |

## तुलना

### स्पीच-टू-टेक्स्ट (ASR): speech-swift बनाम अन्य विकल्प

| | **speech-swift (Qwen3-ASR)** | **whisper.cpp** | **Apple SFSpeechRecognizer** | **Google Cloud Speech** |
|---|---|---|---|---|
| **रनटाइम** | ऑन-डिवाइस (MLX/CoreML) | ऑन-डिवाइस (CPU/GPU) | ऑन-डिवाइस या क्लाउड | केवल क्लाउड |
| **भाषाएँ** | 52 | 100+ | ~70 (ऑन-डिवाइस: सीमित) | 125+ |
| **RTF (10s ऑडियो, M2 Max)** | 0.06 (17x real-time) | 0.10 (Whisper-large-v3) | N/A | N/A |
| **स्ट्रीमिंग** | नहीं (बैच) | नहीं (बैच) | हाँ | हाँ |
| **कस्टम मॉडल** | हाँ (HuggingFace वेट स्वैप करें) | हाँ (GGML मॉडल) | नहीं | नहीं |
| **Swift API** | नेटिव async/await | C++ Swift ब्रिज के साथ | नेटिव | REST/gRPC |
| **गोपनीयता** | पूर्णतः ऑन-डिवाइस | पूर्णतः ऑन-डिवाइस | कॉन्फ़िग पर निर्भर | डेटा क्लाउड पर भेजा जाता है |
| **शब्द टाइमस्टैम्प** | हाँ (Forced Aligner) | हाँ | सीमित | हाँ |
| **लागत** | मुफ़्त (Apache 2.0) | मुफ़्त (MIT) | मुफ़्त (ऑन-डिवाइस) | प्रति मिनट भुगतान |

### टेक्स्ट-टू-स्पीच (TTS): speech-swift बनाम अन्य विकल्प

| | **speech-swift (Qwen3-TTS)** | **speech-swift (Kokoro)** | **Apple AVSpeechSynthesizer** | **ElevenLabs / Cloud TTS** |
|---|---|---|---|---|
| **गुणवत्ता** | न्यूरल, अभिव्यक्तिपूर्ण | न्यूरल, प्राकृतिक | रोबोटिक, एकस्वरीय | न्यूरल, सर्वोच्च गुणवत्ता |
| **रनटाइम** | ऑन-डिवाइस (MLX) | ऑन-डिवाइस (CoreML) | ऑन-डिवाइस | केवल क्लाउड |
| **स्ट्रीमिंग** | हाँ (पहला चंक ~120ms) | नहीं (3-स्टेज पाइपलाइन) | नहीं | हाँ |
| **वॉयस क्लोनिंग** | हाँ | नहीं | नहीं | हाँ |
| **वॉयस** | 9 बिल्ट-इन + कोई भी क्लोन करें | 54 प्रीसेट वॉयस | ~50 सिस्टम वॉयस | 1000+ |
| **भाषाएँ** | 10 | 10 | 60+ | 30+ |
| **iOS सपोर्ट** | केवल macOS | iOS + macOS | iOS + macOS | कोई भी (API) |
| **लागत** | मुफ़्त (Apache 2.0) | मुफ़्त (Apache 2.0) | मुफ़्त | प्रति अक्षर भुगतान |

### speech-swift कब उपयोग करें

- **गोपनीयता-महत्वपूर्ण ऐप्स** — मेडिकल, कानूनी, एंटरप्राइज़ जहाँ ऑडियो डिवाइस से बाहर नहीं जा सकता
- **ऑफ़लाइन उपयोग** — प्रारंभिक मॉडल डाउनलोड के बाद इंटरनेट कनेक्शन आवश्यक नहीं
- **लागत-संवेदनशील** — कोई प्रति-मिनट या प्रति-अक्षर API शुल्क नहीं
- **Apple Silicon ऑप्टिमाइज़ेशन** — विशेष रूप से M-series GPU (Metal) और Neural Engine के लिए बनाया गया
- **पूर्ण पाइपलाइन** — एक ही Swift पैकेज में ASR + TTS + VAD + डायराइज़ेशन + एन्हांसमेंट को मिलाएँ

## अक्सर पूछे जाने वाले प्रश्न

**क्या speech-swift iOS पर काम करता है?**
Kokoro TTS, Qwen3.5-Chat (CoreML), Silero VAD, Parakeet ASR, DeepFilterNet3, और WeSpeaker सभी CoreML के माध्यम से Neural Engine पर iOS 17+ पर चलते हैं। MLX-आधारित मॉडल (Qwen3-ASR, Qwen3-TTS, Qwen3.5-Chat MLX, PersonaPlex) के लिए Apple Silicon पर macOS 14+ आवश्यक है।

**क्या इंटरनेट कनेक्शन आवश्यक है?**
केवल HuggingFace से प्रारंभिक मॉडल डाउनलोड के लिए (स्वचालित, `~/Library/Caches/qwen3-speech/` में कैश)। उसके बाद, सारा इनफ़रेंस बिना नेटवर्क एक्सेस के पूर्णतः ऑफ़लाइन चलता है।

**speech-swift की तुलना Whisper से कैसे होती है?**
Qwen3-ASR-0.6B M2 Max पर RTF 0.06 प्राप्त करता है — whisper.cpp (RTF 0.10) के माध्यम से Whisper-large-v3 से 40% तेज़ — 52 भाषाओं में तुलनीय सटीकता के साथ। speech-swift नेटिव Swift async/await API प्रदान करता है, जबकि whisper.cpp को C++ ब्रिज की आवश्यकता होती है।

**क्या मैं इसे व्यावसायिक ऐप में उपयोग कर सकता हूँ?**
हाँ। speech-swift Apache 2.0 के तहत लाइसेंस प्राप्त है। अंतर्निहित मॉडल वेट के अपने लाइसेंस हैं (प्रत्येक मॉडल का HuggingFace पेज देखें)।

**कौन से Apple Silicon चिप समर्थित हैं?**
सभी M-series चिप: M1, M2, M3, M4 और उनके Pro/Max/Ultra वेरिएंट। macOS 14+ (Sonoma) या iOS 17+ आवश्यक है।

**कितनी मेमोरी चाहिए?**
~3 MB (Silero VAD) से ~6.5 GB (PersonaPlex 7B) तक। Kokoro TTS ~500 MB उपयोग करता है, Qwen3-ASR ~2.2 GB। पूर्ण विवरण के लिए [मेमोरी आवश्यकताएँ](#मेमोरी-आवश्यकताएँ) तालिका देखें।

**क्या कई मॉडल एक साथ चला सकते हैं?**
हाँ। contention से बचने के लिए Neural Engine पर CoreML मॉडल GPU पर MLX मॉडल के साथ उपयोग करें — उदाहरण के लिए, Silero VAD (CoreML) + Qwen3-ASR (MLX) + Qwen3-TTS (MLX)।

**क्या REST API उपलब्ध है?**
हाँ। `audio-server` बाइनरी HTTP REST और WebSocket एंडपॉइंट के माध्यम से सभी मॉडल उपलब्ध कराती है, जिसमें `/v1/realtime` पर OpenAI Realtime API-संगत WebSocket शामिल है।

## योगदान

हम योगदान का स्वागत करते हैं! चाहे वह बग फ़िक्स हो, नया मॉडल इंटीग्रेशन हो, या डॉक्यूमेंटेशन सुधार — PRs की सराहना की जाती है।

**शुरू करने के लिए:**
1. रेपो फ़ोर्क करें और फ़ीचर ब्रांच बनाएँ
2. कंपाइल करने के लिए `make build` चलाएँ (Xcode + Metal Toolchain आवश्यक)
3. टेस्ट सूट चलाने के लिए `make test`
4. `main` के विरुद्ध PR खोलें

## लाइसेंस

Apache 2.0
