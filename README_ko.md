# Speech Swift

MLX Swift와 CoreML 기반의 Apple Silicon용 AI 음성 모델.

📖 Read in: [English](README.md) · [中文](README_zh.md) · [日本語](README_ja.md) · [한국어](README_ko.md) · [Español](README_es.md) · [Deutsch](README_de.md) · [Français](README_fr.md) · [हिन्दी](README_hi.md) · [Português](README_pt.md) · [Русский](README_ru.md)

Mac과 iOS를 위한 온디바이스 음성 인식, 합성 및 이해. Apple Silicon에서 완전히 로컬로 실행 — 클라우드 없이, API 키 없이, 데이터가 기기 밖으로 나가지 않습니다.

[Homebrew로 설치](#homebrew)하거나 Swift Package 의존성으로 추가하세요.

**[문서](https://soniqo.audio)** · **[HuggingFace 모델](https://huggingface.co/aufklarer)** · **[블로그](https://blog.ivan.digital)**

- **Qwen3-ASR** — 음성-텍스트 변환 / 음성 인식 (자동 음성 인식, 52개 언어)
- **Parakeet TDT** — CoreML을 통한 음성-텍스트 변환 (Neural Engine, NVIDIA FastConformer + TDT 디코더, 25개 언어)
- **Qwen3-ForcedAligner** — 단어 수준 타임스탬프 정렬 (오디오 + 텍스트 → 타임스탬프)
- **Qwen3-TTS** — 텍스트-음성 변환 (최고 품질, 스트리밍, 커스텀 화자, 10개 언어)
- **CosyVoice TTS** — 스트리밍, 음성 복제, 다화자 대화, 감정 태그를 지원하는 텍스트-음성 변환 (9개 언어, DiT flow matching, CAM++ 화자 인코더)
- **Kokoro TTS** — 온디바이스 텍스트-음성 변환 (82M 파라미터, CoreML/Neural Engine, 54개 음색, iOS 지원, 10개 언어)
- **Qwen3.5-Chat** — 온디바이스 LLM 채팅 (0.8B, MLX + CoreML, INT4 + CoreML INT8, DeltaNet 하이브리드, 스트리밍 토큰)
- **PersonaPlex** — 전이중 음성-음성 대화 (7B, 오디오 입력 → 오디오 출력, 18개 음색 프리셋)
- **DeepFilterNet3** — 음성 향상 / 노이즈 억제 (2.1M 파라미터, 실시간 48kHz)
- **FireRedVAD** — 오프라인 음성 활동 감지 (DFSMN, CoreML, 100개 이상 언어, 97.6% F1)
- **Silero VAD** — 스트리밍 음성 활동 감지 (32ms 청크, 밀리초 미만 지연 시간)
- **Pyannote VAD** — 오프라인 음성 활동 감지 (10초 윈도우, 다화자 중첩)
- **Speaker Diarization** — 누가 언제 말했는지 (Pyannote 세그멘테이션 + 활동 기반 화자 체이닝, 또는 Neural Engine 상의 엔드투엔드 Sortformer)
- **Speaker Embeddings** — 화자 검증 및 식별 (WeSpeaker ResNet34, 256차원 벡터)

논문: [Qwen3-ASR](https://arxiv.org/abs/2601.21337) (Alibaba), [Qwen3-TTS](https://arxiv.org/abs/2601.15621) (Alibaba), [Qwen3](https://arxiv.org/abs/2505.09388) (Alibaba), [Parakeet TDT](https://arxiv.org/abs/2304.06795) (NVIDIA), [CosyVoice 3](https://arxiv.org/abs/2505.17589) (Alibaba), [Kokoro](https://arxiv.org/abs/2301.01695) (StyleTTS 2), [PersonaPlex](https://arxiv.org/abs/2602.06053) (NVIDIA), [Mimi](https://arxiv.org/abs/2410.00037) (Kyutai), [Sortformer](https://arxiv.org/abs/2409.06656) (NVIDIA)

## 로드맵

[로드맵 토론](https://github.com/soniqo/speech-swift/discussions/81)에서 계획된 내용을 확인하세요 — 댓글과 제안을 환영합니다!

## 소식

- **2026년 3월 20일** — [600M 모델로 Mac에서 Whisper Large v3를 능가하다](https://blog.ivan.digital/we-beat-whisper-large-v3-with-a-600m-model-running-entirely-on-your-mac-20e6ce191174)
- **2026년 2월 26일** — [Apple Silicon에서의 화자 분리 및 음성 활동 감지 — MLX 기반 네이티브 Swift](https://blog.ivan.digital/speaker-diarization-and-voice-activity-detection-on-apple-silicon-native-swift-with-mlx-92ea0c9aca0f)
- **2026년 2월 23일** — [Apple Silicon에서 NVIDIA PersonaPlex 7B — MLX 기반 네이티브 Swift로 전이중 음성-음성 변환](https://blog.ivan.digital/nvidia-personaplex-7b-on-apple-silicon-full-duplex-speech-to-speech-in-native-swift-with-mlx-0aa5276f2e23)
- **2026년 2월 12일** — [Qwen3-ASR Swift: Apple Silicon용 온디바이스 ASR + TTS — 아키텍처 및 벤치마크](https://blog.ivan.digital/qwen3-asr-swift-on-device-asr-tts-for-apple-silicon-architecture-and-benchmarks-27cbf1e4463f)

## 모델

| 모델 | 태스크 | 스트리밍 | 언어 | 크기 |
|-------|------|-----------|-----------|-------|
| Qwen3-ASR-0.6B | 음성 → 텍스트 | 아니오 | 52개 언어 | [4-bit](https://huggingface.co/aufklarer/Qwen3-ASR-0.6B-MLX-4bit) 680 MB · [8-bit](https://huggingface.co/aufklarer/Qwen3-ASR-0.6B-MLX-8bit) 1.0 GB · [CoreML](https://huggingface.co/aufklarer/Qwen3-ASR-CoreML) 180 MB |
| Qwen3-ASR-1.7B | 음성 → 텍스트 | 아니오 | 52개 언어 | [4-bit](https://huggingface.co/aufklarer/Qwen3-ASR-1.7B-MLX-4bit) 2.1 GB · [8-bit](https://huggingface.co/aufklarer/Qwen3-ASR-1.7B-MLX-8bit) 3.2 GB |
| Parakeet-TDT-0.6B | 음성 → 텍스트 | 아니오 | 25개 유럽 언어 | [CoreML INT4](https://huggingface.co/aufklarer/Parakeet-TDT-v3-CoreML-INT4) 315 MB · [CoreML INT8](https://huggingface.co/aufklarer/Parakeet-TDT-v3-CoreML-INT8) 500 MB |
| Qwen3-ForcedAligner-0.6B | 오디오 + 텍스트 → 타임스탬프 | 아니오 | 다국어 | [4-bit](https://huggingface.co/aufklarer/Qwen3-ForcedAligner-0.6B-4bit) 979 MB · [8-bit](https://huggingface.co/aufklarer/Qwen3-ForcedAligner-0.6B-8bit) 1.4 GB · [CoreML INT4](https://huggingface.co/aufklarer/Qwen3-ForcedAligner-0.6B-CoreML-INT4) 630 MB · [CoreML INT8](https://huggingface.co/aufklarer/Qwen3-ForcedAligner-0.6B-CoreML-INT8) 1.0 GB |
| Qwen3-TTS-0.6B Base | 텍스트 → 음성 | 예 (~120ms) | 10개 언어 | [4-bit](https://huggingface.co/aufklarer/Qwen3-TTS-12Hz-0.6B-Base-MLX-4bit) 1.7 GB · [8-bit](https://huggingface.co/aufklarer/Qwen3-TTS-12Hz-0.6B-Base-MLX-8bit) 2.4 GB |
| Qwen3-TTS-0.6B CustomVoice | 텍스트 → 음성 | 예 (~120ms) | 10개 언어 | [4-bit](https://huggingface.co/aufklarer/Qwen3-TTS-12Hz-0.6B-CustomVoice-MLX-4bit) 1.7 GB |
| Qwen3-TTS-1.7B Base | 텍스트 → 음성 | 예 (~120ms) | 10개 언어 | [4-bit](https://huggingface.co/aufklarer/Qwen3-TTS-12Hz-1.7B-Base-MLX-4bit) 3.2 GB · [8-bit](https://huggingface.co/aufklarer/Qwen3-TTS-12Hz-1.7B-Base-MLX-8bit) 4.8 GB |
| CosyVoice3-0.5B | 텍스트 → 음성 | 예 (~150ms) | 9개 언어 | [4-bit](https://huggingface.co/aufklarer/CosyVoice3-0.5B-MLX-4bit) 1.2 GB |
| Kokoro-82M | 텍스트 → 음성 | 아니오 | 10개 언어 | [CoreML](https://huggingface.co/aufklarer/Kokoro-82M-CoreML) ~170 MB |
| Qwen3.5-0.8B Chat | Text → Text (LLM) | Yes (streaming) | Multi | [MLX INT4](https://huggingface.co/aufklarer/Qwen3.5-0.8B-Chat-MLX) 418 MB · [CoreML INT8](https://huggingface.co/aufklarer/Qwen3.5-0.8B-Chat-CoreML) 981 MB |
| PersonaPlex-7B | 음성 → 음성 | 예 (~2초 청크) | EN | [4-bit](https://huggingface.co/aufklarer/PersonaPlex-7B-MLX-4bit) 4.9 GB · [8-bit](https://huggingface.co/aufklarer/PersonaPlex-7B-MLX-8bit) 9.1 GB |
| FireRedVAD | 음성 활동 감지 | 아니오 (오프라인) | 100개 이상 언어 | [CoreML](https://huggingface.co/aufklarer/FireRedVAD-CoreML) ~1.2 MB |
| Silero-VAD-v5 | 음성 활동 감지 | 예 (32ms 청크) | 언어 무관 | [MLX](https://huggingface.co/aufklarer/Silero-VAD-v5-MLX) · [CoreML](https://huggingface.co/aufklarer/Silero-VAD-v5-CoreML) ~1.2 MB |
| Pyannote-Segmentation-3.0 | VAD + 화자 세그멘테이션 | 아니오 (10초 윈도우) | 언어 무관 | [MLX](https://huggingface.co/aufklarer/Pyannote-Segmentation-MLX) ~5.7 MB |
| DeepFilterNet3 | 음성 향상 | 예 (10ms 프레임) | 언어 무관 | [CoreML FP16](https://huggingface.co/aufklarer/DeepFilterNet3-CoreML) ~4.2 MB |
| WeSpeaker-ResNet34-LM | 화자 임베딩 (256차원) | 아니오 | 언어 무관 | [MLX](https://huggingface.co/aufklarer/WeSpeaker-ResNet34-LM-MLX) · [CoreML](https://huggingface.co/aufklarer/WeSpeaker-ResNet34-LM-CoreML) ~25 MB |
| CAM++ | 화자 임베딩 (192차원) | 아니오 | 언어 무관 | [CoreML](https://huggingface.co/aufklarer/CamPlusPlus-Speaker-CoreML) ~14 MB |
| Sortformer | 화자 분리 (엔드투엔드) | 예 (청크) | 언어 무관 | [CoreML](https://huggingface.co/aufklarer/Sortformer-Diarization-CoreML) ~240 MB |

### 메모리 요구 사항

가중치 메모리는 모델 파라미터가 소비하는 GPU (MLX) 또는 ANE (CoreML) 메모리입니다. 최대 추론 메모리에는 KV 캐시, 활성화 값, 중간 텐서가 포함됩니다.

| 모델 | 가중치 메모리 | 최대 추론 메모리 |
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

### TTS 선택 가이드

- **Qwen3-TTS**: 최고 품질, 스트리밍 (~120ms), 9개 내장 화자, 10개 언어, 배치 합성
- **CosyVoice TTS**: 스트리밍 (~150ms), 9개 언어, 음성 복제 (CAM++ 화자 인코더), 다화자 대화 (`[S1] ... [S2] ...`), 인라인 감정/스타일 태그 (`(happy)`, `(whispers)`), DiT flow matching + HiFi-GAN 보코더
- **Kokoro TTS**: iOS에 적합한 경량 TTS (82M 파라미터), CoreML/Neural Engine, 54개 음색, 10개 언어, 비자기회귀 (3단계 파이프라인)
- **PersonaPlex**: 전이중 음성-음성 (오디오 입력 → 오디오 출력), 스트리밍 (~2초 청크), 18개 음색 프리셋, Moshi 아키텍처 기반

## 설치

### Homebrew

네이티브 ARM Homebrew (`/opt/homebrew`)가 필요합니다. Rosetta/x86_64 Homebrew는 지원되지 않습니다.

```bash
brew tap soniqo/speech https://github.com/soniqo/speech-swift
brew install speech
```

사용 방법:

```bash
audio transcribe recording.wav
audio speak "Hello world"
audio speak "Hallo Welt" --engine cosyvoice --language german
audio respond --input question.wav --transcript
```

> 마이크 입력을 사용한 대화형 음성 대화는 **[PersonaPlexDemo](Examples/PersonaPlexDemo/)** 를 참조하세요.

### Swift Package Manager

`Package.swift`에 추가하세요:

```swift
dependencies: [
    .package(url: "https://github.com/soniqo/speech-swift", branch: "main")
]
```

필요한 모듈을 가져오세요:

```swift
import Qwen3ASR      // 음성 인식 (MLX)
import ParakeetASR   // 음성 인식 (CoreML)
import Qwen3TTS      // 텍스트-음성 변환 (Qwen3)
import CosyVoiceTTS  // 텍스트-음성 변환 (스트리밍)
import KokoroTTS     // 텍스트-음성 변환 (CoreML, iOS 지원)
import Qwen3Chat     // 온디바이스 LLM 채팅 (CoreML)
import PersonaPlex   // 음성-음성 (전이중)
import SpeechVAD          // 음성 활동 감지 (pyannote + Silero)
import SpeechEnhancement  // 노이즈 억제 (DeepFilterNet3)
import AudioCommon        // 공용 유틸리티
```

### 요구 사항

- Swift 5.9+
- macOS 14+ 또는 iOS 17+
- Apple Silicon (M1/M2/M3/M4)
- Xcode 15+ (Metal Toolchain 포함 — 없으면 `xcodebuild -downloadComponent MetalToolchain` 실행)

### 소스에서 빌드

```bash
git clone https://github.com/soniqo/speech-swift
cd speech-swift
make build
```

이 명령은 Swift 패키지 **및** MLX Metal 셰이더 라이브러리를 한 번에 컴파일합니다. Metal 라이브러리(`mlx.metallib`)는 GPU 추론에 필수이며 — 없으면 런타임에 `Failed to load the default metallib` 오류가 발생합니다.

디버그 빌드: `make debug`. 단위 테스트 실행: `make test`.

## 음성 어시스턴트 사용해 보기

**[PersonaPlexDemo](Examples/PersonaPlexDemo/)** 는 바로 실행할 수 있는 macOS 음성 어시스턴트입니다 — 탭하여 말하면 실시간으로 음성 응답을 받을 수 있습니다. 자동 음성 감지를 위한 Silero VAD가 포함된 마이크 입력, 전사를 위한 Qwen3-ASR, 음성-음성 생성을 위한 PersonaPlex 7B를 사용합니다. 18개 음색 프리셋과 내면 독백 트랜스크립트 표시를 갖춘 멀티턴 대화를 지원합니다.

```bash
make build  # 리포 루트에서 — MLX metallib 포함 전체 빌드
cd Examples/PersonaPlexDemo
# .app 번들 안내는 Examples/PersonaPlexDemo/README.md를 참조하세요
```

> M2 Max에서 RTF ~0.94 (실시간보다 빠름). 모델은 첫 실행 시 자동 다운로드됩니다 (~5.5 GB PersonaPlex + ~400 MB ASR).

## 데모 앱

- **[PersonaPlexDemo](Examples/PersonaPlexDemo/)** — 대화형 음성 어시스턴트 (마이크 입력, VAD, 멀티턴). 위 내용 참조.
- **[SpeechDemo](Examples/SpeechDemo/)** — 받아쓰기 (Parakeet TDT / 언어 자동 감지가 포함된 Qwen3-ASR) 및 텍스트-음성 합성 (Qwen3-TTS)이 탭 인터페이스로 제공됩니다.

macOS `.app` 번들로 빌드하고 실행하세요 — 각 데모의 README에서 안내를 확인할 수 있습니다.

## 음성-텍스트 변환 (ASR) — Swift로 오디오 전사하기

### 기본 전사

```swift
import Qwen3ASR

// 기본값: 0.6B 모델
let model = try await Qwen3ASRModel.fromPretrained()

// 더 높은 정확도를 위해 1.7B 모델 사용
let model = try await Qwen3ASRModel.fromPretrained(
    modelId: "aufklarer/Qwen3-ASR-1.7B-MLX-8bit"
)

// 오디오는 아무 샘플레이트나 가능 — 내부적으로 16kHz로 자동 리샘플링
let transcription = model.transcribe(audio: audioSamples, sampleRate: 16000)
print(transcription)
```

### CoreML 인코더 (Neural Engine)

하이브리드 모드: Neural Engine의 CoreML 인코더 + GPU의 MLX 텍스트 디코더. 저전력이며 인코더 패스에서 GPU를 확보합니다.

```swift
import Qwen3ASR

let encoder = try await CoreMLASREncoder.fromPretrained()
let model = try await Qwen3ASRModel.fromPretrained()
let text = try model.transcribe(audio: audioSamples, sampleRate: 16000, coremlEncoder: encoder)
```

INT8 (180 MB, 기본값) 및 INT4 (90 MB) 변형을 사용할 수 있습니다. INT8 권장 (FP32 대비 코사인 유사도 > 0.999).

### Parakeet TDT (CoreML)

```swift
import ParakeetASR

let model = try await ParakeetASRModel.fromPretrained()
let transcription = model.transcribe(audio: audioSamples, sampleRate: 16000)
```

CoreML을 통해 Neural Engine에서 실행됩니다 — GPU를 다른 동시 작업에 사용할 수 있습니다. 25개 유럽 언어, ~315 MB.

### ASR CLI

```bash
make build  # 또는: swift build -c release && ./scripts/build_mlx_metallib.sh release

# 기본값 (Qwen3-ASR 0.6B, MLX)
.build/release/audio transcribe audio.wav

# 1.7B 모델 사용
.build/release/audio transcribe audio.wav --model 1.7B

# CoreML 인코더 (Neural Engine + MLX 디코더)
.build/release/audio transcribe --engine qwen3-coreml audio.wav

# Parakeet TDT (CoreML, Neural Engine)
.build/release/audio transcribe --engine parakeet audio.wav
```

## 강제 정렬

### 단어 수준 타임스탬프

```swift
import Qwen3ASR

let aligner = try await Qwen3ForcedAligner.fromPretrained()
// 첫 실행 시 ~979 MB 다운로드

let aligned = aligner.align(
    audio: audioSamples,
    text: "Can you guarantee that the replacement part will be shipped tomorrow?",
    sampleRate: 24000
)

for word in aligned {
    print("[\(String(format: "%.2f", word.startTime))s - \(String(format: "%.2f", word.endTime))s] \(word.text)")
}
```

### 강제 정렬 CLI

```bash
swift build -c release

# 제공된 텍스트로 정렬
.build/release/audio align audio.wav --text "Hello world"

# 먼저 전사한 후 정렬
.build/release/audio align audio.wav
```

출력:
```
[0.12s - 0.45s] Can
[0.45s - 0.72s] you
[0.72s - 1.20s] guarantee
...
```

비자기회귀 — 3단계 파이프라인, 샘플링 루프 없음. 아키텍처 상세 내용은 [Forced Aligner](docs/inference/forced-aligner.md)를 참조하세요.

## 텍스트-음성 변환 (TTS) — Swift로 음성 생성하기

### 기본 합성

```swift
import Qwen3TTS
import AudioCommon  // WAVWriter 사용

let model = try await Qwen3TTSModel.fromPretrained()
// 첫 실행 시 ~1.7 GB 다운로드 (모델 + 코덱 가중치)
let audio = model.synthesize(text: "Hello world", language: "english")
// 출력은 24kHz 모노 float 샘플
try WAVWriter.write(samples: audio, sampleRate: 24000, to: outputURL)
```

### TTS CLI

```bash
make build
.build/release/audio speak "Hello world" --output output.wav --language english
```

### 커스텀 음성 / 화자 선택

**CustomVoice** 모델 변형은 9개의 내장 화자 음색과 톤/스타일 제어를 위한 자연어 지시를 지원합니다. CustomVoice 모델 ID를 전달하여 로드하세요:

```swift
import Qwen3TTS

// CustomVoice 모델 로드 (첫 실행 시 ~1.7 GB 다운로드)
let model = try await Qwen3TTSModel.fromPretrained(
    modelId: TTSModelVariant.customVoice.rawValue
)

// 특정 화자로 합성
let audio = model.synthesize(text: "Hello world", language: "english", speaker: "vivian")

// 사용 가능한 화자 목록
print(model.availableSpeakers)  // ["aiden", "dylan", "eric", ...]
```

CLI:

```bash
# 화자를 지정한 CustomVoice 모델 사용
.build/release/audio speak "Hello world" --model customVoice --speaker vivian --output vivian.wav

# 사용 가능한 화자 목록
.build/release/audio speak --model customVoice --list-speakers
```

### 음성 복제 (Base 모델)

참조 오디오 파일에서 화자의 음성을 복제합니다. 두 가지 모드:

**ICL 모드** (권장) — 참조 오디오를 트랜스크립트와 함께 코덱 토큰으로 인코딩합니다. 더 높은 품질, 안정적인 EOS:

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

**X-vector 모드** — 화자 임베딩만 사용, 트랜스크립트 불필요하지만 품질이 낮음:

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

### 톤 / 스타일 지시 (CustomVoice 전용)

CustomVoice 모델은 발화 스타일, 톤, 감정, 속도를 제어하기 위한 자연어 `instruct` 파라미터를 지원합니다. 지시는 ChatML 형식으로 모델 입력 앞에 추가됩니다.

```swift
// 밝은 톤
let audio = model.synthesize(
    text: "Welcome to our store!",
    language: "english",
    speaker: "ryan",
    instruct: "Speak in a cheerful, upbeat tone"
)

// 느리고 진지하게
let audio = model.synthesize(
    text: "We regret to inform you...",
    language: "english",
    speaker: "aiden",
    instruct: "Read this slowly and solemnly"
)

// 속삭이기
let audio = model.synthesize(
    text: "Can you keep a secret?",
    language: "english",
    speaker: "vivian",
    instruct: "Whisper this softly"
)
```

CLI:

```bash
# 스타일 지시 포함
.build/release/audio speak "Good morning!" --model customVoice --speaker ryan \
    --instruct "Speak in a cheerful, upbeat tone" --output cheerful.wav

# CustomVoice 사용 시 기본 지시 ("Speak naturally.")가 자동 적용됨
.build/release/audio speak "Hello world" --model customVoice --speaker ryan --output natural.wav
```

CustomVoice 모델에서 `--instruct`를 제공하지 않으면 `"Speak naturally."`가 자동으로 적용되어 장황한 출력을 방지합니다. Base 모델은 instruct를 지원하지 않습니다.

### 배치 합성

단일 배치 순전파로 여러 텍스트를 합성하여 처리량을 높입니다:

```swift
let texts = ["Good morning everyone.", "The weather is nice today.", "Please open the window."]
let audioList = model.synthesizeBatch(texts: texts, language: "english", maxBatchSize: 4)
// audioList[i]는 texts[i]에 대한 24kHz 모노 float 샘플
for (i, audio) in audioList.enumerated() {
    try WAVWriter.write(samples: audio, sampleRate: 24000, to: URL(fileURLWithPath: "output_\(i).wav"))
}
```

#### 배치 CLI

```bash
# 한 줄에 하나의 텍스트가 있는 파일 생성
echo "Hello world.\nGoodbye world." > texts.txt
.build/release/audio speak --batch-file texts.txt --output output.wav --batch-size 4
# output_0.wav, output_1.wav, ... 생성
```

> 배치 모드는 항목 간에 모델 가중치 로드를 분산합니다. Apple Silicon에서 B=4 기준 ~1.5-2.5배 처리량 향상을 기대할 수 있습니다. 비슷한 길이의 오디오를 생성하는 텍스트에서 최적의 결과를 얻을 수 있습니다.

### 샘플링 옵션

```swift
let config = SamplingConfig(temperature: 0.9, topK: 50, repetitionPenalty: 1.05)
let audio = model.synthesize(text: "Hello", language: "english", sampling: config)
```

### 스트리밍 합성

첫 패킷 지연 시간을 줄이기 위해 오디오 청크를 점진적으로 출력합니다:

```swift
let stream = model.synthesizeStream(
    text: "Hello, this is streaming synthesis.",
    language: "english",
    streaming: .lowLatency  // 첫 오디오 청크까지 ~120ms
)

for try await chunk in stream {
    // chunk.samples: [Float] PCM @ 24kHz
    // chunk.isFinal: 마지막 청크에서 true
    playAudio(chunk.samples)
}
```

CLI:

```bash
# 기본 스트리밍 (3프레임 첫 청크, ~225ms 지연)
.build/release/audio speak "Hello world" --stream

# 저지연 (1프레임 첫 청크, ~120ms 지연)
.build/release/audio speak "Hello world" --stream --first-chunk-frames 1
```

## 음성-음성 — 전이중 음성 대화

> 마이크 입력이 포함된 대화형 음성 어시스턴트는 **[PersonaPlexDemo](Examples/PersonaPlexDemo/)** 를 참조하세요 — 탭하여 말하기, 자동 음성 감지로 멀티턴 대화를 지원합니다.

### 음성-음성 변환

```swift
import PersonaPlex
import AudioCommon  // WAVWriter, AudioFileLoader 사용

let model = try await PersonaPlexModel.fromPretrained()
// 첫 실행 시 ~5.5 GB 다운로드 (temporal 4-bit + depformer + Mimi 코덱 + 음색 프리셋)

let audio = try AudioFileLoader.load(url: inputURL, targetSampleRate: 24000)
let (response, textTokens) = model.respond(userAudio: audio, voice: .NATM0)
// response: 24kHz 모노 float 샘플
// textTokens: 모델의 내면 독백 (SentencePiece 토큰 ID)
try WAVWriter.write(samples: response.audio, sampleRate: 24000, to: outputURL)
```

### 내면 독백 (텍스트 출력)

PersonaPlex는 오디오와 함께 텍스트 토큰을 생성합니다 — 모델의 내부 추론입니다. 내장된 SentencePiece 디코더로 디코딩하세요:

```swift
let decoder = try SentencePieceDecoder(modelPath: "tokenizer_spm_32k_3.model")
let transcript = decoder.decode(textTokens)
print(transcript)  // 예: "Sure, I can help you with that..."
```

### 스트리밍 음성-음성

```swift
// 생성되는 대로 오디오 청크 수신 (청크당 ~2초)
let stream = model.respondStream(userAudio: audio, voice: .NATM0)
for try await chunk in stream {
    playAudio(chunk.samples)  // 즉시 재생, 24kHz 모노
    // chunk.textTokens에 이 청크의 텍스트 포함; 마지막 청크에 모든 토큰 포함
    if chunk.isFinal { break }
}
```

### 음색 선택

18개 음색 프리셋 사용 가능:
- **자연스러운 여성**: NATF0, NATF1, NATF2, NATF3
- **자연스러운 남성**: NATM0, NATM1, NATM2, NATM3
- **다양한 여성**: VARF0, VARF1, VARF2, VARF3, VARF4
- **다양한 남성**: VARM0, VARM1, VARM2, VARM3, VARM4

### 시스템 프롬프트

시스템 프롬프트는 모델의 대화 동작을 조정합니다. 임의의 커스텀 프롬프트를 일반 문자열로 전달할 수 있습니다:

```swift
// 커스텀 시스템 프롬프트 (자동 토큰화)
let response = model.respond(
    userAudio: audio,
    voice: .NATM0,
    systemPrompt: "You enjoy having a good conversation."
)

// 또는 프리셋 사용
let response = model.respond(
    userAudio: audio,
    voice: .NATM0,
    systemPromptTokens: SystemPromptPreset.customerService.tokens
)
```

사용 가능한 프리셋: `focused` (기본값), `assistant`, `customerService`, `teacher`.

### PersonaPlex CLI

```bash
make build

# 기본 음성-음성
.build/release/audio respond --input question.wav --output response.wav

# 트랜스크립트 포함 (내면 독백 텍스트 디코딩)
.build/release/audio respond --input question.wav --transcript

# JSON 출력 (오디오 경로, 트랜스크립트, 지연 시간 메트릭)
.build/release/audio respond --input question.wav --json

# 커스텀 시스템 프롬프트 텍스트
.build/release/audio respond --input question.wav --system-prompt-text "You enjoy having a good conversation."

# 음색 및 시스템 프롬프트 프리셋 선택
.build/release/audio respond --input question.wav --voice NATF1 --system-prompt focused

# 샘플링 파라미터 조정
.build/release/audio respond --input question.wav --audio-temp 0.6 --repetition-penalty 1.5

# 텍스트 엔트로피 조기 중단 활성화 (텍스트가 수렴하면 중단)
.build/release/audio respond --input question.wav --entropy-threshold 1.0 --entropy-window 5

# 사용 가능한 음색 및 프롬프트 목록
.build/release/audio respond --list-voices
.build/release/audio respond --list-prompts
```

## CosyVoice TTS — 음성 복제를 지원하는 스트리밍 텍스트-음성 변환

### 기본 합성

```swift
import CosyVoiceTTS
import AudioCommon  // WAVWriter 사용

let model = try await CosyVoiceTTSModel.fromPretrained()
// 첫 실행 시 ~1.9 GB 다운로드 (LLM + DiT + HiFi-GAN 가중치)

let audio = model.synthesize(text: "Hello, how are you today?", language: "english")
// 출력은 24kHz 모노 float 샘플
try WAVWriter.write(samples: audio, sampleRate: 24000, to: outputURL)
```

### 스트리밍 합성

```swift
// 스트리밍: 생성되는 대로 오디오 청크 수신 (첫 청크까지 ~150ms)
for try await chunk in model.synthesizeStream(text: "Hello, how are you today?", language: "english") {
    // chunk.audio: [Float], chunk.sampleRate: Int
    playAudio(chunk.audio)  // 즉시 재생
}
```

### 음성 복제 (CosyVoice)

CAM++ 화자 인코더 (192차원, CoreML Neural Engine)를 사용하여 화자의 음성을 복제합니다:

```swift
import CosyVoiceTTS
import AudioCommon

let model = try await CosyVoiceTTSModel.fromPretrained()
let speaker = try await CamPlusPlusSpeaker.fromPretrained()
// 첫 사용 시 ~14 MB CAM++ CoreML 모델 다운로드

let refAudio = try AudioFileLoader.load(url: referenceURL, targetSampleRate: 16000)
let embedding = try speaker.embed(audio: refAudio, sampleRate: 16000)
// embedding: 길이 192의 [Float]

let audio = model.synthesize(
    text: "Hello in a cloned voice!",
    language: "english",
    speakerEmbedding: embedding
)
```

### CosyVoice TTS CLI

```bash
make build

# 기본 합성
.build/release/audio speak "Hello world" --engine cosyvoice --language english --output output.wav

# 음성 복제 (첫 사용 시 CAM++ 화자 인코더 다운로드)
.build/release/audio speak "Hello world" --engine cosyvoice --voice-sample reference.wav --output cloned.wav

# 음성 복제를 포함한 다화자 대화
.build/release/audio speak "[S1] Hello there! [S2] Hey, how are you?" \
    --engine cosyvoice --speakers s1=alice.wav,s2=bob.wav -o dialogue.wav

# 인라인 감정/스타일 태그
.build/release/audio speak "(excited) Wow, amazing! (sad) But I have to go..." \
    --engine cosyvoice -o emotion.wav

# 조합: 대화 + 감정 + 음성 복제
.build/release/audio speak "[S1] (happy) Great news! [S2] (surprised) Really?" \
    --engine cosyvoice --speakers s1=alice.wav,s2=bob.wav -o combined.wav

# 커스텀 스타일 지시
.build/release/audio speak "Hello world" --engine cosyvoice --cosy-instruct "Speak cheerfully" -o cheerful.wav

# 스트리밍 합성
.build/release/audio speak "Hello world" --engine cosyvoice --language english --stream --output output.wav
```

## Kokoro TTS — 경량 온디바이스 텍스트-음성 변환 (iOS + macOS)

### 기본 합성

```swift
import KokoroTTS
import AudioCommon  // WAVWriter 사용

let tts = try await KokoroTTSModel.fromPretrained()
// 첫 실행 시 ~170 MB 다운로드 (CoreML 모델 + 음색 임베딩 + 사전)

let audio = try tts.synthesize(text: "Hello world", voice: "af_heart")
// 출력은 24kHz 모노 float 샘플
try WAVWriter.write(samples: audio, sampleRate: 24000, to: outputURL)
```

10개 언어에 걸쳐 54개 프리셋 음색을 제공합니다. 비자기회귀 — 3단계 CoreML 파이프라인, 샘플링 루프 없음. Neural Engine에서 실행되어 GPU를 완전히 확보합니다.

### Kokoro TTS CLI

```bash
make build

# 기본 합성
.build/release/audio kokoro "Hello world" --voice af_heart --output hello.wav

# 언어 선택
.build/release/audio kokoro "Bonjour le monde" --voice ff_siwis --language fr --output bonjour.wav

# 사용 가능한 음색 목록
.build/release/audio kokoro --list-voices
```

## Qwen3 Chat (온디바이스 LLM)

```swift
import Qwen3Chat

let chat = try await Qwen3ChatModel.fromPretrained()
// 첫 실행 시 ~318 MB 다운로드 (INT4 CoreML 모델 + 토크나이저)

// 단일 응답
let response = try chat.generate("What is Swift?", systemPrompt: "Answer briefly.")
print(response)

// 스트리밍 토큰
let stream = chat.chatStream("Tell me a joke", systemPrompt: "Be funny.")
for try await token in stream {
    print(token, terminator: "")
}
```

Qwen3-0.6B는 CoreML용 INT4로 양자화되었습니다. Neural Engine에서 실행되며 iPhone에서 ~2 tok/s, M 시리즈에서 ~15 tok/s를 달성합니다. KV 캐시를 사용한 멀티턴 대화, 사고 모드 (`<think>` 토큰), 설정 가능한 샘플링 (temperature, top-k, top-p, repetition penalty)을 지원합니다.

## 음성 활동 감지 (VAD) — 오디오에서 음성 감지

### 스트리밍 VAD (Silero)

Silero VAD v5는 32ms 오디오 청크를 밀리초 미만의 지연 시간으로 처리합니다 — 마이크나 스트림에서의 실시간 음성 감지에 이상적입니다.

```swift
import SpeechVAD

let vad = try await SileroVADModel.fromPretrained()
// 또는 CoreML 사용 (Neural Engine, 저전력):
// let vad = try await SileroVADModel.fromPretrained(engine: .coreml)

// 스트리밍: 512 샘플 청크 처리 (16kHz에서 32ms)
let prob = vad.processChunk(samples)  // → 0.0...1.0
vad.resetState()  // 다른 오디오 스트림 간에 호출

// 또는 모든 세그먼트를 한 번에 감지
let segments = vad.detectSpeech(audio: audioSamples, sampleRate: 16000)
for seg in segments {
    print("음성: \(seg.startTime)s - \(seg.endTime)s")
}
```

### 이벤트 기반 스트리밍

```swift
let processor = StreamingVADProcessor(model: vad)

// 임의 길이의 오디오 입력 — 음성이 확인되면 이벤트 발생
let events = processor.process(samples: audioBuffer)
for event in events {
    switch event {
    case .speechStarted(let time):
        print("음성 시작: \(time)s")
    case .speechEnded(let segment):
        print("음성: \(segment.startTime)s - \(segment.endTime)s")
    }
}

// 스트림 종료 시 플러시
let final = processor.flush()
```

### VAD CLI

```bash
make build

# 스트리밍 Silero VAD (32ms 청크)
.build/release/audio vad-stream audio.wav

# CoreML 백엔드 (Neural Engine)
.build/release/audio vad-stream audio.wav --engine coreml

# 커스텀 임계값
.build/release/audio vad-stream audio.wav --onset 0.6 --offset 0.4

# JSON 출력
.build/release/audio vad-stream audio.wav --json

# 배치 pyannote VAD (10초 슬라이딩 윈도우)
.build/release/audio vad audio.wav
```

## 화자 분리 — 누가 언제 말했는지

### 분리 파이프라인

```swift
import SpeechVAD

let pipeline = try await DiarizationPipeline.fromPretrained()
// 또는 CoreML 임베딩 사용 (Neural Engine, GPU 확보):
// let pipeline = try await DiarizationPipeline.fromPretrained(embeddingEngine: .coreml)

let result = pipeline.diarize(audio: samples, sampleRate: 16000)
for seg in result.segments {
    print("화자 \(seg.speakerId): [\(seg.startTime)s - \(seg.endTime)s]")
}
print("\(result.numSpeakers)명의 화자 감지")
```

### 화자 임베딩

```swift
let model = try await WeSpeakerModel.fromPretrained()
// 또는: let model = try await WeSpeakerModel.fromPretrained(engine: .coreml)
let embedding = model.embed(audio: samples, sampleRate: 16000)
// embedding: L2 정규화된 길이 256의 [Float]

// 화자 비교
let similarity = WeSpeakerModel.cosineSimilarity(embeddingA, embeddingB)
```

### 화자 추출

참조 녹음을 사용하여 특정 화자의 세그먼트만 추출합니다:

```swift
let pipeline = try await DiarizationPipeline.fromPretrained()
let targetEmb = pipeline.embeddingModel.embed(audio: enrollmentAudio, sampleRate: 16000)
let segments = pipeline.extractSpeaker(
    audio: meetingAudio, sampleRate: 16000,
    targetEmbedding: targetEmb
)
```

### Sortformer 화자 분리 (엔드투엔드, CoreML)

NVIDIA Sortformer는 최대 4명의 화자에 대해 프레임별 화자 활동을 직접 예측합니다 — 임베딩이나 클러스터링이 필요 없습니다. Neural Engine에서 실행됩니다.

```swift
let diarizer = try await SortformerDiarizer.fromPretrained()
let result = diarizer.diarize(audio: samples, sampleRate: 16000, config: .default)
for seg in result.segments {
    print("화자 \(seg.speakerId): [\(seg.startTime)s - \(seg.endTime)s]")
}
```

### 화자 분리 CLI

```bash
make build

# Pyannote 화자 분리 (기본값)
.build/release/audio diarize meeting.wav

# Sortformer 화자 분리 (CoreML, Neural Engine)
.build/release/audio diarize meeting.wav --engine sortformer

# CoreML 임베딩 (Neural Engine, pyannote 전용)
.build/release/audio diarize meeting.wav --embedding-engine coreml

# JSON 출력
.build/release/audio diarize meeting.wav --json

# 특정 화자 추출 (pyannote 전용)
.build/release/audio diarize meeting.wav --target-speaker enrollment.wav

# 화자 임베딩
.build/release/audio embed-speaker enrollment.wav --json
.build/release/audio embed-speaker enrollment.wav --engine coreml
```

아키텍처 상세 내용은 [Speaker Diarization](docs/inference/speaker-diarization.md)을 참조하세요.

## 음성 향상 — 노이즈 억제 및 오디오 정리

### 노이즈 억제

```swift
import SpeechEnhancement
import AudioCommon  // WAVWriter 사용

let enhancer = try await SpeechEnhancer.fromPretrained()
// 첫 실행 시 ~4.3 MB 다운로드 (Core ML FP16 모델 + 보조 데이터)

let cleanAudio = try enhancer.enhance(audio: noisyAudio, sampleRate: 48000)
try WAVWriter.write(samples: cleanAudio, sampleRate: 48000, to: outputURL)
```

### 노이즈 제거 CLI

```bash
make build

# 기본 노이즈 제거
.build/release/audio denoise noisy.wav

# 커스텀 출력 경로
.build/release/audio denoise noisy.wav --output clean.wav
```

아키텍처 상세 내용은 [Speech Enhancement](docs/inference/speech-enhancement.md)를 참조하세요.

## 파이프라인 — 여러 모델 조합

모든 모델은 공유 프로토콜(`SpeechRecognitionModel`, `SpeechGenerationModel`, `SpeechEnhancementModel` 등)을 준수하며 파이프라인으로 조합할 수 있습니다:

### 잡음 환경 음성 인식 (DeepFilterNet + ASR)

```swift
import SpeechEnhancement
import Qwen3ASR

let enhancer = try await SpeechEnhancer.fromPretrained()
let asr = try await Qwen3ASRModel.fromPretrained()

// 48kHz에서 향상 후 16kHz에서 전사
let clean = try enhancer.enhance(audio: noisyAudio, sampleRate: 48000)
let clean16k = AudioResampler.resample(clean, from: 48000, to: 16000)
let text = asr.transcribe(audio: clean16k, sampleRate: 16000)
```

### 음성-음성 릴레이 (VAD + ASR + TTS)

```swift
import SpeechVAD
import Qwen3ASR
import Qwen3TTS

let vad = try await SileroVADModel.fromPretrained()
let asr = try await Qwen3ASRModel.fromPretrained()
let tts = try await Qwen3TTSModel.fromPretrained()

// 음성 세그먼트 감지, 전사, 재합성
let segments = vad.detectSpeech(audio: audio, sampleRate: 16000)
for seg in segments {
    let chunk = Array(audio[Int(seg.startTime * 16000)..<Int(seg.endTime * 16000)])
    let text = asr.transcribe(audio: chunk, sampleRate: 16000)
    let speech = tts.synthesize(text: text, language: "english")
    // speech: 24kHz 모노 float 샘플
}
```

### 회의 전사 (화자 분리 + ASR)

```swift
import SpeechVAD
import Qwen3ASR

let pipeline = try await DiarizationPipeline.fromPretrained()
let asr = try await Qwen3ASRModel.fromPretrained()

let result = pipeline.diarize(audio: meetingAudio, sampleRate: 16000)
for seg in result.segments {
    let chunk = Array(meetingAudio[Int(seg.startTime * 16000)..<Int(seg.endTime * 16000)])
    let text = asr.transcribe(audio: chunk, sampleRate: 16000)
    print("화자 \(seg.speakerId) [\(seg.startTime)s-\(seg.endTime)s]: \(text)")
}
```

전체 프로토콜 레퍼런스는 [Shared Protocols](docs/shared-protocols.md)를 참조하세요.

## HTTP API 서버

독립형 HTTP 서버가 모든 모델을 REST 및 WebSocket 엔드포인트를 통해 제공합니다. 모델은 첫 번째 요청 시 지연 로드됩니다.

```bash
swift build -c release
.build/release/audio-server --port 8080

# 오디오 전사
curl -X POST http://localhost:8080/transcribe --data-binary @audio.wav -H "Content-Type: audio/wav"

# 텍스트-음성 변환
curl -X POST http://localhost:8080/speak -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "engine": "cosyvoice"}' -o output.wav

# 음성-음성 (PersonaPlex)
curl -X POST http://localhost:8080/respond --data-binary @question.wav -o response.wav

# 음성 향상
curl -X POST http://localhost:8080/enhance --data-binary @noisy.wav -o clean.wav

# 시작 시 모든 모델 사전 로드
.build/release/audio-server --preload --port 8080
```

### WebSocket 스트리밍

#### OpenAI Realtime API (`/v1/realtime`)

기본 WebSocket 엔드포인트는 [OpenAI Realtime API](https://platform.openai.com/docs/api-reference/realtime) 프로토콜을 구현합니다 — 모든 메시지는 `type` 필드가 있는 JSON이며, 오디오는 base64 인코딩된 PCM16 24kHz 모노입니다.

**클라이언트 → 서버 이벤트:**

| 이벤트 | 설명 |
|-------|-------------|
| `session.update` | 엔진, 언어, 오디오 형식 설정 |
| `input_audio_buffer.append` | base64 PCM16 오디오 청크 전송 |
| `input_audio_buffer.commit` | 축적된 오디오 전사 (ASR) |
| `input_audio_buffer.clear` | 오디오 버퍼 초기화 |
| `response.create` | TTS 합성 요청 |

**서버 → 클라이언트 이벤트:**

| 이벤트 | 설명 |
|-------|-------------|
| `session.created` | 세션 초기화됨 |
| `session.updated` | 설정 확인됨 |
| `input_audio_buffer.committed` | 전사를 위한 오디오 커밋됨 |
| `conversation.item.input_audio_transcription.completed` | ASR 결과 |
| `response.audio.delta` | Base64 PCM16 오디오 청크 (TTS) |
| `response.audio.done` | 오디오 스트리밍 완료 |
| `response.done` | 메타데이터 포함 응답 완료 |
| `error` | 유형 및 메시지 포함 오류 |

```javascript
const ws = new WebSocket('ws://localhost:8080/v1/realtime');

// ASR: 오디오 전송, 전사 결과 수신
ws.send(JSON.stringify({ type: 'input_audio_buffer.append', audio: base64PCM16 }));
ws.send(JSON.stringify({ type: 'input_audio_buffer.commit' }));
// → 수신: conversation.item.input_audio_transcription.completed

// TTS: 텍스트 전송, 스트리밍 오디오 수신
ws.send(JSON.stringify({
  type: 'response.create',
  response: { modalities: ['audio', 'text'], instructions: 'Hello world' }
}));
// → 수신: response.audio.delta (base64 청크), response.audio.done, response.done
```

`Examples/websocket-client.html`에 예제 HTML 클라이언트가 있습니다 — 서버 실행 중에 브라우저에서 열어보세요.

서버는 별도의 `AudioServer` 모듈이자 `audio-server` 실행 파일입니다 — 메인 `audio` CLI에 Hummingbird/WebSocket을 추가하지 않습니다.

## 지연 시간 (M2 Max, 64 GB)

### ASR

| 모델 | 백엔드 | RTF | 10초 오디오 처리 시간 |
|-------|---------|-----|------------------------|
| Qwen3-ASR-0.6B (4-bit) | MLX | ~0.06 | ~0.6s |
| Qwen3-ASR-0.6B (INT8) | CoreML + MLX | ~0.09 | ~0.9s |
| Qwen3-ASR-1.7B (8-bit) | MLX | ~0.11 | ~1.1s |
| Parakeet-TDT-0.6B (INT4) | CoreML (Neural Engine) | ~0.12 cold, ~0.03 warm | ~1.2s / ~0.3s |
| Whisper-large-v3 | whisper.cpp (Q5_0) | ~0.10 | ~1.0s |
| Whisper-small | whisper.cpp (Q5_0) | ~0.04 | ~0.4s |

### 강제 정렬

| 모델 | 프레임워크 | 20초 오디오 | RTF |
|-------|-----------|-----------|-----|
| Qwen3-ForcedAligner-0.6B (4-bit) | MLX Swift (debug) | ~365ms | ~0.018 |

> 단일 비자기회귀 순전파 — 샘플링 루프 없음. 오디오 인코더가 대부분을 차지하며 (~328ms), 디코더 단일 패스는 ~37ms. **실시간보다 55배 빠름.**

### TTS

| 모델 | 프레임워크 | 짧은 (1초) | 중간 (3초) | 긴 (6초) | 스트리밍 첫 패킷 |
|-------|-----------|-----------|-------------|------------|----------------------|
| Qwen3-TTS-0.6B (4-bit) | MLX Swift (release) | 1.6s (RTF 1.2) | 2.3s (RTF 0.7) | 3.9s (RTF 0.7) | ~120ms (1-frame) |
| Kokoro-82M | CoreML (Neural Engine) | ~1.4s (RTFx 0.7) | ~4.3s (RTFx 0.7) | ~8.6s (RTFx 0.7) | N/A (비자기회귀) |
| Apple `AVSpeechSynthesizer` | AVFoundation | 0.08s | 0.08s | 0.17s (RTF 0.02) | N/A |

> Qwen3-TTS는 운율과 감정이 담긴 자연스럽고 표현력 있는 음성을 생성하며, **실시간보다 빠르게** (RTF < 1.0) 실행됩니다. 스트리밍 합성은 첫 오디오 청크를 ~120ms에 전달합니다. Kokoro-82M은 3단계 파이프라인으로 Neural Engine에서 완전히 실행됩니다 (RTFx ~0.7), iOS에 이상적입니다. Apple 내장 TTS는 더 빠르지만 기계적이고 단조로운 음성을 생성합니다.

### PersonaPlex (음성-음성)

| 모델 | 프레임워크 | ms/step | RTF | 비고 |
|-------|-----------|---------|-----|-------|
| PersonaPlex-7B (8-bit) | MLX Swift (release) | ~112ms | ~1.4 | 권장 — 일관된 응답, 4-bit보다 30% 빠름 |
| PersonaPlex-7B (4-bit) | MLX Swift (release) | ~158ms | ~1.97 | 비권장 — 출력 품질 저하 |

> **8-bit를 사용하세요.** INT8은 더 빠르고 (112 ms/step vs. 158 ms/step) 일관된 전이중 응답을 생성합니다. INT4 양자화는 생성 품질을 저하시켜 알아들을 수 없는 음성을 생성합니다. INT8은 M2 Max에서 ~112ms/step으로 실행됩니다.

### VAD 및 화자 임베딩

| 모델 | 백엔드 | 호출당 지연 시간 | RTF | 비고 |
|-------|---------|-----------------|-----|-------|
| Silero-VAD-v5 | MLX | ~2.1ms / 청크 | 0.065 | GPU (Metal) |
| Silero-VAD-v5 | CoreML | ~0.27ms / 청크 | 0.008 | Neural Engine, **7.7배 빠름** |
| WeSpeaker ResNet34-LM | MLX | ~310ms / 20초 오디오 | 0.016 | GPU (Metal) |
| WeSpeaker ResNet34-LM | CoreML | ~430ms / 20초 오디오 | 0.021 | Neural Engine, GPU 확보 |

> Silero VAD CoreML은 Neural Engine에서 MLX보다 7.7배 빠르게 실행되어 상시 마이크 입력에 이상적입니다. WeSpeaker MLX는 GPU에서 더 빠르지만, CoreML은 동시 작업(TTS, ASR)을 위해 GPU를 확보합니다. 두 백엔드 모두 동일한 결과를 생성합니다.

### 음성 향상

| 모델 | 백엔드 | 길이 | 지연 시간 | RTF |
|-------|---------|----------|---------|-----|
| DeepFilterNet3 (FP16) | CoreML | 5s | 0.65s | 0.13 |
| DeepFilterNet3 (FP16) | CoreML | 10s | 1.2s | 0.12 |
| DeepFilterNet3 (FP16) | CoreML | 20s | 4.8s | 0.24 |

RTF = Real-Time Factor (낮을수록 좋음, < 1.0 = 실시간보다 빠름). GRU 비용은 ~O(n²)로 증가합니다.

### MLX vs CoreML

두 백엔드 모두 동일한 결과를 생성합니다. 작업 부하에 따라 선택하세요:

| | MLX | CoreML |
|---|---|---|
| **하드웨어** | GPU (Metal 셰이더) | Neural Engine + CPU |
| **적합한 용도** | 최대 처리량, 단일 모델 작업 | 다중 모델 파이프라인, 백그라운드 작업 |
| **전력** | 높은 GPU 활용률 | 저전력, GPU 확보 |
| **지연 시간** | 대형 모델에서 빠름 (WeSpeaker) | 소형 모델에서 빠름 (Silero VAD) |

**데스크톱 추론**: MLX가 기본값입니다 — Apple Silicon에서 가장 빠른 단일 모델 성능을 제공합니다. 여러 모델을 동시에 실행할 때 (예: VAD + ASR + TTS) GPU 경합을 피하려면 CoreML로 전환하거나, 노트북에서 배터리에 민감한 작업에 사용하세요.

CoreML 모델은 Qwen3-ASR 인코더, Silero VAD, WeSpeaker에 사용할 수 있습니다. Qwen3-ASR의 경우 `--engine qwen3-coreml` (하이브리드: ANE의 CoreML 인코더 + GPU의 MLX 텍스트 디코더)을 사용하세요. VAD/임베딩의 경우 생성 시 `engine: .coreml`을 전달하세요 — 추론 API는 동일합니다.

## 정확도 벤치마크

### ASR — 단어 오류율 ([상세](docs/benchmarks/asr-wer.md))

| 모델 | WER% (LibriSpeech test-clean) | RTF |
|-------|-------------------------------|-----|
| Qwen3-ASR 1.7B 8-bit | **2.35** | 0.090 |
| Qwen3-ASR 1.7B 4-bit | 2.57 | 0.045 |
| Parakeet TDT INT8 | 2.74 | 0.089 |
| Qwen3-ASR 0.6B 8-bit | 2.80 | 0.025 |

Qwen3-ASR 1.7B 8-bit는 비슷한 크기에서 Whisper Large v3 Turbo (2.5%)를 능가합니다. 다국어: FLEURS에서 10개 언어 벤치마크 완료.

### TTS — 왕복 명료도 ([상세](docs/benchmarks/tts-roundtrip.md))

| 엔진 | WER% | RTF |
|--------|------|-----|
| CosyVoice3 | **3.25** | 0.59 |
| Qwen3-TTS 1.7B | 3.47 | 0.79 |
| Kokoro-82M | 3.90 | 0.17 |

### VAD — 음성 감지 ([상세](docs/benchmarks/vad-detection.md))

| 엔진 | F1% (FLEURS) | RTF |
|--------|-------------|-----|
| FireRedVAD | **99.12** | 0.007 |
| Silero CoreML | 95.13 | 0.022 |
| Pyannote MLX | 94.86 | 0.358 |

## 아키텍처

**모델:** [ASR Model](docs/models/asr-model.md), [TTS Model](docs/models/tts-model.md), [CosyVoice TTS](docs/models/cosyvoice-tts.md), [Kokoro TTS](docs/models/kokoro-tts.md), [Parakeet TDT](docs/models/parakeet-asr.md), [PersonaPlex](docs/models/personaplex.md), [FireRedVAD](docs/models/fireredvad.md)

**추론:** [Qwen3-ASR](docs/inference/qwen3-asr-inference.md), [Parakeet TDT](docs/inference/parakeet-asr-inference.md), [TTS Inference](docs/inference/qwen3-tts-inference.md), [Forced Aligner](docs/inference/forced-aligner.md), [FireRedVAD](docs/inference/fireredvad.md), [Silero VAD](docs/inference/silero-vad.md), [Speaker Diarization](docs/inference/speaker-diarization.md), [Speech Enhancement](docs/inference/speech-enhancement.md), [오디오 재생](docs/audio/playback.md)

**벤치마크:** [ASR WER](docs/benchmarks/asr-wer.md), [TTS Round-Trip](docs/benchmarks/tts-roundtrip.md), [VAD Detection](docs/benchmarks/vad-detection.md)

**레퍼런스:** [Shared Protocols](docs/shared-protocols.md)

## 캐시 설정

모델 가중치는 로컬에 캐시됩니다. 캐시 위치를 변경하려면:

```bash
export QWEN3_CACHE_DIR=/path/to/cache
```

## MLX Metal 라이브러리

런타임에 `Failed to load the default metallib`이 표시되면 Metal 셰이더 라이브러리가 없는 것입니다. `make build` (또는 수동 `swift build` 후 `./scripts/build_mlx_metallib.sh release`)를 실행하여 컴파일하세요. Metal Toolchain이 없으면 먼저 설치하세요:

```bash
xcodebuild -downloadComponent MetalToolchain
```

## 테스트

단위 테스트 (config, 샘플링, 텍스트 전처리, 타임스탬프 보정)는 모델 다운로드 없이 실행됩니다:

```bash
swift test --filter "Qwen3TTSConfigTests|SamplingTests|CosyVoiceTTSConfigTests|CamPlusPlusMelExtractorTests|PersonaPlexTests|ForcedAlignerTests/testText|ForcedAlignerTests/testTimestamp|ForcedAlignerTests/testLIS|SileroVADTests/testSilero|SileroVADTests/testReflection|SileroVADTests/testProcess|SileroVADTests/testReset|SileroVADTests/testDetect|SileroVADTests/testStreaming|SileroVADTests/testVADEvent|KokoroTTSTests"
```

통합 테스트는 모델 가중치가 필요합니다 (첫 실행 시 자동 다운로드):

```bash
# TTS 왕복: 텍스트 합성, WAV 저장, ASR로 다시 전사
swift test --filter TTSASRRoundTripTests

# ASR만: 테스트 오디오 전사
swift test --filter Qwen3ASRIntegrationTests

# Forced Aligner E2E: 단어 수준 타임스탬프 (~979 MB 다운로드)
swift test --filter ForcedAlignerTests/testForcedAlignerE2E

# PersonaPlex E2E: 음성-음성 파이프라인 (~5.5 GB 다운로드)
PERSONAPLEX_E2E=1 swift test --filter PersonaPlexE2ETests
```

> **참고:** MLX 연산을 사용하는 테스트를 실행하기 전에 MLX Metal 라이브러리를 빌드해야 합니다.
> [MLX Metal 라이브러리](#mlx-metal-라이브러리) 안내를 참조하세요.

## 지원 언어

| 모델 | 언어 |
|-------|-----------|
| Qwen3-ASR | 52개 언어 (CN, EN, 광둥어, DE, FR, ES, JA, KO, RU, + 22개 중국어 방언, ...) |
| Parakeet TDT | 25개 유럽 언어 (BG, CS, DA, DE, EL, EN, ES, ET, FI, FR, HR, HU, IT, LT, LV, MT, NL, PL, PT, RO, RU, SK, SL, SV, UK) |
| Qwen3-TTS | EN, CN, DE, JA, ES, FR, KO, RU, IT, PT (+ CustomVoice를 통한 북경/사천 방언) |
| CosyVoice TTS | CN, EN, JA, KO, DE, ES, FR, IT, RU |
| Kokoro TTS | EN (US/UK), ES, FR, HI, IT, JA, PT, CN, KO, DE |
| PersonaPlex | EN |

## 비교

### 음성-텍스트 (ASR): speech-swift vs 대안

| | **speech-swift (Qwen3-ASR)** | **whisper.cpp** | **Apple SFSpeechRecognizer** | **Google Cloud Speech** |
|---|---|---|---|---|
| **런타임** | 온디바이스 (MLX/CoreML) | 온디바이스 (CPU/GPU) | 온디바이스 또는 클라우드 | 클라우드 전용 |
| **언어** | 52 | 100+ | ~70 (온디바이스: 제한적) | 125+ |
| **RTF (10초 오디오, M2 Max)** | 0.06 (17배 실시간) | 0.10 (Whisper-large-v3) | N/A | N/A |
| **스트리밍** | 아니오 (배치) | 아니오 (배치) | 예 | 예 |
| **커스텀 모델** | 예 (HuggingFace 가중치 교체) | 예 (GGML 모델) | 아니오 | 아니오 |
| **Swift API** | 네이티브 async/await | Swift 브릿지가 필요한 C++ | 네이티브 | REST/gRPC |
| **프라이버시** | 완전 온디바이스 | 완전 온디바이스 | 설정에 따라 다름 | 데이터가 클라우드로 전송됨 |
| **단어 타임스탬프** | 예 (Forced Aligner) | 예 | 제한적 | 예 |
| **비용** | 무료 (Apache 2.0) | 무료 (MIT) | 무료 (온디바이스) | 분당 과금 |

### 텍스트-음성 (TTS): speech-swift vs 대안

| | **speech-swift (Qwen3-TTS)** | **speech-swift (Kokoro)** | **Apple AVSpeechSynthesizer** | **ElevenLabs / 클라우드 TTS** |
|---|---|---|---|---|
| **품질** | 뉴럴, 표현력 있음 | 뉴럴, 자연스러움 | 기계적, 단조로움 | 뉴럴, 최고 품질 |
| **런타임** | 온디바이스 (MLX) | 온디바이스 (CoreML) | 온디바이스 | 클라우드 전용 |
| **스트리밍** | 예 (첫 청크 ~120ms) | 아니오 (3단계 파이프라인) | 아니오 | 예 |
| **음성 복제** | 예 | 아니오 | 아니오 | 예 |
| **음색** | 9개 내장 + 아무 음성 복제 | 54개 프리셋 음색 | ~50개 시스템 음색 | 1000+ |
| **언어** | 10 | 10 | 60+ | 30+ |
| **iOS 지원** | macOS 전용 | iOS + macOS | iOS + macOS | 아무 플랫폼 (API) |
| **비용** | 무료 (Apache 2.0) | 무료 (Apache 2.0) | 무료 | 글자당 과금 |

### speech-swift를 사용해야 할 때

- **프라이버시가 중요한 앱** — 의료, 법률, 기업 환경에서 오디오가 기기를 떠날 수 없는 경우
- **오프라인 사용** — 초기 모델 다운로드 후 인터넷 연결 불필요
- **비용에 민감한 경우** — 분당 또는 글자당 API 요금 없음
- **Apple Silicon 최적화** — M 시리즈 GPU (Metal) 및 Neural Engine에 특화되어 구축됨
- **전체 파이프라인** — ASR + TTS + VAD + 화자 분리 + 음성 향상을 단일 Swift 패키지로 조합

## FAQ

**speech-swift는 iOS에서 작동하나요?**
Kokoro TTS, Qwen3.5-Chat (CoreML), Silero VAD, Parakeet ASR, DeepFilterNet3, WeSpeaker는 모두 CoreML을 통해 Neural Engine에서 iOS 17+에서 실행됩니다. MLX 기반 모델 (Qwen3-ASR, Qwen3-TTS, Qwen3.5-Chat MLX, PersonaPlex)은 Apple Silicon의 macOS 14+ 이상이 필요합니다.

**인터넷 연결이 필요한가요?**
HuggingFace에서 초기 모델 다운로드 시에만 필요합니다 (자동, `~/Library/Caches/qwen3-speech/`에 캐시). 이후에는 네트워크 접속 없이 모든 추론이 완전히 오프라인으로 실행됩니다.

**speech-swift는 Whisper와 어떻게 비교되나요?**
Qwen3-ASR-0.6B는 M2 Max에서 RTF 0.06을 달성합니다 — whisper.cpp를 통한 Whisper-large-v3 (RTF 0.10)보다 40% 빠르며, 52개 언어에 걸쳐 비슷한 정확도를 보여줍니다. speech-swift는 네이티브 Swift async/await API를 제공하는 반면, whisper.cpp는 C++ 브릿지가 필요합니다.

**상용 앱에 사용할 수 있나요?**
예. speech-swift는 Apache 2.0 라이선스입니다. 기반 모델 가중치에는 자체 라이선스가 있습니다 (각 모델의 HuggingFace 페이지를 확인하세요).

**어떤 Apple Silicon 칩이 지원되나요?**
모든 M 시리즈 칩: M1, M2, M3, M4 및 Pro/Max/Ultra 변형. macOS 14+ (Sonoma) 또는 iOS 17+ 이상이 필요합니다.

**메모리는 얼마나 필요한가요?**
~3 MB (Silero VAD)에서 ~6.5 GB (PersonaPlex 7B)까지 다양합니다. Kokoro TTS는 ~500 MB, Qwen3-ASR는 ~2.2 GB를 사용합니다. 전체 세부 사항은 [메모리 요구 사항](#메모리-요구-사항) 표를 참조하세요.

**여러 모델을 동시에 실행할 수 있나요?**
예. GPU 경합을 피하기 위해 Neural Engine의 CoreML 모델과 GPU의 MLX 모델을 함께 사용하세요 — 예를 들어, Silero VAD (CoreML) + Qwen3-ASR (MLX) + Qwen3-TTS (MLX).

**REST API가 있나요?**
예. `audio-server` 바이너리가 HTTP REST 및 WebSocket 엔드포인트를 통해 모든 모델을 제공하며, `/v1/realtime`에서 OpenAI Realtime API 호환 WebSocket을 포함합니다.

## 기여하기

기여를 환영합니다! 버그 수정, 새 모델 통합, 문서 개선 등 — PR을 환영합니다.

**시작하려면:**
1. 리포를 포크하고 기능 브랜치를 생성하세요
2. `make build`로 컴파일하세요 (Xcode + Metal Toolchain 필요)
3. `make test`로 테스트 스위트를 실행하세요
4. `main`을 대상으로 PR을 제출하세요

## 라이선스

Apache 2.0
