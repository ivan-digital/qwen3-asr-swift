# Speech Swift

MLX Swift와 CoreML 기반의 Apple Silicon용 AI 음성 모델.

📖 Read in: [English](README.md) · [中文](README_zh.md) · [日本語](README_ja.md) · [한국어](README_ko.md) · [Español](README_es.md) · [Deutsch](README_de.md) · [Français](README_fr.md) · [हिन्दी](README_hi.md) · [Português](README_pt.md) · [Русский](README_ru.md)

Mac과 iOS를 위한 온디바이스 음성 인식, 합성 및 이해. Apple Silicon에서 완전히 로컬로 실행됩니다 — 클라우드 없이, API 키 없이, 데이터가 기기 밖으로 나가지 않습니다.

**[📚 전체 문서 →](https://soniqo.audio/ko)** · **[🤗 HuggingFace 모델](https://huggingface.co/aufklarer)** · **[📝 블로그](https://blog.ivan.digital)**

- **[Qwen3-ASR](https://soniqo.audio/ko/guides/transcribe)** — 음성-텍스트 변환 (자동 음성 인식, 52개 언어, MLX + CoreML)
- **[Parakeet TDT](https://soniqo.audio/ko/guides/parakeet)** — CoreML을 통한 음성-텍스트 변환 (Neural Engine, NVIDIA FastConformer + TDT 디코더, 25개 언어)
- **[Omnilingual ASR](https://soniqo.audio/ko/guides/omnilingual)** — 음성-텍스트 변환 (Meta wav2vec2 + CTC, **1,672개 언어**, 32개 문자 체계, CoreML 300M + MLX 300M/1B/3B/7B)
- **[스트리밍 받아쓰기](https://soniqo.audio/ko/guides/dictate)** — 부분 결과와 발화 종료 감지를 갖춘 실시간 받아쓰기 (Parakeet-EOU-120M)
- **[Nemotron 스트리밍](https://soniqo.audio/ko/guides/nemotron)** — 네이티브 구두점 및 대소문자 처리를 제공하는 저지연 스트리밍 ASR (NVIDIA Nemotron-Speech-Streaming-0.6B, CoreML, 영어)
- **[Qwen3-ForcedAligner](https://soniqo.audio/ko/guides/align)** — 단어 수준 타임스탬프 정렬 (오디오 + 텍스트 → 타임스탬프)
- **[Qwen3-TTS](https://soniqo.audio/ko/guides/speak)** — 텍스트-음성 변환 (최고 품질, 스트리밍, 커스텀 화자, 10개 언어)
- **[CosyVoice TTS](https://soniqo.audio/ko/guides/cosyvoice)** — 음성 복제, 다화자 대화, 감정 태그를 지원하는 스트리밍 TTS (9개 언어)
- **[Kokoro TTS](https://soniqo.audio/ko/guides/kokoro)** — 온디바이스 TTS (82M, CoreML/Neural Engine, 54개 음색, iOS 지원, 10개 언어)
- **[Qwen3.5-Chat](https://soniqo.audio/ko/guides/chat)** — 온디바이스 LLM 채팅 (0.8B, MLX INT4 + CoreML INT8, DeltaNet 하이브리드, 스트리밍 토큰)
- **[PersonaPlex](https://soniqo.audio/ko/guides/respond)** — 전이중 음성-음성 대화 (7B, 오디오 입력 → 오디오 출력, 18개 음색 프리셋)
- **[DeepFilterNet3](https://soniqo.audio/ko/guides/denoise)** — 실시간 노이즈 억제 (2.1M 파라미터, 48 kHz)
- **[웨이크워드](https://soniqo.audio/ko/guides/wake-word)** — 온디바이스 키워드 감지 (KWS Zipformer 3M, CoreML, 실시간의 26배, 구성 가능한 키워드 목록)
- **[VAD](https://soniqo.audio/ko/guides/vad)** — 음성 활동 감지 (Silero 스트리밍, Pyannote 오프라인, FireRedVAD 100+ 개 언어)
- **[화자 분리](https://soniqo.audio/ko/guides/diarize)** — 누가 언제 말했는지 (Pyannote 파이프라인, Neural Engine 상의 엔드투엔드 Sortformer)
- **[화자 임베딩](https://soniqo.audio/ko/guides/embed-speaker)** — WeSpeaker ResNet34 (256차원), CAM++ (192차원)

논문: [Qwen3-ASR](https://arxiv.org/abs/2601.21337) (Alibaba) · [Qwen3-TTS](https://arxiv.org/abs/2601.15621) (Alibaba) · [Omnilingual ASR](https://arxiv.org/abs/2511.09690) (Meta) · [Parakeet TDT](https://arxiv.org/abs/2304.06795) (NVIDIA) · [CosyVoice 3](https://arxiv.org/abs/2505.17589) (Alibaba) · [Kokoro](https://arxiv.org/abs/2301.01695) (StyleTTS 2) · [PersonaPlex](https://arxiv.org/abs/2602.06053) (NVIDIA) · [Mimi](https://arxiv.org/abs/2410.00037) (Kyutai) · [Sortformer](https://arxiv.org/abs/2409.06656) (NVIDIA)

## 소식

- **2026년 4월 19일** — [Apple Silicon에서의 MLX와 CoreML — 올바른 백엔드 선택을 위한 실용 가이드](https://blog.ivan.digital/mlx-vs-coreml-on-apple-silicon-a-practical-guide-to-picking-the-right-backend-and-why-you-should-f77ddea7b27a)
- **2026년 3월 20일** — [600M 모델로 Mac에서 Whisper Large v3를 능가하다](https://blog.ivan.digital/we-beat-whisper-large-v3-with-a-600m-model-running-entirely-on-your-mac-20e6ce191174)
- **2026년 2월 26일** — [Apple Silicon에서의 화자 분리 및 음성 활동 감지 — MLX 기반 네이티브 Swift](https://blog.ivan.digital/speaker-diarization-and-voice-activity-detection-on-apple-silicon-native-swift-with-mlx-92ea0c9aca0f)
- **2026년 2월 23일** — [Apple Silicon에서 NVIDIA PersonaPlex 7B — MLX 기반 네이티브 Swift로 전이중 음성-음성 변환](https://blog.ivan.digital/nvidia-personaplex-7b-on-apple-silicon-full-duplex-speech-to-speech-in-native-swift-with-mlx-0aa5276f2e23)
- **2026년 2월 12일** — [Qwen3-ASR Swift: Apple Silicon용 온디바이스 ASR + TTS — 아키텍처 및 벤치마크](https://blog.ivan.digital/qwen3-asr-swift-on-device-asr-tts-for-apple-silicon-architecture-and-benchmarks-27cbf1e4463f)

## 빠른 시작

`Package.swift`에 패키지를 추가하세요:

```swift
.package(url: "https://github.com/soniqo/speech-swift", branch: "main")
```

필요한 모듈만 임포트하세요 — 모든 모델이 독립된 SPM 라이브러리이므로 사용하지 않는 것에 비용을 지불할 필요가 없습니다:

```swift
.product(name: "ParakeetStreamingASR", package: "speech-swift"),
.product(name: "SpeechUI",             package: "speech-swift"),  // 선택적 SwiftUI 뷰
```

**3줄로 오디오 버퍼 전사:**

```swift
import ParakeetStreamingASR

let model = try await ParakeetStreamingASRModel.fromPretrained()
let text = try model.transcribeAudio(audioSamples, sampleRate: 16000)
```

**부분 결과가 포함된 라이브 스트리밍:**

```swift
for await partial in model.transcribeStream(audio: samples, sampleRate: 16000) {
    print(partial.isFinal ? "FINAL: \(partial.text)" : "... \(partial.text)")
}
```

**약 10줄짜리 SwiftUI 받아쓰기 뷰:**

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

`SpeechUI`에는 `TranscriptionView`(파이널 + 파셜)와 `TranscriptionStore`(스트리밍 ASR 어댑터)만 포함됩니다. 오디오 시각화와 재생에는 AVFoundation을 사용하세요.

사용 가능한 SPM 프로덕트: `Qwen3ASR`, `Qwen3TTS`, `Qwen3TTSCoreML`, `ParakeetASR`, `ParakeetStreamingASR`, `NemotronStreamingASR`, `OmnilingualASR`, `KokoroTTS`, `CosyVoiceTTS`, `PersonaPlex`, `SpeechVAD`, `SpeechEnhancement`, `Qwen3Chat`, `SpeechCore`, `SpeechUI`, `AudioCommon`.

## 모델

아래는 컴팩트 뷰입니다. **[크기, 양자화, 다운로드 URL, 메모리 테이블을 포함한 전체 모델 카탈로그 → soniqo.audio/architecture](https://soniqo.audio/ko/architecture)**.

| 모델 | 작업 | 백엔드 | 크기 | 언어 |
|-------|------|----------|-------|-----------|
| [Qwen3-ASR](https://soniqo.audio/ko/guides/transcribe) | 음성 → 텍스트 | MLX, CoreML (하이브리드) | 0.6B, 1.7B | 52 |
| [Parakeet TDT](https://soniqo.audio/ko/guides/parakeet) | 음성 → 텍스트 | CoreML (ANE) | 0.6B | 25개 유럽어 |
| [Parakeet EOU](https://soniqo.audio/ko/guides/dictate) | 음성 → 텍스트 (스트리밍) | CoreML (ANE) | 120M | 25개 유럽어 |
| [Nemotron Streaming](https://soniqo.audio/ko/guides/nemotron) | 음성 → 텍스트 (스트리밍, 구두점 포함) | CoreML (ANE) | 0.6B | 영어 |
| [Omnilingual ASR](https://soniqo.audio/ko/guides/omnilingual) | 음성 → 텍스트 | CoreML (ANE), MLX | 300M / 1B / 3B / 7B | **[1,672](https://github.com/facebookresearch/omnilingual-asr/blob/main/src/omnilingual_asr/models/wav2vec2_llama/lang_ids.py)** |
| [Qwen3-ForcedAligner](https://soniqo.audio/ko/guides/align) | 오디오 + 텍스트 → 타임스탬프 | MLX, CoreML | 0.6B | 다언어 |
| [Qwen3-TTS](https://soniqo.audio/ko/guides/speak) | 텍스트 → 음성 | MLX, CoreML | 0.6B, 1.7B | 10 |
| [CosyVoice3](https://soniqo.audio/ko/guides/cosyvoice) | 텍스트 → 음성 | MLX | 0.5B | 9 |
| [Kokoro-82M](https://soniqo.audio/ko/guides/kokoro) | 텍스트 → 음성 | CoreML (ANE) | 82M | 10 |
| [Qwen3.5-Chat](https://soniqo.audio/ko/guides/chat) | 텍스트 → 텍스트 (LLM) | MLX, CoreML | 0.8B | 다언어 |
| [PersonaPlex](https://soniqo.audio/ko/guides/respond) | 음성 → 음성 | MLX | 7B | EN |
| [Silero VAD](https://soniqo.audio/ko/guides/vad) | 음성 활동 감지 | MLX, CoreML | 309K | 언어 무관 |
| [Pyannote](https://soniqo.audio/ko/guides/diarize) | VAD + 화자 분리 | MLX | 1.5M | 언어 무관 |
| [Sortformer](https://soniqo.audio/ko/guides/diarize) | 화자 분리 (E2E) | CoreML (ANE) | — | 언어 무관 |
| [DeepFilterNet3](https://soniqo.audio/ko/guides/denoise) | 음성 향상 | CoreML | 2.1M | 언어 무관 |
| [WeSpeaker](https://soniqo.audio/ko/guides/embed-speaker) | 화자 임베딩 | MLX, CoreML | 6.6M | 언어 무관 |

## 설치

### Homebrew

네이티브 ARM Homebrew(`/opt/homebrew`)가 필요합니다. Rosetta/x86_64 Homebrew는 지원되지 않습니다.

```bash
brew tap soniqo/speech https://github.com/soniqo/speech-swift
brew install speech
```

그런 다음:

```bash
audio transcribe recording.wav
audio speak "Hello world"
audio respond --input question.wav --transcript
audio-server --port 8080            # 로컬 HTTP / WebSocket 서버 (OpenAI 호환 /v1/realtime)
```

**[전체 CLI 레퍼런스 →](https://soniqo.audio/ko/cli)**

### Swift Package Manager

```swift
dependencies: [
    .package(url: "https://github.com/soniqo/speech-swift", branch: "main")
]
```

필요한 것만 임포트하세요 — 모든 모델이 독립된 SPM 타겟입니다:

```swift
import Qwen3ASR             // 음성 인식 (MLX)
import ParakeetASR          // 음성 인식 (CoreML, 배치)
import ParakeetStreamingASR // 부분 결과 + EOU 포함 스트리밍 받아쓰기
import NemotronStreamingASR // 영어 스트리밍 ASR, 네이티브 구두점 (0.6B)
import OmnilingualASR       // 1,672개 언어 (CoreML + MLX)
import Qwen3TTS             // 텍스트-음성 변환
import CosyVoiceTTS         // 음성 복제 포함 텍스트-음성 변환
import KokoroTTS            // 텍스트-음성 변환 (iOS 지원)
import Qwen3Chat            // 온디바이스 LLM 채팅
import PersonaPlex          // 전이중 음성-음성 변환
import SpeechVAD            // VAD + 화자 분리 + 임베딩
import SpeechEnhancement    // 노이즈 억제
import SpeechUI             // 스트리밍 전사를 위한 SwiftUI 컴포넌트
import AudioCommon          // 공유 프로토콜 및 유틸리티
```

### 요구사항

- Swift 6+, Xcode 16+ (Metal Toolchain 포함)
- macOS 15+ (Sequoia) 또는 iOS 18+, Apple Silicon (M1/M2/M3/M4)

macOS 15 / iOS 18 최소 요구사항은 [MLState](https://developer.apple.com/documentation/coreml/mlstate) —— Apple의 영속적 ANE 상태 API —— 에서 비롯됩니다. CoreML 파이프라인(Qwen3-ASR, Qwen3-Chat, Qwen3-TTS)은 MLState를 사용해 KV 캐시를 토큰 스텝 간 Neural Engine에 상주시킵니다.

### 소스 빌드

```bash
git clone https://github.com/soniqo/speech-swift
cd speech-swift
make build
```

`make build`는 Swift 패키지**와** MLX Metal 셰이더 라이브러리를 함께 컴파일합니다. Metal 라이브러리는 GPU 추론에 필요합니다 — 없으면 런타임에 `Failed to load the default metallib`이 발생합니다. 디버그 빌드는 `make debug`, 테스트 스위트는 `make test`로 실행합니다.

**[전체 빌드 및 설치 가이드 →](https://soniqo.audio/ko/getting-started)**

## 데모 앱

- **[DictateDemo](Examples/DictateDemo/)** ([문서](https://soniqo.audio/ko/guides/dictate)) — macOS 메뉴 바 스트리밍 받아쓰기. 라이브 파셜, VAD 기반 발화 종료 감지, 원클릭 복사. 백그라운드 agent로 실행됩니다 (Parakeet-EOU-120M + Silero VAD).
- **[iOSEchoDemo](Examples/iOSEchoDemo/)** — iOS 에코 데모 (Parakeet ASR + Kokoro TTS). 기기 및 시뮬레이터 지원.
- **[PersonaPlexDemo](Examples/PersonaPlexDemo/)** — 마이크 입력, VAD, 멀티턴 컨텍스트를 지원하는 대화형 음성 어시스턴트. macOS. M2 Max에서 RTF 약 0.94 (실시간보다 빠름).
- **[SpeechDemo](Examples/SpeechDemo/)** — 탭형 인터페이스에서 받아쓰기와 TTS 합성. macOS.

각 데모의 README에 빌드 방법이 있습니다.

## 코드 예제

아래 스니펫은 각 도메인의 최소 사용 경로를 보여줍니다. 각 섹션은 [soniqo.audio](https://soniqo.audio/ko)의 전체 가이드로 링크되어 있으며, 설정 옵션, 다양한 백엔드, 스트리밍 패턴 및 CLI 레시피를 다룹니다.

### 음성-텍스트 변환 — [전체 가이드 →](https://soniqo.audio/ko/guides/transcribe)

```swift
import Qwen3ASR

let model = try await Qwen3ASRModel.fromPretrained()
let text = model.transcribe(audio: audioSamples, sampleRate: 16000)
```

대체 백엔드: [Parakeet TDT](https://soniqo.audio/ko/guides/parakeet) (CoreML, 32× 실시간), [Omnilingual ASR](https://soniqo.audio/ko/guides/omnilingual) (1,672개 언어, CoreML 또는 MLX), [스트리밍 받아쓰기](https://soniqo.audio/ko/guides/dictate) (라이브 파셜).

### 강제 정렬 — [전체 가이드 →](https://soniqo.audio/ko/guides/align)

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

### 텍스트-음성 변환 — [전체 가이드 →](https://soniqo.audio/ko/guides/speak)

```swift
import Qwen3TTS
import AudioCommon

let model = try await Qwen3TTSModel.fromPretrained()
let audio = model.synthesize(text: "Hello world", language: "english")
try WAVWriter.write(samples: audio, sampleRate: 24000, to: outputURL)
```

대체 TTS 엔진: [CosyVoice3](https://soniqo.audio/ko/guides/cosyvoice) (스트리밍 + 음성 복제 + 감정 태그), [Kokoro-82M](https://soniqo.audio/ko/guides/kokoro) (iOS 지원, 54개 음색), [음성 복제](https://soniqo.audio/ko/guides/voice-cloning).

### 음성-음성 변환 — [전체 가이드 →](https://soniqo.audio/ko/guides/respond)

```swift
import PersonaPlex

let model = try await PersonaPlexModel.fromPretrained()
let responseAudio = model.respond(userAudio: userSamples)
// 24 kHz 모노 Float32 출력, 재생 준비 완료
```

### LLM 채팅 — [전체 가이드 →](https://soniqo.audio/ko/guides/chat)

```swift
import Qwen3Chat

let chat = try await Qwen35MLXChat.fromPretrained()
chat.chat(messages: [(.user, "Explain MLX in one sentence")]) { token, isFinal in
    print(token, terminator: "")
}
```

### 음성 활동 감지 — [전체 가이드 →](https://soniqo.audio/ko/guides/vad)

```swift
import SpeechVAD

let vad = try await SileroVADModel.fromPretrained()
let segments = vad.detectSpeech(audio: samples, sampleRate: 16000)
for s in segments { print("\(s.startTime)s → \(s.endTime)s") }
```

### 화자 분리 — [전체 가이드 →](https://soniqo.audio/ko/guides/diarize)

```swift
import SpeechVAD

let diarizer = try await DiarizationPipeline.fromPretrained()
let segments = diarizer.diarize(audio: samples, sampleRate: 16000)
for s in segments { print("Speaker \(s.speakerId): \(s.startTime)s - \(s.endTime)s") }
```

### 음성 향상 — [전체 가이드 →](https://soniqo.audio/ko/guides/denoise)

```swift
import SpeechEnhancement

let denoiser = try await DeepFilterNet3Model.fromPretrained()
let clean = try denoiser.enhance(audio: noisySamples, sampleRate: 48000)
```

### 음성 파이프라인 (ASR → LLM → TTS) — [전체 가이드 →](https://soniqo.audio/ko/api)

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

`VoicePipeline`은 실시간 음성 agent 상태 머신으로 ([speech-core](https://github.com/soniqo/speech-core)로 구동), VAD 기반 턴 감지, 인터럽션 처리, 이거(eager) STT를 지원합니다. 임의의 `SpeechRecognitionModel` + `SpeechGenerationModel` + `StreamingVADProvider`를 연결할 수 있습니다.

### HTTP API 서버

```bash
audio-server --port 8080
```

HTTP REST + WebSocket 엔드포인트로 모든 모델을 공개합니다. OpenAI Realtime API 호환 WebSocket `/v1/realtime`도 포함됩니다. [`Sources/AudioServer/`](Sources/AudioServer/)를 참조하세요.

## 아키텍처

speech-swift는 모델당 하나의 SPM 타겟으로 분리되어 있어 사용자는 임포트한 것에 대해서만 비용을 지불합니다. 공유 인프라는 `AudioCommon` (프로토콜, 오디오 I/O, HuggingFace 다운로더, `SentencePieceModel`)과 `MLXCommon` (웨이트 로딩, `QuantizedLinear` 헬퍼, `SDPA` 멀티헤드 어텐션 헬퍼)에 있습니다.

**[백엔드, 메모리 테이블, 모듈 맵이 포함된 전체 아키텍처 다이어그램 → soniqo.audio/architecture](https://soniqo.audio/ko/architecture)** · **[API 레퍼런스 → soniqo.audio/api](https://soniqo.audio/ko/api)** · **[벤치마크 → soniqo.audio/benchmarks](https://soniqo.audio/ko/benchmarks)**

로컬 문서 (리포지토리):
- **모델:** [Qwen3-ASR](docs/models/asr-model.md) · [Qwen3-TTS](docs/models/tts-model.md) · [CosyVoice](docs/models/cosyvoice-tts.md) · [Kokoro](docs/models/kokoro-tts.md) · [Parakeet TDT](docs/models/parakeet-asr.md) · [Parakeet Streaming](docs/models/parakeet-streaming-asr.md) · [Nemotron Streaming](docs/models/nemotron-streaming.md) · [Omnilingual ASR](docs/models/omnilingual-asr.md) · [PersonaPlex](docs/models/personaplex.md) · [FireRedVAD](docs/models/fireredvad.md)
- **추론:** [Qwen3-ASR](docs/inference/qwen3-asr-inference.md) · [Parakeet TDT](docs/inference/parakeet-asr-inference.md) · [Parakeet Streaming](docs/inference/parakeet-streaming-asr-inference.md) · [Nemotron Streaming](docs/inference/nemotron-streaming-inference.md) · [Omnilingual ASR](docs/inference/omnilingual-asr-inference.md) · [TTS](docs/inference/qwen3-tts-inference.md) · [Forced Aligner](docs/inference/forced-aligner.md) · [Silero VAD](docs/inference/silero-vad.md) · [화자 분리](docs/inference/speaker-diarization.md) · [음성 향상](docs/inference/speech-enhancement.md)
- **레퍼런스:** [공유 프로토콜](docs/shared-protocols.md)

## 캐시 설정

모델 웨이트는 첫 사용 시 HuggingFace에서 다운로드되어 `~/Library/Caches/qwen3-speech/`에 캐시됩니다. `QWEN3_CACHE_DIR` (CLI) 또는 `cacheDir:` (Swift API)로 재정의할 수 있습니다. 모든 `fromPretrained()` 엔트리 포인트는 `offlineMode: true`를 지원하여 웨이트가 이미 캐시되어 있으면 네트워크를 건너뜁니다.

샌드박스 iOS 컨테이너 경로를 포함한 자세한 내용은 [`docs/inference/cache-and-offline.md`](docs/inference/cache-and-offline.md)를 참조하세요.

## MLX Metal 라이브러리

런타임에 `Failed to load the default metallib`이 보이면 Metal 셰이더 라이브러리가 없습니다. 수동 `swift build` 후에 `make build` 또는 `./scripts/build_mlx_metallib.sh release`를 실행하세요. Metal Toolchain이 없다면 먼저 설치하세요:

```bash
xcodebuild -downloadComponent MetalToolchain
```

## 테스트

```bash
make test                            # 전체 스위트 (단위 + 모델 다운로드 포함 E2E)
swift test --skip E2E                # 단위만 (CI 안전, 다운로드 없음)
swift test --filter Qwen3ASRTests    # 특정 모듈
```

E2E 테스트 클래스는 `E2E` 접두사를 사용하므로 CI는 `--skip E2E`로 필터링할 수 있습니다. 전체 테스트 규약은 [CLAUDE.md](CLAUDE.md#testing)를 참조하세요.

## 기여

PR 환영합니다 — 버그 수정, 새 모델 통합, 문서. fork, 피처 브랜치 생성, `make build && make test`, `main`에 대해 PR을 여세요.

## 라이선스

Apache 2.0
