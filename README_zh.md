# Speech Swift

面向 Apple Silicon 的 AI 语音模型，基于 MLX Swift 和 CoreML。

📖 Read in: [English](README.md) · [中文](README_zh.md) · [日本語](README_ja.md) · [한국어](README_ko.md) · [Español](README_es.md) · [Deutsch](README_de.md) · [Français](README_fr.md) · [हिन्दी](README_hi.md) · [Português](README_pt.md) · [Русский](README_ru.md)

端侧语音识别、合成与理解，适用于 Mac 与 iOS。完全在 Apple Silicon 上本地运行——无需云端、无需 API 密钥、数据不出设备。

**[📚 完整文档 →](https://soniqo.audio/zh)** · **[🤗 HuggingFace 模型](https://huggingface.co/aufklarer)** · **[📝 博客](https://blog.ivan.digital)**

- **[Qwen3-ASR](https://soniqo.audio/zh/guides/transcribe)** — 语音转文字（自动语音识别，52 种语言，MLX + CoreML）
- **[Parakeet TDT](https://soniqo.audio/zh/guides/parakeet)** — 通过 CoreML 进行语音转文字（神经引擎，NVIDIA FastConformer + TDT 解码器，25 种语言）
- **[Omnilingual ASR](https://soniqo.audio/zh/guides/omnilingual)** — 语音转文字（Meta wav2vec2 + CTC，**1,672 种语言**，覆盖 32 种文字系统，CoreML 300M + MLX 300M/1B/3B/7B）
- **[流式听写](https://soniqo.audio/zh/guides/dictate)** — 带部分结果和句末检测的实时听写（Parakeet-EOU-120M）
- **[Qwen3-ForcedAligner](https://soniqo.audio/zh/guides/align)** — 词级时间戳对齐（音频 + 文本 → 时间戳）
- **[Qwen3-TTS](https://soniqo.audio/zh/guides/speak)** — 文本转语音（最高质量、流式输出、自定义说话人，10 种语言）
- **[CosyVoice TTS](https://soniqo.audio/zh/guides/cosyvoice)** — 流式 TTS，支持声音克隆、多说话人对话、情感标签（9 种语言）
- **[Kokoro TTS](https://soniqo.audio/zh/guides/kokoro)** — 端侧 TTS（82M，CoreML/神经引擎，54 种音色，iOS 就绪，10 种语言）
- **[Qwen3.5-Chat](https://soniqo.audio/zh/guides/chat)** — 端侧 LLM 对话（0.8B，MLX INT4 + CoreML INT8，DeltaNet 混合架构，流式 token）
- **[PersonaPlex](https://soniqo.audio/zh/guides/respond)** — 全双工语音到语音（7B，音频输入 → 音频输出，18 种预设音色）
- **[DeepFilterNet3](https://soniqo.audio/zh/guides/denoise)** — 实时噪声抑制（2.1M 参数，48 kHz）
- **[唤醒词](https://soniqo.audio/zh/guides/wake-word)** — 设备端关键词识别（KWS Zipformer 3M，CoreML，26× 实时，可配置关键词列表）
- **[VAD](https://soniqo.audio/zh/guides/vad)** — 语音活动检测（Silero 流式、Pyannote 离线、FireRedVAD 100+ 种语言）
- **[说话人分离](https://soniqo.audio/zh/guides/diarize)** — 谁在什么时间说话（Pyannote 流水线，神经引擎上的端到端 Sortformer）
- **[说话人嵌入向量](https://soniqo.audio/zh/guides/embed-speaker)** — WeSpeaker ResNet34（256 维）、CAM++（192 维）

论文：[Qwen3-ASR](https://arxiv.org/abs/2601.21337) (Alibaba) · [Qwen3-TTS](https://arxiv.org/abs/2601.15621) (Alibaba) · [Omnilingual ASR](https://arxiv.org/abs/2511.09690) (Meta) · [Parakeet TDT](https://arxiv.org/abs/2304.06795) (NVIDIA) · [CosyVoice 3](https://arxiv.org/abs/2505.17589) (Alibaba) · [Kokoro](https://arxiv.org/abs/2301.01695) (StyleTTS 2) · [PersonaPlex](https://arxiv.org/abs/2602.06053) (NVIDIA) · [Mimi](https://arxiv.org/abs/2410.00037) (Kyutai) · [Sortformer](https://arxiv.org/abs/2409.06656) (NVIDIA)

## 动态

- **2026 年 3 月 20 日** — [我们用一个 600M 模型在 Mac 上击败了 Whisper Large v3](https://blog.ivan.digital/we-beat-whisper-large-v3-with-a-600m-model-running-entirely-on-your-mac-20e6ce191174)
- **2026 年 2 月 26 日** — [Apple Silicon 上的说话人分离与语音活动检测——基于 MLX 的原生 Swift 实现](https://blog.ivan.digital/speaker-diarization-and-voice-activity-detection-on-apple-silicon-native-swift-with-mlx-92ea0c9aca0f)
- **2026 年 2 月 23 日** — [NVIDIA PersonaPlex 7B 在 Apple Silicon 上运行——基于 MLX 的原生 Swift 全双工语音到语音](https://blog.ivan.digital/nvidia-personaplex-7b-on-apple-silicon-full-duplex-speech-to-speech-in-native-swift-with-mlx-0aa5276f2e23)
- **2026 年 2 月 12 日** — [Qwen3-ASR Swift：面向 Apple Silicon 的端侧 ASR + TTS——架构与基准测试](https://blog.ivan.digital/qwen3-asr-swift-on-device-asr-tts-for-apple-silicon-architecture-and-benchmarks-27cbf1e4463f)

## 快速开始

将依赖添加到你的 `Package.swift`：

```swift
.package(url: "https://github.com/soniqo/speech-swift", from: "0.0.9")
```

只引入你需要的模块——每个模型都是独立的 SPM 库，不用为你不使用的东西买单：

```swift
.product(name: "ParakeetStreamingASR", package: "speech-swift"),
.product(name: "SpeechUI",             package: "speech-swift"),  // 可选的 SwiftUI 视图
```

**3 行代码转写音频缓冲区：**

```swift
import ParakeetStreamingASR

let model = try await ParakeetStreamingASRModel.fromPretrained()
let text = try model.transcribeAudio(audioSamples, sampleRate: 16000)
```

**带部分结果的实时流式：**

```swift
for await partial in model.transcribeStream(audio: samples, sampleRate: 16000) {
    print(partial.isFinal ? "FINAL: \(partial.text)" : "... \(partial.text)")
}
```

**约 10 行写出 SwiftUI 听写视图：**

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

`SpeechUI` 只提供 `TranscriptionView`（最终结果 + 部分结果）与 `TranscriptionStore`（流式 ASR 适配器）。音频可视化和播放请使用 AVFoundation。

可用的 SPM products：`Qwen3ASR`、`Qwen3TTS`、`Qwen3TTSCoreML`、`ParakeetASR`、`ParakeetStreamingASR`、`OmnilingualASR`、`KokoroTTS`、`CosyVoiceTTS`、`PersonaPlex`、`SpeechVAD`、`SpeechEnhancement`、`Qwen3Chat`、`SpeechCore`、`SpeechUI`、`AudioCommon`。

## 模型

下方是精简视图。**[完整模型目录（含大小、量化方式、下载地址、内存表）→ soniqo.audio/architecture](https://soniqo.audio/zh/architecture)**。

| 模型 | 任务 | 后端 | 大小 | 语言 |
|-------|------|----------|-------|-----------|
| [Qwen3-ASR](https://soniqo.audio/zh/guides/transcribe) | 语音 → 文字 | MLX、CoreML（混合） | 0.6B、1.7B | 52 |
| [Parakeet TDT](https://soniqo.audio/zh/guides/parakeet) | 语音 → 文字 | CoreML (ANE) | 0.6B | 25 种欧洲语言 |
| [Parakeet EOU](https://soniqo.audio/zh/guides/dictate) | 语音 → 文字（流式） | CoreML (ANE) | 120M | 25 种欧洲语言 |
| [Omnilingual ASR](https://soniqo.audio/zh/guides/omnilingual) | 语音 → 文字 | CoreML (ANE)、MLX | 300M / 1B / 3B / 7B | **[1,672](https://github.com/facebookresearch/omnilingual-asr/blob/main/src/omnilingual_asr/models/wav2vec2_llama/lang_ids.py)** |
| [Qwen3-ForcedAligner](https://soniqo.audio/zh/guides/align) | 音频 + 文本 → 时间戳 | MLX、CoreML | 0.6B | 多语言 |
| [Qwen3-TTS](https://soniqo.audio/zh/guides/speak) | 文本 → 语音 | MLX、CoreML | 0.6B、1.7B | 10 |
| [CosyVoice3](https://soniqo.audio/zh/guides/cosyvoice) | 文本 → 语音 | MLX | 0.5B | 9 |
| [Kokoro-82M](https://soniqo.audio/zh/guides/kokoro) | 文本 → 语音 | CoreML (ANE) | 82M | 10 |
| [Qwen3.5-Chat](https://soniqo.audio/zh/guides/chat) | 文本 → 文本（LLM） | MLX、CoreML | 0.8B | 多语言 |
| [PersonaPlex](https://soniqo.audio/zh/guides/respond) | 语音 → 语音 | MLX | 7B | EN |
| [Silero VAD](https://soniqo.audio/zh/guides/vad) | 语音活动检测 | MLX、CoreML | 309K | 语言无关 |
| [Pyannote](https://soniqo.audio/zh/guides/diarize) | VAD + 说话人分离 | MLX | 1.5M | 语言无关 |
| [Sortformer](https://soniqo.audio/zh/guides/diarize) | 说话人分离（端到端） | CoreML (ANE) | — | 语言无关 |
| [DeepFilterNet3](https://soniqo.audio/zh/guides/denoise) | 语音增强 | CoreML | 2.1M | 语言无关 |
| [WeSpeaker](https://soniqo.audio/zh/guides/embed-speaker) | 说话人嵌入向量 | MLX、CoreML | 6.6M | 语言无关 |

## 安装

### Homebrew

需要原生 ARM Homebrew（`/opt/homebrew`），不支持 Rosetta/x86_64 Homebrew。

```bash
brew tap soniqo/speech https://github.com/soniqo/speech-swift
brew install speech
```

然后：

```bash
audio transcribe recording.wav
audio speak "Hello world"
audio respond --input question.wav --transcript
```

**[完整 CLI 参考 →](https://soniqo.audio/zh/cli)**

### Swift Package Manager

```swift
dependencies: [
    .package(url: "https://github.com/soniqo/speech-swift", from: "0.0.9")
]
```

只引入你需要的模块——每个模型都是独立的 SPM target：

```swift
import Qwen3ASR             // 语音识别 (MLX)
import ParakeetASR          // 语音识别 (CoreML，批量)
import ParakeetStreamingASR // 带部分结果和 EOU 的流式听写
import OmnilingualASR       // 1,672 种语言 (CoreML + MLX)
import Qwen3TTS             // 文本转语音
import CosyVoiceTTS         // 带声音克隆的文本转语音
import KokoroTTS            // 文本转语音 (iOS 就绪)
import Qwen3Chat            // 端侧 LLM 对话
import PersonaPlex          // 全双工语音到语音
import SpeechVAD            // VAD + 说话人分离 + 嵌入向量
import SpeechEnhancement    // 噪声抑制
import SpeechUI             // 流式转写的 SwiftUI 组件
import AudioCommon          // 共享协议与工具
```

### 环境要求

- Swift 5.9+、Xcode 15+（含 Metal Toolchain）
- macOS 14+ 或 iOS 17+、Apple Silicon（M1/M2/M3/M4）

### 从源代码构建

```bash
git clone https://github.com/soniqo/speech-swift
cd speech-swift
make build
```

`make build` 会同时编译 Swift 包**和** MLX Metal 着色器库。Metal 库是 GPU 推理所必需的——若缺失，运行时会出现 `Failed to load the default metallib`。`make debug` 用于调试构建，`make test` 运行测试套件。

**[完整构建与安装指南 →](https://soniqo.audio/zh/getting-started)**

## 演示应用

- **[DictateDemo](Examples/DictateDemo/)**（[文档](https://soniqo.audio/zh/guides/dictate)）— macOS 菜单栏流式听写，带实时部分结果、VAD 驱动的句末检测、一键复制。以后台 agent 方式运行（Parakeet-EOU-120M + Silero VAD）。
- **[iOSEchoDemo](Examples/iOSEchoDemo/)** — iOS 回声 demo（Parakeet ASR + Kokoro TTS）。支持设备和模拟器。
- **[PersonaPlexDemo](Examples/PersonaPlexDemo/)** — 带麦克风输入、VAD 和多轮上下文的对话式语音助手。macOS。M2 Max 上 RTF 约 0.94（快于实时）。
- **[SpeechDemo](Examples/SpeechDemo/)** — 标签式界面中的听写与 TTS 合成。macOS。

每个 demo 的 README 都包含构建说明。

## 代码示例

下方代码片段展示每个领域的最小使用路径。每一节都链接到 [soniqo.audio](https://soniqo.audio/zh) 上的完整指南，涵盖配置选项、多种后端、流式模式和 CLI 示例。

### 语音转文字 — [完整指南 →](https://soniqo.audio/zh/guides/transcribe)

```swift
import Qwen3ASR

let model = try await Qwen3ASRModel.fromPretrained()
let text = model.transcribe(audio: audioSamples, sampleRate: 16000)
```

其他后端：[Parakeet TDT](https://soniqo.audio/zh/guides/parakeet)（CoreML，32× 实时）、[Omnilingual ASR](https://soniqo.audio/zh/guides/omnilingual)（1,672 种语言，CoreML 或 MLX）、[流式听写](https://soniqo.audio/zh/guides/dictate)（实时部分结果）。

### 强制对齐 — [完整指南 →](https://soniqo.audio/zh/guides/align)

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

### 文本转语音 — [完整指南 →](https://soniqo.audio/zh/guides/speak)

```swift
import Qwen3TTS
import AudioCommon

let model = try await Qwen3TTSModel.fromPretrained()
let audio = model.synthesize(text: "Hello world", language: "english")
try WAVWriter.write(samples: audio, sampleRate: 24000, to: outputURL)
```

其他 TTS 引擎：[CosyVoice3](https://soniqo.audio/zh/guides/cosyvoice)（流式 + 声音克隆 + 情感标签）、[Kokoro-82M](https://soniqo.audio/zh/guides/kokoro)（iOS 就绪，54 种音色）、[声音克隆](https://soniqo.audio/zh/guides/voice-cloning)。

### 语音到语音 — [完整指南 →](https://soniqo.audio/zh/guides/respond)

```swift
import PersonaPlex

let model = try await PersonaPlexModel.fromPretrained()
let responseAudio = model.respond(userAudio: userSamples)
// 24 kHz 单声道 Float32 输出，可直接播放
```

### LLM 对话 — [完整指南 →](https://soniqo.audio/zh/guides/chat)

```swift
import Qwen3Chat

let chat = try await Qwen35MLXChat.fromPretrained()
chat.chat(messages: [(.user, "Explain MLX in one sentence")]) { token, isFinal in
    print(token, terminator: "")
}
```

### 语音活动检测 — [完整指南 →](https://soniqo.audio/zh/guides/vad)

```swift
import SpeechVAD

let vad = try await SileroVADModel.fromPretrained()
let segments = vad.detectSpeech(audio: samples, sampleRate: 16000)
for s in segments { print("\(s.startTime)s → \(s.endTime)s") }
```

### 说话人分离 — [完整指南 →](https://soniqo.audio/zh/guides/diarize)

```swift
import SpeechVAD

let diarizer = try await DiarizationPipeline.fromPretrained()
let segments = diarizer.diarize(audio: samples, sampleRate: 16000)
for s in segments { print("Speaker \(s.speakerId): \(s.startTime)s - \(s.endTime)s") }
```

### 语音增强 — [完整指南 →](https://soniqo.audio/zh/guides/denoise)

```swift
import SpeechEnhancement

let denoiser = try await DeepFilterNet3Model.fromPretrained()
let clean = try denoiser.enhance(audio: noisySamples, sampleRate: 48000)
```

### 语音流水线（ASR → LLM → TTS） — [完整指南 →](https://soniqo.audio/zh/api)

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

`VoicePipeline` 是实时语音 agent 的状态机（由 [speech-core](https://github.com/soniqo/speech-core) 提供），支持 VAD 驱动的轮次检测、打断处理和积极式 STT。它可以连接任意 `SpeechRecognitionModel` + `SpeechGenerationModel` + `StreamingVADProvider`。

### HTTP API 服务器

```bash
audio-server --port 8080
```

通过 HTTP REST + WebSocket 接口暴露每个模型，包括一个兼容 OpenAI Realtime API 的 WebSocket 接口 `/v1/realtime`。详见 [`Sources/AudioServer/`](Sources/AudioServer/)。

## 架构

speech-swift 把每个模型拆成独立的 SPM target，因此使用者只为 import 的模块付费。共享基础设施在 `AudioCommon`（协议、音频 I/O、HuggingFace 下载器、`SentencePieceModel`）和 `MLXCommon`（权重加载、`QuantizedLinear` 辅助工具、`SDPA` 多头注意力辅助工具）。

**[完整架构图（含后端、内存表、模块映射）→ soniqo.audio/architecture](https://soniqo.audio/zh/architecture)** · **[API 参考 → soniqo.audio/api](https://soniqo.audio/zh/api)** · **[基准测试 → soniqo.audio/benchmarks](https://soniqo.audio/zh/benchmarks)**

本地文档（仓库内）：
- **模型：** [Qwen3-ASR](docs/models/asr-model.md) · [Qwen3-TTS](docs/models/tts-model.md) · [CosyVoice](docs/models/cosyvoice-tts.md) · [Kokoro](docs/models/kokoro-tts.md) · [Parakeet TDT](docs/models/parakeet-asr.md) · [Parakeet Streaming](docs/models/parakeet-streaming-asr.md) · [Omnilingual ASR](docs/models/omnilingual-asr.md) · [PersonaPlex](docs/models/personaplex.md) · [FireRedVAD](docs/models/fireredvad.md)
- **推理：** [Qwen3-ASR](docs/inference/qwen3-asr-inference.md) · [Parakeet TDT](docs/inference/parakeet-asr-inference.md) · [Parakeet Streaming](docs/inference/parakeet-streaming-asr-inference.md) · [Omnilingual ASR](docs/inference/omnilingual-asr-inference.md) · [TTS](docs/inference/qwen3-tts-inference.md) · [Forced Aligner](docs/inference/forced-aligner.md) · [Silero VAD](docs/inference/silero-vad.md) · [说话人分离](docs/inference/speaker-diarization.md) · [语音增强](docs/inference/speech-enhancement.md)
- **参考：** [共享协议](docs/shared-protocols.md)

## 缓存配置

模型权重在首次使用时从 HuggingFace 下载并缓存到 `~/Library/Caches/qwen3-speech/`。可通过 `QWEN3_CACHE_DIR`（CLI）或 `cacheDir:`（Swift API）覆盖。所有 `fromPretrained()` 入口都接受 `offlineMode: true`，在权重已缓存时跳过网络。

详见 [`docs/inference/cache-and-offline.md`](docs/inference/cache-and-offline.md)，包含 iOS 沙盒容器路径等完整说明。

## MLX Metal 库

如果运行时出现 `Failed to load the default metallib`，说明缺少 Metal 着色器库。在 `swift build` 之后运行 `make build` 或 `./scripts/build_mlx_metallib.sh release`。如果缺少 Metal Toolchain，先安装：

```bash
xcodebuild -downloadComponent MetalToolchain
```

## 测试

```bash
make test                            # 完整套件（单元 + 需要下载模型的 E2E）
swift test --skip E2E                # 仅单元测试（CI 安全，无下载）
swift test --filter Qwen3ASRTests    # 指定模块
```

E2E 测试类使用 `E2E` 前缀，这样 CI 就可以用 `--skip E2E` 过滤掉它们。完整测试规范见 [CLAUDE.md](CLAUDE.md#testing)。

## 贡献

欢迎 PR——bug 修复、新模型集成、文档改进。fork、创建功能分支、`make build && make test`，然后向 `main` 提交 PR。

## 许可证

Apache 2.0
