# Speech Swift

面向 Apple Silicon 的 AI 语音模型，基于 MLX Swift 和 CoreML 构建。

📖 Read in: [English](README.md) · [中文](README_zh.md) · [日本語](README_ja.md) · [한국어](README_ko.md) · [Español](README_es.md) · [Deutsch](README_de.md) · [Français](README_fr.md) · [हिन्दी](README_hi.md) · [Português](README_pt.md) · [Русский](README_ru.md)

端侧语音识别、合成与理解，适用于 Mac 和 iOS。完全在 Apple Silicon 上本地运行——无需云端、无需 API 密钥、数据不出设备。

通过 [Homebrew 安装](#homebrew)或作为 Swift Package 依赖引入。

**[文档](https://soniqo.audio)** · **[HuggingFace 模型](https://huggingface.co/aufklarer)** · **[博客](https://blog.ivan.digital)**

- **Qwen3-ASR** — 语音转文字 / 语音识别（自动语音识别，支持 52 种语言）
- **Parakeet TDT** — 通过 CoreML 进行语音转文字（神经引擎，NVIDIA FastConformer + TDT 解码器，25 种语言）
- **Qwen3-ForcedAligner** — 词级时间戳对齐（音频 + 文本 → 时间戳）
- **Qwen3-TTS** — 文本转语音合成（最高质量，流式输出，自定义说话人，10 种语言）
- **CosyVoice TTS** — 支持流式合成、声音克隆、多说话人对话和情感标签的文本转语音（9 种语言，DiT flow matching，CAM++ 说话人编码器）
- **Kokoro TTS** — 端侧文本转语音（82M 参数，CoreML/神经引擎，54 种音色，iOS 就绪，10 种语言）
- **Qwen3.5-Chat** — 端侧 LLM 对话（0.8B，MLX + CoreML，INT4/INT8，DeltaNet 混合架构，流式 token）
- **PersonaPlex** — 全双工语音到语音对话（7B，音频输入 → 音频输出，18 种预设音色）
- **DeepFilterNet3** — 语音增强 / 噪声抑制（2.1M 参数，实时 48kHz）
- **FireRedVAD** — 离线语音活动检测（DFSMN，CoreML，100+ 种语言，97.6% F1）
- **Silero VAD** — 流式语音活动检测（32ms 分块，亚毫秒延迟）
- **Pyannote VAD** — 离线语音活动检测（10 秒窗口，多说话人重叠）
- **说话人分离** — 谁在什么时间说话（Pyannote 分割 + 基于活动的说话人链接，或基于神经引擎的端到端 Sortformer）
- **说话人嵌入向量** — 说话人验证与识别（WeSpeaker ResNet34，256 维向量）

论文：[Qwen3-ASR](https://arxiv.org/abs/2601.21337) (Alibaba), [Qwen3-TTS](https://arxiv.org/abs/2601.15621) (Alibaba), [Qwen3](https://arxiv.org/abs/2505.09388) (Alibaba), [Parakeet TDT](https://arxiv.org/abs/2304.06795) (NVIDIA), [CosyVoice 3](https://arxiv.org/abs/2505.17589) (Alibaba), [Kokoro](https://arxiv.org/abs/2301.01695) (StyleTTS 2), [PersonaPlex](https://arxiv.org/abs/2602.06053) (NVIDIA), [Mimi](https://arxiv.org/abs/2410.00037) (Kyutai), [Sortformer](https://arxiv.org/abs/2409.06656) (NVIDIA)

## 路线图

请查看[路线图讨论](https://github.com/soniqo/speech-swift/discussions/81)了解未来计划——欢迎评论和建议！

## 动态

- **2026 年 3 月 20 日** — [我们用一个 600M 模型在 Mac 上击败了 Whisper Large v3](https://blog.ivan.digital/we-beat-whisper-large-v3-with-a-600m-model-running-entirely-on-your-mac-20e6ce191174)
- **2026 年 2 月 26 日** — [Apple Silicon 上的说话人分离与语音活动检测——基于 MLX 的原生 Swift 实现](https://blog.ivan.digital/speaker-diarization-and-voice-activity-detection-on-apple-silicon-native-swift-with-mlx-92ea0c9aca0f)
- **2026 年 2 月 23 日** — [NVIDIA PersonaPlex 7B 在 Apple Silicon 上运行——基于 MLX 的原生 Swift 全双工语音到语音](https://blog.ivan.digital/nvidia-personaplex-7b-on-apple-silicon-full-duplex-speech-to-speech-in-native-swift-with-mlx-0aa5276f2e23)
- **2026 年 2 月 12 日** — [Qwen3-ASR Swift：面向 Apple Silicon 的端侧 ASR + TTS——架构与基准测试](https://blog.ivan.digital/qwen3-asr-swift-on-device-asr-tts-for-apple-silicon-architecture-and-benchmarks-27cbf1e4463f)

## 模型

| 模型 | 任务 | 流式 | 语言 | 规格 |
|------|------|------|------|------|
| Qwen3-ASR-0.6B | 语音 → 文本 | 否 | 52 种语言 | [4-bit](https://huggingface.co/aufklarer/Qwen3-ASR-0.6B-MLX-4bit) 680 MB · [8-bit](https://huggingface.co/aufklarer/Qwen3-ASR-0.6B-MLX-8bit) 1.0 GB · [CoreML](https://huggingface.co/aufklarer/Qwen3-ASR-CoreML) 180 MB |
| Qwen3-ASR-1.7B | 语音 → 文本 | 否 | 52 种语言 | [4-bit](https://huggingface.co/aufklarer/Qwen3-ASR-1.7B-MLX-4bit) 2.1 GB · [8-bit](https://huggingface.co/aufklarer/Qwen3-ASR-1.7B-MLX-8bit) 3.2 GB |
| Parakeet-TDT-0.6B | 语音 → 文本 | 否 | 25 种欧洲语言 | [CoreML INT4](https://huggingface.co/aufklarer/Parakeet-TDT-v3-CoreML-INT4) 315 MB · [CoreML INT8](https://huggingface.co/aufklarer/Parakeet-TDT-v3-CoreML-INT8) 500 MB |
| Qwen3-ForcedAligner-0.6B | 音频 + 文本 → 时间戳 | 否 | 多语言 | [4-bit](https://huggingface.co/aufklarer/Qwen3-ForcedAligner-0.6B-4bit) 979 MB · [8-bit](https://huggingface.co/aufklarer/Qwen3-ForcedAligner-0.6B-8bit) 1.4 GB · [CoreML INT4](https://huggingface.co/aufklarer/Qwen3-ForcedAligner-0.6B-CoreML-INT4) 630 MB · [CoreML INT8](https://huggingface.co/aufklarer/Qwen3-ForcedAligner-0.6B-CoreML-INT8) 1.0 GB |
| Qwen3-TTS-0.6B Base | 文本 → 语音 | 是 (~120ms) | 10 种语言 | [4-bit](https://huggingface.co/aufklarer/Qwen3-TTS-12Hz-0.6B-Base-MLX-4bit) 1.7 GB · [8-bit](https://huggingface.co/aufklarer/Qwen3-TTS-12Hz-0.6B-Base-MLX-8bit) 2.4 GB |
| Qwen3-TTS-0.6B CustomVoice | 文本 → 语音 | 是 (~120ms) | 10 种语言 | [4-bit](https://huggingface.co/aufklarer/Qwen3-TTS-12Hz-0.6B-CustomVoice-MLX-4bit) 1.7 GB |
| Qwen3-TTS-1.7B Base | 文本 → 语音 | 是 (~120ms) | 10 种语言 | [4-bit](https://huggingface.co/aufklarer/Qwen3-TTS-12Hz-1.7B-Base-MLX-4bit) 3.2 GB · [8-bit](https://huggingface.co/aufklarer/Qwen3-TTS-12Hz-1.7B-Base-MLX-8bit) 4.8 GB |
| CosyVoice3-0.5B | 文本 → 语音 | 是 (~150ms) | 9 种语言 | [4-bit](https://huggingface.co/aufklarer/CosyVoice3-0.5B-MLX-4bit) 1.2 GB |
| Kokoro-82M | 文本 → 语音 | 否 | 10 种语言 | [CoreML](https://huggingface.co/aufklarer/Kokoro-82M-CoreML) ~170 MB |
| Qwen3.5-0.8B Chat | Text → Text (LLM) | Yes (streaming) | Multi | [MLX INT4](https://huggingface.co/aufklarer/Qwen3.5-0.8B-Chat-MLX) 418 MB · [CoreML INT8](https://huggingface.co/aufklarer/Qwen3.5-0.8B-Chat-CoreML) 981 MB |
| PersonaPlex-7B | 语音 → 语音 | 是 (~2s 分块) | EN | [4-bit](https://huggingface.co/aufklarer/PersonaPlex-7B-MLX-4bit) 4.9 GB · [8-bit](https://huggingface.co/aufklarer/PersonaPlex-7B-MLX-8bit) 9.1 GB |
| FireRedVAD | 语音活动检测 | 否（离线） | 100+ 种语言 | [CoreML](https://huggingface.co/aufklarer/FireRedVAD-CoreML) ~1.2 MB |
| Silero-VAD-v5 | 语音活动检测 | 是 (32ms 分块) | 语言无关 | [MLX](https://huggingface.co/aufklarer/Silero-VAD-v5-MLX) · [CoreML](https://huggingface.co/aufklarer/Silero-VAD-v5-CoreML) ~1.2 MB |
| Pyannote-Segmentation-3.0 | VAD + 说话人分割 | 否 (10s 窗口) | 语言无关 | [MLX](https://huggingface.co/aufklarer/Pyannote-Segmentation-MLX) ~5.7 MB |
| DeepFilterNet3 | 语音增强 | 是 (10ms 帧) | 语言无关 | [CoreML FP16](https://huggingface.co/aufklarer/DeepFilterNet3-CoreML) ~4.2 MB |
| WeSpeaker-ResNet34-LM | 说话人嵌入向量 (256 维) | 否 | 语言无关 | [MLX](https://huggingface.co/aufklarer/WeSpeaker-ResNet34-LM-MLX) · [CoreML](https://huggingface.co/aufklarer/WeSpeaker-ResNet34-LM-CoreML) ~25 MB |
| CAM++ | 说话人嵌入向量 (192 维) | 否 | 语言无关 | [CoreML](https://huggingface.co/aufklarer/CamPlusPlus-Speaker-CoreML) ~14 MB |
| Sortformer | 说话人分离（端到端） | 是（分块） | 语言无关 | [CoreML](https://huggingface.co/aufklarer/Sortformer-Diarization-CoreML) ~240 MB |

### 内存需求

权重内存指模型参数占用的 GPU (MLX) 或 ANE (CoreML) 内存。峰值推理内存包括 KV 缓存、激活值和中间张量。

| 模型 | 权重内存 | 峰值推理内存 |
|------|---------|-------------|
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

### TTS 引擎选择指南

- **Qwen3-TTS**：最佳质量，流式合成 (~120ms)，9 种内置说话人，10 种语言，批量合成
- **CosyVoice TTS**：流式合成 (~150ms)，9 种语言，声音克隆（CAM++ 说话人编码器），多说话人对话（`[S1] ... [S2] ...`），行内情感/风格标签（`(happy)`、`(whispers)`），DiT flow matching + HiFi-GAN 声码器
- **Kokoro TTS**：轻量级 iOS 就绪 TTS（82M 参数），CoreML/神经引擎，54 种音色，10 种语言，非自回归（3 阶段流水线）
- **PersonaPlex**：全双工语音到语音（音频输入 → 音频输出），流式 (~2s 分块)，18 种预设音色，基于 Moshi 架构

## 安装

### Homebrew

需要原生 ARM Homebrew (`/opt/homebrew`)。不支持 Rosetta/x86_64 Homebrew。

```bash
brew tap soniqo/speech https://github.com/soniqo/speech-swift
brew install speech
```

使用方式：

```bash
audio transcribe recording.wav
audio speak "Hello world"
audio speak "Hallo Welt" --engine cosyvoice --language german
audio respond --input question.wav --transcript
```

> 如需使用麦克风进行交互式语音对话，请参见 **[PersonaPlexDemo](Examples/PersonaPlexDemo/)**。

### Swift Package Manager

在 `Package.swift` 中添加：

```swift
dependencies: [
    .package(url: "https://github.com/soniqo/speech-swift", branch: "main")
]
```

导入所需模块：

```swift
import Qwen3ASR      // 语音识别 (MLX)
import ParakeetASR   // 语音识别 (CoreML)
import Qwen3TTS      // 文本转语音 (Qwen3)
import CosyVoiceTTS  // 文本转语音（流式）
import KokoroTTS     // 文本转语音 (CoreML, iOS 就绪)
import Qwen3Chat     // 端侧 LLM 对话 (CoreML)
import PersonaPlex   // 语音到语音（全双工）
import SpeechVAD          // 语音活动检测 (pyannote + Silero)
import SpeechEnhancement  // 噪声抑制 (DeepFilterNet3)
import AudioCommon        // 共享工具库
```

### 系统要求

- Swift 5.9+
- macOS 14+ 或 iOS 17+
- Apple Silicon (M1/M2/M3/M4)
- Xcode 15+（需安装 Metal Toolchain——如缺失请运行 `xcodebuild -downloadComponent MetalToolchain`）

### 从源码构建

```bash
git clone https://github.com/soniqo/speech-swift
cd speech-swift
make build
```

此命令一步完成 Swift 包编译**和** MLX Metal 着色器库构建。Metal 库 (`mlx.metallib`) 是 GPU 推理所必需的——缺少它运行时会报 `Failed to load the default metallib` 错误。

调试构建：`make debug`。运行单元测试：`make test`。

## 体验语音助手

**[PersonaPlexDemo](Examples/PersonaPlexDemo/)** 是一个即开即用的 macOS 语音助手——点击说话，实时获取语音回复。使用麦克风输入配合 Silero VAD 自动检测语音，Qwen3-ASR 进行转录，PersonaPlex 7B 生成语音到语音的回复。支持多轮对话，18 种预设音色，并可显示内心独白转录。

```bash
make build  # 在仓库根目录运行——构建所有内容包括 MLX metallib
cd Examples/PersonaPlexDemo
# 参见 Examples/PersonaPlexDemo/README.md 了解 .app 打包说明
```

> M2 Max 上 RTF ~0.94（快于实时）。首次运行时模型自动下载（PersonaPlex ~5.5 GB + ASR ~400 MB）。

## 示例应用

- **[PersonaPlexDemo](Examples/PersonaPlexDemo/)** — 对话式语音助手（麦克风输入、VAD、多轮对话）。详见上文。
- **[SpeechDemo](Examples/SpeechDemo/)** — 听写（Parakeet TDT / Qwen3-ASR 自动语言检测）与文本转语音合成（Qwen3-TTS），标签式界面。

构建并以 macOS `.app` 包运行——各示例应用的 README 中有详细说明。

## 语音转文字 (ASR)——Swift 音频转录

### 基础转录

```swift
import Qwen3ASR

// 默认：0.6B 模型
let model = try await Qwen3ASRModel.fromPretrained()

// 或使用更大的 1.7B 模型以获得更高精度
let model = try await Qwen3ASRModel.fromPretrained(
    modelId: "aufklarer/Qwen3-ASR-1.7B-MLX-8bit"
)

// 音频可以是任意采样率——内部自动重采样至 16kHz
let transcription = model.transcribe(audio: audioSamples, sampleRate: 16000)
print(transcription)
```

### CoreML 编码器（神经引擎）

混合模式：CoreML 编码器运行在神经引擎上 + MLX 文本解码器运行在 GPU 上。更低功耗，编码器推理时释放 GPU。

```swift
import Qwen3ASR

let encoder = try await CoreMLASREncoder.fromPretrained()
let model = try await Qwen3ASRModel.fromPretrained()
let text = try model.transcribe(audio: audioSamples, sampleRate: 16000, coremlEncoder: encoder)
```

提供 INT8（180 MB，默认）和 INT4（90 MB）两种量化版本。推荐使用 INT8（与 FP32 的余弦相似度 > 0.999）。

### Parakeet TDT (CoreML)

```swift
import ParakeetASR

let model = try await ParakeetASRModel.fromPretrained()
let transcription = model.transcribe(audio: audioSamples, sampleRate: 16000)
```

通过 CoreML 在神经引擎上运行——释放 GPU 用于并行任务。支持 25 种欧洲语言，约 315 MB。

### ASR 命令行

```bash
make build  # 或: swift build -c release && ./scripts/build_mlx_metallib.sh release

# 默认 (Qwen3-ASR 0.6B, MLX)
.build/release/audio transcribe audio.wav

# 使用 1.7B 模型
.build/release/audio transcribe audio.wav --model 1.7B

# CoreML 编码器（神经引擎 + MLX 解码器）
.build/release/audio transcribe --engine qwen3-coreml audio.wav

# Parakeet TDT (CoreML, 神经引擎)
.build/release/audio transcribe --engine parakeet audio.wav
```

## 强制对齐

### 词级时间戳

```swift
import Qwen3ASR

let aligner = try await Qwen3ForcedAligner.fromPretrained()
// 首次运行下载约 979 MB

let aligned = aligner.align(
    audio: audioSamples,
    text: "Can you guarantee that the replacement part will be shipped tomorrow?",
    sampleRate: 24000
)

for word in aligned {
    print("[\(String(format: "%.2f", word.startTime))s - \(String(format: "%.2f", word.endTime))s] \(word.text)")
}
```

### 强制对齐命令行

```bash
swift build -c release

# 使用提供的文本进行对齐
.build/release/audio align audio.wav --text "Hello world"

# 先转录，再对齐
.build/release/audio align audio.wav
```

输出：
```
[0.12s - 0.45s] Can
[0.45s - 0.72s] you
[0.72s - 1.20s] guarantee
...
```

非自回归——3 阶段流水线，无采样循环。架构详情请参阅[强制对齐器](docs/inference/forced-aligner.md)。

## 文本转语音 (TTS)——Swift 语音合成

### 基础合成

```swift
import Qwen3TTS
import AudioCommon  // for WAVWriter

let model = try await Qwen3TTSModel.fromPretrained()
// 首次运行下载约 1.7 GB（模型 + 编解码器权重）
let audio = model.synthesize(text: "Hello world", language: "english")
// 输出为 24kHz 单声道浮点采样
try WAVWriter.write(samples: audio, sampleRate: 24000, to: outputURL)
```

### TTS 命令行

```bash
make build
.build/release/audio speak "Hello world" --output output.wav --language english
```

### 自定义音色 / 说话人选择

**CustomVoice** 模型变体支持 9 种内置说话人音色和自然语言指令控制语调/风格。通过传入 CustomVoice 模型 ID 加载：

```swift
import Qwen3TTS

// 加载 CustomVoice 模型（首次运行下载约 1.7 GB）
let model = try await Qwen3TTSModel.fromPretrained(
    modelId: TTSModelVariant.customVoice.rawValue
)

// 使用指定说话人合成
let audio = model.synthesize(text: "Hello world", language: "english", speaker: "vivian")

// 列出可用说话人
print(model.availableSpeakers)  // ["aiden", "dylan", "eric", ...]
```

命令行：

```bash
# 使用 CustomVoice 模型并指定说话人
.build/release/audio speak "Hello world" --model customVoice --speaker vivian --output vivian.wav

# 列出可用说话人
.build/release/audio speak --model customVoice --list-speakers
```

### 声音克隆（Base 模型）

从参考音频克隆说话人的声音。两种模式：

**ICL 模式**（推荐）——使用转录文本将参考音频编码为编解码器 token。质量更高，EOS 可靠：

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

**X-vector 模式**——仅使用说话人嵌入向量，无需转录文本但质量较低：

```swift
let audio = model.synthesizeWithVoiceClone(
    text: "Hello world",
    referenceAudio: refAudio,
    referenceSampleRate: 24000,
    language: "english"
)
```

命令行：

```bash
.build/release/audio speak "Hello world" --voice-sample reference.wav --output cloned.wav
```

### 语调 / 风格指令（仅 CustomVoice）

CustomVoice 模型支持自然语言 `instruct` 参数来控制说话风格、语调、情感和语速。指令以 ChatML 格式添加到模型输入的前面。

```swift
// 欢快语调
let audio = model.synthesize(
    text: "Welcome to our store!",
    language: "english",
    speaker: "ryan",
    instruct: "Speak in a cheerful, upbeat tone"
)

// 缓慢而严肃
let audio = model.synthesize(
    text: "We regret to inform you...",
    language: "english",
    speaker: "aiden",
    instruct: "Read this slowly and solemnly"
)

// 耳语
let audio = model.synthesize(
    text: "Can you keep a secret?",
    language: "english",
    speaker: "vivian",
    instruct: "Whisper this softly"
)
```

命令行：

```bash
# 带风格指令
.build/release/audio speak "Good morning!" --model customVoice --speaker ryan \
    --instruct "Speak in a cheerful, upbeat tone" --output cheerful.wav

# 使用 CustomVoice 时默认自动应用 instruct ("Speak naturally.")
.build/release/audio speak "Hello world" --model customVoice --speaker ryan --output natural.wav
```

使用 CustomVoice 模型时若未提供 `--instruct`，会自动应用 `"Speak naturally."` 以防止输出过长。Base 模型不支持 instruct。

### 批量合成

在单次批量前向推理中合成多段文本，提高吞吐量：

```swift
let texts = ["Good morning everyone.", "The weather is nice today.", "Please open the window."]
let audioList = model.synthesizeBatch(texts: texts, language: "english", maxBatchSize: 4)
// audioList[i] 是 texts[i] 对应的 24kHz 单声道浮点采样
for (i, audio) in audioList.enumerated() {
    try WAVWriter.write(samples: audio, sampleRate: 24000, to: URL(fileURLWithPath: "output_\(i).wav"))
}
```

#### 批量命令行

```bash
# 创建一个每行一段文本的文件
echo "Hello world.\nGoodbye world." > texts.txt
.build/release/audio speak --batch-file texts.txt --output output.wav --batch-size 4
# 生成 output_0.wav, output_1.wav, ...
```

> 批量模式将模型权重加载成本分摊到多个项目上。在 Apple Silicon 上 B=4 时预计可获得约 1.5-2.5 倍的吞吐量提升。文本生成长度相近时效果最佳。

### 采样选项

```swift
let config = SamplingConfig(temperature: 0.9, topK: 50, repetitionPenalty: 1.05)
let audio = model.synthesize(text: "Hello", language: "english", sampling: config)
```

### 流式合成

增量输出音频分块，降低首包延迟：

```swift
let stream = model.synthesizeStream(
    text: "Hello, this is streaming synthesis.",
    language: "english",
    streaming: .lowLatency  // 首个音频分块约 120ms
)

for try await chunk in stream {
    // chunk.samples: [Float] PCM @ 24kHz
    // chunk.isFinal: 最后一个分块时为 true
    playAudio(chunk.samples)
}
```

命令行：

```bash
# 默认流式（3 帧首块，约 225ms 延迟）
.build/release/audio speak "Hello world" --stream

# 低延迟（1 帧首块，约 120ms 延迟）
.build/release/audio speak "Hello world" --stream --first-chunk-frames 1
```

## 语音到语音——全双工语音对话

> 如需使用麦克风进行交互式语音助手体验，请参见 **[PersonaPlexDemo](Examples/PersonaPlexDemo/)**——点击说话，支持多轮对话和自动语音检测。

### 语音到语音

```swift
import PersonaPlex
import AudioCommon  // for WAVWriter, AudioFileLoader

let model = try await PersonaPlexModel.fromPretrained()
// 首次运行下载约 5.5 GB（temporal 4-bit + depformer + Mimi 编解码器 + 音色预设）

let audio = try AudioFileLoader.load(url: inputURL, targetSampleRate: 24000)
let (response, textTokens) = model.respond(userAudio: audio, voice: .NATM0)
// response: 24kHz 单声道浮点采样
// textTokens: 模型的内心独白（SentencePiece token ID）
try WAVWriter.write(samples: response.audio, sampleRate: 24000, to: outputURL)
```

### 内心独白（文本输出）

PersonaPlex 在生成音频的同时也生成文本 token——这是模型的内部推理过程。使用内置的 SentencePiece 解码器进行解码：

```swift
let decoder = try SentencePieceDecoder(modelPath: "tokenizer_spm_32k_3.model")
let transcript = decoder.decode(textTokens)
print(transcript)  // 例如 "Sure, I can help you with that..."
```

### 流式语音到语音

```swift
// 在生成过程中接收音频分块（每块约 2 秒）
let stream = model.respondStream(userAudio: audio, voice: .NATM0)
for try await chunk in stream {
    playAudio(chunk.samples)  // 即时播放，24kHz 单声道
    // chunk.textTokens 包含此分块的文本；最后一个分块包含所有 token
    if chunk.isFinal { break }
}
```

### 音色选择

提供 18 种预设音色：
- **自然女声**：NATF0, NATF1, NATF2, NATF3
- **自然男声**：NATM0, NATM1, NATM2, NATM3
- **多样女声**：VARF0, VARF1, VARF2, VARF3, VARF4
- **多样男声**：VARM0, VARM1, VARM2, VARM3, VARM4

### 系统提示词

系统提示词引导模型的对话行为。可以将任意自定义提示词作为纯字符串传入：

```swift
// 自定义系统提示词（自动分词）
let response = model.respond(
    userAudio: audio,
    voice: .NATM0,
    systemPrompt: "You enjoy having a good conversation."
)

// 或使用预设
let response = model.respond(
    userAudio: audio,
    voice: .NATM0,
    systemPromptTokens: SystemPromptPreset.customerService.tokens
)
```

可用预设：`focused`（默认）、`assistant`、`customerService`、`teacher`。

### PersonaPlex 命令行

```bash
make build

# 基础语音到语音
.build/release/audio respond --input question.wav --output response.wav

# 带转录（解码内心独白文本）
.build/release/audio respond --input question.wav --transcript

# JSON 输出（音频路径、转录、延迟指标）
.build/release/audio respond --input question.wav --json

# 自定义系统提示词文本
.build/release/audio respond --input question.wav --system-prompt-text "You enjoy having a good conversation."

# 选择音色和系统提示词预设
.build/release/audio respond --input question.wav --voice NATF1 --system-prompt focused

# 调整采样参数
.build/release/audio respond --input question.wav --audio-temp 0.6 --repetition-penalty 1.5

# 启用文本熵早停（文本熵过低时停止生成）
.build/release/audio respond --input question.wav --entropy-threshold 1.0 --entropy-window 5

# 列出可用音色和提示词
.build/release/audio respond --list-voices
.build/release/audio respond --list-prompts
```

## CosyVoice TTS——支持声音克隆的流式文本转语音

### 基础合成

```swift
import CosyVoiceTTS
import AudioCommon  // for WAVWriter

let model = try await CosyVoiceTTSModel.fromPretrained()
// 首次运行下载约 1.9 GB（LLM + DiT + HiFi-GAN 权重）

let audio = model.synthesize(text: "Hello, how are you today?", language: "english")
// 输出为 24kHz 单声道浮点采样
try WAVWriter.write(samples: audio, sampleRate: 24000, to: outputURL)
```

### 流式合成

```swift
// 流式：在生成过程中接收音频分块（首块约 150ms）
for try await chunk in model.synthesizeStream(text: "Hello, how are you today?", language: "english") {
    // chunk.audio: [Float], chunk.sampleRate: Int
    playAudio(chunk.audio)  // 即时播放
}
```

### 声音克隆 (CosyVoice)

使用 CAM++ 说话人编码器（192 维，CoreML 神经引擎）克隆说话人声音：

```swift
import CosyVoiceTTS
import AudioCommon

let model = try await CosyVoiceTTSModel.fromPretrained()
let speaker = try await CamPlusPlusSpeaker.fromPretrained()
// 首次使用时下载约 14 MB CAM++ CoreML 模型

let refAudio = try AudioFileLoader.load(url: referenceURL, targetSampleRate: 16000)
let embedding = try speaker.embed(audio: refAudio, sampleRate: 16000)
// embedding: 长度为 192 的 [Float]

let audio = model.synthesize(
    text: "Hello in a cloned voice!",
    language: "english",
    speakerEmbedding: embedding
)
```

### CosyVoice TTS 命令行

```bash
make build

# 基础合成
.build/release/audio speak "Hello world" --engine cosyvoice --language english --output output.wav

# 声音克隆（首次使用时下载 CAM++ 说话人编码器）
.build/release/audio speak "Hello world" --engine cosyvoice --voice-sample reference.wav --output cloned.wav

# 多说话人对话配合声音克隆
.build/release/audio speak "[S1] Hello there! [S2] Hey, how are you?" \
    --engine cosyvoice --speakers s1=alice.wav,s2=bob.wav -o dialogue.wav

# 行内情感/风格标签
.build/release/audio speak "(excited) Wow, amazing! (sad) But I have to go..." \
    --engine cosyvoice -o emotion.wav

# 组合：对话 + 情感 + 声音克隆
.build/release/audio speak "[S1] (happy) Great news! [S2] (surprised) Really?" \
    --engine cosyvoice --speakers s1=alice.wav,s2=bob.wav -o combined.wav

# 自定义风格指令
.build/release/audio speak "Hello world" --engine cosyvoice --cosy-instruct "Speak cheerfully" -o cheerful.wav

# 流式合成
.build/release/audio speak "Hello world" --engine cosyvoice --language english --stream --output output.wav
```

## Kokoro TTS——轻量级端侧文本转语音 (iOS + macOS)

### 基础合成

```swift
import KokoroTTS
import AudioCommon  // for WAVWriter

let tts = try await KokoroTTSModel.fromPretrained()
// 首次运行下载约 170 MB（CoreML 模型 + 音色嵌入向量 + 字典）

let audio = try tts.synthesize(text: "Hello world", voice: "af_heart")
// 输出为 24kHz 单声道浮点采样
try WAVWriter.write(samples: audio, sampleRate: 24000, to: outputURL)
```

54 种预设音色覆盖 10 种语言。非自回归——3 阶段 CoreML 流水线，无采样循环。完全在神经引擎上运行，不占用 GPU。

### Kokoro TTS 命令行

```bash
make build

# 基础合成
.build/release/audio kokoro "Hello world" --voice af_heart --output hello.wav

# 选择语言
.build/release/audio kokoro "Bonjour le monde" --voice ff_siwis --language fr --output bonjour.wav

# 列出可用音色
.build/release/audio kokoro --list-voices
```

## Qwen3 Chat（端侧 LLM）

```swift
import Qwen3Chat

let chat = try await Qwen3ChatModel.fromPretrained()
// 首次运行下载约 318 MB（INT4 CoreML 模型 + 分词器）

// 单次回复
let response = try chat.generate("What is Swift?", systemPrompt: "Answer briefly.")
print(response)

// 流式 token
let stream = chat.chatStream("Tell me a joke", systemPrompt: "Be funny.")
for try await token in stream {
    print(token, terminator: "")
}
```

Qwen3-0.6B 经 INT4 量化后部署于 CoreML。在神经引擎上运行，iPhone 上约 2 tok/s，M 系列芯片上约 15 tok/s。支持多轮对话（KV 缓存）、思考模式（`<think>` token）以及可配置的采样参数（temperature、top-k、top-p、重复惩罚）。

## 语音活动检测 (VAD)——检测音频中的语音

### 流式 VAD (Silero)

Silero VAD v5 以 32ms 音频分块为单位处理，亚毫秒延迟——非常适合麦克风或音频流的实时语音检测。

```swift
import SpeechVAD

let vad = try await SileroVADModel.fromPretrained()
// 或使用 CoreML（神经引擎，更低功耗）：
// let vad = try await SileroVADModel.fromPretrained(engine: .coreml)

// 流式：处理 512 采样点的分块（16kHz 下 32ms）
let prob = vad.processChunk(samples)  // → 0.0...1.0
vad.resetState()  // 在不同音频流之间调用

// 或一次性检测所有片段
let segments = vad.detectSpeech(audio: audioSamples, sampleRate: 16000)
for seg in segments {
    print("Speech: \(seg.startTime)s - \(seg.endTime)s")
}
```

### 事件驱动流式处理

```swift
let processor = StreamingVADProcessor(model: vad)

// 输入任意长度音频——语音确认后触发事件
let events = processor.process(samples: audioBuffer)
for event in events {
    switch event {
    case .speechStarted(let time):
        print("Speech started at \(time)s")
    case .speechEnded(let segment):
        print("Speech: \(segment.startTime)s - \(segment.endTime)s")
    }
}

// 在流结束时刷新
let final = processor.flush()
```

### VAD 命令行

```bash
make build

# 流式 Silero VAD（32ms 分块）
.build/release/audio vad-stream audio.wav

# CoreML 后端（神经引擎）
.build/release/audio vad-stream audio.wav --engine coreml

# 自定义阈值
.build/release/audio vad-stream audio.wav --onset 0.6 --offset 0.4

# JSON 输出
.build/release/audio vad-stream audio.wav --json

# 批量 pyannote VAD（10 秒滑动窗口）
.build/release/audio vad audio.wav
```

## 说话人分离——谁在什么时间说话

### 分离管线

```swift
import SpeechVAD

let pipeline = try await DiarizationPipeline.fromPretrained()
// 或使用 CoreML 嵌入（神经引擎，释放 GPU）：
// let pipeline = try await DiarizationPipeline.fromPretrained(embeddingEngine: .coreml)

let result = pipeline.diarize(audio: samples, sampleRate: 16000)
for seg in result.segments {
    print("Speaker \(seg.speakerId): [\(seg.startTime)s - \(seg.endTime)s]")
}
print("\(result.numSpeakers) speakers detected")
```

### 说话人嵌入向量

```swift
let model = try await WeSpeakerModel.fromPretrained()
// 或: let model = try await WeSpeakerModel.fromPretrained(engine: .coreml)
let embedding = model.embed(audio: samples, sampleRate: 16000)
// embedding: 长度为 256 的 [Float]，L2 归一化

// 比较说话人
let similarity = WeSpeakerModel.cosineSimilarity(embeddingA, embeddingB)
```

### 说话人提取

使用参考录音提取特定说话人的片段：

```swift
let pipeline = try await DiarizationPipeline.fromPretrained()
let targetEmb = pipeline.embeddingModel.embed(audio: enrollmentAudio, sampleRate: 16000)
let segments = pipeline.extractSpeaker(
    audio: meetingAudio, sampleRate: 16000,
    targetEmbedding: targetEmb
)
```

### Sortformer 分离（端到端，CoreML）

NVIDIA Sortformer 直接预测最多 4 位说话人的逐帧语音活动——无需嵌入或聚类。在神经引擎上运行。

```swift
let diarizer = try await SortformerDiarizer.fromPretrained()
let result = diarizer.diarize(audio: samples, sampleRate: 16000, config: .default)
for seg in result.segments {
    print("Speaker \(seg.speakerId): [\(seg.startTime)s - \(seg.endTime)s]")
}
```

### 分离命令行

```bash
make build

# Pyannote 分离（默认）
.build/release/audio diarize meeting.wav

# Sortformer 分离（CoreML，神经引擎）
.build/release/audio diarize meeting.wav --engine sortformer

# CoreML 嵌入（神经引擎，仅 pyannote）
.build/release/audio diarize meeting.wav --embedding-engine coreml

# JSON 输出
.build/release/audio diarize meeting.wav --json

# 提取特定说话人（仅 pyannote）
.build/release/audio diarize meeting.wav --target-speaker enrollment.wav

# 说话人嵌入向量
.build/release/audio embed-speaker enrollment.wav --json
.build/release/audio embed-speaker enrollment.wav --engine coreml
```

架构详情请参阅[说话人分离](docs/inference/speaker-diarization.md)。

## 语音增强——噪声抑制与音频清理

### 噪声抑制

```swift
import SpeechEnhancement
import AudioCommon  // for WAVWriter

let enhancer = try await SpeechEnhancer.fromPretrained()
// 首次运行下载约 4.3 MB（Core ML FP16 模型 + 辅助数据）

let cleanAudio = try enhancer.enhance(audio: noisyAudio, sampleRate: 48000)
try WAVWriter.write(samples: cleanAudio, sampleRate: 48000, to: outputURL)
```

### 降噪命令行

```bash
make build

# 基础降噪
.build/release/audio denoise noisy.wav

# 自定义输出路径
.build/release/audio denoise noisy.wav --output clean.wav
```

架构详情请参阅[语音增强](docs/inference/speech-enhancement.md)。

## 管线——组合多个模型

所有模型遵循共享协议（`SpeechRecognitionModel`、`SpeechGenerationModel`、`SpeechEnhancementModel` 等），可以组合为管线：

### 嘈杂语音识别（DeepFilterNet + ASR）

```swift
import SpeechEnhancement
import Qwen3ASR

let enhancer = try await SpeechEnhancer.fromPretrained()
let asr = try await Qwen3ASRModel.fromPretrained()

// 以 48kHz 增强，再以 16kHz 转录
let clean = try enhancer.enhance(audio: noisyAudio, sampleRate: 48000)
let clean16k = AudioResampler.resample(clean, from: 48000, to: 16000)
let text = asr.transcribe(audio: clean16k, sampleRate: 16000)
```

### 语音中转（VAD + ASR + TTS）

```swift
import SpeechVAD
import Qwen3ASR
import Qwen3TTS

let vad = try await SileroVADModel.fromPretrained()
let asr = try await Qwen3ASRModel.fromPretrained()
let tts = try await Qwen3TTSModel.fromPretrained()

// 检测语音片段，转录，重新合成
let segments = vad.detectSpeech(audio: audio, sampleRate: 16000)
for seg in segments {
    let chunk = Array(audio[Int(seg.startTime * 16000)..<Int(seg.endTime * 16000)])
    let text = asr.transcribe(audio: chunk, sampleRate: 16000)
    let speech = tts.synthesize(text: text, language: "english")
    // speech: 24kHz 单声道浮点采样
}
```

### 会议转录（分离 + ASR）

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

完整协议参考请参阅[共享协议](docs/shared-protocols.md)。

## HTTP API 服务器

一个独立的 HTTP 服务器通过 REST 和 WebSocket 端点暴露所有模型。模型在首次请求时懒加载。

```bash
swift build -c release
.build/release/audio-server --port 8080

# 音频转录
curl -X POST http://localhost:8080/transcribe --data-binary @audio.wav -H "Content-Type: audio/wav"

# 文本转语音
curl -X POST http://localhost:8080/speak -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "engine": "cosyvoice"}' -o output.wav

# 语音到语音 (PersonaPlex)
curl -X POST http://localhost:8080/respond --data-binary @question.wav -o response.wav

# 语音增强
curl -X POST http://localhost:8080/enhance --data-binary @noisy.wav -o clean.wav

# 启动时预加载所有模型
.build/release/audio-server --preload --port 8080
```

### WebSocket 流式接口

#### OpenAI Realtime API (`/v1/realtime`)

主要 WebSocket 端点实现了 [OpenAI Realtime API](https://platform.openai.com/docs/api-reference/realtime) 协议——所有消息为 JSON 格式，带 `type` 字段，音频为 base64 编码的 PCM16 24kHz 单声道。

**客户端 → 服务端事件：**

| 事件 | 描述 |
|------|------|
| `session.update` | 配置引擎、语言、音频格式 |
| `input_audio_buffer.append` | 发送 base64 PCM16 音频分块 |
| `input_audio_buffer.commit` | 转录累积的音频 (ASR) |
| `input_audio_buffer.clear` | 清空音频缓冲区 |
| `response.create` | 请求 TTS 合成 |

**服务端 → 客户端事件：**

| 事件 | 描述 |
|------|------|
| `session.created` | 会话已初始化 |
| `session.updated` | 配置已确认 |
| `input_audio_buffer.committed` | 音频已提交进行转录 |
| `conversation.item.input_audio_transcription.completed` | ASR 结果 |
| `response.audio.delta` | Base64 PCM16 音频分块 (TTS) |
| `response.audio.done` | 音频流完成 |
| `response.done` | 回复完成，附带元数据 |
| `error` | 错误类型和消息 |

```javascript
const ws = new WebSocket('ws://localhost:8080/v1/realtime');

// ASR：发送音频，获取转录
ws.send(JSON.stringify({ type: 'input_audio_buffer.append', audio: base64PCM16 }));
ws.send(JSON.stringify({ type: 'input_audio_buffer.commit' }));
// → 接收: conversation.item.input_audio_transcription.completed

// TTS：发送文本，获取流式音频
ws.send(JSON.stringify({
  type: 'response.create',
  response: { modalities: ['audio', 'text'], instructions: 'Hello world' }
}));
// → 接收: response.audio.delta (base64 分块), response.audio.done, response.done
```

示例 HTML 客户端位于 `Examples/websocket-client.html`——在服务器运行时用浏览器打开即可。

服务器是独立的 `AudioServer` 模块和 `audio-server` 可执行文件——不会将 Hummingbird/WebSocket 添加到主 `audio` CLI 中。

## 延迟 (M2 Max, 64 GB)

### ASR

| 模型 | 后端 | RTF | 10 秒音频处理耗时 |
|------|------|-----|-------------------|
| Qwen3-ASR-0.6B (4-bit) | MLX | ~0.06 | ~0.6s |
| Qwen3-ASR-0.6B (INT8) | CoreML + MLX | ~0.09 | ~0.9s |
| Qwen3-ASR-1.7B (8-bit) | MLX | ~0.11 | ~1.1s |
| Parakeet-TDT-0.6B (INT4) | CoreML (神经引擎) | ~0.12 冷启动, ~0.03 热启动 | ~1.2s / ~0.3s |
| Whisper-large-v3 | whisper.cpp (Q5_0) | ~0.10 | ~1.0s |
| Whisper-small | whisper.cpp (Q5_0) | ~0.04 | ~0.4s |

### 强制对齐

| 模型 | 框架 | 20 秒音频 | RTF |
|------|------|----------|-----|
| Qwen3-ForcedAligner-0.6B (4-bit) | MLX Swift (debug) | ~365ms | ~0.018 |

> 单次非自回归前向推理——无采样循环。音频编码器耗时占主导（约 328ms），解码器单次推理约 37ms。**比实时快 55 倍。**

### TTS

| 模型 | 框架 | 短文本 (1s) | 中等 (3s) | 长文本 (6s) | 流式首包 |
|------|------|-----------|----------|-----------|---------|
| Qwen3-TTS-0.6B (4-bit) | MLX Swift (release) | 1.6s (RTF 1.2) | 2.3s (RTF 0.7) | 3.9s (RTF 0.7) | ~120ms (1 帧) |
| Kokoro-82M | CoreML (神经引擎) | ~1.4s (RTFx 0.7) | ~4.3s (RTFx 0.7) | ~8.6s (RTFx 0.7) | 不适用（非自回归） |
| Apple `AVSpeechSynthesizer` | AVFoundation | 0.08s | 0.08s | 0.17s (RTF 0.02) | 不适用 |

> Qwen3-TTS 生成自然、富有表现力的语音，具备韵律和情感，运行**快于实时**（RTF < 1.0）。流式合成在约 120ms 内输出首个音频分块。Kokoro-82M 完全在神经引擎上通过 3 阶段流水线运行（RTFx 约 0.7），非常适合 iOS。Apple 内置 TTS 速度更快但语音生硬单调。

### PersonaPlex（语音到语音）

| 模型 | 框架 | ms/步 | RTF | 说明 |
|------|------|-------|-----|------|
| PersonaPlex-7B (8-bit) | MLX Swift (release) | ~112ms | ~1.4 | 推荐——连贯的响应，比 4-bit 快 30% |
| PersonaPlex-7B (4-bit) | MLX Swift (release) | ~158ms | ~1.97 | 不推荐——输出质量下降 |

> **请使用 8-bit。** INT8 更快（112 ms/步 vs. 158 ms/步）且能生成连贯的全双工响应。INT4 量化会降低生成质量，产生不连贯的语音。INT8 在 M2 Max 上以约 112ms/步运行。

### VAD 与说话人嵌入

| 模型 | 后端 | 单次调用延迟 | RTF | 说明 |
|------|------|-------------|-----|------|
| Silero-VAD-v5 | MLX | ~2.1ms / 分块 | 0.065 | GPU (Metal) |
| Silero-VAD-v5 | CoreML | ~0.27ms / 分块 | 0.008 | 神经引擎，**快 7.7 倍** |
| WeSpeaker ResNet34-LM | MLX | ~310ms / 20s 音频 | 0.016 | GPU (Metal) |
| WeSpeaker ResNet34-LM | CoreML | ~430ms / 20s 音频 | 0.021 | 神经引擎，释放 GPU |

> Silero VAD CoreML 在神经引擎上运行，速度是 MLX 的 7.7 倍，非常适合常开麦克风输入场景。WeSpeaker MLX 在 GPU 上更快，但 CoreML 可释放 GPU 用于并发任务（TTS、ASR）。两种后端产生等效结果。

### 语音增强

| 模型 | 后端 | 时长 | 延迟 | RTF |
|------|------|------|------|-----|
| DeepFilterNet3 (FP16) | CoreML | 5s | 0.65s | 0.13 |
| DeepFilterNet3 (FP16) | CoreML | 10s | 1.2s | 0.12 |
| DeepFilterNet3 (FP16) | CoreML | 20s | 4.8s | 0.24 |

RTF = 实时因子（越低越好，< 1.0 = 快于实时）。GRU 计算成本约为 O(n²)。

### MLX 与 CoreML 对比

两种后端产生等效结果。根据您的工作负载选择：

| | MLX | CoreML |
|---|---|---|
| **硬件** | GPU (Metal 着色器) | 神经引擎 + CPU |
| **最适合** | 最大吞吐量，单模型场景 | 多模型管线，后台任务 |
| **功耗** | GPU 利用率较高 | 功耗更低，释放 GPU |
| **延迟** | 大模型更快 (WeSpeaker) | 小模型更快 (Silero VAD) |

**桌面推理**：MLX 为默认选择——在 Apple Silicon 上提供最快的单模型性能。在同时运行多个模型时（如 VAD + ASR + TTS）切换至 CoreML 以避免 GPU 争用，或在笔记本电脑上用于对电池敏感的场景。

CoreML 模型可用于 Qwen3-ASR 编码器、Silero VAD 和 WeSpeaker。对于 Qwen3-ASR，使用 `--engine qwen3-coreml`（混合模式：CoreML 编码器在 ANE 上 + MLX 文本解码器在 GPU 上）。对于 VAD/嵌入，在构造时传入 `engine: .coreml`——推理 API 完全相同。

## 精度基准

### ASR——词错误率（[详情](docs/benchmarks/asr-wer.md)）

| 模型 | WER% (LibriSpeech test-clean) | RTF |
|------|-------------------------------|-----|
| Qwen3-ASR 1.7B 8-bit | **2.35** | 0.090 |
| Qwen3-ASR 1.7B 4-bit | 2.57 | 0.045 |
| Parakeet TDT INT8 | 2.74 | 0.089 |
| Qwen3-ASR 0.6B 8-bit | 2.80 | 0.025 |

Qwen3-ASR 1.7B 8-bit 在同等规模下超越 Whisper Large v3 Turbo (2.5%)。多语言：在 FLEURS 上对 10 种语言进行了基准测试。

### TTS——往返可理解性（[详情](docs/benchmarks/tts-roundtrip.md)）

| 引擎 | WER% | RTF |
|------|------|-----|
| CosyVoice3 | **3.25** | 0.59 |
| Qwen3-TTS 1.7B | 3.47 | 0.79 |
| Kokoro-82M | 3.90 | 0.17 |

### VAD——语音检测（[详情](docs/benchmarks/vad-detection.md)）

| 引擎 | F1% (FLEURS) | RTF |
|------|-------------|-----|
| FireRedVAD | **99.12** | 0.007 |
| Silero CoreML | 95.13 | 0.022 |
| Pyannote MLX | 94.86 | 0.358 |

## 架构

**模型：** [ASR 模型](docs/models/asr-model.md)、[TTS 模型](docs/models/tts-model.md)、[CosyVoice TTS](docs/models/cosyvoice-tts.md)、[Kokoro TTS](docs/models/kokoro-tts.md)、[Parakeet TDT](docs/models/parakeet-asr.md)、[PersonaPlex](docs/models/personaplex.md)、[FireRedVAD](docs/models/fireredvad.md)

**推理：** [ASR 推理](docs/inference/qwen3-asr-inference.md)、[TTS 推理](docs/inference/qwen3-tts-inference.md)、[强制对齐器](docs/inference/forced-aligner.md)、[FireRedVAD](docs/inference/fireredvad.md)、[Silero VAD](docs/inference/silero-vad.md)、[说话人分离](docs/inference/speaker-diarization.md)、[语音增强](docs/inference/speech-enhancement.md)、[音频播放](docs/audio/playback.md)

**基准测试：** [ASR WER](docs/benchmarks/asr-wer.md)、[TTS 往返测试](docs/benchmarks/tts-roundtrip.md)、[VAD 检测](docs/benchmarks/vad-detection.md)

**参考：** [共享协议](docs/shared-protocols.md)

## 缓存配置

模型权重缓存在本地。可通过以下方式覆盖缓存位置：

```bash
export QWEN3_CACHE_DIR=/path/to/cache
```

## MLX Metal 库

如果运行时出现 `Failed to load the default metallib` 错误，说明 Metal 着色器库缺失。运行 `make build`（或在手动 `swift build` 后运行 `./scripts/build_mlx_metallib.sh release`）来编译它。如果缺少 Metal Toolchain，请先安装：

```bash
xcodebuild -downloadComponent MetalToolchain
```

## 测试

单元测试（配置、采样、文本预处理、时间戳校正）无需下载模型即可运行：

```bash
swift test --filter "Qwen3TTSConfigTests|SamplingTests|CosyVoiceTTSConfigTests|CamPlusPlusMelExtractorTests|PersonaPlexTests|ForcedAlignerTests/testText|ForcedAlignerTests/testTimestamp|ForcedAlignerTests/testLIS|SileroVADTests/testSilero|SileroVADTests/testReflection|SileroVADTests/testProcess|SileroVADTests/testReset|SileroVADTests/testDetect|SileroVADTests/testStreaming|SileroVADTests/testVADEvent|KokoroTTSTests"
```

集成测试需要模型权重（首次运行时自动下载）：

```bash
# TTS 往返测试：合成文本，保存 WAV，再用 ASR 转录回来
swift test --filter TTSASRRoundTripTests

# 仅 ASR：转录测试音频
swift test --filter Qwen3ASRIntegrationTests

# 强制对齐器 E2E：词级时间戳（下载约 979 MB）
swift test --filter ForcedAlignerTests/testForcedAlignerE2E

# PersonaPlex E2E：语音到语音管线（下载约 5.5 GB）
PERSONAPLEX_E2E=1 swift test --filter PersonaPlexE2ETests
```

> **注意：** 运行使用 MLX 操作的测试前必须先构建 MLX Metal 库。
> 请参阅 [MLX Metal 库](#mlx-metal-库)了解说明。

## 支持的语言

| 模型 | 语言 |
|------|------|
| Qwen3-ASR | 52 种语言（中文、英文、粤语、德语、法语、西班牙语、日语、韩语、俄语 + 22 种中国方言，...） |
| Parakeet TDT | 25 种欧洲语言 (BG, CS, DA, DE, EL, EN, ES, ET, FI, FR, HR, HU, IT, LT, LV, MT, NL, PL, PT, RO, RU, SK, SL, SV, UK) |
| Qwen3-TTS | EN, CN, DE, JA, ES, FR, KO, RU, IT, PT（CustomVoice 还支持北京/四川方言） |
| CosyVoice TTS | CN, EN, JA, KO, DE, ES, FR, IT, RU |
| Kokoro TTS | EN (US/UK), ES, FR, HI, IT, JA, PT, CN, KO, DE |
| PersonaPlex | EN |

## 与其他方案对比

### 语音转文字 (ASR)：speech-swift 与替代方案

| | **speech-swift (Qwen3-ASR)** | **whisper.cpp** | **Apple SFSpeechRecognizer** | **Google Cloud Speech** |
|---|---|---|---|---|
| **运行方式** | 端侧 (MLX/CoreML) | 端侧 (CPU/GPU) | 端侧或云端 | 仅云端 |
| **语言** | 52 | 100+ | ~70（端侧：有限） | 125+ |
| **RTF（10s 音频，M2 Max）** | 0.06 (17x 实时) | 0.10 (Whisper-large-v3) | 不适用 | 不适用 |
| **流式** | 否（批处理） | 否（批处理） | 是 | 是 |
| **自定义模型** | 是（替换 HuggingFace 权重） | 是（GGML 模型） | 否 | 否 |
| **Swift API** | 原生 async/await | C++ Swift 桥接 | 原生 | REST/gRPC |
| **隐私** | 完全端侧 | 完全端侧 | 取决于配置 | 数据发送至云端 |
| **词级时间戳** | 是（强制对齐器） | 是 | 有限 | 是 |
| **费用** | 免费 (Apache 2.0) | 免费 (MIT) | 免费（端侧） | 按分钟计费 |

### 文本转语音 (TTS)：speech-swift 与替代方案

| | **speech-swift (Qwen3-TTS)** | **speech-swift (Kokoro)** | **Apple AVSpeechSynthesizer** | **ElevenLabs / 云端 TTS** |
|---|---|---|---|---|
| **质量** | 神经网络，富有表现力 | 神经网络，自然 | 机械，单调 | 神经网络，最高质量 |
| **运行方式** | 端侧 (MLX) | 端侧 (CoreML) | 端侧 | 仅云端 |
| **流式** | 是（首块约 120ms） | 否（3 阶段流水线） | 否 | 是 |
| **声音克隆** | 是 | 否 | 否 | 是 |
| **音色** | 9 种内置 + 克隆任意 | 54 种预设 | ~50 种系统音色 | 1000+ |
| **语言** | 10 | 10 | 60+ | 30+ |
| **iOS 支持** | 仅 macOS | iOS + macOS | iOS + macOS | 任意（API） |
| **费用** | 免费 (Apache 2.0) | 免费 (Apache 2.0) | 免费 | 按字符计费 |

### 何时使用 speech-swift

- **隐私关键应用** — 医疗、法律、企业等音频不能离开设备的场景
- **离线使用** — 首次模型下载后无需网络连接
- **成本敏感** — 无按分钟或按字符的 API 费用
- **Apple Silicon 优化** — 专为 M 系列 GPU (Metal) 和神经引擎打造
- **完整管线** — 在单个 Swift 包中组合 ASR + TTS + VAD + 分离 + 增强

## 常见问题

**speech-swift 能在 iOS 上运行吗？**
Kokoro TTS、Qwen3.5-Chat（CoreML）、Silero VAD、Parakeet ASR、DeepFilterNet3 和 WeSpeaker 均可通过 CoreML 在 iOS 17+ 的神经引擎上运行。基于 MLX 的模型（Qwen3-ASR、Qwen3-TTS、Qwen3.5-Chat MLX、PersonaPlex）需要在 Apple Silicon 上运行 macOS 14+。

**需要网络连接吗？**
仅在首次从 HuggingFace 下载模型时需要（自动下载，缓存在 `~/Library/Caches/qwen3-speech/`）。之后所有推理完全离线运行，不进行任何网络访问。

**speech-swift 与 Whisper 相比如何？**
Qwen3-ASR-0.6B 在 M2 Max 上达到 RTF 0.06——比通过 whisper.cpp 运行的 Whisper-large-v3 (RTF 0.10) 快 40%——在 52 种语言上精度相当。speech-swift 提供原生 Swift async/await API，而 whisper.cpp 需要 C++ 桥接。

**可以用于商业应用吗？**
可以。speech-swift 采用 Apache 2.0 许可证。底层模型权重有各自的许可证（请查看各模型的 HuggingFace 页面）。

**支持哪些 Apple Silicon 芯片？**
所有 M 系列芯片：M1、M2、M3、M4 及其 Pro/Max/Ultra 变体。需要 macOS 14+ (Sonoma) 或 iOS 17+。

**需要多少内存？**
从约 3 MB（Silero VAD）到约 6.5 GB（PersonaPlex 7B）不等。Kokoro TTS 约 500 MB，Qwen3-ASR 约 2.2 GB。完整详情请参阅[内存需求](#内存需求)表。

**可以同时运行多个模型吗？**
可以。在神经引擎上使用 CoreML 模型，同时在 GPU 上使用 MLX 模型以避免争用——例如 Silero VAD (CoreML) + Qwen3-ASR (MLX) + Qwen3-TTS (MLX)。

**有 REST API 吗？**
有。`audio-server` 二进制文件通过 HTTP REST 和 WebSocket 端点暴露所有模型，包括 `/v1/realtime` 上兼容 OpenAI Realtime API 的 WebSocket。

## 贡献

欢迎贡献！无论是 bug 修复、新模型集成还是文档改进——PR 都受到欢迎。

**开始贡献：**
1. Fork 仓库并创建功能分支
2. `make build` 编译（需要 Xcode + Metal Toolchain）
3. `make test` 运行测试套件
4. 向 `main` 提交 PR

## 许可证

Apache 2.0
