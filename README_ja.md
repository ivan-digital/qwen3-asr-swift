# Speech Swift

Apple Silicon向けAI音声モデル。MLX SwiftとCoreMLで動作します。

📖 Read in: [English](README.md) · [中文](README_zh.md) · [日本語](README_ja.md) · [한국어](README_ko.md) · [Español](README_es.md) · [Deutsch](README_de.md) · [Français](README_fr.md) · [हिन्दी](README_hi.md) · [Português](README_pt.md) · [Русский](README_ru.md)

Mac・iOS向けのオンデバイス音声認識・合成・理解。Apple Silicon上で完全にローカル動作——クラウド不要、APIキー不要、データはデバイスから外部に送信されません。

[Homebrewでインストール](#homebrew)するか、Swift Packageの依存関係として追加できます。

**[ドキュメント](https://soniqo.audio)** · **[HuggingFace モデル](https://huggingface.co/aufklarer)** · **[ブログ](https://blog.ivan.digital)**

- **Qwen3-ASR** — 音声認識 (自動音声認識、52言語対応)
- **Parakeet TDT** — CoreMLによる音声認識 (Neural Engine、NVIDIA FastConformer + TDTデコーダー、25言語対応)
- **Qwen3-ForcedAligner** — 単語レベルのタイムスタンプ整列 (音声 + テキスト → タイムスタンプ)
- **Qwen3-TTS** — テキスト読み上げ (最高品質、ストリーミング、カスタムスピーカー、10言語対応)
- **CosyVoice TTS** — ストリーミング対応テキスト読み上げ、音声クローン、マルチスピーカー対話、感情タグ (9言語、DiT flow matching、CAM++話者エンコーダー)
- **Kokoro TTS** — オンデバイステキスト読み上げ (82Mパラメーター、CoreML/Neural Engine、54種類の声、iOS対応、10言語)
- **Qwen3.5-Chat** — オンデバイスLLMチャット (0.8B、MLX + CoreML、INT4/INT8、DeltaNetハイブリッド、ストリーミングトークン)
- **PersonaPlex** — 全二重音声間会話 (7B、音声入力 → 音声出力、18種類のボイスプリセット)
- **DeepFilterNet3** — 音声強調 / ノイズ抑制 (2.1Mパラメーター、リアルタイム48kHz)
- **FireRedVAD** — オフライン音声区間検出 (DFSMN、CoreML、100以上の言語、97.6% F1)
- **Silero VAD** — ストリーミング音声区間検出 (32msチャンク、サブミリ秒レイテンシー)
- **Pyannote VAD** — オフライン音声区間検出 (10秒ウィンドウ、マルチスピーカーオーバーラップ)
- **話者ダイアライゼーション** — 誰がいつ話したか (Pyannoteセグメンテーション + アクティビティベース話者チェイニング、またはNeural Engine上のエンドツーエンドSortformer)
- **話者埋め込み** — 話者検証・識別 (WeSpeaker ResNet34、256次元ベクトル)

論文: [Qwen3-ASR](https://arxiv.org/abs/2601.21337) (Alibaba), [Qwen3-TTS](https://arxiv.org/abs/2601.15621) (Alibaba), [Qwen3](https://arxiv.org/abs/2505.09388) (Alibaba), [Parakeet TDT](https://arxiv.org/abs/2304.06795) (NVIDIA), [CosyVoice 3](https://arxiv.org/abs/2505.17589) (Alibaba), [Kokoro](https://arxiv.org/abs/2301.01695) (StyleTTS 2), [PersonaPlex](https://arxiv.org/abs/2602.06053) (NVIDIA), [Mimi](https://arxiv.org/abs/2410.00037) (Kyutai), [Sortformer](https://arxiv.org/abs/2409.06656) (NVIDIA)

## ロードマップ

今後の予定は[ロードマップディスカッション](https://github.com/soniqo/speech-swift/discussions/81)をご覧ください。コメントやご提案も歓迎です！

## ニュース

- **2026年3月20日** — [600Mモデルだけで、Mac上でWhisper Large v3を超えた](https://blog.ivan.digital/we-beat-whisper-large-v3-with-a-600m-model-running-entirely-on-your-mac-20e6ce191174)
- **2026年2月26日** — [Apple Silicon上の話者ダイアライゼーションと音声区間検出 — ネイティブSwift + MLX](https://blog.ivan.digital/speaker-diarization-and-voice-activity-detection-on-apple-silicon-native-swift-with-mlx-92ea0c9aca0f)
- **2026年2月23日** — [Apple Silicon上のNVIDIA PersonaPlex 7B — ネイティブSwift + MLXによる全二重音声間変換](https://blog.ivan.digital/nvidia-personaplex-7b-on-apple-silicon-full-duplex-speech-to-speech-in-native-swift-with-mlx-0aa5276f2e23)
- **2026年2月12日** — [Qwen3-ASR Swift: Apple Silicon向けオンデバイスASR + TTS — アーキテクチャとベンチマーク](https://blog.ivan.digital/qwen3-asr-swift-on-device-asr-tts-for-apple-silicon-architecture-and-benchmarks-27cbf1e4463f)

## モデル

| モデル | タスク | ストリーミング | 対応言語 | サイズ |
|-------|------|-----------|-----------|-------|
| Qwen3-ASR-0.6B | 音声 → テキスト | なし | 52言語 | [4-bit](https://huggingface.co/aufklarer/Qwen3-ASR-0.6B-MLX-4bit) 680 MB · [8-bit](https://huggingface.co/aufklarer/Qwen3-ASR-0.6B-MLX-8bit) 1.0 GB · [CoreML](https://huggingface.co/aufklarer/Qwen3-ASR-CoreML) 180 MB |
| Qwen3-ASR-1.7B | 音声 → テキスト | なし | 52言語 | [4-bit](https://huggingface.co/aufklarer/Qwen3-ASR-1.7B-MLX-4bit) 2.1 GB · [8-bit](https://huggingface.co/aufklarer/Qwen3-ASR-1.7B-MLX-8bit) 3.2 GB |
| Parakeet-TDT-0.6B | 音声 → テキスト | なし | ヨーロッパ25言語 | [CoreML INT4](https://huggingface.co/aufklarer/Parakeet-TDT-v3-CoreML-INT4) 315 MB · [CoreML INT8](https://huggingface.co/aufklarer/Parakeet-TDT-v3-CoreML-INT8) 500 MB |
| Qwen3-ForcedAligner-0.6B | 音声 + テキスト → タイムスタンプ | なし | 多言語 | [4-bit](https://huggingface.co/aufklarer/Qwen3-ForcedAligner-0.6B-4bit) 979 MB · [8-bit](https://huggingface.co/aufklarer/Qwen3-ForcedAligner-0.6B-8bit) 1.4 GB · [CoreML INT4](https://huggingface.co/aufklarer/Qwen3-ForcedAligner-0.6B-CoreML-INT4) 630 MB · [CoreML INT8](https://huggingface.co/aufklarer/Qwen3-ForcedAligner-0.6B-CoreML-INT8) 1.0 GB |
| Qwen3-TTS-0.6B Base | テキスト → 音声 | あり (~120ms) | 10言語 | [4-bit](https://huggingface.co/aufklarer/Qwen3-TTS-12Hz-0.6B-Base-MLX-4bit) 1.7 GB · [8-bit](https://huggingface.co/aufklarer/Qwen3-TTS-12Hz-0.6B-Base-MLX-8bit) 2.4 GB |
| Qwen3-TTS-0.6B CustomVoice | テキスト → 音声 | あり (~120ms) | 10言語 | [4-bit](https://huggingface.co/aufklarer/Qwen3-TTS-12Hz-0.6B-CustomVoice-MLX-4bit) 1.7 GB |
| Qwen3-TTS-1.7B Base | テキスト → 音声 | あり (~120ms) | 10言語 | [4-bit](https://huggingface.co/aufklarer/Qwen3-TTS-12Hz-1.7B-Base-MLX-4bit) 3.2 GB · [8-bit](https://huggingface.co/aufklarer/Qwen3-TTS-12Hz-1.7B-Base-MLX-8bit) 4.8 GB |
| CosyVoice3-0.5B | テキスト → 音声 | あり (~150ms) | 9言語 | [4-bit](https://huggingface.co/aufklarer/CosyVoice3-0.5B-MLX-4bit) 1.2 GB |
| Kokoro-82M | テキスト → 音声 | なし | 10言語 | [CoreML](https://huggingface.co/aufklarer/Kokoro-82M-CoreML) ~170 MB |
| Qwen3.5-0.8B Chat | Text → Text (LLM) | Yes (streaming) | Multi | [MLX INT4](https://huggingface.co/aufklarer/Qwen3.5-0.8B-Chat-MLX) 418 MB · [CoreML INT8](https://huggingface.co/aufklarer/Qwen3.5-0.8B-Chat-CoreML) 981 MB |
| PersonaPlex-7B | 音声 → 音声 | あり (~2秒チャンク) | EN | [4-bit](https://huggingface.co/aufklarer/PersonaPlex-7B-MLX-4bit) 4.9 GB · [8-bit](https://huggingface.co/aufklarer/PersonaPlex-7B-MLX-8bit) 9.1 GB |
| FireRedVAD | 音声区間検出 | なし (オフライン) | 100以上の言語 | [CoreML](https://huggingface.co/aufklarer/FireRedVAD-CoreML) ~1.2 MB |
| Silero-VAD-v5 | 音声区間検出 | あり (32msチャンク) | 言語非依存 | [MLX](https://huggingface.co/aufklarer/Silero-VAD-v5-MLX) · [CoreML](https://huggingface.co/aufklarer/Silero-VAD-v5-CoreML) ~1.2 MB |
| Pyannote-Segmentation-3.0 | VAD + 話者セグメンテーション | なし (10秒ウィンドウ) | 言語非依存 | [MLX](https://huggingface.co/aufklarer/Pyannote-Segmentation-MLX) ~5.7 MB |
| DeepFilterNet3 | 音声強調 | あり (10msフレーム) | 言語非依存 | [CoreML FP16](https://huggingface.co/aufklarer/DeepFilterNet3-CoreML) ~4.2 MB |
| WeSpeaker-ResNet34-LM | 話者埋め込み (256次元) | なし | 言語非依存 | [MLX](https://huggingface.co/aufklarer/WeSpeaker-ResNet34-LM-MLX) · [CoreML](https://huggingface.co/aufklarer/WeSpeaker-ResNet34-LM-CoreML) ~25 MB |
| CAM++ | 話者埋め込み (192次元) | なし | 言語非依存 | [CoreML](https://huggingface.co/aufklarer/CamPlusPlus-Speaker-CoreML) ~14 MB |
| Sortformer | 話者ダイアライゼーション (エンドツーエンド) | あり (チャンク) | 言語非依存 | [CoreML](https://huggingface.co/aufklarer/Sortformer-Diarization-CoreML) ~240 MB |

### メモリ要件

重みメモリは、モデルパラメーターが消費するGPU (MLX) またはANE (CoreML) のメモリです。ピーク推論にはKVキャッシュ、活性化値、中間テンソルが含まれます。

| モデル | 重みメモリ | ピーク推論 |
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

### TTSの選び方

- **Qwen3-TTS**: 最高品質、ストリーミング (~120ms)、9種類の内蔵スピーカー、10言語、バッチ合成
- **CosyVoice TTS**: ストリーミング (~150ms)、9言語、音声クローン (CAM++話者エンコーダー)、マルチスピーカー対話 (`[S1] ... [S2] ...`)、インライン感情・スタイルタグ (`(happy)`, `(whispers)`)、DiT flow matching + HiFi-GANボコーダー
- **Kokoro TTS**: 軽量iOS対応TTS (82Mパラメーター)、CoreML/Neural Engine、54種類の声、10言語、非自己回帰型 (3段階パイプライン)
- **PersonaPlex**: 全二重音声間変換 (音声入力 → 音声出力)、ストリーミング (~2秒チャンク)、18種類のボイスプリセット、Moshiアーキテクチャベース

## インストール

### Homebrew

ネイティブARM Homebrew (`/opt/homebrew`) が必要です。Rosetta/x86_64 Homebrewはサポートされていません。

```bash
brew tap soniqo/speech https://github.com/soniqo/speech-swift
brew install speech
```

使い方：

```bash
audio transcribe recording.wav
audio speak "Hello world"
audio speak "Hallo Welt" --engine cosyvoice --language german
audio respond --input question.wav --transcript
```

> マイク入力によるインタラクティブな音声会話は **[PersonaPlexDemo](Examples/PersonaPlexDemo/)** をご覧ください。

### Swift Package Manager

`Package.swift`に追加：

```swift
dependencies: [
    .package(url: "https://github.com/soniqo/speech-swift", branch: "main")
]
```

必要なモジュールをインポート：

```swift
import Qwen3ASR      // 音声認識 (MLX)
import ParakeetASR   // 音声認識 (CoreML)
import Qwen3TTS      // テキスト読み上げ (Qwen3)
import CosyVoiceTTS  // テキスト読み上げ (ストリーミング)
import KokoroTTS     // テキスト読み上げ (CoreML、iOS対応)
import Qwen3Chat     // オンデバイスLLMチャット (CoreML)
import PersonaPlex   // 音声間変換 (全二重)
import SpeechVAD          // 音声区間検出 (pyannote + Silero)
import SpeechEnhancement  // ノイズ抑制 (DeepFilterNet3)
import AudioCommon        // 共有ユーティリティ
```

### 要件

- Swift 5.9以上
- macOS 14以上またはiOS 17以上
- Apple Silicon (M1/M2/M3/M4)
- Xcode 15以上 (Metal Toolchain付き — 不足している場合は `xcodebuild -downloadComponent MetalToolchain` を実行)

### ソースからビルド

```bash
git clone https://github.com/soniqo/speech-swift
cd speech-swift
make build
```

SwiftパッケージとMLX Metal shaderライブラリをワンステップでコンパイルします。Metalライブラリ (`mlx.metallib`) はGPU推論に必要です。これがないと実行時に `Failed to load the default metallib` エラーが発生します。

デバッグビルド: `make debug`。ユニットテスト実行: `make test`。

## 音声アシスタントを試す

**[PersonaPlexDemo](Examples/PersonaPlexDemo/)** は、すぐに使えるmacOS音声アシスタントです。タップして話しかけると、リアルタイムで音声応答が返ります。マイク入力にSilero VADを使った自動音声検出、Qwen3-ASRによる文字起こし、PersonaPlex 7Bによる音声間生成を搭載。マルチターン会話、18種類のボイスプリセット、内部独白の字幕表示に対応しています。

```bash
make build  # リポジトリのルートから — MLX metallibを含むすべてをビルド
cd Examples/PersonaPlexDemo
# .appバンドルの手順はExamples/PersonaPlexDemo/README.mdを参照
```

> M2 MaxでRTF ~0.94 (リアルタイムより高速)。モデルは初回実行時に自動ダウンロード (~5.5 GB PersonaPlex + ~400 MB ASR)。

## デモアプリ

- **[PersonaPlexDemo](Examples/PersonaPlexDemo/)** — 会話型音声アシスタント (マイク入力、VAD、マルチターン)。上記参照。
- **[SpeechDemo](Examples/SpeechDemo/)** — ディクテーション (Parakeet TDT / Qwen3-ASRの言語自動検出) とテキスト読み上げ (Qwen3-TTS) のタブインターフェース。

macOS `.app`バンドルとしてビルド・実行できます。各デモのREADMEを参照してください。

## 音声認識 (ASR) — Swiftで音声を文字起こし

### 基本的な文字起こし

```swift
import Qwen3ASR

// デフォルト: 0.6Bモデル
let model = try await Qwen3ASRModel.fromPretrained()

// より高精度な1.7Bモデルを使用する場合
let model = try await Qwen3ASRModel.fromPretrained(
    modelId: "aufklarer/Qwen3-ASR-1.7B-MLX-8bit"
)

// 任意のサンプルレートの音声を入力可能 — 内部で自動的に16kHzにリサンプリング
let transcription = model.transcribe(audio: audioSamples, sampleRate: 16000)
print(transcription)
```

### CoreMLエンコーダー (Neural Engine)

ハイブリッドモード: Neural Engine上のCoreMLエンコーダー + GPU上のMLXテキストデコーダー。低消費電力で、エンコーダーパスのGPU負荷を軽減します。

```swift
import Qwen3ASR

let encoder = try await CoreMLASREncoder.fromPretrained()
let model = try await Qwen3ASRModel.fromPretrained()
let text = try model.transcribe(audio: audioSamples, sampleRate: 16000, coremlEncoder: encoder)
```

INT8 (180 MB、デフォルト) とINT4 (90 MB) のバリアントが利用可能。INT8推奨 (FP32とのコサイン類似度 > 0.999)。

### Parakeet TDT (CoreML)

```swift
import ParakeetASR

let model = try await ParakeetASRModel.fromPretrained()
let transcription = model.transcribe(audio: audioSamples, sampleRate: 16000)
```

CoreML経由でNeural Engine上で動作 — GPUを他のワークロードに解放します。25のヨーロッパ言語対応、~315 MB。

### ASR CLI

```bash
make build  # または: swift build -c release && ./scripts/build_mlx_metallib.sh release

# デフォルト (Qwen3-ASR 0.6B, MLX)
.build/release/audio transcribe audio.wav

# 1.7Bモデルを使用
.build/release/audio transcribe audio.wav --model 1.7B

# CoreMLエンコーダー (Neural Engine + MLXデコーダー)
.build/release/audio transcribe --engine qwen3-coreml audio.wav

# Parakeet TDT (CoreML, Neural Engine)
.build/release/audio transcribe --engine parakeet audio.wav
```

## 強制アラインメント

### 単語レベルのタイムスタンプ

```swift
import Qwen3ASR

let aligner = try await Qwen3ForcedAligner.fromPretrained()
// 初回実行時に~979 MBをダウンロード

let aligned = aligner.align(
    audio: audioSamples,
    text: "Can you guarantee that the replacement part will be shipped tomorrow?",
    sampleRate: 24000
)

for word in aligned {
    print("[\(String(format: "%.2f", word.startTime))s - \(String(format: "%.2f", word.endTime))s] \(word.text)")
}
```

### 強制アラインメントCLI

```bash
swift build -c release

# テキストを指定してアラインメント
.build/release/audio align audio.wav --text "Hello world"

# まず文字起こしし、その後アラインメント
.build/release/audio align audio.wav
```

出力：
```
[0.12s - 0.45s] Can
[0.45s - 0.72s] you
[0.72s - 1.20s] guarantee
...
```

非自己回帰型 — 3段階パイプラインでサンプリングループなし。アーキテクチャの詳細は[強制アラインメント](docs/inference/forced-aligner.md)を参照。

## テキスト読み上げ (TTS) — Swiftで音声を生成

### 基本的な音声合成

```swift
import Qwen3TTS
import AudioCommon  // WAVWriter用

let model = try await Qwen3TTSModel.fromPretrained()
// 初回実行時に~1.7 GBをダウンロード (モデル + コーデック重み)
let audio = model.synthesize(text: "Hello world", language: "english")
// 出力は24kHzモノラルfloatサンプル
try WAVWriter.write(samples: audio, sampleRate: 24000, to: outputURL)
```

### TTS CLI

```bash
make build
.build/release/audio speak "Hello world" --output output.wav --language english
```

### カスタムボイス / スピーカー選択

**CustomVoice**モデルバリアントは、9種類の内蔵スピーカーボイスと、トーン・スタイルを制御する自然言語指示をサポートしています。CustomVoiceモデルIDを渡してロードします：

```swift
import Qwen3TTS

// CustomVoiceモデルをロード (初回実行時に~1.7 GBをダウンロード)
let model = try await Qwen3TTSModel.fromPretrained(
    modelId: TTSModelVariant.customVoice.rawValue
)

// 特定のスピーカーで合成
let audio = model.synthesize(text: "Hello world", language: "english", speaker: "vivian")

// 利用可能なスピーカーを表示
print(model.availableSpeakers)  // ["aiden", "dylan", "eric", ...]
```

CLI:

```bash
# CustomVoiceモデルでスピーカーを指定
.build/release/audio speak "Hello world" --model customVoice --speaker vivian --output vivian.wav

# 利用可能なスピーカーを一覧表示
.build/release/audio speak --model customVoice --list-speakers
```

### 音声クローン (Baseモデル)

リファレンス音声ファイルからスピーカーの声をクローンします。2つのモードがあります：

**ICLモード** (推奨) — リファレンス音声をトランスクリプト付きでコーデックトークンにエンコード。高品質で確実なEOS：

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

**X-vectorモード** — 話者埋め込みのみ、トランスクリプト不要ですが品質は低め：

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

### トーン / スタイル指示 (CustomVoiceのみ)

CustomVoiceモデルは、話し方のスタイル、トーン、感情、ペースを制御する自然言語の`instruct`パラメーターを受け付けます。指示はChatML形式でモデル入力の先頭に付加されます。

```swift
// 明るいトーン
let audio = model.synthesize(
    text: "Welcome to our store!",
    language: "english",
    speaker: "ryan",
    instruct: "Speak in a cheerful, upbeat tone"
)

// ゆっくりと真剣に
let audio = model.synthesize(
    text: "We regret to inform you...",
    language: "english",
    speaker: "aiden",
    instruct: "Read this slowly and solemnly"
)

// ささやき
let audio = model.synthesize(
    text: "Can you keep a secret?",
    language: "english",
    speaker: "vivian",
    instruct: "Whisper this softly"
)
```

CLI:

```bash
# スタイル指示付き
.build/release/audio speak "Good morning!" --model customVoice --speaker ryan \
    --instruct "Speak in a cheerful, upbeat tone" --output cheerful.wav

# デフォルトの指示 ("Speak naturally.") はCustomVoice使用時に自動適用
.build/release/audio speak "Hello world" --model customVoice --speaker ryan --output natural.wav
```

CustomVoiceモデルで`--instruct`を指定しない場合、冗長な出力を防ぐために`"Speak naturally."`が自動適用されます。Baseモデルはinstruct非対応です。

### バッチ合成

単一のバッチフォワードパスで複数のテキストを合成し、高スループットを実現：

```swift
let texts = ["Good morning everyone.", "The weather is nice today.", "Please open the window."]
let audioList = model.synthesizeBatch(texts: texts, language: "english", maxBatchSize: 4)
// audioList[i]はtexts[i]の24kHzモノラルfloatサンプル
for (i, audio) in audioList.enumerated() {
    try WAVWriter.write(samples: audio, sampleRate: 24000, to: URL(fileURLWithPath: "output_\(i).wav"))
}
```

#### バッチCLI

```bash
# 1行に1テキストのファイルを作成
echo "Hello world.\nGoodbye world." > texts.txt
.build/release/audio speak --batch-file texts.txt --output output.wav --batch-size 4
# output_0.wav, output_1.wav, ... が生成される
```

> バッチモードはモデル重みのロードを各アイテム間で償却します。Apple SiliconのB=4でスループットが~1.5-2.5倍向上します。テキストが似た長さの音声を生成する場合に最適です。

### サンプリングオプション

```swift
let config = SamplingConfig(temperature: 0.9, topK: 50, repetitionPenalty: 1.05)
let audio = model.synthesize(text: "Hello", language: "english", sampling: config)
```

### ストリーミング合成

最初のパケットの低レイテンシーのために、音声チャンクを段階的に出力：

```swift
let stream = model.synthesizeStream(
    text: "Hello, this is streaming synthesis.",
    language: "english",
    streaming: .lowLatency  // 最初の音声チャンクまで~120ms
)

for try await chunk in stream {
    // chunk.samples: [Float] PCM @ 24kHz
    // chunk.isFinal: 最後のチャンクでtrue
    playAudio(chunk.samples)
}
```

CLI:

```bash
# デフォルトストリーミング (3フレーム初期チャンク、~225msレイテンシー)
.build/release/audio speak "Hello world" --stream

# 低レイテンシー (1フレーム初期チャンク、~120msレイテンシー)
.build/release/audio speak "Hello world" --stream --first-chunk-frames 1
```

## 音声間変換 — 全二重音声会話

> マイク入力によるインタラクティブな音声アシスタントは **[PersonaPlexDemo](Examples/PersonaPlexDemo/)** をご覧ください。タップして話しかけ、自動音声検出によるマルチターン会話が可能です。

### 音声間変換

```swift
import PersonaPlex
import AudioCommon  // WAVWriter, AudioFileLoader用

let model = try await PersonaPlexModel.fromPretrained()
// 初回実行時に~5.5 GBをダウンロード (temporal 4-bit + depformer + Mimiコーデック + ボイスプリセット)

let audio = try AudioFileLoader.load(url: inputURL, targetSampleRate: 24000)
let (response, textTokens) = model.respond(userAudio: audio, voice: .NATM0)
// response: 24kHzモノラルfloatサンプル
// textTokens: モデルの内部独白 (SentencePieceトークンID)
try WAVWriter.write(samples: response.audio, sampleRate: 24000, to: outputURL)
```

### 内部独白 (テキスト出力)

PersonaPlexは音声と並行してテキストトークンを生成します。これはモデルの内部推論です。内蔵のSentencePieceデコーダーでデコードできます：

```swift
let decoder = try SentencePieceDecoder(modelPath: "tokenizer_spm_32k_3.model")
let transcript = decoder.decode(textTokens)
print(transcript)  // 例: "Sure, I can help you with that..."
```

### ストリーミング音声間変換

```swift
// 生成されたそばから音声チャンクを受信 (チャンクあたり~2秒)
let stream = model.respondStream(userAudio: audio, voice: .NATM0)
for try await chunk in stream {
    playAudio(chunk.samples)  // 即座に再生、24kHzモノラル
    // chunk.textTokensにこのチャンクのテキスト; 最終チャンクに全トークン
    if chunk.isFinal { break }
}
```

### ボイス選択

18種類のボイスプリセットが利用可能：
- **ナチュラル女性**: NATF0, NATF1, NATF2, NATF3
- **ナチュラル男性**: NATM0, NATM1, NATM2, NATM3
- **バラエティ女性**: VARF0, VARF1, VARF2, VARF3, VARF4
- **バラエティ男性**: VARM0, VARM1, VARM2, VARM3, VARM4

### システムプロンプト

システムプロンプトはモデルの会話動作を制御します。任意のカスタムプロンプトをプレーンな文字列として渡すことができます：

```swift
// カスタムシステムプロンプト（自動的にトークン化されます）
let response = model.respond(
    userAudio: audio,
    voice: .NATM0,
    systemPrompt: "You enjoy having a good conversation."
)

// またはプリセットを使用
let response = model.respond(
    userAudio: audio,
    voice: .NATM0,
    systemPromptTokens: SystemPromptPreset.customerService.tokens
)
```

利用可能なプリセット: `focused` (デフォルト)、`assistant`、`customerService`、`teacher`。

### PersonaPlex CLI

```bash
make build

# 基本的な音声間変換
.build/release/audio respond --input question.wav --output response.wav

# トランスクリプト付き (内部独白テキストをデコード)
.build/release/audio respond --input question.wav --transcript

# JSON出力 (音声パス、トランスクリプト、レイテンシーメトリクス)
.build/release/audio respond --input question.wav --json

# カスタムシステムプロンプトテキスト
.build/release/audio respond --input question.wav --system-prompt-text "You enjoy having a good conversation."

# ボイスとシステムプロンプトプリセットを選択
.build/release/audio respond --input question.wav --voice NATF1 --system-prompt focused

# サンプリングパラメーターの調整
.build/release/audio respond --input question.wav --audio-temp 0.6 --repetition-penalty 1.5

# テキストエントロピー早期停止を有効化 (テキストが崩壊した場合に停止)
.build/release/audio respond --input question.wav --entropy-threshold 1.0 --entropy-window 5

# 利用可能なボイスとプロンプトを一覧表示
.build/release/audio respond --list-voices
.build/release/audio respond --list-prompts
```

## CosyVoice TTS — 音声クローン対応ストリーミングテキスト読み上げ

### 基本的な音声合成

```swift
import CosyVoiceTTS
import AudioCommon  // WAVWriter用

let model = try await CosyVoiceTTSModel.fromPretrained()
// 初回実行時に~1.9 GBをダウンロード (LLM + DiT + HiFi-GAN重み)

let audio = model.synthesize(text: "Hello, how are you today?", language: "english")
// 出力は24kHzモノラルfloatサンプル
try WAVWriter.write(samples: audio, sampleRate: 24000, to: outputURL)
```

### ストリーミング合成

```swift
// ストリーミング: 生成された音声チャンクを順次受信 (最初のチャンクまで~150ms)
for try await chunk in model.synthesizeStream(text: "Hello, how are you today?", language: "english") {
    // chunk.audio: [Float], chunk.sampleRate: Int
    playAudio(chunk.audio)  // 即座に再生
}
```

### 音声クローン (CosyVoice)

CAM++話者エンコーダー (192次元、CoreML Neural Engine) を使って話者の声をクローン：

```swift
import CosyVoiceTTS
import AudioCommon

let model = try await CosyVoiceTTSModel.fromPretrained()
let speaker = try await CamPlusPlusSpeaker.fromPretrained()
// 初回使用時に~14 MB CAM++ CoreMLモデルをダウンロード

let refAudio = try AudioFileLoader.load(url: referenceURL, targetSampleRate: 16000)
let embedding = try speaker.embed(audio: refAudio, sampleRate: 16000)
// embedding: 長さ192の[Float]

let audio = model.synthesize(
    text: "Hello in a cloned voice!",
    language: "english",
    speakerEmbedding: embedding
)
```

### CosyVoice TTS CLI

```bash
make build

# 基本的な合成
.build/release/audio speak "Hello world" --engine cosyvoice --language english --output output.wav

# 音声クローン (初回使用時にCAM++話者エンコーダーをダウンロード)
.build/release/audio speak "Hello world" --engine cosyvoice --voice-sample reference.wav --output cloned.wav

# 音声クローンによるマルチスピーカー対話
.build/release/audio speak "[S1] Hello there! [S2] Hey, how are you?" \
    --engine cosyvoice --speakers s1=alice.wav,s2=bob.wav -o dialogue.wav

# インライン感情・スタイルタグ
.build/release/audio speak "(excited) Wow, amazing! (sad) But I have to go..." \
    --engine cosyvoice -o emotion.wav

# 組み合わせ: 対話 + 感情 + 音声クローン
.build/release/audio speak "[S1] (happy) Great news! [S2] (surprised) Really?" \
    --engine cosyvoice --speakers s1=alice.wav,s2=bob.wav -o combined.wav

# カスタムスタイル指示
.build/release/audio speak "Hello world" --engine cosyvoice --cosy-instruct "Speak cheerfully" -o cheerful.wav

# ストリーミング合成
.build/release/audio speak "Hello world" --engine cosyvoice --language english --stream --output output.wav
```

## Kokoro TTS — 軽量オンデバイステキスト読み上げ (iOS + macOS)

### 基本的な音声合成

```swift
import KokoroTTS
import AudioCommon  // WAVWriter用

let tts = try await KokoroTTSModel.fromPretrained()
// 初回実行時に~170 MBをダウンロード (CoreMLモデル + ボイス埋め込み + 辞書)

let audio = try tts.synthesize(text: "Hello world", voice: "af_heart")
// 出力は24kHzモノラルfloatサンプル
try WAVWriter.write(samples: audio, sampleRate: 24000, to: outputURL)
```

10言語で54種類のプリセットボイス。非自己回帰型 — 3段階CoreMLパイプラインでサンプリングループなし。Neural Engine上で動作し、GPUを完全に解放します。

### Kokoro TTS CLI

```bash
make build

# 基本的な合成
.build/release/audio kokoro "Hello world" --voice af_heart --output hello.wav

# 言語を選択
.build/release/audio kokoro "Bonjour le monde" --voice ff_siwis --language fr --output bonjour.wav

# 利用可能なボイスを一覧表示
.build/release/audio kokoro --list-voices
```

## Qwen3 Chat (オンデバイスLLM)

```swift
import Qwen3Chat

let chat = try await Qwen3ChatModel.fromPretrained()
// 初回実行時に~318 MBをダウンロード (INT4 CoreMLモデル + トークナイザー)

// 単一応答
let response = try chat.generate("What is Swift?", systemPrompt: "Answer briefly.")
print(response)

// ストリーミングトークン
let stream = chat.chatStream("Tell me a joke", systemPrompt: "Be funny.")
for try await token in stream {
    print(token, terminator: "")
}
```

Qwen3-0.6B INT4量子化CoreML。Neural Engine上で動作し、iPhone ~2 tok/s、Mシリーズ ~15 tok/s。KVキャッシュによるマルチターン会話、思考モード (`<think>`トークン)、設定可能なサンプリング (temperature、top-k、top-p、repetition penalty) をサポート。

## 音声区間検出 (VAD) — 音声中の発話を検出

### ストリーミングVAD (Silero)

Silero VAD v5は32msの音声チャンクをサブミリ秒レイテンシーで処理します。マイクやストリームからのリアルタイム音声検出に最適です。

```swift
import SpeechVAD

let vad = try await SileroVADModel.fromPretrained()
// またはCoreMLを使用 (Neural Engine、低消費電力):
// let vad = try await SileroVADModel.fromPretrained(engine: .coreml)

// ストリーミング: 512サンプルチャンク (16kHzで32ms) を処理
let prob = vad.processChunk(samples)  // → 0.0...1.0
vad.resetState()  // 異なる音声ストリーム間で呼び出し

// または全セグメントを一括検出
let segments = vad.detectSpeech(audio: audioSamples, sampleRate: 16000)
for seg in segments {
    print("Speech: \(seg.startTime)s - \(seg.endTime)s")
}
```

### イベント駆動型ストリーミング

```swift
let processor = StreamingVADProcessor(model: vad)

// 任意の長さの音声を供給 — 発話が確認されるとイベントが発行
let events = processor.process(samples: audioBuffer)
for event in events {
    switch event {
    case .speechStarted(let time):
        print("Speech started at \(time)s")
    case .speechEnded(let segment):
        print("Speech: \(segment.startTime)s - \(segment.endTime)s")
    }
}

// ストリーム終了時にフラッシュ
let final = processor.flush()
```

### VAD CLI

```bash
make build

# ストリーミングSilero VAD (32msチャンク)
.build/release/audio vad-stream audio.wav

# CoreMLバックエンド (Neural Engine)
.build/release/audio vad-stream audio.wav --engine coreml

# カスタム閾値
.build/release/audio vad-stream audio.wav --onset 0.6 --offset 0.4

# JSON出力
.build/release/audio vad-stream audio.wav --json

# バッチpyannote VAD (10秒スライディングウィンドウ)
.build/release/audio vad audio.wav
```

## 話者ダイアライゼーション — 誰がいつ話したか

### ダイアライゼーションパイプライン

```swift
import SpeechVAD

let pipeline = try await DiarizationPipeline.fromPretrained()
// またはCoreML埋め込みを使用 (Neural Engine、GPUを解放):
// let pipeline = try await DiarizationPipeline.fromPretrained(embeddingEngine: .coreml)

let result = pipeline.diarize(audio: samples, sampleRate: 16000)
for seg in result.segments {
    print("Speaker \(seg.speakerId): [\(seg.startTime)s - \(seg.endTime)s]")
}
print("\(result.numSpeakers) speakers detected")
```

### 話者埋め込み

```swift
let model = try await WeSpeakerModel.fromPretrained()
// または: let model = try await WeSpeakerModel.fromPretrained(engine: .coreml)
let embedding = model.embed(audio: samples, sampleRate: 16000)
// embedding: 長さ256、L2正規化済みの[Float]

// 話者を比較
let similarity = WeSpeakerModel.cosineSimilarity(embeddingA, embeddingB)
```

### 話者抽出

リファレンス録音を使って特定の話者のセグメントのみを抽出：

```swift
let pipeline = try await DiarizationPipeline.fromPretrained()
let targetEmb = pipeline.embeddingModel.embed(audio: enrollmentAudio, sampleRate: 16000)
let segments = pipeline.extractSpeaker(
    audio: meetingAudio, sampleRate: 16000,
    targetEmbedding: targetEmb
)
```

### Sortformerダイアライゼーション (エンドツーエンド、CoreML)

NVIDIA Sortformerは最大4人の話者のフレーム単位話者アクティビティを直接予測します。埋め込みやクラスタリングは不要です。Neural Engine上で動作します。

```swift
let diarizer = try await SortformerDiarizer.fromPretrained()
let result = diarizer.diarize(audio: samples, sampleRate: 16000, config: .default)
for seg in result.segments {
    print("Speaker \(seg.speakerId): [\(seg.startTime)s - \(seg.endTime)s]")
}
```

### ダイアライゼーションCLI

```bash
make build

# Pyannoteダイアライゼーション (デフォルト)
.build/release/audio diarize meeting.wav

# Sortformerダイアライゼーション (CoreML, Neural Engine)
.build/release/audio diarize meeting.wav --engine sortformer

# CoreML埋め込み (Neural Engine、pyannoteのみ)
.build/release/audio diarize meeting.wav --embedding-engine coreml

# JSON出力
.build/release/audio diarize meeting.wav --json

# 特定の話者を抽出 (pyannoteのみ)
.build/release/audio diarize meeting.wav --target-speaker enrollment.wav

# 話者埋め込み
.build/release/audio embed-speaker enrollment.wav --json
.build/release/audio embed-speaker enrollment.wav --engine coreml
```

アーキテクチャの詳細は[話者ダイアライゼーション](docs/inference/speaker-diarization.md)を参照。

## 音声強調 — ノイズ抑制とオーディオクリーンアップ

### ノイズ抑制

```swift
import SpeechEnhancement
import AudioCommon  // WAVWriter用

let enhancer = try await SpeechEnhancer.fromPretrained()
// 初回実行時に~4.3 MBをダウンロード (Core ML FP16モデル + 補助データ)

let cleanAudio = try enhancer.enhance(audio: noisyAudio, sampleRate: 48000)
try WAVWriter.write(samples: cleanAudio, sampleRate: 48000, to: outputURL)
```

### ノイズ除去CLI

```bash
make build

# 基本的なノイズ除去
.build/release/audio denoise noisy.wav

# カスタム出力パス
.build/release/audio denoise noisy.wav --output clean.wav
```

アーキテクチャの詳細は[音声強調](docs/inference/speech-enhancement.md)を参照。

## パイプライン — 複数モデルの組み合わせ

すべてのモデルは共有プロトコル (`SpeechRecognitionModel`、`SpeechGenerationModel`、`SpeechEnhancementModel`など) に準拠しており、パイプラインとして組み合わせることができます：

### ノイズの多い音声の認識 (DeepFilterNet + ASR)

```swift
import SpeechEnhancement
import Qwen3ASR

let enhancer = try await SpeechEnhancer.fromPretrained()
let asr = try await Qwen3ASRModel.fromPretrained()

// 48kHzで強調し、16kHzで文字起こし
let clean = try enhancer.enhance(audio: noisyAudio, sampleRate: 48000)
let clean16k = AudioResampler.resample(clean, from: 48000, to: 16000)
let text = asr.transcribe(audio: clean16k, sampleRate: 16000)
```

### 音声リレー (VAD + ASR + TTS)

```swift
import SpeechVAD
import Qwen3ASR
import Qwen3TTS

let vad = try await SileroVADModel.fromPretrained()
let asr = try await Qwen3ASRModel.fromPretrained()
let tts = try await Qwen3TTSModel.fromPretrained()

// 音声セグメントを検出し、文字起こしし、再合成
let segments = vad.detectSpeech(audio: audio, sampleRate: 16000)
for seg in segments {
    let chunk = Array(audio[Int(seg.startTime * 16000)..<Int(seg.endTime * 16000)])
    let text = asr.transcribe(audio: chunk, sampleRate: 16000)
    let speech = tts.synthesize(text: text, language: "english")
    // speech: 24kHzモノラルfloatサンプル
}
```

### 会議の文字起こし (ダイアライゼーション + ASR)

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

プロトコルの完全なリファレンスは[共有プロトコル](docs/shared-protocols.md)を参照。

## HTTP APIサーバー

スタンドアロンのHTTPサーバーが、すべてのモデルをRESTおよびWebSocketエンドポイントで公開します。モデルは最初のリクエスト時に遅延ロードされます。

```bash
swift build -c release
.build/release/audio-server --port 8080

# 音声を文字起こし
curl -X POST http://localhost:8080/transcribe --data-binary @audio.wav -H "Content-Type: audio/wav"

# テキスト読み上げ
curl -X POST http://localhost:8080/speak -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "engine": "cosyvoice"}' -o output.wav

# 音声間変換 (PersonaPlex)
curl -X POST http://localhost:8080/respond --data-binary @question.wav -o response.wav

# 音声強調
curl -X POST http://localhost:8080/enhance --data-binary @noisy.wav -o clean.wav

# 起動時にすべてのモデルをプリロード
.build/release/audio-server --preload --port 8080
```

### WebSocketストリーミング

#### OpenAI Realtime API (`/v1/realtime`)

主要WebSocketエンドポイントは[OpenAI Realtime API](https://platform.openai.com/docs/api-reference/realtime)プロトコルを実装しています。すべてのメッセージは`type`フィールドを持つJSONで、音声はbase64エンコードされたPCM16 24kHzモノラルです。

**クライアント → サーバーイベント:**

| イベント | 説明 |
|-------|-------------|
| `session.update` | エンジン、言語、音声フォーマットの設定 |
| `input_audio_buffer.append` | base64 PCM16音声チャンクの送信 |
| `input_audio_buffer.commit` | 蓄積された音声の文字起こし (ASR) |
| `input_audio_buffer.clear` | 音声バッファのクリア |
| `response.create` | TTS合成のリクエスト |

**サーバー → クライアントイベント:**

| イベント | 説明 |
|-------|-------------|
| `session.created` | セッション初期化完了 |
| `session.updated` | 設定確認 |
| `input_audio_buffer.committed` | 文字起こしのための音声コミット完了 |
| `conversation.item.input_audio_transcription.completed` | ASR結果 |
| `response.audio.delta` | base64 PCM16音声チャンク (TTS) |
| `response.audio.done` | 音声ストリーミング完了 |
| `response.done` | メタデータ付きレスポンス完了 |
| `error` | タイプとメッセージ付きエラー |

```javascript
const ws = new WebSocket('ws://localhost:8080/v1/realtime');

// ASR: 音声を送信し、文字起こしを取得
ws.send(JSON.stringify({ type: 'input_audio_buffer.append', audio: base64PCM16 }));
ws.send(JSON.stringify({ type: 'input_audio_buffer.commit' }));
// → 受信: conversation.item.input_audio_transcription.completed

// TTS: テキストを送信し、ストリーミング音声を取得
ws.send(JSON.stringify({
  type: 'response.create',
  response: { modalities: ['audio', 'text'], instructions: 'Hello world' }
}));
// → 受信: response.audio.delta (base64チャンク), response.audio.done, response.done
```

サンプルHTMLクライアントが`Examples/websocket-client.html`にあります。サーバー稼働中にブラウザで開いてください。

サーバーは独立した`AudioServer`モジュールと`audio-server`実行ファイルです。メインの`audio` CLIにHummingbird/WebSocketを追加することはありません。

## レイテンシー (M2 Max, 64 GB)

### ASR

| モデル | バックエンド | RTF | 10秒音声の処理時間 |
|-------|---------|-----|------------------------|
| Qwen3-ASR-0.6B (4-bit) | MLX | ~0.06 | ~0.6s |
| Qwen3-ASR-0.6B (INT8) | CoreML + MLX | ~0.09 | ~0.9s |
| Qwen3-ASR-1.7B (8-bit) | MLX | ~0.11 | ~1.1s |
| Parakeet-TDT-0.6B (INT4) | CoreML (Neural Engine) | ~0.12 cold, ~0.03 warm | ~1.2s / ~0.3s |
| Whisper-large-v3 | whisper.cpp (Q5_0) | ~0.10 | ~1.0s |
| Whisper-small | whisper.cpp (Q5_0) | ~0.04 | ~0.4s |

### 強制アラインメント

| モデル | フレームワーク | 20秒音声 | RTF |
|-------|-----------|-----------|-----|
| Qwen3-ForcedAligner-0.6B (4-bit) | MLX Swift (debug) | ~365ms | ~0.018 |

> 非自己回帰型の単一フォワードパス — サンプリングループなし。音声エンコーダーが支配的 (~328ms)、デコーダー単一パスは~37ms。**リアルタイムの55倍高速。**

### TTS

| モデル | フレームワーク | 短い (1s) | 中程度 (3s) | 長い (6s) | ストリーミング最初のパケット |
|-------|-----------|-----------|-------------|------------|----------------------|
| Qwen3-TTS-0.6B (4-bit) | MLX Swift (release) | 1.6s (RTF 1.2) | 2.3s (RTF 0.7) | 3.9s (RTF 0.7) | ~120ms (1-frame) |
| Kokoro-82M | CoreML (Neural Engine) | ~1.4s (RTFx 0.7) | ~4.3s (RTFx 0.7) | ~8.6s (RTFx 0.7) | N/A (非自己回帰型) |
| Apple `AVSpeechSynthesizer` | AVFoundation | 0.08s | 0.08s | 0.17s (RTF 0.02) | N/A |

> Qwen3-TTSは韻律と感情を備えた自然で表現力豊かな音声を生成し、**リアルタイムより高速** (RTF < 1.0) で動作します。ストリーミング合成は最初の音声チャンクを~120msで配信します。Kokoro-82MはNeural Engine上で3段階パイプラインで動作（RTFx約0.7）、iOS向けに最適です。Appleの内蔵TTSはより高速ですが、機械的で単調な音声を生成します。

### PersonaPlex (音声間変換)

| モデル | フレームワーク | ms/step | RTF | 備考 |
|-------|-----------|---------|-----|-------|
| PersonaPlex-7B (8-bit) | MLX Swift (release) | ~112ms | ~1.4 | 推奨 — 一貫した応答、4-bitより30%高速 |
| PersonaPlex-7B (4-bit) | MLX Swift (release) | ~158ms | ~1.97 | 非推奨 — 出力品質が劣化 |

> **8-bitを使用してください。** INT8はより高速（112 ms/step vs. 158 ms/step）で、一貫した全二重応答を生成します。INT4量子化は生成品質を劣化させ、意味不明な音声を生成します。INT8はM2 Maxで~112ms/stepで動作します。

### VAD & 話者埋め込み

| モデル | バックエンド | 1回のレイテンシー | RTF | 備考 |
|-------|---------|-----------------|-----|-------|
| Silero-VAD-v5 | MLX | ~2.1ms / chunk | 0.065 | GPU (Metal) |
| Silero-VAD-v5 | CoreML | ~0.27ms / chunk | 0.008 | Neural Engine、**7.7倍高速** |
| WeSpeaker ResNet34-LM | MLX | ~310ms / 20s音声 | 0.016 | GPU (Metal) |
| WeSpeaker ResNet34-LM | CoreML | ~430ms / 20s音声 | 0.021 | Neural Engine、GPUを解放 |

> Silero VAD CoreMLはNeural Engine上でMLXの7.7倍の速度で動作し、常時オンのマイク入力に最適です。WeSpeaker MLXはGPU上でより高速ですが、CoreMLはGPUを並行ワークロード (TTS、ASR) に解放します。両バックエンドで同等の結果が得られます。

### 音声強調

| モデル | バックエンド | 音声長 | レイテンシー | RTF |
|-------|---------|----------|---------|-----|
| DeepFilterNet3 (FP16) | CoreML | 5s | 0.65s | 0.13 |
| DeepFilterNet3 (FP16) | CoreML | 10s | 1.2s | 0.12 |
| DeepFilterNet3 (FP16) | CoreML | 20s | 4.8s | 0.24 |

RTF = Real-Time Factor (低いほど良い、< 1.0 = リアルタイムより高速)。GRUのコストは~O(n^2)でスケールします。

### MLX vs CoreML

両バックエンドで同等の結果が得られます。ワークロードに応じて選択してください：

| | MLX | CoreML |
|---|---|---|
| **ハードウェア** | GPU (Metal shaders) | Neural Engine + CPU |
| **最適な用途** | 最大スループット、単一モデルワークロード | マルチモデルパイプライン、バックグラウンドタスク |
| **消費電力** | GPU使用率が高い | 低消費電力、GPUを解放 |
| **レイテンシー** | 大型モデルで高速 (WeSpeaker) | 小型モデルで高速 (Silero VAD) |

**デスクトップ推論**: MLXがデフォルト — Apple Silicon上で最速の単一モデルパフォーマンス。複数モデルを同時実行する場合 (例: VAD + ASR + TTS) はGPUの競合を避けるためにCoreMLに切り替えるか、ノートPCのバッテリーに配慮したワークロードに使用してください。

CoreMLモデルはQwen3-ASRエンコーダー、Silero VAD、WeSpeakerで利用可能です。Qwen3-ASRの場合は`--engine qwen3-coreml`を使用 (ハイブリッド: ANE上のCoreMLエンコーダー + GPU上のMLXテキストデコーダー)。VAD/埋め込みの場合は構築時に`engine: .coreml`を渡します。推論APIは同一です。

## 精度ベンチマーク

### ASR — 単語誤り率 ([詳細](docs/benchmarks/asr-wer.md))

| モデル | WER% (LibriSpeech test-clean) | RTF |
|-------|-------------------------------|-----|
| Qwen3-ASR 1.7B 8-bit | **2.35** | 0.090 |
| Qwen3-ASR 1.7B 4-bit | 2.57 | 0.045 |
| Parakeet TDT INT8 | 2.74 | 0.089 |
| Qwen3-ASR 0.6B 8-bit | 2.80 | 0.025 |

Qwen3-ASR 1.7B 8-bitは同等サイズでWhisper Large v3 Turbo (2.5%) を超えます。多言語: FLEURSで10言語のベンチマーク済み。

### TTS — ラウンドトリップ明瞭度 ([詳細](docs/benchmarks/tts-roundtrip.md))

| エンジン | WER% | RTF |
|--------|------|-----|
| CosyVoice3 | **3.25** | 0.59 |
| Qwen3-TTS 1.7B | 3.47 | 0.79 |
| Kokoro-82M | 3.90 | 0.17 |

### VAD — 音声検出 ([詳細](docs/benchmarks/vad-detection.md))

| エンジン | F1% (FLEURS) | RTF |
|--------|-------------|-----|
| FireRedVAD | **99.12** | 0.007 |
| Silero CoreML | 95.13 | 0.022 |
| Pyannote MLX | 94.86 | 0.358 |

## アーキテクチャ

**モデル:** [ASRモデル](docs/models/asr-model.md)、[TTSモデル](docs/models/tts-model.md)、[CosyVoice TTS](docs/models/cosyvoice-tts.md)、[Kokoro TTS](docs/models/kokoro-tts.md)、[Parakeet TDT](docs/models/parakeet-asr.md)、[PersonaPlex](docs/models/personaplex.md)、[FireRedVAD](docs/models/fireredvad.md)

**推論:** [ASR推論](docs/inference/qwen3-asr-inference.md)、[TTS推論](docs/inference/qwen3-tts-inference.md)、[強制アラインメント](docs/inference/forced-aligner.md)、[FireRedVAD](docs/inference/fireredvad.md)、[Silero VAD](docs/inference/silero-vad.md)、[話者ダイアライゼーション](docs/inference/speaker-diarization.md)、[音声強調](docs/inference/speech-enhancement.md)、[オーディオ再生](docs/audio/playback.md)

**ベンチマーク:** [ASR WER](docs/benchmarks/asr-wer.md)、[TTSラウンドトリップ](docs/benchmarks/tts-roundtrip.md)、[VAD検出](docs/benchmarks/vad-detection.md)

**リファレンス:** [共有プロトコル](docs/shared-protocols.md)

## キャッシュ設定

モデルの重みはローカルにキャッシュされます。キャッシュの保存先を変更するには：

```bash
export QWEN3_CACHE_DIR=/path/to/cache
```

## MLX Metalライブラリ

実行時に`Failed to load the default metallib`が表示された場合、Metal shaderライブラリが不足しています。`make build` (または手動の`swift build`の後に`./scripts/build_mlx_metallib.sh release`) を実行してコンパイルしてください。Metal Toolchainが不足している場合は、先にインストールしてください：

```bash
xcodebuild -downloadComponent MetalToolchain
```

## テスト

ユニットテスト (設定、サンプリング、テキスト前処理、タイムスタンプ補正) はモデルのダウンロードなしで実行できます：

```bash
swift test --filter "Qwen3TTSConfigTests|SamplingTests|CosyVoiceTTSConfigTests|CamPlusPlusMelExtractorTests|PersonaPlexTests|ForcedAlignerTests/testText|ForcedAlignerTests/testTimestamp|ForcedAlignerTests/testLIS|SileroVADTests/testSilero|SileroVADTests/testReflection|SileroVADTests/testProcess|SileroVADTests/testReset|SileroVADTests/testDetect|SileroVADTests/testStreaming|SileroVADTests/testVADEvent|KokoroTTSTests"
```

統合テストにはモデルの重みが必要です (初回実行時に自動ダウンロード)：

```bash
# TTSラウンドトリップ: テキストを合成、WAV保存、ASRで逆文字起こし
swift test --filter TTSASRRoundTripTests

# ASRのみ: テスト音声を文字起こし
swift test --filter Qwen3ASRIntegrationTests

# 強制アラインメントE2E: 単語レベルのタイムスタンプ (~979 MBダウンロード)
swift test --filter ForcedAlignerTests/testForcedAlignerE2E

# PersonaPlex E2E: 音声間パイプライン (~5.5 GBダウンロード)
PERSONAPLEX_E2E=1 swift test --filter PersonaPlexE2ETests
```

> **注意:** MLX操作を使用するテストの実行前に、MLX Metalライブラリをビルドする必要があります。
> 手順は[MLX Metalライブラリ](#mlx-metalライブラリ)を参照してください。

## 対応言語

| モデル | 対応言語 |
|-------|-----------|
| Qwen3-ASR | 52言語 (CN, EN, 広東語, DE, FR, ES, JA, KO, RU, + 中国語方言22種, ...) |
| Parakeet TDT | ヨーロッパ25言語 (BG, CS, DA, DE, EL, EN, ES, ET, FI, FR, HR, HU, IT, LT, LV, MT, NL, PL, PT, RO, RU, SK, SL, SV, UK) |
| Qwen3-TTS | EN, CN, DE, JA, ES, FR, KO, RU, IT, PT (+ CustomVoiceで北京語/四川方言) |
| CosyVoice TTS | CN, EN, JA, KO, DE, ES, FR, IT, RU |
| Kokoro TTS | EN (US/UK), ES, FR, HI, IT, JA, PT, CN, KO, DE |
| PersonaPlex | EN |

## 他との比較

### 音声認識 (ASR): speech-swift vs 他の選択肢

| | **speech-swift (Qwen3-ASR)** | **whisper.cpp** | **Apple SFSpeechRecognizer** | **Google Cloud Speech** |
|---|---|---|---|---|
| **ランタイム** | オンデバイス (MLX/CoreML) | オンデバイス (CPU/GPU) | オンデバイスまたはクラウド | クラウドのみ |
| **対応言語** | 52 | 100以上 | ~70 (オンデバイス: 限定的) | 125以上 |
| **RTF (10秒音声, M2 Max)** | 0.06 (リアルタイムの17倍) | 0.10 (Whisper-large-v3) | N/A | N/A |
| **ストリーミング** | なし (バッチ) | なし (バッチ) | あり | あり |
| **カスタムモデル** | あり (HuggingFace重みを差し替え) | あり (GGMLモデル) | なし | なし |
| **Swift API** | ネイティブasync/await | C++のSwiftブリッジ | ネイティブ | REST/gRPC |
| **プライバシー** | 完全オンデバイス | 完全オンデバイス | 設定による | クラウドにデータ送信 |
| **単語タイムスタンプ** | あり (強制アラインメント) | あり | 限定的 | あり |
| **コスト** | 無料 (Apache 2.0) | 無料 (MIT) | 無料 (オンデバイス) | 分単位課金 |

### テキスト読み上げ (TTS): speech-swift vs 他の選択肢

| | **speech-swift (Qwen3-TTS)** | **speech-swift (Kokoro)** | **Apple AVSpeechSynthesizer** | **ElevenLabs / クラウドTTS** |
|---|---|---|---|---|
| **品質** | ニューラル、表現力豊か | ニューラル、自然 | 機械的、単調 | ニューラル、最高品質 |
| **ランタイム** | オンデバイス (MLX) | オンデバイス (CoreML) | オンデバイス | クラウドのみ |
| **ストリーミング** | あり (最初のチャンク~120ms) | なし (3段階パイプライン) | なし | あり |
| **音声クローン** | あり | なし | なし | あり |
| **ボイス** | 9種類内蔵 + 任意のクローン | 54種類のプリセット | ~50種類のシステムボイス | 1000以上 |
| **対応言語** | 10 | 10 | 60以上 | 30以上 |
| **iOS対応** | macOSのみ | iOS + macOS | iOS + macOS | あり (API) |
| **コスト** | 無料 (Apache 2.0) | 無料 (Apache 2.0) | 無料 | 文字単位課金 |

### speech-swiftを使うべき場面

- **プライバシー重視のアプリ** — 医療、法律、企業向けなど、音声をデバイス外に送信できない場合
- **オフライン利用** — 初回モデルダウンロード後はインターネット接続不要
- **コスト重視** — 分単位や文字単位のAPI課金なし
- **Apple Silicon最適化** — Mシリーズ GPU (Metal) とNeural Engine専用設計
- **フルパイプライン** — ASR + TTS + VAD + ダイアライゼーション + 音声強調を単一のSwiftパッケージで統合

## FAQ

**speech-swiftはiOSで動作しますか？**
Kokoro TTS、Qwen3.5-Chat (CoreML)、Silero VAD、Parakeet ASR、DeepFilterNet3、WeSpeakerはすべてiOS 17以上でCoreML (Neural Engine) を使って動作します。MLXベースのモデル (Qwen3-ASR、Qwen3-TTS、Qwen3.5-Chat MLX、PersonaPlex) はmacOS 14以上のApple Siliconが必要です。

**インターネット接続は必要ですか？**
HuggingFaceからの初回モデルダウンロード時のみ必要です (自動、`~/Library/Caches/qwen3-speech/`にキャッシュ)。以降はすべての推論がネットワークアクセスなしで完全にオフラインで実行されます。

**speech-swiftはWhisperと比べてどうですか？**
Qwen3-ASR-0.6BはM2 MaxでRTF 0.06を達成 — whisper.cpp経由のWhisper-large-v3 (RTF 0.10) より40%高速で、52言語にわたり同等の精度を備えています。speech-swiftはネイティブSwift async/await APIを提供し、whisper.cppはC++ブリッジが必要です。

**商用アプリに使えますか？**
はい。speech-swiftはApache 2.0ライセンスです。基盤となるモデルの重みにはそれぞれのライセンスがあります (各モデルのHuggingFaceページをご確認ください)。

**どのApple Siliconチップに対応していますか？**
すべてのMシリーズチップ: M1、M2、M3、M4とそのPro/Max/Ultraバリアント。macOS 14以上 (Sonoma) またはiOS 17以上が必要です。

**必要なメモリはどのくらいですか？**
~3 MB (Silero VAD) から~6.5 GB (PersonaPlex 7B) まで。Kokoro TTSは~500 MB、Qwen3-ASRは~2.2 GB使用します。詳細は[メモリ要件](#メモリ要件)の表を参照してください。

**複数のモデルを同時に実行できますか？**
はい。Neural Engine上のCoreMLモデルとGPU上のMLXモデルを併用することで競合を避けられます。例: Silero VAD (CoreML) + Qwen3-ASR (MLX) + Qwen3-TTS (MLX)。

**REST APIはありますか？**
はい。`audio-server`バイナリが、HTTP RESTおよびWebSocketエンドポイントですべてのモデルを公開します。`/v1/realtime`にOpenAI Realtime API互換のWebSocketも備えています。

## コントリビューション

コントリビューションを歓迎します！バグ修正、新しいモデルの統合、ドキュメントの改善など、プルリクエストをお待ちしています。

**始め方：**
1. リポジトリをフォークしてフィーチャーブランチを作成
2. `make build`でコンパイル (Xcode + Metal Toolchainが必要)
3. `make test`でテストスイートを実行
4. `main`に対してPRを作成

## ライセンス

Apache 2.0
