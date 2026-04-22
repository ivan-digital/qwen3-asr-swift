# Speech Swift

Apple Silicon向けAI音声モデル。MLX SwiftとCoreMLで動作します。

📖 Read in: [English](README.md) · [中文](README_zh.md) · [日本語](README_ja.md) · [한국어](README_ko.md) · [Español](README_es.md) · [Deutsch](README_de.md) · [Français](README_fr.md) · [हिन्दी](README_hi.md) · [Português](README_pt.md) · [Русский](README_ru.md)

Mac・iOS向けのオンデバイス音声認識・合成・理解。Apple Silicon上で完全にローカル動作します——クラウド不要、APIキー不要、データはデバイスから外部に送信されません。

**[📚 ドキュメント →](https://soniqo.audio/ja)** · **[🤗 HuggingFaceモデル](https://huggingface.co/aufklarer)** · **[📝 ブログ](https://blog.ivan.digital)**

- **[Qwen3-ASR](https://soniqo.audio/ja/guides/transcribe)** — 音声認識（自動音声認識、52言語、MLX + CoreML）
- **[Parakeet TDT](https://soniqo.audio/ja/guides/parakeet)** — CoreMLによる音声認識（Neural Engine、NVIDIA FastConformer + TDTデコーダー、25言語）
- **[Omnilingual ASR](https://soniqo.audio/ja/guides/omnilingual)** — 音声認識（Meta wav2vec2 + CTC、**1,672言語**、32文字体系、CoreML 300M + MLX 300M/1B/3B/7B）
- **[ストリーミングディクテーション](https://soniqo.audio/ja/guides/dictate)** — 部分結果と発話終端検出付きのリアルタイムディクテーション（Parakeet-EOU-120M）
- **[Nemotron ストリーミング](https://soniqo.audio/ja/guides/nemotron)** — ネイティブな句読点と大文字化を備えた低レイテンシストリーミングASR（NVIDIA Nemotron-Speech-Streaming-0.6B、CoreML、英語）
- **[Qwen3-ForcedAligner](https://soniqo.audio/ja/guides/align)** — 単語レベルのタイムスタンプ整列（音声 + テキスト → タイムスタンプ）
- **[Qwen3-TTS](https://soniqo.audio/ja/guides/speak)** — 音声合成（最高品質、ストリーミング、カスタムスピーカー、10言語）
- **[CosyVoice TTS](https://soniqo.audio/ja/guides/cosyvoice)** — 音声クローン、マルチスピーカー対話、感情タグを備えたストリーミングTTS（9言語）
- **[Kokoro TTS](https://soniqo.audio/ja/guides/kokoro)** — オンデバイスTTS（82M、CoreML/Neural Engine、54ボイス、iOS対応、10言語）
- **[Qwen3.5-Chat](https://soniqo.audio/ja/guides/chat)** — オンデバイスLLMチャット（0.8B、MLX INT4 + CoreML INT8、DeltaNetハイブリッド、ストリーミングトークン）
- **[PersonaPlex](https://soniqo.audio/ja/guides/respond)** — 全二重音声間会話（7B、音声入力 → 音声出力、18種類のボイスプリセット）
- **[DeepFilterNet3](https://soniqo.audio/ja/guides/denoise)** — リアルタイムノイズ抑制（2.1Mパラメーター、48 kHz）
- **[ウェイクワード](https://soniqo.audio/ja/guides/wake-word)** — オンデバイスのキーワード検出（KWS Zipformer 3M、CoreML、リアルタイムの26倍、キーワードリスト設定可能）
- **[VAD](https://soniqo.audio/ja/guides/vad)** — 音声区間検出（Sileroストリーミング、Pyannoteオフライン、FireRedVAD 100以上の言語）
- **[話者ダイアライゼーション](https://soniqo.audio/ja/guides/diarize)** — 誰がいつ話したか（Pyannoteパイプライン、Neural Engine上のエンドツーエンドSortformer）
- **[話者埋め込み](https://soniqo.audio/ja/guides/embed-speaker)** — WeSpeaker ResNet34（256次元）、CAM++（192次元）

論文：[Qwen3-ASR](https://arxiv.org/abs/2601.21337) (Alibaba) · [Qwen3-TTS](https://arxiv.org/abs/2601.15621) (Alibaba) · [Omnilingual ASR](https://arxiv.org/abs/2511.09690) (Meta) · [Parakeet TDT](https://arxiv.org/abs/2304.06795) (NVIDIA) · [CosyVoice 3](https://arxiv.org/abs/2505.17589) (Alibaba) · [Kokoro](https://arxiv.org/abs/2301.01695) (StyleTTS 2) · [PersonaPlex](https://arxiv.org/abs/2602.06053) (NVIDIA) · [Mimi](https://arxiv.org/abs/2410.00037) (Kyutai) · [Sortformer](https://arxiv.org/abs/2409.06656) (NVIDIA)

## ニュース

- **2026年4月19日** — [Apple SiliconにおけるMLXとCoreML — 正しいバックエンドを選ぶための実践ガイド](https://blog.ivan.digital/mlx-vs-coreml-on-apple-silicon-a-practical-guide-to-picking-the-right-backend-and-why-you-should-f77ddea7b27a)
- **2026年3月20日** — [600MモデルだけでMac上でWhisper Large v3を超えた](https://blog.ivan.digital/we-beat-whisper-large-v3-with-a-600m-model-running-entirely-on-your-mac-20e6ce191174)
- **2026年2月26日** — [Apple Silicon上の話者ダイアライゼーションと音声区間検出 — ネイティブSwift + MLX](https://blog.ivan.digital/speaker-diarization-and-voice-activity-detection-on-apple-silicon-native-swift-with-mlx-92ea0c9aca0f)
- **2026年2月23日** — [Apple Silicon上のNVIDIA PersonaPlex 7B — ネイティブSwift + MLXによる全二重音声間変換](https://blog.ivan.digital/nvidia-personaplex-7b-on-apple-silicon-full-duplex-speech-to-speech-in-native-swift-with-mlx-0aa5276f2e23)
- **2026年2月12日** — [Qwen3-ASR Swift: Apple Silicon向けオンデバイスASR + TTS — アーキテクチャとベンチマーク](https://blog.ivan.digital/qwen3-asr-swift-on-device-asr-tts-for-apple-silicon-architecture-and-benchmarks-27cbf1e4463f)

## クイックスタート

`Package.swift` にパッケージを追加します：

```swift
.package(url: "https://github.com/soniqo/speech-swift", branch: "main")
```

必要なモジュールだけをインポートします。各モデルは個別のSPMライブラリなので、使わないものにコストを払う必要はありません：

```swift
.product(name: "ParakeetStreamingASR", package: "speech-swift"),
.product(name: "SpeechUI",             package: "speech-swift"),  // オプションのSwiftUIビュー
```

**3行で音声バッファを文字起こし：**

```swift
import ParakeetStreamingASR

let model = try await ParakeetStreamingASRModel.fromPretrained()
let text = try model.transcribeAudio(audioSamples, sampleRate: 16000)
```

**部分結果付きのライブストリーミング：**

```swift
for await partial in model.transcribeStream(audio: samples, sampleRate: 16000) {
    print(partial.isFinal ? "FINAL: \(partial.text)" : "... \(partial.text)")
}
```

**約10行のSwiftUIディクテーションビュー：**

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

`SpeechUI` には `TranscriptionView`（確定 + 部分）と `TranscriptionStore`（ストリーミングASRアダプター）のみが含まれます。音声の可視化や再生には AVFoundation をお使いください。

利用可能なSPMプロダクト：`Qwen3ASR`、`Qwen3TTS`、`Qwen3TTSCoreML`、`ParakeetASR`、`ParakeetStreamingASR`、`NemotronStreamingASR`、`OmnilingualASR`、`KokoroTTS`、`CosyVoiceTTS`、`PersonaPlex`、`SpeechVAD`、`SpeechEnhancement`、`Qwen3Chat`、`SpeechCore`、`SpeechUI`、`AudioCommon`。

## モデル

以下はコンパクト表示です。**[サイズ、量子化、ダウンロードURL、メモリ表を含む完全なモデルカタログ → soniqo.audio/architecture](https://soniqo.audio/ja/architecture)**。

| モデル | タスク | バックエンド | サイズ | 言語 |
|-------|------|----------|-------|-----------|
| [Qwen3-ASR](https://soniqo.audio/ja/guides/transcribe) | 音声 → テキスト | MLX、CoreML（ハイブリッド） | 0.6B、1.7B | 52 |
| [Parakeet TDT](https://soniqo.audio/ja/guides/parakeet) | 音声 → テキスト | CoreML (ANE) | 0.6B | 25欧州言語 |
| [Parakeet EOU](https://soniqo.audio/ja/guides/dictate) | 音声 → テキスト（ストリーミング） | CoreML (ANE) | 120M | 25欧州言語 |
| [Nemotron Streaming](https://soniqo.audio/ja/guides/nemotron) | 音声 → テキスト（ストリーミング、句読点付き） | CoreML (ANE) | 0.6B | 英語 |
| [Omnilingual ASR](https://soniqo.audio/ja/guides/omnilingual) | 音声 → テキスト | CoreML (ANE)、MLX | 300M / 1B / 3B / 7B | **[1,672](https://github.com/facebookresearch/omnilingual-asr/blob/main/src/omnilingual_asr/models/wav2vec2_llama/lang_ids.py)** |
| [Qwen3-ForcedAligner](https://soniqo.audio/ja/guides/align) | 音声 + テキスト → タイムスタンプ | MLX、CoreML | 0.6B | 多言語 |
| [Qwen3-TTS](https://soniqo.audio/ja/guides/speak) | テキスト → 音声 | MLX、CoreML | 0.6B、1.7B | 10 |
| [CosyVoice3](https://soniqo.audio/ja/guides/cosyvoice) | テキスト → 音声 | MLX | 0.5B | 9 |
| [Kokoro-82M](https://soniqo.audio/ja/guides/kokoro) | テキスト → 音声 | CoreML (ANE) | 82M | 10 |
| [Qwen3.5-Chat](https://soniqo.audio/ja/guides/chat) | テキスト → テキスト（LLM） | MLX、CoreML | 0.8B | 多言語 |
| [PersonaPlex](https://soniqo.audio/ja/guides/respond) | 音声 → 音声 | MLX | 7B | EN |
| [Silero VAD](https://soniqo.audio/ja/guides/vad) | 音声区間検出 | MLX、CoreML | 309K | 言語非依存 |
| [Pyannote](https://soniqo.audio/ja/guides/diarize) | VAD + ダイアライゼーション | MLX | 1.5M | 言語非依存 |
| [Sortformer](https://soniqo.audio/ja/guides/diarize) | ダイアライゼーション（E2E） | CoreML (ANE) | — | 言語非依存 |
| [DeepFilterNet3](https://soniqo.audio/ja/guides/denoise) | 音声強調 | CoreML | 2.1M | 言語非依存 |
| [WeSpeaker](https://soniqo.audio/ja/guides/embed-speaker) | 話者埋め込み | MLX、CoreML | 6.6M | 言語非依存 |

## インストール

### Homebrew

ネイティブARM Homebrew（`/opt/homebrew`）が必要です。Rosetta/x86_64 Homebrewはサポートされません。

```bash
brew tap soniqo/speech https://github.com/soniqo/speech-swift
brew install speech
```

その後：

```bash
audio transcribe recording.wav
audio speak "Hello world"
audio respond --input question.wav --transcript
audio-server --port 8080            # ローカル HTTP / WebSocket サーバー（OpenAI 互換 /v1/realtime）
```

**[完全なCLIリファレンス →](https://soniqo.audio/ja/cli)**

### Swift Package Manager

```swift
dependencies: [
    .package(url: "https://github.com/soniqo/speech-swift", branch: "main")
]
```

必要なものだけをインポート。各モデルは個別のSPMターゲットです：

```swift
import Qwen3ASR             // 音声認識 (MLX)
import ParakeetASR          // 音声認識 (CoreML、バッチ)
import ParakeetStreamingASR // 部分結果 + EOU付きストリーミングディクテーション
import NemotronStreamingASR // 英語ストリーミングASR、ネイティブ句読点付き（0.6B）
import OmnilingualASR       // 1,672言語 (CoreML + MLX)
import Qwen3TTS             // 音声合成
import CosyVoiceTTS         // 音声クローン付き音声合成
import KokoroTTS            // 音声合成 (iOS対応)
import Qwen3Chat            // オンデバイスLLMチャット
import PersonaPlex          // 全二重音声間変換
import SpeechVAD            // VAD + 話者ダイアライゼーション + 埋め込み
import SpeechEnhancement    // ノイズ抑制
import SpeechUI             // ストリーミングトランスクリプト用SwiftUIコンポーネント
import AudioCommon          // 共有プロトコルとユーティリティ
```

### 動作要件

- Swift 6+、Xcode 16+（Metal Toolchainを含む）
- macOS 15+（Sequoia）または iOS 18+、Apple Silicon（M1/M2/M3/M4）

macOS 15 / iOS 18 の最小要件は [MLState](https://developer.apple.com/documentation/coreml/mlstate) —— Apple の永続的 ANE ステート API —— に由来します。CoreML パイプライン（Qwen3-ASR、Qwen3-Chat、Qwen3-TTS）は MLState を用いて、KV キャッシュをトークンステップ間で Neural Engine 上に常駐させます。

### ソースからビルド

```bash
git clone https://github.com/soniqo/speech-swift
cd speech-swift
make build
```

`make build` はSwiftパッケージ**と** MLX Metalシェーダーライブラリを同時にコンパイルします。Metalライブラリは GPU 推論に必要です——これがないと実行時に `Failed to load the default metallib` が出ます。`make debug` でデバッグビルド、`make test` でテストスイートを実行します。

**[完全なビルド・インストールガイド →](https://soniqo.audio/ja/getting-started)**

## デモアプリ

- **[DictateDemo](Examples/DictateDemo/)**（[ドキュメント](https://soniqo.audio/ja/guides/dictate)）— macOSメニューバーのストリーミングディクテーション。ライブ部分結果、VADベースの発話終端検出、ワンクリックコピー付き。バックグラウンドagentとして動作（Parakeet-EOU-120M + Silero VAD）。
- **[iOSEchoDemo](Examples/iOSEchoDemo/)** — iOSエコーデモ（Parakeet ASR + Kokoro TTS）。実機・シミュレーター対応。
- **[PersonaPlexDemo](Examples/PersonaPlexDemo/)** — マイク入力、VAD、マルチターンコンテキスト付きの対話型音声アシスタント。macOS。M2 MaxでRTF約0.94（リアルタイムより高速）。
- **[SpeechDemo](Examples/SpeechDemo/)** — タブ形式インターフェース上でのディクテーションとTTS合成。macOS。

各デモのREADMEにビルド手順があります。

## コード例

以下のスニペットは、各領域の最小限の使い方を示しています。各セクションは [soniqo.audio](https://soniqo.audio/ja) 上の完全ガイドにリンクしており、設定オプション、複数のバックエンド、ストリーミングパターン、CLIレシピが載っています。

### 音声認識 — [完全ガイド →](https://soniqo.audio/ja/guides/transcribe)

```swift
import Qwen3ASR

let model = try await Qwen3ASRModel.fromPretrained()
let text = model.transcribe(audio: audioSamples, sampleRate: 16000)
```

代替バックエンド：[Parakeet TDT](https://soniqo.audio/ja/guides/parakeet)（CoreML、32×リアルタイム）、[Omnilingual ASR](https://soniqo.audio/ja/guides/omnilingual)（1,672言語、CoreMLまたはMLX）、[ストリーミングディクテーション](https://soniqo.audio/ja/guides/dictate)（ライブ部分結果）。

### 強制整列 — [完全ガイド →](https://soniqo.audio/ja/guides/align)

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

### 音声合成 — [完全ガイド →](https://soniqo.audio/ja/guides/speak)

```swift
import Qwen3TTS
import AudioCommon

let model = try await Qwen3TTSModel.fromPretrained()
let audio = model.synthesize(text: "Hello world", language: "english")
try WAVWriter.write(samples: audio, sampleRate: 24000, to: outputURL)
```

代替TTSエンジン：[CosyVoice3](https://soniqo.audio/ja/guides/cosyvoice)（ストリーミング + 音声クローン + 感情タグ）、[Kokoro-82M](https://soniqo.audio/ja/guides/kokoro)（iOS対応、54ボイス）、[音声クローン](https://soniqo.audio/ja/guides/voice-cloning)。

### 音声間変換 — [完全ガイド →](https://soniqo.audio/ja/guides/respond)

```swift
import PersonaPlex

let model = try await PersonaPlexModel.fromPretrained()
let responseAudio = model.respond(userAudio: userSamples)
// 24 kHz モノラル Float32 出力、再生可能
```

### LLMチャット — [完全ガイド →](https://soniqo.audio/ja/guides/chat)

```swift
import Qwen3Chat

let chat = try await Qwen35MLXChat.fromPretrained()
chat.chat(messages: [(.user, "Explain MLX in one sentence")]) { token, isFinal in
    print(token, terminator: "")
}
```

### 音声区間検出 — [完全ガイド →](https://soniqo.audio/ja/guides/vad)

```swift
import SpeechVAD

let vad = try await SileroVADModel.fromPretrained()
let segments = vad.detectSpeech(audio: samples, sampleRate: 16000)
for s in segments { print("\(s.startTime)s → \(s.endTime)s") }
```

### 話者ダイアライゼーション — [完全ガイド →](https://soniqo.audio/ja/guides/diarize)

```swift
import SpeechVAD

let diarizer = try await DiarizationPipeline.fromPretrained()
let segments = diarizer.diarize(audio: samples, sampleRate: 16000)
for s in segments { print("Speaker \(s.speakerId): \(s.startTime)s - \(s.endTime)s") }
```

### 音声強調 — [完全ガイド →](https://soniqo.audio/ja/guides/denoise)

```swift
import SpeechEnhancement

let denoiser = try await DeepFilterNet3Model.fromPretrained()
let clean = try denoiser.enhance(audio: noisySamples, sampleRate: 48000)
```

### 音声パイプライン（ASR → LLM → TTS）— [完全ガイド →](https://soniqo.audio/ja/api)

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

`VoicePipeline` はリアルタイム音声agentのステートマシンで（[speech-core](https://github.com/soniqo/speech-core)が駆動）、VADベースのターン検出、割り込み処理、イーガーSTTを備えています。任意の `SpeechRecognitionModel` + `SpeechGenerationModel` + `StreamingVADProvider` を接続できます。

### HTTP APIサーバー

```bash
audio-server --port 8080
```

HTTP REST + WebSocketエンドポイントですべてのモデルを公開します。OpenAI Realtime API互換のWebSocket `/v1/realtime` も含まれます。[`Sources/AudioServer/`](Sources/AudioServer/) を参照してください。

## アーキテクチャ

speech-swift はモデルごとに1つのSPMターゲットに分割されており、利用者はインポートした分だけコストを負担します。共有インフラは `AudioCommon`（プロトコル、音声I/O、HuggingFaceダウンローダー、`SentencePieceModel`）と `MLXCommon`（ウェイトローディング、`QuantizedLinear` ヘルパー、`SDPA` マルチヘッドアテンションヘルパー）にあります。

**[バックエンド、メモリ表、モジュールマップ付きの完全なアーキテクチャ図 → soniqo.audio/architecture](https://soniqo.audio/ja/architecture)** · **[APIリファレンス → soniqo.audio/api](https://soniqo.audio/ja/api)** · **[ベンチマーク → soniqo.audio/benchmarks](https://soniqo.audio/ja/benchmarks)**

ローカルドキュメント（リポジトリ内）：
- **モデル：** [Qwen3-ASR](docs/models/asr-model.md) · [Qwen3-TTS](docs/models/tts-model.md) · [CosyVoice](docs/models/cosyvoice-tts.md) · [Kokoro](docs/models/kokoro-tts.md) · [Parakeet TDT](docs/models/parakeet-asr.md) · [Parakeet Streaming](docs/models/parakeet-streaming-asr.md) · [Nemotron Streaming](docs/models/nemotron-streaming.md) · [Omnilingual ASR](docs/models/omnilingual-asr.md) · [PersonaPlex](docs/models/personaplex.md) · [FireRedVAD](docs/models/fireredvad.md)
- **推論：** [Qwen3-ASR](docs/inference/qwen3-asr-inference.md) · [Parakeet TDT](docs/inference/parakeet-asr-inference.md) · [Parakeet Streaming](docs/inference/parakeet-streaming-asr-inference.md) · [Nemotron Streaming](docs/inference/nemotron-streaming-inference.md) · [Omnilingual ASR](docs/inference/omnilingual-asr-inference.md) · [TTS](docs/inference/qwen3-tts-inference.md) · [Forced Aligner](docs/inference/forced-aligner.md) · [Silero VAD](docs/inference/silero-vad.md) · [話者ダイアライゼーション](docs/inference/speaker-diarization.md) · [音声強調](docs/inference/speech-enhancement.md)
- **リファレンス：** [共有プロトコル](docs/shared-protocols.md)

## キャッシュ設定

モデルの重みは初回使用時にHuggingFaceからダウンロードされ、`~/Library/Caches/qwen3-speech/` にキャッシュされます。`QWEN3_CACHE_DIR`（CLI）または `cacheDir:`（Swift API）で上書き可能です。すべての `fromPretrained()` エントリーポイントは `offlineMode: true` を受け付け、重みがすでにキャッシュされている場合はネットワークをスキップします。

iOSサンドボックスコンテナのパスを含む詳細は [`docs/inference/cache-and-offline.md`](docs/inference/cache-and-offline.md) を参照してください。

## MLX Metalライブラリ

実行時に `Failed to load the default metallib` と表示された場合は、Metalシェーダーライブラリが不足しています。`make build` または手動 `swift build` の後に `./scripts/build_mlx_metallib.sh release` を実行してください。Metal Toolchainがない場合はまずインストールします：

```bash
xcodebuild -downloadComponent MetalToolchain
```

## テスト

```bash
make test                            # 完全スイート（ユニット + モデルダウンロード付きE2E）
swift test --skip E2E                # ユニットのみ（CIセーフ、ダウンロードなし）
swift test --filter Qwen3ASRTests    # 指定モジュール
```

E2Eテストクラスは `E2E` プレフィックスを使うため、CIは `--skip E2E` でそれらを除外できます。完全なテスト規約は [CLAUDE.md](CLAUDE.md#testing) を参照してください。

## コントリビューション

PR歓迎 — バグ修正、新しいモデル統合、ドキュメント改善。fork、フィーチャーブランチ作成、`make build && make test`、`main` に対してPRを開いてください。

## ライセンス

Apache 2.0
