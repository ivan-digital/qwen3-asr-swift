# Cache Directory & Offline Mode

All `fromPretrained()` methods accept optional `cacheDir` and `offlineMode` parameters for apps that need control over model storage or want to avoid network calls.

## Environment Variables

### QWEN3_CACHE_DIR
Override the default cache directory location:
```bash
export QWEN3_CACHE_DIR=/path/to/custom/cache
```

Note: When set, models will be cached directly in this directory (e.g., `/path/to/custom/cache/models/...`), not in a nested `qwen3-speech` subdirectory.

### QWEN3_MODEL_SOURCE
Choose between HuggingFace and ModelScope for downloading models:
```bash
export QWEN3_MODEL_SOURCE=modelscope  # Use ModelScope (modelscope.cn)
export QWEN3_MODEL_SOURCE=huggingface # Use HuggingFace (default)
```

## Custom Cache Directory

By default, models are cached in `~/Library/Caches/qwen3-speech/models/<org>/<model>/`. Pass `cacheDir` to override:

```swift
let appModels = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
    .appendingPathComponent("MyApp/models")

let asr = try await ParakeetASRModel.fromPretrained(
    cacheDir: appModels.appendingPathComponent("parakeet"))

let tts = try await KokoroTTSModel.fromPretrained(
    cacheDir: appModels.appendingPathComponent("kokoro"))
```

This is useful for:
- **Sandboxed macOS apps** that can't write to `~/Library/Caches/`
- **iOS apps** using the app container
- **Custom storage** (external drive, shared group container)

### Diarization Pipeline

The diarization pipeline downloads 3 models (segmentation, speaker embedding, optional VAD). Use `cacheBaseDir` to set a shared base — each sub-model gets its own subdirectory automatically:

```swift
let pipeline = try await PyannoteDiarizationPipeline.fromPretrained(
    cacheBaseDir: appModels)
// Segmentation → appModels/models/aufklarer/Pyannote-Segmentation-MLX/
// Embedding    → appModels/models/aufklarer/WeSpeaker-ResNet34-LM-MLX/
// VAD (opt.)   → appModels/models/aufklarer/Silero-VAD-v5-MLX/
```

## Offline Mode

When `offlineMode: true`, the downloader skips network requests if weights already exist on disk:

```swift
let model = try await Qwen3ASRModel.fromPretrained(offlineMode: true)
```

Behavior:
- Weights exist → returns immediately (no HuggingFace API calls)
- Weights missing → falls through to normal download (will fail if truly offline)

This avoids unnecessary network latency on app launch when models are already cached.

### Combining Both

```swift
let model = try await ParakeetASRModel.fromPretrained(
    cacheDir: bundledModelsDir,
    offlineMode: true)
```

Ship pre-downloaded models in your app bundle, point `cacheDir` at them, and set `offlineMode: true` to guarantee zero network calls.

## ModelScope Support

Models can be downloaded from ModelScope (modelscope.cn) instead of HuggingFace by setting the environment variable:

```bash
export QWEN3_MODEL_SOURCE=modelscope
```

This is useful for:
- **Users in China** - Faster downloads from domestic servers
- **Corporate environments** - May have better access to ModelScope

Supported models on ModelScope:
- Qwen3-ASR models (0.6B and 1.7B)
- Qwen3-TTS models
- Most other models are also available

The cache directory structure remains identical regardless of the source.

## Supported Models

All models support both parameters:

| Model | Parameter |
|-------|-----------|
| `Qwen3ASRModel` | `cacheDir`, `offlineMode` |
| `ParakeetASRModel` | `cacheDir`, `offlineMode` |
| `CoreMLASRModel` | `cacheDir`, `offlineMode` |
| `KokoroTTSModel` | `cacheDir`, `offlineMode` |
| `Qwen3TTSModel` | `cacheDir`, `offlineMode` |
| `Qwen3TTSCoreMLModel` | `cacheDir`, `offlineMode` |
| `CosyVoiceTTSModel` | `cacheDir`, `offlineMode` |
| `PersonaPlexModel` | `cacheDir`, `offlineMode` |
| `SileroVADModel` | `cacheDir`, `offlineMode` |
| `PyannoteVADModel` | `cacheDir`, `offlineMode` |
| `FireRedVADModel` | `cacheDir`, `offlineMode` |
| `WeSpeakerModel` | `cacheDir`, `offlineMode` |
| `SpeechEnhancer` | `cacheDir`, `offlineMode` |
| `SortformerDiarizer` | `cacheDir`, `offlineMode` |
| `PyannoteDiarizationPipeline` | `cacheBaseDir`, `offlineMode` |
| `Qwen35CoreMLChat` | `cacheDir`, `offlineMode` |
| `Qwen35MLXChat` | `cacheDir`, `offlineMode` |
