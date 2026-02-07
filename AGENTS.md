# Agents Guide (qwen3-asr-swift)

This repo is a Swift Package Manager (SwiftPM) library + CLI for running Qwen3-ASR on Apple Silicon using MLX Swift.

## Repo Layout

- `Package.swift`: SwiftPM manifest (library + executable + tests).
- `Sources/Qwen3ASR/`: Core library code (audio preprocessing, encoder, text decoder, weight loading, tokenizer).
- `Sources/Qwen3ASRCLI/main.swift`: CLI entry point.
- `Tests/Qwen3ASRTests/`: Unit tests and integration tests.
- `Tests/Qwen3ASRTests/Resources/test_audio.wav`: Small WAV used by tests.

## Prerequisites

- macOS 14+ (or iOS 17+ for the library target)
- Xcode 15+
- Apple Silicon recommended (MLX Swift)

## Common Commands

- Build:
  - `swift build`
  - `swift build -c release`
- Test:
  - `swift test`
  - `swift test --filter Qwen3ASRTests`
  - `swift test --filter Qwen3ASRIntegrationTests` (downloads model weights, slow)
- Run CLI (debug):
  - `swift run qwen3-asr-cli /path/to/audio.wav`
- Run CLI (release):
  - `swift build -c release`
  - `.build/release/qwen3-asr-cli /path/to/audio.wav`

## MLX Metal Library (metallib) Troubleshooting

If you see an error like:

- `MLX error: Failed to load the default metallib ...`

You likely need to generate `mlx.metallib` locally (some Xcode installs do not include the Metal toolchain by default):

1. Install the Metal toolchain (once):
   - `xcodebuild -downloadComponent MetalToolchain`
2. Build the metallib next to the SwiftPM output:
   - `swift build -c release --disable-sandbox`
   - `./scripts/build_mlx_metallib.sh release`

## Model Download And Cache

`Qwen3ASRModel.fromPretrained(modelId:)` downloads model artifacts from Hugging Face and caches them locally.

- Download code: `Sources/Qwen3ASR/Qwen3ASR.swift`
- Base URL pattern:
  - `https://huggingface.co/{modelId}/resolve/main/{file}`
- Cache directory (macOS default):
  - `~/Library/Caches/qwen3-asr/{sanitizedModelId}/`
- Cache override (useful for CI/sandboxes):
  - Set `QWEN3_ASR_CACHE_DIR=/path/to/cache-root` to place cache under `/path/to/cache-root/qwen3-asr/{sanitizedModelId}/`.
- Expected cached files:
  - `config.json`
  - `vocab.json`
  - `tokenizer_config.json`
  - One or more `*.safetensors` files (discovered via `model.safetensors.index.json` when present)

Do not commit downloaded weights or cache contents to the repo.

## Tests: What To Run And What To Avoid

- Prefer keeping unit tests offline and fast.
- `Qwen3ASRIntegrationTests` downloads large artifacts (hundreds of MB) and performs real model loading.
  - Only run or modify these when working on download, caching, weight loading, or end-to-end inference.
  - When adding new tests for model loading, consider adding a small offline test first (parsing, path logic, shape checks).

## Audio And Preprocessing Conventions

- Public transcription entry point: `Sources/Qwen3ASR/Qwen3ASR.swift` (`Qwen3ASRModel.transcribe`).
- Audio is expected as mono `Float` samples, typically in `[-1, 1]`.
- Feature extraction is Whisper-style and resamples internally as needed:
  - `Sources/Qwen3ASR/AudioPreprocessing.swift` (`WhisperFeatureExtractor.process`)
  - The extractor’s internal `sampleRate` is the reference rate it resamples to.
- `AudioFileLoader` helpers:
  - `Sources/Qwen3ASR/AudioFileLoader.swift` uses `AVFoundation` for general formats and includes a simple WAV parser.

If you change sample-rate assumptions, hop sizes, or mel parameters, update:

- `Sources/Qwen3ASR/AudioPreprocessing.swift`
- Any callers passing `sampleRate:` to `process` / `transcribe`
- Relevant tests in `Tests/Qwen3ASRTests/`
- `README.md` examples if they mention specific sample rates

## Weight Loading Guidelines

Weight application is keyed to Hugging Face-style tensor names and is sensitive to shape/layout.

- Weight loader: `Sources/Qwen3ASR/WeightLoading.swift`
- Audio tower weights are loaded from `audio_tower.*`
- Text decoder weights are loaded from `model.*`

When changing module structure or layer naming, keep the mapping logic consistent and add a targeted test that validates:

- The expected keys are found.
- A small subset of tensors load into the right modules without crashing.

## CLI Expectations

The current CLI is a minimal “transcribe one file” tool.

- Entry point: `Sources/Qwen3ASRCLI/main.swift`
- Usage string printed by the binary:
  - `qwen3-asr-cli <audio-file>`

If you change CLI behavior, keep the usage text accurate and update `README.md` accordingly.

## Change Hygiene

- Avoid leaving noisy `DEBUG:` prints in library code unless they are behind a flag or are strictly temporary.
- Keep API changes additive when possible.
- If you touch download/caching behavior, ensure failures are surfaced as actionable `LocalizedError`s (see `DownloadError`).
