# Agent Instructions

AI speech models for Apple Silicon (MLX Swift). ASR, TTS, speech-to-speech, VAD, diarization, speech enhancement.

## Workflow

- **Never commit, push, or comment on GitHub without explicit user confirmation.** Draft first, ask to confirm, then execute.
- **Every README.md change must update all 9 translations** (`README_zh.md`, `README_ja.md`, `README_ko.md`, `README_es.md`, `README_de.md`, `README_fr.md`, `README_hi.md`, `README_pt.md`, `README_ru.md`). No exceptions.

## Git Conventions

- Never mention Claude, AI, or any AI tool in commit messages, PR descriptions, or co-author tags
- No `Co-Authored-By` lines in commits
- **Never amend commits or force push** unless the user explicitly asks for it
- Always use branches and PRs — commit history must be preserved

## Build

```bash
# Release build (recommended)
make build

# Debug build
make debug

# Run tests (builds debug first)
make test

# Clean
make clean
```

Or manually:

```bash
swift build -c release --disable-sandbox
./scripts/build_mlx_metallib.sh release
```

The metallib step compiles MLX Metal shaders — without it, inference runs ~5x slower due to JIT shader compilation.

## Skills (Slash Commands)

Project skills in `.claude/skills/`:

| Command | Description |
|---------|-------------|
| `/build` or `/build release` | Release build with metallib |
| `/build debug` | Debug build |
| `/test` or `/test unit` | Run unit tests (skip E2E) |
| `/test e2e` | Full test suite with model downloads |
| `/test FilterName` | Run specific test filter |
| `/benchmark asr` | Benchmark ASR speed |
| `/benchmark tts` | Benchmark TTS speed |
| `/benchmark vad` | VAD benchmark on VoxConverse |
| `/benchmark diarize` | DER benchmark on VoxConverse |

## Project Structure

- `Sources/Qwen3ASR/` — Speech-to-text (Qwen3-ASR)
- `Sources/ParakeetASR/` — Speech-to-text (Parakeet TDT, CoreML)
- `Sources/Qwen3TTS/` — Text-to-speech (Qwen3-TTS)
- `Sources/CosyVoiceTTS/` — Text-to-speech (CosyVoice3, streaming)
- `Sources/KokoroTTS/` — Text-to-speech (Kokoro-82M, CoreML, iOS-ready)
- `Sources/PersonaPlex/` — Speech-to-speech (PersonaPlex 7B, full-duplex)
- `Sources/SpeechVAD/` — VAD (Silero + Pyannote), speaker diarization, speaker embedding (WeSpeaker)
- `Sources/SpeechEnhancement/` — Noise suppression (DeepFilterNet3, CoreML)
- `Sources/Qwen3Chat/` — On-device LLM chat (Qwen3.5-0.8B, MLX + CoreML, INT4/INT8)
- `Sources/MLXCommon/` — Shared MLX utilities (weight loading, quantized layers, memory estimation)
- `Sources/AudioCommon/` — Audio I/O, protocols, HuggingFace downloader
- `Sources/AudioCLILib/` — CLI commands
- `Sources/AudioCLI/` — CLI entry point (`audio` binary)
- `Tests/` — Unit and integration tests
- `scripts/` — Model conversion (PyTorch → MLX/CoreML), benchmarking
- `Examples/` — Demo apps (PersonaPlexDemo, SpeechDemo)

## Key Conventions

- Swift 6, macOS 14+, Apple Silicon (M-series)
- MLX for GPU inference (Metal), CoreML for Neural Engine (DeepFilterNet3, Kokoro, Silero VAD optional)
- Models are downloaded from HuggingFace on first use, cached in `~/Library/Caches/qwen3-speech/`
- All audio processing uses Float32 PCM, resampled to model-specific rates internally
- `DiarizedSegment`, `SpeechSegment`, protocol types defined in `Sources/AudioCommon/Protocols.swift`
- Tests that use MLX arrays require the compiled metallib; config/logic-only tests work without it

## Testing

Safe tests (no GPU/model download required):
```bash
make test
```

Full test suite (requires metallib + model downloads):
```bash
make test
```

### Testing requirements for new code

**Every new feature, model, or module MUST include tests:**

- **Unit tests**: Config parsing, data structures, weight loading, math/DSP logic — no GPU or model downloads needed
- **E2E tests**: Full pipeline with real model weights — verify correct output (e.g., ASR round-trip, correct transcription text)
- **Regression tests**: When fixing bugs, add a test that would have caught the bug

**Test organization**: Place tests in `Tests/<ModuleName>Tests/`. Follow existing patterns (e.g., `Qwen3ASRTests/`, `SpeechVADTests/`).

**E2E test naming**: Prefix E2E test classes with `E2E` (e.g., `E2ETranscriptionTests`, `E2EDiarizationTests`). CI uses `--skip E2E` regex to filter out all E2E tests that require model downloads — only unit tests run in the pipeline. E2E tests run locally with `make test` (full suite). **CRITICAL**: Any test class that downloads models or requires GPU inference MUST be prefixed with `E2E`. Unit test classes must NOT contain `E2E` in their name.

**What to test per category:**
| Change | Required tests |
|--------|---------------|
| New model/module | Unit (config, weight loading) + E2E (inference produces correct output) |
| New CLI command | Unit (argument parsing) + E2E (end-to-end with real files) |
| Bug fix | Regression test reproducing the bug |
| New protocol/type | Unit test for conformance and behavior |
| DSP/audio processing | Unit test with known input/output pairs |

## CLI

The `audio` binary is the main entry point:

```bash
.build/release/audio transcribe recording.wav          # ASR
.build/release/audio speak "Hello" --output hi.wav     # TTS
.build/release/audio respond --input q.wav             # Speech-to-speech
.build/release/audio diarize meeting.wav               # Speaker diarization (pyannote)
.build/release/audio diarize meeting.wav --engine sortformer  # Sortformer (CoreML, end-to-end)
.build/release/audio diarize meeting.wav --rttm        # RTTM output
.build/release/audio vad audio.wav                     # Voice activity detection
.build/release/audio embed-speaker voice.wav           # Speaker embedding
.build/release/audio denoise noisy.wav                 # Speech enhancement
.build/release/audio kokoro "Hello" --voice af_heart   # Kokoro TTS (iOS)
```

## Documentation

### Local docs (`docs/`)

Architecture and implementation docs live in this repo:

```
docs/
  models/                       Model architecture, weights, layers
    asr-model.md                Qwen3-ASR architecture
    tts-model.md                Qwen3-TTS architecture
    cosyvoice-tts.md            CosyVoice3 architecture
    kokoro-tts.md               Kokoro-82M architecture
    parakeet-asr.md             Parakeet TDT architecture
    personaplex.md              PersonaPlex architecture
    fireredvad.md               FireRedVAD (DFSMN) architecture
  inference/                    Pipelines, usage, configs
    qwen3-asr-inference.md      Qwen3-ASR inference pipeline
    parakeet-asr-inference.md   Parakeet TDT inference (CoreML)
    qwen3-tts-inference.md      TTS inference pipeline
    forced-aligner.md           Forced alignment pipeline
    silero-vad.md               Silero VAD streaming
    fireredvad.md               FireRedVAD inference + tuning results
    speaker-diarization.md      Speaker diarization pipeline
    speech-enhancement.md       DeepFilterNet3 pipeline
  audio/                        Audio I/O, playback, voice pipeline
    playback.md                 Streaming playback, pre-buffer, Apple audio architecture
    voice-pipeline.md           VoicePipeline state machine, events, config
  benchmarks/                   WER, DER, RTF results
  shared-protocols.md       Protocol reference (cross-cutting)
```

**Keep local docs in sync when making code changes.**

### Documentation site (soniqo-web)

The public documentation is hosted at **https://soniqo.audio** (Firebase Hosting) and lives in a separate private repository: **soniqo-web**.

**Whenever code changes are made in this repo, both local docs AND the soniqo-web site must be updated.**

### What requires a docs update

- New features or capabilities added
- CLI commands added, removed, or flags changed
- Public API changes (protocols, types, function signatures)
- New models or model variants added
- Performance characteristics changed
- Build requirements or installation steps changed
- New modules or source structure changes

### Documentation site structure

```
soniqo-web/public/
  index.html                Landing page (feature grid, performance stats)
  getting-started/          Installation, build instructions, quick start
  guides/
    transcribe/             Qwen3-ASR guide
    parakeet/               Parakeet TDT guide
    speak/                  Qwen3-TTS guide
    cosyvoice/              CosyVoice3 guide
    voice-cloning/          Voice cloning guide
    respond/                PersonaPlex guide
    vad/                    VAD guide (Pyannote + Silero)
    diarize/                Speaker diarization guide
    embed-speaker/          Speaker embeddings guide
    denoise/                Speech enhancement guide
    align/                  Forced alignment guide
    kokoro/                 Kokoro-82M guide (iOS TTS)
  cli/                      CLI command reference (all flags/options)
  api/                      Protocols and shared types
  architecture/             Module structure, backends, weight formats
```

### README translations

Translated READMEs live in the repo root: `README_zh.md`, `README_ja.md`, `README_ko.md`, `README_es.md`, `README_de.md`, `README_fr.md`, `README_hi.md`, `README_pt.md`, `README_ru.md`. **Whenever README.md is updated, all translations must be updated to match.** Each translation links back to the main README and lists all available languages at the top.

### Mapping: code changes → docs pages

| Code change | Local docs | soniqo-web page(s) |
|---|---|---|
| CLI flag added/changed | Relevant inference doc | `/cli/index.html` + relevant guide |
| New model/module | New model + inference doc | Landing page + new guide + architecture |
| Protocol change | `shared-protocols.md` | `/api/index.html` |
| Performance improvement | `benchmarks/` | Landing page perf section + relevant guide |
| Build/install change | — | `/getting-started/index.html` |
| New CLI command | Relevant inference doc | `/cli/index.html` + new guide + landing page |
