# Kokoro TTS Architecture

Kokoro-82M is a non-autoregressive text-to-speech model based on [StyleTTS 2](  1) with an ISTFTNet vocoder. It runs entirely on the Neural Engine via CoreML, producing 24 kHz speech in a single forward pass.

## Overview

- **Parameters**: 82M
- **Backend**: CoreML (Neural Engine)
- **Output**: 24 kHz mono Float32 PCM
- **Inference**: Non-autoregressive (single forward pass, ~45ms constant latency)
- **Voices**: 50 presets across 10 languages
- **License**: Apache-2.0

## Pipeline

```
Text → Phonemizer → CoreML Model → 24kHz Audio
         ↓              ↓
   [Dictionary]    [Style Embedding]
   [G2P BART]     [Random Phases]
```

1. **Phonemizer** converts text to phoneme token IDs via dictionary lookup, suffix stemming, or neural G2P fallback
2. **CoreML model** takes tokens + voice embedding + random phases → outputs audio waveform directly

## Model I/O

**Inputs:**

| Name | Shape | Type | Description |
|------|-------|------|-------------|
| `input_ids` | [1, N] | Int32 | Phoneme token IDs |
| `attention_mask` | [1, N] | Int32 | 1 for real tokens, 0 for padding |
| `ref_s` | [1, 256] | Float32 | Voice style embedding |
| `random_phases` | [1, 9] | Float32 | Random phases for ISTFTNet vocoder |

**Outputs:**

| Name | Shape | Type | Description |
|------|-------|------|-------------|
| `audio` | [1, 1, S] | Float32 | 24 kHz waveform |
| `audio_length_samples` | [1] | Int32 | Valid sample count |
| `pred_dur` | [1, N] | Float32 | Predicted phoneme durations |

## Model Variants

Five pre-compiled CoreML buckets handle different output lengths:

| Variant | Max Tokens | Max Audio | Target |
|---------|-----------|-----------|--------|
| `kokoro_24_10s` | 242 | 10.0s | iOS 17+ / macOS 14+ |
| `kokoro_24_15s` | 242 | 15.0s | iOS 17+ / macOS 14+ |
| `kokoro_21_5s` | 124 | 7.3s | iOS 16+ / macOS 13+ |
| `kokoro_21_10s` | 168 | 10.6s | iOS 16+ / macOS 13+ |
| `kokoro_21_15s` | 249 | 15.5s | iOS 16+ / macOS 13+ |

The runtime selects the smallest bucket that fits the input token count. v2.4 models use newer CoreML operations (iOS 17+); v2.1 models provide backward compatibility.

## Phonemizer

Three-tier pipeline, all Apache-2.0 licensed (no GPL dependencies):

1. **Dictionary lookup** — US and British English pronunciation dictionaries with heteronym support (POS-based disambiguation)
2. **Suffix stemming** — Morphological decomposition for known suffixes
3. **BART G2P** — Neural grapheme-to-phoneme fallback using a separate CoreML encoder-decoder model for OOV words

## Voice Embeddings

Each voice is a 256-dimensional Float32 vector stored in a per-voice JSON file. The embedding captures speaker identity and style characteristics.

Naming convention: `[language][gender]_[name]`
- `a` = American English, `b` = British English
- `e` = Spanish, `f` = French, `h` = Hindi, `i` = Italian
- `j` = Japanese, `k` = Korean, `p` = Portuguese, `z` = Chinese
- `f` = female, `m` = male

## Weight Files

| Component | Size | Format |
|-----------|------|--------|
| CoreML models (5 variants) | ~280 MB | .mlmodelc |
| Voice embeddings (50 voices) | ~1 MB | JSON |
| G2P encoder + decoder | ~40 MB | .mlmodelc |
| Dictionaries + vocab | ~4 MB | JSON |
| **Total** | **~325 MB** | |

Weights: [aufklarer/Kokoro-82M-CoreML](https://huggingface.co/aufklarer/Kokoro-82M-CoreML)

## Performance (M2 Max)

| Metric | Value |
|--------|-------|
| Inference latency | ~45ms (constant) |
| Weight memory | 325 MB |
| Peak inference memory | ~500 MB |
| Backend | CoreML (Neural Engine) |

Non-autoregressive — latency is constant regardless of output length, unlike autoregressive models (Qwen3-TTS, CosyVoice3) where latency scales with audio duration.

## Source Files

```
Sources/KokoroTTS/
  Configuration.swift      Model config, bucket selection
  KokoroModel.swift        CoreML model loading and inference
  KokoroTTS.swift          High-level API (fromPretrained, synthesize)
  Phonemizer.swift         Text → phoneme tokenization
  KokoroTTS+Protocols.swift Protocol conformance
  KokoroTTS+Memory.swift   Memory reporting
```
