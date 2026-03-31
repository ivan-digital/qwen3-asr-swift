# Kokoro TTS Architecture

Kokoro-82M is a non-autoregressive text-to-speech model based on [StyleTTS 2](https://github.com/yl4579/StyleTTS2) with an ISTFTNet vocoder. It runs on the Neural Engine via a 3-stage CoreML pipeline, producing 24 kHz speech.

## Overview

- **Parameters**: 82M
- **Backend**: CoreML (Neural Engine)
- **Output**: 24 kHz mono Float32 PCM
- **Inference**: Non-autoregressive, 3-stage pipeline
- **Voices**: 54 presets across 10 languages
- **License**: Apache-2.0

## Pipeline

```
Text → Phonemizer → Duration Model → Alignment → Prosody Model → Decoder → 24kHz Audio
         ↓              ↓                              ↓             ↓
   [Dictionary]    [Style Embedding]             [Style Embed]  [Style Embed]
   [G2P BART]         [Speed]
```

1. **Phonemizer** converts text to phoneme token IDs via dictionary lookup, suffix stemming, or neural G2P fallback
2. **Duration model** predicts per-phoneme durations + intermediate features
3. **Alignment** (Swift-side) builds a frame-to-phoneme matrix from predicted durations
4. **Prosody model** predicts F0 (pitch) and noise features from aligned prosody features
5. **Decoder** synthesizes the audio waveform from aligned text features + prosody

## Model I/O

### Stage 1: Duration Model

**Inputs:**

| Name | Shape | Type | Description |
|------|-------|------|-------------|
| `input_ids` | [1, N] | Int32 | Phoneme token IDs (N = phoneme bucket: 16, 32, 64, or 128) |
| `attention_mask` | [1, N] | Int32 | 1 for real tokens, 0 for padding |
| `ref_s` | [1, 256] | Float32 | Voice style embedding |
| `speed` | [1] | Float32 | Speed multiplier (1.0 = normal) |

**Outputs:**

| Name | Shape | Type | Description |
|------|-------|------|-------------|
| `pred_dur` | [1, N] | Float16 | Predicted phoneme durations (frames per phoneme) |
| `d_transposed` | [1, 640, N] | Float16 | Prosody features for alignment |
| `t_en` | [1, 512, N] | Float16 | Text encoding for alignment |

### Stage 2: Prosody Model

**Inputs:**

| Name | Shape | Type | Description |
|------|-------|------|-------------|
| `en` | [1, 640, F] | Float32 | Aligned prosody features (F = decoder bucket frames) |
| `s` | [1, 128] | Float32 | Style embedding (second half of ref_s) |

**Outputs:**

| Name | Shape | Type | Description |
|------|-------|------|-------------|
| `F0_pred` | [1, F*2] | Float16 | Predicted pitch contour |
| `N_pred` | [1, F*2] | Float16 | Predicted noise features |

### Stage 3: Decoder

**Inputs:**

| Name | Shape | Type | Description |
|------|-------|------|-------------|
| `asr` | [1, 512, F] | Float32 | Aligned text features |
| `F0_pred` | [1, F*2] | Float32 | Pitch contour |
| `N_pred` | [1, F*2] | Float32 | Noise features |
| `ref_s` | [1, 128] | Float32 | Style embedding (first half) |

**Outputs:**

| Name | Shape | Type | Description |
|------|-------|------|-------------|
| `audio` | [1, 1, F*600] | Float16 | 24 kHz waveform |

### Swift-Side Alignment

Between stages 1 and 2, Swift builds an alignment matrix from predicted durations:

```
pred_aln_trg[phoneme_idx, frame_idx] = 1.0  // for each phoneme's allocated frames
en  = d_transposed @ pred_aln_trg   // [640, F] aligned prosody features
asr = t_en @ pred_aln_trg           // [512, F] aligned text features
```

## Model Buckets

### Phoneme Buckets (Duration Model)

The duration model uses enumerated input shapes to minimize backward LSTM contamination from padding. Input is padded to the smallest bucket that fits:

| Bucket | Max Phonemes | Use Case |
|--------|-------------|----------|
| p16 | 16 | Short phrases (1-2 words) |
| p32 | 32 | Short sentences |
| p64 | 64 | Medium sentences |
| p128 | 128 | Long sentences |

### Decoder Buckets

Fixed-shape decoder models for different maximum output lengths:

| Bucket | Max Frames | Max Audio | Samples |
|--------|-----------|-----------|---------|
| `decoder_5s` | 200 | 5.0s | 120,000 |
| `decoder_10s` | 400 | 10.0s | 240,000 |
| `decoder_15s` | 600 | 15.0s | 360,000 |

Each frame produces 600 audio samples at 24 kHz (25ms per frame).

## Phonemizer

Three-tier pipeline, all Apache-2.0 licensed (no GPL dependencies):

1. **Dictionary lookup** — US and British English pronunciation dictionaries with heteronym support (POS-based disambiguation)
2. **Suffix stemming** — Morphological decomposition for known suffixes
3. **BART G2P** — Neural grapheme-to-phoneme fallback using a separate CoreML encoder-decoder model for OOV words

## Voice Embeddings

Each voice is a 256-dimensional Float32 vector stored in a per-voice JSON file. The embedding captures speaker identity and style characteristics. The first 128 dimensions are used by the decoder, the second 128 by the prosody model.

Naming convention: `[language][gender]_[name]`
- `a` = American English, `b` = British English
- `e` = Spanish, `f` = French, `h` = Hindi, `i` = Italian
- `j` = Japanese, `k` = Korean, `p` = Portuguese, `z` = Chinese
- `f` = female, `m` = male

## Weight Files

| Component | Size | Format |
|-----------|------|--------|
| Duration model | ~39 MB | .mlmodelc |
| Prosody model | ~17 MB | .mlmodelc |
| Decoder models (3 buckets) | ~107 MB each | .mlmodelc |
| Voice embeddings (54 voices) | ~0.3 MB | JSON |
| G2P encoder + decoder | ~1.5 MB | .mlmodelc |
| Dictionaries + vocab | ~6 MB | JSON |
| **Total (1 decoder)** | **~170 MB** | |

Weights: [aufklarer/Kokoro-82M-CoreML](https://huggingface.co/aufklarer/Kokoro-82M-CoreML)

## Performance (M2 Max)

| Metric | Value |
|--------|-------|
| Inference RTFx | ~0.7 (faster than real-time) |
| Weight memory | ~170 MB (1 decoder bucket) |
| Backend | CoreML (Neural Engine) |

Non-autoregressive — no sampling loop, latency scales linearly with output length but remains faster than real-time.

## Conversion

```bash
python scripts/convert_kokoro_coreml.py --output /tmp/kokoro-coreml
python scripts/convert_kokoro_coreml.py --output /tmp/kokoro-coreml --quantize int8
```

## Source Files

```
Sources/KokoroTTS/
  Configuration.swift      Model config, phoneme/decoder bucket selection
  KokoroModel.swift        3-stage CoreML model loading and inference
  KokoroTTS.swift          High-level API (fromPretrained, synthesize, alignment)
  Phonemizer.swift         Text → phoneme tokenization
  KokoroTTS+Protocols.swift Protocol conformance
  KokoroTTS+Memory.swift   Memory reporting
```
