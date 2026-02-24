# Forced Aligner ([Qwen3-ForcedAligner-0.6B](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B))

## Overview

Qwen3-ForcedAligner predicts word-level timestamps for audio+text pairs. It shares the same encoder-decoder architecture as Qwen3-ASR but replaces the vocabulary lm_head with a 5000-class timestamp classification head. Inference is non-autoregressive (single forward pass through the decoder).

```
Audio (16kHz) + Text
    |            |
    v            v
+------------------+   +---------------------+
|  Mel → Audio     |   |  Word splitting     |
|  Encoder (24L)   |   |  + timestamp slots  |
+--------+---------+   +---------+-----------+
         |                        |
         v                        v
+------------------------------------------------+
|  Text Decoder (28L, single forward pass)       |
|  Audio embeds injected at <audio_pad> positions |
|  Timestamp tokens at word boundaries            |
+-----------------------+------------------------+
                        |
                        v
+------------------------------------------------+
|  Classify Head (Linear 1024 → 5000)            |
|  argmax at <timestamp> positions                |
+-----------------------+------------------------+
                        |
                        v
+------------------------------------------------+
|  LIS Monotonicity Correction                   |
|  Index × 80ms → timestamps in seconds          |
+-----------------------+------------------------+
                        |
                        v
               [AlignedWord] array
        (word, startTime, endTime)
```

## Architecture

| Component | Config |
|-----------|--------|
| Audio encoder | 24 layers, d_model=1024, 16 heads, FFN=4096, output→1024 |
| Text decoder | 28 layers, hidden=1024, 16Q/8KV heads, headDim=128, 4-bit quantized |
| Classify head | Linear(1024, 5000), float16 (NOT tied to embeddings) |
| Timestamp resolution | 80ms per class (5000 classes = 400s max) |

## Key Difference from ASR

| | ASR | Forced Aligner |
|---|-----|----------------|
| Decoder mode | Autoregressive (token by token) | Non-autoregressive (single pass) |
| Output head | Tied embedding lm_head (vocab 151936) | Classify head (5000 timestamp classes) |
| KV cache | Yes (grows with each token) | None |
| Input | Audio only | Audio + text with `<timestamp>` slots |
| Audio encoder | 18L/896D (0.6B) | 24L/1024D (larger) |

## Inference Pipeline

### 1. Audio Encoding
Same as ASR: mel spectrogram → chunked Conv2D → transformer → projector.

### 2. Text Preprocessing (TextPreprocessing.swift)

Text is split into words (language-specific) and `<timestamp>` tokens inserted:

**English:** Split on whitespace
```
"Can you guarantee" → ["Can", "you", "guarantee"]
```

**CJK:** Character-level splitting
```
"你好世界" → ["你", "好", "世", "界"]
```

Each word gets `<timestamp>` pairs:
```
<ts>Can<ts> <ts>you<ts> <ts>guarantee<ts>
```

### 3. Single Forward Pass

Build the full sequence with chat template:
```
<|im_start|>system\n<|im_end|>\n
<|im_start|>user\n<|audio_start|>[audio_pad × N]<|audio_end|><|im_end|>\n
<|im_start|>assistant\n
<ts>word1_tokens<ts> <ts>word2_tokens<ts> ...
```

One forward pass through the decoder (no cache, no loop). Apply classify head to all hidden states → logits `[1, seqLen, 5000]`.

### 4. Timestamp Extraction

1. Extract logits only at `<timestamp>` positions
2. argmax → raw timestamp class indices
3. Multiply by 80ms → raw timestamps in seconds
4. Pair consecutive timestamps as (start, end) per word

### 5. LIS Monotonicity Correction (TimestampCorrection.swift)

Raw timestamps may not be monotonic. Fix via:
1. Find Longest Increasing Subsequence (O(n log n))
2. Small gaps (≤2 positions): nearest-neighbor correction
3. Larger gaps: linear interpolation between LIS anchors
4. Final pass: enforce non-decreasing order

## Performance (M2 Max, 64 GB)

| Stage | Time | Notes |
|-------|------|-------|
| Audio encoder | ~328ms | Mel extraction + 24L transformer + projector |
| Decoder + classify | ~37ms | Single forward pass, no autoregressive loop |
| **Total (20s audio)** | **~365ms** | **RTF ~0.018 (55x faster than real-time)** |

Debug build. Release would be faster.

## Weight Structure

Weights use a `thinker.` prefix:

| Key pattern | Component |
|-------------|-----------|
| `thinker.audio_tower.*` | Audio encoder (float16) |
| `thinker.model.*` | Text decoder (4-bit quantized) |
| `thinker.lm_head.weight` | Classify head (float16, NOT quantized) |

## Model Files

| Model | ID | Size |
|-------|----|------|
| 4-bit quantized | `aufklarer/Qwen3-ForcedAligner-0.6B-4bit` | ~979 MB |
| bf16 (original) | `mlx-community/Qwen3-ForcedAligner-0.6B-bf16` | ~1.84 GB |

## CLI Usage

```bash
# Align with provided text
audio align audio.wav --text "Can you guarantee that the replacement part will be shipped tomorrow?"

# Transcribe first, then align
audio align audio.wav

# Custom aligner model
audio align audio.wav --aligner-model mlx-community/Qwen3-ForcedAligner-0.6B-bf16
```

Output format:
```
[0.12s - 0.45s] Can
[0.45s - 0.72s] you
[0.72s - 1.20s] guarantee
...
```

## Swift API

```swift
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

## Conversion

```bash
python scripts/convert_forced_aligner.py \
    --source Qwen/Qwen3-ForcedAligner-0.6B \
    --output-dir ./forced-aligner-4bit \
    --upload --repo-id aufklarer/Qwen3-ForcedAligner-0.6B-4bit
```

Quantizes text decoder (attention + MLP + embeddings) to 4-bit. Audio encoder and classify head kept as float16.
