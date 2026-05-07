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
| Text decoder | 28 layers, hidden=1024, 16Q/8KV heads, headDim=128 (4-bit, 8-bit, or bf16) |
| Classify head | Linear(1024, 5000), float16 (NOT tied to embeddings) |
| Timestamp resolution | 80ms per class (5000 classes = 400s addressable, ~270s reliable in practice — see *Long-audio handling* below) |

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

Language dispatch:

| Language | Tokenizer |
|---|---|
| Japanese | `NLTokenizer(.japanese)` — morpheme-level |
| Korean | `NLTokenizer(.korean)` — word-level |
| Thai / Lao / Khmer / Burmese / Tibetan | `NLTokenizer(<lang>)` — native segmentation for scripts without word-level whitespace |
| Everything else | whitespace split + per-Han ideograph break |

A universal token filter keeps Unicode Letters (`L*`), Numbers (`N*`), combining Marks (`Mn`/`Mc`/`Me`), and the ASCII apostrophe — punctuation, symbols, and separators are stripped. Combining marks must be preserved so scripts like Devanagari, Thai, Bengali, and Tibetan keep their vowel and tone marks intact (e.g. `नमस्ते`, `สวัสดี`). The Han-ideograph break covers CJK Unified `0x4E00–0x9FFF`, Extensions A–E, and Compatibility `0xF900–0xFAFF`; hiragana, katakana, and Hangul are deliberately **not** broken per character — those scripts are handled by the language-specific tokenizers.

**English / European / Hindi / Arabic / etc. (default path):**
```
"Can you guarantee" → ["Can", "you", "guarantee"]
"Hello, world!"     → ["Hello", "world"]    # punctuation stripped
"don't stop"        → ["don't", "stop"]     # apostrophe kept
```

**Chinese (default path, no whitespace → per-Han split):**
```
"你好世界"           → ["你", "好", "世", "界"]
"Hello你好world"     → ["Hello", "你", "好", "world"]
```

**Japanese (NLTokenizer, morpheme-level):**
```
"今日はいい天気ですね"     → ["今日", "は", "いい", "天気", "です", "ね"]
"コンピュータ"             → ["コンピュータ"]   # 1 token, NOT 6 per-char
"こんにちは"               → ["こんにちは"]     # 1 token, NOT 5
"iPhoneを使います"         → ["iPhone", "を", "使い", "ます"]
```

**Korean (NLTokenizer, word-level):**
```
"안녕하세요 반갑습니다" → ["안녕하세요", "반갑습니다"]   # 2 word tokens, NOT 11 per-syllable
```

#### Why NLTokenizer

`NLTokenizer` is built into the OS — zero extra binary size, on-device, and produces morpheme-level Japanese and word-level Korean output suitable for the aligner. The model's timestamp head only cares about morpheme **boundaries**, not exact morpheme identity, so small differences (e.g. NLTokenizer splits `"5G"` → `"5", "G"`) don't affect timestamp quality.

Each token gets `<timestamp>` pairs:
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

## Long-audio handling

The classify head addresses 400 s in principle (5000 classes × 80 ms), but the trained reliable range on `Qwen3-ForcedAligner-0.6B` is **~270 s**. Past that, the model emits low / non-monotonic indices for the remaining words, and the LIS pass collapses them onto the last anchor — every trailing word ends up with the same timestamp.

`Qwen3ForcedAligner.alignLong(...)` wraps `align()` to handle this:

1. Run `align()` on the full audio.
2. Detect a trailing **plateau** (≥ 5 consecutive words whose start times differ by < 0.1 s) — the LIS-clamp signature.
3. If found: keep the reliable prefix, slice the remaining audio, re-align the remaining words with a time offset.
4. Iterate until no plateau remains or the remainder is below 5 s.

For audio under the 240 s bypass threshold this is a one-pass passthrough into `align()`. For longer audio the cost is one extra `align` pass per chunk; in practice the second chunk is short so the overhead is small (≤ ~0.5 s wall-clock on the 306 s TED-Ed test clip).

The CLI (`audio align`) calls `alignLong` automatically and prints a one-line message when chunking kicks in:

```
Audio 306.2s saturated after word 690 (272.6s); chunking remaining 33.6s (pass 2)
```

Set `ALIGN_DEBUG=1` to dump raw vs corrected timestamp indices when investigating misaligned outputs.

### Known limitations

- **Audio with leading non-speech** (music intro, long silence): the model often stamps the first word at `~0 s` because the classifier head has no notion of "speech hasn't started yet". Workaround: trim leading non-speech before alignment, or pre-pass with a VAD.
- **Single-language audio assumed**: the language hint applies to the whole alignment; no per-segment language switching.

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
| MLX 4-bit | `aufklarer/Qwen3-ForcedAligner-0.6B-4bit` | ~979 MB |
| MLX 8-bit | `aufklarer/Qwen3-ForcedAligner-0.6B-8bit` | ~1.3 GB |
| MLX bf16 | `aufklarer/Qwen3-ForcedAligner-0.6B-bf16` | ~1.8 GB |
| CoreML INT4 | `aufklarer/Qwen3-ForcedAligner-0.6B-CoreML-INT4` | ~662 MB |
| CoreML INT8 | `aufklarer/Qwen3-ForcedAligner-0.6B-CoreML-INT8` | ~1.1 GB |

Variant is auto-detected from `quantize_config.json` in the model directory. The bf16 variant uses a float text decoder (`FloatTextModel`) instead of a quantized one.

## CLI Usage

```bash
# Align with provided text
audio align audio.wav --text "Can you guarantee that the replacement part will be shipped tomorrow?"

# Transcribe first, then align
audio align audio.wav

# Custom aligner model (8-bit or bf16)
audio align audio.wav --aligner-model aufklarer/Qwen3-ForcedAligner-0.6B-8bit
audio align audio.wav --aligner-model aufklarer/Qwen3-ForcedAligner-0.6B-bf16
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

// Short-form audio (≤240s): single forward pass.
let aligned = aligner.align(
    audio: audioSamples,
    text: "Can you guarantee that the replacement part will be shipped tomorrow?",
    sampleRate: 24000
)

// Long-form audio: auto-chunks past the model's reliable range so trailing
// words don't all collapse onto the last anchor's timestamp. Behaves identically
// to `align()` for short audio.
let alignedLong = aligner.alignLong(
    audio: longAudioSamples,
    text: longText,
    sampleRate: 24000,
    progressHandler: { print($0) }   // optional: notified when a chunk hand-off happens
)

for word in alignedLong {
    print("[\(word.startTime)s - \(word.endTime)s] \(word.text)")
}
```

## Conversion

```bash
# MLX variants
python scripts/convert_forced_aligner.py --bits 4 --upload --repo-id aufklarer/Qwen3-ForcedAligner-0.6B-4bit
python scripts/convert_forced_aligner.py --bits 8 --upload --repo-id aufklarer/Qwen3-ForcedAligner-0.6B-8bit
python scripts/convert_forced_aligner.py --bits 0 --upload --repo-id aufklarer/Qwen3-ForcedAligner-0.6B-bf16

# CoreML variants
python scripts/convert_forced_aligner.py --coreml --coreml-bits 4 --upload --repo-id aufklarer/Qwen3-ForcedAligner-0.6B-CoreML-INT4
python scripts/convert_forced_aligner.py --coreml --coreml-bits 8 --upload --repo-id aufklarer/Qwen3-ForcedAligner-0.6B-CoreML-INT8
```

MLX: quantizes text decoder (attention + MLP + embeddings) to N-bit. Audio encoder and classify head kept as float16. `--bits 0` keeps everything as float16.

CoreML: traces audio encoder, text decoder + classify head, and embedding as separate models with INT4/INT8 palettization. Published as pre-compiled `.mlmodelc` bundles.
