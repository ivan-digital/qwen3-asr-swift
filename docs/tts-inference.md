# TTS Inference Pipeline (Qwen3-TTS)

> Reference for Swift MLX port. Based on Qwen3-TTS-12Hz-0.6B.

## Overview

```
Text -> [Prepare] -> [Talker] -> [Code Predictor] -> [Codec Decode] -> Audio (24kHz)
         <1%          ~55%         ~40%                ~5%
```

## Stage 1: Text Preparation

```
Input text + language tag
    |
    v
Chat template formatting:
    <|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n
    |
    v
Qwen2 BPE tokenizer -> token IDs
    |
    v
text_embedding(token_ids)  [151936, 2048]
    |
    v
Text projection MLP:  Linear(2048, 2048) -> SiLU -> Linear(2048, 1024)
    |
    v
Text hidden states [B, T_text, 1024]
```

## Stage 2: Codec Prefix

Before autoregressive generation, a 6-token codec prefix is constructed:

```
[think, think_bos, language_id, think_eos, pad, bos]
```

- `codec_think`: 2151, `codec_think_bos`: 2153, `codec_think_eos`: 2154
- `codec_pad`: 2148, `codec_bos`: 2149
- `language_id`: e.g. 2050 (English), 2052 (German), 2055 (Chinese), 2058 (Japanese)

> **Note:** Voice cloning mode adds a speaker embedding from ECAPA-TDNN. This is not yet implemented in the Swift port.

## Stage 3: Talker Generation (First Codebook)

```
Combined embeddings = [codec_prefix | text_embeddings | trailing_tokens]
    |
    v
28-layer Qwen3 transformer (autoregressive, MRoPE, GQA, KV cache)
    |
    v
codec_head (Linear -> 3072 vocab) -> logits
    |
    v
Sampling: temperature=0.9, top_k=50, top_p=1.0, repetition_penalty=1.05
    |
    v
First codebook token sequence (until codec_eos = 2150)
```

**MRoPE position tracking:** Three separate position counters (temporal, height, width) are maintained and incremented according to the token type (text vs codec).

## Stage 4: Code Predictor (Remaining 15 Codebooks)

For each generated first-codebook token:

```
Hidden state from Talker
    |
    v
For codebook_group in 2..16:
    5-layer transformer (standard RoPE)
        |
        v
    lm_head[group] -> sample token for this codebook
    Feed predicted embedding back for next codebook group
    |
    v
Result: 16 codebook tokens per time step
```

## Stage 5: Codec Decode (Speech Tokenizer)

```
16 x T codebook indices
    |
    v
SplitRVQ.decode():
    semantic_quantizer.decode(codebook_1)     -> [T, 512]
    acoustic_quantizer.decode(codebooks_2-16) -> [T, 512]
    sum embeddings                            -> [T, 512]
    |
    v
Pre-conv (CausalConv1d, k=3) -> [T, 1024]
    |
    v
Pre-transformer (8 layers, causal, RoPE, SwiGLU+LayerScale):
    input_proj: 1024 -> 512 bottleneck
    8x DecoderTransformerLayer (hidden=512)
    output_proj: 512 -> 1024                  -> [T, 1024]
    |
    v
Pre-upsample (TransposedConv1d 2x + ConvNeXt) x2 = 4x -> [4T, 1024]
    |
    v
Input conv -> [4T, 1536]
    |
    v
SEANet decoder blocks (SnakeBeta + TransposedConv1d + residual units):
    [4T, 1536] -> 8x -> 5x -> 4x -> 3x = 480x
    Total: 4 * 480 = 1920x upsample (12.5 Hz -> 24000 Hz)
    |
    v
Audio waveform [1, T*1920, 1] at 24kHz
```

## Performance (M2 Max, 64 GB)

### Release Build

| Metric | Short (1s) | Medium (3s) | Long (6s) |
|--------|-----------|-------------|------------|
| Tokens generated | 19 | 45 | 80 |
| Generate (talker + code pred) | 1.4s | 2.0s | 3.5s |
| Codec decode | 0.16s | 0.20s | 0.35s |
| **Total** | **1.6s** | **2.3s** | **3.9s** |
| **RTF** | **1.2** | **0.7** | **0.7** |
| Per-step | 73ms | 45ms | 43ms |

### Debug Build

| Metric | Short (1s) | Medium (3s) | Long (6s) |
|--------|-----------|-------------|------------|
| **Total** | **1.9s** | **4.8s** | **6.8s** |
| **RTF** | **2.1** | **1.7** | **1.7** |
| Per-step | 125ms | 120ms | 120ms |

### vs Apple AVSpeechSynthesizer (M2 Max)

| | Qwen3-TTS (release) | Apple TTS |
|---|-----------|-----------|
| RTF (long text) | ~0.7 | ~0.02 |
| Latency (6s audio) | 3.9s | 0.17s |
| Speech quality | Natural, expressive | Robotic, monotone |
| Voice cloning | Planned | No |
| Languages | EN/ZH/DE/JA | 60+ |
| On-device | Yes (MLX) | Yes (AVFoundation) |
| Model size | ~1.7 GB | Built-in |

### Optimizations Applied

1. **Chunked codec decoding** — Process codec frames in overlapping chunks (`chunkSize=25, leftContext=10`) reducing O(T²) attention to O(chunk²). 45-100x faster decode.
2. **Batch embedding lookups** — Sum all 15 codebook group embeddings in one call instead of 15 separate allocations per step.
3. **Bulk float extraction** — Replace per-element `.item()` loop with single `.asArray(Float.self)` call for waveform extraction.
4. **Removed GPU sync barriers** — Eliminated 5 unnecessary `eval()` calls per generation step that prevented GPU pipelining.
5. **Causal mask in codec decoder transformer** — Correctness fix matching Python's `create_additive_causal_mask`. Required for chunked decoding.

## Streaming Mode (Not Yet Implemented)

> The following describes the reference Python implementation. The Swift port does not yet support streaming.

The Talker generates in chunks for low-latency streaming:

1. Generate `streaming_interval` tokens (~25 = ~2 seconds of audio)
2. Keep 25-token context overlap for smooth transitions
3. Decode each chunk independently through Code Predictor + Codec
4. Stitch audio chunks with proportional trimming at boundaries

## Voice Cloning (Not Yet Implemented)

> The following describes the reference Python implementation. The Swift port does not yet support voice cloning.

```
Reference audio (24kHz)
    |
    +---> Speech Tokenizer Encoder -> 16 codebook indices per frame
    |     (prepended to generation as in-context learning)
    |
    +---> Speaker Encoder (ECAPA-TDNN) -> 1024-dim x-vector
          (injected as speaker embedding in codec prefix)
```

## Comparison with ASR Pipeline

| Aspect | ASR | TTS |
|--------|-----|-----|
| Direction | Audio -> Text | Text -> Audio |
| Input | 16kHz audio | Text string |
| Output | Token string | 24kHz waveform |
| Audio processing | Mel spectrogram (Accelerate) | Neural codec (Mimi, 682MB) |
| Main transformer | 28 layers, 1D RoPE | 28 layers, **MRoPE** |
| Additional stages | None | Code Predictor (5 layers) + Codec Decoder |
| Decoding strategy | Greedy argmax | Sampling (top-k/p/temperature) |
| Generation length | Short (text) | Long (codec tokens at 12.5 Hz) |

## Swift Port: Reusable Components

Directly reusable from the ASR codebase:

| ASR Component | TTS Usage |
|---------------|-----------|
| `QuantizedTextDecoderLayer` (transformer block) | Talker layers (swap RoPE for MRoPE) |
| `QuantizedTextAttention` (GQA + SDPA) | Both Talker and Code Predictor |
| `QuantizedTextMLP` (SwiGLU) | All transformer blocks |
| `PreQuantizedEmbedding` | Text embeddings |
| KV cache management | All transformers |
| `WeightLoader` + HF download | Extended for TTS weights |
| `Tokenizer` (Qwen2 BPE) | Same tokenizer |

## Implementation Status

| Component | Status | Lines |
|-----------|--------|-------|
| MRoPE | Done | ~120 |
| Talker (28-layer transformer) | Done | ~350 |
| Code Predictor (5-layer + 15 heads) | Done | ~200 |
| Speech Tokenizer Decoder | Done | ~600 |
| RVQ / Codebooks | Done | ~200 |
| Sampling (top-k/p/temp/rep penalty) | Done | ~110 |
| Inference orchestration | Done | ~280 |
| Weight loading (new keys) | Done | ~360 |
| Configuration | Done | ~150 |
| Speech Tokenizer Encoder | Not yet | — |
| Speaker Encoder (ECAPA-TDNN) | Not yet | — |
| Streaming inference | Not yet | — |
