# TTS Inference Pipeline (Qwen3-TTS)

> Reference for Swift MLX port. Based on [Qwen3-TTS-12Hz-0.6B](https://arxiv.org/abs/2601.15621). Speech tokenizer decoder based on [Mimi](https://arxiv.org/abs/2410.00037) (Kyutai).

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

## Stage 1b: Instruct Preparation (CustomVoice only)

When an instruct string is provided (or the default `"Speak naturally."` is applied), it is tokenized and prepended to the prefill embeddings:

```
Instruct text (e.g. "Speak in a cheerful, upbeat tone")
    |
    v
ChatML format:
    <|im_start|>user\n{instruct}<|im_end|>\n
    token IDs: [151644, 872, 198, ...instruct tokens..., 151645, 198]
    |
    v
text_embedding -> text_projection
    |
    v
Instruct hidden states [B, T_instruct, 1024]
```

The instruct embeddings are concatenated at the start of the prefill sequence:

```
Without instruct: [role_embed | text_embeddings | trailing_tokens]
With instruct:    [instruct_embeddings | role_embed | text_embeddings | trailing_tokens]
```

This positions the style instruction early in the context so the Talker's attention can condition all subsequent generation on it.

## Stage 2: Codec Prefix

Before autoregressive generation, a 6-token codec prefix is constructed:

```
[think, think_bos, language_id, think_eos, pad, bos]
```

- `codec_think`: 2151, `codec_think_bos`: 2153, `codec_think_eos`: 2154
- `codec_pad`: 2148, `codec_bos`: 2149
- `language_id`: e.g. 2050 (English), 2052 (German), 2054 (Spanish), 2055 (Chinese), 2058 (Japanese), 2061 (French), 2064 (Korean), 2069 (Russian), 2070 (Italian)

> **CustomVoice model:** When a speaker is selected, a 7th token (the speaker token ID) is appended to the codec prefix. See `tts-model.md` → Model Variants for the full speaker list and language mappings.

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

## vs Apple AVSpeechSynthesizer (M2 Max, 64 GB)

| | Qwen3-TTS (release) | Apple TTS |
|---|-----------|-----------|
| RTF (long text) | ~0.7 | ~0.02 |
| Latency (6s audio) | 3.9s | 0.17s |
| Speech quality | Natural, expressive | Robotic, monotone |
| Voice cloning | Planned | No |
| Languages | EN/ZH/DE/JA/ES/FR/KO/RU/IT | 60+ |
| On-device | Yes (MLX) | Yes (AVFoundation) |
| Model size | ~1.7 GB | Built-in |

### Implementation Notes

- **Chunked codec decoding** — Codec frames processed in overlapping chunks (`chunkSize=25, leftContext=10`), reducing O(T²) attention to O(chunk²)
- **Batch embedding lookups** — All 15 codebook group embeddings summed in one call per step
- **Bulk float extraction** — Waveform extracted via single `.asArray(Float.self)` call
- **Causal mask in decoder transformer** — Additive causal mask for pre-transformer attention (required for chunked decoding correctness)
- **Lazy code predictor chain** — 15 sequential codebook predictions use lazy MLXArray (Gumbel-max trick) with a single `eval()` at the end, reducing 15 GPU sync barriers to 1 per timestep
- **Compiled talker + code predictor** — `compile(shapeless: true)` for the 28-layer talker (growing KV cache); `compile(shapeless: false)` for the 5-layer code predictor (14 fixed cache sizes)

## Streaming Synthesis

The Talker, Code Predictor, and Mimi decoder are all fully causal, enabling chunk-by-chunk audio emission during generation.

```
[Prefill] → emit first chunk → [Generate + emit subsequent chunks] → final chunk
    ~25ms        ~120ms                53ms/step
```

**`synthesizeStream()`** returns `AsyncThrowingStream<AudioChunk>` (from `Qwen3Common`):

1. **First chunk** — emitted after `firstChunkFrames` tokens (default 3, or 1 for low-latency)
2. **Subsequent chunks** — emitted every `chunkFrames` tokens (default 25 = 2s audio)
3. **Codec decode** — each chunk runs the Mimi decoder with left-context overlap for quality

### Zero-Pad Decode

When `firstChunkFrames < 4` (the codec's minimum input size due to ConvNeXt kernel=7 after 2x pre-upsample), the decoder input is zero-padded on the left. The decoder is fully causal, so zero frames produce silence. Output is trimmed from the right to keep exactly `realChunkFrames × 1920` samples.

### First-Packet Latency (M2 Max, release)

| Config | First Chunk | Latency |
|--------|-------------|---------|
| Default (3 frames) | 240ms audio | ~225ms |
| Low-latency (1 frame) | 80ms audio | ~120ms |

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

