# Marvis TTS Architecture

> CSM (Conversational Speech Model) dual-transformer with Mimi codec for streaming text-to-speech. Based on [CSM](https://arxiv.org/abs/2503.01758) (Sesame) and [Mimi](https://arxiv.org/abs/2410.00037) (Kyutai).

## Overview

```
Text -> [Tokenize] -> [Backbone] -> [Decoder] -> [Mimi Decode] -> Audio (24kHz)
          <1%           ~50%          ~40%          ~10%
```

## Model

| Component | Parameters | Layers | Hidden | Heads (Q/KV) |
|-----------|-----------|--------|--------|--------------|
| Backbone | ~250M | 16 | 2048 | 32Q / 8KV |
| Decoder | ~60M | 4 | 1024 | 8Q / 2KV |
| Mimi Codec | ~100M | — | — | — |
| **Total** | **~310M (8-bit: ~350 MB)** | | | |

Default model: `Marvis-AI/marvis-tts-250m-v0.2-MLX-8bit`

## Stage 1: Text Tokenization

```
Input text + reference context
    |
    v
Llama-3 tokenizer (via HuggingFace AutoTokenizer)
    "[speaker_id]" + text -> token IDs
    |
    v
Token frame: [T, K+1] where K = 32 codebooks
    text IDs in last column, audio codebooks in first K columns
    mask: 1 where data exists, 0 otherwise
```

## Stage 2: Reference Audio Encoding

```
Reference audio (24kHz)
    |
    v
Mimi encoder (SEANet encoder + encoder transformer + RVQ quantization)
    Downsample: 8×6×5×4 = 960x (24kHz -> 25 Hz)
    Quantize: 32 codebooks × 2048 vocab
    |
    v
Audio codes [K, T_ref]
    Transposed to [T_ref, K] and framed as [T_ref, K+1]
    mask: audio columns = 1, text column = 0
```

## Stage 3: Backbone Generation (First Codebook)

```
Context tokens (ref text + ref audio) + generation text
    |
    v
Token embedding:
    text_embeddings(text_ids) + sum(audio_embeddings(codebook_i))
    Masked and summed per frame -> [B, T, 2048]
    |
    v
16-layer Llama-3 transformer (autoregressive, KV cache)
    Llama3 Scaled RoPE (wavelength-dependent frequency scaling)
    GQA: 32 query heads, 8 KV heads
    SwiGLU MLP (gate + up + down projections)
    |
    v
codebook0_head (Linear -> 2048 vocab) -> logits
    |
    v
Sampling: temperature=0.9, top-p=0.8
    |
    v
First codebook token (until all-zero EOS frame)
```

## Stage 4: Decoder (Remaining 31 Codebooks)

For each generated first-codebook token:

```
Backbone hidden state -> projection (2048 -> 1024)
    |
    v
For codebook_group in 1..31:
    Embed previous codebook token (audio_embeddings)
    Add to projected hidden state
    |
    v
    4-layer Llama-3 decoder transformer (KV cache)
        GQA: 8 query heads, 2 KV heads
        SwiGLU MLP
    |
    v
    audio_head[group] (Linear -> 2048 vocab) -> sample token
    |
    v
Result: 32 codebook tokens per time step
```

## Stage 5: Mimi Codec Decode (Streaming)

```
32 codebook indices per frame
    |
    v
SplitResidualVectorQuantizer:
    rvq_first.decode(codebook_1)    -> [T, 256] -> project to [T, 512]
    rvq_rest.decode(codebooks_2-32) -> [T, 256] -> project to [T, 512]
    sum embeddings                  -> [T, 512]
    |
    v
Decoder transformer (8 layers, RoPE, SiLU-gated MLP, LayerScale):
    input_proj: 512 -> 256 bottleneck
    8x MimiTransformerLayer (hidden=256)
    output_proj: 256 -> 512              -> [T, 512]
    |
    v
SEANet decoder:
    init_conv -> [T, 1024]
    Upsample blocks (4×5×6×8 = 960x):
        TransposedConv1d + ResnetBlocks (ELU + dilated convs)
    final_conv -> [1, 1, T*960]
    |
    v
Audio waveform at 24kHz (960x upsample from 25 Hz frame rate)
```

### Streaming Decode

Mimi supports frame-by-frame streaming decode. Each new frame of 32 codebook tokens can be decoded immediately to produce ~960 audio samples (40ms at 24kHz). The streaming decoder maintains internal state across frames.

## Key Differences from Qwen3-TTS

| | Qwen3-TTS | Marvis TTS |
|---|-----------|------------|
| Architecture | Single transformer + code predictor | Dual transformer (backbone + decoder) |
| Codebooks | 16 | 32 |
| Frame rate | 12.5 Hz | 25 Hz |
| Streaming | No (sequential code predictor) | Yes (frame-by-frame Mimi decode) |
| Tokenizer | Custom Qwen2 BPE | Llama-3 (HuggingFace) |
| RoPE | MRoPE (multi-dimensional) | Llama3 Scaled RoPE |
| Voice cloning | Planned (speaker encoder) | Reference audio + text |
| Quantization | 4-bit | 8-bit |

## Weight Sources

- CSM model: `Marvis-AI/marvis-tts-250m-v0.2-MLX-8bit` (HuggingFace)
- Mimi codec: `kyutai/moshiko-pytorch-bf16` (HuggingFace)
- Voice prompts: bundled in model repo (`prompts/conversational_a.wav`, etc.)
