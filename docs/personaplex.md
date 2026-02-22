# PersonaPlex: Full-Duplex Speech-to-Speech Model

## Overview

PersonaPlex is NVIDIA's 7B parameter full-duplex speech-to-speech model built on Kyutai's [Moshi](https://github.com/kyutai-labs/moshi) architecture. It enables simultaneous listening and speaking with controllable voice and role.

This implementation ports offline inference to Swift/MLX with 4-bit quantization (~3.5 GB for the temporal transformer).

## Architecture

```
[User Audio 24kHz] → [Mimi Encoder] → 16 codebook tokens @ 12.5Hz
                                              ↓
              [Temporal Transformer: 32L, dim=4096, 7B params]
                  17 streams summed: text + 8 user audio + 8 agent audio
                                              ↓
              [Depformer: 6L, dim=1024, per-codebook weights]
                  16 sequential steps → 16 agent audio codebook tokens
                                              ↓
[Agent Audio 24kHz] ← [Mimi Decoder] ← 16 codebook tokens @ 12.5Hz
```

## Components

### Mimi Codec (Kyutai)
- **Encoder**: SEANet convolutional encoder → 8-layer transformer → RVQ
- **Decoder**: RVQ decode → 8-layer transformer → SEANet convolutional decoder
- **Codebooks**: 16 (1 semantic via split RVQ + 15 acoustic)
- **Frame rate**: 12.5 Hz (80ms per frame)
- **Sample rate**: 24 kHz
- **Architecture**: Same `tokenizer-e351c8d8-checkpoint125.safetensors` as Moshi

### Temporal Transformer (7B, 4-bit quantized)
- **Layers**: 32
- **Dimension**: 4096
- **Heads**: 32 (head_dim=128)
- **FFN**: SiLU-gated (SwiGLU), intermediate=16896 (dim × 4.125)
- **Norm**: RMSNorm computed in float32
- **Position**: RoPE (base=10000)
- **Context**: 3000 tokens
- **Quantization**: 4-bit with group_size=64

**Embeddings (17 streams)**:
- Stream 0: Text embedding (vocab=32001)
- Streams 1-8: User audio embeddings (8 codebooks, vocab=2049)
- Streams 9-16: Agent audio embeddings (8 codebooks, vocab=2049)

All embeddings are summed before entering the transformer.

### Depformer (per-codebook weights)
- **Layers**: 6
- **Dimension**: 1024
- **Heads**: 16 (head_dim=64)
- **FFN**: SiLU-gated (SwiGLU), intermediate=4224
- **Context**: 8 tokens
- **Steps**: 16 (expanded from 8 in base Moshi)
- **No positional embedding** (depformer_pos_emb="none")

**Key feature — MultiLinear**:
Each attention and FFN layer uses `weights_per_step=True`, meaning separate weight matrices for each of the 16 codebook steps. Weights are stored as `[16 * outDim, inDim]` and sliced at runtime.

**Generation sequence** (per timestep):
```
for k in 0..<16:
  input = depformer_in[k](temporal_hidden)
  if k == 0: input += text_embedding(text_token)
  else:      input += audio_embedding[k-1](prev_audio_token)
  for layer in 6_layers:
    input = layer(input, step=k)  # uses weight[k]
  logits = linears[k](rms_norm(input))
  token = sample(logits)
```

## Delay Pattern

The 17 streams use temporal delays to handle autoregressive dependencies:

```
Stream  0 (text):           delay=0
Stream  1 (user audio cb0): delay=0  (semantic)
Stream  2 (user audio cb1): delay=1  (acoustic)
...
Stream  8 (user audio cb7): delay=1
Stream  9 (agent audio cb0): delay=0  (semantic)
Stream 10 (agent audio cb1): delay=1  (acoustic)
...
Stream 16 (agent audio cb7): delay=1
```

Semantic codebooks (cb0) and text have no delay; acoustic codebooks (cb1-7) have delay=1.

## Sampling

- **Audio**: temperature=0.8, top_k=250
- **Text**: temperature=0.7, top_k=25

## Weight Files

| File | Size | Contents |
|------|------|----------|
| `temporal.safetensors` | ~3.5 GB | 32-layer transformer (4-bit quantized) |
| `depformer.safetensors` | ~50 MB | 6-layer depformer with MultiLinear (BF16) |
| `embeddings.safetensors` | ~500 MB | 17 embeddings + output heads (BF16) |
| `mimi.safetensors` | ~385 MB | Mimi codec encoder/decoder/quantizer |
| `voices/*.safetensors` | ~6 MB | 18 voice preset embeddings |
| `tokenizer_spm_32k_3.model` | ~553 KB | SentencePiece text tokenizer |

## Voices

18 presets available:
- **Natural Female**: NATF0, NATF1, NATF2, NATF3
- **Natural Male**: NATM0, NATM1, NATM2, NATM3
- **Variety Female**: VARF0, VARF1, VARF2, VARF3, VARF4
- **Variety Male**: VARM0, VARM1, VARM2, VARM3, VARM4

## Memory Requirements

- Temporal transformer (4-bit): ~3.5 GB
- KV cache (context=3000): ~1 GB
- Mimi codec: ~400 MB
- Depformer + embeddings: ~550 MB
- **Total**: ~5.5 GB (fits comfortably in 64 GB M2 Max)

## References

- [PersonaPlex paper](https://arxiv.org/abs/2602.06053)
- [NVIDIA PersonaPlex](https://github.com/NVIDIA/personaplex)
- [Moshi/Mimi paper](https://arxiv.org/abs/2410.00037)
- [Kyutai Moshi](https://github.com/kyutai-labs/moshi)
- [HuggingFace model](https://huggingface.co/nvidia/personaplex-7b-v1)
