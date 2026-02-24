# PersonaPlex: Full-Duplex Speech-to-Speech Model

## Overview

PersonaPlex is NVIDIA's 7B parameter full-duplex speech-to-speech model built on Kyutai's [Moshi](https://github.com/kyutai-labs/moshi) architecture. It enables simultaneous listening and speaking with controllable voice and role.

This implementation ports inference to Swift/MLX with 4-bit quantization (~3.5 GB temporal transformer, ~650 MB depformer), supporting both offline and streaming output.

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
- **FFN**: SiLU-gated (SwiGLU), intermediate=11264 (dim × 2/3 × 4.125, LLaMA-style)
- **Norm**: RMSNorm computed in float32
- **Position**: RoPE (base=10000)
- **Context**: 3000 tokens
- **Quantization**: 4-bit with group_size=64

**Embeddings (17 streams)**:
- Stream 0: Text embedding (vocab=32001)
- Streams 1-8: User audio embeddings (8 codebooks, vocab=2049)
- Streams 9-16: Agent audio embeddings (8 codebooks, vocab=2049)

All embeddings are summed before entering the transformer.

### Depformer (per-codebook weights, 4-bit quantized)
- **Layers**: 6
- **Dimension**: 1024
- **Heads**: 16 (head_dim=64)
- **FFN**: SiLU-gated (SwiGLU), intermediate=2816 (dim × 2/3 × 4.125, LLaMA-style)
- **Context**: 8 tokens
- **Steps**: 16 (expanded from 8 in base Moshi)
- **No positional embedding** (depformer_pos_emb="none")
- **Quantization**: 4-bit with group_size=64 (MultiLinear attention/FFN + input projections)

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
  logits = linears[k](input)
  token = sample(logits)
```

## Inference Pipeline

```
1. Encode user audio with Mimi → [1, 16, T] codebook tokens
2. Replay voice prompt embeddings (50 frames, ~4s)
3. Silence spacer (0.5s)
4. Text system prompt (one token per frame)
5. Silence spacer (0.5s)
6. User audio frames — agent generates simultaneously (full-duplex)
7. Post-user generation (optional, up to maxSteps)
8. Decode agent tokens with Mimi → 24kHz response audio
```

**Example run** (M2 Max, 4-bit):
- Input: "Can you guarantee that the replacement part will be shipped tomorrow?" (20s)
- Output: "I can't promise a specific time, but we'll do our best to get it out tomorrow. It's one of the top priorities, so yes, we'll try to get it done as soon as possible and ship it first thing in the morning." (36s)
- RTF: ~0.87 (faster than real-time, both transformers 4-bit quantized)

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

## System Prompts

PersonaPlex accepts a text system prompt that steers the model's behavior. Prompts are pre-tokenized with SentencePiece (`tokenizer_spm_32k_3.model`) and injected between silence spacers before the user audio.

Several built-in presets are available (`--list-prompts` to see all). The default is a general helpful assistant prompt. The prompt significantly affects output quality — without focused instructions, the model tends to ramble off-topic.

## Sampling

- **Audio**: temperature=0.8, top_k=250, repetition_penalty=1.2 (window=30)
- **Text**: temperature=0.7, top_k=25

## Weight Files

| File | Size | Contents |
|------|------|----------|
| `temporal.safetensors` | ~3.4 GB | 32-layer transformer (4-bit quantized, including in_proj QKV) |
| `depformer.safetensors` | ~650 MB | 6-layer depformer with 16-step MultiLinear (4-bit quantized) |
| `embeddings.safetensors` | ~943 MB | 17 embeddings + output heads (BF16) |
| `mimi.safetensors` | ~367 MB | Mimi codec encoder/decoder/quantizer |
| `voices/*.safetensors` | ~6 MB | 18 voice preset embeddings |
| `tokenizer_spm_32k_3.model` | ~553 KB | SentencePiece text tokenizer |

### Weight Key Sanitization

The conversion script (`scripts/convert_personaplex.py`) maps PyTorch key conventions to Swift module paths:

- **RMSNorm**: `*.alpha` (1,1,D) → `*.weight` (D)
- **Packed QKV**: `*.in_proj_weight` → `*.in_proj.weight` (+ `_scales`/`_biases` for 4-bit)
- **Packed out_proj**: `*.out_proj_weight` → `*.out_proj.weight` (+ `_scales`/`_biases` for 4-bit)
- **Per-step FFN**: `layers.{l}.gating.{step}.linear_in.weight` → concatenated `layers.{l}.gating.linear_in.weight` (MultiLinear format, + `scales`/`biases` when quantized)
- **Embeddings split**: `embeddings.safetensors` contains mixed temporal + depformer keys, split at load time

## Voices

18 presets available:
- **Natural Female**: NATF0, NATF1, NATF2, NATF3
- **Natural Male**: NATM0, NATM1, NATM2, NATM3
- **Variety Female**: VARF0, VARF1, VARF2, VARF3, VARF4
- **Variety Male**: VARM0, VARM1, VARM2, VARM3, VARM4

## Memory Requirements

- Temporal transformer (4-bit): ~3.4 GB
- Depformer (4-bit, 16-step MultiLinear): ~650 MB
- Embeddings + output heads: ~943 MB
- Mimi codec: ~367 MB
- KV cache (context=3000): ~1 GB
- **Total**: ~6.4 GB (fits comfortably on M-series Macs with 16+ GB RAM)

## Streaming Inference

PersonaPlex supports streaming output via `respondStream()`, emitting audio chunks as they're generated (~2s per chunk at 25 frames).

### How It Works

During the autoregressive generation loop, agent audio codebook tokens are accumulated. Once enough frames are collected (default: 25 frames = ~2s), they're decoded incrementally through Mimi's streaming decoder (`MimiStreamingDecoder.decodeFrames()`) and emitted as an `AudioChunk` (from `Qwen3Common`).

```
Generation loop (12.5 Hz):
  temporal.forward() → depformer.generate() → accumulate tokens
  every 25 frames → Mimi decodeStep → emit audio chunk (~2s of 24kHz audio)
```

### Streaming Config

| Parameter | Default | Description |
|-----------|---------|-------------|
| `firstChunkFrames` | 25 | Frames before first audio emission (~2s) |
| `chunkFrames` | 25 | Frames per subsequent chunk (~2s) |

### API

```swift
let stream = model.respondStream(
    userAudio: audio,
    voice: .NATM0,
    streaming: PersonaPlexModel.PersonaPlexStreamingConfig(
        firstChunkFrames: 25, chunkFrames: 25)
)
for try await chunk in stream {
    playAudio(chunk.samples)  // 24kHz mono
    if chunk.isFinal { break }
}
```

## Performance Optimizations

The following optimizations are applied relative to the baseline offline implementation:

| Optimization | Impact | Description |
|-------------|--------|-------------|
| **eval() consolidation** | ~15-25% | Reduced GPU sync barriers from 3 to 1 per generation step |
| **Bulk audio extraction** | Decode phase | Single `.asArray(Float.self)` instead of per-sample `.item()` calls |
| **Prefill batching** | Prefill phase | Voice prompt + silence/text/silence batched into 2 forward passes (was ~300 individual) |
| **Compiled temporal transformer** | ~30% per step | `compile(shapeless: true)` fuses ~450 Metal kernels per step; opt-in via `warmUp()` |

### Compiled Inference

The temporal transformer can be compiled for Metal kernel fusion:

```swift
let model = try await PersonaPlexModel.fromPretrained(modelId: modelId)
model.warmUp()  // compile + warmup pass
// Subsequent respond() / respondStream() calls use compiled path
```

CLI: `personaplex-cli --input audio.wav --compile --output response.wav`

## References

- [PersonaPlex paper](https://arxiv.org/abs/2602.06053)
- [NVIDIA PersonaPlex](https://github.com/NVIDIA/personaplex)
- [Moshi/Mimi paper](https://arxiv.org/abs/2410.00037)
- [Kyutai Moshi](https://github.com/kyutai-labs/moshi)
- [HuggingFace model](https://huggingface.co/nvidia/personaplex-7b-v1)
