# ASR Model Architecture ([paper](https://arxiv.org/abs/2601.21337))

## Overview

Qwen3-ASR is an encoder-decoder model: audio encoder extracts features, text decoder generates transcription tokens autoregressively.

Two model sizes are supported:
- **0.6B** (`aufklarer/Qwen3-ASR-0.6B-MLX-4bit`) — 4-bit quantized, ~0.4 GB
- **1.7B** (`aufklarer/Qwen3-ASR-1.7B-MLX-8bit`) — 8-bit quantized, ~2.5 GB

```
Audio (16kHz mono)
    |
    v
+-------------------+
|  Mel Spectrogram  |   WhisperFeatureExtractor
|  128 bins, 8ms    |   Accelerate FFT (vDSP_fft_zrip)
+--------+----------+
         |
         v
+-------------------+
|  Audio Encoder    |   Conv2D (3 layers, stride-2) + Transformer
|                   |   Block attention, sinusoidal pos embeddings
+--------+----------+
         |
         v
+-------------------+
|  Projector        |   2-layer MLP (dModel -> outputDim)
+--------+----------+
         |  audio embeddings injected into decoder (no variance scaling)
         v
+-------------------+
|  Text Decoder     |   Qwen3 LLM (28 layers, quantized)
|                   |   GQA, RoPE, SwiGLU, KV cache
+--------+----------+
         |
         v
    Transcription
```

## Audio Encoder

| Parameter | 0.6B | 1.7B |
|-----------|------|------|
| Hidden size (d_model) | 896 | 1024 |
| Layers | 18 | 24 |
| Attention heads | 14 | 16 |
| FFN dim | 3584 | 4096 |
| Output dim (projector) | 1024 | 2048 |
| Conv2D layers | 3 (stride 2 each = 8x downsample) | 3 |
| Downsample hidden size | 480 | 480 |
| Position encoding | Sinusoidal (cached) | Sinusoidal |
| Attention type | Block attention (chunked) | Block attention |
| Chunk size | 100 frames (configurable) | 100 frames |

**Block attention:** Audio is split into fixed-size chunks. Attention is restricted within each chunk via a block diagonal mask, reducing complexity from O(T^2) to O(T * chunk_size).

**Conv2D frontend:** Three Conv2D layers with GELU activation downsample the mel spectrogram 8x in the time dimension before the transformer layers.

## Text Decoder (Qwen3)

| Parameter | 0.6B | 1.7B |
|-----------|------|------|
| Hidden size | 1024 | 2048 |
| Layers | 28 | 28 |
| Attention heads (Q) | 16 | 16 |
| KV heads (GQA) | 8 | 8 |
| Head dimension | 128 | 128 |
| Intermediate size (MLP) | 3072 | 6144 |
| Vocab size | 151936 | 151936 |
| RoPE base | 1,000,000 | 1,000,000 |
| RoPE type | MRoPE [24,20,20]* | MRoPE [24,20,20]* |
| Quantization | 4-bit (group=64) | 8-bit (group=64) |
| Activation | SwiGLU | SwiGLU |
| Norm | RMSNorm (eps=1e-6) | RMSNorm |
| Q/K normalization | RMSNorm per head | RMSNorm per head |

*Both models use MRoPE config in HuggingFace, but for ASR (single-modal, no image input) all 3 position dimensions are identical, so it reduces to standard 1D RoPE at inference time.

**Transformer block:**
```
x -> RMSNorm -> Attention(Q/K/V projections, Q/K RMSNorm, RoPE, GQA via SDPA) -> + residual
  -> RMSNorm -> SwiGLU MLP(gate_proj, up_proj, down_proj)                      -> + residual
```

**GQA (Grouped Query Attention):** 16 query heads share 8 KV heads (2:1 ratio). SDPA handles this natively without manual tiling.

**Audio injection:** Audio embeddings from the projector are concatenated into the token embedding sequence at designated positions (marked by special tokens). No variance scaling is applied — direct dtype cast only.

## Tokenizer

- Qwen2 BPE tokenizer (vocab size 151936)
- Special tokens: `<|im_start|>`, `<|im_end|>`, `<asr_text>`, language tags
- Byte-level BPE decoding with GPT-2 byte-to-unicode mapping
- Language auto-detection via model output

## Weight Files

| File | Purpose |
|------|---------|
| `model-00001-of-00002.safetensors` | Audio encoder + text decoder weights (part 1) |
| `model-00002-of-00002.safetensors` | Text decoder weights (part 2) |
| `vocab.json` | Token-to-ID mapping |
| `tokenizer_config.json` | Tokenizer settings + added tokens |

Total size: ~0.4 GB (0.6B 4-bit), ~2.5 GB (1.7B 8-bit)
