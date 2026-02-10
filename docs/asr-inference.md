# ASR Inference Pipeline

## Overview

```
Audio (16kHz) → [Preprocessing] → [Audio Encoding] → [Text Generation] → Transcription
                 ~5% time          ~20% time           ~75% time
```

## Stage 1: Preprocessing (AudioPreprocessing.swift)

Converts raw audio to a mel spectrogram `[128, T]`.

**Optimizations applied:**

| Operation | Before | After | Speedup |
|-----------|--------|-------|---------|
| STFT | Manual DFT (O(N^2) per frame) | `vDSP_fft_zrip` (Accelerate FFT, zero-padded 400->512) | ~20x |
| Mel filterbank | Triple-nested scalar loop | `vDSP_mmul` (BLAS matrix multiply) | ~100x |
| Log10 | Scalar `log10(max(x, eps))` loop | `vDSP_vclip` + `vvlog10f` (vForce) | ~10x |
| Clamp/normalize | Two scalar loops | `vDSP_vclip` + `vDSP_vsmsa` | ~10x |
| Windowing | Scalar multiply loop | `vDSP_vmul` | ~5x |

**Key design decisions:**
- Hann window and FFT setup (`vDSP_create_fftsetup`) precomputed once in `init()`
- All temporary buffers preallocated outside the frame loop
- `vDSP_fft_zrip` uses in-place real FFT: input zero-padded from 400 to 512, split-complex packing with DC in `realp[0]` and Nyquist in `imagp[0]`
- Mel filterbank bin frequencies use padded FFT size (`k * fs / 512`)

## Stage 2: Audio Encoding (AudioEncoder.swift)

18-layer transformer with block attention over chunked mel features.

**Optimizations applied:**

| Operation | Before | After |
|-----------|--------|-------|
| Self-attention | Manual matmul chain | `MLXFast.scaledDotProductAttention` (Metal kernel) |
| Position embeddings | Recomputed every call | Cached by sequence length |
| Block attention mask | Scalar O(n^2) loop | MLXArray broadcast (`blockIds .== blockIds^T`) |

## Stage 3: Text Generation (QuantizedTextDecoder.swift)

28-layer quantized Qwen3 decoder with GQA and RoPE.

**Optimizations applied:**

| Operation | Before | After | Impact |
|-----------|--------|-------|--------|
| RoPE | Manual split-half rotation | `MLXNN.RoPE` -> `MLXFast.RoPE` (fused Metal kernel) | ~1.3x |
| Attention | Manual GQA tile + matmul chain | `MLXFast.scaledDotProductAttention` (native GQA) | ~2.5x |
| Causal mask | Scalar O(n^2) loop every call | `nil` for autoregressive (seqLen=1), broadcast for prefill | Eliminates mask for 99%+ of steps |

**Autoregressive generation flow:**
1. **Prefill** (seqLen > 1): Full causal mask via broadcast, all prompt tokens in one forward pass
2. **Decode** (seqLen = 1): No mask needed, SDPA uses optimized T_q=1 Metal kernel

## Performance

| Model | Framework | RTF | 10s audio processed in |
|-------|-----------|-----|------------------------|
| Qwen3-ASR-0.6B (4-bit) | MLX Swift | ~0.06 | ~0.6s |
| Whisper-large-v3 | whisper.cpp (Q5_0) | ~0.10 | ~1.0s |
| Whisper-small | whisper.cpp (Q5_0) | ~0.04 | ~0.4s |
