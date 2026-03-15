# ASR Inference Pipeline ([Qwen3-ASR](https://arxiv.org/abs/2601.21337))

## Overview

```
Audio (16kHz) → [Preprocessing] → [Audio Encoding] → [Text Generation] → Transcription
                 ~5% time          ~20% time           ~75% time
```

## Stage 1: Preprocessing (AudioPreprocessing.swift)

Converts raw audio to a mel spectrogram `[128, T]` using Accelerate framework.

- STFT via `vDSP_fft_zrip` (in-place real FFT, zero-padded 400→512)
- Mel filterbank via `vDSP_mmul`, bin frequencies use padded FFT size (`k * fs / 512`)
- Log-mel via `vDSP_vclip` + `vvlog10f` (vForce vectorized)
- Hann window and FFT setup precomputed once in `init()`
- All temporary buffers preallocated outside the frame loop

## Stage 2: Audio Encoding (AudioEncoder.swift)

18-layer transformer with block attention over chunked mel features.

- Self-attention via `MLXFast.scaledDotProductAttention` (Metal kernel)
- Sinusoidal position embeddings cached by sequence length
- Block attention mask via MLXArray broadcast (`blockIds .== blockIds^T`)

## Stage 3: Text Generation (QuantizedTextDecoder.swift)

28-layer quantized Qwen3 decoder with GQA and RoPE.

- RoPE via `MLXFast.RoPE` (fused Metal kernel)
- GQA via `MLXFast.scaledDotProductAttention` (native GQA support, no manual tiling)
- Causal mask: `nil` for autoregressive steps (seqLen=1), broadcast for prefill
- **Prefill** (seqLen > 1): all prompt tokens in one forward pass
- **Decode** (seqLen = 1): SDPA uses optimized T_q=1 Metal kernel

## Performance

| Model | Framework | RTF | 10s audio processed in |
|-------|-----------|-----|------------------------|
| Qwen3-ASR-0.6B (4-bit) | MLX Swift | ~0.06 | ~0.6s |
| Whisper-large-v3 | whisper.cpp (Q5_0) | ~0.10 | ~1.0s |
| Whisper-small | whisper.cpp (Q5_0) | ~0.04 | ~0.4s |
