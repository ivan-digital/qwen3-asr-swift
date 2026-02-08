# Inference Architecture & Performance

## Pipeline Overview

Qwen3-ASR inference has three stages:

```
Audio → [Preprocessing] → [Audio Encoding] → [Text Generation] → Transcription
         ~5% time          ~20% time           ~75% time
```

## Preprocessing (AudioPreprocessing.swift)

Converts raw audio to a mel spectrogram `[128, T]`.

**Optimizations applied:**

| Operation | Before | After | Speedup |
|-----------|--------|-------|---------|
| STFT | Manual DFT with twiddle factors (O(N^2) per frame) | `vDSP_fft_zrip` (Accelerate FFT, O(N log N), zero-padded 400→512) | ~20x |
| Mel filterbank | Triple-nested scalar loop | `vDSP_mmul` (BLAS matrix multiply) | ~100x |
| Log10 | Scalar `log10(max(x, eps))` loop | `vDSP_vclip` + `vvlog10f` (vForce) | ~10x |
| Clamp/normalize | Two scalar loops | `vDSP_vclip` + `vDSP_vsmsa` | ~10x |
| Windowing | Scalar multiply loop | `vDSP_vmul` | ~5x |

**Key design decisions:**
- Hann window and FFT setup (`vDSP_create_fftsetup`) are precomputed once in `init()` and reused across calls
- All temporary buffers (windowed frame, split-complex I/O) are preallocated outside the frame loop
- `vDSP_fft_zrip` uses in-place real FFT: input zero-padded from 400 to 512 (power-of-2), split-complex packing with DC in `realp[0]` and Nyquist in `imagp[0]`
- Mel filterbank bin frequencies use the padded FFT size (`k * fs / 512`) for correct frequency mapping

## Audio Encoder (AudioEncoder.swift)

18-layer transformer with block attention over chunked mel features.

**Optimizations applied:**

| Operation | Before | After |
|-----------|--------|-------|
| Self-attention | Manual `matmul → scale → mask → softmax → matmul → transpose` | `MLXFast.scaledDotProductAttention` (optimized Metal kernel) |
| Position embeddings | Recomputed every call | Cached by sequence length in `cachedPosEmbeddings` dictionary |
| Block attention mask | Scalar O(n^2) loop filling `[seqLen, seqLen]` array | MLXArray broadcast comparison (`blockIds .== blockIds^T`) |

**SDPA details:**
- Input: `[B, N_heads, T, D]` for Q/K/V
- Returns: `[B, N_heads, T, D]` (same layout as input), requires `.transposed(0, 2, 1, 3)` to get `[B, T, N_heads, D]`
- Dispatches to optimized Metal kernel — significant speedup for encoder's fixed-length sequences

## Text Decoder (QuantizedTextDecoder.swift)

28-layer quantized Qwen3 decoder with GQA (Grouped Query Attention) and RoPE.

**Optimizations applied:**

| Operation | Before | After | Impact |
|-----------|--------|-------|--------|
| RoPE | Manual split-half rotation: slice, broadcast cos/sin, element-wise multiply, concatenate | `MLXNN.RoPE` → `MLXFast.RoPE` (single fused Metal kernel) | ~1.3x |
| Attention | Manual: GQA tile (expand+tile+reshape) → matmul → scale → mask → softmax → matmul → transpose | `MLXFast.scaledDotProductAttention` (handles GQA natively, no tiling needed) | ~2.5x |
| Causal mask | Scalar O(n^2) loop for every call | `nil` for autoregressive steps (seqLen=1), MLX broadcast for prefill | Eliminates mask overhead for 99%+ of steps |

**SDPA + GQA details:**
- SDPA accepts `queries: [B, N_q, T, D]` and `keys/values: [B, N_kv, T, D]` where `N_q != N_kv`
- Eliminates the expensive GQA tiling step (expand → tile → reshape was O(N_q/N_kv) memory overhead)
- When `T_q == 1` (every autoregressive step), dispatches to a specialized Metal kernel optimized for single-query attention

**Autoregressive generation flow:**
1. **Prefill** (seqLen > 1): Full causal mask via broadcast comparison, processes all prompt tokens in one forward pass
2. **Decode** (seqLen = 1): No mask needed (single query attends to all cached positions), SDPA uses optimized T_q=1 kernel
