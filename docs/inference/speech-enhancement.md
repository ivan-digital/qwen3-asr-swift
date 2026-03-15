# Speech Enhancement — DeepFilterNet3

## Overview

DeepFilterNet3 (Interspeech 2023) removes background noise from speech in real-time. Runs on Apple Neural Engine via Core ML (~2.1M params, FP16, ~4.2 MB).

```
Audio 48kHz → STFT (960-pt, 480 hop, Vorbis window) → 481 complex bins
  → Encoder (Conv2d + SqueezedGRU) → ERB mask + DF coefficients
  → ERB mask applied to full spectrum (broadband noise removal)
  → Deep filtering on lowest 96 bins (fine-grained enhancement)
  → iSTFT → Enhanced audio 48kHz
```

Neural network runs on **Neural Engine** (Core ML). Signal processing (STFT, ERB filterbank, deep filtering) runs on **CPU** (Accelerate/vDSP).

## Parameters

| Parameter | Value |
|-----------|-------|
| FFT / hop | 960 / 480 (10ms frames @ 48kHz) |
| ERB bands | 32 |
| DF bins / order | 96 / 5 taps |
| GRU hidden | 256 |
| Params | ~2.1M (~4.2 MB FP16) |
| Max duration | ~48s (6000 frames) |

## Latency (M2 Max)

| Duration | Time | RTF |
|----------|------|-----|
| 5s | 0.65s | 0.13 |
| 10s | 1.2s | 0.12 |
| 20s | 4.8s | 0.24 |

Core ML GRU cost scales ~O(n²) due to sequential hidden state processing. Short audio is proportionally faster.

## CLI

```bash
swift run audio denoise noisy.wav
swift run audio denoise noisy.wav --output clean.wav
```

## Conversion

```bash
python scripts/convert_deepfilternet3.py [--output OUTPUT_DIR]
```

Outputs:
- `DeepFilterNet3.mlpackage` (~4.2 MB) — Core ML FP16 model
- `auxiliary.npz` (~126 KB) — ERB filterbank, Vorbis window, normalization states

## References

- [DeepFilterNet3 Paper](https://arxiv.org/abs/2305.08227) (Interspeech 2023)
- [GitHub: Rikorose/DeepFilterNet](https://github.com/Rikorose/DeepFilterNet)
- MIT/Apache-2.0 dual license
