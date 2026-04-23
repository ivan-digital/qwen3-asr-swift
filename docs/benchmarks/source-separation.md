# Source Separation Benchmark

## Dataset

**MUSDB18-HQ** — 50 test tracks, full-length stereo music at 44.1kHz. Standard benchmark for music source separation (SiSEC 2018).

## Results

| Target | Our SDR (dB) | Published UMX-HQ | Gap |
|--------|-------------|-------------------|-----|
| Vocals | 6.23 | 6.32 | -0.09 |
| Drums | **6.44** | 5.73 | **+0.71** |
| Bass | 4.56 | 5.23 | -0.67 |
| Other | 3.41 | 4.02 | -0.61 |

**Machine**: Apple M2 Max, 64 GB, macOS 14, release build with compiled metallib.

**Key observations:**
- Vocals and drums match or exceed published numbers
- Bass and other are ~0.6 dB lower due to simplified Wiener post-filtering (soft-mask vs published EM)
- SDR computed globally per track (published uses BSSEval v4 per-window median, which can differ by ~0.5 dB)

## Performance

| Metric | Value |
|--------|-------|
| RTF | 0.23 |
| Speed | 4.3x real-time |
| Total audio | 12,471s (~3.5 hours) |
| Total processing | 2,878s (~48 min) |
| Memory | ~550 MB (4 models × 34 MB weights + STFT buffers) |

## Comparison with published models

| Model | Vocals | Drums | Bass | Other | Framework |
|-------|--------|-------|------|-------|-----------|
| **Open-Unmix HQ (MLX)** | **6.23** | **6.44** | **4.56** | **3.41** | **This benchmark** |
| Open-Unmix HQ (PyTorch) | 6.32 | 5.73 | 5.23 | 4.02 | PyTorch (published) |
| Demucs v3 | 7.68 | 7.08 | 7.41 | 4.42 | PyTorch |
| MDXC-Q | 8.90 | 7.58 | 7.30 | 6.25 | PyTorch |

Open-Unmix is a lightweight baseline (8.9M params per stem). Larger models like Demucs (83M) and MDXC achieve higher SDR but require significantly more compute.

## Reproduction

Benchmarks are run with the internal `museval`-based harness against the
[MUSDB18-HQ](https://zenodo.org/records/3338373) test set (~48 min on M2 Max for
the full 50-track run). Numbers above reflect the published
[`aufklarer/OpenUnmix-HQ-MLX`](https://huggingface.co/aufklarer/OpenUnmix-HQ-MLX)
weights; re-running locally requires `musdb` + `museval` in a Python env.
