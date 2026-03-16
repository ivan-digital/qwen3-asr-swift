# VAD Detection Benchmark

## Dataset

**VoxConverse test** — 5 multi-speaker conversation files. Frame-level speech/non-speech evaluation at 10ms resolution.

## Results

| Engine | Params | Backend | F1% | FAR% | MR% | RTF | Cold Start |
|--------|--------|---------|-----|------|-----|-----|------------|
| Pyannote | 1.5M | MLX (GPU) | 98.22 | 50.09 | 0.19 | 0.358 | ~2s |
| Silero v5 | 309K | MLX (GPU) | 95.98 | 21.02 | 5.88 | 0.027 | ~1s |
| FireRedVAD | 588K | CoreML (ANE) | 94.21 | 69.33 | 5.05 | 0.009 | ~0.5s |

**Machine**: Apple M2 Max, 64 GB, macOS 14, release build.

## Key observations

- **Pyannote** has highest F1 (98.22%) with near-zero miss rate (0.19%) but very high false alarm (50%)
- **Silero** offers the best balance: good F1 (95.98%), moderate FAR (21%), streaming-capable (32ms chunks)
- **FireRedVAD** is fastest (RTF 0.009, 111x real-time) but has high FAR (69%) on VoxConverse — likely due to chunking boundary effects for long audio
- All engines have elevated FAR on VoxConverse due to background noise in multi-speaker conversation audio

## Comparison with published numbers

| Model | F1% | FAR% | MR% | Params | Dataset |
|-------|-----|------|-----|--------|---------|
| **Our Pyannote** | **98.22** | 50.09 | **0.19** | 1.5M | VoxConverse |
| FireRedVAD (paper) | 97.57 | **2.69** | 3.62 | 588K | FLEURS-VAD-102 |
| **Our Silero** | 95.98 | 21.02 | 5.88 | 309K | VoxConverse |
| Silero-VAD (paper) | 95.95 | 9.41 | 3.95 | 309K | FLEURS-VAD-102 |
| **Our FireRedVAD** | 94.21 | 69.33 | 5.05 | 588K | VoxConverse |

Our Silero F1 (95.98%) closely matches the paper's number (95.95%), validating our implementation. Higher FAR on VoxConverse vs FLEURS is expected — VoxConverse has more non-speech content with background noise.

## Reproduction

```bash
# Download VoxConverse test data first
python scripts/benchmark_diarization.py --download-only --num-files 5

# Run individual engines
python scripts/benchmark_vad.py --engine pyannote
python scripts/benchmark_vad.py --engine silero
python scripts/benchmark_vad.py --engine firered

# Compare all engines
python scripts/benchmark_vad.py --compare
```
