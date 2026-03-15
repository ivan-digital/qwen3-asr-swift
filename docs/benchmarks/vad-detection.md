# VAD Detection Benchmark

## Dataset

**VoxConverse test** — multi-speaker conversation audio. Frame-level speech/non-speech evaluation at 10ms resolution.

## Our Results

| Engine | F1% | FAR% | MR% | Precision% | Recall% | RTF |
|--------|-----|------|-----|------------|---------|-----|
| Pyannote (MLX) | 97.07 | 31.07 | 0.79 | 95.01 | 99.21 | 0.358 |

**Machine**: Apple M2 Max, 64 GB, macOS 14, release build.

Note: High FAR (31%) indicates Pyannote aggressively labels non-speech as speech on VoxConverse. Very low miss rate (0.79%) means almost no speech is missed.

## Comparison with Published Numbers

FireRedVAD paper (FLEURS-VAD-102, 102 languages):

| Model | F1% | FAR% | MR% | AUC-ROC% | Params | Dataset |
|-------|-----|------|-----|----------|--------|---------|
| FireRedVAD | 97.57 | 2.69 | 3.62 | 99.60 | 0.6M | FLEURS-VAD-102 |
| Silero-VAD | 95.95 | 9.41 | 3.95 | 97.99 | 0.3M | FLEURS-VAD-102 |
| TEN-VAD | 95.19 | 15.47 | 2.95 | 97.81 | — | FLEURS-VAD-102 |
| Our Pyannote (MLX) | 97.07 | 31.07 | 0.79 | — | 1.5M | VoxConverse |

Different datasets — direct comparison is indicative only. Our F1 is competitive but FAR is higher (VoxConverse has more non-speech content with background noise).

## Reproduction

```bash
# Download VoxConverse test data first
python scripts/benchmark_diarization.py --download-only --num-files 5

# Run VAD benchmark
python scripts/benchmark_vad.py --engine mlx --num-files 5
python scripts/benchmark_vad.py --compare  # both MLX and CoreML
```
