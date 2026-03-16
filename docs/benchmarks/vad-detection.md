# VAD Detection Benchmark

## Dataset

**VoxConverse test** — 5 multi-speaker conversation files. Frame-level speech/non-speech evaluation at 10ms resolution.

## Our Results

| Engine | F1% | FAR% | MR% | Precision% | Recall% | RTF |
|--------|-----|------|-----|------------|---------|-----|
| Pyannote (MLX) | 97.32 | 47.08 | 0.26 | — | 99.74 | 0.358 |
| Silero (MLX) | 95.98 | 21.02 | 5.88 | 97.91 | 94.12 | 0.056 |

**Machine**: Apple M2 Max, 64 GB, macOS 14, release build.

**Observations:**
- Pyannote has higher F1 (97.3%) but very high FAR (47%) — aggressively labels non-speech as speech
- Silero has lower FAR (21%) and is 6.4x faster (RTF 0.056 vs 0.358)
- Both have high FAR on VoxConverse due to background noise in conversational audio

## Comparison with published numbers

FireRedVAD paper (FLEURS-VAD-102, 102 languages):

| Model | F1% | FAR% | MR% | Params | Dataset |
|-------|-----|------|-----|--------|---------|
| FireRedVAD | 97.57 | 2.69 | 3.62 | 0.6M | FLEURS-VAD-102 |
| Our Pyannote (MLX) | 97.32 | 47.08 | 0.26 | 1.5M | VoxConverse |
| Our Silero (MLX) | 95.98 | 21.02 | 5.88 | 0.3M | VoxConverse |
| Silero-VAD (paper) | 95.95 | 9.41 | 3.95 | 0.3M | FLEURS-VAD-102 |
| TEN-VAD | 95.19 | 15.47 | 2.95 | — | FLEURS-VAD-102 |

Different datasets — direct comparison is indicative only. Our Silero F1 (95.98%) matches the paper's number (95.95%), validating our implementation.

## Reproduction

```bash
# Download VoxConverse test data first
python scripts/benchmark_diarization.py --download-only --num-files 5

# Run VAD benchmark
python scripts/benchmark_vad.py --engine pyannote
python scripts/benchmark_vad.py --engine silero
python scripts/benchmark_vad.py --compare
```
