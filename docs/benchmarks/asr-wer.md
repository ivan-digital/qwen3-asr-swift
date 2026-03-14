# ASR Word Error Rate (WER) Benchmark

## Dataset

**LibriSpeech test-clean** — 2620 utterances, ~5.4 hours of read English speech.

## Results

| Model | Engine | Bits | Size | WER% | RTF | Model Load | Warmup |
|-------|--------|------|------|------|-----|------------|--------|
| Qwen3-ASR 0.6B | MLX (GPU) | 4-bit | 675 MB | 3.34 | 0.023 | 2.4s | 0.3s |
| Qwen3-ASR 0.6B | MLX (GPU) | 8-bit | 960 MB | 2.80 | 0.025 | 2.4s | 0.5s |
| Parakeet TDT 0.6B | CoreML (ANE) | INT4 | 332 MB | — | 0.295 | 23.3s | 2.4s |

Parakeet WER pending (full run in progress).

**Machine**: Apple M2 Max, 64 GB, macOS 14, release build with compiled metallib.

## Comparison with published models

| Model | Params | Size | Precision | WER% (test-clean) | Source |
|-------|--------|------|-----------|-------------------|--------|
| Whisper Large v3 Turbo | 809M | 1.6 GB | FP16 | 2.5 | OpenAI (2024) |
| Whisper Large v3 | 1.5B | 3.1 GB | FP16 | 2.7 | OpenAI (2023) |
| **Qwen3-ASR 0.6B 8-bit** | **600M** | **960 MB** | **8-bit** | **2.80** | **This benchmark** |
| Whisper Medium | 769M | 1.5 GB | FP16 | 3.0 | OpenAI (2022) |
| **Qwen3-ASR 0.6B 4-bit** | **600M** | **675 MB** | **4-bit** | **3.34** | **This benchmark** |
| Whisper Small | 244M | 483 MB | FP16 | 3.4 | OpenAI (2022) |
| FireRedASR2-AED | 1B | ~2 GB | FP16 | 4.57 | Xiaohongshu (2025) |
| Whisper Base | 74M | 142 MB | FP16 | 5.0 | OpenAI (2022) |

Whisper numbers are from the original papers using FP16 inference. Qwen3-ASR 0.6B at 8-bit (960 MB) matches Whisper Large v3 (3.1 GB) quality at 3x smaller size.

## Compression delta

| Variant | WER% | Substitutions | Insertions | Deletions | Total errors |
|---------|------|---------------|------------|-----------|-------------|
| 0.6B 8-bit | 2.80 | 1111 | 92 | 268 | 1471 |
| 0.6B 4-bit | 3.34 | 1323 | 123 | 308 | 1754 |
| Delta | +0.54 | +212 | +31 | +40 | +283 |

4-bit adds 0.54% WER (19% more errors). Model size: 675 MB (4-bit) vs 960 MB (8-bit) — 30% smaller.

## Reproduction

```bash
make build
python scripts/benchmark_asr.py --batch --engine qwen3 --model 0.6B
python scripts/benchmark_asr.py --batch --engine qwen3 --model 0.6B-8bit
python scripts/benchmark_asr.py --batch --engine parakeet
python scripts/benchmark_asr.py --batch --engine parakeet --model int8
```

First run downloads LibriSpeech test-clean (~350 MB). Results saved to `benchmarks/librispeech/`.
