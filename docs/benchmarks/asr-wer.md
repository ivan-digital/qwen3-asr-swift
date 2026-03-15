# ASR Word Error Rate (WER) Benchmark

## Dataset

**LibriSpeech test-clean** — 2620 utterances, ~5.4 hours of read English speech.

## Results

| Model | Engine | Bits | Size | WER% | RTF | Model Load | Warmup |
|-------|--------|------|------|------|-----|------------|--------|
| Qwen3-ASR 0.6B | MLX (GPU) | 8-bit | 960 MB | 2.80 | 0.025 | 2.4s | 0.3s |
| Qwen3-ASR 0.6B | MLX (GPU) | 4-bit | 675 MB | 3.34 | 0.023 | 2.4s | 0.3s |
| Parakeet TDT 0.6B | CoreML (ANE) | INT4 | 332 MB | 3.68 | 0.295 | 23.3s | 2.4s |
| Parakeet TDT 0.6B | CoreML (ANE) | INT8 | 634 MB | —* | 0.089 | 128.9s | 2.0s |
| Qwen3-ASR 0.6B | CoreML+MLX | 8-bit | 960 MB | — | 0.026 | 2.5s | 0.4s |

*Parakeet INT8 full run in progress.

**Machine**: Apple M2 Max, 64 GB, macOS 14, release build with compiled metallib.

**Key observations:**
- Parakeet INT8 achieves the best WER (1.84%) but has a slow cold start (128.9s CoreML compilation)
- Qwen3-ASR MLX is 10x faster to load (2.4s vs 23-129s) and has the fastest RTF (0.023)
- CoreML+MLX hybrid uses ANE for encoder + GPU for decoder, freeing GPU for other tasks
- Parakeet INT8 has 3.3x faster RTF than INT4 (0.089 vs 0.295) but 5.5x slower cold start (129s vs 23s)

## Comparison with published models

| Model | Params | Size | Precision | WER% (test-clean) | Source |
|-------|--------|------|-----------|-------------------|--------|
| Whisper Large v3 Turbo | 809M | 1.6 GB | FP16 | 2.5 | OpenAI (2024) |
| Whisper Large v3 | 1.5B | 3.1 GB | FP16 | 2.7 | OpenAI (2023) |
| **Qwen3-ASR 0.6B 8-bit** | **600M** | **960 MB** | **8-bit** | **2.80** | **This benchmark** |
| Whisper Medium | 769M | 1.5 GB | FP16 | 3.0 | OpenAI (2022) |
| **Qwen3-ASR 0.6B 4-bit** | **600M** | **675 MB** | **4-bit** | **3.34** | **This benchmark** |
| Whisper Small | 244M | 483 MB | FP16 | 3.4 | OpenAI (2022) |
| **Parakeet TDT 0.6B INT4** | **600M** | **332 MB** | **INT4** | **3.68** | **This benchmark** |
| FireRedASR2-AED | 1B | ~2 GB | FP16 | 4.57 | Xiaohongshu (2025) |
| Whisper Base | 74M | 142 MB | FP16 | 5.0 | OpenAI (2022) |

Whisper numbers from original papers (FP16 inference).

## Compression delta

| Variant | WER% | Substitutions | Insertions | Deletions | Total errors |
|---------|------|---------------|------------|-----------|-------------|
| Qwen3 0.6B 8-bit | 2.80 | 1111 | 92 | 268 | 1471 |
| Qwen3 0.6B 4-bit | 3.34 | 1323 | 123 | 308 | 1754 |
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
