# ASR Word Error Rate (WER) Benchmark

## Dataset

**LibriSpeech test-clean** — 2620 utterances, ~5.4 hours of read English speech.

## Results

| Model | Engine | Bits | WER% | RTF | Model Load | Warmup |
|-------|--------|------|------|-----|------------|--------|
| Qwen3-ASR 0.6B | MLX (GPU) | 4-bit | 3.34 | 0.023 | 2.4s | 0.3s |
| Qwen3-ASR 0.6B | MLX (GPU) | 8-bit | 2.80 | 0.025 | 2.4s | 0.5s |
| Parakeet TDT 0.6B | CoreML (ANE) | INT4 | — | 0.295 | 23.3s | 2.4s |

Parakeet WER pending (full run in progress).

**Machine**: Apple M2 Max, 64 GB, macOS 14, release build with compiled metallib.

## Comparison with published models

| Model | Params | WER% (test-clean) | Backend | Source |
|-------|--------|--------------------|---------|--------|
| Whisper Large v3 Turbo | 809M | 2.5 | — | OpenAI (2024) |
| Whisper Large v3 | 1.5B | 2.7 | — | OpenAI (2023) |
| **Qwen3-ASR 0.6B 8-bit** | **600M** | **2.80** | **MLX** | **This benchmark** |
| Whisper Medium | 769M | 3.0 | — | OpenAI (2022) |
| **Qwen3-ASR 0.6B 4-bit** | **600M** | **3.34** | **MLX** | **This benchmark** |
| Whisper Small | 244M | 3.4 | — | OpenAI (2022) |
| FireRedASR2-AED | 1B | 4.57 | — | Xiaohongshu (2025) |
| Whisper Base | 74M | 5.0 | — | OpenAI (2022) |

Qwen3-ASR 0.6B at 8-bit matches Whisper Large v3 quality at 40% of the parameters. At 4-bit it matches Whisper Small/Medium.

## Compression delta

4-bit quantization adds 0.54% WER vs 8-bit (3.34% vs 2.80%). 8-bit is 16% better on error count. Model size: ~200 MB (8-bit) vs ~120 MB (4-bit).

## Error breakdown

| Model | Substitutions | Insertions | Deletions | Total errors | Words |
|-------|---------------|------------|-----------|-------------|-------|
| 0.6B 4-bit | 1323 | 123 | 308 | 1754 | 52,576 |
| 0.6B 8-bit | 1111 | 92 | 268 | 1471 | 52,576 |

## Reproduction

```bash
make build
python scripts/benchmark_asr.py --batch --engine qwen3 --model 0.6B
python scripts/benchmark_asr.py --batch --engine qwen3 --model 0.6B-8bit
python scripts/benchmark_asr.py --batch --engine parakeet
```

First run downloads LibriSpeech test-clean (~350 MB). Results saved to `benchmarks/librispeech/`.
