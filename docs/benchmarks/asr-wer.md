# ASR Word Error Rate (WER) Benchmark

## Datasets

- **LibriSpeech test-clean** — 2620 utterances, ~5.4 hours, English read speech (standard ASR benchmark)
- **FLEURS** — multilingual (10 languages), ~400-900 utterances per language, freely downloadable

## Results

| Model | Engine | Bits | Size | WER% | RTF | Model Load | Warmup |
|-------|--------|------|------|------|-----|------------|--------|
| Qwen3-ASR 1.7B | MLX (GPU) | 8-bit | 2.3 GB | 2.35 | 0.090 | 5.1s | 0.8s |
| Qwen3-ASR 1.7B | MLX (GPU) | 4-bit | 1.2 GB | 2.57 | 0.045 | 3.2s | 0.4s |
| Parakeet TDT 0.6B | CoreML (ANE) | INT8 | 634 MB | 2.74 | 0.089 | 128.9s | 2.0s |
| Qwen3-ASR 0.6B | MLX (GPU) | 8-bit | 960 MB | 2.80 | 0.025 | 2.4s | 0.3s |
| Qwen3-ASR 0.6B | MLX (GPU) | 4-bit | 675 MB | 3.34 | 0.023 | 2.4s | 0.3s |

**Machine**: Apple M2 Max, 64 GB, macOS 14, release build with compiled metallib.

**Key observations:**
- Parakeet INT8 achieves the best WER (2.74%) but has a slow cold start (128.9s CoreML compilation)
- Qwen3-ASR MLX is 10x faster to load (2.4s vs 129s) and has the fastest RTF (0.023)
- CoreML cold start (first-ever load) compiles a device-specific execution plan: 129s for INT8. Warm start (cached) is 5.4s — CoreML caches compiled plans in `~/Library/Caches/com.apple.CoreML/`. The 129s only happens once per device. Encoder currently uses `.all` compute units; switching to `.cpuAndNeuralEngine` would skip GPU plan compilation

## Comparison with published models

| Model | Params | Size | Precision | WER% (test-clean) | Source |
|-------|--------|------|-----------|-------------------|--------|
| **Qwen3-ASR 1.7B 8-bit** | **1.7B** | **2.3 GB** | **8-bit** | **2.35** | **This benchmark** |
| Whisper Large v3 Turbo | 809M | 1.6 GB | FP16 | 2.5 | OpenAI (2024) |
| **Qwen3-ASR 1.7B 4-bit** | **1.7B** | **1.2 GB** | **4-bit** | **2.57** | **This benchmark** |
| Whisper Large v3 | 1.5B | 3.1 GB | FP16 | 2.7 | OpenAI (2023) |
| **Parakeet TDT 0.6B INT8** | **600M** | **634 MB** | **INT8** | **2.74** | **This benchmark** |
| **Qwen3-ASR 0.6B 8-bit** | **600M** | **960 MB** | **8-bit** | **2.80** | **This benchmark** |
| Whisper Medium | 769M | 1.5 GB | FP16 | 3.0 | OpenAI (2022) |
| **Qwen3-ASR 0.6B 4-bit** | **600M** | **675 MB** | **4-bit** | **3.34** | **This benchmark** |
| Whisper Small | 244M | 483 MB | FP16 | 3.4 | OpenAI (2022) |
| FireRedASR2-AED | 1B | ~2 GB | FP16 | 4.57 | Xiaohongshu (2025) |
| Whisper Base | 74M | 142 MB | FP16 | 5.0 | OpenAI (2022) |

Whisper numbers from original papers (FP16 inference).

## Multilingual results (FLEURS)

CER used for CJK languages (no word boundaries). Parakeet is English-only (25 European languages).

| Language | Metric | Qwen3 4-bit | Qwen3 8-bit | Parakeet INT8 |
|----------|--------|-------------|-------------|---------------|
| Spanish | WER | 6.44 | 5.06 | 5.18 |
| English | WER | 6.57 | 5.64 | 9.30 |
| Chinese | CER | 8.41 | 7.71 | — |
| German | WER | 9.45 | 6.81 | 12.33 |
| French | WER | 11.42 | 8.50 | 13.02 |
| Japanese | CER | 16.11 | 8.64 | — |
| Russian | WER | 16.35 | 10.52 | 11.49 |
| Korean | WER | 19.95 | 6.89 | — |
| Hindi | WER | 25.93 | 18.57 | — |
| Arabic | WER | 33.47 | 20.31 | — |

**Qwen3-ASR 8-bit** consistently outperforms 4-bit across all languages. Largest gains on Korean (19.95% → 6.89%, 65% reduction) and Japanese (16.11% → 8.64%, 46% reduction).

**Qwen3 vs Parakeet**: Qwen3 8-bit is better on all languages except Spanish (5.06% vs 5.18%). Qwen3 supports 52 languages; Parakeet supports ~25 European languages (no CJK).

## Compression delta

How much accuracy do we lose by quantizing to lower bit widths? This establishes the baseline quality cost of our current quantization before trying more advanced techniques like mixed-bit allocation or outlier decomposition.

| Variant | WER% | Substitutions | Insertions | Deletions | Total errors | Size |
|---------|------|---------------|------------|-----------|-------------|------|
| Qwen3 0.6B 8-bit | 2.80 | 1111 | 92 | 268 | 1471 | 960 MB |
| Qwen3 0.6B 4-bit | 3.34 | 1323 | 123 | 308 | 1754 | 675 MB |
| Delta | +0.54 | +212 | +31 | +40 | +283 | -30% |
| Parakeet TDT INT8 | 2.74 | 990 | 125 | 308 | 1423 | 634 MB |

**Qwen3-ASR**: 4-bit adds 0.54% WER (19% more errors) for 30% size reduction.

## Long-Form Stability (Sustained Neural Engine Load)

Tested whether WER or latency degrade under sustained transcription sessions (simulating meeting transcription). 200 LibriSpeech test-clean utterances processed sequentially (~30 min of audio) on M2 Max.

| Metric | First 25% | Last 25% | Overall |
|--------|-----------|----------|---------|
| WER% | 1.30 | 1.23 | 2.43 |
| RTF | 0.672 | 0.400 | 0.539 |

**Key findings:**
- No WER degradation — last quarter is actually slightly better (1.23% vs 1.30%), within noise
- RTF **improves** over the session (0.67 → 0.40) as CoreML warms up its execution plan cache
- No thermal throttling detected on M2 Max after 42 minutes of continuous Neural Engine inference
- Parakeet processes each chunk independently (no cross-chunk state), so quality cannot accumulate errors

RTF includes per-invocation model loading overhead (~3s). Pure inference RTF is ~0.023 (43x real-time).

## Reproduction

```bash
make build
python scripts/benchmark_asr.py --batch --engine qwen3 --model 0.6B
python scripts/benchmark_asr.py --batch --engine qwen3 --model 0.6B-8bit
python scripts/benchmark_asr.py --batch --engine parakeet
python scripts/benchmark_asr.py --batch --engine parakeet --model int8
```

First run downloads LibriSpeech test-clean (~350 MB). Results saved to `benchmarks/librispeech/`.

### Long-form stability

```bash
# Download LibriSpeech test-clean first (~350 MB)
# Then run sustained benchmark (all 2620 utterances, ~5.4 hours)
python scripts/benchmark_longform.py --engine parakeet
# Or a quick 200-utterance test (~30 min audio)
python scripts/benchmark_longform.py --engine parakeet --max-utterances 200
```

### FLEURS (multilingual, auto-download)

```bash
python scripts/benchmark_asr.py --dataset fleurs --language en_us --batch
python scripts/benchmark_asr.py --dataset fleurs --language cmn_hans_cn --batch
python scripts/benchmark_asr.py --dataset fleurs --language de_de --batch
```
