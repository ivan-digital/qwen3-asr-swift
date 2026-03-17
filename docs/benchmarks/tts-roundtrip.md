# TTS Round-Trip WER Benchmark

## Method

Synthesize text → transcribe audio back (Qwen3-ASR 0.6B) → compute WER vs original text. Measures TTS intelligibility end-to-end.

## Results (30 built-in English sentences, conversational style)

| TTS Engine | Model | Params | Size | WER% | RTF | Cold Start | TTFT |
|------------|-------|--------|------|------|-----|------------|------|
| CosyVoice3 | 0.5B 4-bit | 500M | ~1.9 GB | 3.25 | 0.59 | 0.9s | — |
| Qwen3-TTS | 1.7B 4-bit | 1.7B | ~2.3 GB | 3.47 | 0.79 | 1.1s | 1ms |
| Qwen3-TTS | 1.7B 8-bit | 1.7B | ~3.5 GB | 3.66 | 0.85 | 1.1s | 1ms |
| Kokoro-82M | 82M CoreML | 82M | ~170 MB | 3.90 | 0.17 | 20.5s | ~450ms |
| Qwen3-TTS | 0.6B 8-bit | 600M | ~960 MB | 9.74 | 0.76 | 1.1s | 2ms |
| Qwen3-TTS | 0.6B 4-bit | 600M | ~675 MB | 15.58 | 0.76 | 1.1s | 3ms |

**Machine**: Apple M2 Max, 64 GB, macOS 14, release build.

On harder literary text (111 LibriSpeech sentences), Qwen3-TTS 0.6B 4-bit scores 19.15% WER — higher due to archaic vocabulary and long sentences.

## Key observations

- **CosyVoice3** achieves best intelligibility (3.25% WER) and fastest RTF (0.59)
- **Qwen3-TTS 1.7B** matches CosyVoice quality (3.47%) but is slower (RTF 0.79)
- **0.6B → 1.7B** dramatically improves quality (15.58% → 3.47% for 4-bit, 4.5x fewer errors)
- **4-bit → 8-bit** makes little difference for 1.7B (3.47% → 3.66%) but hurts 0.6B (15.58% → 9.74%)
- All engines are faster than real-time (RTF < 1.0)
- **Kokoro** has fastest RTF (0.17) but slowest cold start (20.5s CoreML compilation, cached after first use)
- **MLX models** (Qwen3, CosyVoice) load in ~1s — no cold start penalty
- **TTFT** (time to first token): Qwen3-TTS embed is 1-3ms, Kokoro ~450ms (CoreML prediction overhead)
- Generation dominates Qwen3-TTS latency at ~92% of total time (52-58ms/step)

## Latency breakdown (Qwen3-TTS)

| Stage | Time | % of total | Description |
|-------|------|-----------|-------------|
| Embed | 1-3ms | <1% | Text embedding (TTFT) |
| Generate | 2-6s | ~92% | Autoregressive codec tokens |
| Decode | 244-457ms | ~8% | Codec decoder → waveform |

## Common error types

- **Number compounding**: "twenty three" → "twentythree"
- **Word drops**: Determiners ("the", "a") occasionally dropped
- **Spelling variants**: "honour" → "honor", "colour" → "color"
- **Truncation**: Very long sentences cut short near max token limit

## Reproduction

```bash
make build
python scripts/benchmark_tts.py                            # Qwen3-TTS base (30 sentences)
python scripts/benchmark_tts.py --tts-engine cosyvoice     # CosyVoice3
python scripts/benchmark_tts.py --tts-model 1.7b           # Qwen3-TTS 1.7B
python scripts/benchmark_tts.py --compare                  # All engines
python scripts/benchmark_tts.py --input-file corpus.txt    # Custom corpus
```
