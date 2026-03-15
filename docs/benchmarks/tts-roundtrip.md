# TTS Round-Trip WER Benchmark

## Method

Synthesize text → transcribe audio back → compute WER vs original text. Measures TTS intelligibility end-to-end.

## Results (Qwen3-TTS base, 30 sentences)

| Metric | Value |
|--------|-------|
| Round-trip WER | 2.27% |
| Sentences | 30/30 (0 failed) |
| Mean TTS RTF | 0.56 |
| Mean embed | 2ms |
| Mean generate | 2526ms (38ms/step) |
| Mean decode | 211ms |

**Machine**: Apple M2 Max, 64 GB, macOS 14, release build.

## Latency breakdown

| Stage | Time | Description |
|-------|------|-------------|
| Embed | 2ms | Text embedding preparation |
| Generate | 2526ms | Autoregressive codec token generation |
| Decode | 211ms | Codec decoder → waveform |
| Total | ~2.7s | For ~4.8s of audio (RTF 0.56) |

Generation dominates at 92% of total time (38ms/step). Decode is fast (211ms for full waveform).

## Error analysis

5/30 sentences had errors (WER > 0):

- **Number compounding** (2 cases): "twenty three" → "twentythree", "forty five" → "fortyfive". TTS joins compound numbers into single words.
- **Word drops** (2 cases): Missing "the" or "a" — minor determiners dropped during synthesis.
- **Truncation** (1 case): "will be shipped tomorrow" → "will be shipped" — sentence cut short near max token limit.

25/30 sentences had perfect round-trip (0% WER).

## Reproduction

```bash
make build
python scripts/benchmark_tts.py                           # Qwen3-TTS (30 sentences)
python scripts/benchmark_tts.py --compare                 # All TTS engines
python scripts/benchmark_tts.py --input-file sentences.txt # Custom corpus
```
