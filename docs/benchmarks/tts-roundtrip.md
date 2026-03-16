# TTS Round-Trip WER Benchmark

## Method

Synthesize text → transcribe audio back → compute WER vs original text. Measures TTS intelligibility end-to-end.

## Results

| Corpus | Sentences | WER% | TTS RTF | ms/step | Embed | Generate | Decode |
|--------|-----------|------|---------|---------|-------|----------|--------|
| Built-in (conversational) | 30 | 2.27 | 0.56 | 38 | 2ms | 2526ms | 211ms |
| LibriSpeech transcripts | 47/50 | 18.55 | 0.57 | 40 | 2ms | 6353ms | 469ms |

**Machine**: Apple M2 Max, 64 GB, macOS 14, release build.

**Observations:**
- Built-in corpus (short, conversational): 2.27% WER — 25/30 perfect round-trip
- LibriSpeech transcripts (long, literary): 18.55% WER — archaic vocabulary and long sentences increase errors
- Generation dominates at ~92% of total time (38-40ms/step)
- RTF ~0.57 (faster than real-time) regardless of input complexity

## Latency breakdown

| Stage | Time | Description |
|-------|------|-------------|
| Embed | 2ms | Text embedding preparation |
| Generate | 2.5-6.4s | Autoregressive codec token generation |
| Decode | 0.2-0.5s | Codec decoder → waveform |

Longer sentences produce more tokens → longer generation time, but ms/step is constant (~39ms).

## Error analysis

Common error types:
- **Number compounding**: "twenty three" → "twentythree"
- **Word drops**: Determiners ("the", "a") occasionally dropped
- **Archaic words**: "honour" → "honor", "connexion" → "connection"
- **Truncation**: Very long sentences cut short near max token limit (2 failures in 50)

## Reproduction

```bash
make build
python scripts/benchmark_tts.py                           # Built-in 30 sentences
python scripts/benchmark_tts.py --input-file corpus.txt   # Custom corpus
python scripts/benchmark_tts.py --compare                 # All TTS engines
```
