---
name: benchmark
description: Run performance benchmarks. Arguments: asr, tts, vad, diarize (with optional --num-files N).
disable-model-invocation: true
argument-hint: [asr|tts|vad|diarize]
allowed-tools: Bash
---

# Benchmark

Run benchmarks using the release build. Build first with `/build`.

## Usage

- `/benchmark asr` — transcribe test audio, report RTF
- `/benchmark tts` — synthesize test text, report RTF
- `/benchmark vad` — VAD on VoxConverse (all engines)
- `/benchmark diarize` — DER on VoxConverse (requires downloaded test set)

```bash
module="$ARGUMENTS"
cli=".build/release/audio"

case "$module" in
  asr)
    $cli transcribe Tests/Qwen3ASRTests/Resources/test_audio.wav 2>&1
    ;;
  tts)
    $cli speak "The quick brown fox jumps over the lazy dog." --output /tmp/bench_tts.wav 2>&1
    ;;
  vad)
    python3 scripts/benchmark_vad.py --compare --num-files 5 2>&1
    ;;
  diarize)
    python3 scripts/benchmark_diarization.py --num-files 5 2>&1
    ;;
  *)
    echo "Usage: /benchmark [asr|tts|vad|diarize]"
    ;;
esac
```

## Performance targets (M2 Max)

| Module | Metric | Target |
|--------|--------|--------|
| ASR (Qwen3) | RTF | ~0.06 |
| ASR (Parakeet) | RTF | ~0.025 |
| TTS | RTF | ~0.7 |
| VAD (Silero) | RTF | >20x real-time |
| Diarization | DER | <10% (VoxConverse) |
