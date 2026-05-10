# ASR Inference Pipeline ([Qwen3-ASR](https://arxiv.org/abs/2601.21337))

## Overview

```
Audio (16kHz) → [Preprocessing] → [Audio Encoding] → [Text Generation] → Transcription
                 ~5% time          ~20% time           ~75% time
```

## Stage 1: Preprocessing (AudioPreprocessing.swift)

Converts raw audio to a mel spectrogram `[128, T]` using Accelerate framework.

- STFT via `vDSP_fft_zrip` (in-place real FFT, zero-padded 400→512)
- Mel filterbank via `vDSP_mmul`, bin frequencies use padded FFT size (`k * fs / 512`)
- Log-mel via `vDSP_vclip` + `vvlog10f` (vForce vectorized)
- Hann window and FFT setup precomputed once in `init()`
- All temporary buffers preallocated outside the frame loop

## Stage 2: Audio Encoding (AudioEncoder.swift)

18-layer transformer with block attention over chunked mel features.

- Self-attention via `MLXFast.scaledDotProductAttention` (Metal kernel)
- Sinusoidal position embeddings cached by sequence length
- Block attention mask via MLXArray broadcast (`blockIds .== blockIds^T`)

## Stage 3: Text Generation (QuantizedTextDecoder.swift)

28-layer quantized Qwen3 decoder with GQA and RoPE.

- RoPE via `MLXFast.RoPE` (fused Metal kernel)
- GQA via `MLXFast.scaledDotProductAttention` (native GQA support, no manual tiling)
- Causal mask: `nil` for autoregressive steps (seqLen=1), broadcast for prefill
- **Prefill** (seqLen > 1): all prompt tokens in one forward pass
- **Decode** (seqLen = 1): SDPA uses optimized T_q=1 Metal kernel
- **Greedy fast path** (default options): decode loop is double-buffered via `MLX.asyncEval` — step N+1's forward pass is queued *before* step N's token syncs to CPU, so host-side EOS check and append overlap with GPU work. Bit-identical to the legacy loop (snapshot-tested); ~5 % faster on long-form audio (1.7B-8bit, 71 s clip)
- **Batched greedy decode** (`transcribeBatch` / `audio transcribe-batch --batch-size N`): the default batch path keeps per-row decoder forwards serial for correctness, but syncs the whole `[B, 1]` token tensor once per decode step instead of doing B scalar `.item()` calls. On repeated 10s chunks with the 0.6B 4-bit model, batch size 6 improved aggregate inference time from 3.95s to 3.55s for 24 chunks (+11.3%). The true `[B,1,H]` MLX decoder forward remains gated behind `QWEN3_ASR_EXPERIMENTAL_BATCH_DECODE=1` until row-level bit-exactness is resolved.

## Decoder Options

`transcribe(audio:sampleRate:options:)` accepts a `Qwen3DecodingOptions`
struct that exposes the HuggingFace-style decoding knobs:

| Field | Default | Notes |
|---|---|---|
| `maxTokens` | `448` | Cap on decoder output per chunk. |
| `language` | `nil` | Hint; `nil` → auto-detect. |
| `context` | `nil` | Prefix prepended to the decoder prompt. |
| `repetitionPenalty` | `1.0` | HF divisor; `1.1`–`1.3` typical. Positive logits divide, negative logits multiply — matches the HF sign-aware branch so the penalty always reduces the probability of the already-generated token. |
| `noRepeatNgramSize` | `0` | Masks tokens that would form a repeated n-gram of this size. `0` disables. |
| `temperature` | `0.0` | `0` = greedy (argmax). `> 0` = sample via Gumbel-max (`argmax(logits/T + Gumbel(0,1)) ~ softmax(logits/T)`). |

All defaults take the asyncEval greedy fast path described above; any
non-default option falls back to a per-token-sync loop because the
sampler pulls full logits to CPU and defeats the overlap. Output is
byte-identical to the legacy loop in either mode.

The canonical defence against "percent percent percent..." loops on silence
or ambiguous audio is `repetitionPenalty = 1.15`:

```swift
let text = model.transcribe(
    audio: samples,
    sampleRate: 16000,
    options: Qwen3DecodingOptions(repetitionPenalty: 1.15)
)
```

The legacy overload `transcribe(audio:sampleRate:language:maxTokens:context:)`
remains available and forwards into the new path with default options.

## Batched Chunk Decode

`transcribeBatch(audios:sampleRate:language:maxTokens:context:options:)`
adds a throughput-oriented API for offline ASR workloads that already split
long recordings into fixed-duration chunks:

```swift
let texts = model.transcribeBatch(
    audios: chunks,
    sampleRate: 24000,
    language: "en"
)
```

By default this API preserves the same per-chunk greedy semantics as
`transcribe(audio:)`, while sharing one CPU token synchronization across the
batch at each decode step.

Non-greedy decoding options always fall back to serial per-chunk
`transcribe(audio:)` calls because repetition penalty, n-gram masking, and
temperature sampling each need CPU-side logits at every decode step that the
batched path doesn't surface. "Non-greedy" means any of:

- `repetitionPenalty != 1.0`
- `noRepeatNgramSize > 0`
- `temperature > 0.0`

The gate is `Qwen3ASRModel.isGreedyFastPath`, locked in by unit tests in
`Tests/Qwen3ASRTests/Qwen3ASRBatchedDecodeTests.swift` (one per non-greedy
flag) and an E2E `testBatchFallsBackToSerialForNonGreedyOptions` that asserts
batched output equals per-chunk serial output when non-greedy options are set.

The experimental MLX decoder batch can be enabled with
`QWEN3_ASR_EXPERIMENTAL_BATCH_DECODE=1` while developing the true
batched-GEMM scheduler. It currently truncates row 1 at batch size 2 on
repeated identical chunks; that bug is locked in by
`E2EQwen3ASRExperimentalBatchedDecodeTests` so the gate is a hard constraint,
not a soft warning.

The current experimental true-batch path batches only equal-length encoder
outputs. Mixed-duration inputs are grouped by encoder sequence length, decoded
bucket by bucket, and returned in the caller's original order. This matches
fixed-duration chunking and avoids silently leaving padding keys in the KV
cache during decode.

CLI usage:

```bash
audio transcribe-batch ./chunks --engine qwen3 --model 1.7B-8bit --batch-size 4
```

The JSONL output exposes both per-row and per-batch timings:

- `time` / `rtf` — per-row inference time and RTF. The default path runs
  decoder forwards serially per row, so this is the group elapsed split
  evenly across rows in the batch. It is an attribution, not a measured
  per-row wall-clock.
- `batch_time` / `batch_rtf` — true wall-clock for the whole group; use
  these when computing aggregate throughput.

Experimental batched decoder run:

```bash
QWEN3_ASR_EXPERIMENTAL_BATCH_DECODE=1 \
  audio transcribe-batch ./chunks --engine qwen3 --model 1.7B-8bit --batch-size 4
```

For mixed speech pipelines, batch Qwen3-ASR within MLX rather than trying to
run multiple MLX models concurrently in one process. MLX uses a single GPU
dispatch queue; CoreML-backed models can run on the Neural Engine in parallel,
but this Qwen3-ASR decoder optimization is specifically an MLX/GPU path.

## Performance

| Model | Framework | RTF | 10s audio processed in |
|-------|-----------|-----|------------------------|
| Qwen3-ASR-0.6B (4-bit) | MLX Swift | ~0.06 | ~0.6s |
| Whisper-large-v3 | whisper.cpp (Q5_0) | ~0.10 | ~1.0s |
| Whisper-small | whisper.cpp (Q5_0) | ~0.04 | ~0.4s |

## Streaming / Partial Transcription

Qwen3-ASR operates in batch mode only. The entire audio input is processed in a single forward pass — there is no streaming or partial transcription support. The audio encoder uses block attention over the full mel spectrogram, and the text decoder generates tokens autoregressively conditioned on the complete encoder output.

For real-time transcription use cases where partial results are needed, consider chunking the audio externally and running separate inference passes on each chunk.

## Language Detection

The model automatically detects the spoken language from the audio content. No language hint or locale parameter is required. The text decoder emits a language token at the start of generation, followed by the transcribed text. Supported languages include English, Chinese, Japanese, Korean, and many European languages.

## Model Architecture Reference

See [docs/models/asr-model.md](../models/asr-model.md) for detailed architecture documentation including layer dimensions, weight formats, and quantization details.
