# Nemotron Speech Streaming 0.6B Architecture

NVIDIA Nemotron Speech Streaming is a cache-aware streaming ASR model optimized for low-latency voice agents. Native punctuation and capitalization are emitted as regular BPE tokens — there is no separate EOU/EOB head; the caller signals end of stream via `finalize()`.

## Overview

- **Parameters**: 600M (INT8-palettized encoder)
- **Backend**: CoreML (Neural Engine / GPU)
- **Architecture**: Cache-aware FastConformer encoder + RNN-T decoder
- **Languages**: English only
- **Input**: 16 kHz mono PCM
- **Chunk sizes**: 80 ms, 160 ms, 560 ms, 1120 ms (released 160 ms variant first)
- **License**: Check NVIDIA model card (derivative CoreML bundle at `aufklarer/Nemotron-Speech-Streaming-0.6B-CoreML-INT8`)

## Pipeline

```
audio (16 kHz) → Mel (128 bins, 10 ms hop)
                   ↓
  Cache-aware FastConformer encoder (24 layers, 1024 hidden)
    • Left attention context: 70 frames
    • Right context: selects chunk size (0/1/6/13 = 80/160/560/1120 ms)
    • Subsampling: 8x (80 ms output stride)
                   ↓
  RNN-T:
    • Prediction net (2-layer LSTM, 640 hidden)
    • Joint network — 1025 outputs (1024 BPE + blank)
                   ↓
  Greedy decode with SentencePiece vocab (punctuation inline)
```

## Streaming Model I/O

### Encoder

**Inputs:**

| Name | Shape | Type | Description |
|---|---|---|---|
| `audio_signal` | `[1, 128, melFrames]` | Float32 | Mel spectrogram for the current chunk |
| `audio_length` | `[1]` | Int32 | Valid mel frame count (typically `melFrames`) |
| `pre_cache` | `[1, 128, preCacheSize]` | Float32 | Rolling mel context from the previous chunk |
| `cache_last_channel` | `[24, 1, 70, 1024]` | Float32 | Attention cache across all encoder layers |
| `cache_last_time` | `[24, 1, 1024, 8]` | Float32 | Depthwise conv cache |
| `cache_last_channel_len` | `[1]` | Int32 | Valid attention cache length |

**Outputs:**

| Name | Shape | Type | Description |
|---|---|---|---|
| `encoded_output` | `[1, T, 1024]` | Float16 | Encoded frames in `[B, T, D]` layout |
| `encoded_length` | `[1]` | Int32 | Valid encoded frame count |
| `new_pre_cache` | `[1, 128, preCacheSize]` | Float32 | Rolling mel context for next chunk |
| `new_cache_last_channel` | `[24, 1, 70, 1024]` | Float32 | Updated attention cache |
| `new_cache_last_time` | `[24, 1, 1024, 8]` | Float32 | Updated conv cache |
| `new_cache_last_channel_len` | `[1]` | Int32 | Updated attention cache length |

### Decoder (prediction network)

**Inputs:** `token [1, 1] Int32`, `h [2, 1, 640] Float16`, `c [2, 1, 640] Float16`
**Outputs:** `decoder_output [1, 1, 640] Float16`, `h_out [2, 1, 640] Float16`, `c_out [2, 1, 640] Float16`

### Joint

**Inputs:** `encoder_output [1, 1, 1024] Float16`, `decoder_output [1, 1, 640] Float16`
**Output:** `logits [1, 1, 1025] Float16`

## Chunk configuration

| `chunkMs` | `chunkSize` (80 ms frames) | `rightContext` | `melFrames` | `preCacheSize` | `outputFrames` |
|---|---|---|---|---|---|
| 80   | 1  | 0  | 9   | 16 | 1  |
| 160  | 2  | 1  | 17  | 16 | 2  |
| 560  | 7  | 6  | 64  | 9  | 7  |
| 1120 | 14 | 13 | 121 | 9  | 14 |

The published HuggingFace bundle targets **160 ms** by default (best low-latency / quality balance). The other three chunk sizes (80, 560, 1120 ms) can also be converted from the upstream NeMo checkpoint; the published repo just pins the 160 ms variant.

## Weight Files

| Component | Size | Format |
|---|---|---|
| Encoder | 562 MB | INT8-palettized `.mlmodelc` |
| Decoder | 14 MB | Float16 `.mlmodelc` |
| Joint | 3.3 MB | Float16 `.mlmodelc` |
| Vocab + config | ~20 KB | JSON |
| **Total** | **~580 MB** | |

Weights: [aufklarer/Nemotron-Speech-Streaming-0.6B-CoreML-INT8](https://huggingface.co/aufklarer/Nemotron-Speech-Streaming-0.6B-CoreML-INT8)
Upstream source: [nvidia/nemotron-speech-streaming-en-0.6b](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b)

## Source Files

```
Sources/NemotronStreamingASR/
  Configuration.swift                   NemotronStreamingConfig + .default (160 ms)
  StreamingMelPreprocessor.swift        NeMo mel (symmetric Hann, zero-center, raw log)
  RNNTGreedyDecoder.swift               Blank advances, non-blank emits (no EOU handling)
  StreamingSession.swift                Chunk buffer, encoder cache, finalize()
  Vocabulary.swift                      SentencePiece decode with inline punctuation
  NemotronStreamingASR.swift            fromPretrained / createSession / transcribeStream
  NemotronStreamingASR+Memory.swift     ModelMemoryManageable conformance
  NemotronStreamingASR+Protocols.swift  SpeechRecognitionModel conformance
```

