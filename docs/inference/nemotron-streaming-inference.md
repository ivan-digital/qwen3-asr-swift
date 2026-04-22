# Nemotron Speech Streaming — Inference Pipeline

Low-latency English streaming ASR using `NemotronStreamingASRModel`. Runs a cache-aware FastConformer encoder + RNN-T decoder on CoreML.

For architecture, see [nemotron-streaming.md](../models/nemotron-streaming.md).

## Quick start — batch

```swift
import NemotronStreamingASR

let model = try await NemotronStreamingASRModel.fromPretrained()
let text = try model.transcribeAudio(audioSamples, sampleRate: 16000)
```

Conforms to `SpeechRecognitionModel`, so it slots into any code path that takes a generic STT model (voice pipelines, diarization + ASR, etc.).

## Quick start — AsyncStream

```swift
for await partial in model.transcribeStream(audio: samples, sampleRate: 16000) {
    if partial.isFinal { print("FINAL: \(partial.text)") }
    else               { print("... \(partial.text)") }
}
```

`transcribeStream` spins up an internal `StreamingSession`, chunks input at session-native boundaries, and yields `PartialTranscript` values as the model emits them:

| Field | Meaning |
|---|---|
| `text` | Decoded text so far (SentencePiece, with inline punctuation + capitalization) |
| `isFinal` | `true` only on the last partial returned by `finalize()` |
| `confidence` | `exp(mean(logProb))` across all emitted tokens |
| `segmentIndex` | Monotonic session index (always 0 for now — Nemotron has no internal EOU boundary) |

## Long-lived session (mic input)

```swift
let session = try model.createSession()

// on each mic chunk:
let partials = try session.pushAudio(float32Chunk16kHz)
for p in partials {
    showPartial(p.text)  // isFinal is always false mid-stream
}

// when the stream ends:
let trailing = try session.finalize()
for p in trailing {
    commit(p.text)  // isFinal == true
}
```

### Chunk sizing

`pushAudio` buffers internally and runs the encoder once enough samples accumulate. The math for the published 160 ms bundle:

- `samplesPerChunk = melFrames * hopLength` — `17 * 160 = 2720` (~170 ms of input incl. pre-cache overlap)
- `shiftSamples = outputFrames * subsamplingFactor * hopLength` — `2 * 8 * 160 = 2560` (160 ms advance)

Push mic audio in arbitrary sizes; the session runs whenever enough samples have accumulated.

## End-of-utterance

Unlike Parakeet-EOU, Nemotron **does not** emit an explicit EOU token. Two options for utterance boundaries:

1. **External VAD**: use `SpeechVAD` (Silero) to detect silence and call `finalize()` to commit a sentence, then immediately `createSession()` again for the next utterance.
2. **Punctuation boundary**: since Nemotron emits `.`, `?`, `!` inline, you can treat a trailing sentence-ending punctuation in the partial text as a commit cue. Works without an extra VAD model but requires the model to actually emit terminal punctuation.

## Picking between Parakeet-EOU and Nemotron

| | Parakeet-EOU 120M | Nemotron 0.6B |
|---|---|---|
| Params | 120M | 600M |
| Encoder | 17-layer FastConformer, 512 hidden | 24-layer FastConformer, 1024 hidden |
| Decoder | 1-layer LSTM, RNN-T | 2-layer LSTM, RNN-T |
| EOU detection | Built-in `<EOU>` token | External (VAD or punctuation) |
| Punctuation | No (post-process) | Native (inline BPE tokens) |
| Bundle size | ~150 MB | ~580 MB |
| Latency | 320 ms chunk (default) | 160 ms chunk (default) |
| Use case | Constrained iOS, EOU-driven UX | Higher fidelity + punctuation, voice agents |

## CLI

```bash
audio transcribe recording.wav --engine nemotron           # batch transcription
audio transcribe recording.wav --engine nemotron --stream  # streaming (prints final)
audio transcribe recording.wav --engine nemotron --stream --partial  # prints partials too
```
