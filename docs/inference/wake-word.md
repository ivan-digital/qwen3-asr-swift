# Wake-word / keyword-spotting inference

The `SpeechWakeWord` module runs the KWS Zipformer (see
[model docs](../models/kws-zipformer.md)) as a streaming keyword spotter.
Registered keywords are boosted and thresholded; anything else is suppressed.

## Swift API

```swift
import SpeechWakeWord

let detector = try await WakeWordDetector.fromPretrained(
    keywords: [
        KeywordSpec(phrase: "hey soniqo", acThreshold: 0.15, boost: 0.5),
        KeywordSpec(phrase: "cancel")
    ]
)

// Streaming — push audio chunks from your capture source.
let session = try detector.createSession()
for chunk in audioStream {                         // Float32 @ 16 kHz
    for detection in try session.pushAudio(chunk) {
        print("[\(detection.time(frameShiftSeconds: 0.04))s] \(detection.phrase)")
    }
}
// Flush whatever's buffered when the stream ends:
for detection in try session.finalize() { print(detection.phrase) }

// Batch — one-shot detection over a full file.
let detections = try detector.detect(audio: samples, sampleRate: 16000)
```

`KeywordSpec` fields:

- `phrase` — display string, lowercased internally and BPE-tokenized against
  `bpe.model`. Multi-word phrases are matched as contiguous BPE sequences.
- `acThreshold` — mean acoustic probability over the matched span required to
  emit. `0` → use the tuned default (0.15).
- `boost` — per-token boost applied while the context-graph is matched. Positive
  values make the phrase easier to trigger; negative discourage it. `0` → use
  the tuned default (0.5).

`KeywordDetection` fields:

- `phrase` — the matching `KeywordSpec.phrase`.
- `tokenIds` / `timestamps` — BPE ids and their encoder-frame offsets inside
  the emission.
- `frameIndex` — encoder frame at which the emission fired (40 ms / frame).

## CLI

```bash
# Bare phrase (uses tuned defaults):
audio wake recording.wav --keywords "hey soniqo"

# Per-phrase tuning (phrase[:ac_threshold[:boost]]):
audio wake recording.wav --keywords "hey soniqo:0.1:0.5" "cancel:0.2"

# JSON output:
audio wake recording.wav --keywords "hey soniqo" --json

# Keyword file (one entry per line, `#` for comments):
audio wake recording.wav --keywords-file keywords.txt
```

## Streaming pipeline

```
mic PCM @ 16 kHz
  │
  ▼
KaldiFbank  (25/10 ms frames, 80 mel bins, no CMVN)
  │   one mel frame per 10 ms
  ▼
mel window (sliding, 45 frames in → 16 new frames per encoder step)
  │
  ▼
CoreML encoder  (+ 38-tensor state)  ─► 8 joiner-space frames
  │
  ▼
StreamingKwsDecoder
  ├── ContextGraph (Aho-Corasick trie over BPE ids)
  ├── Beam search (beam=4, blank-aware)
  └── Emission: `num_tailing_blanks > N` && `mean_ac_prob >= threshold`
```

The detector is **not thread-safe** — create one `WakeWordSession` per audio
source. Models can be safely shared between sessions (CoreML serialises their
calls internally).

## Pipeline integration

The module exposes `WakeWordProvider` to wire into `VoicePipeline` as an
activation gate — mirror the shape of `StreamingVADProvider` so the pipeline
can switch between VAD-only and VAD + wake-word gating without code changes.

```swift
let adapter = try WakeWordStreamingAdapter(detector: detector)
```

## Threshold tuning

The tuned defaults (0.15 / 0.5 / 1) are a good starting point on read speech.
For noisy conditions or far-field mics, raise `acThreshold` toward 0.2–0.3 to
cut false positives, or increase `boost` to 1.0–2.0 for better recall on
short phrases. Use `audio wake --json` to dump matched spans + timestamps
and iterate.

## Known limitations

- English only — gigaspeech fine-tune, no multilingual variant exported.
- Single-stream decoder — multiple concurrent streams need independent
  `WakeWordSession` instances (models themselves are shareable).
- Greedy BPE encoder — good enough for common phrases; fall back to the full
  SentencePiece encoder in the model bundle for unusual casings or unicode.
