# Feature Plan: Live Real-Time Translation

## Goal

Add a “live mode” that listens to microphone audio, transcribes continuously, and emits near-real-time translated text in a target language (with incremental updates and low latency).

## Non-Goals (MVP)

- Perfect word-level timestamps/alignments.
- Speaker diarization.
- Full duplex conversation UI (we can add later).

## Constraints / Current Gaps

- `Qwen3Tokenizer.encode(_:)` is currently a placeholder (character-level) and is not suitable for building robust prompts or doing text-only generation/translation.
- “True” streaming (incremental audio tokens with stable decoder KV-cache while audio grows) is non-trivial; start with a pragmatic sliding-window approach.
- MLX/Metal runtime availability: support a clean failure mode if Metal is unavailable (CI/sandbox).

## Proposed UX

### CLI (macOS)

Add a new subcommand (or a separate executable entrypoint if desired):

```bash
swift run qwen3-asr-cli realtime --to en
swift run qwen3-asr-cli realtime --to ja --from auto --model mlx-community/Qwen3-ASR-0.6B-4bit
```

Flags:

- `--to <lang>`: target translation language (required).
- `--from <lang|auto>`: source language hint; default `auto`.
- `--model <hf_model_id>`: ASR model id.
- `--window-seconds <N>`: sliding audio window size, default 10–20s.
- `--step-ms <N>`: update interval, default 250–500ms.
- `--vad <on|off>`: simple silence detection; default `on`.
- `--device <gpu|cpu>`: best-effort; if Metal unavailable, exit with actionable error.

Output modes:

- `--format plain`: print incremental translation deltas.
- `--format jsonl`: emit structured events (`partial`, `final`, `metrics`).

### Library API

Add an API that yields incremental results (tokens or segments):

```swift
public struct RealtimeTranslationOptions: Sendable {
    public var targetLanguage: String
    public var sourceLanguage: String? // nil = auto
    public var windowSeconds: Double
    public var stepMs: Int
    public var enableVAD: Bool
}

public struct RealtimeTranslationEvent: Sendable {
    public enum Kind: Sendable { case partial, final }
    public var kind: Kind
    public var transcript: String
    public var translation: String
    public var isStable: Bool
}

public protocol AudioFrameSource: Sendable {
    func frames() -> AsyncStream<[Float]> // mono float samples at a declared sample rate
    var sampleRate: Int { get }
}

public extension Qwen3ASRModel {
    func realtimeTranslate(
        audio: AudioFrameSource,
        options: RealtimeTranslationOptions
    ) -> AsyncStream<RealtimeTranslationEvent>
}
```

## Architecture (Phased)

### Phase 0: Foundations (required before translation)

1. **Tokenizer: implement real encoding**
   - Parse and load the correct tokenizer artifacts from the model repo (likely `tokenizer.json` or merges/BPE files depending on Qwen3’s tokenizer format).
   - Replace placeholder `Qwen3Tokenizer.encode(_:)` with a correct implementation.
   - Add tests that round-trip known strings and validate specific token ids used by prompts.
   - Update model downloader to fetch required tokenizer files (beyond `vocab.json` / `tokenizer_config.json`) if needed.

2. **Text-only generation helper**
   - Add a `generateText(prompt:)` path that does not require audio embeddings.
   - Ensure it supports KV-cache and max token limits.

### Phase 1: Realtime Transcription MVP (sliding window)

1. **Microphone capture**
   - Implement `MicrophoneAudioSource` using `AVAudioEngine` (macOS first; iOS later).
   - Output small fixed-size frames (e.g. 20ms or 40ms) as `[Float]`.
   - Resample to the model’s expected rate (or let `WhisperFeatureExtractor.process` resample).

2. **Windowed ASR loop**
   - Maintain a ring buffer of the last `windowSeconds` of audio.
   - Every `stepMs`, run `transcribe()` on the current window.
   - Emit events:
     - `partial`: whenever the best-guess transcript changes.
     - `final`: when VAD indicates end-of-utterance (or when the tail segment stabilizes over N iterations).

3. **Stability heuristics**
   - Implement a “stabilizer” that compares successive transcripts and only finalizes stable prefixes.
   - Keep a “committed” transcript and a “live” suffix to reduce churn.

### Phase 2: Translation MVP (incremental, segment-based)

1. **Segment translation strategy**
   - Translate only finalized segments (from Phase 1) to avoid re-translating the entire window every tick.
   - For partial output, optionally translate the current live suffix at a reduced rate (e.g. every 1s).

2. **Translation engine**
   - Option A (preferred for on-device, minimal deps): use the existing Qwen3 text decoder weights for text-only translation via a prompt.
     - Prompt sketch: `Translate the following speech transcript to {targetLanguage}. Output only the translation.\n\n{transcript}`
     - Requires correct tokenizer encoding (Phase 0).
   - Option B: pluggable translator protocol to allow an external provider later.

3. **Rate limiting + caching**
   - Cache translations of finalized segments to avoid recomputation.
   - Debounce partial translations.

### Phase 3: True Streaming (optional, harder)

Replace sliding-window recomputation with genuine incremental decoding:

- Incremental feature extraction and audio encoder chunking.
- Maintain decoder KV-cache across audio growth.
- Emit tokens as they are generated (`AsyncStream<String>`).
- Requires careful alignment between audio token positions and prompt embedding replacement.

## Testing

- Unit tests:
  - Tokenizer encode/decode correctness for known fixtures.
  - “Stabilizer” behavior (prefix commit, churn reduction).
  - Translation prompt formatting and output post-processing.
- Integration tests (opt-in):
  - Realtime mode with a prerecorded WAV stream (simulate mic) to keep tests deterministic.
  - Skip by default unless an env var is set (avoid CI downloading huge weights).

## Performance Targets (initial)

- Update cadence: 250–500ms.
- End-to-end latency (speech to partial text): < 1.5s on an M-series Mac for the 0.6B 4-bit model.
- Translation latency for finalized segments: < 1–2s per short segment (best effort; depends on model + prompt).

## Deliverables Checklist

- [ ] Tokenizer: real `encode` + any missing tokenizer artifacts downloaded
- [ ] Text-only generation helper
- [ ] `MicrophoneAudioSource` (macOS)
- [ ] Sliding-window realtime transcription loop + stabilizer
- [ ] Segment-based translation pipeline + events API
- [ ] CLI `realtime` command + JSONL output mode
- [ ] Tests + docs updates (`README.md` + `AGENTS.md`)

