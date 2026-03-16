# Shared Protocols: Model-Agnostic Interfaces

## Overview

The `AudioCommon` module defines shared protocols that provide model-agnostic interfaces for speech processing. These allow generic code to work with any conforming model without knowing its concrete type.

```
┌─────────────────────────────────────────────────────────┐
│                    AudioCommon                          │
│                                                         │
│  AudioChunk          SpeechGenerationModel (TTS)        │
│  AlignedWord         SpeechRecognitionModel (STT)       │
│  SpeechSegment       ForcedAlignmentModel                │
│  TranscriptionResult SpeechToSpeechModel                 │
│                      VoiceActivityDetectionModel (VAD)   │
│                      StreamingVADProvider (pipeline)      │
│                      SpeakerEmbeddingModel               │
│                      SpeakerDiarizationModel             │
│                      SpeakerExtractionCapable            │
│                      SpeechEnhancementModel              │
└─────────────────────────────────────────────────────────┘
        ▲                    ▲                    ▲
        │                    │                    │
   ┌────┴────┐        ┌─────┴─────┐       ┌─────┴─────┐       ┌─────┴─────┐
   │Qwen3TTS │        │  Qwen3ASR │       │PersonaPlex │       │ SpeechVAD │
   │CosyVoice│        │ParakeetASR│       └───────────┘       └───────────┘
   │Kokoro   │        │ForcedAlign│
   └─────────┘        └───────────┘
```

## Protocols

### SpeechGenerationModel (TTS)

Text-to-speech models that generate audio from text.

```swift
public protocol SpeechGenerationModel: AnyObject {
    var sampleRate: Int { get }
    func generate(text: String, language: String?) async throws -> [Float]
    func generateStream(text: String, language: String?) -> AsyncThrowingStream<AudioChunk, Error>
}
```

**Conforming types:** `Qwen3TTSModel`, `CosyVoiceTTSModel`, `KokoroTTSModel`

### SpeechRecognitionModel (STT)

Speech-to-text models that transcribe audio.

```swift
public protocol SpeechRecognitionModel: AnyObject {
    var inputSampleRate: Int { get }
    func transcribe(audio: [Float], sampleRate: Int, language: String?) -> String
    func transcribeWithLanguage(audio: [Float], sampleRate: Int, language: String?) -> TranscriptionResult
}
```

The `transcribeWithLanguage` method has a default implementation that delegates to `transcribe()` with no language detection. Models that detect language (e.g. ParakeetASR via `NLLanguageRecognizer`) override it to return `TranscriptionResult` with a detected language — used by the voice pipeline to forward language to TTS.

**Conforming types:** `Qwen3ASRModel`, `ParakeetASRModel`

### ForcedAlignmentModel

Models that align text to audio at the word level.

```swift
public protocol ForcedAlignmentModel: AnyObject {
    func align(audio: [Float], text: String, sampleRate: Int, language: String?) -> [AlignedWord]
}
```

**Conforming types:** `Qwen3ForcedAligner`

### SpeechToSpeechModel

Speech-to-speech models that generate spoken responses from spoken input.

```swift
public protocol SpeechToSpeechModel: AnyObject {
    var sampleRate: Int { get }
    func respond(userAudio: [Float]) -> [Float]
    func respondStream(userAudio: [Float]) -> AsyncThrowingStream<AudioChunk, Error>
}
```

**Conforming types:** `PersonaPlexModel`

### VoiceActivityDetectionModel (VAD)

Models that detect speech activity regions in audio.

```swift
public protocol VoiceActivityDetectionModel: AnyObject {
    var inputSampleRate: Int { get }
    func detectSpeech(audio: [Float], sampleRate: Int) -> [SpeechSegment]
}
```

**Conforming types:** `PyannoteVADModel`, `SileroVADModel`

### StreamingVADProvider (Pipeline)

Streaming VAD that processes fixed-size audio chunks and returns speech probability. Used by the `SpeechCore` voice pipeline via the C vtable FFI.

```swift
public protocol StreamingVADProvider: AnyObject {
    var inputSampleRate: Int { get }
    var chunkSize: Int { get }
    func processChunk(_ samples: [Float]) -> Float
    func resetState()
}
```

**Conforming types:** `SileroVADModel`

### SpeakerEmbeddingModel

Models that extract speaker embeddings from audio.

```swift
public protocol SpeakerEmbeddingModel: AnyObject {
    var inputSampleRate: Int { get }
    var embeddingDimension: Int { get }
    func embed(audio: [Float], sampleRate: Int) -> [Float]
}
```

**Conforming types:** `WeSpeakerModel`

### SpeakerDiarizationModel

Models that assign speaker identities to speech segments.

```swift
public protocol SpeakerDiarizationModel: AnyObject {
    var inputSampleRate: Int { get }
    func diarize(audio: [Float], sampleRate: Int) -> [DiarizedSegment]
}
```

**Conforming types:** `PyannoteDiarizationPipeline` (aliased as `DiarizationPipeline`), `SortformerDiarizer`

### SpeakerExtractionCapable

Extended diarization protocol for engines that support extracting a target speaker's segments using a reference embedding. Not all engines support this — Sortformer is end-to-end and does not produce speaker embeddings.

```swift
public protocol SpeakerExtractionCapable: SpeakerDiarizationModel {
    func extractSpeaker(audio: [Float], sampleRate: Int, targetEmbedding: [Float]) -> [SpeechSegment]
}
```

**Conforming types:** `PyannoteDiarizationPipeline`

### SpeechEnhancementModel

Models that enhance speech by removing noise.

```swift
public protocol SpeechEnhancementModel: AnyObject {
    var inputSampleRate: Int { get }
    func enhance(audio: [Float], sampleRate: Int) throws -> [Float]
}
```

**Conforming types:** `DeepFilterNet3Model`

## Voice Pipeline (SpeechCore)

The `SpeechCore` module provides `VoicePipeline` — a real-time voice agent pipeline powered by [speech-core](https://github.com/soniqo/speech-core) (C++ engine, distributed as xcframework). It connects `SpeechRecognitionModel`, `SpeechGenerationModel`, and `StreamingVADProvider` through a state machine with VAD-driven turn detection, interruption handling, and eager STT.

```swift
import SpeechCore

let pipeline = VoicePipeline(
    stt: parakeetASR,
    tts: qwen3TTS,
    vad: sileroVAD,
    config: .init(mode: .echo),
    onEvent: { event in print(event) }
)
pipeline.start()
pipeline.pushAudio(micSamples)  // feed mic audio continuously
```

### Pipeline Modes

| Mode | Flow | Use case |
|------|------|----------|
| **voicePipeline** | audio → VAD → STT → LLM → TTS → audio | Full voice agent |
| **echo** | audio → VAD → STT → TTS → audio | Testing (speaks back transcription) |
| **transcribeOnly** | audio → VAD → STT → text | Transcription only |

### Configuration

```swift
var config = PipelineConfig()
config.mode = .echo
config.minSilenceDuration = 0.6     // seconds to confirm end of speech
config.eagerSTT = true              // start STT before silence confirms
config.eagerSTTDelay = 0.3          // seconds in silence before eager fires
config.allowInterruptions = true    // user can barge-in during playback
config.minInterruptionDuration = 1.0 // seconds of speech to confirm barge-in
config.maxResponseDuration = 5.0    // cap TTS output (prevents hallucination)
config.postPlaybackGuard = 0.3      // suppress VAD after playback (AEC settle)
config.warmupSTT = true             // warm up Neural Engine at pipeline start
```

### Events

| Event | When |
|-------|------|
| `speechStarted` | VAD confirms user speech |
| `speechEnded` | User utterance finalized |
| `transcriptionCompleted` | STT returns text + language + confidence |
| `responseCreated` | TTS synthesis starting |
| `responseAudioDelta` | TTS audio chunk ready (PCM Float32) |
| `responseInterrupted` | User barged in during playback |
| `responseDone` | TTS synthesis complete |
| `error` | STT/LLM/TTS failure |

## Shared Types

### AudioChunk

Unified audio chunk type returned by all streaming methods:

```swift
public struct AudioChunk: Sendable {
    public let samples: [Float]    // PCM audio samples
    public let sampleRate: Int     // Hz (e.g. 24000)
    public let frameIndex: Int     // First frame index in this chunk
    public let isFinal: Bool       // Last chunk flag
    public let elapsedTime: Double? // Wall-clock seconds (nil if not tracked)
    public let textTokens: [Int32] // Text tokens for this chunk (PersonaPlex streaming)
}
```

**Note on `textTokens`**: In `PersonaPlexModel.respondStream()`, each non-final chunk contains the text tokens generated during that chunk. The final chunk contains all text tokens from the entire generation. For non-PersonaPlex streams, this field defaults to empty.

### TranscriptionResult

Result of speech recognition including detected language:

```swift
public struct TranscriptionResult: Sendable {
    public let text: String
    public let language: String?  // e.g. "english", "russian"
}
```

### SpeechSegment

Time segment where speech was detected, returned by `VoiceActivityDetectionModel`:

```swift
public struct SpeechSegment: Sendable {
    public let startTime: Float    // seconds
    public let endTime: Float      // seconds
    public var duration: Float     // computed: endTime - startTime
}
```

### AlignedWord

Word with timestamps, returned by `ForcedAlignmentModel`:

```swift
public struct AlignedWord: Sendable {
    public let text: String
    public let startTime: Float    // seconds
    public let endTime: Float      // seconds
}
```

### DiarizedSegment

Speech segment with speaker identity, returned by `SpeakerDiarizationModel`:

```swift
public struct DiarizedSegment: Sendable {
    public let startTime: Float    // seconds
    public let endTime: Float      // seconds
    public let speakerId: Int      // 0-based speaker identifier
    public var duration: Float     // computed: endTime - startTime
}
```

## Usage

### Generic TTS Function

```swift
import AudioCommon

func synthesizeAny(
    _ model: any SpeechGenerationModel,
    text: String,
    language: String? = nil
) async throws -> [Float] {
    try await model.generate(text: text, language: language)
}

// Works with any TTS model:
let qwen = try await Qwen3TTSModel.fromPretrained()
let cosy = try await CosyVoiceTTSModel.fromPretrained()

let audio1 = try await synthesizeAny(qwen, text: "Hello")
let audio2 = try await synthesizeAny(cosy, text: "Hello")
```

### Generic Streaming

```swift
func streamAny(
    _ model: any SpeechGenerationModel,
    text: String
) -> AsyncThrowingStream<AudioChunk, Error> {
    model.generateStream(text: text, language: nil)
}
```

### Existential Collections

```swift
let ttsModels: [any SpeechGenerationModel] = [qwen, cosy]

for model in ttsModels {
    let audio = try await model.generate(text: "Hello", language: "english")
    print("Generated \(audio.count) samples at \(model.sampleRate) Hz")
}
```

## Module Structure

```
Sources/
├── AudioCommon/               Shared types, protocols, utilities
│   ├── Protocols.swift        AudioChunk, AlignedWord, SpeechSegment, DiarizedSegment, 9 protocols
│   ├── AudioModelError.swift  Unified error type for all model operations
│   ├── Logging.swift          Centralized os.Logger instances (AudioLog)
│   ├── AudioFileLoader.swift  WAV/audio file loading
│   ├── WAVWriter.swift        WAV file writing
│   ├── WeightLoading.swift    Safetensors loading, HuggingFace download
│   ├── Tokenizer.swift        BPE tokenizer
│   ├── QuantizedMLP.swift     Shared 4-bit SwiGLU MLP
│   └── PreQuantizedEmbedding.swift  4-bit packed embedding table
│
├── Qwen3ASR/                  Speech-to-text (ASR + Forced Aligner)
│   ├── Qwen3ASR.swift         Qwen3ASRModel: SpeechRecognitionModel
│   ├── ForcedAligner.swift    Qwen3ForcedAligner: ForcedAlignmentModel
│   ├── Qwen3ASR+Protocols.swift
│   └── ForcedAligner+Protocols.swift
│
├── Qwen3TTS/                  Text-to-speech (Talker + Code Predictor + Mimi)
│   ├── Qwen3TTS.swift         Qwen3TTSModel: SpeechGenerationModel
│   └── Qwen3TTS+Protocols.swift
│
├── CosyVoiceTTS/              Text-to-speech (LLM + DiT + HiFi-GAN)
│   ├── CosyVoiceTTS.swift     CosyVoiceTTSModel: SpeechGenerationModel
│   └── CosyVoiceTTS+Protocols.swift
│
├── PersonaPlex/               Speech-to-speech (Temporal + Depformer + Mimi)
│   ├── PersonaPlex.swift      PersonaPlexModel: SpeechToSpeechModel
│   └── PersonaPlex+Protocols.swift
│
├── SpeechVAD/                 VAD, diarization, speaker embedding
│   ├── SpeechVAD.swift        PyannoteVADModel: VoiceActivityDetectionModel
│   ├── SpeechVAD+Protocols.swift  Protocol conformances
│   ├── SileroVAD.swift        SileroVADModel: VoiceActivityDetectionModel, StreamingVADProvider
│   ├── SileroModel.swift      Silero VAD v5 network (STFT + encoder + LSTM)
│   ├── StreamingVADProcessor.swift  Event-driven streaming wrapper
│   ├── DiarizationPipeline.swift  PyannoteDiarizationPipeline: SpeakerDiarizationModel, SpeakerExtractionCapable
│   ├── DiarizationHelpers.swift   Shared helpers (merge, compact IDs, resample)
│   ├── SortformerDiarizer.swift   SortformerDiarizer: SpeakerDiarizationModel (CoreML)
│   ├── WeSpeaker.swift        WeSpeakerModel: SpeakerEmbeddingModel
│   └── PowersetDecoder.swift  7-class powerset → per-speaker probabilities
│
├── SpeechCore/                Voice pipeline (wraps speech-core C++ engine)
│   └── VoicePipeline.swift    VoicePipeline: bridges STT/TTS/VAD to C pipeline
│
├── AudioCLILib/               CLI commands and utilities (library)
└── AudioCLI/                  Thin launcher (main.swift → AudioCLILib)
```

### Dependencies

```
AudioCommon  ← Qwen3ASR      ─┐
             ← Qwen3TTS      │
             ← CosyVoiceTTS  ├── AudioCLILib ── AudioCLI (executable)
             ← KokoroTTS     │
             ← PersonaPlex   │
             ← SpeechVAD    ─┘
             ← SpeechCore (CSpeechCore xcframework + AudioCommon)
```

Each model target depends only on `AudioCommon` and MLX. No cross-dependencies between model targets. `SpeechCore` depends on `AudioCommon` for protocols and the `CSpeechCore` binary target for the C++ pipeline engine.

## Thread Safety

All model classes are **not thread-safe** by design. ML inference is inherently sequential on a shared GPU, and MLX's `Module` system does not support actor isolation. Adding synchronization primitives would introduce overhead for a scenario no caller exercises.

**Not thread-safe** (create separate instances for concurrent use):
- `Qwen3ASRModel`, `StreamingASR`
- `Qwen3TTSModel`
- `CosyVoiceTTSModel`
- `PersonaPlexModel`
- `SileroVADModel`, `StreamingVADProcessor`, `PyannoteVADModel`
- `PyannoteDiarizationPipeline` (aliased as `DiarizationPipeline`), `SortformerDiarizer`

**Thread-safe** (all `let` properties, pure computation):
- `WeSpeakerModel`

**Sendable config types** — The following value types conform to `Sendable` and can be safely passed across concurrency boundaries:
`SegmentationConfig`, `VADConfig`, `DiarizationConfig`, `VADPipeline`, `Qwen3AudioEncoderConfig`, `Qwen3ASRTokens`, `SlottedText`, `TextChunker`

## Error Handling

### AudioModelError

Unified error type in `AudioCommon` for cross-module error reporting:

| Case | Fields | When |
|------|--------|------|
| `modelLoadFailed` | `modelId`, `reason`, `underlying?` | Model download or initialization fails |
| `weightLoadingFailed` | `path`, `underlying?` | Safetensors file cannot be read |
| `inferenceFailed` | `operation`, `reason` | Generation or decoding step fails |
| `invalidConfiguration` | `model`, `reason` | Config values are incompatible |
| `voiceNotFound` | `voice`, `searchPath` | Voice preset file missing |

Each case produces a human-readable `errorDescription` with full context including underlying errors.

### Per-module errors

Modules may also define their own error types for domain-specific failures:
- `TTSError` (Qwen3TTS) — tokenizer and language errors
- `CosyVoiceTTSError` (CosyVoiceTTS) — load, download, input, generation errors
- `DownloadError` (AudioCommon) — HuggingFace download failures

## Logging

Centralized structured logging via `os.Logger` (Apple's unified logging system):

```swift
import AudioCommon

// Available loggers:
AudioLog.modelLoading  // Weight loading, initialization, voice preset errors
AudioLog.inference     // Generation, decoding, pipeline steps
AudioLog.download      // HuggingFace downloads, cache operations
```

All loggers use subsystem `com.qwen3speech`. Messages are visible in Console.app and `log stream`.

Used in:
- `PersonaPlexModel` — voice preset loading failures (`.warning`)
- `HuggingFaceDownloader` — directory listing errors (`.debug`)

## Design Decisions

1. **`AnyObject` constraint** — All protocols require reference semantics since ML models hold large weight buffers
2. **Optional `language`** — Protocol methods use `String?` to allow model-specific defaults (Qwen3 defaults to "english", CosyVoice to "english")
3. **Optional `elapsedTime`** — `AudioChunk.elapsedTime` is `Double?` because not all models track wall-clock time (e.g. CosyVoice)
4. **No `ModelLoadable`** — Each model has different loading parameters (TTS needs `tokenizerModelId`, PersonaPlex needs voice presets), so loading stays on concrete types
5. **Unified `AudioChunk`** — All streaming methods return the shared `AudioChunk` type directly. The previous per-model chunk types (`TTSAudioChunk`, `CosyVoiceAudioChunk`, `PersonaPlexAudioChunk`) were removed
6. **Separate `ForcedAlignmentModel`** — Distinct from `SpeechRecognitionModel` because input/output differ (audio+text → timestamps vs audio → text)
7. **Document-only thread safety** — No locks or actors; document the single-threaded contract instead. This matches standard ML library practice (PyTorch, Core ML)
8. **Sendable on value types** — Config structs with only primitive fields get `Sendable` so they can cross `Task` boundaries without warnings
