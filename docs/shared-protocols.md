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
│                      SpeechToSpeechModel                 │
│                      VoiceActivityDetectionModel (VAD)   │
└─────────────────────────────────────────────────────────┘
        ▲                    ▲                    ▲
        │                    │                    │
   ┌────┴────┐        ┌─────┴─────┐       ┌─────┴─────┐       ┌─────┴─────┐
   │Qwen3TTS │        │  Qwen3ASR │       │PersonaPlex │       │ SpeechVAD │
   │CosyVoice│        │ForcedAlign│       └───────────┘       └───────────┘
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

**Conforming types:** `Qwen3TTSModel`, `CosyVoiceTTSModel`

### SpeechRecognitionModel (STT)

Speech-to-text models that transcribe audio.

```swift
public protocol SpeechRecognitionModel: AnyObject {
    var inputSampleRate: Int { get }
    func transcribe(audio: [Float], sampleRate: Int, language: String?) -> String
}
```

**Conforming types:** `Qwen3ASRModel`

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
│   ├── Protocols.swift        AudioChunk, AlignedWord, SpeechSegment, 5 protocols
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
├── SpeechVAD/                 Voice Activity Detection (pyannote + Silero)
│   ├── SpeechVAD.swift        PyannoteVADModel: VoiceActivityDetectionModel
│   ├── SpeechVAD+Protocols.swift
│   ├── SileroVAD.swift        SileroVADModel: VoiceActivityDetectionModel
│   ├── SileroModel.swift      Silero VAD v5 network (STFT + encoder + LSTM)
│   └── StreamingVADProcessor.swift  Event-driven streaming wrapper
│
├── AudioCLILib/               CLI commands and utilities (library)
└── AudioCLI/                  Thin launcher (main.swift → AudioCLILib)
```

### Dependencies

```
AudioCommon  ← Qwen3ASR      ─┐
             ← Qwen3TTS      │
             ← CosyVoiceTTS  ├── AudioCLILib ── AudioCLI (executable)
             ← PersonaPlex   │
             ← SpeechVAD    ─┘
```

Each model target depends only on `AudioCommon` and MLX. No cross-dependencies between model targets.

## Design Decisions

1. **`AnyObject` constraint** — All protocols require reference semantics since ML models hold large weight buffers
2. **Optional `language`** — Protocol methods use `String?` to allow model-specific defaults (Qwen3 defaults to "english", CosyVoice to "english")
3. **Optional `elapsedTime`** — `AudioChunk.elapsedTime` is `Double?` because not all models track wall-clock time (e.g. CosyVoice)
4. **No `ModelLoadable`** — Each model has different loading parameters (TTS needs `tokenizerModelId`, PersonaPlex needs voice presets), so loading stays on concrete types
5. **Unified `AudioChunk`** — All streaming methods return the shared `AudioChunk` type directly. The previous per-model chunk types (`TTSAudioChunk`, `CosyVoiceAudioChunk`, `PersonaPlexAudioChunk`) were removed
6. **Separate `ForcedAlignmentModel`** — Distinct from `SpeechRecognitionModel` because input/output differ (audio+text → timestamps vs audio → text)
