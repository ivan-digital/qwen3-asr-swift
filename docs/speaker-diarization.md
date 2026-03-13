# Speaker Diarization & Speaker Embedding

## Overview

Speaker diarization identifies **who spoke when** in an audio recording. Two engines are available:

1. **Pyannote** (default) — three-stage pipeline: Pyannote segmentation → WeSpeaker embedding → agglomerative clustering
2. **Sortformer** (CoreML) — NVIDIA's end-to-end neural diarization model, runs on Neural Engine

## Architecture

### Engine Selection

```bash
audio diarize meeting.wav                    # Pyannote (default)
audio diarize meeting.wav --engine sortformer  # Sortformer (CoreML)
```

### Sortformer (End-to-End, CoreML)

NVIDIA Sortformer predicts per-frame speaker activity for up to 4 speakers directly from mel features. No separate embedding or clustering stages.

```
Audio → [128-dim Mel] → [Chunk Sliding Window] → [CoreML Neural Engine] → [Sigmoid + Binarize] → Segments
                             ↕ streaming state
                        (spkcache + fifo buffers)
```

- **Input**: 128-dim log-mel features (Hann window, nFFT=400, hop=160, 16kHz)
- **Chunking**: 112 mel frames per chunk (6 core + 1 left + 7 right context × 8 subsampling)
- **CoreML model**: `[1,112,128]` chunk + `[1,188,512]` spkcache + `[1,40,512]` fifo → `[1,242,4]` speaker preds
- **State management**: FIFO overflow → spkcache (streaming state carried across chunks)
- **Post-processing**: Sigmoid → hysteresis binarization (onset=0.5, offset=0.3) → segment merging
- **Frame duration**: 0.08s per prediction frame

No speaker embeddings are produced — `--target-speaker` and `--embedding-engine` are not available with Sortformer.

### Pyannote Three-Stage Pipeline

```
Audio → [Stage 1: Segmentation] → [Stage 2: Embedding] → [Stage 3: Clustering] → Diarized Segments
```

**Stage 1 — Segmentation**: Pyannote model processes 10s sliding windows. Instead of collapsing the 7-class powerset to binary VAD, we use `PowersetDecoder` to extract per-speaker probabilities:
- spk1 = P(class 1) + P(class 4) + P(class 5)
- spk2 = P(class 2) + P(class 4) + P(class 6)
- spk3 = P(class 3) + P(class 5) + P(class 6)

Hysteresis binarization produces local speaker segments per window.

**Stage 2 — Embedding**: For each local segment, crop the audio and extract a 256-dim speaker embedding using WeSpeaker ResNet34-LM.

**Stage 3 — Clustering**: Spectral clustering with GMM-BIC automatically estimates the number of speakers and assigns global speaker IDs. Cosine affinity → normalized Laplacian → LAPACK eigendecomposition → GMM-BIC model selection → k-means on spectral embedding. No manual threshold tuning required.

### WeSpeaker ResNet34-LM

~6.6M params, 256-dim output, ~25 MB.

```
Input: [B, T, 80, 1] log-mel spectrogram (80 fbank, 16kHz)
  │
  ├─ Conv2d(1→32, k=3, p=1) + ReLU           (BN fused)
  ├─ Layer1: 3× BasicBlock(32→32)
  ├─ Layer2: 4× BasicBlock(32→64, s=2)
  ├─ Layer3: 6× BasicBlock(64→128, s=2)
  ├─ Layer4: 3× BasicBlock(128→256, s=2)
  │
  ├─ Statistics Pooling: mean + std → [B, 5120]
  ├─ Linear(5120→256) → L2 normalize
  │
  Output: 256-dim L2-normalized speaker embedding
```

BatchNorm is **fused into Conv2d at conversion time** — no BN layers in the Swift model. This simplifies the model and avoids train/eval mode differences.

### CoreML Backend

WeSpeaker supports a CoreML backend (`engine: .coreml`) that runs on the Neural Engine, freeing the GPU for concurrent workloads.

```swift
let model = try await WeSpeakerModel.fromPretrained(engine: .coreml)
// Same API — embed(), cosineSimilarity()

let pipeline = try await DiarizationPipeline.fromPretrained(embeddingEngine: .coreml)
```

The CoreML model uses EnumeratedShapes for variable mel lengths (20–2000 frames, covering ~0.3s–32s audio) and float16 I/O. Input: `[1, 1, T, 80]` NCHW mel spectrogram. Output: `[1, 256]` L2-normalized embedding.

| Backend | Latency (20s audio) | Hardware | Model |
|---------|-------------------|----------|-------|
| MLX | ~310ms | GPU (Metal) | [aufklarer/WeSpeaker-ResNet34-LM-MLX](https://huggingface.co/aufklarer/WeSpeaker-ResNet34-LM-MLX) |
| CoreML | ~430ms | Neural Engine + CPU | [aufklarer/WeSpeaker-ResNet34-LM-CoreML](https://huggingface.co/aufklarer/WeSpeaker-ResNet34-LM-CoreML) |

MLX and CoreML embeddings are **not interchangeable** — NHWC vs NCHW layout causes stats pooling to flatten features in different orders. Each backend is self-consistent (cosine sim 1.0 for same input) but cross-backend similarity is low (~0.15). Use the same backend for enrollment and comparison.

### Mel Feature Extraction

80-dim log-mel spectrogram via vDSP (same pipeline as WhisperFeatureExtractor but with different parameters):
- **Hamming window** (not Hann): `0.54 - 0.46 * cos(2π*i/N)`
- nFFT=400, hop=160, 16kHz
- 80 mel bins with Slaney normalization
- Simple `log(max(mel, 1e-10))` — no Whisper-specific normalization

## Usage

### Speaker Diarization

```swift
let pipeline = try await DiarizationPipeline.fromPretrained()
let result = pipeline.diarize(audio: samples, sampleRate: 16000)

for seg in result.segments {
    print("Speaker \(seg.speakerId): [\(seg.startTime)s - \(seg.endTime)s]")
}
print("\(result.numSpeakers) speakers detected")
```

### Speaker Embedding

```swift
let model = try await WeSpeakerModel.fromPretrained()
let embedding = model.embed(audio: samples, sampleRate: 16000)
// embedding: [Float] of length 256, L2-normalized
```

### Speaker Extraction

Given a reference audio of a target speaker, extract only their segments:

```swift
let pipeline = try await DiarizationPipeline.fromPretrained()

// Get target speaker embedding from enrollment audio
let targetEmb = pipeline.embeddingModel.embed(audio: enrollmentAudio, sampleRate: 16000)

// Extract target speaker's segments from the main audio
let segments = pipeline.extractSpeaker(
    audio: meetingAudio, sampleRate: 16000,
    targetEmbedding: targetEmb
)
```

### Sortformer Swift API

```swift
let diarizer = try await SortformerDiarizer.fromPretrained()
let result = diarizer.diarize(audio: samples, sampleRate: 16000, config: .default)

for seg in result.segments {
    print("Speaker \(seg.speakerId): [\(seg.startTime)s - \(seg.endTime)s]")
}
// result.speakerEmbeddings is empty (end-to-end model)
```

### CLI Commands

```bash
# Pyannote diarization (default)
audio diarize meeting.wav

# Sortformer diarization (CoreML, Neural Engine)
audio diarize meeting.wav --engine sortformer

# CoreML embeddings (Neural Engine, pyannote only)
audio diarize meeting.wav --embedding-engine coreml

# With options
audio diarize meeting.wav --min-speakers 2 --max-speakers 3 --json

# Speaker extraction (pyannote only)
audio diarize meeting.wav --target-speaker enrollment.wav

# Embed a speaker's voice
audio embed-speaker enrollment.wav
audio embed-speaker enrollment.wav --engine coreml --json
```

## Model Weights

- **Segmentation**: `aufklarer/Pyannote-Segmentation-MLX` (~5.7 MB)
- **Speaker Embedding (MLX)**: `aufklarer/WeSpeaker-ResNet34-LM-MLX` (~25 MB)
- **Speaker Embedding (CoreML)**: `aufklarer/WeSpeaker-ResNet34-LM-CoreML` (~13 MB)
- **Sortformer (CoreML)**: `FluidInference/diar-streaming-sortformer-coreml` (~50 MB)
- Cache: `~/Library/Caches/qwen3-speech/`

### Weight Conversion

Both backends fuse BatchNorm into Conv2d at conversion time: `w_fused = w * γ/√(σ²+ε)`, `b_fused = β - μ·γ/√(σ²+ε)`. MLX additionally transposes to channels-last `[O,H,W,I]`. Conversion scripts are in `scripts/`.

## Protocols

The module provides protocol conformances in `AudioCommon`:

```swift
// SpeakerEmbeddingModel
extension WeSpeakerModel: SpeakerEmbeddingModel {}

// SpeakerDiarizationModel
extension DiarizationPipeline: SpeakerDiarizationModel {
    func diarize(audio: [Float], sampleRate: Int) -> [DiarizedSegment]
}

// SpeakerDiarizationModel (end-to-end, CoreML)
extension SortformerDiarizer: SpeakerDiarizationModel {
    func diarize(audio: [Float], sampleRate: Int) -> [DiarizedSegment]
}
```

## File Structure

```
Sources/SpeechVAD/
├── MelFeatureExtractor.swift          80-dim fbank via vDSP (WeSpeaker)
├── WeSpeakerModel.swift               ResNet34 network (BN-fused Conv2d)
├── WeSpeakerWeightLoading.swift       Weight loading from safetensors
├── WeSpeaker.swift                    Public API: embed(), fromPretrained(), engine selection
├── CoreMLWeSpeakerInference.swift     CoreML inference (EnumeratedShapes, float16)
├── PowersetDecoder.swift              7-class powerset → per-speaker probs
├── SpectralClustering.swift           GMM-BIC + spectral clustering (Accelerate/LAPACK)
├── DiarizationPipeline.swift          Full Pyannote pipeline + speaker extraction
├── SortformerConfig.swift             Sortformer model configuration
├── SortformerMelExtractor.swift       128-dim log-mel for Sortformer (Hann window)
├── SortformerModel.swift              CoreML wrapper for Sortformer inference
├── SortformerDiarizer.swift           End-to-end Sortformer pipeline (streaming)
└── SpeechVAD+Protocols.swift          Protocol conformances

Sources/AudioCommon/Protocols.swift    DiarizedSegment, SpeakerEmbeddingModel, SpeakerDiarizationModel
Sources/AudioCLILib/DiarizeCommand.swift       `audio diarize` (--engine, --embedding-engine)
Sources/AudioCLILib/EmbedSpeakerCommand.swift  `audio embed-speaker` (--engine)
scripts/convert_wespeaker.py                    MLX weight conversion
scripts/convert_wespeaker_coreml.py             CoreML weight conversion
```
