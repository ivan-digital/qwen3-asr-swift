# Speaker Diarization & Speaker Embedding

## Overview

Speaker diarization identifies **who spoke when** in an audio recording. This module combines two models:

1. **Pyannote Segmentation** (PyanNet) — already used for VAD, outputs 7-class powerset probabilities for up to 3 local speakers per window
2. **WeSpeaker ResNet34-LM** — speaker embedding model that produces 256-dim vectors from speech, used to link local speakers across windows

## Architecture

### Three-Stage Pipeline

```
Audio → [Stage 1: Segmentation] → [Stage 2: Embedding] → [Stage 3: Clustering] → Diarized Segments
```

**Stage 1 — Segmentation**: Pyannote model processes 10s sliding windows. Instead of collapsing the 7-class powerset to binary VAD, we use `PowersetDecoder` to extract per-speaker probabilities:
- spk1 = P(class 1) + P(class 4) + P(class 5)
- spk2 = P(class 2) + P(class 4) + P(class 6)
- spk3 = P(class 3) + P(class 5) + P(class 6)

Hysteresis binarization produces local speaker segments per window.

**Stage 2 — Embedding**: For each local segment, crop the audio and extract a 256-dim speaker embedding using WeSpeaker ResNet34-LM.

**Stage 3 — Clustering**: Agglomerative clustering with average linkage and cosine distance merges local speaker embeddings into global speaker IDs.

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

### CLI Commands

```bash
# Full diarization
audio diarize meeting.wav
# Speaker 0: [0.50s - 3.20s] (2.70s)
# Speaker 1: [3.50s - 7.10s] (3.60s)
# Speaker 0: [7.30s - 9.80s] (2.50s)
# --- 2 speaker(s) ---

# With options
audio diarize meeting.wav --threshold 0.6 --max-speakers 3 --json

# Speaker extraction
audio diarize meeting.wav --target-speaker enrollment.wav

# Embed a speaker's voice
audio embed-speaker enrollment.wav
audio embed-speaker enrollment.wav --json
```

## Model Weights

- **Segmentation**: `aufklarer/Pyannote-Segmentation-MLX` (~5.7 MB)
- **Speaker Embedding**: `aufklarer/WeSpeaker-ResNet34-LM-MLX` (~25 MB)
- Cache: `~/Library/Caches/qwen3-speech/`

### Weight Conversion

```bash
# Convert WeSpeaker weights (requires HuggingFace token for gated model)
python scripts/convert_wespeaker.py --token YOUR_HF_TOKEN

# Upload to HuggingFace
python scripts/convert_wespeaker.py --upload --repo-id aufklarer/WeSpeaker-ResNet34-LM-MLX
```

The conversion script:
1. Downloads `pyannote/wespeaker-voxceleb-resnet34-LM`
2. Fuses BatchNorm into Conv2d: `w_fused = w * γ/√(σ²+ε)`, `b_fused = β - μ·γ/√(σ²+ε)`
3. Transposes Conv2d weights: `[O,I,H,W]` → `[O,H,W,I]` for MLX channels-last
4. Renames `seg_1` → `embedding`
5. Saves as safetensors + config.json

## Protocols

The module provides protocol conformances in `AudioCommon`:

```swift
// SpeakerEmbeddingModel
extension WeSpeakerModel: SpeakerEmbeddingModel {}

// SpeakerDiarizationModel
extension DiarizationPipeline: SpeakerDiarizationModel {
    func diarize(audio: [Float], sampleRate: Int) -> [DiarizedSegment]
}
```

## File Structure

```
Sources/SpeechVAD/
├── MelFeatureExtractor.swift      80-dim fbank via vDSP
├── WeSpeakerModel.swift           ResNet34 network (BN-fused Conv2d)
├── WeSpeakerWeightLoading.swift   Weight loading from safetensors
├── WeSpeaker.swift                Public API: embed(), fromPretrained()
├── PowersetDecoder.swift          7-class powerset → per-speaker probs
├── DiarizationPipeline.swift      Full pipeline + speaker extraction
└── SpeechVAD+Protocols.swift      Protocol conformances

Sources/AudioCommon/Protocols.swift    DiarizedSegment, SpeakerEmbeddingModel, SpeakerDiarizationModel
Sources/AudioCLILib/DiarizeCommand.swift       `audio diarize`
Sources/AudioCLILib/EmbedSpeakerCommand.swift  `audio embed-speaker`
scripts/convert_wespeaker.py                    Weight conversion
```
