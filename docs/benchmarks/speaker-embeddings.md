# Speaker Embedding Benchmark

## Models

| Model | Architecture | Embedding Dim | Params | Backend | Weights |
|-------|-------------|---------------|--------|---------|---------|
| WeSpeaker ResNet34-LM | ResNet34 + Stats Pooling | 256 | 6.6M | MLX (GPU) | 25 MB |
| WeSpeaker ResNet34-LM | ResNet34 + Stats Pooling | 256 | 6.6M | CoreML (ANE) | 25 MB |
| CAM++ (3D-Speaker) | CAM++ | 192 | ~7M | CoreML (ANE) | 14 MB |

## Extraction Latency (M2 Max, 64 GB)

20s audio clip, 10 iterations after warmup.

| Engine | Dim | Mean | Std | Min |
|--------|-----|------|-----|-----|
| WeSpeaker MLX | 256 | 65 ms | 3.9 ms | 60 ms |
| WeSpeaker CoreML | 256 | 148 ms | 10.7 ms | 141 ms |
| CAM++ CoreML | 192 | 12 ms | 0.6 ms | 11 ms |

## Speaker Verification (LibriSpeech test-clean)

40 speakers, 2780 trial pairs (2000 positive, 780 negative) generated from LibriSpeech test-clean. Scoring with cosine similarity.

| Engine | EER% | minDCF (p=0.01) | Pos Mean Sim | Neg Mean Sim |
|--------|------|-----------------|--------------|--------------|
| WeSpeaker MLX | **0.98** | **0.084** | 0.718 | 0.091 |
| CAM++ CoreML | 9.49 | 0.440 | 0.742 | 0.424 |

### Comparison with published results (VoxCeleb1-O)

| Model | Published EER% | Params | Source |
|-------|---------------|--------|--------|
| CAM++ | 0.65 | 7.2M | 3D-Speaker (Interspeech 2023) |
| WeSpeaker ResNet34-LM | 0.72 | 6.6M | WeSpeaker VoxSRC2023 |

LibriSpeech test-clean is easier than VoxCeleb1-O (read speech, cleaner audio), so direct comparison is indicative only. WeSpeaker's 0.98% EER is consistent with published VoxCeleb1-O numbers (0.72%). CAM++'s 9.49% EER is higher than its published VoxCeleb1-O number (0.65%) — the CoreML model uses a fixed 500-frame input shape (short audio is tiled, long audio is center-cropped). Re-exporting with EnumeratedShapes would improve this.

## Embedding Quality (VoxConverse)

Cosine similarity between segment-level embeddings extracted from VoxConverse test set (5 multi-speaker recordings). Measures how well embeddings discriminate speakers in real conversational audio.

- **Intra-speaker**: cosine similarity between different segments of the **same** speaker
- **Inter-speaker**: cosine similarity between segments of **different** speakers
- **Separation**: intra - inter (higher = more discriminative)

| Engine | Intra (mean +/- std) | Inter (mean +/- std) | Separation |
|--------|-------------------|-------------------|------------|
| WeSpeaker MLX | 0.726 +/- 0.210 | 0.142 +/- 0.145 | **0.584** |
| CAM++ CoreML | 0.693 +/- 0.162 | 0.436 +/- 0.132 | 0.257 |

WeSpeaker MLX matches the Python pyannote reference (0.577 separation on same segments, cosine similarity 0.974 between Swift and Python embeddings).

CAM++ trades discrimination for speed — 5x faster (12 ms vs 65 ms), suited for real-time diarization on Neural Engine.

## Reproduction

```bash
make build

# Speaker verification (LibriSpeech test-clean, auto-downloaded)
python scripts/benchmark_speaker.py --librispeech --engine mlx

# VoxConverse embedding quality
python scripts/benchmark_speaker.py --voxconverse --engine mlx

# Latency benchmark
python scripts/benchmark_speaker.py --latency --engine mlx

# All engines comparison
python scripts/benchmark_speaker.py --compare
```
