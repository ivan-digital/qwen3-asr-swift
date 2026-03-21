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

## Embedding Quality (VoxConverse)

Cosine similarity between segment-level embeddings extracted from VoxConverse test set (5 multi-speaker recordings). Measures how well embeddings discriminate speakers in real conversational audio.

- **Intra-speaker**: cosine similarity between different segments of the **same** speaker
- **Inter-speaker**: cosine similarity between segments of **different** speakers
- **Separation**: intra - inter (higher = more discriminative)

| Engine | Intra (mean +/- std) | Inter (mean +/- std) | Separation |
|--------|-------------------|-------------------|------------|
| WeSpeaker MLX | 0.726 +/- 0.210 | 0.142 +/- 0.145 | **0.584** |
| CAM++ CoreML | 0.693 +/- 0.162 | 0.436 +/- 0.132 | 0.257 |

WeSpeaker MLX produces the most discriminative embeddings on same-channel audio, with 0.584 separation — matching the Python pyannote reference (0.577 on same segments). Cosine similarity of 0.974 between Swift and Python embeddings on identical audio.

### Bugs fixed during benchmarking

The initial benchmark revealed near-zero separation (0.008) for WeSpeaker MLX. Root cause analysis found two implementation bugs:

1. **Input dimension ordering**: Python WeSpeaker permutes `(B,T,F)` → `(B,F,T)` before conv — frequency is the height dimension. Our implementation had time as height, causing the ResNet to process spatial dimensions incorrectly.

2. **Feature preprocessing**: Missing CMN (cepstral mean normalization) and wrong window function. pyannote uses hamming window + global mean subtraction over time. We had Povey window and no CMN.

After fixing both issues, Swift embeddings match Python with 0.974 cosine similarity.

## Reproduction

```bash
make build

# Latency benchmark
python scripts/benchmark_speaker.py --latency --engine mlx
python scripts/benchmark_speaker.py --latency --engine coreml
python scripts/benchmark_speaker.py --latency --engine camplusplus

# VoxConverse embedding quality
python scripts/benchmark_speaker.py --voxconverse --engine mlx

# All engines comparison
python scripts/benchmark_speaker.py --compare

# VoxCeleb1-O verification (requires audio download)
python scripts/benchmark_speaker.py --download-voxceleb
python scripts/benchmark_speaker.py --voxceleb --engine mlx
```
