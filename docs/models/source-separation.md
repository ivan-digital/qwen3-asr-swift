# Open-Unmix HQ — Music Source Separation

## Architecture

Open-Unmix HQ (UMX-HQ) separates stereo music into 4 stems: vocals, drums, bass, other. Each stem has an independent model with identical architecture but separate weights.

```
Stereo 44.1kHz → STFT (4096/1024) → magnitude → normalize
→ FC1 (2974→512) → BN1 → tanh → BiLSTM (3-layer, 256/dir)
→ skip connection → FC2 (1024→512) → BN2 → ReLU
→ FC3 (512→4098) → BN3 → denormalize → ReLU mask × input → iSTFT → stem
```

### Signal Flow

1. **STFT**: 4096-point FFT, 1024-hop, periodic Hann window, center-padded with reflect mode. Produces 2049 frequency bins per frame.

2. **Input normalization**: Crop to 1487 bins (~16kHz), apply learned per-bin mean/scale (from training statistics).

3. **Encoder**: FC layer (2974→512, no bias) + BatchNorm + tanh. The 2974 input = 2 channels × 1487 bins.

4. **BiLSTM**: 3-layer bidirectional LSTM, 256 hidden per direction (512 total). Captures temporal context across frames.

5. **Decoder**: Skip connection from encoder output concatenated with LSTM output (1024→512), then FC + BN + ReLU, then FC + BN to full spectrum (4098 = 2 channels × 2049 bins).

6. **Output denormalization**: Apply learned output mean/scale, ReLU to ensure non-negative.

7. **Masking**: Element-wise multiply with input magnitude spectrogram. The model predicts a magnitude mask, not direct spectral values.

8. **Wiener post-filtering**: Soft-mask refinement across all 4 targets. Computes power-ratio masks from all source estimates to enforce that sources sum to the mixture.

9. **iSTFT**: Overlap-add synthesis with window normalization. Center padding removed to match original length.

### Model Parameters

| Component | Parameters |
|-----------|-----------|
| FC1 | 2974 × 512 = 1,522,688 |
| BN1 | 512 × 4 = 2,048 |
| BiLSTM (3 layers) | ~5.2M |
| FC2 | 1024 × 512 = 524,288 |
| BN2 | 512 × 4 = 2,048 |
| FC3 | 512 × 4098 = 2,098,176 |
| BN3 | 4098 × 4 = 16,392 |
| Normalization params | 2 × (1487 + 2049) = 7,072 |
| **Total per stem** | **~8.9M** |
| **Total (4 stems)** | **~35.6M** |

### Weights

- Format: safetensors (MLX-compatible)
- Size: ~34 MB per stem, ~136 MB total
- HuggingFace: [aufklarer/OpenUnmix-HQ-MLX](https://huggingface.co/aufklarer/OpenUnmix-HQ-MLX)

## STFT Configuration

| Parameter | Value |
|-----------|-------|
| n_fft | 4096 |
| n_hop | 1024 |
| Window | Periodic Hann |
| Center padding | Reflect |
| Frequency bins | 2049 (one-sided) |
| Max bin (model input) | 1487 (~16kHz at 44.1kHz) |
| Sample rate | 44,100 Hz |

## Reference

- [Open-Unmix (GitHub)](https://github.com/sigsep/open-unmix-pytorch)
- Stöter et al., "Open-Unmix — A Reference Implementation for Music Source Separation" (JOSS, 2019)
