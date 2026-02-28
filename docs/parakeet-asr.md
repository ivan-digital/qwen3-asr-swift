# Parakeet TDT ASR

CoreML-based automatic speech recognition using NVIDIA's Parakeet-TDT 0.6B v3.

## Architecture

**FastConformer Encoder**: 24-layer conformer with depthwise-separable convolutions and 8x subsampling. Input: 128-dim mel spectrogram at 16kHz. Output: 1024-dim encoded representations.

**TDT Decoder**: Token-and-Duration Transducer — an extension of RNN-T that adds a duration prediction head alongside the standard token prediction head.

- **Prediction network**: 2-layer LSTM (640 hidden) predicts from previously emitted tokens
- **Joint network**: Combines encoder and prediction outputs, produces dual logits:
  - Token logits (8193 classes: 8192 SentencePiece tokens + 1 blank)
  - Duration logits (5 classes: `[0, 1, 2, 3, 4]` frames to advance)

### TDT Decoding Algorithm

Standard RNN-T always advances 1 encoder frame on blank. TDT enhances this:

```
t = 0
while t < encoded_length:
    (token_logits, dur_logits) = joint(encoder[t], decoder_state)
    token = argmax(token_logits)
    if token == blank:
        t += 1
    else:
        emit(token)
        duration = duration_bins[argmax(dur_logits)]
        t += max(duration, 1)
        decoder_state = lstm(token, decoder_state)
```

This variable-rate alignment is more accurate and efficient than fixed-rate RNN-T.

## CoreML Models

The model is split into 4 CoreML sub-models for optimal compute unit placement:

| Model | Compute | Quantization | Purpose |
|-------|---------|--------------|---------|
| `preprocessor.mlmodelc` | CPU only | None | Raw audio → 128-dim mel features |
| `encoder.mlmodelc` | CPU + Neural Engine | INT4 palettized | Mel → 1024-dim encoded features |
| `decoder.mlmodelc` | CPU + Neural Engine | None | LSTM prediction network |
| `joint.mlmodelc` | CPU + Neural Engine | None | Dual-head token + duration logits |

The encoder is INT4 palettized (~75% size reduction) with minimal accuracy impact. Decoder and joint are small enough that quantization isn't necessary.

## Audio Preprocessing

- Sample rate: 16kHz
- Mel bins: 128
- n_fft: 512, hop: 160, win: 400
- Window: Hanning
- Pre-emphasis: 0.97 (applied in the CoreML preprocessor model)

## CLI Usage

```bash
# Transcribe with Parakeet (CoreML, fast on Neural Engine)
audio transcribe --engine parakeet audio.wav

# Transcribe with Qwen3 (MLX, default)
audio transcribe audio.wav
audio transcribe --engine qwen3 audio.wav
```

## Performance

On Apple Silicon with Neural Engine:
- ~110x real-time factor on M4 Pro
- First inference includes CoreML compilation (~2-3s), subsequent runs are faster
- Encoder runs on Neural Engine for maximum throughput

## Model Weights

- [aufklarer/Parakeet-TDT-v3-CoreML-INT4](https://huggingface.co/aufklarer/Parakeet-TDT-v3-CoreML-INT4)

## Thread Safety

- `ParakeetConfig` — `Sendable` (immutable value type)
- `ParakeetVocabulary` — `Sendable` (immutable dictionary)
- `ParakeetASRModel` — NOT thread-safe (LSTM decoder state). Create separate instances for concurrent use.
- CoreML `MLModel.prediction()` is thread-safe for stateless models, but the TDT decode loop manages LSTM h/c state as local variables (safe per-call, not re-entrant).
