# CosyVoice3 TTS Architecture

## Overview
Three-stage pipeline: LLM → DiT Flow Matching → HiFi-GAN Vocoder → 24kHz audio

## Languages
9 languages supported: Chinese, English, Japanese, Korean, German, Spanish, French, Italian, Russian

## Pipeline Stages

### Stage 1: LLM (Qwen2.5-0.5B)
- Speech token generator based on Qwen2.5-0.5B
- 24 transformer layers, 896 hidden, 14Q/2KV heads
- FSQ vocabulary: 6561 tokens at 25 Hz
- Input: [sos, text_tokens..., task_id, speech_tokens...]
- Autoregressive with KV cache

### Stage 2: DiT Flow Matching
- 22-layer Diffusion Transformer (1024 hidden, 16 heads)
- Conditional flow matching with Euler ODE solver (10 steps)
- Classifier-free guidance (rate=0.7)
- AdaLN (Adaptive Layer Norm) for timestep conditioning
- Converts speech tokens → 80-band mel spectrogram
- Token upsampling: 25 Hz → 50 Hz via linear interpolation

### Stage 3: HiFi-GAN Vocoder
- Neural Source Filter (NSF) with 8 harmonics
- F0 prediction from mel spectrogram
- 3-stage upsampling [8, 5, 3] = 120x + ISTFT (hop=4) = 480x total
- Snake activation in residual blocks
- ISTFT reconstruction (n_fft=16) → 24kHz waveform

## Streaming
- Chunk-aware causal masking in DiT
- 25-token chunks (~1 second of audio)
- Target: ~150ms latency to first chunk

## Weight Conversion
- Source: FunAudioLLM/Fun-CosyVoice3-0.5B-2512 (PyTorch .pt)
- LLM: 4-bit quantized (group_size=64)
- DiT Flow: bfloat16
- HiFi-GAN: float32
- Conv1d weights transposed: PyTorch [out,in,k] → MLX [out,k,in]
- Total: ~1.9 GB (quantized)

## Configuration
Key parameters from cosyvoice3.yaml:
- Sample rate: 24000 Hz
- Mel: 80 bins, n_fft=1920, hop=480
- Token frame rate: 25 Hz, mel frame rate: 50 Hz

## References
- CosyVoice 3 paper: https://arxiv.org/abs/2505.17589
- Model: https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512
