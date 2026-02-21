# TTS Model Architecture (Qwen3-TTS)

> Reference for Swift MLX port. Based on [Qwen3-TTS-12Hz-0.6B](https://arxiv.org/abs/2601.15621). Speech tokenizer decoder based on [Mimi](https://arxiv.org/abs/2410.00037) (Kyutai).

## Overview

Qwen3-TTS has four components: a Talker (main LM), Code Predictor (residual codebooks), Speech Tokenizer (neural audio codec), and optional Speaker Encoder (voice cloning). The Swift port currently implements the Talker, Code Predictor, and Speech Tokenizer Decoder. The Speech Tokenizer Encoder and Speaker Encoder are not yet ported.

```
Text input
    |
    v
+--------------------+
|  Text Embedding    |   Qwen2 BPE -> Embedding(151936, 2048) -> MLP projection -> 1024
+--------+-----------+
         |
         v
+--------------------+
|  Talker            |   28-layer Qwen3 transformer (MRoPE, GQA, SwiGLU)
|  1024 hidden dim   |   Generates first codebook autoregressively
+--------+-----------+
         |  hidden states + first codebook tokens
         v
+--------------------+
|  Code Predictor    |   5-layer transformer (standard RoPE)
|  1024 hidden dim   |   Predicts remaining 15 codebooks sequentially
+--------+-----------+
         |  16 codebook indices per frame
         v
+--------------------+
|  Speech Tokenizer  |   Mimi-based neural codec
|  Decoder only      |   RVQ decode -> Transformer -> Upsampling convs -> waveform
+--------+-----------+
         |
         v
    Audio (24kHz)
```

## Component A: Talker

The primary autoregressive transformer. Generates the first codebook of speech tokens from text.

| Parameter | 0.6B | 1.7B |
|-----------|------|------|
| Hidden size | 1024 | 2048 |
| Layers | 28 | 28 |
| Attention heads (Q) | 16 | 16 |
| KV heads (GQA) | 8 | 8 |
| Head dimension | 128 | 128 |
| Intermediate size | 3072 | 8192 |
| Codec vocab size | 3072 | 3072 |
| Text vocab size | 151936 | 151936 |
| RoPE type | **MRoPE** (3D sections [24,20,20], interleaved) | MRoPE |
| RoPE base | 1,000,000 | 1,000,000 |
| Quantization | 4-bit | 4-bit |
| Q/K normalization | RMSNorm per head | RMSNorm per head |

**MRoPE (Multimodal RoPE):** Unlike ASR's standard 1D RoPE, the Talker uses 3D position encoding with sections `[24, 20, 20]` across the 64 rotation dimensions (head_dim/2 = 64). Positions are interleaved as `[T, H, W, T, H, W, ...]` across the dimension.

**Two embedding tables:**
- `text_embedding`: 151936 tokens, dim 2048 (projected to 1024 via MLP)
- `codec_embedding`: per-codebook, dim 1024

**Special codec tokens (0.6B):**
| Token | ID |
|-------|-----|
| `codec_pad` | 2148 |
| `codec_bos` | 2149 |
| `codec_eos` | 2150 |
| Language: English | 2050 |
| Language: German | 2052 |
| Language: Spanish | 2054 |
| Language: Chinese | 2055 |
| Language: Japanese | 2058 |
| Language: French | 2061 |
| Language: Korean | 2064 |
| Language: Russian | 2069 |
| Language: Italian | 2070 |

**Transformer block (identical to ASR except RoPE):**
```
x -> RMSNorm -> Attention(Q/K projections, Q/K RMSNorm, MRoPE, GQA via SDPA) -> + residual
  -> RMSNorm -> SwiGLU MLP(gate_proj, up_proj, down_proj)                     -> + residual
```

## Component B: Code Predictor

A smaller transformer that predicts the remaining 15 codebooks given the first.

| Parameter | Value |
|-----------|-------|
| Hidden size | 1024 |
| Layers | 5 |
| Attention heads (Q) | 16 |
| KV heads (GQA) | 8 |
| Head dimension | 128 |
| Intermediate size | 3072 |
| Vocab size | 2048 (per codebook) |
| RoPE type | Standard 1D |
| LM heads | 15 (one per remaining codebook) |

**Sequential prediction:** For each time step, predicts codebook 2 from codebook 1's hidden state, then codebook 3, and so on through codebook 16.

## Component C: Speech Tokenizer (Mimi Codec)

A neural audio codec that converts between waveforms and discrete multi-codebook tokens at **12.5 Hz**.

### Encoder (Not Yet Implemented)

> The encoder converts audio to codebook tokens and is used for voice cloning input. It is not yet ported to Swift.

```
Audio (24kHz) -> SeanetEncoder (Conv1d downsampling, residual blocks)
             -> ProjectedTransformer (8 layers, causal, RoPE)
             -> ConvDownsample1d (to 12.5 Hz)
             -> SplitResidualVectorQuantizer
                  - 1 semantic quantizer (4096 codebook)
                  - 15 acoustic quantizers (2048 codebook each)
             -> 16 codebook indices per frame
```

| Parameter | Value |
|-----------|-------|
| Input sample rate | 24,000 Hz |
| Frame rate | 12.5 Hz |
| Downsample rate | 1920x (24000 / 12.5) |
| Encoder hidden size | 512 |
| Transformer layers | 8 |
| Transformer heads | 8 |
| Num quantizers | 16 (1 semantic + 15 acoustic) |
| Codebook size | 2048 (acoustic), 4096 (semantic) |
| Codebook dim | 256 |

### Decoder (tokens -> audio, used for synthesis output)

```
16 codebook indices -> SplitRVQ.decode() (sum embeddings)       -> [T, 512]
                    -> Pre-conv (CausalConv1d k=3)              -> [T, 1024]
                    -> Pre-transformer (8 layers, causal RoPE,
                       1024->512 bottleneck, SwiGLU+LayerScale) -> [T, 1024]
                    -> Pre-upsample (TransposedConv1d 2x + ConvNeXt) x2 -> [4T, 1024]
                    -> Input conv                               -> [4T, 1536]
                    -> SEANet decoder (8x,5x,4x,3x = 480x):
                       SnakeBeta + TransposedConv1d + 3x dilated residual units
                    -> SnakeBeta + CausalConv1d(7,1) + clip     -> [T*1920, 1]
                    -> Audio waveform (24kHz)
```

| Parameter | Value |
|-----------|-------|
| Output sample rate | 24,000 Hz |
| Decoder dim | 1536 |
| Latent dim | 1024 |
| Transformer layers | 8 |
| Transformer heads | 16 |
| Upsample rates | [2, 2] pre-upsample (4x) then [8, 5, 4, 3] SEANet decoder (480x) = 1920x |

**SnakeBeta activation:** `x + (1/b) * sin^2(a * x)` — learnable periodic activation used in the decoder upsampling blocks for high-quality audio reconstruction.

## Model Variants

Qwen3-TTS ships in two variants with identical architecture (Talker + Code Predictor + Speech Tokenizer). The difference is fine-tuning and how speaker identity is provided.

| Variant | HuggingFace ID (0.6B, 4-bit) | Speaker Selection |
|---------|-------------------------------|-------------------|
| **Base** | `mlx-community/Qwen3-TTS-12Hz-0.6B-Base-4bit` | None (single default voice) |
| **CustomVoice** | `mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-4bit` | 9 preset voices + instruction control |

### CustomVoice Speakers

The CustomVoice model includes 9 preset voices. Each speaker is selected by prepending a speaker token ID to the codec prefix.

| Speaker | Language | Codec Token ID | Description |
|---------|----------|---------------:|-------------|
| Vivian | Chinese (Mandarin) | 3065 | Bright young female |
| Serena | Chinese (Mandarin) | 3066 | Warm, gentle young female |
| Uncle_Fu | Chinese (Mandarin) | 3010 | Seasoned male, mellow timbre |
| Dylan | Chinese (Beijing dialect) | 2878 | Youthful Beijing male |
| Eric | Chinese (Sichuan dialect) | 2875 | Lively Chengdu male |
| Ryan | English | 3061 | Dynamic male with rhythm |
| Aiden | English | 2861 | Sunny American male |
| Ono_Anna | Japanese | 2873 | Playful female |
| Sohee | Korean | 2864 | Warm female |

**Dialect handling:** Dylan and Eric have dialect overrides in `spk_is_dialect` — when selected, the language ID is replaced with the corresponding dialect token instead of standard Chinese.

### Supported Languages

Both variants support these language IDs in the codec prefix:

| Language | Codec Token ID |
|----------|---------------:|
| English | 2050 |
| German | 2052 |
| Spanish | 2054 |
| Chinese | 2055 |
| Japanese | 2058 |
| French | 2061 |
| Korean | 2064 |
| Russian | 2069 |
| Italian | 2070 |

> **Note:** The CustomVoice preset speakers are each trained on specific languages (see table above). Using a speaker with a mismatched language (e.g., Ryan with Chinese text) will produce output but quality may degrade.

### Instruction Control (CustomVoice only)

CustomVoice accepts a natural language `instruct` string to control tone, emotion, and prosody. The instruction is prepended to the text input in ChatML format and interpreted by the Talker transformer.

Examples:
- `"Speak in a cheerful, upbeat tone"`
- `"Read this slowly and solemnly"`
- `"Whisper this softly"`

### Codec Prefix Construction

**Base model (6 tokens):**
```
[think, think_bos, language_id, think_eos, pad, bos]
```

**CustomVoice with speaker (7 tokens):**
```
[think, think_bos, language_id, think_eos, pad, bos, speaker_token]
```

## Component D: Speaker Encoder (Not Yet Implemented)

> The speaker encoder is used for voice cloning in the Base model. It is not yet ported to Swift. The CustomVoice model does not need a speaker encoder — voices are selected by token ID.

ECAPA-TDNN network extracting speaker embeddings from reference audio.

| Parameter | Value |
|-----------|-------|
| Input | 128-bin mel spectrogram (24kHz, n_fft=1024) |
| Output | 1024-dim speaker embedding |
| Architecture | TimeDelayNet -> Res2Net x4 -> Squeeze-Excitation -> AttentiveStatisticsPooling |
| Channels | [512, 512, 512, 512, 1536] |

## Generation Config

| Parameter | Talker | Code Predictor |
|-----------|--------|----------------|
| Sampling | Yes | Yes |
| Temperature | 0.9 | 0.9 |
| Top-k | 50 | 50 |
| Top-p | 1.0 | 1.0 |
| Repetition penalty | 1.05 | - |
| Max tokens | 8192 | - |

## Weight Files

| File                                 | Unquantized | 4-bit Quantized  | Purpose                                   |
| ------------------------------------ | ----------- | ---------------- | ----------------------------------------- |
| `model.safetensors`                  | 1.83 GB     | 977 MB           | Talker + Code Predictor + text embeddings |
| `speech_tokenizer/model.safetensors` | 682 MB      | 651 MB (float32) | Audio codec (encoder + decoder + RVQ)     |
| `config.json`                        | 4.5 kB      | 4.5 kB           | Main model config                         |
| `speech_tokenizer/config.json`       | 2.3 kB      | 2.3 kB           | Codec config                              |
| `vocab.json` + `merges.txt`          | 4.5 MB      | 4.5 MB           | BPE tokenizer                             |

Total: ~2.5 GB unquantized, ~1.6 GB 4-bit quantized (speech tokenizer stays float32).

Pre-converted MLX weights: [mlx-community/Qwen3-TTS-12Hz-0.6B-Base-4bit](https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-0.6B-Base-4bit)
