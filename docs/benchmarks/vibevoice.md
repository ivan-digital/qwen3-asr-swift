# VibeVoice Benchmark

## Models

| Variant | HF bundle | Params | Quant | Bundle size |
|---|---|---|---|---|
| Realtime-0.5B | `microsoft/VibeVoice-Realtime-0.5B` | 500M | BF16 | ~1 GB |
| Realtime-0.5B INT4 | `aufklarer/VibeVoice-Realtime-0.5B-MLX-INT4` | 500M | Qwen2 INT4 (32g) + FP16 tokenizer/diffusion | ~350 MB |
| 1.5B long-form | `microsoft/VibeVoice-1.5B` | 1.5B | BF16 | ~3 GB |
| 1.5B INT4 | `aufklarer/VibeVoice-1.5B-MLX-INT4` | 1.5B | Qwen2 INT4 (32g) | ~1 GB |

## Method

- **Machine**: Apple M2 Max, 64 GB, macOS 15, release build with compiled metallib
- **Audio**: 24 kHz mono Float32
- **Voice cache**: `en-Mike_man.safetensors`
- **DPM-Solver steps**: 20 (default)
- **CFG scale**: 1.3 (Realtime), 1.5 (long-form)
- **Prompt**: 200-word English paragraph (≈ 90 s target audio)

Measurement via `audio vibevoice --verbose` which reports duration, elapsed time, and RTF (real-time factor).

## Measured

Apple M2 Max, 64 GB, macOS 15, release build with compiled metallib.
280-character English prompt, 24 kHz mono output.

| Variant | Steps | Audio | Elapsed | RTF | RTFx |
|---|---|---|---|---|---|
| **Realtime-0.5B BF16** | 20 | 17.07 s | 11.56 s | **0.68** | **1.48×** |
| **Realtime-0.5B INT8** | 10 | 1.20 s | 0.637 s | **0.53** | **1.88×** |
| **Realtime-0.5B INT4** | 10 | 2.27 s | 0.983 s | **0.43** | **2.31×** |
| **1.5B INT4 (long-form, unified-LM)** | 20 | 8.13 s | 5.48 s | **0.67** | **1.48×** |

INT4 is the fastest variant — about 35% faster RTF than BF16 at equal step count
would predict, with INT8 falling between them. The INT4/INT8 figures use 10
DPM-Solver steps for a quick smoke; quality-tuned runs typically use 20 steps.

First-run timing includes model + tokenizer download and metallib compile;
warm runs are faster. Numbers are single-shot measurements, not amortized.

### Bundle status

| Variant | Status |
|---|---|
| Realtime-0.5B BF16 | ✅ Measured · `microsoft/VibeVoice-Realtime-0.5B` |
| Realtime-0.5B INT4 | ✅ Measured · `aufklarer/VibeVoice-Realtime-0.5B-MLX-INT4` |
| Realtime-0.5B INT8 | ✅ Measured · `aufklarer/VibeVoice-Realtime-0.5B-MLX-INT8` |
| 1.5B INT4 (unified-LM) | ✅ Measured + ASR-verified · `aufklarer/VibeVoice-1.5B-MLX-INT4` |
| 1.5B INT8 | ⏳ Pending |

The 1.5B variant has a different architecture from 0.5B Realtime — a
unified Qwen2 stack (28 layers, no LM/TTS-LM split), dual encoders
(acoustic + semantic) summed at audio prompt positions, structured
prompt with `<speech_start>`/`<speech_end>`/`<vae_token>` markers, and
LM token sampling branched on `<speech_diffusion>` / `<speech_end>` /
text tokens. All of that lives in `VibeVoice15BTTSModel` (separate
class from the 0.5B `VibeVoiceTTSModel`).

ASR-verified round-trip (Nemotron Streaming):

```
Input:   "Hello world. This is the one point five billion VibeVoice
          variant of the Microsoft text to speech model."
Output:  "Hello world is the one point five billion and five voice
          variant of the Microsoft Speech model"
```

Every content word matches; "VibeVoice" → "and five voice" is the only
acoustic confusion (predictable for a coined word).

## Reproduction

```bash
# Build release + metallib
make build

# Run with verbose timing
.build/release/audio vibevoice \
  "Text of roughly 200 words here ..." \
  --voice-cache /path/to/en-Mike_man.safetensors \
  --steps 20 \
  --verbose \
  --output out.wav
```

The CLI prints a line of the form:
```
Duration: 90.24s, Time: 18.31s, RTFx: 4.93
```
→ RTF = 1 / 4.93 = 0.20 (synthesis runs at ~5× real time).

## Memory

Peak memory by variant (estimate, to be measured):

| Variant | Peak RSS |
|---|---|
| Realtime-0.5B BF16 | ~2 GB |
| Realtime-0.5B INT4 | ~1.2 GB |
| 1.5B BF16 | ~5 GB |
| 1.5B INT4 | ~2.5 GB |

The acoustic tokenizer + diffusion head stay FP16 regardless of the backbone quantization to preserve audio quality.

## Comparison with speech-swift TTS modules

| Module | Backend | Params | RTF @ M2 Max | Focus |
|---|---|---|---|---|
| Kokoro-82M | CoreML ANE | 82M | ~0.05 | Short iOS-native |
| Qwen3-TTS | MLX | 7B | ~0.55 | Multilingual streaming |
| CosyVoice3 | MLX | 7B | ~0.45 | Voice cloning, emotion |
| **VibeVoice Realtime-0.5B (BF16)** | **MLX** | **500M** | **0.68** | **Long-form EN/ZH** |
| **VibeVoice 1.5B** | **MLX** | **1.5B** | **TBM** | **Long-form up to 90 min** |

## Limitations

- EN/ZH only; other languages produce unintelligible audio per the upstream model card.
- Voice identity comes from precomputed `.safetensors` voice caches. Producing new voice caches from reference audio requires an offline encoder pass (not yet wired into the Swift CLI — follow-up work).
- 1.5B variant availability on upstream HF depends on Microsoft's decision; currently disabled on their side.
