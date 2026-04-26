# VibeVoice (Microsoft) — Long-form Multi-Speaker TTS

VibeVoice is a long-form, multi-speaker text-to-speech model from Microsoft. It
targets podcast / audiobook / dialogue synthesis — not short one-shot
utterances. Two variants are supported:

| Variant | Params | Focus | Context |
|---|---|---|---|
| **Realtime-0.5B** | ~500M | Low-latency streaming | ~8K tokens |
| **1.5B long-form** | ~1.5B + diffusion | Up to 90 min / 4 speakers / single pass | 64K tokens |

**The two variants have different architectures**, so they have separate
Swift classes:

- **0.5B Realtime** → `VibeVoiceTTSModel` — split LM (4-layer base + 20-layer
  TTS), type embeddings (text/speech), EOS classifier, voice cache flow
- **1.5B Long-form** → `VibeVoice15BTTSModel` — unified 28-layer Qwen2 stack,
  dual encoders (acoustic + semantic, summed at audio prompt positions),
  structured prompt (`<system>` + voice exemplar + `<text input>` +
  `<speech output>` + `<speech_start>`), LM-head token sampling branched
  on `<speech_diffusion>` / `<speech_end>` / text tokens

Both train on English + Chinese only. Non-EN/ZH input produces unintelligible
audio per Microsoft's model card.

- **License**: MIT
- **Output**: 24 kHz mono Float32 PCM
- **Backend**: MLX (Apple Silicon GPU)
- **HF (upstream)**: [microsoft/VibeVoice-Realtime-0.5B](https://huggingface.co/microsoft/VibeVoice-Realtime-0.5B), [microsoft/VibeVoice-1.5B](https://huggingface.co/microsoft/VibeVoice-1.5B)

## 0.5B Realtime pipeline

```
text (EN/ZH) → Qwen2.5 BPE tokenizer → token ids
                                           ↓
                    language_model (4 lower Qwen2 layers)
                                           ↓
                tts_language_model (20 upper Qwen2 layers, GQA)
                                           ↓
            TTS input-type embedding (text=1, speech=0)
                                           ↓
         sample_speech_tokens: 20-step DPM-Solver diffusion head
                 + classifier-free guidance (cfg=1.3 default)
                                           ↓
              speech latents (acoustic_vae_dim=64, 7.5 Hz)
                                           ↓
            acoustic_tokenizer.decode (σ-VAE streaming conv stack)
                                           ↓
                        audio (24 kHz mono Float32)
```

At inference, the model alternates between:
1. **Text window** — consumes `TTSConstants.textWindowSize = 5` tokens, runs
   the split LM (base LM → TTS LM) to produce a conditioning hidden state.
2. **Speech window** — emits `TTSConstants.speechWindowSize = 6` speech latents
   via diffusion + decodes each to waveform via the acoustic tokenizer.

End-of-speech is detected by a per-step binary classifier on the TTS LM's
final hidden state (`> 0.5` → stop).

## 1.5B long-form pipeline

```
reference audio (24 kHz mono) ──┬─ acoustic_tokenizer.encode → mean (64-d)
                                │      ↓
                                │      acoustic_connector ──┐
                                │                           +  combined_audio_embed
                                └─ semantic_tokenizer.encode → mean (128-d)
                                       ↓                    │
                                       semantic_connector ──┘

text + structured prompt → embed_tokens → text embeds
       ↓                           ↓
       merge: text_embeds[mask] = combined_audio_embed
       ↓
unified Qwen2 (28 layers, hidden 1536, GQA 12/2)
       ↓
LM head (tied embed_tokens.T) → vocab logits → argmax → next_token

  if next_token == <speech_diffusion>:
      sample acoustic latent via DPM-Solver (cfg=1.5) → decode → audio chunk
      feed acoustic_connector(latent) back as next embed
  if next_token == <speech_end>: terminate
  else (text token): embed + advance
```

Structured prompt format (canonical, from
`vibevoice/processor/vibevoice_processor.py`):

```
<bos>
" Transform the text provided by various speakers into speech output, ...\n"
" Speaker 0:" <speech_start> [vae_token]*N <speech_end> "\n"
" Text input:\n Speaker 0:<text>\n"
" Speech output:\n" <speech_start>
```

The `[vae_token]` placeholders are replaced with `combined_audio_embed` rows
via an `acoustic_input_mask` at forward time. Generation begins after the
final `<speech_start>` cue.

## Voice cloning — 0.5B path (voice cache)

Speaker identity does **not** come from a reference waveform at generation
time. Instead, a precomputed **voice cache** (`.safetensors`) holds the
conditioning KV cache + hidden states for a specific speaker:

- `lm_hidden`, `tts_lm_hidden`, `neg_tts_lm_hidden`
- `lm_key_{i}`, `lm_value_{i}` for each base-LM layer
- `tts_lm_key_{i}`, `tts_lm_value_{i}` for each TTS-LM layer
- `neg_tts_lm_key_{i}`, `neg_tts_lm_value_{i}` for CFG negative conditioning

Loading a voice cache is instantaneous at runtime. Mint your own from any
reference audio + transcript via `VibeVoiceTTSModel.encodeAndSaveVoice(...)`
or the CLI:

```bash
audio vibevoice-encode-voice reference.wav "exact transcript" \
    --output voices/my-voice.safetensors
```

The encoder runs the audio through `acoustic_tokenizer.encode` and the
transcript through both LMs, capturing per-layer KV caches and hidden
states. Encoding is fast: a 17-second clip on M2 Max takes ~2 s.

## Voice conditioning — 1.5B path (single-shot)

The 1.5B variant doesn't use precomputed voice caches. Reference audio +
transcript + text-to-synthesize go in a single call:

```swift
let tts = try await VibeVoice15BTTSModel.fromPretrained()
let pcm = try await tts.generate(
    text: "Long English script.",
    referenceAudio: refSamples,           // [Float] mono 24 kHz
    referenceTranscript: "",
    sampleRate: 24000
)
```

Internally `generate(...)`:
1. Resamples `referenceAudio` to 24 kHz
2. Runs `acoustic_tokenizer.encode` and `semantic_tokenizer.encode` on the
   reference (both return mean directly — no scaling, per `vllm_plugin/model.py`)
3. Computes `audio_embed = acoustic_connector(ac) + semantic_connector(sem)`
4. Builds the structured prompt with `<speech_start>`/`<speech_end>` markers
   and `[vae_token]` placeholders for the audio segment
5. Embeds text via `embed_tokens`, replaces at `acoustic_input_mask` positions
   with `audio_embed` rows
6. Forwards through the unified LM, capturing the prefill hidden state
7. Token-samples each step branched on `<speech_diffusion>` / `<speech_end>` /
   text — runs diffusion only when LM emits `<speech_diffusion>`

## Model parameters (Realtime-0.5B)

| Component | Params |
|---|---|
| Qwen2 backbone (24 layers, hidden 896, intermediate 4864, GQA 14/2) | ~460M |
| σ-VAE acoustic tokenizer (encoder + decoder) | ~40M |
| Diffusion head (4 layers, adaLN modulation, latent 64) | ~4M |
| Speech connector + TTS-input-types + EOS classifier | < 1M |
| **Total** | **~500M** |

## Weight bundles

| Bundle | Quantization | Size |
|---|---|---|
| `microsoft/VibeVoice-Realtime-0.5B` | BF16 | ~1 GB |
| `aufklarer/VibeVoice-Realtime-0.5B-MLX-INT4` | Qwen2 INT4, tokenizer+diffusion FP16 | ~350 MB |
| `aufklarer/VibeVoice-Realtime-0.5B-MLX-INT8` | Qwen2 INT8 | ~570 MB |
| `microsoft/VibeVoice-1.5B` | BF16 | ~3 GB |
| `aufklarer/VibeVoice-1.5B-MLX-INT4` | Qwen2 INT4 | ~1 GB |

Quantization is produced by `models/vibevoice/export/convert.py` in
`soniqo/speech-models` and uses MLX group-wise affine quantization (32-group,
INT4/INT8). Embeddings, norms, acoustic-tokenizer convs, and the EOS
classifier are kept in their source dtype.

## Source files

```
Sources/VibeVoiceTTS/
  VibeVoiceTTSModel.swift          Primary public API (fromPretrained, loadVoice, generate)
  VibeVoiceTTS+Protocols.swift     SpeechGenerationModel conformance
  Constants.swift                  AudioConstants, TTSConstants, TokenConstants
  Models/
    VibeVoiceStreamModel.swift     Split LM + inference loop + voice-cache loader
    Qwen2*.swift                   Qwen2 backbone layers
    AcousticTokenizer*.swift       σ-VAE encoder/decoder
    DiffusionHead.swift            4-layer DDPM head with adaLN modulation
    EOSClassifier.swift
  Layers/                          TimestepEmbedder, Normalization, StreamingConv1d
  Inference/                       DPMSolver, KVCache, WeightLoader
  Quantization/                    VibeVoiceQuantizer + manifest
  Errors/VibeVoiceError.swift
```

## License

Microsoft's VibeVoice model weights are MIT. The long-form 1.5B is currently
disabled by Microsoft on the upstream HF repo — check availability before
running.
