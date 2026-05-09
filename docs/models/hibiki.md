# Hibiki — Streaming Speech-to-Speech Translation (Kyutai)

Hibiki is Kyutai's streaming speech-to-speech translation model, built on the
Moshi/Mimi stack (same RVQ codec + delay-pattern decoding as PersonaPlex). This
repo currently ships the **Zero-3B** variant.

## Variants and language coverage

| Variant | Source → Target | Params | Status |
|---|---|---|---|
| Hibiki 1B | FR → EN | 1.7 B | converter only (`models/hibiki/export/convert.py --variant 1b`) |
| Hibiki 2B | FR → EN | 2.7 B | converter only (`models/hibiki/export/convert.py --variant 2b`) |
| **Hibiki Zero-3B** | **FR / ES / PT / DE → EN** | **3.1 B** | **shipped** (`Sources/HibikiTranslate/`) |

Pre-converted MLX weights (CC-BY-4.0):
- `aufklarer/Hibiki-Zero-3B-MLX-4bit` (~2.7 GB)
- `aufklarer/Hibiki-Zero-3B-MLX-8bit` (~3.9 GB)

## Architecture

```
Source-language audio (24 kHz)
        │
        ▼  Mimi streaming encoder (12.5 Hz, 16 codebooks, RVQ)
        │
        ▼  Source codebooks → temporal audio embeddings (streams 1..16)
        │
[Temporal Transformer · GQA · 28 layers · dim=2048]
        │  (text + 32 audio streams summed; 16 KV heads with kv_repeat=2)
        ▼
[Depformer · 6 layers · 16-step scheduled MultiLinear]
        │  (9 unique slice weights, schedule = [0..8, 8×8])
        ▼  Target codebooks (streams 17..32)
        │
        ▼  Mimi streaming decoder (12.5 Hz, 16 codebooks)
        │
Target-language audio (24 kHz)
```

### Architectural deltas vs PersonaPlex

| Component | PersonaPlex 7B | Hibiki Zero-3B |
|---|---|---|
| Temporal dim / layers | 4096 / 32 | **2048 / 28** |
| Heads / KV heads | 32 / 32 (MHA) | **16 / 8 (GQA, kv_repeat=2)** |
| Hidden scale (FFN) | 4.125 → 11264 intermediate | **6 → 8192 intermediate** |
| RoPE | interleaved (`traditional: true`) | **split-half (`rope_concat`, `traditional: false`)** |
| RoPE max period | 10000 | **20000** |
| Audio codebooks (n_q / dep_q) | 16 / 8 (8 user + 8 agent) | **32 / 16 (16 source + 16 target)** |
| Streams | 17 (1 + 8 + 8) | **33 (1 + 16 + 16)** |
| Max delay | 1 | **2** |
| Conditioner | none (system prompt) | **none (Zero is unconditional)** |
| Voice presets | 18 | **none** |
| Depformer schedule | one slice per step (16 unique) | **9 unique slices over 16 steps** |
| Depformer dim_feedforward | 2816 (depformer.dim×2/3×4.125) | **4096 (depformer.dim×2/3×6)** |
| Tokenizer | SPM 32k (tokenizer_spm_32k_3.model) | **SPM 48k (tokenizer_spm_48k_multi6_2.model)** |

## Synchronous 1:1 generation

Hibiki is **synchronous**: each Mimi frame of input (80 ms) produces exactly
one Mimi frame of output. The generation loop runs `tSrc` steps where
`tSrc = ceil(input_pcm_len / 1920)`. Output duration ≈ input duration. There
is no separate prefill phase (no voice prompt, no system prompt).

## Files

```
Sources/HibikiTranslate/
  Configuration.swift              HibikiConfig.zero3B + JSON loader
  HibikiTemporalTransformer.swift  GQA + rope_concat (28 layers, dim=2048)
  HibikiDepformer.swift            ScheduledMultiLinear (9 unique slices)
  HibikiTranslateModel.swift       Module shell + fromPretrained()
  HibikiTranslate.swift            translate() / translateStream() driver
  HibikiWeightLoading.swift        4-file safetensors loader
```

## Usage

```swift
import HibikiTranslate
import AudioCommon

let model = try await HibikiTranslateModel.fromPretrained(
    modelId: HibikiTranslateModel.defaultModelId  // 4-bit
)

let pcm = try AudioFileLoader.load(url: input, targetSampleRate: 24000)
let (englishAudio, textTokens) = model.translate(
    sourceAudio: pcm, sourceLanguage: .fr, verbose: true
)
try WAVWriter.write(samples: englishAudio, sampleRate: 24000, to: output)
```

CLI:

```bash
audio audio-translate input_fr.wav --output translated_en.wav --source-lang fr
audio audio-translate input.wav --quantization 8bit --verbose --transcript
```

## Conversion

The Python converter at `models/hibiki/export/convert.py` (in the speech-models
repo) handles all three Hibiki variants:

```bash
python convert.py --variant 3b-zero --bits 4 \
    --upload --repo-id aufklarer/Hibiki-Zero-3B-MLX-4bit
python convert.py --variant 3b-zero --bits 8 \
    --upload --repo-id aufklarer/Hibiki-Zero-3B-MLX-8bit
```

It downloads the upstream PyTorch bf16 weights from `kyutai/hibiki-{1b,2b}-pytorch-bf16`
or `kyutai/hibiki-zero-3b-pytorch-bf16` and produces MLX-compatible safetensors:

- `temporal.safetensors` (quantized)
- `depformer.safetensors` (quantized; per-step slices packed by step index)
- `embeddings.safetensors` (BF16; text + 32 audio + per-codebook output heads)
- `mimi.safetensors` (Mimi codec, copied as-is)
- `tokenizer_spm_48k_multi6_2.model`
- `config.json`

## Known limitations

- **Translation quality is broken — open bug.** A closed-loop round-trip test
  (`testClosedLoopTTSToHibikiToASR`: known FR text → Qwen3TTS → FR audio →
  Hibiki → EN audio → Parakeet ASR) produces 0/3 keyword hits on simple
  in-distribution sentences. Decoding Hibiki's inner-monologue text tokens
  (which the model emits at 12.5 Hz alongside the audio codes) reveals that
  the temporal transformer is producing valid English subwords that are
  **completely unrelated to the source content** — meaning the audio
  embeddings aren't conditioning the LM as intended.
  The structural pipeline (weight load, forward shape, durations, RTF) is
  correct end-to-end. The bug is in the audio-conditioning path. Most likely
  suspects: GQA KV-cache head broadcast layout, or a per-stream embedding
  scaling upstream Moshi applies that this port is missing. Needs comparison
  against Kyutai's `hibiki-rs` reference output to isolate further.
- **`translateStream()` is single-chunk** — The streaming entry point currently
  wraps `translate()` and emits one final `AudioChunk`. True per-chunk Mimi
  streaming decode is a v2 follow-up.
- **No SentencePiece decoder** — `translate()` returns text token IDs but
  doesn't decode them through `tokenizer_spm_48k_multi6_2.model`. The CLI
  `--transcript` flag prints raw token IDs.
- **Quantization-only** — The repo currently exposes Zero-3B 4-bit and 8-bit
  only. The 1B and 2B converters exist (`models/hibiki/export/convert.py`)
  but the Swift driver targets Zero-3B's GQA + rope_concat + non-conditioned
  layout. Adding 1B/2B variants to the Swift side is a follow-up.

## References

- [Hibiki paper (Kyutai, 2025)](https://arxiv.org/abs/2502.03382)
- [Kyutai Hibiki repo](https://github.com/kyutai-labs/hibiki)
- [Moshi-swift reference](https://github.com/kyutai-labs/moshi-swift) (lib-level Hibiki support)
- [PersonaPlex doc](personaplex.md) (shared Mimi/Depformer stack)
