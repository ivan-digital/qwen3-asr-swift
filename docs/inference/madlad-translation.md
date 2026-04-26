# MADLAD Translation Inference Pipeline

Many-to-many translation across 400+ languages, MLX on Apple Silicon. Source language is auto-detected; the only required input is the **target** language.

## Pipeline

```
"Hello, how are you?" + target="es"
       │
       ▼  MADLADTokenizer.encode
[▁, <2es>, ▁Hello, ',', ▁how, ▁are, ▁you, '?', </s>]
       │
       ▼  encoder (32 T5Blocks, run ONCE)
encoder_output [1, T_src, 1024]
       │
       ▼  decoder loop (start with token 0 = <pad>)
       │   ── self-attn KV cache grows per step
       │   ── cross-attn KV cache computed once, reused
"Hola, ¿cómo estás?"
```

## Stage 1: Tokenize source

`MADLADTokenizer` wraps a HuggingFace `Tokenizer` (loaded from `tokenizer.json`) and prepends two synthetic prefix tokens that match HF's training format:

```
[▁, <2{target}>, ...sentencepiece-encoded source..., </s>]
```

The leading `▁` (U+2581, id 805) is the SentencePiece sequence-start word-boundary marker. The `<2{target}>` token tells MADLAD what language to translate *into* — source language is inferred from the text.

`<2{target}>` is resolved via direct vocab lookup (`convertTokenToId` + round-trip verify), not via `tokenizer.encode("<2es>")`. The Unigram tokenizer would otherwise split it into sub-pieces because it isn't in `added_tokens`.

## Stage 2: Encode source

`MADLADTranslationModel.encode(inputIds:)` runs the 32-layer encoder once. Internally:

- Embedding lookup (`shared`, INT4/INT8 quantized)
- 32× T5 encoder blocks: `pre-RMSNorm → self-attn (with rel-pos bias on layer 0, propagated thereafter) → residual → pre-RMSNorm → gated-GeLU FFN → residual`
- `encoder.final_layer_norm`

Result: `[1, T_src, d_model]` hidden states.

## Stage 3: Autoregressive decode

```swift
var caches = (0..<numDecoderLayers).map { _ in DecoderLayerCache() }
var nextToken = config.decoderStartTokenId  // 0 = <pad>

while !done {
    let logits = model.decodeStep(
        inputIds: MLXArray([Int32(nextToken)]).expandedDimensions(axis: 0),
        encoderOutput: encoderOutput,
        caches: &caches)
    nextToken = sample(logits)
    if nextToken == config.eosTokenId { break }   // </s> = 2
}
```

Each layer carries two caches:

| Cache | Grows | When updated |
|-------|-------|--------------|
| `selfAttn` | yes — past `[K,V]` of decoder positions, concatenated each step | every step |
| `crossAttn` | no — `[K,V]` projected from the encoder output | computed once at first step, reused for all subsequent steps |

The relative position bias is computed only on layer 0 (with the correct `offset = past_length` so query position aligns with absolute decoder position) and propagated to layers 1..31.

## Sampling

- `temperature ≤ 0` → **greedy** (argmax). Default. Recommended for translation.
- `temperature > 0` → temperature scaling + optional `topK` cutoff + `topP` (nucleus) cutoff + `repetitionPenalty` over the last 64 generated tokens.

## Streaming

`MADLADTranslator.translateStream(...)` is an `AsyncThrowingStream<String, Error>` that yields incremental text fragments. Each step decodes the **full** accumulated token list and emits the suffix diff vs the previously-yielded text — decoding tokens one at a time strips the SentencePiece `▁` word-boundary marker and collapses spaces ("Hola mundo" → "Holamundo").

## CLI

```bash
audio translate "Hello" --to es
audio translate "Bonjour" --to en --quantization int8
audio translate --to fr --json   # JSON output with timing metrics
audio transcribe meeting.wav | audio translate --to es   # pipe from ASR
audio translate "Hello world" --to es --stream            # incremental output
```

## Performance characteristics

- INT4 weights: ~1.7 GB on disk, ~2 GB in memory at runtime.
- Encoder runs once per source sentence (parallel over all source tokens).
- Decoder is the bottleneck — autoregressive, one forward pass per output token.
- Cross-attention K/V is computed once (see cache table above) — saves ~32× cross-attn projection work on long outputs.

## Failure modes to know about

| Symptom | Cause | Fix |
|---|---|---|
| All-zero logits | `shared.weight` not loaded — converter dropped it as a "duplicate" | converter remaps `decoder.embed_tokens.weight` → `shared.weight` |
| Repetitive degenerate output ("¿ ¿ ¿ donde está la?") | Missing leading `▁` token before `<2{lang}>` in source | prepend id 805 (`▁`) in `MADLADTokenizer.encode` |
| `unsupportedLanguage` thrown for known language | Wrong tokenizer ids — using T5 defaults instead of MADLAD's `eos=2, pad=1` | pass `config.eosTokenId` / `config.padTokenId` to `MADLADTokenizer.load` |
| Spaces dropped in streaming output | Per-token decode strips `▁` | decode full sequence each step, yield suffix diff |
