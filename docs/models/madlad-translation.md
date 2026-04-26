# MADLAD-400 Translation Model

On-device, many-to-many machine translation across 400+ languages using the [`google/madlad400-3b-mt`](https://huggingface.co/google/madlad400-3b-mt) (Apache 2.0) T5 v1.1 encoder-decoder, quantized to INT4/INT8 for Apple Silicon via MLX.

## Architecture

T5 v1.1 encoder-decoder with relative position bias. ~3B parameters total.

| Block | Count | Inner shape | Notes |
|-------|-------|-------------|-------|
| Encoder | 32 | self-attn + gated-GeLU FFN | bidirectional, RMSNorm pre-norm |
| Decoder | 32 | self-attn (causal) + cross-attn + gated-GeLU FFN | autoregressive, KV cache + cross-attn cache |

### Dimensions

| Parameter | Value |
|-----------|-------|
| `d_model` | 1024 |
| `d_kv` (head dim) | 128 |
| `num_heads` | 16 |
| `inner_dim` | 16 × 128 = 2048 |
| `d_ff` | 8192 |
| Vocab | 256,000 (SentencePiece Unigram, 400+ `<2xx>` language tokens) |
| Tied embeddings | **No** — separate `lm_head` |

### Attention specifics (T5 quirks)

- **No `1/√d_k` scaling.** Attention scores are unscaled. Position information arrives via an additive **relative position bias** rather than position embeddings.
- **Relative position bias** is a learned `[num_buckets=32, num_heads=16]` table indexed by bucketed `key_pos − query_pos`. Half the buckets handle exact small distances; the other half are log-spaced up to `max_distance=128`. The encoder uses **bidirectional** bucketing (sign-distinguishing); the decoder uses **unidirectional** (past-only). The bias table lives only on the **first layer** of each stack — subsequent layers receive the precomputed bias as input and reuse it.
- **No biases** on Q/K/V/O projections.
- **Cross-attention** has no relative position bias and no causal mask.
- **Causal mask** in decoder self-attention is added on top of the position bias as an additive `-FLT_MAX` mask above the diagonal.

### FFN

T5 v1.1 gated-GeLU: `wo(gelu_new(wi_0(x)) * wi_1(x))`. The "new" GeLU is the `tanh` approximation `0.5x · (1 + tanh(√(2/π)(x + 0.044715 x³)))`.

### Tokenizer

SentencePiece Unigram, 256k vocab. 400+ `<2xx>` language target tokens (e.g. `<2en>`, `<2es>`, `<2zh>`) are stored as user-defined pieces with score `0.0` but **not** registered in `added_tokens` — direct vocab lookup via `convertTokenToId` is required, since plain `tokenizer.encode("<2es>")` runs the Unigram algorithm and splits the token into sub-pieces.

Special tokens: `decoder_start_token_id = 0`, `pad_token_id = 1`, `eos_token_id = 2` (all different from vanilla T5).

A leading `▁` (U+2581, SentencePiece word-boundary marker, id 805) is prepended before the language token in MADLAD's training format. `Sources/MADLADTranslation/MADLADTokenizer.swift` reproduces this:

```
encode(text="Hello, how are you?", target="es")
  → [▁=805, <2es>=40, ▁Hello=88912, ',', ▁how, ▁are, ▁you, '?', </s>=2]
```

## Weight key layout

Direct 1:1 with HF safetensors except for one rename. The conversion script at `speech-models/models/madlad-translation/export/convert_mlx.py` handles it:

| Swift module path | HF key | Notes |
|---|---|---|
| `shared` | `decoder.embed_tokens.weight` | **Renamed.** MADLAD has no `shared.weight` or `encoder.embed_tokens.weight` in HF — the encoder reuses the decoder's embedding table. |
| `lm_head` | `lm_head.weight` | Separate output projection (untied). |
| `encoder.block.{i}.layer.0.SelfAttention.{q,k,v,o}` | same | |
| `encoder.block.0.layer.0.SelfAttention.relative_attention_bias` | same | Only block 0; FP16, not quantized. |
| `encoder.block.{i}.layer.0.layer_norm` | same | |
| `encoder.block.{i}.layer.1.DenseReluDense.{wi_0,wi_1,wo}` | same | |
| `encoder.block.{i}.layer.1.layer_norm` | same | |
| `encoder.final_layer_norm` | same | |
| `decoder.block.{i}.layer.0.SelfAttention.{q,k,v,o}` | same | |
| `decoder.block.0.layer.0.SelfAttention.relative_attention_bias` | same | |
| `decoder.block.{i}.layer.0.layer_norm` | same | |
| `decoder.block.{i}.layer.1.EncDecAttention.{q,k,v,o}` | same | Cross-attention. |
| `decoder.block.{i}.layer.1.layer_norm` | same | |
| `decoder.block.{i}.layer.2.DenseReluDense.{wi_0,wi_1,wo}` | same | |
| `decoder.block.{i}.layer.2.layer_norm` | same | |
| `decoder.final_layer_norm` | same | |

### Quantization

`mx.quantize(group_size=64, bits∈{4,8})` packs each linear weight into uint32 + per-group fp16 `scales` and `biases`. Quantized: `q/k/v/o`, `wi_0/wi_1/wo`, `lm_head`, `shared` (input embedding). Kept as fp16: all `*.layer_norm.weight`, `final_layer_norm.weight`, `relative_attention_bias.weight` (small).

| Variant | Size |
|---------|------|
| INT4 | ~1.7 GB |
| INT8 | ~3.1 GB |

Both variants live under `int4/` and `int8/` subdirectories of [`aufklarer/MADLAD400-3B-MT-MLX`](https://huggingface.co/aufklarer/MADLAD400-3B-MT-MLX).
