# Qwen3.5-0.8B Chat Model

On-device LLM for chat using the Qwen3.5-0.8B hybrid architecture (DeltaNet + GatedAttention).

## Architecture

Qwen3.5-0.8B is a **hybrid recurrent-attention** model with 24 transformer layers:

| Layer type | Count | Indices | Mechanism |
|-----------|-------|---------|-----------|
| DeltaNet (linear attention) | 18 | 0,1,2, 4,5,6, 8,9,10, 12,13,14, 16,17,18, 20,21,22 | Gated delta rule recurrence |
| GatedAttention (full attention) | 6 | 3, 7, 11, 15, 19, 23 | Scaled dot-product with KV cache |

Pattern: `[linear, linear, linear, full] × 6`

### Why Hybrid?

- **DeltaNet layers** process tokens with O(1) memory per step — no KV cache needed. The recurrent state `S ∈ R^{H×D×D}` is fixed-size regardless of sequence length.
- **GatedAttention layers** (every 4th layer) provide global context via full attention, but only 6 layers need KV cache instead of 24.
- **Result**: ~75% less KV cache memory vs pure attention, enabling much longer conversations on memory-constrained devices (iOS).

### Model Dimensions

| Parameter | Value |
|-----------|-------|
| Hidden size | 1024 |
| Layers | 24 (18 DeltaNet + 6 GatedAttention) |
| Vocab | 248,320 |
| Tied embeddings | Yes (lm_head = embed_tokens) |

**DeltaNet layers:**
| Parameter | Value |
|-----------|-------|
| Heads | 16 |
| Head dim (key/value) | 128 |
| QKV projection | 6144 (3 × 16 × 128) |
| Gate projection (Z) | 2048 (2 × hidden) |
| Conv1d kernel | 4 (causal) |

**GatedAttention layers:**
| Parameter | Value |
|-----------|-------|
| Q heads | 8 |
| KV heads | 2 (GQA 4:1) |
| Head dim | 256 |
| Q projection | 4096 (Q + gate interleaved per head) |
| Partial RoPE | 25% (64 of 256 dims) |
| QK norm | RMSNorm per head |

## DeltaNet: Gated Delta Rule

Each DeltaNet layer maintains a recurrent state matrix `S ∈ R^{B×H×D_v×D_k}`:

```
1. Decay:  S = g · S                        # g = exp(-exp(A_log) · softplus(a + dt_bias))
2. Predict: kv_mem = (S · k).sum(dim=-1)    # project state onto key direction
3. Error:   delta = (v - kv_mem) · beta      # beta = sigmoid(b_proj), error correction
4. Update:  S = S + k ⊗ delta               # rank-1 update with key × delta outer product
5. Output:  y = (S · q).sum(dim=-1)          # project state onto query direction
```

**Q/K normalization**: `q = (1/D) · rms_norm(q)`, `k = (1/√D) · rms_norm(k)` (without learnable weights).

**Conv1d**: A causal depthwise conv (kernel=4) is applied to QKV before the attention, providing short-range local context.

**Gated output**: `output = out_proj(rms_norm(y) · silu(z))` where `z` comes from a separate gate projection.

## GatedAttention

Standard multi-head attention with these specifics:

- **Q+gate interleaving**: `q_proj` outputs `[B, T, 2·H·D]`. Reshape to `[B, T, H, 2·D]`, then split into Q `[B, T, H, D]` and gate `[B, T, H·D]`. This interleaved-per-head split is critical for correctness.
- **Gated output**: `output = o_proj(sdpa_result · sigmoid(gate))` — note `sigmoid`, not `silu`.
- **Partial RoPE**: Only first 25% of head dimensions (64 of 256) get rotary encoding.
- **QK norm**: Learned RMSNorm per head applied to Q and K before RoPE.

## Weight Formats

### MLX (Mac GPU)

Repo: `aufklarer/Qwen3.5-0.8B-Chat-MLX` with `int4/` and `int8/` subdirectories.

| Variant | Size | Group size |
|---------|------|-----------|
| INT4 | 404 MB | 64 |
| INT8 | 763 MB | 64 |

Weights are in safetensors format with quantized linear layers (`weight` uint32, `scales` bfloat16, `biases` bfloat16).

Key naming:
- DeltaNet: `layers.{i}.linear_attn.{in_proj_qkv, in_proj_z, in_proj_a, in_proj_b, conv1d, out_proj, norm, dt_bias, A_log}`
- GatedAttention: `layers.{i}.self_attn.{q_proj, k_proj, v_proj, o_proj, q_norm, k_norm}`
- MLP: `layers.{i}.mlp.{gate_proj, up_proj, down_proj}`

### CoreML (iOS Neural Engine)

Repo: `aufklarer/Qwen3.5-0.8B-Chat-CoreML` with `int4/` and `int8/` subdirectories.

Split into two models:
- `embedding.mlpackage` — token embedding lookup
- `decoder.mlpackage` — full transformer with stateful DeltaNet (MLState) and KV cache

## Swift API

```swift
import Qwen3Chat

// MLX (Mac)
let model = try await Qwen35MLXChat.fromPretrained(quantization: .int4)
let response = try model.generate(
    messages: [
        ChatMessage(role: .system, content: "You are a helpful assistant."),
        ChatMessage(role: .user, content: "What is 2+2?")
    ],
    sampling: ChatSamplingConfig(temperature: 0.3, maxTokens: 100)
)

// Streaming
for try await chunk in model.generateStream(messages: messages) {
    print(chunk, terminator: "")
}
```

## Token IDs

| Token | ID |
|-------|-----|
| `<\|im_start\|>` | 248045 |
| `<\|im_end\|>` | 248046 |
| `<\|endoftext\|>` | 248044 |
| `<think>` | 248068 |
| `</think>` | 248069 |
| `\n` | 198 |
| `\n\n` | 271 (BPE-merged) |

## Non-Thinking Mode

By default, generation uses `enableThinking: false` which injects an empty `<think>\n\n</think>\n\n` block in the prompt. The model then generates the answer directly without chain-of-thought reasoning.

## Conversion

- **MLX**: `scripts/convert_qwen35_chat_mlx.py` — quantizes FP16 HuggingFace weights to INT4/INT8 using MLX native quantization
- **CoreML**: `scripts/convert_qwen35_chat_coreml.py` — converts FP16 HuggingFace weights to CoreML with INT4/INT8 quantization
