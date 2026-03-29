#!/usr/bin/env python3
"""Convert Qwen3-TTS to CoreML for Neural Engine inference.

Architecture patterns for ANE optimization:
1. Conv2d instead of Linear (ANE-optimal NCHW layout)
2. NCHW data layout: (batch, channels, 1, seq_len)
3. Precomputed RoPE cos/sin tables as model inputs
4. Stacked KV cache with scatter-write mask
5. W8A16 palettization (sensitivity-aware)
6. -1e4 mask values (not -inf, prevents FP16 NaN)

Usage:
    python scripts/convert_qwen3_tts_coreml.py --output models/Qwen3-TTS-CoreML
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from safetensors import safe_open

sys.stdout.reconfigure(line_buffering=True)

MASK_VALUE = -1e4  # -10000, NOT -inf (prevents FP16 NaN in softmax)


# ============================================================================
# Conv2d-based modules (ANE-optimal)
# ============================================================================

class RMSNormConv(nn.Module):
    """RMSNorm on channel dimension of NCHW tensors."""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.eps = eps

    def forward(self, x):
        # x: (B, C, 1, T)
        norm = torch.rsqrt(x.pow(2).mean(dim=1, keepdim=True) + self.eps)
        return x * norm * self.weight


class Conv2dLinear(nn.Module):
    """Conv2d with kernel_size=1, equivalent to Linear but ANE-friendly."""
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_features, out_features, kernel_size=1, bias=bias)

    def forward(self, x):
        return self.conv(x)


class TalkerAttentionV2(nn.Module):
    """Attention with Conv2d projections and precomputed RoPE."""
    def __init__(self, hidden=1024, num_heads=16, num_kv_heads=8, head_dim=128):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.d = num_kv_heads * head_dim  # KV dim per layer
        self.scale = head_dim ** -0.5

        self.q_proj = Conv2dLinear(hidden, num_heads * head_dim)
        self.k_proj = Conv2dLinear(hidden, num_kv_heads * head_dim)
        self.v_proj = Conv2dLinear(hidden, num_kv_heads * head_dim)
        self.o_proj = Conv2dLinear(num_heads * head_dim, hidden)
        self.q_norm = RMSNormConv(head_dim)
        self.k_norm = RMSNormConv(head_dim)

    def forward(self, x, rope_cos, rope_sin, key_padding_mask,
                kv_cache_update_mask, layer_key_cache, layer_value_cache):
        """
        x: (B, C, 1, 1) — single token embedding
        rope_cos, rope_sin: (B, head_dim, 1) — precomputed for this position
        key_padding_mask: (B, max_seq) — additive mask (-1e4 for masked)
        kv_cache_update_mask: (B, max_seq) — one-hot write position
        layer_key_cache: (B, kv_dim, 1, max_seq)
        layer_value_cache: (B, kv_dim, 1, max_seq)
        """
        B = x.shape[0]

        # Project Q, K, V via Conv2d
        q = self.q_proj(x)  # (B, num_heads*head_dim, 1, 1)
        k = self.k_proj(x)  # (B, num_kv_heads*head_dim, 1, 1)
        v = self.v_proj(x)  # (B, num_kv_heads*head_dim, 1, 1)

        # Q/K norm on head_dim before multi-head reshape
        # Reshape to (B*heads, head_dim, 1, 1) for RMSNormConv
        q_flat = q.view(B * self.num_heads, self.head_dim, 1, 1)
        k_flat = k.view(B * self.num_kv_heads, self.head_dim, 1, 1)
        q_flat = self.q_norm(q_flat)
        k_flat = self.k_norm(k_flat)

        # Reshape to multi-head: (B, heads, head_dim, 1)
        q = q_flat.view(B, self.num_heads, self.head_dim, 1)
        k = k_flat.view(B, self.num_kv_heads, self.head_dim, 1)
        v = v.view(B, self.num_kv_heads, self.head_dim, 1)

        # Apply RoPE (rotate-half on head_dim, which is dim=2 here)
        def rotate_half(t):
            t1 = t[:, :, :self.head_dim // 2, :]
            t2 = t[:, :, self.head_dim // 2:, :]
            return torch.cat([-t2, t1], dim=2)

        cos = rope_cos[:, None, :, :]  # (B, 1, head_dim, 1) — broadcasts over heads
        sin = rope_sin[:, None, :, :]
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin

        # Flatten back: (B, kv_dim, 1, 1)
        k_flat = k.view(B, self.d, 1, 1)
        v_flat = v.view(B, self.d, 1, 1)

        # Scatter-write into KV cache
        update_mask = kv_cache_update_mask[:, None, None, :]  # (B, 1, 1, max_seq)
        new_key_cache = layer_key_cache * (1.0 - update_mask) + k_flat * update_mask
        new_value_cache = layer_value_cache * (1.0 - update_mask) + v_flat * update_mask

        # GQA expand cached keys/values
        kv_per_head = self.d // self.num_kv_heads  # = head_dim
        if self.num_kv_heads < self.num_heads:
            rep = self.num_heads // self.num_kv_heads
            kc = new_key_cache.view(B, self.num_kv_heads, kv_per_head, -1)
            vc = new_value_cache.view(B, self.num_kv_heads, kv_per_head, -1)
            kc = kc.repeat_interleave(rep, dim=1)
            vc = vc.repeat_interleave(rep, dim=1)
        else:
            kc = new_key_cache.view(B, self.num_heads, kv_per_head, -1)
            vc = new_value_cache.view(B, self.num_heads, kv_per_head, -1)

        # Attention: Q @ K^T / sqrt(d) + mask
        # q: (B, heads, head_dim, 1), kc: (B, heads, head_dim, max_seq)
        scores = torch.einsum('bhdi,bhdj->bhij', q, kc) * self.scale  # (B, heads, 1, max_seq)
        scores = scores + key_padding_mask[:, None, None, :]  # broadcast mask
        attn = torch.softmax(scores, dim=-1)

        # Attn @ V
        # attn: (B, heads, 1, max_seq), vc: (B, heads, head_dim, max_seq)
        out = torch.einsum('bhij,bhdj->bhdi', attn, vc)  # (B, heads, head_dim, 1)
        out = out.reshape(B, self.num_heads * self.head_dim, 1, 1)

        # Output projection
        out = self.o_proj(out)  # (B, C, 1, 1)

        return out, new_key_cache, new_value_cache


class TalkerMLPV2(nn.Module):
    """SwiGLU MLP with Conv2d layers."""
    def __init__(self, hidden, intermediate):
        super().__init__()
        self.gate_proj = Conv2dLinear(hidden, intermediate)
        self.up_proj = Conv2dLinear(hidden, intermediate)
        self.down_proj = Conv2dLinear(intermediate, hidden)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TalkerLayerV2(nn.Module):
    def __init__(self, hidden=1024, intermediate=3072, num_heads=16,
                 num_kv_heads=8, head_dim=128, eps=1e-6):
        super().__init__()
        self.self_attn = TalkerAttentionV2(hidden, num_heads, num_kv_heads, head_dim)
        self.mlp = TalkerMLPV2(hidden, intermediate)
        self.input_layernorm = RMSNormConv(hidden, eps)
        self.post_attention_layernorm = RMSNormConv(hidden, eps)

    def forward(self, x, rope_cos, rope_sin, key_padding_mask,
                kv_cache_update_mask, layer_key_cache, layer_value_cache):
        residual = x
        h = self.input_layernorm(x)
        h, new_kc, new_vc = self.self_attn(
            h, rope_cos, rope_sin, key_padding_mask,
            kv_cache_update_mask, layer_key_cache, layer_value_cache)
        h = residual + h

        residual = h
        h = self.post_attention_layernorm(h)
        h = self.mlp(h)
        h = residual + h

        return h, new_kc, new_vc


class TalkerWrapperV2(nn.Module):
    """Qwen3-TTS Talker with ANE-optimized Conv2d architecture.

    Single-token decode: input_embeds (B, C, 1, 1) + precomputed RoPE + mask
    Stacked KV cache: all layers concatenated along channel dim.
    """
    def __init__(self, num_layers=28, hidden=1024, intermediate=3072,
                 num_heads=16, num_kv_heads=8, head_dim=128,
                 eps=1e-6, codec_vocab=3072, max_seq_len=256):
        super().__init__()
        self.num_layers = num_layers
        self.d = num_kv_heads * head_dim  # KV dim per layer
        self.hidden = hidden
        self.max_seq_len = max_seq_len

        self.layers = nn.ModuleList([
            TalkerLayerV2(hidden, intermediate, num_heads, num_kv_heads, head_dim, eps)
            for _ in range(num_layers)
        ])
        self.norm = RMSNormConv(hidden, eps)
        # codec_head as Conv2d
        self.codec_head = nn.Conv2d(hidden, codec_vocab, kernel_size=1, bias=False)

    def forward(self, input_embeds, rope_cos, rope_sin,
                key_padding_mask, kv_cache_update_mask,
                key_cache, value_cache):
        """
        input_embeds: (B, C, 1, 1)
        rope_cos: (B, head_dim, 1)
        rope_sin: (B, head_dim, 1)
        key_padding_mask: (B, max_seq)
        kv_cache_update_mask: (B, max_seq)
        key_cache: (B, d*num_layers, 1, max_seq)
        value_cache: (B, d*num_layers, 1, max_seq)
        """
        h = input_embeds

        # Split stacked cache per layer
        layer_key_caches = key_cache.split(self.d, dim=1)
        layer_value_caches = value_cache.split(self.d, dim=1)

        key_cache_updates = []
        value_cache_updates = []

        for i, layer in enumerate(self.layers):
            h, new_kc, new_vc = layer(
                h, rope_cos, rope_sin, key_padding_mask,
                kv_cache_update_mask,
                layer_key_caches[i], layer_value_caches[i])

            # Extract just the update (the written position)
            # new_kc is the full cache — extract the delta
            update_k = (new_kc * kv_cache_update_mask[:, None, None, :]).sum(dim=-1, keepdim=True)
            update_v = (new_vc * kv_cache_update_mask[:, None, None, :]).sum(dim=-1, keepdim=True)
            key_cache_updates.append(update_k)
            value_cache_updates.append(update_v)

        h = self.norm(h)  # (B, C, 1, 1)

        # Logits
        logits = self.codec_head(h)  # (B, vocab, 1, 1)
        logits = logits.squeeze(-1).squeeze(-1)  # (B, vocab)

        # Hidden states for code predictor
        hidden_states = h  # (B, C, 1, 1)

        # Stack cache updates
        all_key_updates = torch.cat(key_cache_updates, dim=1)    # (B, d*layers, 1, 1)
        all_value_updates = torch.cat(value_cache_updates, dim=1)

        return logits, hidden_states, all_key_updates, all_value_updates


# ============================================================================
# Weight Loading (HuggingFace -> Conv2d)
# ============================================================================

def load_talker_weights_v2(wrapper: TalkerWrapperV2, weights: dict):
    """Load HF weights into Conv2d wrapper with [out, in] -> [out, in, 1, 1] reshape."""
    sd = wrapper.state_dict()
    loaded = 0

    for i in range(wrapper.num_layers):
        src = f"talker.model.layers.{i}."
        dst = f"layers.{i}."

        # Attention projections: Linear [out, in] -> Conv2d [out, in, 1, 1]
        for attn_name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            w_key = f"self_attn.{attn_name}.weight"
            src_key = src + w_key.replace("self_attn.", "self_attn.")
            dst_key = dst + f"self_attn.{attn_name}.conv.weight"
            if src_key in weights and dst_key in sd:
                w = weights[src_key].float()
                sd[dst_key] = w[:, :, None, None]  # Add spatial dims
                loaded += 1

        # Q/K norm: [dim] -> [1, dim, 1, 1]
        for norm_name in ["q_norm", "k_norm"]:
            src_key = src + f"self_attn.{norm_name}.weight"
            dst_key = dst + f"self_attn.{norm_name}.weight"
            if src_key in weights and dst_key in sd:
                w = weights[src_key].float()
                sd[dst_key] = w.view(1, -1, 1, 1)
                loaded += 1

        # MLP: gate/up/down proj
        for mlp_name in ["gate_proj", "up_proj", "down_proj"]:
            src_key = src + f"mlp.{mlp_name}.weight"
            dst_key = dst + f"mlp.{mlp_name}.conv.weight"
            if src_key in weights and dst_key in sd:
                w = weights[src_key].float()
                sd[dst_key] = w[:, :, None, None]
                loaded += 1

        # Layer norms: [dim] -> [1, dim, 1, 1]
        for ln_name in ["input_layernorm", "post_attention_layernorm"]:
            src_key = src + f"{ln_name}.weight"
            dst_key = dst + f"{ln_name}.weight"
            if src_key in weights and dst_key in sd:
                w = weights[src_key].float()
                sd[dst_key] = w.view(1, -1, 1, 1)
                loaded += 1

    # Final norm
    if "talker.model.norm.weight" in weights:
        sd["norm.weight"] = weights["talker.model.norm.weight"].float().view(1, -1, 1, 1)
        loaded += 1

    # Codec head: Linear [vocab, hidden] -> Conv2d [vocab, hidden, 1, 1]
    if "talker.codec_head.weight" in weights:
        sd["codec_head.weight"] = weights["talker.codec_head.weight"].float()[:, :, None, None]
        loaded += 1

    wrapper.load_state_dict(sd)
    print(f"  Loaded {loaded} weights (Conv2d format)")


# ============================================================================
# Conversion
# ============================================================================

def convert_talker_v2(weights: dict, output_dir: Path, quantize: str = "int8",
                      max_seq_len: int = 256):
    print("\n=== Converting Talker (V2: Conv2d + stacked KV) ===")

    wrapper = TalkerWrapperV2(max_seq_len=max_seq_len)
    load_talker_weights_v2(wrapper, weights)
    wrapper.eval()

    num_layers = 28
    d = 8 * 128  # kv_dim per layer
    total_d = d * num_layers  # stacked KV dim

    # Trace inputs (single-token decode)
    B = 1
    dummy_embeds = torch.randn(B, 1024, 1, 1)
    dummy_cos = torch.randn(B, 128, 1)
    dummy_sin = torch.randn(B, 128, 1)
    dummy_pad_mask = torch.zeros(B, max_seq_len)
    dummy_pad_mask[:, 1:] = MASK_VALUE  # mask all except first position
    dummy_update_mask = torch.zeros(B, max_seq_len)
    dummy_update_mask[:, 0] = 1.0  # write to first position
    dummy_kc = torch.zeros(B, total_d, 1, max_seq_len)
    dummy_vc = torch.zeros(B, total_d, 1, max_seq_len)

    print("  Tracing...", flush=True)
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (
            dummy_embeds, dummy_cos, dummy_sin,
            dummy_pad_mask, dummy_update_mask,
            dummy_kc, dummy_vc))
    print("  Trace complete.", flush=True)

    import coremltools as ct

    inputs = [
        ct.TensorType("input_embeds", shape=(1, 1024, 1, 1), dtype=np.float16),
        ct.TensorType("rope_cos", shape=(1, 128, 1), dtype=np.float16),
        ct.TensorType("rope_sin", shape=(1, 128, 1), dtype=np.float16),
        ct.TensorType("key_padding_mask", shape=(1, max_seq_len), dtype=np.float16),
        ct.TensorType("kv_cache_update_mask", shape=(1, max_seq_len), dtype=np.float16),
        ct.TensorType("key_cache", shape=(1, total_d, 1, max_seq_len), dtype=np.float16),
        ct.TensorType("value_cache", shape=(1, total_d, 1, max_seq_len), dtype=np.float16),
    ]

    outputs = [
        ct.TensorType("logits", dtype=np.float16),
        ct.TensorType("hidden_states", dtype=np.float16),
        ct.TensorType("key_cache_updates", dtype=np.float16),
        ct.TensorType("value_cache_updates", dtype=np.float16),
    ]

    print("  Converting to CoreML...", flush=True)
    mlmodel = ct.convert(
        traced, inputs=inputs, outputs=outputs,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT16,
    )

    if quantize == "int8":
        print("  Quantizing to W8A16 (palettization)...", flush=True)
        op_config = ct.optimize.coreml.OpPalettizerConfig(
            mode="kmeans", nbits=8)
        config = ct.optimize.coreml.OptimizationConfig(global_config=op_config)
        mlmodel = ct.optimize.coreml.palettize_weights(mlmodel, config=config)

    model_path = output_dir / "Talker.mlpackage"
    print(f"  Saving to {model_path}...", flush=True)
    mlmodel.save(str(model_path))
    size_mb = sum(f.stat().st_size for f in model_path.rglob("*") if f.is_file()) / 1e6
    print(f"  Saved ({size_mb:.0f} MB)")

    return mlmodel


# ============================================================================
# V1-style Linear modules (used by CodePredictor)
# ============================================================================

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class TalkerMLP(nn.Module):
    def __init__(self, hidden, intermediate):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TalkerAttention(nn.Module):
    def __init__(self, hidden=1024, num_heads=16, num_kv_heads=8, head_dim=128,
                 rope_theta=1e6):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.rope_theta = rope_theta
        self.q_proj = nn.Linear(hidden, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden, bias=False)
        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)

    def forward(self, x, position_ids, causal_mask, k_cache, v_cache):
        B, S, _ = x.shape
        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        # RoPE (standard 1D — MRoPE reduces to this for TTS where T=H=W)
        dim = self.head_dim
        freqs = 1.0 / (self.rope_theta ** (
            torch.arange(0, dim, 2, dtype=torch.float32, device=x.device) / dim))
        pos = position_ids.float().unsqueeze(-1)
        angles = pos * freqs
        cos_val = torch.cos(angles).unsqueeze(1)
        sin_val = torch.sin(angles).unsqueeze(1)

        def apply_rope(t):
            # MLX traditional=False: stride half (x[:d/2], x[d/2:])
            t1, t2 = t[..., :dim // 2], t[..., dim // 2:]
            return torch.cat([t1 * cos_val - t2 * sin_val,
                              t2 * cos_val + t1 * sin_val], dim=-1)

        q = apply_rope(q)
        k = apply_rope(k)

        # KV cache concat
        k = torch.cat([k_cache, k], dim=2)
        v = torch.cat([v_cache, v], dim=2)

        # GQA expand
        if self.num_kv_heads < self.num_heads:
            rep = self.num_heads // self.num_kv_heads
            k_exp = k.repeat_interleave(rep, dim=1)
            v_exp = v.repeat_interleave(rep, dim=1)
        else:
            k_exp, v_exp = k, v

        scores = torch.matmul(q, k_exp.transpose(-2, -1)) * self.scale
        scores = scores + causal_mask
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v_exp)

        out = out.transpose(1, 2).contiguous().view(B, S, -1)
        return self.o_proj(out), k, v


class TalkerLayer(nn.Module):
    def __init__(self, hidden=1024, intermediate=3072, num_heads=16,
                 num_kv_heads=8, head_dim=128, rope_theta=1e6, eps=1e-6):
        super().__init__()
        self.self_attn = TalkerAttention(hidden, num_heads, num_kv_heads, head_dim, rope_theta)
        self.mlp = TalkerMLP(hidden, intermediate)
        self.input_layernorm = RMSNorm(hidden, eps)
        self.post_attention_layernorm = RMSNorm(hidden, eps)

    def forward(self, x, position_ids, causal_mask, k_cache, v_cache):
        residual = x
        h = self.input_layernorm(x)
        h, new_k, new_v = self.self_attn(h, position_ids, causal_mask, k_cache, v_cache)
        h = residual + h

        residual = h
        h = self.post_attention_layernorm(h)
        h = self.mlp(h)
        h = residual + h

        return h, new_k, new_v


def make_causal_mask(query_len, key_len, dtype=torch.float32):
    mask = torch.full((query_len, key_len), float('-inf'), dtype=dtype)
    past_len = key_len - query_len
    for q in range(query_len):
        for k in range(key_len):
            if k <= past_len + q:
                mask[q, k] = 0.0
    return mask.unsqueeze(0).unsqueeze(0)


# ============================================================================
# Code Predictor
# ============================================================================

class CodePredictorWrapper(nn.Module):
    """Code Predictor: 5-layer transformer predicting 15 residual codebooks."""

    def __init__(self, num_layers=5, hidden=1024, intermediate=3072,
                 num_heads=16, num_kv_heads=8, head_dim=128,
                 rope_theta=1e6, eps=1e-6, num_groups=15, vocab_size=2048):
        super().__init__()
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_groups = num_groups
        self.layers = nn.ModuleList([
            TalkerLayer(hidden, intermediate, num_heads, num_kv_heads,
                        head_dim, rope_theta, eps)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(hidden, eps)
        self.lm_heads = nn.ModuleList([
            nn.Linear(hidden, vocab_size, bias=False) for _ in range(num_groups)
        ])

    def forward(self, input_embeds, position_ids, causal_mask, *kv_states):
        h = input_embeds
        new_kv = []
        for i, layer in enumerate(self.layers):
            k_cache = kv_states[2 * i]
            v_cache = kv_states[2 * i + 1]
            h, new_k, new_v = layer(h, position_ids, causal_mask, k_cache, v_cache)
            new_kv.append(new_k)
            new_kv.append(new_v)

        h = self.norm(h)

        # Output normalized hidden states — lm_heads applied in Swift per group
        hidden_states = h[:, -1:, :]  # [1, 1, 1024]

        return (hidden_states, *new_kv)


def load_code_predictor_weights(wrapper: CodePredictorWrapper, weights: dict):
    sd = wrapper.state_dict()
    mapping = {}

    for i in range(wrapper.num_layers):
        prefix_src = f"talker.code_predictor.model.layers.{i}."
        prefix_dst = f"layers.{i}."
        for suffix in [
            "self_attn.q_proj.weight", "self_attn.k_proj.weight",
            "self_attn.v_proj.weight", "self_attn.o_proj.weight",
            "self_attn.q_norm.weight", "self_attn.k_norm.weight",
            "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
            "input_layernorm.weight", "post_attention_layernorm.weight",
        ]:
            mapping[prefix_dst + suffix] = prefix_src + suffix

    mapping["norm.weight"] = "talker.code_predictor.model.norm.weight"

    for i in range(wrapper.num_groups):
        mapping[f"lm_heads.{i}.weight"] = f"talker.code_predictor.lm_head.{i}.weight"

    loaded = 0
    for dst_key, src_key in mapping.items():
        if src_key in weights and dst_key in sd:
            sd[dst_key] = weights[src_key].float()
            loaded += 1

    wrapper.load_state_dict(sd)
    print(f"  Loaded {loaded}/{len(mapping)} CodePredictor weights")


def convert_code_predictor(weights: dict, output_dir: Path, quantize: str = "int8"):
    """Convert Code Predictor to CoreML."""
    print("\n=== Converting Code Predictor ===")

    wrapper = CodePredictorWrapper(
        num_layers=5, hidden=1024, intermediate=3072,
        num_heads=16, num_kv_heads=8, head_dim=128,
        rope_theta=1e6, eps=1e-6, num_groups=15, vocab_size=2048,
    )
    load_code_predictor_weights(wrapper, weights)
    wrapper.eval()

    num_layers = 5
    num_kv_heads = 8
    head_dim = 128
    max_seq_len = 20  # CP has very short sequences (2 prefill + 15 decode = 17 max)

    seq_len = 2
    past_len = 1
    dummy_embeds = torch.randn(1, seq_len, 1024)
    dummy_pos = torch.tensor([[past_len, past_len + 1]], dtype=torch.int32)
    dummy_mask = make_causal_mask(seq_len, seq_len + past_len)
    dummy_kv = []
    for _ in range(num_layers):
        dummy_kv.append(torch.zeros(1, num_kv_heads, past_len, head_dim))
        dummy_kv.append(torch.zeros(1, num_kv_heads, past_len, head_dim))

    print("  Tracing...", flush=True)
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (dummy_embeds, dummy_pos, dummy_mask, *dummy_kv))
    print("  Trace complete.", flush=True)

    import coremltools as ct

    # CodePredictor uses FP32 — FP16 causes NaN in CoreML's attention computation
    inputs = [
        ct.TensorType("input_embeds",
                       shape=ct.Shape(shape=(1, ct.RangeDim(1, max_seq_len), 1024)),
                       dtype=np.float32),
        ct.TensorType("position_ids",
                       shape=ct.Shape(shape=(1, ct.RangeDim(1, max_seq_len))),
                       dtype=np.int32),
        ct.TensorType("causal_mask",
                       shape=ct.Shape(shape=(1, 1,
                                             ct.RangeDim(1, max_seq_len),
                                             ct.RangeDim(1, max_seq_len + 1))),
                       dtype=np.float32),
    ]
    for i in range(num_layers):
        for name in ["key", "value"]:
            inputs.append(ct.TensorType(
                f"layer_{i}_{name}_cache",
                shape=ct.Shape(shape=(1, num_kv_heads,
                                      ct.RangeDim(1, max_seq_len), head_dim)),
                dtype=np.float32))

    outputs = [ct.TensorType("hidden_states", dtype=np.float32)]
    for i in range(num_layers):
        for name in ["key", "value"]:
            outputs.append(ct.TensorType(f"layer_{i}_{name}_cache_out", dtype=np.float32))
    print("  Converting to CoreML (FP32)...", flush=True)
    mlmodel = ct.convert(
        traced, inputs=inputs, outputs=outputs,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT32,
    )

    model_path = output_dir / "CodePredictor.mlpackage"
    print(f"  Saving to {model_path}...", flush=True)
    mlmodel.save(str(model_path))

    size_mb = sum(f.stat().st_size for f in model_path.rglob("*") if f.is_file()) / 1e6
    print(f"  Saved ({size_mb:.0f} MB)")


# ============================================================================
# Embeddings extraction
# ============================================================================

def extract_embeddings(weights: dict, output_dir: Path):
    """Extract embedding tables for Swift-side lookup."""
    print("\n=== Extracting Embeddings ===")
    from safetensors.numpy import save_file

    embeds = {}

    # Codec embedding: [3072, 1024]
    key = "talker.model.codec_embedding.weight"
    if key in weights:
        embeds["codec_embedding"] = weights[key].half().numpy()
        print(f"  codec_embedding: {embeds['codec_embedding'].shape}")

    # Text embedding: [151936, 2048]
    key = "talker.model.embed_tokens.weight"
    if key not in weights:
        key = "talker.model.text_embedding.weight"
    if key in weights:
        embeds["text_embedding"] = weights[key].half().numpy()
        print(f"  text_embedding: {embeds['text_embedding'].shape}")

    # Text projection: fc1.weight [2048,2048], fc1.bias [2048], fc2.weight [1024,2048], fc2.bias [1024]
    for name in ["linear_fc1.weight", "linear_fc1.bias", "linear_fc2.weight", "linear_fc2.bias"]:
        key = f"talker.text_projection.{name}"
        if key in weights:
            embeds[f"text_projection.{name}"] = weights[key].half().numpy()
            print(f"  text_projection.{name}: {embeds[f'text_projection.{name}'].shape}")

    # Code predictor codec embeddings: [16 groups, each 2048 x 1024]
    for i in range(16):
        key = f"talker.code_predictor.model.codec_embedding.{i}.weight"
        if key in weights:
            embeds[f"cp_codec_embedding.{i}"] = weights[key].half().numpy()

    # CP group embedding
    key = "talker.code_predictor.cp_group_embedding.weight"
    if key in weights:
        embeds["cp_group_embedding"] = weights[key].half().numpy()
        print(f"  cp_group_embedding: {embeds['cp_group_embedding'].shape}")

    # Code predictor lm_heads: 15 separate [2048, 1024] weight matrices
    for i in range(15):
        key = f"talker.code_predictor.lm_head.{i}.weight"
        if key in weights:
            embeds[f"cp_lm_head.{i}"] = weights[key].half().numpy()
    cp_heads_found = sum(1 for i in range(15) if f"cp_lm_head.{i}" in embeds)
    print(f"  cp_lm_heads: {cp_heads_found}/15 found")

    save_path = output_dir / "embeddings.safetensors"
    save_file(embeds, str(save_path))
    size_mb = save_path.stat().st_size / 1e6
    print(f"  Saved embeddings.safetensors ({size_mb:.0f} MB)")


# ============================================================================
# Minimal PyTorch MimiDecoder (for tracing)
# ============================================================================

class SnakeBeta(nn.Module):
    """SnakeBeta activation: x + (1/exp(beta)) * sin^2(exp(alpha) * x).
    Alpha and beta are learnable parameters stored in log-space.
    """
    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(1, channels, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1))

    def forward(self, x):
        # x: [B, C, T] (NCT format for PyTorch conv)
        a = torch.exp(self.alpha)
        b = torch.exp(self.beta)
        sin_term = torch.sin(a * x)
        return x + (1.0 / b) * (sin_term * sin_term)


class CausalConv1d(nn.Module):
    """Conv1d with left padding for causality."""
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        super().__init__()
        self.pad_amount = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride,
                              padding=0, dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        # x: [B, C, T]
        if self.pad_amount > 0:
            x = F.pad(x, (self.pad_amount, 0))
        return self.conv(x)


class CausalTransposeConv1d(nn.Module):
    """ConvTranspose1d with right trimming for causality."""
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, bias=True):
        super().__init__()
        self.trim_right = kernel_size - stride
        self.conv = nn.ConvTranspose1d(in_ch, out_ch, kernel_size, stride=stride,
                                       padding=0, bias=bias)

    def forward(self, x):
        # x: [B, C, T]
        out = self.conv(x)
        if self.trim_right > 0:
            out = out[:, :, :-self.trim_right]
        return out


class LayerScale(nn.Module):
    """Per-channel learnable scale factor."""
    def __init__(self, channels, init_value=0.01):
        super().__init__()
        self.scale = nn.Parameter(torch.full((1, channels, 1), init_value))

    def forward(self, x):
        return x * self.scale


class ConvNeXtBlock(nn.Module):
    """ConvNeXt block: depthwise causal conv -> LayerNorm -> Linear -> GELU -> Linear -> LayerScale -> residual."""
    def __init__(self, dim, intermediate_scale=4, kernel_size=7):
        super().__init__()
        intermediate_dim = dim * intermediate_scale
        self.dwconv = CausalConv1d(dim, dim, kernel_size, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)
        self.pwconv2 = nn.Linear(intermediate_dim, dim)
        self.gamma = nn.Parameter(torch.full((1, dim, 1), 0.01))

    def forward(self, x):
        # x: [B, C, T]
        residual = x
        h = self.dwconv(x)
        # LayerNorm expects [B, T, C], so transpose
        h = h.transpose(1, 2)  # [B, T, C]
        h = self.norm(h)
        h = self.pwconv1(h)
        h = F.gelu(h)
        h = self.pwconv2(h)
        h = h.transpose(1, 2)  # [B, C, T]
        h = self.gamma * h
        return h + residual


class DecoderResidualUnit(nn.Module):
    """Dilated residual unit: SnakeBeta -> CausalConv1d(dilated) -> SnakeBeta -> CausalConv1d(1x1) -> residual."""
    def __init__(self, dim, dilation):
        super().__init__()
        self.snake1 = SnakeBeta(dim)
        self.conv1 = CausalConv1d(dim, dim, kernel_size=7, dilation=dilation)
        self.snake2 = SnakeBeta(dim)
        self.conv2 = CausalConv1d(dim, dim, kernel_size=1)

    def forward(self, x):
        residual = x
        h = self.snake1(x)
        h = self.conv1(h)
        h = self.snake2(h)
        h = self.conv2(h)
        return h + residual


class DecoderBlock(nn.Module):
    """Upsample block: SnakeBeta -> CausalTransposeConv1d -> 3x DecoderResidualUnit."""
    def __init__(self, in_dim, out_dim, stride):
        super().__init__()
        self.snake = SnakeBeta(in_dim)
        self.upsample = CausalTransposeConv1d(in_dim, out_dim,
                                               kernel_size=stride * 2, stride=stride)
        self.res_units = nn.ModuleList([
            DecoderResidualUnit(out_dim, dilation=d)
            for d in [1, 3, 9]
        ])

    def forward(self, x):
        h = self.snake(x)
        h = self.upsample(h)
        for unit in self.res_units:
            h = unit(h)
        return h


class MimiDecoderRMSNorm(nn.Module):
    """RMSNorm for MimiDecoder (eps=1e-8)."""
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class DecoderTransformerLayer(nn.Module):
    """Decoder transformer layer: RMSNorm -> Attention(RoPE) -> LayerScale -> RMSNorm -> SwiGLU MLP -> LayerScale."""
    def __init__(self, hidden=512, num_heads=16, head_dim=64, rope_theta=10000.0, eps=1e-8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.rope_theta = rope_theta

        self.q_proj = nn.Linear(hidden, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, num_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, num_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden, bias=False)

        intermediate = hidden * 2  # 1024
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

        self.norm1 = MimiDecoderRMSNorm(hidden, eps)
        self.norm2 = MimiDecoderRMSNorm(hidden, eps)
        self.attn_layer_scale = LayerScale(hidden)
        self.mlp_layer_scale = LayerScale(hidden)

    def forward(self, x, cos_val, sin_val, causal_mask):
        # x: [B, T, hidden]
        B, S, _ = x.shape
        residual = x
        h = self.norm1(x)

        q = self.q_proj(h).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE — MLX traditional=False: stride half (x[:d/2], x[d/2:])
        q1, q2 = q[..., :self.head_dim // 2], q[..., self.head_dim // 2:]
        q = torch.cat([q1 * cos_val - q2 * sin_val, q2 * cos_val + q1 * sin_val], dim=-1)
        k1, k2 = k[..., :self.head_dim // 2], k[..., self.head_dim // 2:]
        k = torch.cat([k1 * cos_val - k2 * sin_val, k2 * cos_val + k1 * sin_val], dim=-1)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        scores = scores + causal_mask
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)

        out = out.transpose(1, 2).contiguous().view(B, S, -1)
        attn_out = self.o_proj(out)
        # LayerScale in NCT format: transpose -> scale -> transpose back
        attn_out = self.attn_layer_scale(attn_out.transpose(1, 2)).transpose(1, 2)
        h = residual + attn_out

        residual2 = h
        h = self.norm2(h)
        h = F.silu(self.gate_proj(h)) * self.up_proj(h)
        h = self.down_proj(h)
        h = self.mlp_layer_scale(h.transpose(1, 2)).transpose(1, 2)
        h = residual2 + h

        return h


class DecoderTransformer(nn.Module):
    """Pre-transformer: input_proj(1024->512) -> 8 layers -> norm -> output_proj(512->1024)."""
    def __init__(self, latent_dim=1024, hidden=512, num_heads=16, head_dim=64,
                 num_layers=8, rope_theta=10000.0, eps=1e-8, max_seq_len=200):
        super().__init__()
        self.head_dim = head_dim
        self.rope_theta = rope_theta
        self.input_proj = nn.Linear(latent_dim, hidden)
        self.layers = nn.ModuleList([
            DecoderTransformerLayer(hidden, num_heads, head_dim, rope_theta, eps)
            for _ in range(num_layers)
        ])
        self.norm = MimiDecoderRMSNorm(hidden, eps)
        self.output_proj = nn.Linear(hidden, latent_dim)

        # Pre-compute RoPE and causal mask for max_seq_len.
        dim = head_dim
        freqs = 1.0 / (rope_theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        positions = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(-1)
        angles = positions * freqs
        self.register_buffer('cos_table', torch.cos(angles).unsqueeze(0).unsqueeze(0))  # [1, 1, max, dim//2]
        self.register_buffer('sin_table', torch.sin(angles).unsqueeze(0).unsqueeze(0))

        rows = torch.arange(max_seq_len).unsqueeze(1)
        cols = torch.arange(max_seq_len).unsqueeze(0)
        mask = torch.where(cols > rows, torch.tensor(float('-inf')), torch.tensor(0.0))
        self.register_buffer('causal_mask_table', mask.unsqueeze(0).unsqueeze(0))  # [1, 1, max, max]

    def forward(self, x):
        # x: [B, T, 1024] (NLC format)
        h = self.input_proj(x)  # [B, T, 512]

        S = h.shape[1]
        # Slice precomputed RoPE and mask to actual length
        cos_val = self.cos_table[:, :, :S, :]
        sin_val = self.sin_table[:, :, :S, :]
        causal_mask = self.causal_mask_table[:, :, :S, :S]

        for layer in self.layers:
            h = layer(h, cos_val, sin_val, causal_mask)

        h = self.norm(h)
        h = self.output_proj(h)  # [B, T, 1024]
        return h


class MimiDecoderWrapper(nn.Module):
    """Full Mimi-based speech tokenizer decoder.
    Converts 16-codebook indices [B, 16, T] to 24kHz waveform [B, 1, T*1920].

    Architecture:
      SplitRVQ -> pre_conv(512->1024) -> DecoderTransformer(1024->512->1024) ->
      pre_upsample(2x,2x) -> input_conv(1024->1536) ->
      4 DecoderBlocks([8,5,4,3], 1536->768->384->192->96) ->
      SnakeBeta -> final_conv(96->1) -> clip(-1,1)
    """

    def __init__(self):
        super().__init__()
        # RVQ codebooks
        # rvq_first: 1 codebook, size 2048, dim 256
        self.rvq_first_embed = nn.Embedding(2048, 256)
        # rvq_first output_proj: Conv1d(256, 512, k=1, no bias)
        self.rvq_first_output_proj = nn.Conv1d(256, 512, 1, bias=False)
        # rvq_rest: 15 codebooks, each size 2048, dim 256
        self.rvq_rest_embeds = nn.ModuleList([nn.Embedding(2048, 256) for _ in range(15)])
        # rvq_rest output_proj: Conv1d(256, 512, k=1, no bias)
        self.rvq_rest_output_proj = nn.Conv1d(256, 512, 1, bias=False)

        # Pre-conv: CausalConv1d(512, 1024, k=3)
        self.pre_conv = CausalConv1d(512, 1024, kernel_size=3)

        # Decoder transformer
        self.transformer = DecoderTransformer(
            num_layers=8, hidden=512, num_heads=16, head_dim=64,
            latent_dim=1024, rope_theta=10000.0, eps=1e-8)

        # Pre-upsample: 2 stages of CausalTransposeConv1d(1024,1024,k=2,s=2) + ConvNeXtBlock(1024)
        self.pre_upsample1 = CausalTransposeConv1d(1024, 1024, kernel_size=2, stride=2)
        self.pre_convnext1 = ConvNeXtBlock(1024)
        self.pre_upsample2 = CausalTransposeConv1d(1024, 1024, kernel_size=2, stride=2)
        self.pre_convnext2 = ConvNeXtBlock(1024)

        # Input conv: CausalConv1d(1024, 1536, k=7)
        self.input_conv = CausalConv1d(1024, 1536, kernel_size=7)

        # 4 decoder blocks: upsample rates [8, 5, 4, 3], dims [1536, 768, 384, 192, 96]
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(1536, 768, stride=8),
            DecoderBlock(768, 384, stride=5),
            DecoderBlock(384, 192, stride=4),
            DecoderBlock(192, 96, stride=3),
        ])

        # Final: SnakeBeta(96) + CausalConv1d(96, 1, k=7)
        self.final_snake = SnakeBeta(96)
        self.final_conv = CausalConv1d(96, 1, kernel_size=7)

    def forward(self, codes):
        """
        codes: [B, 16, T] — int32 codebook indices
        returns: [B, 1, T*1920] — waveform audio clipped to [-1, 1]
        """
        # RVQ decode: sum embeddings from first + rest quantizers
        # First codebook (semantic)
        first_codes = codes[:, 0, :]  # [B, T]
        first_embed = self.rvq_first_embed(first_codes)  # [B, T, 256]
        first_embed = first_embed.transpose(1, 2)  # [B, 256, T]
        first_out = self.rvq_first_output_proj(first_embed)  # [B, 512, T]

        # Rest codebooks (acoustic) — sum all 15 embeddings then project
        rest_sum = self.rvq_rest_embeds[0](codes[:, 1, :])  # [B, T, 256]
        rest_sum = rest_sum + self.rvq_rest_embeds[1](codes[:, 2, :])
        rest_sum = rest_sum + self.rvq_rest_embeds[2](codes[:, 3, :])
        rest_sum = rest_sum + self.rvq_rest_embeds[3](codes[:, 4, :])
        rest_sum = rest_sum + self.rvq_rest_embeds[4](codes[:, 5, :])
        rest_sum = rest_sum + self.rvq_rest_embeds[5](codes[:, 6, :])
        rest_sum = rest_sum + self.rvq_rest_embeds[6](codes[:, 7, :])
        rest_sum = rest_sum + self.rvq_rest_embeds[7](codes[:, 8, :])
        rest_sum = rest_sum + self.rvq_rest_embeds[8](codes[:, 9, :])
        rest_sum = rest_sum + self.rvq_rest_embeds[9](codes[:, 10, :])
        rest_sum = rest_sum + self.rvq_rest_embeds[10](codes[:, 11, :])
        rest_sum = rest_sum + self.rvq_rest_embeds[11](codes[:, 12, :])
        rest_sum = rest_sum + self.rvq_rest_embeds[12](codes[:, 13, :])
        rest_sum = rest_sum + self.rvq_rest_embeds[13](codes[:, 14, :])
        rest_sum = rest_sum + self.rvq_rest_embeds[14](codes[:, 15, :])
        rest_sum = rest_sum.transpose(1, 2)  # [B, 256, T]
        rest_out = self.rvq_rest_output_proj(rest_sum)  # [B, 512, T]

        # Combined RVQ output: [B, 512, T]
        h = first_out + rest_out

        # Pre-conv: [B, 512, T] -> [B, 1024, T]
        h = self.pre_conv(h)

        # Transformer operates in NLC format: [B, T, 1024]
        h = h.transpose(1, 2)
        h = self.transformer(h)
        h = h.transpose(1, 2)  # back to [B, 1024, T]

        # Pre-upsample (2x, 2x = 4x total)
        h = self.pre_upsample1(h)
        h = self.pre_convnext1(h)
        h = self.pre_upsample2(h)
        h = self.pre_convnext2(h)

        # Input conv: [B, 1024, T*4] -> [B, 1536, T*4]
        h = self.input_conv(h)

        # Main decoder blocks (8x, 5x, 4x, 3x = 480x)
        for block in self.decoder_blocks:
            h = block(h)

        # Final output
        h = self.final_snake(h)
        h = self.final_conv(h)
        h = torch.clamp(h, -1.0, 1.0)

        return h  # [B, 1, T*1920]


def load_mimi_decoder_weights(wrapper: MimiDecoderWrapper, weights: dict):
    """Load speech_tokenizer weights into MimiDecoderWrapper.

    Weight keys use 'decoder.' prefix from the speech_tokenizer/model.safetensors file.
    Conv1d weights in safetensors are PyTorch format [out, in, kernel].
    ConvTranspose1d weights are [in, out, kernel].
    """
    sd = wrapper.state_dict()
    loaded = 0
    total = len(sd)

    def _set(dst_key, tensor):
        nonlocal loaded
        if dst_key in sd:
            sd[dst_key] = tensor.float()
            loaded += 1
        else:
            print(f"  WARNING: {dst_key} not in model state_dict")

    # --- RVQ codebooks ---
    # rvq_first: use embed or compute from cluster_usage + embedding_sum
    prefix = "decoder.quantizer.rvq_first.vq.layers.0._codebook"
    if f"{prefix}.embed" in weights:
        _set("rvq_first_embed.weight", weights[f"{prefix}.embed"])
    elif f"{prefix}.cluster_usage" in weights:
        usage = weights[f"{prefix}.cluster_usage"].float()
        emb_sum = weights[f"{prefix}.embedding_sum"].float()
        embed = emb_sum / usage.clamp(min=1e-7).unsqueeze(-1)
        _set("rvq_first_embed.weight", embed)

    # rvq_first output_proj (Conv1d, k=1, no bias)
    if "decoder.quantizer.rvq_first.output_proj.weight" in weights:
        _set("rvq_first_output_proj.weight",
             weights["decoder.quantizer.rvq_first.output_proj.weight"])

    # rvq_rest codebooks
    for i in range(15):
        prefix = f"decoder.quantizer.rvq_rest.vq.layers.{i}._codebook"
        if f"{prefix}.embed" in weights:
            _set(f"rvq_rest_embeds.{i}.weight", weights[f"{prefix}.embed"])
        elif f"{prefix}.cluster_usage" in weights:
            usage = weights[f"{prefix}.cluster_usage"].float()
            emb_sum = weights[f"{prefix}.embedding_sum"].float()
            embed = emb_sum / usage.clamp(min=1e-7).unsqueeze(-1)
            _set(f"rvq_rest_embeds.{i}.weight", embed)

    # rvq_rest output_proj
    if "decoder.quantizer.rvq_rest.output_proj.weight" in weights:
        _set("rvq_rest_output_proj.weight",
             weights["decoder.quantizer.rvq_rest.output_proj.weight"])

    # --- Pre-conv ---
    _set("pre_conv.conv.weight", weights["decoder.pre_conv.conv.weight"])
    _set("pre_conv.conv.bias", weights["decoder.pre_conv.conv.bias"])

    # --- Decoder transformer ---
    _set("transformer.input_proj.weight", weights["decoder.pre_transformer.input_proj.weight"])
    _set("transformer.input_proj.bias", weights["decoder.pre_transformer.input_proj.bias"])
    _set("transformer.output_proj.weight", weights["decoder.pre_transformer.output_proj.weight"])
    _set("transformer.output_proj.bias", weights["decoder.pre_transformer.output_proj.bias"])
    _set("transformer.norm.weight", weights["decoder.pre_transformer.norm.weight"])

    for i in range(8):
        src = f"decoder.pre_transformer.layers.{i}"
        dst = f"transformer.layers.{i}"

        # Attention projections
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            _set(f"{dst}.{proj}.weight", weights[f"{src}.self_attn.{proj}.weight"])

        # RMS norms
        _set(f"{dst}.norm1.weight", weights[f"{src}.input_layernorm.weight"])
        _set(f"{dst}.norm2.weight", weights[f"{src}.post_attention_layernorm.weight"])

        # SwiGLU MLP
        _set(f"{dst}.gate_proj.weight", weights[f"{src}.mlp.gate_proj.weight"])
        _set(f"{dst}.up_proj.weight", weights[f"{src}.mlp.up_proj.weight"])
        _set(f"{dst}.down_proj.weight", weights[f"{src}.mlp.down_proj.weight"])

        # LayerScale — stored as 1D [hidden], reshape to [1, hidden, 1] for NCT format
        attn_scale_key = f"{src}.self_attn_layer_scale.scale"
        if attn_scale_key in weights:
            _set(f"{dst}.attn_layer_scale.scale",
                 weights[attn_scale_key].reshape(1, -1, 1))
        mlp_scale_key = f"{src}.mlp_layer_scale.scale"
        if mlp_scale_key in weights:
            _set(f"{dst}.mlp_layer_scale.scale",
                 weights[mlp_scale_key].reshape(1, -1, 1))

    # --- Pre-upsample stages ---
    # Stage 0: transposed conv + ConvNeXt
    _set("pre_upsample1.conv.weight", weights["decoder.upsample.0.0.conv.weight"])
    _set("pre_upsample1.conv.bias", weights["decoder.upsample.0.0.conv.bias"])
    # ConvNeXt 0
    _set("pre_convnext1.dwconv.conv.weight", weights["decoder.upsample.0.1.dwconv.conv.weight"])
    _set("pre_convnext1.dwconv.conv.bias", weights["decoder.upsample.0.1.dwconv.conv.bias"])
    _set("pre_convnext1.norm.weight", weights["decoder.upsample.0.1.norm.weight"])
    _set("pre_convnext1.norm.bias", weights["decoder.upsample.0.1.norm.bias"])
    _set("pre_convnext1.pwconv1.weight", weights["decoder.upsample.0.1.pwconv1.weight"])
    _set("pre_convnext1.pwconv1.bias", weights["decoder.upsample.0.1.pwconv1.bias"])
    _set("pre_convnext1.pwconv2.weight", weights["decoder.upsample.0.1.pwconv2.weight"])
    _set("pre_convnext1.pwconv2.bias", weights["decoder.upsample.0.1.pwconv2.bias"])
    if "decoder.upsample.0.1.gamma" in weights:
        _set("pre_convnext1.gamma", weights["decoder.upsample.0.1.gamma"].reshape(1, -1, 1))

    # Stage 1: transposed conv + ConvNeXt
    _set("pre_upsample2.conv.weight", weights["decoder.upsample.1.0.conv.weight"])
    _set("pre_upsample2.conv.bias", weights["decoder.upsample.1.0.conv.bias"])
    _set("pre_convnext2.dwconv.conv.weight", weights["decoder.upsample.1.1.dwconv.conv.weight"])
    _set("pre_convnext2.dwconv.conv.bias", weights["decoder.upsample.1.1.dwconv.conv.bias"])
    _set("pre_convnext2.norm.weight", weights["decoder.upsample.1.1.norm.weight"])
    _set("pre_convnext2.norm.bias", weights["decoder.upsample.1.1.norm.bias"])
    _set("pre_convnext2.pwconv1.weight", weights["decoder.upsample.1.1.pwconv1.weight"])
    _set("pre_convnext2.pwconv1.bias", weights["decoder.upsample.1.1.pwconv1.bias"])
    _set("pre_convnext2.pwconv2.weight", weights["decoder.upsample.1.1.pwconv2.weight"])
    _set("pre_convnext2.pwconv2.bias", weights["decoder.upsample.1.1.pwconv2.bias"])
    if "decoder.upsample.1.1.gamma" in weights:
        _set("pre_convnext2.gamma", weights["decoder.upsample.1.1.gamma"].reshape(1, -1, 1))

    # --- Input conv: decoder.decoder.0 ---
    _set("input_conv.conv.weight", weights["decoder.decoder.0.conv.weight"])
    _set("input_conv.conv.bias", weights["decoder.decoder.0.conv.bias"])

    # --- Decoder blocks: decoder.decoder.{1,2,3,4} ---
    for i in range(4):
        src_block = f"decoder.decoder.{i + 1}"
        dst_block = f"decoder_blocks.{i}"

        # block.0 = Snake activation
        _set(f"{dst_block}.snake.alpha", weights[f"{src_block}.block.0.alpha"].reshape(1, -1, 1))
        _set(f"{dst_block}.snake.beta", weights[f"{src_block}.block.0.beta"].reshape(1, -1, 1))

        # block.1 = Transposed conv (upsample)
        _set(f"{dst_block}.upsample.conv.weight", weights[f"{src_block}.block.1.conv.weight"])
        _set(f"{dst_block}.upsample.conv.bias", weights[f"{src_block}.block.1.conv.bias"])

        # block.{2,3,4} = 3 residual units
        for j in range(3):
            src_res = f"{src_block}.block.{j + 2}"
            dst_res = f"{dst_block}.res_units.{j}"

            _set(f"{dst_res}.snake1.alpha", weights[f"{src_res}.act1.alpha"].reshape(1, -1, 1))
            _set(f"{dst_res}.snake1.beta", weights[f"{src_res}.act1.beta"].reshape(1, -1, 1))
            _set(f"{dst_res}.conv1.conv.weight", weights[f"{src_res}.conv1.conv.weight"])
            _set(f"{dst_res}.conv1.conv.bias", weights[f"{src_res}.conv1.conv.bias"])
            _set(f"{dst_res}.snake2.alpha", weights[f"{src_res}.act2.alpha"].reshape(1, -1, 1))
            _set(f"{dst_res}.snake2.beta", weights[f"{src_res}.act2.beta"].reshape(1, -1, 1))
            _set(f"{dst_res}.conv2.conv.weight", weights[f"{src_res}.conv2.conv.weight"])
            _set(f"{dst_res}.conv2.conv.bias", weights[f"{src_res}.conv2.conv.bias"])

    # --- Final: decoder.decoder.5 (snake) + decoder.decoder.6 (conv) ---
    _set("final_snake.alpha", weights["decoder.decoder.5.alpha"].reshape(1, -1, 1))
    _set("final_snake.beta", weights["decoder.decoder.5.beta"].reshape(1, -1, 1))
    _set("final_conv.conv.weight", weights["decoder.decoder.6.conv.weight"])
    _set("final_conv.conv.bias", weights["decoder.decoder.6.conv.bias"])

    wrapper.load_state_dict(sd)
    print(f"  Loaded {loaded}/{total} MimiDecoder weights")


def convert_mimi_decoder(st_weights: dict, output_dir: Path):
    """Convert MimiDecoder to CoreML.

    Args:
        st_weights: weights from speech_tokenizer/model.safetensors
        output_dir: output directory for the .mlpackage
    """
    print("\n=== Converting MimiDecoder ===")

    wrapper = MimiDecoderWrapper()
    load_mimi_decoder_weights(wrapper, st_weights)
    wrapper.eval()

    # Trace with a medium-length input
    # EnumeratedShapes: (1,16,4), (1,16,14), (1,16,35), (1,16,50)
    trace_t = 14
    dummy_codes = torch.zeros(1, 16, trace_t, dtype=torch.int32)
    print("  Tracing...", flush=True)
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, dummy_codes)
    print("  Trace complete.", flush=True)

    import coremltools as ct

    enum_shapes = [
        (1, 16, 4),
        (1, 16, 14),
        (1, 16, 35),
        (1, 16, 50),
    ]
    inputs = [
        ct.TensorType("codes",
                       shape=ct.EnumeratedShapes(shapes=enum_shapes),
                       dtype=np.int32),
    ]
    outputs = [
        ct.TensorType("waveform", dtype=np.float16),
    ]

    print("  Converting to CoreML...", flush=True)
    mlmodel = ct.convert(
        traced, inputs=inputs, outputs=outputs,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT16,
    )

    # No quantization for vocoder — convolutions are sensitive to weight quantization
    model_path = output_dir / "MimiDecoder.mlpackage"
    print(f"  Saving to {model_path}...", flush=True)
    mlmodel.save(str(model_path))

    size_mb = sum(f.stat().st_size for f in model_path.rglob("*") if f.is_file()) / 1e6
    print(f"  Saved ({size_mb:.0f} MB)")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Convert Qwen3-TTS to CoreML (V2)")
    parser.add_argument("--model-id", default="Qwen/Qwen3-TTS-12Hz-0.6B-Base",
                        help="HuggingFace model ID")
    parser.add_argument("--output", default="models/Qwen3-TTS-CoreML",
                        help="Output directory")
    parser.add_argument("--quantize", choices=["none", "int8"],
                        default="int8", help="Quantization: int8 (W8A16 palettization) or none (FP16)")
    parser.add_argument("--max-seq-len", type=int, default=256,
                        help="Maximum sequence length for Talker")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download
    print(f"Downloading {args.model_id}...")
    model_dir = Path(snapshot_download(args.model_id))
    print(f"  Source: {model_dir}")

    # Load main model weights (Talker + CodePredictor)
    print("Loading main model weights...")
    weights = {}
    for f in sorted(model_dir.glob("*.safetensors")):
        with safe_open(str(f), framework="pt") as sf:
            for key in sf.keys():
                weights[key] = sf.get_tensor(key)
    print(f"  {len(weights)} tensors")

    # Load speech tokenizer weights (separate file)
    st_weights = {}
    st_path = model_dir / "speech_tokenizer" / "model.safetensors"
    if st_path.exists():
        print("Loading speech tokenizer weights...")
        with safe_open(str(st_path), framework="pt") as sf:
            for key in sf.keys():
                st_weights[key] = sf.get_tensor(key)
        print(f"  {len(st_weights)} tensors")
    else:
        print(f"  WARNING: {st_path} not found — MimiDecoder conversion will be skipped")

    # Phase 1: Talker (V2 Conv2d architecture)
    convert_talker_v2(weights, output_dir, args.quantize, args.max_seq_len)

    # Phase 2: Extract embeddings
    extract_embeddings(weights, output_dir)

    # Phase 3: Code Predictor
    convert_code_predictor(weights, output_dir, args.quantize)

    # Phase 4: MimiDecoder
    if st_weights:
        convert_mimi_decoder(st_weights, output_dir)
    else:
        print("\n[MimiDecoder conversion skipped — speech_tokenizer weights not found]")

    # Verify Talker
    print("\n=== Verification ===")
    import coremltools as ct
    mlmodel = ct.models.MLModel(str(output_dir / "Talker.mlpackage"))

    total_d = 28 * 8 * 128
    test_inputs = {
        "input_embeds": np.random.randn(1, 1024, 1, 1).astype(np.float16),
        "rope_cos": np.ones((1, 128, 1), dtype=np.float16),
        "rope_sin": np.zeros((1, 128, 1), dtype=np.float16),
        "key_padding_mask": np.full((1, args.max_seq_len), MASK_VALUE, dtype=np.float16),
        "kv_cache_update_mask": np.zeros((1, args.max_seq_len), dtype=np.float16),
        "key_cache": np.zeros((1, total_d, 1, args.max_seq_len), dtype=np.float16),
        "value_cache": np.zeros((1, total_d, 1, args.max_seq_len), dtype=np.float16),
    }
    test_inputs["key_padding_mask"][0, 0] = 0  # unmask first position
    test_inputs["kv_cache_update_mask"][0, 0] = 1.0

    result = mlmodel.predict(test_inputs)
    logits = result["logits"]
    print(f"  Talker logits: shape={logits.shape}, range=[{logits.min():.2f}, {logits.max():.2f}]")
    print(f"  Talker hidden: shape={result['hidden_states'].shape}")
    print(f"  Talker KV updates: shape={result['key_cache_updates'].shape}")
    print("  Talker: OK")

    # Save config
    config = {
        "model_type": "qwen3_tts_coreml_v2",
        "model_id": args.model_id,
        "hidden_size": 1024,
        "num_layers": 28,
        "num_heads": 16,
        "num_kv_heads": 8,
        "head_dim": 128,
        "intermediate_size": 3072,
        "codec_vocab_size": 3072,
        "text_hidden_size": 2048,
        "text_vocab_size": 151936,
        "max_seq_len": args.max_seq_len,
        "quantization": args.quantize,
        "architecture": "conv2d_stacked_kv",
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Summary with file sizes
    print(f"\nDone! Output: {output_dir}")
    for f in sorted(output_dir.iterdir()):
        if f.is_file():
            print(f"  {f.name}: {f.stat().st_size / 1e6:.0f} MB")
        elif f.is_dir():
            size = sum(p.stat().st_size for p in f.rglob("*") if p.is_file())
            print(f"  {f.name}/: {size / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
