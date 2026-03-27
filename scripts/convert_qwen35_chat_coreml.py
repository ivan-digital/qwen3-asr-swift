#!/usr/bin/env python3
"""Convert Qwen3.5-0.8B to CoreML with stateful KV cache and DeltaNet state.

Qwen3.5-0.8B uses a hybrid architecture:
  - 18 DeltaNet layers (linear attention with gated output, causal conv1d)
  - 6 GatedAttention layers (full attention with output gate, partial RoPE)
  - Pattern: layers 0,1,2 = DeltaNet, layer 3 = GatedAttention, repeat 6 times
  - Tied embeddings (lm_head = embed_tokens)

DeltaNet layers use linear attention (no softmax). The recurrence
  S_t = alpha_t * S_t-1 + beta_t * (v_t @ k_t^T)
is expanded into a chunk-wise parallel form for CoreML tracing.

The conv1d state (kernel=4) and DeltaNet recurrent state are carried
across inference steps via MLState (ct.StateType).

GatedAttention layers use standard KV cache (MLState) with partial RoPE
(25% of head_dim = 64 rotary dims out of 256) and output gating.

Produces two CoreML models:
  1. embedding.mlpackage  -- token_id -> embedding lookup (stateless)
  2. decoder.mlpackage    -- single-step decoder with all states

Requires:
  pip install torch transformers coremltools safetensors numpy huggingface_hub

Usage:
  python scripts/convert_qwen35_chat_coreml.py
  python scripts/convert_qwen35_chat_coreml.py --hf-model Qwen/Qwen3.5-0.8B
  python scripts/convert_qwen35_chat_coreml.py --output models/Qwen35-0.8B-Chat-CoreML --quantize int4
  python scripts/convert_qwen35_chat_coreml.py --compile
"""

import argparse
import gc
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Force unbuffered output so we see prints before a crash
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# ── Model config for Qwen3.5-0.8B ──

MAX_SEQ = 2048  # Fixed KV cache / mask size for GatedAttention layers

MODEL_CONFIG = {
    "hidden_size": 1024,
    "num_layers": 24,
    "vocab_size": 248320,
    "intermediate_size": 3584,
    "rms_norm_eps": 1e-6,

    # DeltaNet config (18 layers: indices 0,1,2, 4,5,6, 8,9,10, 12,13,14, 16,17,18, 20,21,22)
    "deltanet_num_heads": 16,
    "deltanet_head_dim": 128,       # 6144 / 3 / 16 = 128
    "deltanet_conv_kernel": 4,
    "deltanet_gate_dim": 2048,      # in_proj_z output dim
    "deltanet_qkv_dim": 6144,      # in_proj_qkv output dim (3 * 16 * 128)
    "deltanet_norm_dim": 128,       # group norm per head

    # GatedAttention config (6 layers: indices 3, 7, 11, 15, 19, 23)
    "attn_num_heads": 16,           # 4096 / 256 = 16
    "attn_num_kv_heads": 2,         # 512 / 256 = 2
    "attn_head_dim": 256,
    "attn_out_gate_dim": 2048,      # o_proj output is 2*hidden
    "rope_theta": 10000000.0,
    "partial_rotary_factor": 0.25,  # rotary_dim = 256 * 0.25 = 64
}

# Layer type pattern: 3 DeltaNet + 1 GatedAttention, repeated 6 times
LAYER_PATTERN = ["deltanet", "deltanet", "deltanet", "gated_attn"] * 6


def layer_type(layer_idx):
    """Return 'deltanet' or 'gated_attn' for a given layer index."""
    return LAYER_PATTERN[layer_idx]


def deltanet_layer_indices():
    """Return sorted list of DeltaNet layer indices."""
    return [i for i in range(MODEL_CONFIG["num_layers"]) if layer_type(i) == "deltanet"]


def gated_attn_layer_indices():
    """Return sorted list of GatedAttention layer indices."""
    return [i for i in range(MODEL_CONFIG["num_layers"]) if layer_type(i) == "gated_attn"]


# ── Shared components ──

class RMSNorm(nn.Module):
    """RMSNorm."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        x_float = x.float()
        norm = torch.rsqrt(x_float.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x_float * norm * self.weight).to(x.dtype)


class SwiGLUMLP(nn.Module):
    """SwiGLU MLP shared by both layer types."""

    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ── DeltaNet layer ──

class DeltaNetAttention(nn.Module):
    """DeltaNet linear attention with gated output, single-step inference.

    For single-token generation (T=1), the DeltaNet recurrence simplifies to:
      1. Project input to Q, K, V via in_proj_qkv
      2. Apply causal conv1d (kernel=4) using carried state
      3. Compute alpha = sigmoid(in_proj_a(x)), beta = sigmoid(in_proj_b(x))
      4. Update recurrent state: S = alpha * S + beta * (v outer k)
      5. Output = S @ q, gated by z = silu(in_proj_z(x))
      6. Group norm + output projection

    State carried across steps:
      - conv_state: [1, 6144, 3] -- last 3 inputs for causal conv1d (kernel=4)
      - recurrent_state: [1, 16, 128, 128] -- per-head S matrix (value outer key)
    """

    def __init__(self, config):
        super().__init__()
        self.num_heads = config["deltanet_num_heads"]
        self.head_dim = config["deltanet_head_dim"]
        self.conv_kernel = config["deltanet_conv_kernel"]
        hidden = config["hidden_size"]
        qkv_dim = config["deltanet_qkv_dim"]
        gate_dim = config["deltanet_gate_dim"]

        # Projections
        self.in_proj_qkv = nn.Linear(hidden, qkv_dim, bias=False)    # [6144, 1024]
        self.in_proj_z = nn.Linear(hidden, gate_dim, bias=False)      # [2048, 1024]
        self.in_proj_b = nn.Linear(hidden, self.num_heads, bias=False)  # [16, 1024]
        self.in_proj_a = nn.Linear(hidden, self.num_heads, bias=False)  # [16, 1024]

        # Causal conv1d: applied to QKV concatenated, kernel=4, groups=6144
        # Weight shape: [6144, 1, 4] (depthwise)
        self.conv1d = nn.Conv1d(
            qkv_dim, qkv_dim, kernel_size=self.conv_kernel,
            groups=qkv_dim, bias=False,
        )

        # Per-head learnable parameters
        self.dt_bias = nn.Parameter(torch.zeros(self.num_heads))
        self.A_log = nn.Parameter(torch.zeros(self.num_heads))

        # Group norm (per head, dim=128)
        # RMSNormGated: weight shape [head_dim], applied per-head then gated by silu(z)
        self.norm = nn.Module()
        self.norm.weight = nn.Parameter(torch.ones(self.head_dim))

        # Output projection: [1024, 2048] — gated output, takes half
        self.out_proj = nn.Linear(gate_dim, hidden, bias=False)

    def forward(self, x, conv_state, recurrent_state):
        """Single-step DeltaNet forward.

        Args:
            x: [1, 1, hidden_size] -- single token input
            conv_state: [1, 6144, K-1] -- causal conv1d state (K-1=3 previous steps)
            recurrent_state: [1, 16, 128, 128] -- per-head recurrent state S

        Returns:
            output: [1, 1, hidden_size]
            new_conv_state: [1, 6144, K-1]
            new_recurrent_state: [1, 16, 128, 128]
        """
        B = x.shape[0]  # 1
        hidden = x.shape[2]

        # Project to QKV: [1, 1, 6144]
        qkv = self.in_proj_qkv(x)  # [1, 1, 6144]

        # Gate: z = silu(in_proj_z(x)), [1, 1, 2048]
        z = F.silu(self.in_proj_z(x))

        # Causal conv1d with carried state
        # conv_state: [1, 6144, 3], new input: [1, 6144, 1]
        qkv_t = qkv.transpose(1, 2)  # [1, 6144, 1]

        # Concatenate state + new input: [1, 6144, 4]
        conv_input = torch.cat([conv_state, qkv_t], dim=2)

        # Apply depthwise conv (no padding needed, we have exact kernel_size inputs)
        qkv_conv = F.conv1d(
            conv_input,
            self.conv1d.weight,
            bias=None,
            groups=self.conv1d.weight.shape[0],
        )  # [1, 6144, 1]

        # Update conv state: shift left, append new input
        # New state = last (kernel-1) columns of [old_state, new_input]
        new_conv_state = conv_input[:, :, 1:]  # [1, 6144, 3]

        # Apply SiLU activation to conv output (DeltaNet uses silu after conv)
        qkv_conv = F.silu(qkv_conv)  # [1, 6144, 1]
        qkv_conv = qkv_conv.transpose(1, 2)  # [1, 1, 6144]

        # Split QKV: each [1, 1, 2048] -> reshape to [1, 16, 1, 128]
        q, k, v = qkv_conv.chunk(3, dim=-1)
        q = q.view(B, 1, self.num_heads, self.head_dim)  # [1, 1, 16, 128]
        k = k.view(B, 1, self.num_heads, self.head_dim)  # [1, 1, 16, 128]
        v = v.view(B, 1, self.num_heads, self.head_dim)  # [1, 1, 16, 128]

        # Compute beta: sigmoid(in_proj_b(x)), [1, 1, 16]
        beta = torch.sigmoid(self.in_proj_b(x))  # [1, 1, 16]

        # Compute g (gated decay): g = -A_log.exp() * softplus(a + dt_bias)
        # g is negative so exp(g) < 1 (decay)
        a = self.in_proj_a(x)  # [1, 1, 16]
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)  # [1, 1, 16]

        # L2-normalize q and k
        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)

        # Scale q
        scale = 1.0 / (self.head_dim ** 0.5)
        q = q * scale

        # Transpose to [B, H, T, D] for recurrence
        q = q.transpose(1, 2)  # [1, 16, 1, 128]
        k = k.transpose(1, 2)  # [1, 16, 1, 128]
        v = v.transpose(1, 2)  # [1, 16, 1, 128]
        beta = beta.transpose(1, 2)  # [1, 16, 1]
        g = g.transpose(1, 2)  # [1, 16, 1]

        # Gated delta rule recurrence (single step):
        #   S = exp(g) * S + k^T * (beta * (v - S @ k))
        #   output = S @ q
        g_t = g[:, :, 0].exp().unsqueeze(-1).unsqueeze(-1)  # [1, 16, 1, 1]
        beta_t = beta[:, :, 0].unsqueeze(-1)  # [1, 16, 1]
        k_t = k[:, :, 0]  # [1, 16, 128]
        v_t = v[:, :, 0]  # [1, 16, 128]
        q_t = q[:, :, 0]  # [1, 16, 128]

        # Decay existing state
        new_recurrent_state = recurrent_state * g_t  # [1, 16, 128, 128]

        # Predict current v from state: kv_mem = sum_d(S[h,d,:] * k[h,d]) for each v-dim
        kv_mem = (new_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)  # [1, 16, 128]

        # Delta correction: how much v differs from prediction
        delta = (v_t - kv_mem) * beta_t  # [1, 16, 128]

        # Update state: S += k^T * delta (outer product update)
        new_recurrent_state = new_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)  # [1, 16, 128, 128]

        # Read output using query
        o = (new_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)  # [1, 16, 128]

        # Reshape to [1, 1, 2048] (flatten heads)
        o = o.unsqueeze(1).reshape(B, 1, self.num_heads * self.head_dim)

        # RMSNorm gated by z: norm(o) * silu(z)
        # z: [1, 1, 2048], o: [1, 1, 2048]
        z_reshaped = z.view(B, 1, self.num_heads, self.head_dim)
        o_reshaped = o.view(B, 1, self.num_heads, self.head_dim)
        # Per-head RMSNorm
        o_normed = o_reshaped.float()
        variance = o_normed.pow(2).mean(-1, keepdim=True)
        o_normed = o_normed * torch.rsqrt(variance + 1e-6)
        o_normed = self.norm.weight * o_normed.to(o.dtype)
        # Apply silu gate
        o_normed = o_normed * F.silu(z_reshaped.float()).to(o.dtype)
        o = o_normed.reshape(B, 1, self.num_heads * self.head_dim)

        # Project to hidden_size
        output = self.out_proj(o)  # [1, 1, 1024]

        return output, new_conv_state, new_recurrent_state


# ── GatedAttention layer ──

class PartialRoPE(nn.Module):
    """Partial split-half rotary position embeddings.

    Only the first rotary_dim dimensions get rotation; the rest pass through.
    For Qwen3.5: head_dim=256, partial_rotary_factor=0.25 -> rotary_dim=64.
    """

    def __init__(self, head_dim, rotary_dim, theta=10000000.0):
        super().__init__()
        self.rotary_dim = rotary_dim
        half = rotary_dim // 2
        inv_freq = 1.0 / (theta ** (torch.arange(0, half, dtype=torch.float32) / half))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x, position):
        """Apply partial RoPE.

        Args:
            x: [B, N_heads, 1, head_dim]
            position: [1] int32

        Returns:
            x with first rotary_dim dimensions rotated
        """
        freqs = position.float() * self.inv_freq  # [half]
        cos_val = torch.cos(freqs).view(1, 1, 1, -1)  # [1, 1, 1, half]
        sin_val = torch.sin(freqs).view(1, 1, 1, -1)  # [1, 1, 1, half]

        half = self.rotary_dim // 2
        # Split into rotary and passthrough parts
        x_rot = x[..., :self.rotary_dim]
        x_pass = x[..., self.rotary_dim:]

        # Split-half rotation on the rotary portion
        x1, x2 = x_rot[..., :half], x_rot[..., half:]
        x_rotated = torch.cat([
            x1 * cos_val - x2 * sin_val,
            x2 * cos_val + x1 * sin_val,
        ], dim=-1)

        return torch.cat([x_rotated, x_pass], dim=-1)


class GatedSelfAttention(nn.Module):
    """GatedAttention with KV cache (MLState), partial RoPE, and output gating.

    GatedAttention differs from standard attention:
    - o_proj outputs 2*hidden_size, split into [hidden, gate]
    - Output = hidden * sigmoid(gate)
    - Q/K have per-head RMSNorm
    - Partial RoPE (only 25% of head_dim)
    """

    def __init__(self, config):
        super().__init__()
        self.num_heads = config["attn_num_heads"]
        self.num_kv_heads = config["attn_num_kv_heads"]
        self.head_dim = config["attn_head_dim"]
        self.groups = self.num_heads // self.num_kv_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        hidden = config["hidden_size"]
        gate_dim = config["attn_out_gate_dim"]
        rotary_dim = int(self.head_dim * config["partial_rotary_factor"])

        self.q_proj = nn.Linear(hidden, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, self.num_kv_heads * self.head_dim, bias=False)
        # Output projection: outputs 2*hidden for gating
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, gate_dim, bias=False)

        self.q_norm = RMSNorm(self.head_dim, eps=config["rms_norm_eps"])
        self.k_norm = RMSNorm(self.head_dim, eps=config["rms_norm_eps"])
        self.rope = PartialRoPE(self.head_dim, rotary_dim, theta=config["rope_theta"])

    def forward(self, x, position, mask, k_cache, v_cache, onehot):
        """Single-step GatedAttention forward with KV cache.

        Args:
            x: [1, 1, hidden_size]
            position: [1] int32
            mask: [1, 1, 1, MAX_SEQ]
            k_cache: [1, num_kv_heads, MAX_SEQ, head_dim] buffer
            v_cache: [1, num_kv_heads, MAX_SEQ, head_dim] buffer
            onehot: [1, 1, MAX_SEQ, 1] precomputed

        Returns:
            output: [1, 1, hidden_size]
        """
        B, T, _ = x.shape  # T=1
        hidden = x.shape[2]

        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Per-head Q/K normalization
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Partial RoPE
        q = self.rope(q, position)
        k = self.rope(k, position)

        # In-place KV cache update via one-hot scatter
        k_cache.mul_(1 - onehot)
        k_cache.add_(k * onehot)
        v_cache.mul_(1 - onehot)
        v_cache.add_(v * onehot)

        # GQA: expand KV heads
        k_exp = k_cache.unsqueeze(2).expand(-1, -1, self.groups, -1, -1)
        k_exp = k_exp.reshape(B, self.num_heads, MAX_SEQ, self.head_dim)
        v_exp = v_cache.unsqueeze(2).expand(-1, -1, self.groups, -1, -1)
        v_exp = v_exp.reshape(B, self.num_heads, MAX_SEQ, self.head_dim)

        # Scaled dot-product attention
        attn = torch.matmul(q, k_exp.transpose(-2, -1)) * self.scale
        attn = attn + mask
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v_exp)

        out = out.transpose(1, 2).reshape(B, T, -1)  # [1, 1, num_heads*head_dim]

        # Gated output: o_proj -> [1, 1, 2*hidden], split and gate
        projected = self.o_proj(out)  # [1, 1, 2048]
        out_val, gate = projected.chunk(2, dim=-1)  # each [1, 1, 1024]
        output = out_val * torch.sigmoid(gate)  # [1, 1, 1024]

        return output


# ── Unified layer wrappers ──

class DeltaNetLayer(nn.Module):
    """DeltaNet layer with pre-norm, attention, and SwiGLU MLP."""

    def __init__(self, config):
        super().__init__()
        self.linear_attn = DeltaNetAttention(config)
        self.mlp = SwiGLUMLP(config["hidden_size"], config["intermediate_size"])
        self.input_layernorm = RMSNorm(config["hidden_size"], eps=config["rms_norm_eps"])
        self.post_attention_layernorm = RMSNorm(config["hidden_size"], eps=config["rms_norm_eps"])

    def forward(self, x, conv_state, recurrent_state):
        residual = x
        x_norm = self.input_layernorm(x)
        attn_out, new_conv_state, new_recurrent_state = self.linear_attn(
            x_norm, conv_state, recurrent_state
        )
        x = residual + attn_out

        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x

        return x, new_conv_state, new_recurrent_state


class GatedAttnLayer(nn.Module):
    """GatedAttention layer with pre-norm, attention, and SwiGLU MLP."""

    def __init__(self, config):
        super().__init__()
        self.self_attn = GatedSelfAttention(config)
        self.mlp = SwiGLUMLP(config["hidden_size"], config["intermediate_size"])
        self.input_layernorm = RMSNorm(config["hidden_size"], eps=config["rms_norm_eps"])
        self.post_attention_layernorm = RMSNorm(config["hidden_size"], eps=config["rms_norm_eps"])

    def forward(self, x, position, mask, k_cache, v_cache, onehot):
        residual = x
        x_norm = self.input_layernorm(x)
        attn_out = self.self_attn(x_norm, position, mask, k_cache, v_cache, onehot)
        x = residual + attn_out

        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x

        return x


# ── Full decoder model ──

class Qwen35Decoder(nn.Module):
    """Qwen3.5-0.8B hybrid decoder with DeltaNet + GatedAttention layers.

    All state tensors (DeltaNet conv/recurrent, GatedAttention KV cache) are
    registered as named buffers so coremltools maps them to MLState.

    Forward signature: (input_embeds, position, mask) -> logits
    """

    def __init__(self, config=None):
        super().__init__()
        c = config or MODEL_CONFIG
        self.config = c
        self.num_layers = c["num_layers"]

        # Build layers according to pattern
        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            if layer_type(i) == "deltanet":
                self.layers.append(DeltaNetLayer(c))
            else:
                self.layers.append(GatedAttnLayer(c))

        self.norm = RMSNorm(c["hidden_size"], eps=c["rms_norm_eps"])
        # LM head (tied with embedding)
        self.lm_head = nn.Linear(c["hidden_size"], c["vocab_size"], bias=False)

        # Register state buffers for MLState mapping

        # DeltaNet states
        qkv_dim = c["deltanet_qkv_dim"]
        conv_k = c["deltanet_conv_kernel"]
        dn_heads = c["deltanet_num_heads"]
        dn_head_dim = c["deltanet_head_dim"]

        for i in deltanet_layer_indices():
            # Conv state: last (kernel-1) inputs
            self.register_buffer(
                f"conv_state_{i}",
                torch.zeros(1, qkv_dim, conv_k - 1)
            )
            # Recurrent state: per-head S matrix
            self.register_buffer(
                f"recurrent_state_{i}",
                torch.zeros(1, dn_heads, dn_head_dim, dn_head_dim)
            )

        # GatedAttention KV cache states
        attn_kv_heads = c["attn_num_kv_heads"]
        attn_head_dim = c["attn_head_dim"]

        for i in gated_attn_layer_indices():
            self.register_buffer(
                f"k_cache_{i}",
                torch.zeros(1, attn_kv_heads, MAX_SEQ, attn_head_dim)
            )
            self.register_buffer(
                f"v_cache_{i}",
                torch.zeros(1, attn_kv_heads, MAX_SEQ, attn_head_dim)
            )

    def forward(self, input_embeds, position, mask):
        """Single-step forward pass.

        Args:
            input_embeds: [1, 1, hidden_size] -- token embedding
            position: [1] int32 -- current position
            mask: [1, 1, 1, MAX_SEQ] -- attention mask for GatedAttention layers

        Returns:
            logits: [1, 1, vocab_size]
        """
        x = input_embeds

        # Precompute one-hot for GatedAttention KV cache updates
        onehot = F.one_hot(position.long(), MAX_SEQ).float().view(1, 1, MAX_SEQ, 1)

        for i in range(self.num_layers):
            if layer_type(i) == "deltanet":
                conv_buf = getattr(self, f"conv_state_{i}")
                rec_buf = getattr(self, f"recurrent_state_{i}")

                x, new_conv, new_rec = self.layers[i](x, conv_buf, rec_buf)

                # In-place state update for MLState tracing
                conv_buf.mul_(0)
                conv_buf.add_(new_conv)
                rec_buf.mul_(0)
                rec_buf.add_(new_rec)
            else:
                k_buf = getattr(self, f"k_cache_{i}")
                v_buf = getattr(self, f"v_cache_{i}")
                # GatedAttnLayer updates k_buf/v_buf in-place via mul_() + add_()
                x = self.layers[i](x, position, mask, k_buf, v_buf, onehot)

        x = self.norm(x)
        logits = self.lm_head(x)  # [1, 1, vocab_size]
        return logits


class EmbeddingLookup(nn.Module):
    """Token embedding lookup for CoreML conversion."""

    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)

    def forward(self, token_id):
        # token_id: [1, 1] int32
        return self.embedding(token_id)  # [1, 1, hidden_size]


# ── Weight loading ──

def download_weights(model_id, weights_dir=None):
    """Load model weights from HuggingFace or local directory.

    Returns flat dict with original HuggingFace key names.
    """
    from safetensors.torch import load_file

    if weights_dir:
        cache_dir = Path(weights_dir)
        print(f"Loading weights from {cache_dir}...")
    else:
        from huggingface_hub import snapshot_download
        print(f"Downloading weights from {model_id}...")
        cache_dir = Path(snapshot_download(
            model_id,
            allow_patterns=["*.safetensors", "model.safetensors.index.json"],
        ))

    all_weights = {}
    for f in sorted(cache_dir.glob("*.safetensors")):
        print(f"  Loading {f.name}...")
        all_weights.update(load_file(str(f)))

    print(f"  Total keys: {len(all_weights)}")

    # Strip common prefixes (model., thinker.model., etc.)
    stripped = {}
    for k, v in all_weights.items():
        for prefix in ["thinker.model.", "model."]:
            if k.startswith(prefix):
                stripped[k[len(prefix):]] = v
                break
        else:
            stripped[k] = v

    # Ensure embed_tokens is available
    if "embed_tokens.weight" not in stripped:
        for k, v in all_weights.items():
            if k.endswith("embed_tokens.weight"):
                stripped["embed_tokens.weight"] = v
                print(f"  Found embed_tokens via: {k}")
                break

    return stripped


def load_weights(decoder, embedding_model, weights):
    """Load weights into PyTorch decoder and embedding models.

    Maps HuggingFace weight keys to our module structure:
      HF: layers.{i}.self_attn.{in_proj_qkv,in_proj_z,...}.weight
      Us: layers.{i}.self_attn.{in_proj_qkv,in_proj_z,...}.weight

    DeltaNet and GatedAttention layers have different sub-modules but
    the layer index mapping is 1:1.
    """
    decoder_sd = decoder.state_dict()
    loaded = 0
    skipped = 0
    missing = []

    for key in decoder_sd:
        # Skip state buffers
        if any(key.startswith(p) for p in [
            "conv_state_", "recurrent_state_", "k_cache_", "v_cache_"
        ]):
            skipped += 1
            continue

        # Skip RoPE inv_freq buffers
        if "inv_freq" in key:
            skipped += 1
            continue

        # Map our key to HF key
        src_key = key

        # LM head uses tied embedding weight
        if key == "lm_head.weight":
            src_key = "embed_tokens.weight"

        if src_key in weights:
            if decoder_sd[key].shape == weights[src_key].shape:
                decoder_sd[key] = weights[src_key].float()
                loaded += 1
            else:
                print(f"  Shape mismatch: {key} "
                      f"model={list(decoder_sd[key].shape)} "
                      f"weight={list(weights[src_key].shape)}")
        else:
            missing.append(key)

    if missing:
        print(f"  Missing {len(missing)} weights:")
        for k in missing[:20]:
            print(f"    {k}")
        if len(missing) > 20:
            print(f"    ... and {len(missing) - 20} more")

    decoder.load_state_dict(decoder_sd)
    total = len(decoder_sd) - skipped
    print(f"  Decoder: loaded {loaded}/{total} weights (skipped {skipped} state buffers)")

    # Load embedding weights
    emb_sd = embedding_model.state_dict()
    if "embed_tokens.weight" in weights:
        emb_sd["embedding.weight"] = weights["embed_tokens.weight"].float()
        embedding_model.load_state_dict(emb_sd)
        print(f"  Embedding: loaded 1/1 weights")
    else:
        print(f"  WARNING: embed_tokens.weight not found!")

    return loaded == total


# ── CoreML conversion ──

def convert_embedding(traced_embedding, quantize_mode=None):
    """Convert embedding lookup to CoreML."""
    import coremltools as ct

    print("Converting embedding model...")
    mlmodel = ct.convert(
        traced_embedding,
        inputs=[ct.TensorType(name="token_id", shape=(1, 1), dtype=np.int32)],
        outputs=[ct.TensorType(name="embedding")],
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS18,
    )

    if quantize_mode == "int4":
        print("  Applying INT4 quantization to embedding...")
        op_config = ct.optimize.coreml.OpLinearQuantizerConfig(
            mode="linear_symmetric", dtype="int4",
            granularity="per_block", block_size=64,
        )
        config = ct.optimize.coreml.OptimizationConfig(global_config=op_config)
        mlmodel = ct.optimize.coreml.linear_quantize_weights(mlmodel, config=config)
    elif quantize_mode == "int8":
        print("  Applying INT8 palettization to embedding...")
        from coremltools.optimize.coreml import (
            OpPalettizerConfig, OptimizationConfig, palettize_weights,
        )
        op_config = OpPalettizerConfig(mode="kmeans", nbits=8)
        config = OptimizationConfig(global_config=op_config)
        mlmodel = palettize_weights(mlmodel, config)

    return mlmodel


def convert_decoder(traced_decoder, config, quantize_mode=None):
    """Convert decoder to CoreML with MLState for all stateful buffers.

    State mapping:
      - DeltaNet layers: conv_state_{i} [1, 6144, 3], recurrent_state_{i} [1, 16, 128, 128]
      - GatedAttention layers: k_cache_{i} [1, 2, MAX_SEQ, 256], v_cache_{i} [1, 2, MAX_SEQ, 256]
    """
    import coremltools as ct

    hidden = config["hidden_size"]

    inputs = [
        ct.TensorType(name="input_embeds", shape=(1, 1, hidden), dtype=np.float32),
        ct.TensorType(name="position", shape=(1,), dtype=np.int32),
        ct.TensorType(name="attention_mask", shape=(1, 1, 1, MAX_SEQ), dtype=np.float32),
    ]

    states = []

    # DeltaNet states
    qkv_dim = config["deltanet_qkv_dim"]
    conv_k = config["deltanet_conv_kernel"]
    dn_heads = config["deltanet_num_heads"]
    dn_head_dim = config["deltanet_head_dim"]

    for i in deltanet_layer_indices():
        states.append(ct.StateType(
            wrapped_type=ct.TensorType(shape=(1, qkv_dim, conv_k - 1)),
            name=f"conv_state_{i}",
        ))
        states.append(ct.StateType(
            wrapped_type=ct.TensorType(shape=(1, dn_heads, dn_head_dim, dn_head_dim)),
            name=f"recurrent_state_{i}",
        ))

    # GatedAttention KV cache states
    attn_kv_heads = config["attn_num_kv_heads"]
    attn_head_dim = config["attn_head_dim"]

    for i in gated_attn_layer_indices():
        states.append(ct.StateType(
            wrapped_type=ct.TensorType(shape=(1, attn_kv_heads, MAX_SEQ, attn_head_dim)),
            name=f"k_cache_{i}",
        ))
        states.append(ct.StateType(
            wrapped_type=ct.TensorType(shape=(1, attn_kv_heads, MAX_SEQ, attn_head_dim)),
            name=f"v_cache_{i}",
        ))

    outputs = [ct.TensorType(name="logits")]

    n_dn = len(deltanet_layer_indices())
    n_ga = len(gated_attn_layer_indices())
    print(f"Converting decoder ({n_dn} DeltaNet + {n_ga} GatedAttention layers, "
          f"{len(states)} state tensors)...")

    mlmodel = ct.convert(
        traced_decoder,
        inputs=inputs,
        states=states,
        outputs=outputs,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS18,
    )

    if quantize_mode == "int4":
        print("  Applying INT4 quantization...")
        op_config = ct.optimize.coreml.OpLinearQuantizerConfig(
            mode="linear_symmetric", dtype="int4",
            granularity="per_block", block_size=64,
        )
        quant_config = ct.optimize.coreml.OptimizationConfig(global_config=op_config)
        mlmodel = ct.optimize.coreml.linear_quantize_weights(mlmodel, config=quant_config)
    elif quantize_mode == "int8":
        print("  Applying INT8 palettization...")
        from coremltools.optimize.coreml import (
            OpPalettizerConfig, OptimizationConfig, palettize_weights,
        )
        op_config = OpPalettizerConfig(mode="kmeans", nbits=8)
        config = OptimizationConfig(global_config=op_config)
        mlmodel = palettize_weights(mlmodel, config)

    return mlmodel


def compile_mlpackage(output_dir, name):
    """Compile .mlpackage to .mlmodelc using xcrun."""
    pkg = output_dir / f"{name}.mlpackage"
    compiled = output_dir / f"{name}.mlmodelc"

    if compiled.exists():
        shutil.rmtree(compiled)

    print(f"Compiling {name}.mlpackage -> {name}.mlmodelc...")
    result = subprocess.run(
        ["xcrun", "coremlcompiler", "compile", str(pkg), str(output_dir)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"  xcrun failed: {result.stderr.strip()}")
        print("  Falling back to Python compilation...")
        import coremltools as ct
        compiled_url = ct.utils.compile_model(str(pkg))
        shutil.move(str(compiled_url), str(compiled))

    if compiled.exists():
        print(f"  Compiled to {compiled}")
    else:
        print(f"  ERROR: {compiled} not found after compilation")


# ── Verification ──

def verify_embedding(pt_model, coreml_path):
    """Verify CoreML embedding matches PyTorch."""
    import coremltools as ct

    print("\nVerifying embedding model...")
    coreml_model = ct.models.MLModel(str(coreml_path))
    max_diff = 0.0

    for token in [0, 1, 100, 1000, 5000]:
        tok = torch.tensor([[token]], dtype=torch.int32)
        with torch.no_grad():
            pt_out = pt_model(tok).numpy().flatten().astype(np.float32)

        cm_out = np.array(
            coreml_model.predict({"token_id": tok.numpy()})["embedding"]
        ).flatten().astype(np.float32)

        cos = float(np.dot(pt_out, cm_out) / (
            np.linalg.norm(pt_out) * np.linalg.norm(cm_out) + 1e-10))
        diff = 1.0 - cos
        max_diff = max(max_diff, diff)
        print(f"  token={token:6d}: cosine_sim={cos:.6f}")

    status = "PASS" if max_diff < 0.01 else "WARNING"
    print(f"  {status}: max (1-cosine_sim) = {max_diff:.6f}")
    return max_diff


def reset_all_states(decoder):
    """Zero out all state buffers in the decoder."""
    for i in deltanet_layer_indices():
        getattr(decoder, f"conv_state_{i}").zero_()
        getattr(decoder, f"recurrent_state_{i}").zero_()
    for i in gated_attn_layer_indices():
        getattr(decoder, f"k_cache_{i}").zero_()
        getattr(decoder, f"v_cache_{i}").zero_()


def verify_decoder(pt_decoder, pt_embedding, coreml_decoder_path, coreml_embedding_path):
    """Run a few decoder steps and verify against PyTorch."""
    import coremltools as ct

    print("\nVerifying decoder (5-step autoregressive)...")

    cml_decoder = ct.models.MLModel(str(coreml_decoder_path))
    cml_embedding = ct.models.MLModel(str(coreml_embedding_path))

    # Create MLState for stateful prediction
    decoder_state = cml_decoder.make_state()

    # Reset PyTorch state buffers
    reset_all_states(pt_decoder)

    # Test tokens (simple prompt)
    tokens = [151644, 8948, 198, 151645, 198]  # <|im_start|>system\n<|im_end|>\n
    match_count = 0

    for step, token in enumerate(tokens):
        # Build mask: allow positions 0..step
        mask = torch.full((1, 1, 1, MAX_SEQ), -1e4)
        mask[:, :, :, :step + 1] = 0
        mask_np = mask.numpy()
        position = torch.tensor([step], dtype=torch.int32)

        # PyTorch path
        with torch.no_grad():
            tok_t = torch.tensor([[token]], dtype=torch.int32)
            embed = pt_embedding(tok_t)
            pt_logits_t = pt_decoder(embed, position, mask)
            pt_logits = pt_logits_t.detach().numpy().flatten()
            pt_token = int(np.argmax(pt_logits))

        # CoreML path
        cm_embed = cml_embedding.predict({"token_id": np.array([[token]], dtype=np.int32)})
        cm_input = {
            "input_embeds": cm_embed["embedding"].astype(np.float32),
            "position": np.array([step], dtype=np.int32),
            "attention_mask": mask_np,
        }
        cm_result = cml_decoder.predict(cm_input, state=decoder_state)
        cm_logits = np.array(cm_result["logits"]).flatten().astype(np.float32)
        cm_token = int(np.argmax(cm_logits))

        cos = float(np.dot(pt_logits, cm_logits) / (
            np.linalg.norm(pt_logits) * np.linalg.norm(cm_logits) + 1e-10))
        marker = "ok" if pt_token == cm_token else "MISMATCH"
        if pt_token == cm_token:
            match_count += 1
        print(f"  step={step}: cos_sim={cos:.6f}, "
              f"pt_token={pt_token}, cm_token={cm_token} {marker}")

    print(f"  Token match: {match_count}/{len(tokens)}")


# ── Main ──

def main():
    global MAX_SEQ

    parser = argparse.ArgumentParser(
        description="Convert Qwen3.5-0.8B (hybrid DeltaNet + GatedAttention) to CoreML"
    )
    parser.add_argument("--hf-model", default="Qwen/Qwen3.5-0.8B",
                        help="HuggingFace model ID")
    parser.add_argument("--weights-dir",
                        help="Local directory with safetensors (skip HF download)")
    parser.add_argument("--output", default="models/Qwen35-0.8B-Chat-CoreML",
                        help="Output directory")
    parser.add_argument("--quantize", choices=["none", "int4", "int8"],
                        default="int4", help="Quantization mode")
    parser.add_argument("--max-seq-len", type=int, default=MAX_SEQ,
                        help=f"Maximum sequence length for KV cache (default: {MAX_SEQ})")
    parser.add_argument("--compile", action="store_true",
                        help="Compile .mlpackage to .mlmodelc")
    parser.add_argument("--skip-verify", action="store_true",
                        help="Skip verification step")
    parser.add_argument("--upload", type=str, default=None,
                        help="HuggingFace repo ID to upload")
    args = parser.parse_args()

    MAX_SEQ = args.max_seq_len
    quantize_mode = args.quantize if args.quantize != "none" else None
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = MODEL_CONFIG

    # ── Phase 1: Download weights ──
    print("=" * 60)
    print("Phase 1: Download weights")
    print("=" * 60)
    weights = download_weights(args.hf_model, args.weights_dir)

    # Print layer type summary
    print(f"\n  Layer pattern ({config['num_layers']} layers):")
    for i in range(config["num_layers"]):
        lt = layer_type(i)
        marker = "DeltaNet" if lt == "deltanet" else "GatedAttn"
        print(f"    Layer {i:2d}: {marker}")

    # ── Phase 2: Build PyTorch models ──
    print("\n" + "=" * 60)
    print("Phase 2: Build PyTorch models")
    print("=" * 60)

    decoder = Qwen35Decoder(config)
    embedding = EmbeddingLookup(config["vocab_size"], config["hidden_size"])

    dec_params = sum(p.numel() for p in decoder.parameters())
    emb_params = sum(p.numel() for p in embedding.parameters())
    print(f"  Decoder: {dec_params:,} params ({dec_params * 4 / 1024 / 1024:.1f} MB FP32)")
    print(f"  Embedding: {emb_params:,} params ({emb_params * 4 / 1024 / 1024:.1f} MB FP32)")

    # Count state buffer sizes
    dn_conv_size = sum(
        getattr(decoder, f"conv_state_{i}").numel()
        for i in deltanet_layer_indices()
    )
    dn_rec_size = sum(
        getattr(decoder, f"recurrent_state_{i}").numel()
        for i in deltanet_layer_indices()
    )
    ga_kv_size = sum(
        getattr(decoder, f"k_cache_{i}").numel() + getattr(decoder, f"v_cache_{i}").numel()
        for i in gated_attn_layer_indices()
    )
    print(f"  DeltaNet conv states: {dn_conv_size:,} elements "
          f"({dn_conv_size * 2 / 1024 / 1024:.1f} MB FP16)")
    print(f"  DeltaNet recurrent states: {dn_rec_size:,} elements "
          f"({dn_rec_size * 2 / 1024 / 1024:.1f} MB FP16)")
    print(f"  GatedAttention KV cache: {ga_kv_size:,} elements "
          f"({ga_kv_size * 2 / 1024 / 1024:.1f} MB FP16)")

    # ── Phase 3: Load weights ──
    print("\n" + "=" * 60)
    print("Phase 3: Load weights")
    print("=" * 60)
    load_weights(decoder, embedding, weights)
    decoder.eval()
    embedding.eval()
    del weights
    gc.collect()

    # ── Phase 4: Trace models ──
    print("\n" + "=" * 60)
    print("Phase 4: Trace models")
    print("=" * 60)

    # Trace embedding
    example_token = torch.tensor([[5]], dtype=torch.int32)
    with torch.no_grad():
        traced_embedding = torch.jit.trace(embedding, (example_token,))

    # Verify embedding trace
    with torch.no_grad():
        ref = embedding(example_token)
        trc = traced_embedding(example_token)
    diff = (ref - trc).abs().max().item()
    print(f"  Embedding trace diff: {diff:.2e}")
    assert diff < 1e-5, f"Embedding trace mismatch: {diff}"

    # Trace decoder
    hidden = config["hidden_size"]
    example_embeds = torch.randn(1, 1, hidden)
    example_position = torch.tensor([0], dtype=torch.int32)
    example_mask = torch.zeros(1, 1, 1, MAX_SEQ)

    n_states = (len(deltanet_layer_indices()) * 2 + len(gated_attn_layer_indices()) * 2)
    print(f"  Tracing decoder ({config['num_layers']} layers, "
          f"{n_states} state buffers)...")

    with torch.no_grad():
        traced_decoder = torch.jit.trace(
            decoder,
            (example_embeds, example_position, example_mask),
        )

    # Reset buffers after trace (in-place updates modify them)
    reset_all_states(decoder)

    # Verify decoder trace
    with torch.no_grad():
        ref = decoder(example_embeds, example_position, example_mask)
        reset_all_states(decoder)
        trc = traced_decoder(example_embeds, example_position, example_mask)
    diff = (ref - trc).abs().max().item()
    print(f"  Decoder trace diff: {diff:.2e}")
    assert diff < 1e-5, f"Decoder trace mismatch: {diff}"
    print("  Trace verified successfully.")

    # ── Phase 5: Convert to CoreML ──
    print("\n" + "=" * 60)
    print("Phase 5: Convert to CoreML")
    print("=" * 60)

    # Import coremltools AFTER tracing to avoid segfault with torch
    import coremltools as ct

    # Convert embedding
    mlmodel_emb = convert_embedding(traced_embedding, quantize_mode=quantize_mode)
    emb_path = output_dir / "embedding.mlpackage"
    if emb_path.exists():
        shutil.rmtree(emb_path)
    mlmodel_emb.save(str(emb_path))
    emb_size = sum(f.stat().st_size for f in emb_path.rglob("*") if f.is_file()) / 1024 / 1024
    print(f"  Saved embedding.mlpackage ({emb_size:.1f} MB)")
    del mlmodel_emb
    gc.collect()

    # Convert decoder
    mlmodel_dec = convert_decoder(traced_decoder, config, quantize_mode=quantize_mode)
    dec_path = output_dir / "decoder.mlpackage"
    if dec_path.exists():
        shutil.rmtree(dec_path)
    mlmodel_dec.save(str(dec_path))
    dec_size = sum(f.stat().st_size for f in dec_path.rglob("*") if f.is_file()) / 1024 / 1024
    print(f"  Saved decoder.mlpackage ({dec_size:.1f} MB)")
    del mlmodel_dec, traced_embedding, traced_decoder
    gc.collect()

    # ── Phase 6: Verify ──
    if not args.skip_verify:
        print("\n" + "=" * 60)
        print("Phase 6: Verify")
        print("=" * 60)
        verify_embedding(embedding, emb_path)
        verify_decoder(decoder, embedding, dec_path, emb_path)

    # ── Phase 7: Compile ──
    if args.compile:
        print("\n" + "=" * 60)
        print("Phase 7: Compile")
        print("=" * 60)
        compile_mlpackage(output_dir, "embedding")
        if (output_dir / "embedding.mlpackage").exists():
            shutil.rmtree(output_dir / "embedding.mlpackage")
        compile_mlpackage(output_dir, "decoder")
        if (output_dir / "decoder.mlpackage").exists():
            shutil.rmtree(output_dir / "decoder.mlpackage")

    del decoder, embedding
    gc.collect()

    # ── Save config ──
    tokenizer_info = {}
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.hf_model)
        tokenizer.save_pretrained(str(output_dir))
        tokenizer_info = {
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        }
        print(f"  Saved tokenizer to {output_dir}")
    except Exception as e:
        print(f"  Tokenizer save skipped: {e}")

    out_config = {
        "model_type": "qwen3.5-hybrid",
        "source_model": args.hf_model,
        "architecture": "deltanet+gated_attention",
        "max_seq_length": MAX_SEQ,
        "hidden_size": config["hidden_size"],
        "num_layers": config["num_layers"],
        "vocab_size": config["vocab_size"],
        "intermediate_size": config["intermediate_size"],
        "rms_norm_eps": config["rms_norm_eps"],
        "tied_embeddings": True,

        "deltanet": {
            "num_layers": len(deltanet_layer_indices()),
            "layer_indices": deltanet_layer_indices(),
            "num_heads": config["deltanet_num_heads"],
            "head_dim": config["deltanet_head_dim"],
            "conv_kernel": config["deltanet_conv_kernel"],
            "qkv_dim": config["deltanet_qkv_dim"],
            "gate_dim": config["deltanet_gate_dim"],
        },
        "gated_attention": {
            "num_layers": len(gated_attn_layer_indices()),
            "layer_indices": gated_attn_layer_indices(),
            "num_heads": config["attn_num_heads"],
            "num_kv_heads": config["attn_num_kv_heads"],
            "head_dim": config["attn_head_dim"],
            "rope_theta": config["rope_theta"],
            "partial_rotary_factor": config["partial_rotary_factor"],
        },

        "quantization": args.quantize,
        "files": {
            "embedding": "embedding.mlpackage",
            "decoder": "decoder.mlpackage",
        },
        **tokenizer_info,
    }

    config_path = output_dir / "chat_config.json"
    with open(config_path, "w") as f:
        json.dump(out_config, f, indent=2)
    print(f"\nSaved chat_config.json")

    # Summary
    print(f"\nDone! Output in: {output_dir}")
    for f in sorted(output_dir.iterdir()):
        if f.is_dir():
            sz = sum(ff.stat().st_size for ff in f.rglob("*") if ff.is_file())
        else:
            sz = f.stat().st_size
        print(f"  {f.name}: {sz / 1024 / 1024:.1f} MB")

    if args.upload:
        upload_to_hf(str(output_dir), args.upload)


def upload_to_hf(output_dir, repo_id):
    """Upload converted model to HuggingFace Hub."""
    from huggingface_hub import HfApi, create_repo

    api = HfApi()
    out_path = Path(output_dir)

    print(f"\nCreating/updating repo: {repo_id}")
    create_repo(repo_id, repo_type="model", exist_ok=True)

    files = [f for f in out_path.rglob("*") if f.is_file()]
    print(f"Uploading {len(files)} files...")
    api.upload_folder(
        folder_path=str(out_path),
        repo_id=repo_id,
        repo_type="model",
    )
    print(f"Uploaded to: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
