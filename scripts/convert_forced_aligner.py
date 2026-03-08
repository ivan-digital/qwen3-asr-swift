#!/usr/bin/env python3
"""
Convert Qwen3-ForcedAligner-0.6B to MLX safetensors or CoreML.

Produces five variants from Qwen/Qwen3-ForcedAligner-0.6B:
  MLX 4-bit:   text decoder 4-bit quantized  (~980 MB)
  MLX 8-bit:   text decoder 8-bit quantized  (~1.4 GB)
  MLX bf16:    text decoder float16           (~2.2 GB)
  CoreML INT4: full model palettized INT4     (~350 MB)
  CoreML INT8: full model palettized INT8     (~600 MB)

Usage:
  python scripts/convert_forced_aligner.py --bits 4
  python scripts/convert_forced_aligner.py --bits 8
  python scripts/convert_forced_aligner.py --bits 0   # bf16 (no quantization)
  python scripts/convert_forced_aligner.py --coreml --coreml-bits 4
  python scripts/convert_forced_aligner.py --coreml --coreml-bits 8
  python scripts/convert_forced_aligner.py --bits 4 --upload --repo-id aufklarer/Qwen3-ForcedAligner-0.6B-4bit
"""

import argparse
import gc
import json
import shutil
from pathlib import Path

import torch
import numpy as np
from huggingface_hub import snapshot_download
from safetensors.numpy import save_file
from safetensors.torch import load_file


# ---------------------------------------------------------------------------
# Group quantization (MLX-compatible format)
# ---------------------------------------------------------------------------

def quantize_nbit(weight: torch.Tensor, bits: int, group_size: int = 64):
    """Quantize a 2-D float weight to N-bit with per-group scales and biases.

    Returns (packed_uint32, scales, biases) matching MLX QuantizedLinear format.
    """
    assert weight.ndim == 2, f"Expected 2-D tensor, got {weight.ndim}-D"
    assert bits in (4, 8), f"Supported bits: 4, 8 (got {bits})"
    rows, cols = weight.shape
    assert cols % group_size == 0, (
        f"Columns ({cols}) must be divisible by group_size ({group_size})"
    )

    max_val = (1 << bits) - 1  # 15 for 4-bit, 255 for 8-bit
    elems_per_uint32 = 32 // bits  # 8 for 4-bit, 4 for 8-bit

    w = weight.float().reshape(rows, cols // group_size, group_size)
    w_min = w.min(dim=-1).values
    w_max = w.max(dim=-1).values

    scales = (w_max - w_min) / float(max_val)
    biases = w_min
    scales = scales.clamp(min=1e-10)

    q = ((w - biases.unsqueeze(-1)) / scales.unsqueeze(-1))
    q = q.round().clamp(0, max_val).to(torch.uint8).reshape(rows, cols)

    assert cols % elems_per_uint32 == 0
    packed_cols = cols // elems_per_uint32
    packed = torch.zeros(rows, packed_cols, dtype=torch.int64)
    for i in range(elems_per_uint32):
        packed |= q[:, i::elems_per_uint32].to(torch.int64) << (bits * i)

    packed_np = packed.to(torch.int32).numpy().view(np.uint32)
    packed = torch.from_numpy(packed_np.copy())

    return packed, scales.to(torch.float16), biases.to(torch.float16)


def tensors_to_numpy(tensors: dict) -> dict:
    """Convert all torch tensors to numpy arrays for safetensors.numpy.save_file."""
    result = {}
    for key, tensor in tensors.items():
        if isinstance(tensor, torch.Tensor):
            if tensor.dtype == torch.bfloat16:
                result[key] = tensor.to(torch.float16).numpy()
            else:
                result[key] = tensor.numpy()
        else:
            result[key] = tensor
    return result


# ---------------------------------------------------------------------------
# Weight classification
# ---------------------------------------------------------------------------

TEXT_DECODER_QUANTIZE_SUFFIXES = {
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
}


def should_quantize(key: str) -> bool:
    if not key.startswith("thinker.model.layers."):
        return False
    if not key.endswith(".weight"):
        return False
    for suffix in TEXT_DECODER_QUANTIZE_SUFFIXES:
        if f".{suffix}.weight" in key:
            return True
    return False


def should_quantize_embedding(key: str) -> bool:
    return key == "thinker.model.embed_tokens.weight"


# ---------------------------------------------------------------------------
# MLX conversion
# ---------------------------------------------------------------------------

def convert_mlx(
    source_model: str = "Qwen/Qwen3-ForcedAligner-0.6B",
    output_dir: str = "./forced-aligner-mlx",
    bits: int = 4,
    group_size: int = 64,
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Downloading model from {source_model}...")
    model_dir = Path(snapshot_download(
        source_model,
        allow_patterns=["*.safetensors", "*.json", "vocab.json", "merges.txt"],
    ))

    print("Loading weights...")
    all_weights = {}
    for sf_file in sorted(model_dir.glob("*.safetensors")):
        print(f"  Loading {sf_file.name}")
        all_weights.update(load_file(str(sf_file)))

    print(f"Total weight tensors: {len(all_weights)}")

    output_tensors = {}
    quantized_count = 0
    float_count = 0

    for key, tensor in sorted(all_weights.items()):
        if bits == 0:
            # bf16: keep everything as float16
            output_tensors[key] = tensor
            float_count += 1
        elif should_quantize_embedding(key):
            print(f"  Quantizing embedding ({bits}-bit): {key} {list(tensor.shape)}")
            packed, scales, biases = quantize_nbit(tensor, bits, group_size)
            output_tensors[key] = packed
            output_tensors[key.replace(".weight", ".scales")] = scales
            output_tensors[key.replace(".weight", ".biases")] = biases
            quantized_count += 1
        elif should_quantize(key):
            print(f"  Quantizing ({bits}-bit): {key} {list(tensor.shape)}")
            packed, scales, biases = quantize_nbit(tensor, bits, group_size)
            output_tensors[key] = packed
            output_tensors[key.replace(".weight", ".scales")] = scales
            output_tensors[key.replace(".weight", ".biases")] = biases
            quantized_count += 1
        else:
            output_tensors[key] = tensor
            float_count += 1

    if bits > 0:
        print(f"Quantized {quantized_count} layers to {bits}-bit, kept {float_count} as float")
    else:
        print(f"Kept all {float_count} tensors as float16 (bf16 mode)")

    output_file = output_path / "model.safetensors"
    print(f"Saving to {output_file}...")
    save_file(tensors_to_numpy(output_tensors), str(output_file))

    file_size_mb = output_file.stat().st_size / (1024 * 1024)
    print(f"  Output size: {file_size_mb:.1f} MB")

    for fname in ["vocab.json", "merges.txt", "tokenizer_config.json", "config.json"]:
        src = model_dir / fname
        if src.exists():
            shutil.copy2(src, output_path / fname)
            print(f"  Copied {fname}")

    if bits > 0:
        quant_config = {
            "quantization": {
                "group_size": group_size,
                "bits": bits,
                "quantized_components": ["text_decoder"],
                "float_components": ["audio_encoder", "classify_head", "norms"],
            }
        }
    else:
        quant_config = {
            "quantization": {
                "bits": 0,
                "quantized_components": [],
                "float_components": ["audio_encoder", "text_decoder", "classify_head"],
            }
        }

    with open(output_path / "quantize_config.json", "w") as f:
        json.dump(quant_config, f, indent=2)

    print(f"\nMLX conversion complete! Output in: {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# CoreML conversion
# ---------------------------------------------------------------------------

def build_audio_encoder(config: dict, weights: dict):
    """Build a PyTorch audio encoder matching the Qwen3-ASR architecture."""
    import torch.nn as nn
    import math

    d_model = config["d_model"]
    num_heads = config["encoder_attention_heads"]
    ffn_dim = config["encoder_ffn_dim"]
    num_layers = config["encoder_layers"]
    num_mel_bins = config["num_mel_bins"]
    output_dim = config.get("output_dim", d_model)
    downsample_hidden = config.get("downsample_hidden_size", 480)

    class SelfAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(d_model, d_model)
            self.k_proj = nn.Linear(d_model, d_model)
            self.v_proj = nn.Linear(d_model, d_model)
            self.out_proj = nn.Linear(d_model, d_model)
            self.head_dim = d_model // num_heads
            self.num_heads = num_heads
            self.scale = 1.0 / math.sqrt(self.head_dim)

        def forward(self, x):
            B, T, _ = x.shape
            q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
            k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            out = (attn @ v).transpose(1, 2).reshape(B, T, d_model)
            return self.out_proj(out)

    class EncoderLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = SelfAttention()
            self.self_attn_layer_norm = nn.LayerNorm(d_model)
            self.fc1 = nn.Linear(d_model, ffn_dim)
            self.fc2 = nn.Linear(ffn_dim, d_model)
            self.final_layer_norm = nn.LayerNorm(d_model)

        def forward(self, x):
            residual = x
            x = self.self_attn_layer_norm(x)
            x = self.self_attn(x)
            x = residual + x
            residual = x
            x = self.final_layer_norm(x)
            x = nn.functional.gelu(self.fc1(x))
            x = self.fc2(x)
            x = residual + x
            return x

    class AudioEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            # 3 conv2d layers for mel feature downsampling
            self.conv2d1 = nn.Conv2d(1, downsample_hidden, kernel_size=3, stride=2, padding=1)
            self.conv2d2 = nn.Conv2d(downsample_hidden, downsample_hidden, kernel_size=3, stride=2, padding=1)
            self.conv2d3 = nn.Conv2d(downsample_hidden, downsample_hidden, kernel_size=3, stride=2, padding=1)
            self.conv_out = nn.Linear(downsample_hidden * (num_mel_bins // 8), d_model, bias=False)
            self.layers = nn.ModuleList([EncoderLayer() for _ in range(num_layers)])
            self.ln_post = nn.LayerNorm(d_model)
            self.proj1 = nn.Linear(d_model, output_dim)
            self.proj2 = nn.Linear(output_dim, output_dim)

        def forward(self, mel_features):
            # mel_features: [B, num_mel_bins, T]
            x = mel_features.unsqueeze(1)  # [B, 1, 128, T]
            x = nn.functional.gelu(self.conv2d1(x))
            x = nn.functional.gelu(self.conv2d2(x))
            x = nn.functional.gelu(self.conv2d3(x))
            # [B, downsample_hidden, 128//8, T//8] -> reshape
            B, C, F, T = x.shape
            x = x.permute(0, 3, 1, 2).reshape(B, T, C * F)  # [B, T, C*F]
            x = self.conv_out(x)  # [B, T, d_model]
            for layer in self.layers:
                x = layer(x)
            x = self.ln_post(x)
            x = self.proj1(x)
            x = nn.functional.gelu(x)
            x = self.proj2(x)
            return x

    encoder = AudioEncoder()

    # Load weights: keys are like "conv2d1.weight", "layers.0.self_attn.q_proj.weight", etc.
    state_dict = {}
    for key, tensor in weights.items():
        if key.startswith("audio_tower."):
            name = key[len("audio_tower."):]
            # Conv2d weights stay in PyTorch layout [outC, inC, kH, kW]
            state_dict[name] = tensor
    missing, unexpected = encoder.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  Warning: missing audio encoder keys: {missing[:5]}...")
    if unexpected:
        print(f"  Warning: unexpected audio encoder keys: {unexpected[:5]}...")
    return encoder


def build_text_decoder(config: dict, weights: dict):
    """Build a traceable PyTorch text decoder matching Qwen3 architecture."""
    import torch.nn as nn
    import math

    hidden_size = config["hidden_size"]
    num_heads = config["num_attention_heads"]
    num_kv_heads = config["num_key_value_heads"]
    head_dim = config.get("head_dim", hidden_size // num_heads)
    intermediate_size = config["intermediate_size"]
    num_layers = config["num_hidden_layers"]
    vocab_size = config["vocab_size"]
    rms_norm_eps = config.get("rms_norm_eps", 1e-6)
    rope_theta = config.get("rope_theta", 1000000.0)

    class RMSNorm(nn.Module):
        def __init__(self, dim, eps=1e-6):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(dim))
            self.eps = eps

        def forward(self, x):
            norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
            return (x.float() * norm).to(x.dtype) * self.weight

    class RotaryEmbedding(nn.Module):
        def __init__(self, dim, max_seq_len=4096, theta=10000.0):
            super().__init__()
            inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
            self.register_buffer("inv_freq", inv_freq)
            t = torch.arange(max_seq_len).float()
            freqs = torch.outer(t, inv_freq)
            self.register_buffer("cos_cached", freqs.cos())
            self.register_buffer("sin_cached", freqs.sin())

        def forward(self, seq_len):
            return self.cos_cached[:seq_len], self.sin_cached[:seq_len]

    def apply_rotary(x, cos, sin):
        # x: [B, heads, T, head_dim]
        half = x.shape[-1] // 2
        x1, x2 = x[..., :half], x[..., half:]
        cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, T, half]
        sin = sin.unsqueeze(0).unsqueeze(0)
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)

    class Attention(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
            self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
            self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
            self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
            self.q_norm = RMSNorm(head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(head_dim, eps=rms_norm_eps)

        def forward(self, x, cos, sin):
            B, T, _ = x.shape
            q = self.q_proj(x).view(B, T, num_heads, head_dim).transpose(1, 2)
            k = self.k_proj(x).view(B, T, num_kv_heads, head_dim).transpose(1, 2)
            v = self.v_proj(x).view(B, T, num_kv_heads, head_dim).transpose(1, 2)

            # Apply QK norms then RoPE
            q = self.q_norm(q)
            k = self.k_norm(k)
            q = apply_rotary(q, cos, sin)
            k = apply_rotary(k, cos, sin)

            # GQA: repeat KV heads
            if num_kv_heads < num_heads:
                repeats = num_heads // num_kv_heads
                k = k.repeat_interleave(repeats, dim=1)
                v = v.repeat_interleave(repeats, dim=1)

            # Causal attention
            scale = 1.0 / math.sqrt(head_dim)
            attn = (q @ k.transpose(-2, -1)) * scale
            # Causal mask
            causal = torch.triu(torch.full((T, T), float("-inf"), device=x.device), diagonal=1)
            attn = attn + causal
            attn = attn.softmax(dim=-1)
            out = (attn @ v).transpose(1, 2).reshape(B, T, num_heads * head_dim)
            return self.o_proj(out)

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
            self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
            self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

        def forward(self, x):
            return self.down_proj(nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))

    class DecoderLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = Attention()
            self.mlp = MLP()
            self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
            self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)

        def forward(self, x, cos, sin):
            residual = x
            x = self.input_layernorm(x)
            x = self.self_attn(x, cos, sin)
            x = residual + x
            residual = x
            x = self.post_attention_layernorm(x)
            x = self.mlp(x)
            x = residual + x
            return x

    class TextDecoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
            self.layers = nn.ModuleList([DecoderLayer() for _ in range(num_layers)])
            self.norm = RMSNorm(hidden_size, eps=rms_norm_eps)
            self.rotary = RotaryEmbedding(head_dim, max_seq_len=4096, theta=rope_theta)

        def forward(self, inputs_embeds):
            cos, sin = self.rotary(inputs_embeds.shape[1])
            x = inputs_embeds
            for layer in self.layers:
                x = layer(x, cos, sin)
            return self.norm(x)

    model = TextDecoder()

    # Load weights
    text_state = {}
    for key, tensor in weights.items():
        if key.startswith("model."):
            name = key[len("model."):]
            text_state[name] = tensor

    # Remove rotary keys (we compute them)
    text_state = {k: v for k, v in text_state.items() if "rotary" not in k}

    missing, unexpected = model.load_state_dict(text_state, strict=False)
    if missing:
        filtered_missing = [k for k in missing if "rotary" not in k]
        if filtered_missing:
            print(f"  Warning: missing text decoder keys: {filtered_missing[:5]}...")
    if unexpected:
        print(f"  Warning: unexpected text decoder keys: {unexpected[:5]}...")
    print(f"  Text decoder: {num_layers} layers, hidden={hidden_size}")
    return model


def convert_coreml(
    source_model: str = "Qwen/Qwen3-ForcedAligner-0.6B",
    output_dir: str = "./forced-aligner-coreml",
    coreml_bits: int = 4,
    compile: bool = False,
):
    import coremltools as ct

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    traced_dir = output_path / "_traced"
    traced_dir.mkdir(exist_ok=True)

    # ─── Phase 1: Load weights from safetensors and build model ───
    print("=" * 60)
    print("Phase 1: Loading weights and building model")
    print("=" * 60)

    print(f"Downloading/loading model from {source_model}...")
    model_dir = Path(snapshot_download(
        source_model,
        allow_patterns=["*.safetensors", "*.json", "vocab.json", "merges.txt"],
    ))

    # Load config
    with open(model_dir / "config.json") as f:
        full_config = json.load(f)

    thinker_config = full_config.get("thinker_config", full_config)
    audio_config = thinker_config["audio_config"]
    text_config = thinker_config["text_config"]
    classify_num = thinker_config.get("classify_num", 5000)
    hidden_size = text_config["hidden_size"]

    # Load all safetensors weights
    all_weights = {}
    for sf_file in sorted(model_dir.glob("*.safetensors")):
        print(f"  Loading {sf_file.name}")
        all_weights.update(load_file(str(sf_file)))

    # Strip "thinker." prefix
    weights = {}
    for key, tensor in all_weights.items():
        if key.startswith("thinker."):
            weights[key[len("thinker."):]] = tensor.float()
        else:
            weights[key] = tensor.float()

    # Copy tokenizer files
    for fname in ["vocab.json", "merges.txt", "tokenizer_config.json", "config.json"]:
        src = model_dir / fname
        if src.exists():
            shutil.copy2(src, output_path / fname)

    # ─── Build audio encoder ───
    print("Building audio encoder from weights...")
    audio_encoder = build_audio_encoder(audio_config, weights)
    audio_encoder.eval()

    # ─── Build text decoder ───
    print("Building text decoder...")
    text_model = build_text_decoder(text_config, weights)
    text_model.eval()

    # ─── Build classify head ───
    print("Building classify head...")
    lm_head = torch.nn.Linear(hidden_size, classify_num, bias=False)
    lm_weights = {}
    for key, tensor in weights.items():
        if key.startswith("lm_head."):
            lm_weights[key[len("lm_head."):]] = tensor
    lm_head.load_state_dict(lm_weights)
    lm_head.eval()

    del all_weights, weights
    gc.collect()

    # ─── Trace audio encoder ───
    print("Tracing audio encoder...")

    example_mel = torch.randn(1, 128, 100)
    with torch.no_grad():
        traced_enc = torch.jit.trace(audio_encoder, example_mel)
    traced_enc.save(str(traced_dir / "audio_encoder.pt"))

    with torch.no_grad():
        example_audio_embeds = audio_encoder(example_mel)
    print(f"  Audio encoder output shape: {example_audio_embeds.shape}")

    # ─── Trace text decoder + classify head ───
    print("Tracing text decoder + classify head...")

    class TextDecoderWrapper(torch.nn.Module):
        def __init__(self, text_model, lm_head):
            super().__init__()
            self.text_model = text_model
            self.lm_head = lm_head

        def forward(self, inputs_embeds):
            hidden = self.text_model(inputs_embeds)
            logits = self.lm_head(hidden)
            return logits

    dec_wrapper = TextDecoderWrapper(text_model, lm_head)
    example_embeds = torch.randn(1, 20, hidden_size)
    with torch.no_grad():
        traced_dec = torch.jit.trace(dec_wrapper, example_embeds)
    traced_dec.save(str(traced_dir / "text_decoder.pt"))
    print(f"  Text decoder traced with hidden_size={hidden_size}")

    # ─── Trace embedding layer ───
    print("Tracing embedding layer...")

    class EmbeddingWrapper(torch.nn.Module):
        def __init__(self, embed_tokens):
            super().__init__()
            self.embed_tokens = embed_tokens

        def forward(self, input_ids):
            return self.embed_tokens(input_ids)

    embed_wrapper = EmbeddingWrapper(text_model.embed_tokens)
    example_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    with torch.no_grad():
        traced_embed = torch.jit.trace(embed_wrapper, example_ids)
    traced_embed.save(str(traced_dir / "embedding.pt"))

    # Free PyTorch models
    del audio_encoder, text_model, lm_head
    del dec_wrapper, embed_wrapper
    del traced_enc, traced_dec, traced_embed
    gc.collect()

    # ─── Phase 2: Convert to CoreML ───
    print()
    print("=" * 60)
    print("Phase 2: Converting traced models to CoreML")
    print("=" * 60)

    from coremltools.optimize.coreml import (
        OpPalettizerConfig,
        OptimizationConfig,
        palettize_weights,
    )

    def palettize(mlmodel, nbits):
        op_config = OpPalettizerConfig(mode="kmeans", nbits=nbits)
        config = OptimizationConfig(global_config=op_config)
        return palettize_weights(mlmodel, config)

    # Audio encoder with EnumeratedShapes for variable mel lengths
    print("\nConverting audio encoder...")
    mel_shapes = [
        ct.Shape(shape=(1, 128, l))
        for l in [100, 200, 300, 400, 500, 750, 1000, 1500, 2000, 3000]
    ]
    traced = torch.jit.load(str(traced_dir / "audio_encoder.pt"))
    enc_ml = ct.convert(
        traced,
        inputs=[ct.TensorType(
            name="mel_features",
            shape=ct.EnumeratedShapes(shapes=mel_shapes),
            dtype=np.float32,
        )],
        outputs=[ct.TensorType(name="audio_embeddings")],
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS17,
        skip_model_load=True,
    )
    print(f"  Palettizing audio encoder to INT{coreml_bits}...")
    enc_ml = palettize(enc_ml, coreml_bits)
    enc_ml.save(str(output_path / "audio_encoder.mlpackage"))
    del enc_ml, traced
    gc.collect()
    print("  Audio encoder saved.")

    # Embedding layer
    print("\nConverting embedding layer...")
    traced = torch.jit.load(str(traced_dir / "embedding.pt"))
    embed_shapes = [
        ct.Shape(shape=(1, l))
        for l in [10, 20, 50, 100, 200, 500, 1000, 2000]
    ]
    embed_ml = ct.convert(
        traced,
        inputs=[ct.TensorType(
            name="input_ids",
            shape=ct.EnumeratedShapes(shapes=embed_shapes),
            dtype=np.int32,
        )],
        outputs=[ct.TensorType(name="embeddings")],
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS17,
        skip_model_load=True,
    )
    embed_ml.save(str(output_path / "embedding.mlpackage"))
    del embed_ml, traced
    gc.collect()
    print("  Embedding saved.")

    # Text decoder + classify head
    print("\nConverting text decoder + classify head...")
    traced = torch.jit.load(str(traced_dir / "text_decoder.pt"))
    embed_shapes = [
        ct.Shape(shape=(1, l, hidden_size))
        for l in [10, 20, 50, 100, 200, 500, 1000, 2000]
    ]
    dec_ml = ct.convert(
        traced,
        inputs=[ct.TensorType(
            name="inputs_embeds",
            shape=ct.EnumeratedShapes(shapes=embed_shapes),
            dtype=np.float32,
        )],
        outputs=[ct.TensorType(name="logits")],
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS17,
        skip_model_load=True,
    )
    print(f"  Palettizing text decoder to INT{coreml_bits}...")
    dec_ml = palettize(dec_ml, coreml_bits)
    dec_ml.save(str(output_path / "text_decoder.mlpackage"))
    del dec_ml, traced
    gc.collect()
    print("  Text decoder saved.")

    # Clean up traced models
    shutil.rmtree(traced_dir)

    # Save config
    config = {
        "format": "coreml",
        "palettization_bits": coreml_bits,
        "hidden_size": hidden_size,
        "classify_num": 5000,
        "timestamp_segment_time": 0.08,
    }
    with open(output_path / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Optionally compile
    if compile:
        print("\nCompiling CoreML models...")
        for name in ["audio_encoder", "embedding", "text_decoder"]:
            pkg_path = output_path / f"{name}.mlpackage"
            compiled_path = output_path / f"{name}.mlmodelc"
            if compiled_path.exists():
                shutil.rmtree(compiled_path)
            compiled_url = ct.utils.compile_model(str(pkg_path))
            shutil.move(str(compiled_url), str(compiled_path))
            shutil.rmtree(pkg_path)
            print(f"  Compiled {name}.mlmodelc")

    print(f"\nCoreML conversion complete! Output in: {output_path}")
    print("Files:")
    for f in sorted(output_path.iterdir()):
        size = (
            sum(ff.stat().st_size for ff in f.rglob("*") if ff.is_file())
            if f.is_dir()
            else f.stat().st_size
        )
        print(f"  {f.name}: {size / 1024 / 1024:.1f} MB")

    return output_path


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------

def upload(output_dir: str, repo_id: str):
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, exist_ok=True)
    api.upload_folder(
        folder_path=output_dir,
        repo_id=repo_id,
        commit_message=f"Upload Qwen3-ForcedAligner-0.6B for speech-swift",
    )
    print(f"Uploaded to: https://huggingface.co/{repo_id}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert Qwen3-ForcedAligner to MLX or CoreML format"
    )
    parser.add_argument("--source", default="Qwen/Qwen3-ForcedAligner-0.6B",
                        help="Source model ID on HuggingFace")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (auto-named if omitted)")
    parser.add_argument("--bits", type=int, default=4, choices=[0, 4, 8],
                        help="MLX quantization bits (0=bf16, 4=4bit, 8=8bit)")
    parser.add_argument("--group-size", type=int, default=64,
                        help="Quantization group size")
    parser.add_argument("--coreml", action="store_true",
                        help="Convert to CoreML instead of MLX")
    parser.add_argument("--coreml-bits", type=int, default=4, choices=[4, 8],
                        help="CoreML palettization bits")
    parser.add_argument("--compile", action="store_true",
                        help="Compile CoreML .mlpackage to .mlmodelc")
    parser.add_argument("--upload", action="store_true",
                        help="Upload to HuggingFace Hub")
    parser.add_argument("--repo-id", default=None,
                        help="HuggingFace repo ID for upload")
    args = parser.parse_args()

    if args.coreml:
        default_dir = f"./forced-aligner-coreml-int{args.coreml_bits}"
        output_path = convert_coreml(
            source_model=args.source,
            output_dir=args.output_dir or default_dir,
            coreml_bits=args.coreml_bits,
            compile=args.compile,
        )
    else:
        bits_name = {0: "bf16", 4: "4bit", 8: "8bit"}[args.bits]
        default_dir = f"./forced-aligner-{bits_name}"
        output_path = convert_mlx(
            source_model=args.source,
            output_dir=args.output_dir or default_dir,
            bits=args.bits,
            group_size=args.group_size,
        )

    if args.upload:
        if not args.repo_id:
            print("Error: --repo-id required for upload")
            exit(1)
        upload(str(output_path), args.repo_id)
