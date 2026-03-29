#!/usr/bin/env python3
"""Convert Qwen3.5-0.8B to quantized MLX format for on-device chat.

Downloads the official FP16/BF16 weights from Qwen/Qwen3.5-0.8B, extracts the
text-only language model, quantizes to INT4 or INT8, and saves as MLX safetensors.

Usage:
    # INT4 (default, 404 MB)
    python scripts/convert_qwen35_chat_mlx.py --output /tmp/Qwen3.5-0.8B-Chat-MLX/int4

    # INT8 (763 MB)
    python scripts/convert_qwen35_chat_mlx.py --output /tmp/Qwen3.5-0.8B-Chat-MLX/int8 --bits 8

    # Upload both
    huggingface-cli upload aufklarer/Qwen3.5-0.8B-Chat-MLX /tmp/Qwen3.5-0.8B-Chat-MLX
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import torch
from huggingface_hub import hf_hub_download, snapshot_download


# ── Weight keys that should be quantized ──

QUANTIZE_SUFFIXES = {
    # DeltaNet projections
    "in_proj_qkv.weight", "in_proj_z.weight", "in_proj_a.weight", "in_proj_b.weight",
    "out_proj.weight",
    # GatedAttention projections
    "q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight",
    # MLP
    "gate_proj.weight", "up_proj.weight", "down_proj.weight",
}

# Keys that stay float (norms, scalars, conv1d, embeddings handled separately)
FLOAT_PATTERNS = {"layernorm", "norm.weight", "A_log", "dt_bias", "conv1d"}


def should_quantize(key: str) -> bool:
    """Check if a weight key should be quantized."""
    return any(key.endswith(s) for s in QUANTIZE_SUFFIXES)


# ── Main ──

def main():
    parser = argparse.ArgumentParser(description="Convert Qwen3.5-0.8B to quantized MLX")
    parser.add_argument("--hf-model", default="Qwen/Qwen3.5-0.8B",
                        help="Source HuggingFace model ID")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--bits", type=int, default=4, choices=[4, 8],
                        help="Quantization bits (4 or 8)")
    parser.add_argument("--group-size", type=int, default=64)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    bits = args.bits
    group_size = args.group_size

    print(f"Source: {args.hf_model}")
    print(f"Output: {output_dir}")
    print(f"Quantization: INT{bits}, group_size={group_size}")

    # Download weights
    print("\n[1/4] Downloading weights...")
    model_dir = snapshot_download(
        args.hf_model,
        allow_patterns=["*.safetensors", "*.json", "*.safetensors.index.json"],
    )
    print(f"  Downloaded to: {model_dir}")

    # Load all safetensors
    from safetensors.torch import load_file
    import glob
    all_weights = {}
    for f in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
        print(f"  Loading {os.path.basename(f)}...")
        all_weights.update(load_file(f))
    print(f"  Total tensors: {len(all_weights)}")

    # Strip prefix and extract text-only weights
    print("\n[2/4] Extracting text model weights...")
    text_weights = {}
    skipped = 0
    for key, value in all_weights.items():
        stripped = key
        for prefix in ["model.language_model.", "model."]:
            if stripped.startswith(prefix):
                stripped = stripped[len(prefix):]
                break

        if stripped.startswith("visual.") or stripped.startswith("vision_tower."):
            skipped += 1
            continue
        if stripped == "lm_head.weight":
            skipped += 1  # tied to embed_tokens
            continue

        text_weights[stripped] = value

    print(f"  Text model tensors: {len(text_weights)}")
    print(f"  Skipped (vision/lm_head): {skipped}")

    # Convert torch tensors to MLX arrays
    print(f"\n[3/4] Quantizing to INT{bits} using MLX...")
    mlx_weights = {}
    for key, w in text_weights.items():
        # Convert to MLX via numpy
        if w.dtype == torch.bfloat16:
            mlx_weights[key] = mx.array(w.to(torch.float16).numpy())
        else:
            mlx_weights[key] = mx.array(w.numpy())

    # Apply sanitization (norm +1, conv1d transpose)
    # Only specific norms need +1 (HF stores as value-1).
    # DeltaNet linear_attn.norm.weight and final norm.weight do NOT need +1.
    norm_suffixes = (
        ".input_layernorm.weight",
        ".post_attention_layernorm.weight",
        ".q_norm.weight",
        ".k_norm.weight",
    )
    for key in list(mlx_weights.keys()):
        w = mlx_weights[key]
        if w.ndim == 1 and any(key.endswith(s) for s in norm_suffixes):
            mlx_weights[key] = w.astype(mx.float32) + 1.0
        # Conv1d: PyTorch [C, 1, K] → MLX [C, K, 1]
        if "conv1d.weight" in key and w.ndim == 3 and w.shape[1] == 1:
            mlx_weights[key] = w.moveaxis(2, 1)

    # Use MLX native quantization (matches QuantizedLinear format exactly)
    output_tensors = {}
    quantized_count = 0
    float_count = 0

    for key in sorted(mlx_weights.keys()):
        w = mlx_weights[key]

        if key == "embed_tokens.weight":
            q, s, b = mx.quantize(w, group_size=group_size, bits=bits)
            output_tensors["embed_tokens.weight"] = q
            output_tensors["embed_tokens.scales"] = s
            output_tensors["embed_tokens.biases"] = b
            quantized_count += 1
            continue

        if should_quantize(key):
            base = key[:-len(".weight")]
            q, s, b = mx.quantize(w, group_size=group_size, bits=bits)
            output_tensors[f"{base}.weight"] = q
            output_tensors[f"{base}.scales"] = s
            output_tensors[f"{base}.biases"] = b
            quantized_count += 1
        else:
            output_tensors[key] = w.astype(mx.bfloat16) if w.ndim > 1 else w
            float_count += 1

    print(f"  Quantized: {quantized_count} layers")
    print(f"  Float: {float_count} tensors (norms, scalars, conv1d)")
    print(f"  Total output: {len(output_tensors)} tensors")

    # Save using MLX native format
    print(f"\n[4/4] Saving...")
    output_path = output_dir / "model.safetensors"
    mx.save_safetensors(str(output_path), output_tensors)
    file_size = os.path.getsize(output_path)
    print(f"  model.safetensors: {file_size / 1024 / 1024:.1f} MB")

    # Config
    hf_config_path = os.path.join(model_dir, "config.json")
    with open(hf_config_path) as f:
        hf_config = json.load(f)

    text_cfg = hf_config.get("text_config", hf_config)
    rope_params = text_cfg.get("rope_parameters", {})

    config = {
        "hidden_size": text_cfg.get("hidden_size", 1024),
        "num_hidden_layers": text_cfg.get("num_hidden_layers", 24),
        "num_attention_heads": text_cfg.get("num_attention_heads", 8),
        "num_key_value_heads": text_cfg.get("num_key_value_heads", 2),
        "head_dim": text_cfg.get("head_dim", 256),
        "intermediate_size": text_cfg.get("intermediate_size", 3584),
        "vocab_size": text_cfg.get("vocab_size", 248320),
        "max_seq_len": 2048,
        "rope_theta": rope_params.get("rope_theta", 10_000_000),
        "rms_norm_eps": text_cfg.get("rms_norm_eps", 1e-6),
        "eos_token_id": 248046,
        "pad_token_id": 248044,
        "quantization": f"int{bits}",
        "model_type": "qwen3_5_text",
        "layer_types": text_cfg.get("layer_types", []),
        "full_attention_interval": text_cfg.get("full_attention_interval", 4),
        "linear_num_key_heads": text_cfg.get("linear_num_key_heads", 16),
        "linear_key_head_dim": text_cfg.get("linear_key_head_dim", 128),
        "linear_num_value_heads": text_cfg.get("linear_num_value_heads", 16),
        "linear_value_head_dim": text_cfg.get("linear_value_head_dim", 128),
        "linear_conv_kernel_dim": text_cfg.get("linear_conv_kernel_dim", 4),
        "partial_rotary_factor": rope_params.get("partial_rotary_factor", 0.25),
        "tie_word_embeddings": hf_config.get("tie_word_embeddings", True),
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Tokenizer
    for filename in ["tokenizer.json", "tokenizer_config.json"]:
        src = os.path.join(model_dir, filename)
        if os.path.exists(src):
            shutil.copy2(src, output_dir / filename)
            print(f"  Copied {filename}")

    print(f"\nDone! INT{bits} model at: {output_dir}")
    print(f"  model.safetensors: {file_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
