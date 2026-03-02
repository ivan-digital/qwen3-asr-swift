#!/usr/bin/env python3
"""
Convert Qwen3-TTS safetensors to MLX-compatible format with N-bit quantization.

Downloads from mlx-community (already safetensors format) and re-quantizes:
  - model.safetensors  (talker. prefixed keys + code predictor)
  - config.json

Supports both 0.6B and 1.7B model sizes, and 4-bit or 8-bit quantization.

Usage:
  python scripts/convert_qwen3_tts.py
  python scripts/convert_qwen3_tts.py --bits 8
  python scripts/convert_qwen3_tts.py --model-size 1.7b --bits 4
  python scripts/convert_qwen3_tts.py --model-size 1.7b --bits 8 --upload

Requires:
  pip install torch safetensors huggingface_hub numpy
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download, snapshot_download
from safetensors import safe_open
from safetensors.numpy import save_file


# ---------------------------------------------------------------------------
# N-bit group quantization (MLX-compatible format)
# ---------------------------------------------------------------------------

def quantize_nbit(weight: torch.Tensor, group_size: int = 64, bits: int = 4):
    """Quantize a 2-D float weight to N-bit with per-group scales and biases.

    Supports bits=4 (packed 8 per uint32) and bits=8 (packed 4 per uint32).
    Returns (packed_uint32, scales, biases) matching MLX QuantizedLinear format.
    """
    assert bits in (4, 8), f"Only 4-bit and 8-bit quantization supported, got {bits}"
    assert weight.ndim == 2, f"Expected 2-D tensor, got {weight.ndim}-D"
    rows, cols = weight.shape
    assert cols % group_size == 0, (
        f"Columns ({cols}) must be divisible by group_size ({group_size})"
    )
    num_groups = cols // group_size
    max_val = (1 << bits) - 1  # 15 for 4-bit, 255 for 8-bit
    elems_per_uint32 = 32 // bits  # 8 for 4-bit, 4 for 8-bit

    w = weight.float().reshape(rows, num_groups, group_size)
    w_min = w.min(dim=-1).values
    w_max = w.max(dim=-1).values

    scales = (w_max - w_min) / max_val
    biases = w_min
    scales = scales.clamp(min=1e-10)

    scales_expanded = scales.unsqueeze(-1)
    biases_expanded = biases.unsqueeze(-1)
    q = ((w - biases_expanded) / scales_expanded).round().clamp(0, max_val).to(torch.uint8)
    q = q.reshape(rows, cols)

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
# Source model IDs
# ---------------------------------------------------------------------------

SOURCE_MODELS = {
    "0.6b": "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-4bit",
    "1.7b": "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-4bit",
}

OUTPUT_REPOS = {
    ("0.6b", 4): "aufklarer/Qwen3-TTS-12Hz-0.6B-Base-MLX-4bit",
    ("0.6b", 8): "aufklarer/Qwen3-TTS-12Hz-0.6B-Base-MLX-8bit",
    ("1.7b", 4): "aufklarer/Qwen3-TTS-12Hz-1.7B-Base-MLX-4bit",
    ("1.7b", 8): "aufklarer/Qwen3-TTS-12Hz-1.7B-Base-MLX-8bit",
}

# Suffixes that indicate a linear layer weight to quantize
QUANTIZE_SUFFIXES = {
    "q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight",
    "gate_proj.weight", "up_proj.weight", "down_proj.weight",
    "linear_fc1.weight", "linear_fc2.weight",
    "codec_head.weight",
}


def should_quantize(key: str) -> bool:
    """Check if a key should be quantized."""
    # Talker and code predictor linear layers
    for suffix in QUANTIZE_SUFFIXES:
        if key.endswith(suffix):
            return True
    # Code predictor LM heads: talker.code_predictor.lm_head.{i}.weight
    if "lm_head." in key and key.endswith(".weight"):
        return True
    return False


def is_already_quantized(key: str, all_keys: set) -> bool:
    """Check if a weight key has associated .scales/.biases (already quantized)."""
    base = key.replace(".weight", "")
    return f"{base}.scales" in all_keys or f"{base}.biases" in all_keys


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

def load_safetensors(directory: Path) -> dict:
    """Load all safetensors files from a directory into a flat dict."""
    tensors = {}
    for f in sorted(directory.glob("*.safetensors")):
        with safe_open(str(f), framework="pt") as sf:
            for key in sf.keys():
                tensors[key] = sf.get_tensor(key)
    return tensors


def convert_weights(tensors: dict, bits: int, group_size: int = 64):
    """Convert and re-quantize weights to the target bit width."""
    all_keys = set(tensors.keys())
    output = {}
    quantized_count = 0
    skipped_quant = 0

    for key in sorted(tensors.keys()):
        tensor = tensors[key]

        # Skip existing .scales/.biases from source quantization — we'll regenerate
        if key.endswith(".scales") or key.endswith(".biases"):
            # Check if we're going to re-quantize the parent
            parent_weight = key.rsplit(".", 1)[0] + ".weight"
            if parent_weight in all_keys and should_quantize(parent_weight):
                continue  # Will be regenerated
            # Keep non-quantizable scales/biases as-is
            output[key] = tensor
            continue

        if should_quantize(key):
            # If source is already packed uint32 (from 4-bit), dequantize first
            if tensor.dtype == torch.uint32 or (
                tensor.dtype == torch.int32 and is_already_quantized(key, all_keys)
            ):
                # Dequantize from source format
                base = key.replace(".weight", "")
                scales_key = f"{base}.scales"
                biases_key = f"{base}.biases"
                if scales_key in tensors and biases_key in tensors:
                    tensor = dequantize(
                        tensor, tensors[scales_key], tensors[biases_key],
                        group_size=group_size, bits=4  # Source is always 4-bit from mlx-community
                    )
                else:
                    print(f"  WARNING: {key} looks quantized but missing scales/biases, keeping as-is")
                    output[key] = tensor
                    skipped_quant += 1
                    continue

            # Ensure 2-D for quantization
            if tensor.ndim != 2:
                print(f"  SKIP quantize {key}: not 2-D (shape {list(tensor.shape)})")
                output[key] = tensor if tensor.dtype == torch.float16 else tensor.to(torch.float16)
                skipped_quant += 1
                continue

            # Check divisibility
            if tensor.shape[1] % group_size != 0:
                print(f"  SKIP quantize {key}: cols {tensor.shape[1]} not divisible by {group_size}")
                output[key] = tensor.to(torch.float16)
                skipped_quant += 1
                continue

            # Quantize to target bits
            packed, scales, biases = quantize_nbit(tensor, group_size, bits)
            output[key] = packed
            output[key.replace(".weight", ".scales")] = scales
            output[key.replace(".weight", ".biases")] = biases
            quantized_count += 1
        else:
            # Non-quantized: convert to float16
            if tensor.dtype in (torch.float32, torch.float64, torch.bfloat16):
                tensor = tensor.to(torch.float16)
            output[key] = tensor

    print(f"  Quantized {quantized_count} layers to {bits}-bit")
    if skipped_quant > 0:
        print(f"  Skipped {skipped_quant} layers (not suitable for quantization)")

    return output


def dequantize(packed: torch.Tensor, scales: torch.Tensor, biases: torch.Tensor,
               group_size: int = 64, bits: int = 4) -> torch.Tensor:
    """Dequantize a packed uint32 weight back to float for re-quantization."""
    max_val = (1 << bits) - 1
    elems_per_uint32 = 32 // bits
    rows = packed.shape[0]
    packed_cols = packed.shape[1]
    cols = packed_cols * elems_per_uint32

    # Unpack
    packed_long = packed.to(torch.int64)
    q = torch.zeros(rows, cols, dtype=torch.float32)
    for i in range(elems_per_uint32):
        q[:, i::elems_per_uint32] = ((packed_long >> (bits * i)) & max_val).float()

    # Dequantize: w = q * scale + bias
    num_groups = cols // group_size
    q = q.reshape(rows, num_groups, group_size)
    scales_expanded = scales.float().unsqueeze(-1)
    biases_expanded = biases.float().unsqueeze(-1)
    w = q * scales_expanded + biases_expanded
    return w.reshape(rows, cols)


def make_config(model_size: str, bits: int, group_size: int = 64) -> dict:
    """Generate config.json for the converted model."""
    if model_size == "0.6b":
        talker_config = {
            "hidden_size": 1024,
            "num_hidden_layers": 28,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "intermediate_size": 3072,
            "rope_theta": 1000000.0,
            "rms_norm_eps": 1e-6,
            "text_hidden_size": 2048,
        }
    else:  # 1.7b
        talker_config = {
            "hidden_size": 2048,
            "num_hidden_layers": 28,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "intermediate_size": 6144,
            "rope_theta": 1000000.0,
            "rms_norm_eps": 1e-6,
            "text_hidden_size": 2048,
        }

    return {
        "model_type": "qwen3_tts",
        "model_size": model_size.upper(),
        "talker_config": talker_config,
        "quantization_config": {
            "bits": bits,
            "group_size": group_size,
            "quant_method": "minmax",
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert Qwen3-TTS to MLX-compatible safetensors with N-bit quantization"
    )
    parser.add_argument(
        "--model-size",
        choices=["0.6b", "1.7b"],
        default="0.6b",
        help="Model size (default: 0.6b)",
    )
    parser.add_argument(
        "--bits",
        type=int,
        choices=[4, 8],
        default=4,
        help="Quantization bits (default: 4)",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=64,
        help="Group size for quantization (default: 64)",
    )
    parser.add_argument(
        "--source-model-id",
        default=None,
        help="Override source HuggingFace model ID",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: auto-generated from model size and bits)",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload to HuggingFace after conversion",
    )
    parser.add_argument(
        "--repo-id",
        default=None,
        help="HuggingFace repo ID for upload (default: auto-generated)",
    )
    args = parser.parse_args()

    source_id = args.source_model_id or SOURCE_MODELS[args.model_size]
    output_dir = Path(args.output_dir or f"./qwen3-tts-{args.model_size}-{args.bits}bit")
    output_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Download source model
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Downloading {source_id}...")
    print(f"{'='*60}")
    src_dir = Path(snapshot_download(source_id))
    print(f"  Source: {src_dir}")

    # -----------------------------------------------------------------------
    # Load and convert weights
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Converting to {args.bits}-bit (group_size={args.group_size})...")
    print(f"{'='*60}")
    tensors = load_safetensors(src_dir)
    print(f"  Loaded {len(tensors)} keys from source")

    output_tensors = convert_weights(tensors, args.bits, args.group_size)
    del tensors

    # Save
    model_path = output_dir / "model.safetensors"
    save_file(tensors_to_numpy(output_tensors), str(model_path))
    model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"\n  Saved {model_path} ({model_size_mb:.1f} MB, {len(output_tensors)} keys)")
    del output_tensors

    # -----------------------------------------------------------------------
    # Copy tokenizer files from source
    # -----------------------------------------------------------------------
    for fname in ["vocab.json", "merges.txt", "tokenizer_config.json"]:
        src_file = src_dir / fname
        if src_file.exists():
            import shutil
            shutil.copy2(src_file, output_dir / fname)
            print(f"  Copied {fname}")

    # -----------------------------------------------------------------------
    # Config
    # -----------------------------------------------------------------------
    config = make_config(args.model_size, args.bits, args.group_size)
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Saved {config_path}")

    # -----------------------------------------------------------------------
    # Upload
    # -----------------------------------------------------------------------
    if args.upload:
        from huggingface_hub import HfApi
        repo_id = args.repo_id or OUTPUT_REPOS.get(
            (args.model_size, args.bits),
            f"aufklarer/Qwen3-TTS-12Hz-{args.model_size.upper()}-Base-MLX-{args.bits}bit"
        )
        print(f"\n  Uploading to {repo_id}...")
        api = HfApi()
        api.create_repo(repo_id, exist_ok=True)
        api.upload_folder(folder_path=str(output_dir), repo_id=repo_id)
        print(f"  Uploaded to https://huggingface.co/{repo_id}")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Conversion complete!")
    print(f"{'='*60}")
    print(f"  Model size: {args.model_size.upper()}")
    print(f"  Quantization: {args.bits}-bit, group_size={args.group_size}")
    print(f"  Output: {output_dir}")
    print(f"  Model file: {model_size_mb:.1f} MB")


if __name__ == "__main__":
    main()
