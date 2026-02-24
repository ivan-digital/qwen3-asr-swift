#!/usr/bin/env python3
"""
Convert Qwen3-ForcedAligner-0.6B weights to 4-bit quantized safetensors for Swift/MLX.

Downloads from Qwen/Qwen3-ForcedAligner-0.6B and produces:
  - model.safetensors (audio encoder float, text decoder 4-bit, classify head float)
  - vocab.json, merges.txt, tokenizer_config.json (tokenizer files)
  - config.json

Usage:
  python scripts/convert_forced_aligner.py
  python scripts/convert_forced_aligner.py --output-dir ./forced-aligner-mlx
  python scripts/convert_forced_aligner.py --upload --repo-id your-name/Qwen3-ForcedAligner-0.6B-4bit
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import torch
import numpy as np
from huggingface_hub import hf_hub_download, snapshot_download
from safetensors.numpy import save_file
from safetensors.torch import load_file


# ---------------------------------------------------------------------------
# 4-bit group quantization (MLX-compatible format)
# ---------------------------------------------------------------------------

def quantize_4bit(weight: torch.Tensor, group_size: int = 64):
    """Quantize a 2-D float weight to 4-bit with per-group scales and biases.

    Returns (packed_uint32, scales, biases) matching MLX QuantizedLinear format.
    """
    assert weight.ndim == 2, f"Expected 2-D tensor, got {weight.ndim}-D"
    rows, cols = weight.shape
    assert cols % group_size == 0, (
        f"Columns ({cols}) must be divisible by group_size ({group_size})"
    )
    num_groups = cols // group_size
    max_val = 15  # 4-bit: 0-15
    elems_per_uint32 = 8  # 32 / 4

    w = weight.float().reshape(rows, num_groups, group_size)
    w_min = w.min(dim=-1).values
    w_max = w.max(dim=-1).values

    scales = (w_max - w_min) / float(max_val)
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
        packed |= q[:, i::elems_per_uint32].to(torch.int64) << (4 * i)

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

# Suffixes of linear layers in the text decoder to quantize
TEXT_DECODER_QUANTIZE_SUFFIXES = {
    "q_proj", "k_proj", "v_proj", "o_proj",   # Attention projections
    "gate_proj", "up_proj", "down_proj",        # SwiGLU MLP
}


def should_quantize(key: str) -> bool:
    """Check if a weight key should be 4-bit quantized."""
    # Only quantize text decoder linear layers
    if not key.startswith("thinker.model.layers."):
        return False
    # Only quantize .weight tensors (not biases, norms, etc.)
    if not key.endswith(".weight"):
        return False
    # Check if the layer name matches quantize targets
    for suffix in TEXT_DECODER_QUANTIZE_SUFFIXES:
        if f".{suffix}.weight" in key:
            return True
    return False


def should_quantize_embedding(key: str) -> bool:
    """Check if this is an embedding weight to quantize."""
    return key == "thinker.model.embed_tokens.weight"


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def convert(
    source_model: str = "Qwen/Qwen3-ForcedAligner-0.6B",
    output_dir: str = "./forced-aligner-4bit",
    group_size: int = 64,
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Downloading model from {source_model}...")
    model_dir = snapshot_download(
        source_model,
        allow_patterns=["*.safetensors", "*.json", "vocab.json", "merges.txt"],
    )
    model_dir = Path(model_dir)

    # Load all safetensors
    print("Loading weights...")
    all_weights = {}
    for sf_file in sorted(model_dir.glob("*.safetensors")):
        print(f"  Loading {sf_file.name}")
        weights = load_file(str(sf_file))
        all_weights.update(weights)

    print(f"Total weight tensors: {len(all_weights)}")

    # Process weights
    output_tensors = {}
    quantized_count = 0
    float_count = 0

    for key, tensor in sorted(all_weights.items()):
        if should_quantize_embedding(key):
            # Quantize embedding
            print(f"  Quantizing embedding: {key} {list(tensor.shape)}")
            packed, scales, biases = quantize_4bit(tensor, group_size)
            output_tensors[key] = packed
            output_tensors[key.replace(".weight", ".scales")] = scales
            output_tensors[key.replace(".weight", ".biases")] = biases
            quantized_count += 1

        elif should_quantize(key):
            # Quantize linear layer
            print(f"  Quantizing: {key} {list(tensor.shape)}")
            packed, scales, biases = quantize_4bit(tensor, group_size)
            output_tensors[key] = packed
            output_tensors[key.replace(".weight", ".scales")] = scales
            output_tensors[key.replace(".weight", ".biases")] = biases
            quantized_count += 1

        else:
            # Keep as float (audio encoder, classify head, norms, biases)
            output_tensors[key] = tensor
            float_count += 1

    print(f"Quantized {quantized_count} layers, kept {float_count} as float")

    # Save weights
    output_file = output_path / "model.safetensors"
    print(f"Saving to {output_file}...")
    save_file(tensors_to_numpy(output_tensors), str(output_file))

    file_size_mb = output_file.stat().st_size / (1024 * 1024)
    print(f"  Output size: {file_size_mb:.1f} MB")

    # Copy tokenizer files
    for fname in ["vocab.json", "merges.txt", "tokenizer_config.json", "config.json"]:
        src = model_dir / fname
        if src.exists():
            shutil.copy2(src, output_path / fname)
            print(f"  Copied {fname}")

    # Write quantization config
    quant_config = {
        "quantization": {
            "group_size": group_size,
            "bits": 4,
            "quantized_components": ["text_decoder"],
            "float_components": ["audio_encoder", "classify_head", "norms"],
        }
    }
    with open(output_path / "quantize_config.json", "w") as f:
        json.dump(quant_config, f, indent=2)

    print(f"\nConversion complete! Output in: {output_path}")
    return output_path


def upload(output_dir: str, repo_id: str):
    """Upload converted model to HuggingFace Hub."""
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, exist_ok=True)
    api.upload_folder(
        folder_path=output_dir,
        repo_id=repo_id,
        commit_message="Upload 4-bit quantized Qwen3-ForcedAligner-0.6B for MLX",
    )
    print(f"Uploaded to: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Qwen3-ForcedAligner to 4-bit MLX format")
    parser.add_argument("--source", default="Qwen/Qwen3-ForcedAligner-0.6B",
                        help="Source model ID on HuggingFace")
    parser.add_argument("--output-dir", default="./forced-aligner-4bit",
                        help="Output directory")
    parser.add_argument("--group-size", type=int, default=64,
                        help="Quantization group size")
    parser.add_argument("--upload", action="store_true",
                        help="Upload to HuggingFace Hub after conversion")
    parser.add_argument("--repo-id", default=None,
                        help="HuggingFace repo ID for upload")
    args = parser.parse_args()

    output_path = convert(
        source_model=args.source,
        output_dir=args.output_dir,
        group_size=args.group_size,
    )

    if args.upload:
        if not args.repo_id:
            print("Error: --repo-id required for upload")
            exit(1)
        upload(str(output_path), args.repo_id)
