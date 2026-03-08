#!/usr/bin/env python3
"""
Convert Qwen3-ASR bf16 safetensors to MLX-compatible format with N-bit quantization.

Downloads from official Qwen repos (bf16) and quantizes text decoder to 4-bit or 8-bit.
Audio encoder weights are kept as float16 (not quantized).

Supports both 0.6B and 1.7B model sizes, and 4-bit or 8-bit quantization.

Usage:
  python scripts/convert_qwen3_asr.py
  python scripts/convert_qwen3_asr.py --bits 8
  python scripts/convert_qwen3_asr.py --model-size 1.7b --bits 4
  python scripts/convert_qwen3_asr.py --model-size 1.7b --bits 8 --upload

Requires:
  pip install torch safetensors huggingface_hub numpy
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.numpy import save_file


# ---------------------------------------------------------------------------
# N-bit group quantization (MLX-compatible format)
# ---------------------------------------------------------------------------

def quantize_nbit(weight: torch.Tensor, group_size: int = 64, bits: int = 4):
    """Quantize a 2-D float weight to N-bit with per-group scales and biases.

    Returns (packed_uint32, scales, biases) matching MLX QuantizedLinear format.
    """
    assert bits in (4, 8), f"Only 4-bit and 8-bit quantization supported, got {bits}"
    assert weight.ndim == 2, f"Expected 2-D tensor, got {weight.ndim}-D"
    rows, cols = weight.shape
    assert cols % group_size == 0, (
        f"Columns ({cols}) must be divisible by group_size ({group_size})"
    )
    num_groups = cols // group_size
    max_val = (1 << bits) - 1
    elems_per_uint32 = 32 // bits

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
# Source model IDs (official Qwen bf16 repos)
# ---------------------------------------------------------------------------

SOURCE_MODELS = {
    "0.6b": "Qwen/Qwen3-ASR-0.6B",
    "1.7b": "Qwen/Qwen3-ASR-1.7B",
}

OUTPUT_REPOS = {
    ("0.6b", 4): "aufklarer/Qwen3-ASR-0.6B-MLX-4bit",
    ("0.6b", 8): "aufklarer/Qwen3-ASR-0.6B-MLX-8bit",
    ("1.7b", 4): "aufklarer/Qwen3-ASR-1.7B-MLX-4bit",
    ("1.7b", 8): "aufklarer/Qwen3-ASR-1.7B-MLX-8bit",
}

# Text decoder linear layer suffixes to quantize
QUANTIZE_SUFFIXES = {
    "q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight",
    "gate_proj.weight", "up_proj.weight", "down_proj.weight",
}


def is_text_decoder(key: str) -> bool:
    """Check if key belongs to text decoder.

    Keys may be prefixed with thinker. (e.g. thinker.model.layers.0.*)
    """
    return ".model." in key or key.startswith("model.")


def should_quantize(key: str) -> bool:
    """Check if a key should be quantized.

    Only text decoder linear layers and embeddings are quantized.
    Audio encoder weights are kept as float16.
    """
    if not is_text_decoder(key):
        return False
    for suffix in QUANTIZE_SUFFIXES:
        if key.endswith(suffix):
            return True
    # Embeddings are also quantized (PreQuantizedEmbedding)
    if key.endswith("embed_tokens.weight"):
        return True
    return False


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


def strip_prefix(key: str) -> str:
    """Strip thinker. prefix from keys (Qwen source uses thinker.model.* / thinker.audio_tower.*)."""
    if key.startswith("thinker."):
        return key[len("thinker."):]
    return key


def convert_weights(tensors: dict, bits: int, group_size: int = 64):
    """Quantize text decoder weights, keep audio encoder as float16."""
    output = {}
    quantized_count = 0
    skipped_quant = 0

    for key in sorted(tensors.keys()):
        tensor = tensors[key]
        out_key = strip_prefix(key)

        if should_quantize(key):
            tensor = tensor.float()

            if tensor.ndim != 2:
                print(f"  SKIP quantize {out_key}: not 2-D (shape {list(tensor.shape)})")
                output[out_key] = tensor.to(torch.float16)
                skipped_quant += 1
                continue

            if tensor.shape[1] % group_size != 0:
                print(f"  SKIP quantize {out_key}: cols {tensor.shape[1]} not divisible by {group_size}")
                output[out_key] = tensor.to(torch.float16)
                skipped_quant += 1
                continue

            packed, scales, biases = quantize_nbit(tensor, group_size, bits)
            output[out_key] = packed
            output[out_key.replace(".weight", ".scales")] = scales
            output[out_key.replace(".weight", ".biases")] = biases
            quantized_count += 1
        else:
            # Audio encoder and non-quantized layers -> float16
            if tensor.dtype in (torch.float32, torch.float64, torch.bfloat16):
                tensor = tensor.to(torch.float16)
            # Transpose conv weights: PyTorch [outC, inC, kH, kW] -> MLX [outC, kH, kW, inC]
            if out_key.startswith("audio_tower.") and out_key.endswith(".weight") and tensor.ndim == 4:
                tensor = tensor.permute(0, 2, 3, 1).contiguous()
            output[out_key] = tensor

    print(f"  Quantized {quantized_count} layers to {bits}-bit")
    if skipped_quant > 0:
        print(f"  Skipped {skipped_quant} layers (not suitable for quantization)")

    return output


def make_config(model_size: str, bits: int, group_size: int = 64) -> dict:
    """Generate config.json for the converted model."""
    if model_size == "0.6b":
        audio_encoder = {
            "hidden_dim": 768,
            "num_layers": 12,
            "num_heads": 12,
            "head_dim": 64,
            "ffn_dim": 3072,
            "num_mel_bins": 128,
        }
        text_decoder = {
            "hidden_size": 1024,
            "num_layers": 28,
            "num_heads": 16,
            "num_kv_heads": 8,
            "head_dim": 128,
            "intermediate_size": 3072,
            "vocab_size": 151936,
            "rope_theta": 1000000.0,
            "rms_norm_eps": 1e-6,
        }
    else:  # 1.7b
        audio_encoder = {
            "hidden_dim": 1024,
            "num_layers": 24,
            "num_heads": 16,
            "head_dim": 64,
            "ffn_dim": 4096,
            "num_mel_bins": 128,
        }
        text_decoder = {
            "hidden_size": 2048,
            "num_layers": 28,
            "num_heads": 16,
            "num_kv_heads": 8,
            "head_dim": 128,
            "intermediate_size": 6144,
            "vocab_size": 151936,
            "rope_theta": 1000000.0,
            "rms_norm_eps": 1e-6,
        }

    return {
        "model_type": "qwen3_asr",
        "model_size": model_size.upper(),
        "audio_encoder": audio_encoder,
        "text_decoder": text_decoder,
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
        description="Convert Qwen3-ASR to MLX-compatible safetensors with N-bit quantization"
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
        help="Output directory (default: auto-generated)",
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
    output_dir = Path(args.output_dir or f"./qwen3-asr-{args.model_size}-{args.bits}bit")
    output_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Download source model (bf16 from Qwen)
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Downloading {source_id}...")
    print(f"{'='*60}")
    src_dir = Path(snapshot_download(source_id))
    print(f"  Source: {src_dir}")

    # -----------------------------------------------------------------------
    # Load and quantize weights
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Loading weights and quantizing to {args.bits}-bit...")
    print(f"{'='*60}")
    tensors = load_safetensors(src_dir)
    print(f"  Loaded {len(tensors)} tensors")

    output_tensors = convert_weights(tensors, args.bits, args.group_size)
    print(f"  Output: {len(output_tensors)} tensors")

    # Save quantized weights
    model_path = output_dir / "model.safetensors"
    save_file(tensors_to_numpy(output_tensors), str(model_path))
    model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"  Saved {model_path} ({model_size_mb:.1f} MB)")

    del tensors, output_tensors

    # -----------------------------------------------------------------------
    # Copy tokenizer files
    # -----------------------------------------------------------------------
    for fname in ["vocab.json", "merges.txt", "tokenizer_config.json"]:
        src_file = src_dir / fname
        if src_file.exists():
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
            f"aufklarer/Qwen3-ASR-{args.model_size.upper()}-MLX-{args.bits}bit"
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
