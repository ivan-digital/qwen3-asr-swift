#!/usr/bin/env python3
"""
Convert PersonaPlex 7B weights to quantized safetensors for Swift/MLX.

Downloads from nvidia/personaplex-7b-v1 (bf16) and quantizes to 4-bit or 8-bit:
  - temporal.safetensors   (quantized, ~3.5 GB 4-bit / ~6.5 GB 8-bit)
  - depformer.safetensors  (quantized, ~650 MB 4-bit / ~1.3 GB 8-bit)
  - embeddings.safetensors (BF16, ~500 MB)
  - mimi.safetensors       (from tokenizer file, ~385 MB)
  - voices/                (per-voice safetensors)
  - config.json

Usage:
  python scripts/convert_personaplex.py
  python scripts/convert_personaplex.py --bits 8
  python scripts/convert_personaplex.py --bits 4 --upload --repo-id aufklarer/PersonaPlex-7B-MLX-4bit
  python scripts/convert_personaplex.py --bits 8 --upload --repo-id aufklarer/PersonaPlex-7B-MLX-8bit

Requires:
  pip install torch safetensors huggingface_hub numpy
"""

import argparse
import json
import os
import shutil
import tarfile
from pathlib import Path

import torch
import numpy as np
from huggingface_hub import hf_hub_download
from safetensors.numpy import save_file


# ---------------------------------------------------------------------------
# 4-bit group quantization (MLX-compatible format)
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

    scales = (w_max - w_min) / float(max_val)
    biases = w_min
    scales = scales.clamp(min=1e-10)

    scales_expanded = scales.unsqueeze(-1)
    biases_expanded = biases.unsqueeze(-1)
    q = ((w - biases_expanded) / scales_expanded).round().clamp(0, max_val).to(torch.uint8)
    q = q.reshape(rows, cols)

    assert cols % elems_per_uint32 == 0, (
        f"Columns ({cols}) must be divisible by {elems_per_uint32} for {bits}-bit packing"
    )
    packed_cols = cols // elems_per_uint32
    packed = torch.zeros(rows, packed_cols, dtype=torch.int64)
    for i in range(elems_per_uint32):
        packed |= q[:, i::elems_per_uint32].to(torch.int64) << (bits * i)

    packed_np = packed.to(torch.int32).numpy().view(np.uint32)
    packed = torch.from_numpy(packed_np.copy())

    return packed, scales.to(torch.float16), biases.to(torch.float16)


def quantize_4bit(weight: torch.Tensor, group_size: int = 64):
    """Backward-compatible 4-bit quantization wrapper."""
    return quantize_nbit(weight, group_size=group_size, bits=4)


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

# Temporal transformer weight prefixes
TEMPORAL_PREFIXES = [
    "transformer.",
    "out_norm.",     # Top-level output norm belongs to temporal transformer
]

# Depformer weight prefixes
DEPFORMER_PREFIXES = [
    "depformer.",
    "depformer_in.",
    "depformer_norms.",
]

# Embedding / output head prefixes
EMBEDDING_PREFIXES = [
    "text_emb.",
    "emb.",
    "text_linear.",
    "linears.",
    "depformer_emb.",
    "depformer_text_emb.",
]

# Linear layers to quantize in temporal transformer
TEMPORAL_QUANTIZE_SUFFIXES = {
    "in_proj", "out_proj",       # Attention
    "linear_in", "linear_out",   # FFN (SwiGLU)
}

# Depformer linear layers to quantize (same suffixes as temporal, plus depformer_in)
DEPFORMER_QUANTIZE_SUFFIXES = {
    "in_proj", "out_proj",       # Attention (MultiLinear packed)
    "linear_in", "linear_out",   # FFN (per-step, packed later)
}



def classify_key(key: str):
    """Classify a model.safetensors key into temporal/depformer/embedding."""
    for prefix in TEMPORAL_PREFIXES:
        if key.startswith(prefix):
            return "temporal"
    for prefix in DEPFORMER_PREFIXES:
        if key.startswith(prefix):
            return "depformer"
    for prefix in EMBEDDING_PREFIXES:
        if key.startswith(prefix):
            return "embedding"
    return "unknown"


def should_quantize_temporal(key: str, tensor: torch.Tensor) -> bool:
    """Check if a temporal transformer key should be 4-bit quantized."""
    if tensor.ndim != 2:
        return False
    rows, cols = tensor.shape
    if cols % 64 != 0:
        return False

    # Standard submodule weight: e.g. "layers.0.self_attn.out_proj.weight"
    if key.endswith(".weight"):
        parts = key.rsplit(".", 2)
        if len(parts) >= 2 and parts[-2] in TEMPORAL_QUANTIZE_SUFFIXES:
            return True

    # Flat packed param: e.g. "layers.0.self_attn.in_proj_weight"
    if key.endswith("_weight"):
        stem = key.rsplit(".", 1)[-1].replace("_weight", "")
        if stem in TEMPORAL_QUANTIZE_SUFFIXES:
            return True

    return False


def should_quantize_depformer(key: str, tensor: torch.Tensor) -> bool:
    """Check if a depformer key should be 4-bit quantized.

    Quantizes:
    - Attention in_proj/out_proj (MultiLinear packed: [numSteps*outDim, inDim])
    - FFN linear_in/linear_out (per-step: [outDim, inDim])
    - depformer_in.{i}.weight (per-step input projection: [depDim, temporalDim])

    Does NOT quantize:
    - Embeddings (lookup tables)
    - Output heads (linears.{i}.weight — small)
    - Norms (1-D)
    """
    if tensor.ndim != 2:
        return False
    rows, cols = tensor.shape
    if cols % 64 != 0:
        return False

    # depformer_in.{i}.weight — per-step input projections
    if key.startswith("depformer_in.") and key.endswith(".weight"):
        return True

    # Standard submodule weight: e.g. "layers.0.self_attn.out_proj.weight"
    if key.endswith(".weight"):
        parts = key.rsplit(".", 2)
        if len(parts) >= 2 and parts[-2] in DEPFORMER_QUANTIZE_SUFFIXES:
            return True

    # Flat packed param: e.g. "layers.0.self_attn.in_proj_weight"
    if key.endswith("_weight"):
        stem = key.rsplit(".", 1)[-1].replace("_weight", "")
        if stem in DEPFORMER_QUANTIZE_SUFFIXES:
            return True

    return False


def remap_temporal_key(key: str) -> str:
    """Remap temporal transformer key to match Swift module hierarchy.

    transformer.layers.{i}.* -> layers.{i}.*
    transformer.out_norm.* -> out_norm.*
    """
    if key.startswith("transformer."):
        return key[len("transformer."):]
    return key


def remap_depformer_key(key: str) -> str:
    """Remap depformer key to match Swift module hierarchy.

    depformer.layers.{i}.* -> layers.{i}.*
    depformer_in.{i}.* -> depformer_in.{i}.*  (keep)
    depformer_norms.{i}.* -> depformer_norms.{i}.*  (keep)
    """
    if key.startswith("depformer."):
        return key[len("depformer."):]
    return key


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_files(model_id: str, cache_dir: str = None):
    """Download PersonaPlex files from HuggingFace."""
    files = {}
    needed = [
        ("model", "model.safetensors"),
        ("tokenizer", "tokenizer-e351c8d8-checkpoint125.safetensors"),
        ("spm", "tokenizer_spm_32k_3.model"),
        ("voices", "voices.tgz"),
    ]

    for name, filename in needed:
        print(f"  Downloading {filename}...")
        path = hf_hub_download(
            repo_id=model_id,
            filename=filename,
            cache_dir=cache_dir,
        )
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"    -> {path} ({size_mb:.1f} MB)")
        files[name] = path

    return files


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

def convert_temporal(state_dict: dict, quantize: bool, group_size: int = 64, bits: int = 4):
    """Convert and optionally quantize temporal transformer weights."""
    output = {}
    param_count = 0

    for key, tensor in sorted(state_dict.items()):
        new_key = remap_temporal_key(key)
        numel = tensor.numel()
        param_count += numel

        if quantize and should_quantize_temporal(new_key, tensor):
            packed, scales, biases = quantize_nbit(tensor, group_size, bits=bits)
            # Handle both "foo.weight" and "foo_weight" naming for quantized keys
            if new_key.endswith("_weight"):
                base = new_key[:-len("_weight")]
                output[new_key] = packed
                output[base + "_scales"] = scales
                output[base + "_biases"] = biases
            else:
                output[new_key] = packed
                output[new_key.replace(".weight", ".scales")] = scales
                output[new_key.replace(".weight", ".biases")] = biases
            print(f"  [Q{bits}] {key} -> {new_key} {list(packed.shape)} uint32")
        else:
            if tensor.dtype in (torch.float32, torch.float64):
                tensor = tensor.to(torch.bfloat16)
            output[new_key] = tensor
            print(f"  {key} -> {new_key} {list(tensor.shape)} {tensor.dtype}")

    return output, param_count


def convert_depformer(state_dict: dict, quantize: bool = True,
                      group_size: int = 64, bits: int = 4):
    """Convert and optionally quantize depformer weights."""
    output = {}
    param_count = 0

    for key, tensor in sorted(state_dict.items()):
        new_key = remap_depformer_key(key)
        numel = tensor.numel()
        param_count += numel

        if quantize and should_quantize_depformer(new_key, tensor):
            packed, scales, biases = quantize_nbit(tensor, group_size, bits=bits)
            # Handle both "foo.weight" and "foo_weight" naming for quantized keys
            if new_key.endswith("_weight"):
                base = new_key[:-len("_weight")]
                output[new_key] = packed
                output[base + "_scales"] = scales
                output[base + "_biases"] = biases
            else:
                output[new_key] = packed
                output[new_key.replace(".weight", ".scales")] = scales
                output[new_key.replace(".weight", ".biases")] = biases
            print(f"  [Q{bits}] {key} -> {new_key} {list(packed.shape)} uint32")
        else:
            if tensor.dtype in (torch.float32, torch.float64):
                tensor = tensor.to(torch.bfloat16)
            output[new_key] = tensor
            print(f"  {key} -> {new_key} {list(tensor.shape)} {tensor.dtype}")

    return output, param_count


def convert_embeddings(state_dict: dict):
    """Convert embedding and output head weights (kept in BF16)."""
    output = {}
    param_count = 0

    for key, tensor in sorted(state_dict.items()):
        numel = tensor.numel()
        param_count += numel

        if tensor.dtype in (torch.float32, torch.float64):
            tensor = tensor.to(torch.bfloat16)
        output[key] = tensor
        print(f"  {key} {list(tensor.shape)} {tensor.dtype}")

    return output, param_count


def extract_voices(voices_tgz: str, output_dir: Path):
    """Extract voices.tgz into individual safetensors files."""
    voices_dir = output_dir / "voices"
    voices_dir.mkdir(exist_ok=True)

    print(f"  Extracting {voices_tgz}...")
    with tarfile.open(voices_tgz, "r:gz") as tar:
        tar.extractall(path=str(voices_dir))

    # Convert any .pt files to .safetensors
    for pt_file in sorted(voices_dir.glob("**/*.pt")):
        voice_name = pt_file.stem
        print(f"  Converting voice: {voice_name}")
        data = torch.load(str(pt_file), map_location="cpu", weights_only=True)
        if isinstance(data, dict):
            out_tensors = {}
            for k, v in data.items():
                if isinstance(v, torch.Tensor):
                    if v.dtype == torch.bfloat16:
                        out_tensors[k] = v.to(torch.float16).numpy()
                    else:
                        out_tensors[k] = v.float().numpy()
            sf_path = voices_dir / f"{voice_name}.safetensors"
            save_file(out_tensors, str(sf_path))
            print(f"    -> {sf_path}")
        elif isinstance(data, torch.Tensor):
            out_tensors = {"embeddings": data.float().numpy()}
            sf_path = voices_dir / f"{voice_name}.safetensors"
            save_file(out_tensors, str(sf_path))
            print(f"    -> {sf_path}")

    # Clean up .pt files and any nested directories left from tar extraction
    for pt_file in sorted(voices_dir.glob("**/*.pt")):
        pt_file.unlink()
    for sub in sorted(voices_dir.iterdir(), reverse=True):
        if sub.is_dir():
            try:
                sub.rmdir()  # Only removes if empty
            except OSError:
                pass

    # List resulting voice files
    voice_files = sorted(voices_dir.glob("*.safetensors"))
    print(f"  {len(voice_files)} voice files extracted")
    return [f.stem for f in voice_files]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def make_config(voice_names: list):
    """PersonaPlex model configuration."""
    return {
        "model_type": "personaplex",
        "version": "personaplex-7b-v1",
        "base_model": "kyutai/moshiko-pytorch-bf16",

        "temporal": {
            "dim": 4096,
            "num_layers": 32,
            "num_heads": 32,
            "hidden_scale": 4.125,  # LLaMA-style: intermediate = dim * 2/3 * hidden_scale = 11264
            "n_q": 8,
            "card": 2048,
            "text_card": 32000,
            "context": 3000,
            "max_period": 10000,
        },

        "depformer": {
            "dim": 1024,
            "num_layers": 6,
            "num_heads": 16,
            "dim_feedforward": 2816,  # = dim * 2/3 * hidden_scale (LLaMA-style SwiGLU)
            "num_steps": 16,
            "card": 2048,
            "text_card": 32000,
            "context": 8,
            "weights_per_step": True,
            "multi_linear": True,
        },

        "mimi": {
            "sample_rate": 24000,
            "frame_rate": 12.5,
            "num_codebooks": 16,
            "codebook_size": 2048,
            "codebook_dim": 256,
            "dimension": 512,
            "seanet_ratios": [8, 6, 5, 4],
            "transformer_layers": 8,
        },

        "sampling": {
            "audio_temp": 0.8,
            "audio_top_k": 250,
            "text_temp": 0.7,
            "text_top_k": 25,
        },

        "delays": [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],

        "quantization": {
            "bits": 4,
            "group_size": 64,
            "quantized_components": ["temporal", "depformer"],
        },

        "voices": voice_names,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert PersonaPlex 7B to MLX-compatible 4-bit safetensors"
    )
    parser.add_argument(
        "--model-id", default="nvidia/personaplex-7b-v1",
        help="HuggingFace model ID",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Output directory (default: auto-generated from bits)",
    )
    parser.add_argument(
        "--no-quantize", action="store_true",
        help="Skip quantization of temporal transformer",
    )
    parser.add_argument(
        "--bits", type=int, default=4, choices=[4, 8],
        help="Quantization bits (4 or 8, default: 4)",
    )
    parser.add_argument(
        "--group-size", type=int, default=64,
        help="Group size for quantization",
    )
    parser.add_argument(
        "--cache-dir", default=None,
        help="HuggingFace cache directory",
    )
    parser.add_argument(
        "--upload", action="store_true",
        help="Upload converted model to HuggingFace",
    )
    parser.add_argument(
        "--repo-id", default=None,
        help="HuggingFace repo ID for upload (e.g. your-name/PersonaPlex-7B-MLX-4bit)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir or f"./personaplex-mlx-{args.bits}bit")
    output_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Download
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Downloading from {args.model_id}...")
    print(f"{'='*60}")
    files = download_files(args.model_id, args.cache_dir)

    # -----------------------------------------------------------------------
    # Copy Mimi tokenizer weights
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Copying Mimi codec weights...")
    print(f"{'='*60}")
    mimi_src = files["tokenizer"]
    mimi_dst = output_dir / "mimi.safetensors"
    if not mimi_dst.exists():
        shutil.copy2(mimi_src, str(mimi_dst))
    mimi_size = os.path.getsize(str(mimi_dst)) / (1024 * 1024)
    print(f"  Saved {mimi_dst} ({mimi_size:.1f} MB)")

    # Copy SentencePiece tokenizer
    spm_dst = output_dir / "tokenizer_spm_32k_3.model"
    if not spm_dst.exists():
        shutil.copy2(files["spm"], str(spm_dst))
    print(f"  Copied tokenizer_spm_32k_3.model")

    # -----------------------------------------------------------------------
    # Extract voices
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Extracting voice embeddings...")
    print(f"{'='*60}")
    voice_names = extract_voices(files["voices"], output_dir)

    # -----------------------------------------------------------------------
    # Load and split model weights
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Loading model.safetensors (this may take a while for 16.7 GB)...")
    print(f"{'='*60}")

    from safetensors import safe_open

    temporal_weights = {}
    depformer_weights = {}
    embedding_weights = {}
    unknown_weights = {}

    with safe_open(files["model"], framework="pt", device="cpu") as f:
        keys = list(f.keys())
        print(f"  {len(keys)} keys found")

        for i, key in enumerate(keys):
            if i % 100 == 0:
                print(f"  Processing key {i}/{len(keys)}...")
            tensor = f.get_tensor(key)
            category = classify_key(key)

            if category == "temporal":
                temporal_weights[key] = tensor
            elif category == "depformer":
                depformer_weights[key] = tensor
            elif category == "embedding":
                embedding_weights[key] = tensor
            else:
                unknown_weights[key] = tensor

    print(f"\n  Temporal: {len(temporal_weights)} keys")
    print(f"  Depformer: {len(depformer_weights)} keys")
    print(f"  Embeddings: {len(embedding_weights)} keys")
    if unknown_weights:
        print(f"  Unknown: {len(unknown_weights)} keys")
        for k in sorted(unknown_weights.keys()):
            print(f"    - {k} {list(unknown_weights[k].shape)}")

    # -----------------------------------------------------------------------
    # Convert temporal transformer
    # -----------------------------------------------------------------------
    quantize = not args.no_quantize
    bits = args.bits
    print(f"\n{'='*60}")
    print(f"Converting temporal transformer {'(' + str(bits) + '-bit)' if quantize else '(float)'}...")
    print(f"{'='*60}")
    temporal_out, temporal_params = convert_temporal(
        temporal_weights, quantize=quantize, group_size=args.group_size, bits=bits)
    temporal_path = output_dir / "temporal.safetensors"
    save_file(tensors_to_numpy(temporal_out), str(temporal_path))
    temporal_size = os.path.getsize(str(temporal_path)) / (1024 * 1024)
    print(f"\n  Saved {temporal_path} ({temporal_size:.1f} MB)")
    print(f"  Parameters: {temporal_params:,}")
    del temporal_weights, temporal_out

    # -----------------------------------------------------------------------
    # Convert depformer
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    if quantize:
        print(f"Converting depformer ({bits}-bit)...")
    else:
        print("Converting depformer (BF16)...")
    print(f"{'='*60}")
    depformer_out, depformer_params = convert_depformer(
        depformer_weights, quantize=quantize, group_size=args.group_size, bits=bits)
    depformer_path = output_dir / "depformer.safetensors"
    save_file(tensors_to_numpy(depformer_out), str(depformer_path))
    depformer_size = os.path.getsize(str(depformer_path)) / (1024 * 1024)
    print(f"\n  Saved {depformer_path} ({depformer_size:.1f} MB)")
    print(f"  Parameters: {depformer_params:,}")
    del depformer_weights, depformer_out

    # -----------------------------------------------------------------------
    # Convert embeddings
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Converting embeddings (BF16)...")
    print(f"{'='*60}")
    emb_out, emb_params = convert_embeddings(embedding_weights)
    emb_path = output_dir / "embeddings.safetensors"
    save_file(tensors_to_numpy(emb_out), str(emb_path))
    emb_size = os.path.getsize(str(emb_path)) / (1024 * 1024)
    print(f"\n  Saved {emb_path} ({emb_size:.1f} MB)")
    print(f"  Parameters: {emb_params:,}")
    del embedding_weights, emb_out

    # -----------------------------------------------------------------------
    # Config
    # -----------------------------------------------------------------------
    config = make_config(voice_names)
    if args.no_quantize:
        config.pop("quantization", None)
    elif args.bits != 4 or args.group_size != 64:
        config["quantization"]["bits"] = args.bits
        config["quantization"]["group_size"] = args.group_size
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"\n  Saved {config_path}")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    total_params = temporal_params + depformer_params + emb_params
    print(f"\n{'='*60}")
    print("Conversion complete!")
    print(f"{'='*60}")
    print(f"  Output directory: {output_dir}")
    print(f"  Temporal:    {temporal_size:8.1f} MB  ({temporal_params:>14,} params)")
    print(f"  Depformer:   {depformer_size:8.1f} MB  ({depformer_params:>14,} params)")
    print(f"  Embeddings:  {emb_size:8.1f} MB  ({emb_params:>14,} params)")
    print(f"  Mimi:        {mimi_size:8.1f} MB")
    total_size = temporal_size + depformer_size + emb_size + mimi_size
    print(f"  Total:       {total_size:8.1f} MB  ({total_params:>14,} params)")
    print(f"  Voices:      {len(voice_names)}")
    print()

    # -----------------------------------------------------------------------
    # Upload (optional)
    # -----------------------------------------------------------------------
    if args.upload:
        repo_id = args.repo_id
        if not repo_id:
            print("Error: --repo-id is required when using --upload")
            return

        print(f"\n{'='*60}")
        print(f"Uploading to {repo_id}...")
        print(f"{'='*60}")

        from huggingface_hub import HfApi
        api = HfApi()

        # Create repo if needed
        api.create_repo(repo_id=repo_id, exist_ok=True, repo_type="model")

        # Upload all files
        api.upload_folder(
            repo_id=repo_id,
            folder_path=str(output_dir),
            commit_message=f"Upload PersonaPlex 7B MLX {args.bits}-bit weights",
        )
        print(f"  Uploaded to https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
