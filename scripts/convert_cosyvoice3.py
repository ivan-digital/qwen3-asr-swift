#!/usr/bin/env python3
"""
Convert CosyVoice3 PyTorch weights to safetensors format for Swift/MLX.

Downloads llm.pt, flow.pt, hift.pt from HuggingFace and converts them to:
  - llm.safetensors   (4-bit quantized by default)
  - flow.safetensors
  - hifigan.safetensors
  - config.json

Usage:
  python scripts/convert_cosyvoice3.py
  python scripts/convert_cosyvoice3.py --output-dir ./my-output
  python scripts/convert_cosyvoice3.py --no-quantize
"""

import argparse
import json
import os
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
import numpy as np
from safetensors.numpy import save_file


# ---------------------------------------------------------------------------
# 4-bit group quantization (MLX-compatible format)
# ---------------------------------------------------------------------------

def quantize_4bit(weight: torch.Tensor, group_size: int = 64):
    """Quantize a 2-D float weight to 4-bit with per-group scales and biases.

    Returns (packed_uint32, scales, biases) matching MLX QuantizedLinear format:
      - packed: [rows, cols // 8] uint32   (8 x 4-bit values per uint32)
      - scales: [rows, cols // group_size] float16
      - biases: [rows, cols // group_size] float16

    The quantization maps the [min, max] range of each group to [0, 15].
    Values are packed little-endian: element i occupies bits [4*i, 4*i+3].
    """
    assert weight.ndim == 2, f"Expected 2-D tensor, got {weight.ndim}-D"
    rows, cols = weight.shape
    assert cols % group_size == 0, (
        f"Columns ({cols}) must be divisible by group_size ({group_size})"
    )
    num_groups = cols // group_size

    # Reshape to [rows, num_groups, group_size] for per-group stats
    w = weight.float().reshape(rows, num_groups, group_size)
    w_min = w.min(dim=-1).values  # [rows, num_groups]
    w_max = w.max(dim=-1).values  # [rows, num_groups]

    # scales = (max - min) / 15, biases = min
    scales = (w_max - w_min) / 15.0
    biases = w_min

    # Avoid division by zero for constant groups
    scales = scales.clamp(min=1e-10)

    # Quantize: q = round((w - bias) / scale), clamp to [0, 15]
    scales_expanded = scales.unsqueeze(-1)  # [rows, num_groups, 1]
    biases_expanded = biases.unsqueeze(-1)
    q = ((w - biases_expanded) / scales_expanded).round().clamp(0, 15).to(torch.uint8)

    # Flatten back to [rows, cols]
    q = q.reshape(rows, cols)

    # Pack 8 x 4-bit values into each uint32 (little-endian nibble order)
    # Use int64 for bitwise ops (PyTorch doesn't support shifts on uint32 CPU)
    assert cols % 8 == 0, f"Columns ({cols}) must be divisible by 8 for 4-bit packing"
    packed_cols = cols // 8
    packed = torch.zeros(rows, packed_cols, dtype=torch.int64)
    for i in range(8):
        packed |= q[:, i::8].to(torch.int64) << (4 * i)
    # Convert to uint32 via numpy (safetensors.torch doesn't support uint32,
    # but safetensors.numpy does)
    packed_np = packed.to(torch.int32).numpy().view(np.uint32)
    packed = torch.from_numpy(packed_np.copy())

    return packed, scales.to(torch.float16), biases.to(torch.float16)


def tensors_to_numpy(tensors: dict) -> dict:
    """Convert all torch tensors to numpy arrays for safetensors.numpy.save_file."""
    result = {}
    for key, tensor in tensors.items():
        if isinstance(tensor, torch.Tensor):
            # Handle bfloat16 -> float16 since numpy doesn't support bfloat16
            if tensor.dtype == torch.bfloat16:
                result[key] = tensor.to(torch.float16).numpy()
            else:
                result[key] = tensor.numpy()
        else:
            result[key] = tensor
    return result


# ---------------------------------------------------------------------------
# Key remapping helpers
# ---------------------------------------------------------------------------

LLM_PREFIX = "llm.model.model."
LLM_LM_HEAD = "llm.model.lm_head."

# Linear layers to quantize in the LLM
LLM_QUANTIZE_SUFFIXES = {
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
}


def remap_llm_key(key: str):
    """Remap a single LLM key. Returns (new_key, skip) tuple."""
    # Skip LM head (we use llm_decoder/speech_head instead)
    if key.startswith(LLM_LM_HEAD):
        return None, True

    # speech_embedding stays as-is
    if key.startswith("speech_embedding."):
        return key, False

    # llm_decoder -> speech_head
    if key.startswith("llm_decoder."):
        new_key = key.replace("llm_decoder.", "speech_head.", 1)
        return new_key, False

    # Strip llm.model.model. prefix for transformer weights
    if key.startswith(LLM_PREFIX):
        stripped = key[len(LLM_PREFIX):]
        # embed_tokens -> text_embedding
        if stripped.startswith("embed_tokens."):
            return stripped.replace("embed_tokens.", "text_embedding.", 1), False
        return stripped, False

    # Anything else: keep as-is with a warning
    return key, False


def should_quantize_llm_key(key: str) -> bool:
    """Check if a remapped LLM key should be 4-bit quantized."""
    # Only quantize weight matrices of linear layers
    if not key.endswith(".weight"):
        return False
    # Check if the second-to-last component is a quantizable layer
    parts = key.rsplit(".", 2)  # e.g. ["layers.0.self_attn", "q_proj", "weight"]
    if len(parts) >= 2:
        layer_name = parts[-2]
        if layer_name in LLM_QUANTIZE_SUFFIXES:
            return True
    # speech_head is also quantized
    if key == "speech_head.weight":
        return True
    return False


# Linear layers to quantize in the DiT (flow model)
# All target layers have cols divisible by 64.
# NOT quantized: projOut (1024x80, 80%64!=0), Conv1d layers, Embeddings.
DIT_QUANTIZE_SUFFIXES = {
    "to_q", "to_k", "to_v", "to_out",  # DiTAttention
    "linear1", "linear2",               # DiTFeedForward, TimestepEmbedding
}

DIT_QUANTIZE_PREFIXES = [
    "decoder.transformer_blocks.",  # 22 DiT layers
    "decoder.time_embed.",          # TimestepEmbedding
    "decoder.input_embed.proj",     # InputEmbedding projection
    "decoder.norm_out.linear",      # AdaLayerNormZeroFinal
]

# Keys explicitly excluded from DiT quantization
DIT_NO_QUANTIZE = {
    "decoder.proj_out.weight",  # 1024x80, 80 % 64 != 0
}


def should_quantize_dit_key(key: str, tensor: torch.Tensor) -> bool:
    """Check if a remapped flow key should be 4-bit quantized (DiT only)."""
    if not key.endswith(".weight"):
        return False
    if key in DIT_NO_QUANTIZE:
        return False
    if tensor.ndim != 2:
        return False  # Skip Conv1d (3-D) and bias (1-D)
    rows, cols = tensor.shape
    if cols % 64 != 0:
        return False  # Must be divisible by group_size for quantization

    # Check if the key matches a quantizable DiT prefix
    for prefix in DIT_QUANTIZE_PREFIXES:
        if key.startswith(prefix):
            return True

    return False


def remap_flow_key(key: str):
    """Remap a flow.pt key. Returns new_key or None to skip."""
    # Strip decoder.estimator. -> decoder.
    if key.startswith("decoder.estimator."):
        return key.replace("decoder.estimator.", "decoder.", 1)
    return key


def remap_hifigan_key(key: str):
    """Remap a hift.pt key. Returns new_key or None to skip."""
    # Many hift.pt files wrap everything under a prefix like 'generator.'
    # Strip it if present
    if key.startswith("generator."):
        return key[len("generator."):]
    return key


# ---------------------------------------------------------------------------
# Conv1d weight transposition
# ---------------------------------------------------------------------------

# Patterns that indicate Conv1d weights in the flow model
FLOW_CONV1D_PATTERNS = [
    "encoder.",  # Pre-lookahead encoder may have Conv1d
]

# Keys that should NOT be transposed even if they match patterns
FLOW_NO_TRANSPOSE = {
    # DiT uses Linear layers, not Conv1d
    "decoder.",
}


def is_flow_conv1d_weight(key: str, tensor: torch.Tensor) -> bool:
    """Check if a flow tensor is a Conv1d weight needing transpose."""
    if not key.endswith(".weight"):
        return False
    if tensor.ndim != 3:
        return False
    # All 3-D .weight tensors in the flow model are Conv1d
    # (Linear weights are 2-D, so ndim==3 is a sufficient check)
    return True


def is_hifigan_conv_weight(key: str, tensor: torch.Tensor) -> bool:
    """Check if a HiFi-GAN tensor is a Conv/ConvTranspose1d weight."""
    if not key.endswith(".weight"):
        return False
    if tensor.ndim != 3:
        return False
    # Skip non-conv weights (e.g. layer norm, linear)
    # Snake activation alpha/beta are 1-D, so ndim==3 check is sufficient
    return True


def is_hifigan_convtranspose_weight(key: str) -> bool:
    """Check if a HiFi-GAN key is a ConvTranspose1d weight.

    Note: HiFTGenerator 'ups' are regular Conv1d (verified by bias shape matching
    weight dim0, not ConvTranspose1d). ISTFT handles the upsampling.
    Currently no ConvTranspose1d layers exist in this model.
    """
    return False


def transpose_conv1d(tensor: torch.Tensor) -> torch.Tensor:
    """Transpose Conv1d weight [out, in, kernel] -> [out, kernel, in]."""
    assert tensor.ndim == 3, f"Expected 3-D tensor, got {tensor.ndim}-D"
    return tensor.permute(0, 2, 1).contiguous()


def transpose_convtranspose1d(tensor: torch.Tensor) -> torch.Tensor:
    """Transpose ConvTranspose1d weight [in, out, kernel] -> [out, kernel, in]."""
    assert tensor.ndim == 3, f"Expected 3-D tensor, got {tensor.ndim}-D"
    return tensor.permute(1, 2, 0).contiguous()


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def download_model_files(model_id: str, cache_dir: str = None):
    """Download the .pt files from HuggingFace. Returns dict of name->path."""
    files = {}
    # The files may be in a subdirectory like pretrained_models/Fun-CosyVoice3-0.5B/
    possible_paths = [
        # Direct in repo root
        {"llm": "llm.pt", "flow": "flow.pt", "hift": "hift.pt"},
        # In pretrained_models subdirectory
        {
            "llm": "pretrained_models/Fun-CosyVoice3-0.5B/llm.pt",
            "flow": "pretrained_models/Fun-CosyVoice3-0.5B/flow.pt",
            "hift": "pretrained_models/Fun-CosyVoice3-0.5B/hift.pt",
        },
        # In pretrained_models with 2512 suffix
        {
            "llm": "pretrained_models/Fun-CosyVoice3-0.5B-2512/llm.pt",
            "flow": "pretrained_models/Fun-CosyVoice3-0.5B-2512/flow.pt",
            "hift": "pretrained_models/Fun-CosyVoice3-0.5B-2512/hift.pt",
        },
    ]

    for path_set in possible_paths:
        try:
            for name, filename in path_set.items():
                print(f"  Trying {filename}...")
                path = hf_hub_download(
                    repo_id=model_id,
                    filename=filename,
                    cache_dir=cache_dir,
                )
                files[name] = path
            print(f"  Found all files with paths: {list(path_set.values())}")
            return files
        except Exception as e:
            print(f"  Not found at this path set: {e}")
            files.clear()
            continue

    raise RuntimeError(
        f"Could not find llm.pt, flow.pt, hift.pt in {model_id}. "
        "Check the repository structure."
    )


# ---------------------------------------------------------------------------
# Conversion functions
# ---------------------------------------------------------------------------

def convert_llm(state_dict: dict, quantize: bool, group_size: int = 64):
    """Convert LLM state dict to remapped (and optionally quantized) tensors."""
    output = {}
    skipped = []
    param_count = 0

    for key, tensor in sorted(state_dict.items()):
        new_key, skip = remap_llm_key(key)
        if skip or new_key is None:
            skipped.append(key)
            continue

        numel = tensor.numel()
        param_count += numel

        if quantize and should_quantize_llm_key(new_key):
            # 4-bit quantize this weight
            packed, scales, biases = quantize_4bit(tensor, group_size)
            base = new_key  # e.g. "layers.0.self_attn.q_proj.weight"
            output[base] = packed
            output[base.replace(".weight", ".scales")] = scales
            output[base.replace(".weight", ".biases")] = biases
            print(f"  [Q4] {key}")
            print(f"       -> {base} {list(packed.shape)} uint32")
            print(f"       -> {base.replace('.weight', '.scales')} {list(scales.shape)} float16")
            print(f"       -> {base.replace('.weight', '.biases')} {list(biases.shape)} float16")
        else:
            # Convert to bfloat16 for non-quantized weights
            if tensor.dtype in (torch.float32, torch.float64):
                tensor = tensor.to(torch.bfloat16)
            output[new_key] = tensor
            print(f"  {key}")
            print(f"       -> {new_key} {list(tensor.shape)} {tensor.dtype}")

    if skipped:
        print(f"\n  Skipped {len(skipped)} keys:")
        for k in skipped:
            print(f"    - {k}")

    return output, param_count


def convert_flow(state_dict: dict, quantize_dit: bool = False, group_size: int = 64):
    """Convert flow state dict with key remapping, Conv1d transposition, and optional DiT quantization."""
    output = {}
    param_count = 0
    quantized_count = 0

    for key, tensor in sorted(state_dict.items()):
        new_key = remap_flow_key(key)
        if new_key is None:
            continue

        numel = tensor.numel()
        param_count += numel

        # Check if this is a Conv1d weight needing transpose
        if is_flow_conv1d_weight(new_key, tensor):
            original_shape = list(tensor.shape)
            tensor = transpose_conv1d(tensor)
            if tensor.dtype in (torch.float32, torch.float64):
                tensor = tensor.to(torch.bfloat16)
            output[new_key] = tensor
            print(f"  {key}")
            print(f"       -> {new_key} {original_shape} -> {list(tensor.shape)} (Conv1d transposed)")
        elif quantize_dit and should_quantize_dit_key(new_key, tensor):
            # 4-bit quantize this DiT weight
            packed, scales, biases = quantize_4bit(tensor, group_size)
            base = new_key
            output[base] = packed
            output[base.replace(".weight", ".scales")] = scales
            output[base.replace(".weight", ".biases")] = biases
            quantized_count += 1
            print(f"  [Q4] {key}")
            print(f"       -> {base} {list(packed.shape)} uint32")
            print(f"       -> {base.replace('.weight', '.scales')} {list(scales.shape)} float16")
            print(f"       -> {base.replace('.weight', '.biases')} {list(biases.shape)} float16")
        else:
            if tensor.dtype in (torch.float32, torch.float64):
                tensor = tensor.to(torch.bfloat16)
            output[new_key] = tensor
            print(f"  {key}")
            print(f"       -> {new_key} {list(tensor.shape)} {tensor.dtype}")

    if quantize_dit:
        print(f"\n  Quantized {quantized_count} DiT Linear layers to 4-bit")

    return output, param_count


def fold_weight_norm(state_dict: dict) -> dict:
    """Fold PyTorch weight normalization parametrizations into plain .weight tensors.

    Converts: *.parametrizations.weight.original0 (g) + original1 (v) -> *.weight
    Formula: weight = g * v / ||v||
    where g has shape [out, 1, 1] and v has the full weight shape.
    """
    # Collect parametrized layer paths
    g_keys = {}  # path -> g tensor
    v_keys = {}  # path -> v tensor
    other = {}   # non-parametrized keys

    for key, tensor in state_dict.items():
        if ".parametrizations.weight.original0" in key:
            # Extract the layer path: "conv_pre.parametrizations.weight.original0" -> "conv_pre"
            path = key.replace(".parametrizations.weight.original0", "")
            g_keys[path] = tensor
        elif ".parametrizations.weight.original1" in key:
            path = key.replace(".parametrizations.weight.original1", "")
            v_keys[path] = tensor
        else:
            other[key] = tensor

    # Fold weight norm
    result = dict(other)
    for path in sorted(g_keys.keys()):
        if path not in v_keys:
            print(f"  WARNING: found g but not v for {path}, keeping as-is")
            result[f"{path}.parametrizations.weight.original0"] = g_keys[path]
            continue
        g = g_keys[path].float()  # [out, 1, 1]
        v = v_keys[path].float()  # [out, in, kernel]
        # weight = g * v / ||v|| (norm over all dims except first)
        dims = tuple(range(1, v.ndim))
        v_norm = torch.norm(v, dim=dims, keepdim=True)
        weight = g * v / (v_norm + 1e-12)
        result[f"{path}.weight"] = weight
        print(f"  Folded weight_norm: {path} g{list(g.shape)} + v{list(v.shape)} -> weight{list(weight.shape)}")

    # Check for orphan v keys
    for path in v_keys:
        if path not in g_keys:
            print(f"  WARNING: found v but not g for {path}, keeping as-is")
            result[f"{path}.parametrizations.weight.original1"] = v_keys[path]

    return result


def convert_hifigan(state_dict: dict):
    """Convert HiFi-GAN state dict with weight norm folding, key remapping, and Conv transposition."""
    # Step 1: Remap keys first
    remapped = {}
    for key, tensor in state_dict.items():
        new_key = remap_hifigan_key(key)
        if new_key is not None:
            remapped[new_key] = tensor

    # Step 2: Fold weight normalization parametrizations
    print("  --- Folding weight normalization ---")
    folded = fold_weight_norm(remapped)

    # Step 3: Transpose conv weights and count params
    output = {}
    param_count = 0

    for key, tensor in sorted(folded.items()):
        numel = tensor.numel()
        param_count += numel

        # Transpose conv weights
        if is_hifigan_conv_weight(key, tensor):
            original_shape = list(tensor.shape)
            if is_hifigan_convtranspose_weight(key):
                # ConvTranspose1d: [in, out, kernel] -> [out, kernel, in]
                tensor = transpose_convtranspose1d(tensor)
                label = "ConvTranspose1d transposed"
            else:
                # Conv1d: [out, in, kernel] -> [out, kernel, in]
                tensor = transpose_conv1d(tensor)
                label = "Conv1d transposed"

            if tensor.dtype in (torch.float32, torch.float64):
                tensor = tensor.to(torch.float32)  # HiFi-GAN stays float32
            output[key] = tensor
            print(f"  {key} {original_shape} -> {list(tensor.shape)} ({label})")
        else:
            if tensor.dtype in (torch.float32, torch.float64):
                tensor = tensor.to(torch.float32)  # HiFi-GAN stays float32
            output[key] = tensor
            print(f"  {key} {list(tensor.shape)} {tensor.dtype}")

    return output, param_count


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def make_config():
    """CosyVoice3 model configuration derived from cosyvoice3.yaml."""
    return {
        "model_type": "cosyvoice3",
        "version": "Fun-CosyVoice3-0.5B-2512",

        # --- LLM (Qwen2.5-0.5B backbone) ---
        "llm": {
            "hidden_size": 896,
            "num_hidden_layers": 24,
            "num_attention_heads": 14,
            "num_key_value_heads": 2,
            "intermediate_size": 4864,
            "head_dim": 64,
            "max_position_embeddings": 32768,
            "vocab_size": 151936,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1000000.0,
            "tie_word_embeddings": True,
            "speech_token_size": 6561,
            "text_token_size": 151936,
        },

        # --- Flow matching (DiT) ---
        "flow": {
            "input_size": 512,
            "output_size": 80,
            "vocab_size": 6561,
            "spk_embed_dim": 192,
            "token_frame_rate": 25,
            "token_mel_ratio": 2,
            "pre_lookahead_len": 3,
            "dit": {
                "dim": 1024,
                "depth": 22,
                "heads": 16,
                "dim_head": 64,
                "ff_mult": 2,
                "mel_dim": 80,
                "spk_dim": 80,
                "static_chunk_size": 50,
            },
        },

        # --- HiFi-GAN vocoder ---
        "hifigan": {
            "sampling_rate": 24000,
            "in_channels": 80,
            "base_channels": 512,
            "nb_harmonics": 8,
            "upsample_rates": [8, 5, 3],
            "upsample_kernel_sizes": [16, 11, 7],
            "istft_n_fft": 16,
            "istft_hop_len": 4,
            "resblock_kernel_sizes": [3, 7, 11],
            "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            "source_resblock_kernel_sizes": [7, 7, 11],
            "nsf_alpha": 0.1,
            "nsf_sigma": 0.003,
            "nsf_voiced_threshold": 10,
            "audio_limit": 0.99,
        },

        # --- Mel spectrogram ---
        "mel": {
            "n_fft": 1920,
            "num_mels": 80,
            "hop_size": 480,
            "win_size": 1920,
            "sample_rate": 24000,
        },

        # --- Tokenizer ---
        "tokenizer": {
            "type": "fsq",
            "codebook_size": 6561,
            "frame_rate": 25,
        },

        # --- Quantization (LLM only) ---
        "quantization": {
            "bits": 4,
            "group_size": 64,
            "quantized_layers": [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
                "speech_head",
            ],
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert CosyVoice3 PyTorch weights to MLX-compatible safetensors"
    )
    parser.add_argument(
        "--model-id",
        default="FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
        help="HuggingFace model ID (default: FunAudioLLM/Fun-CosyVoice3-0.5B-2512)",
    )
    parser.add_argument(
        "--output-dir",
        default="./cosyvoice3-mlx-4bit",
        help="Output directory (default: ./cosyvoice3-mlx-4bit)",
    )
    parser.add_argument(
        "--no-quantize",
        action="store_true",
        help="Skip 4-bit quantization of LLM weights",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=64,
        help="Group size for 4-bit quantization (default: 64)",
    )
    parser.add_argument(
        "--quantize-dit",
        action="store_true",
        help="4-bit quantize DiT Linear layers in flow model (experimental)",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="HuggingFace cache directory (default: system default)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Download
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Downloading from {args.model_id}...")
    print(f"{'='*60}")
    files = download_model_files(args.model_id, args.cache_dir)
    for name, path in files.items():
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"  {name}: {path} ({size_mb:.1f} MB)")

    # -----------------------------------------------------------------------
    # Convert LLM
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Converting LLM {'(4-bit quantized)' if not args.no_quantize else '(float)'}...")
    print(f"{'='*60}")
    llm_sd = torch.load(files["llm"], map_location="cpu", weights_only=True)
    print(f"  Loaded {len(llm_sd)} keys from llm.pt\n")
    llm_tensors, llm_params = convert_llm(
        llm_sd,
        quantize=not args.no_quantize,
        group_size=args.group_size,
    )
    llm_path = output_dir / "llm.safetensors"
    save_file(tensors_to_numpy(llm_tensors), str(llm_path))
    llm_size = os.path.getsize(llm_path) / (1024 * 1024)
    print(f"\n  Saved {llm_path} ({llm_size:.1f} MB)")
    print(f"  Total parameters: {llm_params:,}")
    print(f"  Output keys: {len(llm_tensors)}")
    del llm_sd, llm_tensors

    # -----------------------------------------------------------------------
    # Convert Flow
    # -----------------------------------------------------------------------
    dit_label = " (4-bit DiT)" if args.quantize_dit else ""
    print(f"\n{'='*60}")
    print(f"Converting Flow (DiT + encoder){dit_label}...")
    print(f"{'='*60}")
    flow_sd = torch.load(files["flow"], map_location="cpu", weights_only=True)
    print(f"  Loaded {len(flow_sd)} keys from flow.pt\n")
    flow_tensors, flow_params = convert_flow(
        flow_sd,
        quantize_dit=args.quantize_dit,
        group_size=args.group_size,
    )
    flow_path = output_dir / "flow.safetensors"
    save_file(tensors_to_numpy(flow_tensors), str(flow_path))
    flow_size = os.path.getsize(flow_path) / (1024 * 1024)
    print(f"\n  Saved {flow_path} ({flow_size:.1f} MB)")
    print(f"  Total parameters: {flow_params:,}")
    print(f"  Output keys: {len(flow_tensors)}")
    del flow_sd, flow_tensors

    # -----------------------------------------------------------------------
    # Convert HiFi-GAN
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Converting HiFi-GAN...")
    print(f"{'='*60}")
    hift_sd = torch.load(files["hift"], map_location="cpu", weights_only=True)
    print(f"  Loaded {len(hift_sd)} keys from hift.pt\n")
    hifigan_tensors, hifigan_params = convert_hifigan(hift_sd)
    hifigan_path = output_dir / "hifigan.safetensors"
    save_file(tensors_to_numpy(hifigan_tensors), str(hifigan_path))
    hifigan_size = os.path.getsize(hifigan_path) / (1024 * 1024)
    print(f"\n  Saved {hifigan_path} ({hifigan_size:.1f} MB)")
    print(f"  Total parameters: {hifigan_params:,}")
    print(f"  Output keys: {len(hifigan_tensors)}")
    del hift_sd, hifigan_tensors

    # -----------------------------------------------------------------------
    # Config
    # -----------------------------------------------------------------------
    config = make_config()
    if args.no_quantize:
        config.pop("quantization", None)
    if args.quantize_dit:
        config["dit_quantization"] = {
            "bits": 4,
            "group_size": args.group_size,
            "quantized_prefixes": [
                "decoder.transformer_blocks.",
                "decoder.time_embed.",
                "decoder.input_embed.proj",
                "decoder.norm_out.linear",
            ],
            "excluded_keys": ["decoder.proj_out.weight"],
        }
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"\n  Saved {config_path}")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    total_params = llm_params + flow_params + hifigan_params
    print(f"\n{'='*60}")
    print("Conversion complete!")
    print(f"{'='*60}")
    print(f"  Output directory: {output_dir}")
    print(f"  LLM:      {llm_size:7.1f} MB  ({llm_params:>12,} params)")
    print(f"  Flow:     {flow_size:7.1f} MB  ({flow_params:>12,} params)")
    print(f"  HiFi-GAN: {hifigan_size:7.1f} MB  ({hifigan_params:>12,} params)")
    print(f"  Total:    {llm_size + flow_size + hifigan_size:7.1f} MB  ({total_params:>12,} params)")
    print()


if __name__ == "__main__":
    main()
