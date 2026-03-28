#!/usr/bin/env python3
"""Repackage mlx-community/Qwen3.5-0.8B-4bit for text-only Qwen3Chat.

Downloads the pre-quantized VLM model from mlx-community, extracts only the
language model weights, strips the `language_model.model.` prefix, and saves
as a text-only package ready for upload to HuggingFace.

Usage:
    python scripts/repackage_qwen35_chat_mlx.py \
        --source mlx-community/Qwen3.5-0.8B-4bit \
        --output /tmp/Qwen3.5-0.8B-Chat-MLX-4bit
"""

import argparse
import json
import os
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files
import mlx.core as mx


def main():
    parser = argparse.ArgumentParser(description="Repackage Qwen3.5 MLX weights for text-only use")
    parser.add_argument("--source", default="mlx-community/Qwen3.5-0.8B-4bit",
                        help="Source HuggingFace repo")
    parser.add_argument("--output", default="/tmp/Qwen3.5-0.8B-Chat-MLX-4bit",
                        help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Source: {args.source}")
    print(f"Output: {output_dir}")

    # Download safetensors
    print("\nDownloading model weights...")
    safetensors_path = hf_hub_download(args.source, "model.safetensors")
    print(f"  Downloaded: {safetensors_path}")

    # Load weights using MLX (handles bfloat16)
    print("\nLoading weights...")
    weights = dict(mx.load(safetensors_path))
    print(f"  Total tensors: {len(weights)}")

    # Extract text-only weights, strip prefix
    text_weights = {}
    skipped = {"vision_tower": 0, "lm_head": 0, "other": 0}
    prefix = "language_model.model."

    for key, value in weights.items():
        if key.startswith(prefix):
            new_key = key[len(prefix):]
            text_weights[new_key] = value
        elif key.startswith("vision_tower."):
            skipped["vision_tower"] += 1
        elif key.startswith("lm_head."):
            skipped["lm_head"] += 1
        else:
            skipped["other"] += 1
            print(f"  Skipping unknown key: {key}")

    print(f"\n  Text model tensors: {len(text_weights)}")
    print(f"  Skipped vision_tower: {skipped['vision_tower']}")
    print(f"  Skipped lm_head: {skipped['lm_head']}")
    print(f"  Skipped other: {skipped['other']}")

    # Print some key samples
    print("\nSample keys:")
    for i, key in enumerate(sorted(text_weights.keys())):
        if i < 10 or "layers.3." in key:
            print(f"  {key}: {text_weights[key].shape} {text_weights[key].dtype}")

    # Save as safetensors using MLX
    output_path = output_dir / "model.safetensors"
    print(f"\nSaving to {output_path}...")
    mx.save_safetensors(str(output_path), text_weights)

    file_size = os.path.getsize(output_path)
    print(f"  Size: {file_size / 1024 / 1024:.1f} MB")

    # Download and adapt config
    print("\nCreating config.json...")
    source_config_path = hf_hub_download(args.source, "config.json")
    with open(source_config_path) as f:
        source_config = json.load(f)

    # Extract text config and flatten for our format
    text_config = source_config.get("text_config", {})
    rope_params = text_config.get("rope_parameters", {})

    config = {
        "hidden_size": text_config.get("hidden_size", 1024),
        "num_hidden_layers": text_config.get("num_hidden_layers", 24),
        "num_attention_heads": text_config.get("num_attention_heads", 8),
        "num_key_value_heads": text_config.get("num_key_value_heads", 2),
        "head_dim": text_config.get("head_dim", 256),
        "intermediate_size": text_config.get("intermediate_size", 3584),
        "vocab_size": text_config.get("vocab_size", 248320),
        "max_seq_len": 2048,  # Practical limit for on-device
        "rope_theta": rope_params.get("rope_theta", 10_000_000),
        "rms_norm_eps": text_config.get("rms_norm_eps", 1e-6),
        "eos_token_id": text_config.get("eos_token_id", 248044),
        "pad_token_id": 248043,
        "quantization": "int4",
        "model_type": "qwen3_5_text",
        "layer_types": text_config.get("layer_types", []),
        "full_attention_interval": text_config.get("full_attention_interval", 4),
        "linear_num_key_heads": text_config.get("linear_num_key_heads", 16),
        "linear_key_head_dim": text_config.get("linear_key_head_dim", 128),
        "linear_num_value_heads": text_config.get("linear_num_value_heads", 16),
        "linear_value_head_dim": text_config.get("linear_value_head_dim", 128),
        "linear_conv_kernel_dim": text_config.get("linear_conv_kernel_dim", 4),
        "partial_rotary_factor": rope_params.get("partial_rotary_factor", 0.25),
        "tie_word_embeddings": source_config.get("tie_word_embeddings", True),
    }

    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Written: {config_path}")

    # Copy tokenizer files
    print("\nCopying tokenizer files...")
    for filename in ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja"]:
        try:
            src = hf_hub_download(args.source, filename)
            dst = output_dir / filename
            shutil.copy2(src, dst)
            print(f"  Copied: {filename}")
        except Exception as e:
            print(f"  Skipped {filename}: {e}")

    print(f"\nDone! Output at: {output_dir}")
    print(f"\nTo upload to HuggingFace:")
    print(f"  huggingface-cli upload aufklarer/Qwen3.5-0.8B-Chat-MLX-4bit {output_dir}")


if __name__ == "__main__":
    main()
