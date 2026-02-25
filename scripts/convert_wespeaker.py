#!/usr/bin/env python3
"""
Convert pyannote/wespeaker-voxceleb-resnet34-LM to safetensors for Swift/MLX.

ResNet34 speaker embedding model (~6.6M params, 256-dim output, ~25 MB).
Key transformations:
  - Fuse BatchNorm into Conv2d: w_fused = w * γ/√(σ²+ε), b_fused = β - μ·γ/√(σ²+ε)
  - Transpose Conv2d: [O, I, H, W] → [O, H, W, I] (PyTorch to MLX channels-last)
  - Rename: strip "resnet." prefix, "seg_1" → "embedding"
  - Drop num_batches_tracked keys

Loads pytorch_model.bin with a custom unpickler — does NOT require pyannote.audio or wespeaker.

Requires: pip install torch safetensors numpy huggingface_hub

Usage:
  python scripts/convert_wespeaker.py --token YOUR_HF_TOKEN
  python scripts/convert_wespeaker.py --output-dir ./wespeaker-mlx
  python scripts/convert_wespeaker.py --upload --repo-id aufklarer/WeSpeaker-ResNet34-LM-MLX
"""

import argparse
import io
import json
import pickle
import zipfile
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors.numpy import save_file


# ── Custom unpickler to load pyannote checkpoints without pyannote installed ──

class _StubModule(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

class _StubObject:
    def __init__(self, *args, **kwargs):
        pass

class _WeSpeakerUnpickler(pickle.Unpickler):
    """Custom unpickler that stubs out pyannote/wespeaker classes."""
    def find_class(self, module, name):
        # Stub out any pyannote or wespeaker classes
        if module.startswith(("pyannote", "wespeaker")):
            if any(kw in name for kw in ("Model", "Net", "LSTM", "Linear", "ResNet", "Block")):
                return _StubModule
            return _StubObject
        # Handle TorchVersion
        if module == "torch.torch_version" and name == "TorchVersion":
            return str
        # Handle collections.OrderedDict
        if module == "collections" and name == "OrderedDict":
            from collections import OrderedDict
            return OrderedDict
        return super().find_class(module, name)


def _load_checkpoint(path):
    """Load a PyTorch checkpoint with class stubs to avoid dependency hell."""
    with open(path, "rb") as f:
        data = f.read()

    # Check if it's a zip file (PyTorch save format)
    if zipfile.is_zipfile(io.BytesIO(data)):
        zf = zipfile.ZipFile(io.BytesIO(data))
        pkl_names = [n for n in zf.namelist() if n.endswith(".pkl")]
        if not pkl_names:
            raise ValueError("No .pkl found in checkpoint zip")
        pkl_name = pkl_names[0]
        data_prefix = pkl_name.rsplit("/", 1)[0] + "/data/" if "/" in pkl_name else "data/"

        class _TorchUnpickler(_WeSpeakerUnpickler):
            def __init__(self, file, zf, data_prefix):
                super().__init__(file)
                self.zf = zf
                self.data_prefix = data_prefix
                self._storages = {}

            def persistent_load(self, saved_id):
                if isinstance(saved_id, tuple) and saved_id[0] == "storage":
                    _, storage_type, key, location, numel = saved_id
                    if key not in self._storages:
                        raw = self.zf.read(self.data_prefix + str(key))
                        storage = storage_type.from_buffer(raw, byte_order="little")
                        self._storages[key] = storage
                    return self._storages[key]
                raise RuntimeError(f"Unknown persistent_id: {saved_id}")

        pkl_data = zf.read(pkl_name)
        result = _TorchUnpickler(io.BytesIO(pkl_data), zf, data_prefix).load()
    else:
        raise ValueError("Checkpoint is not a zip file")

    # Extract state_dict from various wrapper formats
    if isinstance(result, dict):
        if "state_dict" in result:
            return result["state_dict"]
        # Might be a plain state dict already
        has_tensors = any(isinstance(v, torch.Tensor) for v in result.values())
        if has_tensors:
            return result
    if hasattr(result, "state_dict"):
        sd = result.state_dict()
        if isinstance(sd, dict):
            return sd

    return result


def fuse_bn_into_conv(conv_weight, bn_weight, bn_bias, bn_mean, bn_var, eps=1e-5):
    """Fuse BatchNorm parameters into Conv2d weight and bias.

    w_fused = w * (γ / √(σ² + ε))
    b_fused = β - μ * γ / √(σ² + ε)
    """
    scale = bn_weight / np.sqrt(bn_var + eps)  # [O]
    fused_weight = conv_weight * scale[:, None, None, None]
    fused_bias = bn_bias - bn_mean * scale
    return fused_weight.astype(np.float32), fused_bias.astype(np.float32)


def convert(
    source_model: str = "pyannote/wespeaker-voxceleb-resnet34-LM",
    output_dir: str = "./wespeaker-resnet34-lm-mlx",
    token: str = None,
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if not token:
        token_path = Path.home() / ".cache" / "huggingface" / "token"
        if token_path.exists():
            token = token_path.read_text().strip()

    print(f"Downloading pytorch_model.bin from {source_model}...")
    path = hf_hub_download(source_model, "pytorch_model.bin", token=token)

    print("Loading state dict with custom unpickler...")
    state_dict = _load_checkpoint(path)

    # Filter to only tensors
    for k in list(state_dict.keys()):
        if not isinstance(state_dict[k], torch.Tensor):
            del state_dict[k]

    print(f"Loaded {len(state_dict)} tensors")
    for k, v in sorted(state_dict.items()):
        print(f"  {k}: {list(v.shape)} {v.dtype}")

    output_tensors = {}

    # ── 1. Initial conv1 + bn1 ──
    print("\n--- conv1 + bn1 ---")
    conv1_w = state_dict["resnet.conv1.weight"].numpy()  # [32, 1, 3, 3]
    bn1_w = state_dict["resnet.bn1.weight"].numpy()
    bn1_b = state_dict["resnet.bn1.bias"].numpy()
    bn1_m = state_dict["resnet.bn1.running_mean"].numpy()
    bn1_v = state_dict["resnet.bn1.running_var"].numpy()

    fused_w, fused_b = fuse_bn_into_conv(conv1_w, bn1_w, bn1_b, bn1_m, bn1_v)
    # Transpose [O, I, H, W] → [O, H, W, I] for MLX
    output_tensors["conv1.weight"] = np.transpose(fused_w, (0, 2, 3, 1))
    output_tensors["conv1.bias"] = fused_b
    print(f"  conv1: {conv1_w.shape} → {output_tensors['conv1.weight'].shape}")

    # ── 2. ResNet layers ──
    layer_configs = [
        ("layer1", 3),   # 3 blocks, 32→32
        ("layer2", 4),   # 4 blocks, 32→64, first stride=2
        ("layer3", 6),   # 6 blocks, 64→128, first stride=2
        ("layer4", 3),   # 3 blocks, 128→256, first stride=2
    ]

    for layer_name, num_blocks in layer_configs:
        print(f"\n--- {layer_name} ({num_blocks} blocks) ---")
        for block_idx in range(num_blocks):
            prefix = f"resnet.{layer_name}.{block_idx}"
            out_prefix = f"{layer_name}.{block_idx}"

            # conv1 + bn1, conv2 + bn2
            for conv_idx in [1, 2]:
                conv_key = f"{prefix}.conv{conv_idx}.weight"
                bn_prefix = f"{prefix}.bn{conv_idx}"

                conv_w = state_dict[conv_key].numpy()
                bn_w = state_dict[f"{bn_prefix}.weight"].numpy()
                bn_b = state_dict[f"{bn_prefix}.bias"].numpy()
                bn_m = state_dict[f"{bn_prefix}.running_mean"].numpy()
                bn_v = state_dict[f"{bn_prefix}.running_var"].numpy()

                fused_w, fused_b = fuse_bn_into_conv(conv_w, bn_w, bn_b, bn_m, bn_v)
                out_key = f"{out_prefix}.conv{conv_idx}"
                output_tensors[f"{out_key}.weight"] = np.transpose(fused_w, (0, 2, 3, 1))
                output_tensors[f"{out_key}.bias"] = fused_b
                print(f"  {out_key}: {conv_w.shape} → {output_tensors[f'{out_key}.weight'].shape}")

            # Shortcut (downsample) if present
            shortcut_conv_key = f"{prefix}.shortcut.0.weight"
            if shortcut_conv_key in state_dict:
                conv_w = state_dict[shortcut_conv_key].numpy()  # [O, I, 1, 1]
                bn_w = state_dict[f"{prefix}.shortcut.1.weight"].numpy()
                bn_b = state_dict[f"{prefix}.shortcut.1.bias"].numpy()
                bn_m = state_dict[f"{prefix}.shortcut.1.running_mean"].numpy()
                bn_v = state_dict[f"{prefix}.shortcut.1.running_var"].numpy()

                fused_w, fused_b = fuse_bn_into_conv(conv_w, bn_w, bn_b, bn_m, bn_v)
                out_key = f"{out_prefix}.shortcut"
                output_tensors[f"{out_key}.weight"] = np.transpose(fused_w, (0, 2, 3, 1))
                output_tensors[f"{out_key}.bias"] = fused_b
                print(f"  {out_key}: {conv_w.shape} → {output_tensors[f'{out_key}.weight'].shape}")

    # ── 3. Embedding layer (seg_1) ──
    print("\n--- embedding (seg_1) ---")
    output_tensors["embedding.weight"] = state_dict["resnet.seg_1.weight"].numpy()
    output_tensors["embedding.bias"] = state_dict["resnet.seg_1.bias"].numpy()
    print(f"  embedding: {output_tensors['embedding.weight'].shape}")

    # ── Save ──
    output_file = output_path / "model.safetensors"
    print(f"\nSaving to {output_file}...")
    save_file(output_tensors, str(output_file))

    file_size_mb = output_file.stat().st_size / (1024 * 1024)
    print(f"  Output size: {file_size_mb:.1f} MB")

    config = {
        "model_type": "wespeaker-resnet34-lm",
        "sample_rate": 16000,
        "n_mels": 80,
        "embedding_dim": 256,
        "layers": [3, 4, 6, 3],
        "channels": [32, 64, 128, 256],
        "pooling_output_dim": 5120,
    }
    with open(output_path / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print("  Saved config.json")

    print(f"\nConversion complete! Output in: {output_path}")
    total_params = sum(np.prod(v.shape) for v in output_tensors.values())
    print(f"Total parameters: {total_params:,}")
    return output_path


def upload(output_dir: str, repo_id: str):
    """Upload converted model to HuggingFace Hub."""
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, exist_ok=True)
    api.upload_folder(
        folder_path=output_dir,
        repo_id=repo_id,
        commit_message="Upload WeSpeaker ResNet34-LM speaker embedding model for MLX Swift",
    )
    print(f"Uploaded to: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert pyannote/wespeaker-voxceleb-resnet34-LM to safetensors for MLX Swift"
    )
    parser.add_argument("--source", default="pyannote/wespeaker-voxceleb-resnet34-LM",
                        help="Source model ID on HuggingFace")
    parser.add_argument("--output-dir", default="./wespeaker-resnet34-lm-mlx",
                        help="Output directory")
    parser.add_argument("--token", default=None,
                        help="HuggingFace token (reads ~/.cache/huggingface/token if not set)")
    parser.add_argument("--upload", action="store_true",
                        help="Upload to HuggingFace Hub after conversion")
    parser.add_argument("--repo-id", default="aufklarer/WeSpeaker-ResNet34-LM-MLX",
                        help="HuggingFace repo ID for upload")
    args = parser.parse_args()

    output_path = convert(
        source_model=args.source,
        output_dir=args.output_dir,
        token=args.token,
    )

    if args.upload:
        upload(str(output_path), args.repo_id)
