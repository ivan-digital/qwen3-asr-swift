#!/usr/bin/env python3
"""
Convert pyannote/segmentation-3.0 (PyanNet) weights to safetensors for Swift/MLX.

The model uses SincNet → BiLSTM → Linear → Classifier architecture (~1.5M params).
Key transformations:
  - SincNet: pre-compute 80 sinc filters (40 cos + 40 sin) from learned low_hz_/band_hz_
    using the same formulation as asteroid_filterbanks.ParamSincFB
  - Conv1d: transpose weights from PyTorch [O,I,K] → MLX [O,K,I]
  - BiLSTM: split into forward/backward, sum bias_ih + bias_hh
  - Linear/classifier: keep as-is (already [O,I])

Loads pytorch_model.bin directly with a custom unpickler — does NOT require pyannote.audio.

Requires: pip install torch safetensors numpy huggingface_hub
Note: pyannote/segmentation-3.0 is a gated model — you need a HuggingFace token
      with access granted at https://huggingface.co/pyannote/segmentation-3.0

Usage:
  python scripts/convert_pyannote.py --token YOUR_HF_TOKEN
  python scripts/convert_pyannote.py --output-dir ./pyannote-mlx
  python scripts/convert_pyannote.py --upload --repo-id you/Pyannote-Segmentation-MLX
"""

import argparse
import io
import json
import os
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

class _PyannoteUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("pyannote"):
            if "Model" in name or "Net" in name or "LSTM" in name or "Linear" in name:
                return _StubModule
            return _StubObject
        return super().find_class(module, name)


def _load_checkpoint(path):
    """Load a PyTorch Lightning checkpoint with pyannote class stubs."""
    with open(path, "rb") as f:
        data = f.read()

    if not zipfile.is_zipfile(io.BytesIO(data)):
        raise ValueError("Checkpoint is not a zip file")

    zf = zipfile.ZipFile(io.BytesIO(data))
    pkl_names = [n for n in zf.namelist() if n.endswith(".pkl")]
    if not pkl_names:
        raise ValueError("No .pkl found in checkpoint zip")
    pkl_name = pkl_names[0]
    data_prefix = pkl_name.rsplit("/", 1)[0] + "/data/" if "/" in pkl_name else "data/"

    class _TorchUnpickler(_PyannoteUnpickler):
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

    # PyTorch Lightning checkpoints nest the model weights under "state_dict"
    if isinstance(result, dict) and "state_dict" in result:
        return result["state_dict"]
    if hasattr(result, "state_dict"):
        return result.state_dict()
    return result


# ── SincNet filter computation (matching asteroid ParamSincFB) ──

def compute_sinc_filters(low_hz_param, band_hz_param, n_buffer, window_buffer,
                         min_low_hz=50, min_band_hz=50, sample_rate=16000):
    """Compute 80 sinc bandpass filters (40 cos + 40 sin) from 40 learned parameters.

    This matches asteroid_filterbanks.ParamSincFB.filters() exactly:
    - Cosine filters (even, symmetric): standard sinc bandpass
    - Sine filters (odd, antisymmetric): Hilbert transform of the bandpass

    Args:
        low_hz_param: (40, 1) learned low frequency parameters
        band_hz_param: (40, 1) learned bandwidth parameters
        n_buffer: (1, 125) time buffer: 2π * [-125, ..., -1] / sample_rate
        window_buffer: (125,) left half of Hamming window
        min_low_hz: minimum low frequency offset (50 Hz for pyannote)
        min_band_hz: minimum bandwidth offset (50 Hz for pyannote)
        sample_rate: audio sample rate

    Returns:
        filters: (80, 1, 251) pre-computed filter kernels [PyTorch format: O, I, K]
    """
    # Apply parameter transforms (matching asteroid's forward pass)
    low = min_low_hz + np.abs(low_hz_param)          # (40, 1)
    high = np.clip(
        low + min_band_hz + np.abs(band_hz_param),
        min_low_hz,
        sample_rate / 2
    )  # (40, 1)

    n_filters_half = len(low_hz_param)
    kernel_size = 2 * len(window_buffer) + 1  # 125 * 2 + 1 = 251

    def make_filters(low, high, filt_type):
        band = (high - low)[:, 0]            # (40,)
        ft_low = low @ n_buffer               # (40, 125)
        ft_high = high @ n_buffer             # (40, 125)

        if filt_type == "cos":
            bp_left = ((np.sin(ft_high) - np.sin(ft_low)) / (n_buffer / 2)) * window_buffer
            bp_center = 2 * band.reshape(-1, 1)
            bp_right = np.flip(bp_left, axis=1)
        elif filt_type == "sin":
            bp_left = ((np.cos(ft_low) - np.cos(ft_high)) / (n_buffer / 2)) * window_buffer
            bp_center = np.zeros_like(band.reshape(-1, 1))
            bp_right = -np.flip(bp_left, axis=1)
        else:
            raise ValueError(f"Invalid filter type {filt_type}")

        bp = np.concatenate([bp_left, bp_center, bp_right], axis=1)  # (40, 251)
        bp = bp / (2 * band[:, None])
        return bp.reshape(n_filters_half, 1, kernel_size)

    cos_filters = make_filters(low, high, "cos")  # (40, 1, 251)
    sin_filters = make_filters(low, high, "sin")  # (40, 1, 251)
    filters = np.concatenate([cos_filters, sin_filters], axis=0)  # (80, 1, 251)
    return filters.astype(np.float32)


def convert(
    source_model: str = "pyannote/segmentation-3.0",
    output_dir: str = "./pyannote-segmentation-mlx",
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

    # ── 1. SincNet filterbank (40 params → 80 cos+sin filters) ──
    low_hz = state_dict["sincnet.conv1d.0.filterbank.low_hz_"].numpy()     # (40, 1)
    band_hz = state_dict["sincnet.conv1d.0.filterbank.band_hz_"].numpy()   # (40, 1)
    n_buffer = state_dict["sincnet.conv1d.0.filterbank.n_"].numpy()        # (1, 125)
    window_buffer = state_dict["sincnet.conv1d.0.filterbank.window_"].numpy()  # (125,)

    print(f"\nSincNet: {len(low_hz)} parameter pairs → 80 filters (cos + sin)")
    print(f"  low_hz range: [{low_hz.min():.1f}, {low_hz.max():.1f}]")
    print(f"  band_hz range: [{band_hz.min():.1f}, {band_hz.max():.1f}]")

    filters = compute_sinc_filters(low_hz, band_hz, n_buffer, window_buffer)
    # Transpose to MLX Conv1d format: [O, I, K] → [O, K, I]
    output_tensors["sincnet.conv.0.weight"] = np.transpose(filters, (0, 2, 1))
    print(f"  Pre-computed filters: {filters.shape} → MLX {output_tensors['sincnet.conv.0.weight'].shape}")

    # ── 2. Conv layers 1, 2 ──
    for i in [1, 2]:
        w = state_dict[f"sincnet.conv1d.{i}.weight"].numpy()
        output_tensors[f"sincnet.conv.{i}.weight"] = np.transpose(w, (0, 2, 1))
        output_tensors[f"sincnet.conv.{i}.bias"] = state_dict[f"sincnet.conv1d.{i}.bias"].numpy()
        print(f"  Conv {i}: {w.shape} → {output_tensors[f'sincnet.conv.{i}.weight'].shape}")

    # ── 3. InstanceNorm ──
    for i in range(3):
        for p in ["weight", "bias"]:
            output_tensors[f"sincnet.norm.{i}.{p}"] = state_dict[f"sincnet.norm1d.{i}.{p}"].numpy()
    for p in ["weight", "bias"]:
        output_tensors[f"sincnet.wav_norm.{p}"] = state_dict[f"sincnet.wav_norm1d.{p}"].numpy()

    # ── 4. BiLSTM ──
    num_layers = 0
    while f"lstm.weight_ih_l{num_layers}" in state_dict:
        num_layers += 1
    print(f"\nBiLSTM: {num_layers} layers")

    for i in range(num_layers):
        for direction, suffix in [("fwd", ""), ("bwd", "_reverse")]:
            wih = state_dict[f"lstm.weight_ih_l{i}{suffix}"].numpy()
            whh = state_dict[f"lstm.weight_hh_l{i}{suffix}"].numpy()
            bih = state_dict[f"lstm.bias_ih_l{i}{suffix}"].numpy()
            bhh = state_dict[f"lstm.bias_hh_l{i}{suffix}"].numpy()
            output_tensors[f"lstm_{direction}.layers.{i}.Wx"] = wih
            output_tensors[f"lstm_{direction}.layers.{i}.Wh"] = whh
            output_tensors[f"lstm_{direction}.layers.{i}.bias"] = bih + bhh
            print(f"  {direction} layer {i}: Wx={wih.shape}, Wh={whh.shape}")

    # ── 5. Linear + classifier ──
    print("\nLinear layers:")
    for k in sorted(state_dict):
        if k.startswith("linear.") or k.startswith("classifier."):
            output_tensors[k] = state_dict[k].numpy()
            print(f"  {k}: {output_tensors[k].shape}")

    # ── Save ──
    output_file = output_path / "model.safetensors"
    print(f"\nSaving to {output_file}...")
    save_file(output_tensors, str(output_file))

    file_size_mb = output_file.stat().st_size / (1024 * 1024)
    print(f"  Output size: {file_size_mb:.1f} MB")

    config = {
        "model_type": "pyannote-segmentation",
        "sample_rate": 16000,
        "sincnet": {
            "n_filters": [80, 60, 60],
            "kernel_sizes": [251, 5, 5],
            "strides": [10, 1, 1],
            "pool_sizes": [3, 3, 3],
        },
        "lstm": {
            "hidden_size": 128,
            "num_layers": num_layers,
            "bidirectional": True,
        },
        "linear": {
            "hidden_size": 128,
            "num_layers": 2,
        },
        "num_classes": 7,
        "max_speakers": 3,
        "powerset_max_classes": 2,
        "num_frames_per_chunk": 589,
        "chunk_duration": 10.0,
        "chunk_step_ratio": 0.1,
        "warm_up": [0.0, 0.0],
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
        commit_message="Upload pyannote segmentation model for MLX Swift",
    )
    print(f"Uploaded to: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert pyannote/segmentation-3.0 to safetensors for MLX Swift"
    )
    parser.add_argument("--source", default="pyannote/segmentation-3.0",
                        help="Source model ID on HuggingFace")
    parser.add_argument("--output-dir", default="./pyannote-segmentation-mlx",
                        help="Output directory")
    parser.add_argument("--token", default=None,
                        help="HuggingFace token (reads ~/.cache/huggingface/token if not set)")
    parser.add_argument("--upload", action="store_true",
                        help="Upload to HuggingFace Hub after conversion")
    parser.add_argument("--repo-id", default="aufklarer/Pyannote-Segmentation-MLX",
                        help="HuggingFace repo ID for upload")
    args = parser.parse_args()

    output_path = convert(
        source_model=args.source,
        output_dir=args.output_dir,
        token=args.token,
    )

    if args.upload:
        upload(str(output_path), args.repo_id)
