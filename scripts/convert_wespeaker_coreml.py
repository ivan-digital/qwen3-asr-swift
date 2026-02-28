#!/usr/bin/env python3
"""Convert WeSpeaker ResNet34-LM to CoreML format.

Pipeline: download pyannote checkpoint -> fuse BN into Conv2d -> build plain
PyTorch ResNet34 -> trace -> CoreML convert with EnumeratedShapes -> compile .mlmodelc

Uses EnumeratedShapes for the variable time dimension T to avoid BNNS crashes
with RangeDim on Neural Engine.

Requires: pip install torch coremltools safetensors numpy huggingface_hub

Usage:
    python3 scripts/convert_wespeaker_coreml.py
    python3 scripts/convert_wespeaker_coreml.py --upload
    python3 scripts/convert_wespeaker_coreml.py --output-dir ./wespeaker-coreml
"""

import argparse
import io
import json
import os
import pickle
import shutil
import subprocess
import zipfile
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download


# Enumerated T values for mel spectrogram time dimension.
# T=20 ~ 0.3s, T=100 ~ 1.6s, T=500 ~ 8s, T=2000 ~ 32s
T_VALUES = [20, 50, 100, 200, 300, 500, 750, 1000, 1500, 2000]


# -- Custom unpickler (same as convert_wespeaker.py) --

class _StubModule(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

class _StubObject:
    def __init__(self, *args, **kwargs):
        pass

class _WeSpeakerUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith(("pyannote", "wespeaker")):
            if any(kw in name for kw in ("Model", "Net", "LSTM", "Linear", "ResNet", "Block")):
                return _StubModule
            return _StubObject
        if module == "torch.torch_version" and name == "TorchVersion":
            return str
        if module == "collections" and name == "OrderedDict":
            from collections import OrderedDict
            return OrderedDict
        return super().find_class(module, name)


def _load_checkpoint(path):
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

    if isinstance(result, dict):
        if "state_dict" in result:
            return result["state_dict"]
        if any(isinstance(v, torch.Tensor) for v in result.values()):
            return result
    if hasattr(result, "state_dict"):
        sd = result.state_dict()
        if isinstance(sd, dict):
            return sd
    return result


def fuse_bn_into_conv(conv_weight, bn_weight, bn_bias, bn_mean, bn_var, eps=1e-5):
    """Fuse BatchNorm into Conv2d: w_fused = w * gamma/sqrt(var+eps), b_fused = beta - mu*gamma/sqrt(var+eps)"""
    scale = bn_weight / np.sqrt(bn_var + eps)
    fused_weight = conv_weight * scale[:, None, None, None]
    fused_bias = bn_bias - bn_mean * scale
    return fused_weight.astype(np.float32), fused_bias.astype(np.float32)


# -- PyTorch ResNet34 for CoreML --

class BasicBlockCoreML(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=True)
        self.shortcut = None
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=True)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        residual = self.shortcut(x) if self.shortcut is not None else x
        return F.relu(out + residual)


class WeSpeakerCoreML(nn.Module):
    """WeSpeaker ResNet34-LM with BN fused into Conv2d, PyTorch NCHW layout."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1, bias=True)

        # Layer1: 3 blocks, 32->32
        self.layer1 = nn.Sequential(*[BasicBlockCoreML(32, 32) for _ in range(3)])

        # Layer2: 4 blocks, 32->64, first stride=2
        blocks2 = [BasicBlockCoreML(32, 64, stride=2)]
        for _ in range(3):
            blocks2.append(BasicBlockCoreML(64, 64))
        self.layer2 = nn.Sequential(*blocks2)

        # Layer3: 6 blocks, 64->128, first stride=2
        blocks3 = [BasicBlockCoreML(64, 128, stride=2)]
        for _ in range(5):
            blocks3.append(BasicBlockCoreML(128, 128))
        self.layer3 = nn.Sequential(*blocks3)

        # Layer4: 3 blocks, 128->256, first stride=2
        blocks4 = [BasicBlockCoreML(128, 256, stride=2)]
        for _ in range(2):
            blocks4.append(BasicBlockCoreML(256, 256))
        self.layer4 = nn.Sequential(*blocks4)

        # Embedding: stats pool -> Linear(5120, 256) -> L2 norm
        self.embedding = nn.Linear(5120, 256)

    def forward(self, mel):
        """
        Args:
            mel: [1, 1, T, 80] NCHW mel spectrogram

        Returns:
            embedding: [1, 256] L2-normalized speaker embedding
        """
        x = F.relu(self.conv1(mel))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # x: [1, 256, T/8, 10]

        # Reshape for stats pooling: [1, 256 * 10, T/8] = [1, 2560, T/8]
        B, C, T, F_ = x.shape
        x = x.reshape(B, C * F_, T)

        # Stats pooling: mean + std over time
        mean = x.mean(dim=2)  # [1, 2560]
        std = x.std(dim=2)    # [1, 2560]
        pooled = torch.cat([mean, std], dim=-1)  # [1, 5120]

        # Embedding
        emb = self.embedding(pooled)  # [1, 256]

        # L2 normalize
        emb = F.normalize(emb, p=2, dim=-1)
        return emb


def load_fused_weights(model, state_dict):
    """Load BN-fused weights from pyannote checkpoint into our plain Conv2d model."""
    numpy_sd = {k: v.numpy() for k, v in state_dict.items() if isinstance(v, torch.Tensor)}

    def fuse_and_load(conv_module, conv_key, bn_key):
        """Fuse BN into Conv2d and load. Weights stay in PyTorch [O,I,H,W]."""
        conv_w = numpy_sd[f"{conv_key}.weight"]
        bn_w = numpy_sd[f"{bn_key}.weight"]
        bn_b = numpy_sd[f"{bn_key}.bias"]
        bn_m = numpy_sd[f"{bn_key}.running_mean"]
        bn_v = numpy_sd[f"{bn_key}.running_var"]
        fw, fb = fuse_bn_into_conv(conv_w, bn_w, bn_b, bn_m, bn_v)
        conv_module.weight.data.copy_(torch.from_numpy(fw))
        conv_module.bias.data.copy_(torch.from_numpy(fb))

    # conv1 + bn1
    fuse_and_load(model.conv1, "resnet.conv1", "resnet.bn1")

    # ResNet layers
    layer_configs = [
        ("layer1", model.layer1, 3),
        ("layer2", model.layer2, 4),
        ("layer3", model.layer3, 6),
        ("layer4", model.layer4, 3),
    ]

    for layer_name, layer_module, num_blocks in layer_configs:
        for block_idx in range(num_blocks):
            prefix = f"resnet.{layer_name}.{block_idx}"
            block = layer_module[block_idx]

            fuse_and_load(block.conv1, f"{prefix}.conv1", f"{prefix}.bn1")
            fuse_and_load(block.conv2, f"{prefix}.conv2", f"{prefix}.bn2")

            shortcut_key = f"{prefix}.shortcut.0.weight"
            if shortcut_key in numpy_sd and block.shortcut is not None:
                fuse_and_load(block.shortcut, f"{prefix}.shortcut.0", f"{prefix}.shortcut.1")

    # Embedding
    model.embedding.weight.data.copy_(state_dict["resnet.seg_1.weight"])
    model.embedding.bias.data.copy_(state_dict["resnet.seg_1.bias"])

    print("Weights loaded successfully")


def verify_pytorch_coreml(pt_model, coreml_model, num_tests=5):
    """Verify CoreML model outputs match PyTorch float32."""
    print("\nVerifying CoreML against PyTorch...")
    torch.manual_seed(42)

    max_cos_diff = 0.0
    for i, T in enumerate(T_VALUES[:num_tests]):
        mel = torch.randn(1, 1, T, 80)

        with torch.no_grad():
            pt_emb = pt_model(mel).numpy().flatten()

        mel_np = mel.numpy().astype(np.float16)
        result = coreml_model.predict({"mel": mel_np})
        cm_emb = np.array(result["embedding"]).flatten()

        cos_sim = float(np.dot(pt_emb, cm_emb) / (np.linalg.norm(pt_emb) * np.linalg.norm(cm_emb) + 1e-10))
        diff = 1.0 - cos_sim
        max_cos_diff = max(max_cos_diff, diff)
        print(f"  T={T:4d}: cosine_sim={cos_sim:.6f}  (1-sim={diff:.6f})")

    print(f"  Max (1 - cosine_sim): {max_cos_diff:.6f}")
    if max_cos_diff < 0.001:
        print("  PASS: cosine similarity > 0.999")
    else:
        print("  WARNING: cosine similarity below 0.999 threshold")

    return max_cos_diff


def main():
    parser = argparse.ArgumentParser(description="Convert WeSpeaker ResNet34-LM to CoreML")
    parser.add_argument("--source", default="pyannote/wespeaker-voxceleb-resnet34-LM",
                        help="Source model on HuggingFace")
    parser.add_argument("--output-dir", default="./wespeaker-resnet34-lm-coreml",
                        help="Output directory")
    parser.add_argument("--token", default=None, help="HuggingFace token")
    parser.add_argument("--upload", action="store_true", help="Upload to HuggingFace")
    parser.add_argument("--repo-id", default="aufklarer/WeSpeaker-ResNet34-LM-CoreML",
                        help="HuggingFace repo ID for upload")
    parser.add_argument("--skip-verify", action="store_true", help="Skip verification")
    args = parser.parse_args()

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Read token
    token = args.token
    if not token:
        token_path = Path.home() / ".cache" / "huggingface" / "token"
        if token_path.exists():
            token = token_path.read_text().strip()

    # Step 1: Download checkpoint
    print(f"Downloading pytorch_model.bin from {args.source}...")
    path = hf_hub_download(args.source, "pytorch_model.bin", token=token)

    print("Loading state dict...")
    state_dict = _load_checkpoint(path)
    for k in list(state_dict.keys()):
        if not isinstance(state_dict[k], torch.Tensor):
            del state_dict[k]
    print(f"Loaded {len(state_dict)} tensors")

    # Step 2: Build plain PyTorch model and load fused weights
    print("\nBuilding PyTorch model with fused BN...")
    pt_model = WeSpeakerCoreML()
    load_fused_weights(pt_model, state_dict)
    pt_model.eval()

    total_params = sum(p.numel() for p in pt_model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Step 3: Trace with a representative T value
    print("\nTracing model...")
    example_mel = torch.randn(1, 1, 200, 80)
    with torch.no_grad():
        traced = torch.jit.trace(pt_model, (example_mel,))

    # Step 4: Convert to CoreML with EnumeratedShapes
    print("Converting to CoreML with EnumeratedShapes...")
    print(f"  T values: {T_VALUES}")

    shapes = [(1, 1, t, 80) for t in T_VALUES]
    mel_input = ct.TensorType(
        "mel",
        shape=ct.EnumeratedShapes(shapes=shapes),
    )

    mlmodel = ct.convert(
        traced,
        inputs=[mel_input],
        outputs=[ct.TensorType("embedding")],
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS17,
    )

    # Step 5: Save .mlpackage
    mlpackage_path = str(output_path / "wespeaker.mlpackage")
    if os.path.exists(mlpackage_path):
        shutil.rmtree(mlpackage_path)
    mlmodel.save(mlpackage_path)
    print(f"Saved .mlpackage to {mlpackage_path}")

    # Step 6: Compile to .mlmodelc
    print("Compiling to .mlmodelc...")
    mlmodelc_path = str(output_path / "wespeaker.mlmodelc")
    if os.path.exists(mlmodelc_path):
        shutil.rmtree(mlmodelc_path)

    result = subprocess.run(
        ["xcrun", "coremlcompiler", "compile", mlpackage_path, str(output_path)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"xcrun compilation failed: {result.stderr}")
        print("Falling back to Python compilation...")
        compiled = mlmodel.get_compiled_model_path()
        if os.path.exists(mlmodelc_path):
            shutil.rmtree(mlmodelc_path)
        shutil.copytree(compiled, mlmodelc_path)
    else:
        print(f"Compiled to {mlmodelc_path}")

    if not os.path.exists(mlmodelc_path):
        print("ERROR: .mlmodelc not found after compilation")
        return

    # Step 7: Verify
    if not args.skip_verify:
        print("\nLoading CoreML model for verification...")
        coreml_model = ct.models.MLModel(mlpackage_path)
        verify_pytorch_coreml(pt_model, coreml_model)

    # Step 8: Save config
    config = {
        "model_type": "wespeaker-resnet34-lm-coreml",
        "sample_rate": 16000,
        "n_mels": 80,
        "embedding_dim": 256,
        "layers": [3, 4, 6, 3],
        "channels": [32, 64, 128, 256],
        "pooling_output_dim": 5120,
        "enumerated_mel_lengths": T_VALUES,
        "compute_precision": "float16",
    }
    config_path = str(output_path / "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved config to {config_path}")

    print(f"\nConversion complete! Output in: {output_path}")

    # Step 9: Upload
    if args.upload:
        try:
            from huggingface_hub import HfApi

            api = HfApi()
            api.create_repo(args.repo_id, exist_ok=True)

            print(f"\nUploading to {args.repo_id}...")
            api.upload_folder(
                folder_path=mlmodelc_path,
                path_in_repo="wespeaker.mlmodelc",
                repo_id=args.repo_id,
            )
            api.upload_file(
                path_or_fileobj=config_path,
                path_in_repo="config.json",
                repo_id=args.repo_id,
            )
            print(f"Uploaded to https://huggingface.co/{args.repo_id}")
        except Exception as e:
            print(f"\nUpload failed: {e}")
            print(f"Upload manually:")
            print(f"  huggingface-cli upload {args.repo_id} {mlmodelc_path} wespeaker.mlmodelc")
            print(f"  huggingface-cli upload {args.repo_id} {config_path} config.json")

    print("\nDone!")


if __name__ == "__main__":
    main()
