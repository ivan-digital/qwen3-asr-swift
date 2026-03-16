#!/usr/bin/env python3
"""Convert FireRedVAD (DFSMN-based) PyTorch checkpoint to CoreML.

FireRedVAD is a lightweight 588K-param VAD model using DFSMN (Deep Feedforward
Sequential Memory Network) blocks with depthwise Conv1d for temporal context.

Architecture:
  Input: [1, T, 80] log Mel fbank (25ms window, 10ms shift)
  CMVN: subtract mean, multiply inverse std (from cmvn.ark)
  DFSMN: fc1(80->256,ReLU) -> fc2(256->128,ReLU) -> FSMN(depthwise Conv1d, k=20)
  7x DFSMNBlock: fc1(128->256,ReLU) -> fc2(256->128,no_bias) -> FSMN + skip
  DNN: Linear(128->256,ReLU)
  Output: Linear(256->1) -> sigmoid -> [1, T, 1] speech probability

Source: https://github.com/FireRedTeam/FireRedASR (Xiaohongshu)

Usage:
    python scripts/convert_fireredvad.py \\
        --model-dir /tmp/FireRedVAD/VAD \\
        --output ./fireredvad-coreml

    # With INT8 quantization:
    python scripts/convert_fireredvad.py \\
        --model-dir /tmp/FireRedVAD/VAD \\
        --output ./fireredvad-coreml \\
        --quantize int8

    # Upload to HuggingFace:
    huggingface-cli upload aufklarer/FireRedVAD-CoreML ./fireredvad-coreml

Output:
    fireredvad.mlpackage  - CoreML model (CPU + Neural Engine)
    fireredvad.mlmodelc   - Compiled CoreML model (for distribution)
    cmvn.json             - CMVN mean/inv_std for Swift feature extraction
    config.json           - Model configuration
"""

import argparse
import json
import os
import shutil
import struct
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import coremltools AFTER torch to avoid SIGSEGV on macOS (coremltools 9.0
# + torch 2.7 import order issue with shared Metal/MPS libraries).
import coremltools as ct


# ─── Model Architecture ──────────────────────────────────────────────────────
# Faithfully reproduced from FireRedASR detect_model.py, with modifications
# for CoreML traceability (no caches, no masking, no dropout).


class FSMN(nn.Module):
    """Feedforward Sequential Memory Network layer.

    Uses depthwise Conv1d for lookback and lookahead temporal context.
    Non-streaming: processes full sequence at once, no caching needed.
    """

    def __init__(self, P, N1, S1, N2=0, S2=0):
        super().__init__()
        assert N1 >= 1
        self.N1, self.S1, self.N2, self.S2 = N1, S1, N2, S2
        self.lookback_padding = (N1 - 1) * S1
        self.lookback_filter = nn.Conv1d(
            in_channels=P,
            out_channels=P,
            kernel_size=N1,
            stride=1,
            padding=self.lookback_padding,
            dilation=S1,
            groups=P,
            bias=False,
        )
        if self.N2 > 0:
            self.lookahead_filter = nn.Conv1d(
                in_channels=P,
                out_channels=P,
                kernel_size=N2,
                stride=1,
                padding=(N2 - 1) * S2,
                dilation=S2,
                groups=P,
                bias=False,
            )

    def forward(self, inputs):
        # inputs: [B, T, P]
        x = inputs.permute(0, 2, 1).contiguous()  # [B, P, T]
        residual = x

        # Lookback filter with causal trimming
        lookback = self.lookback_filter(x)
        if self.N1 > 1:
            lookback = lookback[:, :, : -(self.lookback_padding)]
        memory = residual + lookback

        # Lookahead filter (non-streaming only)
        if self.N2 > 0:
            lookahead = self.lookahead_filter(x)
            memory = memory + F.pad(
                lookahead[:, :, self.N2 * self.S2 :], (0, self.S2)
            )

        memory = memory.permute(0, 2, 1).contiguous()  # [B, T, P]
        return memory


class DFSMNBlock(nn.Module):
    """DFSMN block: Affine+ReLU -> Affine -> FSMN + skip connection."""

    def __init__(self, H, P, N1, S1, N2=0, S2=0):
        super().__init__()
        self.fc1 = nn.Sequential(nn.Linear(P, H, bias=True), nn.ReLU())
        self.fc2 = nn.Linear(H, P, bias=False)
        self.fsmn = FSMN(P, N1, S1, N2, S2)

    def forward(self, inputs):
        # inputs: [B, T, P]
        residual = inputs
        h = self.fc1(inputs)
        p = self.fc2(h)
        memory = self.fsmn(p)
        return memory + residual


class FireRedVADModel(nn.Module):
    """FireRedVAD non-streaming model for CoreML conversion.

    Combines CMVN normalization, DFSMN blocks, DNN, and sigmoid output
    into a single traceable module.
    """

    def __init__(self, cmvn_mean, cmvn_inv_std, D, R, M, H, P, N1, S1, N2, S2):
        super().__init__()
        # CMVN as registered buffers (baked into CoreML model)
        self.register_buffer("cmvn_mean", torch.tensor(cmvn_mean, dtype=torch.float32))
        self.register_buffer(
            "cmvn_inv_std", torch.tensor(cmvn_inv_std, dtype=torch.float32)
        )

        # First FSMN block (no skip connection, connects input layer)
        self.fc1 = nn.Sequential(nn.Linear(D, H, bias=True), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(H, P, bias=True), nn.ReLU())
        self.fsmn1 = FSMN(P, N1, S1, N2, S2)

        # R-1 DFSMN blocks with skip connections
        self.fsmns = nn.ModuleList(
            [DFSMNBlock(H, P, N1, S1, N2, S2) for _ in range(R - 1)]
        )

        # M DNN layers
        dnn_layers = [nn.Linear(P, H, bias=True), nn.ReLU()]
        for _ in range(M - 1):
            dnn_layers += [nn.Linear(H, H, bias=True), nn.ReLU()]
        self.dnns = nn.Sequential(*dnn_layers)

        # Output head
        self.out = nn.Linear(H, 1)

    def forward(self, feat):
        """Non-streaming forward pass.

        Args:
            feat: [1, T, 80] — raw log Mel features (before CMVN)

        Returns:
            probs: [1, T, 1] — speech probability per frame
        """
        # CMVN normalization
        x = (feat - self.cmvn_mean) * self.cmvn_inv_std

        # First FSMN block
        h = self.fc1(x)
        p = self.fc2(h)
        memory = self.fsmn1(p)

        # R-1 DFSMN blocks
        for fsmn in self.fsmns:
            memory = fsmn(memory)

        # DNN + output
        output = self.dnns(memory)
        logits = self.out(output)
        probs = torch.sigmoid(logits)
        return probs


# ─── CMVN Parsing ────────────────────────────────────────────────────────────


def parse_kaldi_cmvn(ark_path):
    """Parse Kaldi binary CMVN ark file.

    Kaldi binary CMVN format:
      \\x00 B         — binary marker (2 bytes)
      D M             — double matrix type (2 bytes)
      space           — separator (1 byte)
      \\x04 rows      — int32 size marker + rows (5 bytes)
      \\x04 cols       — int32 size marker + cols (5 bytes)
      data            — float64 values (rows * cols * 8 bytes)

    Row 0: sums (last element = count)
    Row 1: sum of squares (last element = 0)

    Returns:
        mean: [D] — per-feature mean
        inv_std: [D] — per-feature inverse standard deviation
    """
    with open(ark_path, "rb") as f:
        data = f.read()

    # Validate binary marker
    assert data[0] == 0x00 and data[1] == ord("B"), "Not a Kaldi binary file"

    # Matrix type
    mtype = data[2:4]
    assert mtype in (b"DM", b"FM"), f"Unexpected matrix type: {mtype}"
    is_double = mtype == b"DM"
    dtype = np.float64 if is_double else np.float32
    elem_size = 8 if is_double else 4

    pos = 5  # skip marker(2) + type(2) + space(1)

    # Read rows
    assert data[pos] == 4, f"Expected size marker 4, got {data[pos]}"
    pos += 1
    rows = struct.unpack("<i", data[pos : pos + 4])[0]
    pos += 4

    # Read cols
    assert data[pos] == 4, f"Expected size marker 4, got {data[pos]}"
    pos += 1
    cols = struct.unpack("<i", data[pos : pos + 4])[0]
    pos += 4

    # Read matrix
    matrix = np.frombuffer(
        data[pos : pos + rows * cols * elem_size], dtype=dtype
    ).reshape(rows, cols)

    # Extract mean and inverse std
    count = matrix[0, -1]
    mean = matrix[0, :-1] / count
    variance = matrix[1, :-1] / count - mean**2
    inv_std = 1.0 / np.sqrt(variance + 1e-20)

    return mean.astype(np.float32), inv_std.astype(np.float32)


# ─── Weight Loading ──────────────────────────────────────────────────────────


def load_weights(model, state_dict):
    """Load PyTorch weights into our non-streaming model.

    The original checkpoint uses the same layer structure but wraps fc1/fc2
    in nn.Sequential with dropout. Our model drops the dropout layers but
    keeps the same weight keys (fc1.0.weight == Sequential[0].weight, etc).
    """
    our_sd = model.state_dict()
    loaded = {}

    for key, value in state_dict.items():
        # Map from original "dfsmn.X" and "out.X" keys to our flat structure
        if key.startswith("dfsmn."):
            new_key = key[len("dfsmn.") :]
        elif key.startswith("out."):
            new_key = key
        else:
            print(f"  WARNING: skipping unknown key {key}")
            continue

        if new_key in our_sd:
            if our_sd[new_key].shape == value.shape:
                loaded[new_key] = value
            else:
                print(
                    f"  WARNING: shape mismatch for {new_key}: "
                    f"expected {our_sd[new_key].shape}, got {value.shape}"
                )
        else:
            print(f"  WARNING: key {new_key} (from {key}) not in model")

    # Check for missing keys (exclude CMVN buffers)
    missing = set(our_sd.keys()) - set(loaded.keys()) - {"cmvn_mean", "cmvn_inv_std"}
    if missing:
        print(f"  WARNING: missing keys: {missing}")

    model.load_state_dict(loaded, strict=False)
    print(f"  Loaded {len(loaded)}/{len(our_sd) - 2} weight tensors")


# ─── Verification ────────────────────────────────────────────────────────────


def verify_against_pytorch(our_model, original_forward, cmvn_mean, cmvn_inv_std):
    """Verify our model matches the original PyTorch model output."""
    print("\nVerifying against original PyTorch model...")

    torch.manual_seed(42)
    for T in [50, 100, 200, 500]:
        # Raw features (before CMVN)
        raw_feat = torch.randn(1, T, 80)

        # Our model: takes raw features, applies CMVN internally
        with torch.no_grad():
            our_out = our_model(raw_feat)

        # Original model: needs CMVN-normalized input
        cmvn_feat = (raw_feat - cmvn_mean) * cmvn_inv_std
        with torch.no_grad():
            orig_out = original_forward(cmvn_feat)

        diff = (our_out - orig_out).abs().max().item()
        print(f"  T={T}: max_diff={diff:.8f} {'PASS' if diff < 1e-5 else 'FAIL'}")


def verify_coreml(our_model, coreml_model, num_tests=5):
    """Verify CoreML model matches PyTorch model."""
    print("\nVerifying CoreML against PyTorch...")

    torch.manual_seed(42)
    max_diff = 0

    for i in range(num_tests):
        T = 100 + i * 50
        feat = torch.randn(1, T, 80)

        # PyTorch
        with torch.no_grad():
            pt_out = our_model(feat)

        # CoreML
        feat_np = feat.numpy().astype(np.float16)
        result = coreml_model.predict({"features": feat_np})
        cm_out = result["probabilities"]
        cm_tensor = torch.tensor(np.array(cm_out), dtype=torch.float32)

        diff = (pt_out - cm_tensor).abs().max().item()
        max_diff = max(max_diff, diff)
        print(
            f"  T={T}: PyTorch range=[{pt_out.min():.4f}, {pt_out.max():.4f}], "
            f"CoreML range=[{cm_tensor.min():.4f}, {cm_tensor.max():.4f}], "
            f"max_diff={diff:.6f}"
        )

    print(f"  Max difference: {max_diff:.6f}")
    if max_diff < 0.05:
        print("  PASS: CoreML matches PyTorch within float16 tolerance")
    else:
        print("  WARNING: CoreML diverges from PyTorch")


# ─── Main ────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Convert FireRedVAD to CoreML",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="/tmp/FireRedVAD/VAD",
        help="Directory containing model.pth.tar and cmvn.ark",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./fireredvad-coreml",
        help="Output directory",
    )
    parser.add_argument(
        "--quantize",
        type=str,
        choices=["none", "int8"],
        default="none",
        help="Quantization mode (default: none, model is only 2.3MB)",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Compile .mlpackage to .mlmodelc",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip verification against original model",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload to HuggingFace",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="aufklarer/FireRedVAD-CoreML",
        help="HuggingFace repo ID for upload",
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ─── Step 1: Parse CMVN ──────────────────────────────────────────────
    print("=" * 60)
    print("Step 1: Parsing CMVN normalization")
    print("=" * 60)

    cmvn_path = model_dir / "cmvn.ark"
    if not cmvn_path.exists():
        raise FileNotFoundError(f"CMVN file not found: {cmvn_path}")

    cmvn_mean, cmvn_inv_std = parse_kaldi_cmvn(cmvn_path)
    print(f"  Features: {len(cmvn_mean)}")
    print(f"  Mean range: [{cmvn_mean.min():.4f}, {cmvn_mean.max():.4f}]")
    print(f"  InvStd range: [{cmvn_inv_std.min():.4f}, {cmvn_inv_std.max():.4f}]")

    # Save CMVN as JSON (for Swift-side feature extraction if needed separately)
    cmvn_json = {
        "mean": cmvn_mean.tolist(),
        "inv_std": cmvn_inv_std.tolist(),
    }
    cmvn_json_path = output_dir / "cmvn.json"
    with open(cmvn_json_path, "w") as f:
        json.dump(cmvn_json, f, indent=2)
    print(f"  Saved CMVN to {cmvn_json_path}")

    # ─── Step 2: Load original model ─────────────────────────────────────
    print()
    print("=" * 60)
    print("Step 2: Loading original PyTorch model")
    print("=" * 60)

    ckpt_path = model_dir / "model.pth.tar"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    pkg = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_args = pkg["args"]
    state_dict = pkg["model_state_dict"]

    print(f"  Config: R={model_args.R}, M={model_args.M}, H={model_args.H}, "
          f"P={model_args.P}, N1={model_args.N1}, S1={model_args.S1}, "
          f"N2={model_args.N2}, S2={model_args.S2}")
    print(f"  Input dim: {model_args.idim}, Output dim: {model_args.odim}")
    print(f"  Weights: {len(state_dict)} tensors")

    # Load original model for verification (import directly to avoid kaldiio)
    original_model = None
    original_forward = None
    if not args.skip_verify:
        try:
            import importlib.util

            spec = importlib.util.spec_from_file_location(
                "detect_model",
                str(
                    Path(__file__).parent.parent
                    / "fireredasr2s"
                    / "fireredvad"
                    / "core"
                    / "detect_model.py"
                ),
            )
            if spec is None:
                # Try the /tmp location
                spec = importlib.util.spec_from_file_location(
                    "detect_model",
                    "/tmp/FireRedASR2S/fireredasr2s/fireredvad/core/detect_model.py",
                )
            if spec is not None:
                dm = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(dm)
                original_model = dm.DetectModel(model_args)
                original_model.load_state_dict(state_dict, strict=True)
                original_model.eval()

                def _orig_forward(cmvn_feat):
                    probs, _ = original_model(cmvn_feat, caches=None)
                    return probs

                original_forward = _orig_forward
                print("  Original model loaded for verification")
        except Exception as e:
            print(f"  Could not load original model for verification: {e}")
            print("  Skipping PyTorch-vs-PyTorch verification")

    # ─── Step 3: Build our model ─────────────────────────────────────────
    print()
    print("=" * 60)
    print("Step 3: Building non-streaming model with baked CMVN")
    print("=" * 60)

    model = FireRedVADModel(
        cmvn_mean=cmvn_mean,
        cmvn_inv_std=cmvn_inv_std,
        D=model_args.idim,
        R=model_args.R,
        M=model_args.M,
        H=model_args.H,
        P=model_args.P,
        N1=model_args.N1,
        S1=model_args.S1,
        N2=model_args.N2,
        S2=model_args.S2,
    )

    load_weights(model, state_dict)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")

    # Quick sanity check
    with torch.no_grad():
        test_out = model(torch.randn(1, 100, 80))
    print(f"  Sanity check: input [1,100,80] -> output {test_out.shape}, "
          f"range=[{test_out.min():.4f}, {test_out.max():.4f}]")

    # ─── Step 4: Verify against original ─────────────────────────────────
    if original_forward is not None:
        cmvn_mean_t = torch.tensor(cmvn_mean, dtype=torch.float32)
        cmvn_inv_std_t = torch.tensor(cmvn_inv_std, dtype=torch.float32)
        verify_against_pytorch(model, original_forward, cmvn_mean_t, cmvn_inv_std_t)

    # ─── Step 5: Trace ───────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("Step 5: Tracing model")
    print("=" * 60)

    # Trace with a representative input size
    example_feat = torch.randn(1, 200, 80)
    with torch.no_grad():
        traced = torch.jit.trace(model, (example_feat,))

    # Verify traced model with different lengths
    for T in [50, 100, 200, 500]:
        test_feat = torch.randn(1, T, 80)
        with torch.no_grad():
            ref = model(test_feat)
            traced_out = traced(test_feat)
        diff = (ref - traced_out).abs().max().item()
        print(f"  T={T}: traced max_diff={diff:.8f}")

    print("  Tracing OK")

    # ─── Step 6: Convert to CoreML ───────────────────────────────────────
    print()
    print("=" * 60)
    print("Step 6: Converting to CoreML")
    print("=" * 60)

    # Use RangeDim for variable sequence length.
    # Typical range: 10 frames (100ms) to 6000 frames (60s of audio).
    seq_range = ct.RangeDim(lower_bound=1, upper_bound=6000, default=200)
    input_shape = ct.Shape(shape=(1, seq_range, 80))

    print("  Converting with variable sequence length [1..6000]...")
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType("features", shape=input_shape, dtype=np.float32),
        ],
        outputs=[
            ct.TensorType("probabilities"),
        ],
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS17,
    )

    # ─── Step 7: Quantize (optional) ─────────────────────────────────────
    if args.quantize == "int8":
        print()
        print("=" * 60)
        print("Step 7: Quantizing to INT8")
        print("=" * 60)

        from coremltools.optimize.coreml import (
            OpLinearQuantizerConfig,
            OptimizationConfig,
            linear_quantize_weights,
        )

        op_config = OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8")
        config = OptimizationConfig(global_config=op_config)
        mlmodel = linear_quantize_weights(mlmodel, config)
        print("  INT8 quantization applied")

    # ─── Step 8: Save ────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("Step 8: Saving CoreML model")
    print("=" * 60)

    mlpackage_path = output_dir / "fireredvad.mlpackage"
    if mlpackage_path.exists():
        shutil.rmtree(mlpackage_path)
    mlmodel.save(str(mlpackage_path))
    print(f"  Saved {mlpackage_path}")

    # ─── Step 9: Compile (optional) ──────────────────────────────────────
    if args.compile:
        print()
        print("=" * 60)
        print("Step 9: Compiling to .mlmodelc")
        print("=" * 60)

        mlmodelc_path = output_dir / "fireredvad.mlmodelc"
        if mlmodelc_path.exists():
            shutil.rmtree(mlmodelc_path)

        compiled_url = ct.utils.compile_model(str(mlpackage_path))
        shutil.move(str(compiled_url), str(mlmodelc_path))
        print(f"  Compiled to {mlmodelc_path}")

    # ─── Step 10: Verify CoreML ──────────────────────────────────────────
    if not args.skip_verify:
        print()
        print("=" * 60)
        print("Step 10: Verifying CoreML model")
        print("=" * 60)

        coreml_model = ct.models.MLModel(str(mlpackage_path))
        verify_coreml(model, coreml_model)

    # ─── Step 11: Save config ────────────────────────────────────────────
    print()
    print("=" * 60)
    print("Step 11: Saving configuration")
    print("=" * 60)

    config = {
        "model_type": "fireredvad_coreml",
        "architecture": "DFSMN",
        "num_params": total_params,
        "sample_rate": 16000,
        "frame_length_ms": 25,
        "frame_shift_ms": 10,
        "num_mel_bins": model_args.idim,
        "dfsmn_blocks": model_args.R,
        "dnn_layers": model_args.M,
        "hidden_size": model_args.H,
        "projection_size": model_args.P,
        "lookback_order": model_args.N1,
        "lookback_stride": model_args.S1,
        "lookahead_order": model_args.N2,
        "lookahead_stride": model_args.S2,
        "output_dim": model_args.odim,
        "cmvn_embedded": True,
        "compute_precision": "float16",
        "quantization": args.quantize if args.quantize != "none" else None,
    }

    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Saved config to {config_path}")

    # ─── Summary ─────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)

    print(f"  Parameters: {total_params:,}")
    print(f"  Output directory: {output_dir}")
    print("  Files:")
    for f in sorted(output_dir.iterdir()):
        if f.is_dir():
            size = sum(ff.stat().st_size for ff in f.rglob("*") if ff.is_file())
        else:
            size = f.stat().st_size
        print(f"    {f.name}: {size / 1024:.1f} KB")

    # ─── Upload (optional) ───────────────────────────────────────────────
    if args.upload:
        print()
        print("=" * 60)
        print("Uploading to HuggingFace")
        print("=" * 60)

        try:
            from huggingface_hub import HfApi

            api = HfApi()
            api.create_repo(args.repo_id, exist_ok=True)

            print(f"  Uploading to {args.repo_id}...")
            api.upload_folder(
                folder_path=str(output_dir),
                repo_id=args.repo_id,
            )
            print(f"  Uploaded to https://huggingface.co/{args.repo_id}")
        except Exception as e:
            print(f"  Upload failed: {e}")
            print(f"  Upload manually:")
            print(f"    huggingface-cli upload {args.repo_id} {output_dir}")

    print("\nDone!")


if __name__ == "__main__":
    main()
