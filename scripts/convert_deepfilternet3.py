#!/usr/bin/env python3
"""Convert DeepFilterNet3 PyTorch checkpoint to Core ML.

Downloads the pretrained model, loads via the df package, wraps for Core ML
conversion, and saves the .mlpackage + auxiliary signal processing data.

Requires: pip install deepfilternet coremltools

Usage:
    python scripts/convert_deepfilternet3.py [--output OUTPUT_DIR]

Output:
    DeepFilterNet3.mlpackage — Core ML model (runs on Neural Engine)
    auxiliary.npz — signal processing constants:
      - erb_fb [481, 32] — forward ERB filterbank
      - erb_inv_fb [32, 481] — inverse ERB filterbank
      - window [960] — Vorbis analysis/synthesis window
      - mean_norm_state [32] — initial ERB mean normalization state
      - unit_norm_state [96] — initial spec unit normalization state
"""

import argparse
import os
import sys
import types
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Download ────────────────────────────────────────────────────────────────

MODEL_URL = "https://github.com/Rikorose/DeepFilterNet/raw/main/models/DeepFilterNet3.zip"


def download_model(cache_dir: Path) -> Path:
    """Download and extract DeepFilterNet3 checkpoint."""
    extract_dir = cache_dir / "DeepFilterNet3"
    ckpt_path = extract_dir / "checkpoints" / "model_120.ckpt.best"
    config_path = extract_dir / "config.ini"

    if ckpt_path.exists() and config_path.exists():
        print(f"  Using cached checkpoint: {ckpt_path}")
        return ckpt_path

    cache_dir.mkdir(parents=True, exist_ok=True)
    zip_path = cache_dir / "DeepFilterNet3.zip"

    if not zip_path.exists():
        print(f"  Downloading from {MODEL_URL}...")
        import urllib.request
        urllib.request.urlretrieve(MODEL_URL, zip_path)
        print(f"  Downloaded: {zip_path}")

    print("  Extracting...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(cache_dir)

    if not ckpt_path.exists():
        for p in sorted(extract_dir.rglob("*")):
            print(f"    {p.relative_to(extract_dir)}")
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    return ckpt_path


# ─── Patch torchaudio import ────────────────────────────────────────────────

def patch_torchaudio():
    """Stub out torchaudio.backend.common which df.io imports."""
    torchaudio_backend = types.ModuleType("torchaudio.backend")
    torchaudio_backend_common = types.ModuleType("torchaudio.backend.common")
    class AudioMetaData:
        pass
    torchaudio_backend_common.AudioMetaData = AudioMetaData
    sys.modules["torchaudio.backend"] = torchaudio_backend
    sys.modules["torchaudio.backend.common"] = torchaudio_backend_common


# ─── Core ML Wrapper ────────────────────────────────────────────────────────

class DeepFilterNet3CoreML(nn.Module):
    """Wrapper for Core ML export.

    Takes ERB and spec features, returns ERB mask and DF coefficients.
    Lookahead padding is done via slice+pad (no negative padding).
    GRU states are internal (batch mode, reset each call).
    """
    def __init__(self, dfnet, conv_lookahead: int):
        super().__init__()
        self.enc = dfnet.enc
        self.erb_dec = dfnet.erb_dec
        self.df_dec = dfnet.df_dec
        self.df_out_transform = dfnet.df_out_transform
        self.conv_lookahead = conv_lookahead

    def forward(self, feat_erb: torch.Tensor, feat_spec: torch.Tensor):
        """
        Args:
            feat_erb: [1, 1, T, 32] — normalized ERB features
            feat_spec: [1, 2, T, 96] — normalized complex spec features

        Returns:
            erb_mask: [1, 1, T, 32] — sigmoid ERB gain mask
            df_coefs: [1, 5, T, 96, 2] — deep filter coefficients (order, real/imag)
        """
        la = self.conv_lookahead
        # Lookahead: shift time axis (trim beginning, pad end with zeros)
        feat_erb = F.pad(feat_erb[:, :, la:, :], (0, 0, 0, la))
        feat_spec = F.pad(feat_spec[:, :, la:, :], (0, 0, 0, la))

        e0, e1, e2, e3, emb, c0, lsnr = self.enc(feat_erb, feat_spec)
        m = self.erb_dec(emb, e3, e2, e1, e0)
        df_coefs = self.df_dec(emb, c0)
        df_coefs = self.df_out_transform(df_coefs)

        return m, df_coefs


# ─── Signal Processing Constants ─────────────────────────────────────────────

FFT_SIZE = 960
HOP_SIZE = 480
SAMPLE_RATE = 48000
ERB_BANDS = 32
DF_BINS = 96


def compute_vorbis_window(size: int) -> np.ndarray:
    """w[n] = sin(pi/2 * sin^2(pi * (n + 0.5) / N))"""
    n = np.arange(size, dtype=np.float32)
    x = np.pi * (n + 0.5) / size
    return np.sin(np.pi / 2 * np.sin(x) ** 2).astype(np.float32)


# ─── Conversion ──────────────────────────────────────────────────────────────

def convert(ckpt_path: Path, output_dir: Path):
    """Convert PyTorch checkpoint to Core ML .mlpackage."""
    import coremltools as ct

    # Load config from the config.ini next to checkpoint
    config_dir = ckpt_path.parent.parent
    config_path = config_dir / "config.ini"
    if not config_path.exists():
        raise FileNotFoundError(f"config.ini not found at {config_path}")

    os.environ["DF_LOG_LEVEL"] = "error"

    from df.config import config
    config.load(str(config_path), allow_defaults=True)

    from df.deepfilternet3 import DfNet, ModelParams
    from df.modules import erb_fb
    import libdf

    p = ModelParams()
    print(f"  Model config: conv_ch={p.conv_ch}, emb_hidden={p.emb_hidden_dim}, "
          f"erb={p.nb_erb}, df_bins={p.nb_df}, df_order={p.df_order}")

    # Create model
    df_state = libdf.DF(
        sr=p.sr, fft_size=p.fft_size, hop_size=p.hop_size, nb_bands=p.nb_erb)
    erb = erb_fb(df_state.erb_widths(), p.sr, inverse=False)
    erb_inverse = erb_fb(df_state.erb_widths(), p.sr, inverse=True)
    model = DfNet(erb, erb_inverse, run_df=True, train_mask=True)

    # Load weights
    print(f"  Loading checkpoint: {ckpt_path}")
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_sd = {k: v for k, v in sd.items()
                if not k.startswith("erb_fb") and not k.startswith("mask.")}
    missing, unexpected = model.load_state_dict(model_sd, strict=False)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")
    if missing:
        print(f"  Missing keys (expected): {missing}")

    # Wrap for Core ML
    wrapper = DeepFilterNet3CoreML(model, conv_lookahead=p.conv_lookahead)
    wrapper.eval()

    # Verify wrapper matches original
    T = 15
    test_erb = torch.randn(1, 1, T, p.nb_erb)
    test_spec_5d = torch.randn(1, 1, T, p.nb_df, 2)
    test_spec_4d = test_spec_5d.squeeze(1).permute(0, 3, 1, 2)
    test_full = torch.randn(1, 1, T, p.fft_size // 2 + 1, 2)

    with torch.no_grad():
        _, m_orig, _, _ = model(test_full, test_erb, test_spec_5d)
        m_wrap, _ = wrapper(test_erb, test_spec_4d)

    assert torch.allclose(m_orig, m_wrap, atol=1e-5), "Wrapper output mismatch!"
    print("  Wrapper verification passed")

    # Trace
    print("  Tracing model...")
    traced = torch.jit.trace(wrapper, (test_erb, test_spec_4d))

    # Convert to Core ML
    print("  Converting to Core ML...")
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(
                name="feat_erb",
                shape=ct.Shape(shape=(1, 1, ct.RangeDim(1, 6000), ERB_BANDS))),
            ct.TensorType(
                name="feat_spec",
                shape=ct.Shape(shape=(1, 2, ct.RangeDim(1, 6000), DF_BINS))),
        ],
        outputs=[
            ct.TensorType(name="erb_mask"),
            ct.TensorType(name="df_coefs"),
        ],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS14,
    )

    # Save model
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "DeepFilterNet3.mlpackage"
    mlmodel.save(str(model_path))
    print(f"\n  Model saved to: {model_path}")

    # Verify Core ML output
    print("  Verifying Core ML model...")
    T2 = 20
    v_erb = torch.randn(1, 1, T2, p.nb_erb)
    v_spec = torch.randn(1, 2, T2, p.nb_df)
    with torch.no_grad():
        m_pt, c_pt = wrapper(v_erb, v_spec)

    pred = mlmodel.predict({
        "feat_erb": v_erb.numpy(),
        "feat_spec": v_spec.numpy(),
    })
    m_diff = np.abs(pred["erb_mask"] - m_pt.numpy()).max()
    c_diff = np.abs(pred["df_coefs"] - c_pt.numpy()).max()
    print(f"  Max diff — erb_mask: {m_diff:.6f}, df_coefs: {c_diff:.6f}")

    # Save auxiliary signal processing data
    erb_fb_np = erb_fb(df_state.erb_widths(), p.sr, inverse=False).numpy()
    erb_inv_fb_np = erb_fb(df_state.erb_widths(), p.sr, inverse=True).numpy()
    window = compute_vorbis_window(FFT_SIZE)
    mean_state = np.linspace(-60, -90, ERB_BANDS, dtype=np.float32)
    unit_state = np.array(libdf.unit_norm_init(DF_BINS), dtype=np.float32)

    aux_path = output_dir / "auxiliary.npz"
    np.savez(aux_path,
        erb_fb=erb_fb_np,
        erb_inv_fb=erb_inv_fb_np,
        window=window,
        mean_norm_state=mean_state,
        unit_norm_state=unit_state,
    )
    print(f"  Auxiliary data saved to: {aux_path}")

    # Summary
    import shutil
    model_size = sum(f.stat().st_size for f in model_path.rglob("*") if f.is_file())
    aux_size = aux_path.stat().st_size
    print(f"\n  Summary:")
    print(f"    Model: {model_size / 1024 / 1024:.1f} MB ({model_path.name})")
    print(f"    Auxiliary: {aux_size / 1024:.0f} KB ({aux_path.name})")
    print(f"    Parameters: {n_params:,}")
    print(f"    Target: Neural Engine (macOS 14+)")


def main():
    parser = argparse.ArgumentParser(
        description="Convert DeepFilterNet3 to Core ML")
    parser.add_argument("--output", "-o", type=str, default=None,
                       help="Output directory (default: auto-detect cache)")
    parser.add_argument("--checkpoint", "-c", type=str, default=None,
                       help="Path to existing checkpoint (skip download)")
    args = parser.parse_args()

    if args.output:
        output_dir = Path(args.output)
    else:
        cache_base = Path.home() / "Library" / "Caches" / "qwen3-speech"
        output_dir = cache_base / "aufklarer_DeepFilterNet3-CoreML"

    print("Step 1: Download/locate checkpoint")
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    else:
        cache_dir = Path.home() / ".cache" / "deepfilternet3"
        ckpt_path = download_model(cache_dir)

    print("\nStep 2: Convert to Core ML")
    patch_torchaudio()
    convert(ckpt_path, output_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
