#!/usr/bin/env python3
"""
Convert Qwen3-ASR-0.6B audio encoder to CoreML for iOS/iPad deployment.

Loads audio_tower weights from HuggingFace, builds a clean PyTorch encoder,
traces it, converts to CoreML with EnumeratedShapes for variable-length audio,
and applies INT4 palettization for Neural Engine efficiency.

The audio encoder processes mel spectrograms → audio embeddings that feed
into the text decoder. This converts the encoder only; the text decoder
conversion (with KV cache) is a follow-up task.

Note: The encoder processes mel spectrograms WITHOUT chunking/block attention.
The original model chunks mel into 100-frame segments, but for typical audio
lengths (< 30s) full attention works well and simplifies the CoreML model.

Requires:
  pip install torch coremltools safetensors numpy huggingface_hub

Usage:
  python scripts/convert_qwen3_asr_coreml.py
  python scripts/convert_qwen3_asr_coreml.py --output-dir ./qwen3-asr-coreml --compile
  python scripts/convert_qwen3_asr_coreml.py --no-quantize  # skip INT4
"""

import argparse
import gc
import json
import math
import shutil
import subprocess
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Encoder config for Qwen3-ASR-0.6B ──

ENCODER_CONFIG = {
    "d_model": 896,
    "num_heads": 14,
    "ffn_dim": 3584,
    "num_layers": 18,
    "num_mel_bins": 128,
    "downsample_hidden": 480,
    "output_dim": 1024,
    "layer_norm_eps": 1e-5,
    "conv_out_dim": 7680,  # 480 channels * 16 spatial
    "max_pos_embed": 1500,
}

# Enumerated mel time lengths for CoreML EnumeratedShapes.
# Each value = number of mel frames. At hop=160, 16kHz: T=100 ~ 1s.
# After 3x stride-2 conv (8x downsampling): T/8 output tokens.
MEL_T_VALUES = [100, 200, 400, 600, 800, 1000, 1500, 2000, 3000]


# ── PyTorch encoder modules ──

class AudioSelfAttention(nn.Module):
    """Multi-head self-attention matching audio_tower.layers.N.self_attn weights."""

    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, _ = x.shape
        q = self.q_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)

        out = out.transpose(1, 2).reshape(B, T, -1)
        return self.out_proj(out)


class AudioEncoderLayer(nn.Module):
    """Transformer layer matching audio_tower.layers.N weights.

    Pre-norm architecture: LN → Attn → residual → LN → FFN(GELU) → residual.
    """

    def __init__(self, d_model, num_heads, ffn_dim, eps):
        super().__init__()
        self.self_attn = AudioSelfAttention(d_model, num_heads)
        self.self_attn_layer_norm = nn.LayerNorm(d_model, eps=eps)
        self.fc1 = nn.Linear(d_model, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, d_model)
        self.final_layer_norm = nn.LayerNorm(d_model, eps=eps)

    def forward(self, x):
        residual = x
        x = self.self_attn(self.self_attn_layer_norm(x))
        x = residual + x

        residual = x
        x = self.final_layer_norm(x)
        x = self.fc2(F.gelu(self.fc1(x)))
        x = residual + x

        return x


def _sinusoidal_pos_embed(max_len, d_model):
    """Sinusoidal position embeddings matching mlx-audio / Whisper convention."""
    half = d_model // 2
    inv = torch.exp(torch.arange(half, dtype=torch.float32) * -(math.log(10000.0) / (half - 1)))
    pos = torch.arange(max_len, dtype=torch.float32)
    scaled = pos.unsqueeze(1) * inv.unsqueeze(0)
    return torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=1).unsqueeze(0)


class Qwen3ASREncoder(nn.Module):
    """Qwen3-ASR audio encoder for CoreML conversion.

    Architecture:
      mel [1, 128, T] → 3x Conv2d(stride=2) → Linear(7680→896)
      → sinusoidal pos embed → 18x Transformer → LN → proj(896→1024)

    Weight keys match audio_tower.* from HuggingFace safetensors directly.
    """

    def __init__(self, config=None):
        super().__init__()
        c = config or ENCODER_CONFIG
        d = c["d_model"]
        h = c["downsample_hidden"]

        self.conv2d1 = nn.Conv2d(1, h, 3, stride=2, padding=1)
        self.conv2d2 = nn.Conv2d(h, h, 3, stride=2, padding=1)
        self.conv2d3 = nn.Conv2d(h, h, 3, stride=2, padding=1)

        self.conv_out = nn.Linear(c["conv_out_dim"], d, bias=False)

        self.register_buffer("pos_embed", _sinusoidal_pos_embed(c["max_pos_embed"], d))

        self.layers = nn.ModuleList([
            AudioEncoderLayer(d, c["num_heads"], c["ffn_dim"], c["layer_norm_eps"])
            for _ in range(c["num_layers"])
        ])

        self.ln_post = nn.LayerNorm(d, eps=c["layer_norm_eps"])
        self.proj1 = nn.Linear(d, d)
        self.proj2 = nn.Linear(d, c["output_dim"])

    def forward(self, mel):
        """Process mel spectrogram to audio embeddings.

        Args:
            mel: [1, 128, T] mel spectrogram (variable T via EnumeratedShapes)
        Returns:
            [1, T/8, 1024] audio embeddings for text decoder
        """
        # Conv frontend: [1, 128, T] → [1, 1, 128, T] → conv → [1, 480, 16, T/8]
        x = mel.unsqueeze(1)
        x = F.gelu(self.conv2d1(x))
        x = F.gelu(self.conv2d2(x))
        x = F.gelu(self.conv2d3(x))

        # Reshape NCHW to [B, T/8, C*H]: permute(0,3,1,2) matches Swift flatten order
        B, C, H, W = x.shape
        x = x.permute(0, 3, 1, 2).reshape(B, W, C * H)

        x = self.conv_out(x)
        x = x + self.pos_embed[:, :x.shape[1], :]

        for layer in self.layers:
            x = layer(x)

        x = self.ln_post(x)
        x = F.gelu(self.proj1(x))
        x = self.proj2(x)
        return x


# ── Weight loading ──

def download_encoder_weights(model_id):
    """Download safetensors from HuggingFace and extract audio_tower weights."""
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file

    print(f"Downloading weights from {model_id}...")
    cache_dir = Path(snapshot_download(
        model_id,
        allow_patterns=["*.safetensors", "model.safetensors.index.json"],
    ))

    all_weights = {}
    for f in sorted(cache_dir.glob("*.safetensors")):
        print(f"  Loading {f.name}...")
        all_weights.update(load_file(str(f)))

    # Strip prefix: audio_tower.X, model.audio_tower.X, thinker.audio_tower.X
    audio_weights = {}
    for k, v in all_weights.items():
        for prefix in ["thinker.audio_tower.", "model.audio_tower.", "audio_tower."]:
            if k.startswith(prefix):
                audio_weights[k[len(prefix):]] = v
                break

    print(f"  Extracted {len(audio_weights)} audio_tower weights")
    if not audio_weights:
        print("  WARNING: No audio_tower weights found. Available prefixes:")
        prefixes = set(k.rsplit(".", 1)[0].split(".")[0] for k in all_weights)
        print(f"    {sorted(prefixes)}")

    return audio_weights


def load_weights(model, weights):
    """Load HuggingFace weights into the PyTorch encoder."""
    sd = model.state_dict()
    loaded = 0

    for key in sd:
        if key == "pos_embed":
            continue  # computed buffer, not from weights
        if key in weights:
            if sd[key].shape == weights[key].shape:
                sd[key] = weights[key].float()
                loaded += 1
            else:
                print(f"  Shape mismatch: {key} model={sd[key].shape} weight={weights[key].shape}")
        else:
            print(f"  Missing: {key}")

    model.load_state_dict(sd)
    n_params = len(sd) - 1  # exclude pos_embed
    print(f"  Loaded {loaded}/{n_params} weights")
    return loaded == n_params


# ── CoreML conversion ──

def convert_to_coreml(traced, quantize):
    """Convert traced encoder to CoreML with EnumeratedShapes and optional INT4."""
    mel_shapes = [ct.Shape(shape=(1, 128, t)) for t in MEL_T_VALUES]

    print(f"Converting to CoreML (shapes: {MEL_T_VALUES})...")
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(
            name="mel",
            shape=ct.EnumeratedShapes(shapes=mel_shapes),
            dtype=np.float32,
        )],
        outputs=[ct.TensorType(name="audio_embeddings")],
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS17,
    )

    if quantize:
        print("  Applying INT4 palettization...")
        from coremltools.optimize.coreml import (
            OpPalettizerConfig,
            OptimizationConfig,
            palettize_weights,
        )
        op_config = OpPalettizerConfig(mode="kmeans", nbits=4)
        config = OptimizationConfig(global_config=op_config)
        mlmodel = palettize_weights(mlmodel, config)

    return mlmodel


def verify(pt_model, coreml_path, n=5):
    """Verify CoreML outputs match PyTorch float32."""
    print(f"\nVerifying CoreML against PyTorch ({n} shapes)...")
    coreml_model = ct.models.MLModel(str(coreml_path))
    torch.manual_seed(42)
    max_diff = 0.0

    for T in MEL_T_VALUES[:n]:
        mel = torch.randn(1, 128, T)
        with torch.no_grad():
            pt_out = pt_model(mel).numpy().flatten().astype(np.float32)

        cm_out = np.array(
            coreml_model.predict({"mel": mel.numpy()})["audio_embeddings"]
        ).flatten().astype(np.float32)

        cos = float(np.dot(pt_out, cm_out) / (np.linalg.norm(pt_out) * np.linalg.norm(cm_out) + 1e-10))
        diff = 1.0 - cos
        max_diff = max(max_diff, diff)
        tokens = T // 8 + (1 if T % 8 else 0)
        # Account for stride-2 conv formula: (T-1)//2+1 applied 3 times
        print(f"  T={T:4d}: cosine_sim={cos:.6f}")

    status = "PASS" if max_diff < 0.01 else "WARNING"
    print(f"  {status}: max (1-cosine_sim) = {max_diff:.6f}")
    return max_diff


def compile_mlpackage(output_dir, name):
    """Compile .mlpackage → .mlmodelc using xcrun."""
    pkg = output_dir / f"{name}.mlpackage"
    compiled = output_dir / f"{name}.mlmodelc"

    if compiled.exists():
        shutil.rmtree(compiled)

    print(f"Compiling {name}.mlpackage → {name}.mlmodelc...")
    result = subprocess.run(
        ["xcrun", "coremlcompiler", "compile", str(pkg), str(output_dir)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"  xcrun failed: {result.stderr.strip()}")
        print("  Falling back to Python compilation...")
        compiled_url = ct.utils.compile_model(str(pkg))
        shutil.move(str(compiled_url), str(compiled))

    if compiled.exists():
        shutil.rmtree(pkg)
        print(f"  Compiled to {compiled}")
    else:
        print(f"  ERROR: {compiled} not found after compilation")


# ── Main ──

def main():
    parser = argparse.ArgumentParser(description="Convert Qwen3-ASR encoder to CoreML")
    parser.add_argument("--model-id", default="Qwen/Qwen3-ASR-0.6B",
                        help="HuggingFace model ID (default: Qwen/Qwen3-ASR-0.6B)")
    parser.add_argument("--output-dir", default="./qwen3-asr-coreml",
                        help="Output directory (default: ./qwen3-asr-coreml)")
    parser.add_argument("--no-quantize", action="store_true",
                        help="Skip INT4 palettization")
    parser.add_argument("--compile", action="store_true",
                        help="Compile .mlpackage to .mlmodelc")
    parser.add_argument("--skip-verify", action="store_true",
                        help="Skip verification step")
    parser.add_argument("--upload", action="store_true",
                        help="Upload to HuggingFace")
    parser.add_argument("--repo-id", default="aufklarer/Qwen3-ASR-CoreML-INT4",
                        help="HuggingFace repo ID for upload")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Phase 1: Download weights ──
    print("=" * 60)
    print("Phase 1: Download audio_tower weights")
    print("=" * 60)
    audio_weights = download_encoder_weights(args.model_id)

    # ── Phase 2: Build PyTorch encoder and load weights ──
    print("\n" + "=" * 60)
    print("Phase 2: Build PyTorch encoder")
    print("=" * 60)

    model = Qwen3ASREncoder()
    load_weights(model, audio_weights)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,} ({total_params * 4 / 1024 / 1024:.1f} MB FP32)")

    del audio_weights
    gc.collect()

    # ── Phase 3: Trace ──
    print("\n" + "=" * 60)
    print("Phase 3: Trace model")
    print("=" * 60)

    example_mel = torch.randn(1, 128, 1000)
    with torch.no_grad():
        traced = torch.jit.trace(model, (example_mel,))

    # Sanity check
    with torch.no_grad():
        ref = model(example_mel)
        trc = traced(example_mel)
    diff = (ref - trc).abs().max().item()
    print(f"  Traced vs original max diff: {diff:.2e}")
    assert diff < 1e-5, f"Trace mismatch: {diff}"

    # ── Phase 4: Convert to CoreML ──
    print("\n" + "=" * 60)
    print("Phase 4: Convert to CoreML")
    print("=" * 60)

    mlmodel = convert_to_coreml(traced, quantize=not args.no_quantize)

    pkg_path = output_dir / "encoder.mlpackage"
    if pkg_path.exists():
        shutil.rmtree(pkg_path)
    mlmodel.save(str(pkg_path))

    size_mb = sum(f.stat().st_size for f in pkg_path.rglob("*") if f.is_file()) / 1024 / 1024
    print(f"  Saved encoder.mlpackage ({size_mb:.1f} MB)")

    del traced, mlmodel
    gc.collect()

    # ── Phase 5: Verify ──
    if not args.skip_verify:
        print("\n" + "=" * 60)
        print("Phase 5: Verify")
        print("=" * 60)
        verify(model, pkg_path)

    del model
    gc.collect()

    # ── Phase 6: Compile ──
    if args.compile:
        compile_mlpackage(output_dir, "encoder")

    # ── Save config ──
    config = {
        "model_type": "qwen3-asr-encoder-coreml",
        "source_model": args.model_id,
        "num_mel_bins": 128,
        "sample_rate": 16000,
        "hop_length": 160,
        "encoder_hidden": ENCODER_CONFIG["d_model"],
        "encoder_layers": ENCODER_CONFIG["num_layers"],
        "encoder_heads": ENCODER_CONFIG["num_heads"],
        "encoder_ffn": ENCODER_CONFIG["ffn_dim"],
        "output_dim": ENCODER_CONFIG["output_dim"],
        "conv_stride": 8,
        "enumerated_mel_lengths": MEL_T_VALUES,
        "quantization": "int4_palettize" if not args.no_quantize else "none",
        "compute_precision": "float16",
    }
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"\nSaved config.json")

    # Summary
    print(f"\nDone! Output in: {output_dir}")
    for f in sorted(output_dir.iterdir()):
        sz = sum(ff.stat().st_size for ff in f.rglob("*") if ff.is_file()) if f.is_dir() else f.stat().st_size
        print(f"  {f.name}: {sz / 1024 / 1024:.1f} MB")

    # ── Upload ──
    if args.upload:
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            api.create_repo(args.repo_id, exist_ok=True)
            print(f"\nUploading to {args.repo_id}...")
            api.upload_folder(folder_path=str(output_dir), repo_id=args.repo_id)
            print(f"Uploaded to https://huggingface.co/{args.repo_id}")
        except Exception as e:
            print(f"\nUpload failed: {e}")
            print(f"Upload manually: huggingface-cli upload {args.repo_id} {output_dir}")


if __name__ == "__main__":
    main()
