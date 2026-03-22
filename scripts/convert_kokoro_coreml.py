#!/usr/bin/env python3
"""Convert Kokoro-82M PyTorch model to CoreML.

Downloads Kokoro-82M, converts to a single end-to-end CoreML model
(tokens → audio), with optional INT8 palettization for iOS.

The model is compiled with a fixed input/output shape (one bucket).
Default: 5s max output (124 tokens) — sufficient for voice assistants.

Requires:
    pip install torch kokoro coremltools numpy huggingface_hub

Usage:
    # Default: single 5s bucket, FP16
    python scripts/convert_kokoro_coreml.py

    # INT8 palettized for iOS (~80MB instead of ~310MB)
    python scripts/convert_kokoro_coreml.py --palettize 8

    # Custom duration bucket
    python scripts/convert_kokoro_coreml.py --max-tokens 242 --max-samples 240000 --name kokoro_24_10s

    # All standard buckets (for CLI/server use)
    python scripts/convert_kokoro_coreml.py --all-buckets

Output:
    kokoro_21_5s.mlpackage   — End-to-end TTS model (ANE)
    G2PEncoder.mlpackage     — G2P encoder for OOV words
    G2PDecoder.mlpackage     — G2P decoder for OOV words
    voices/                  — Voice style embeddings (per-voice JSON)
    vocab_index.json         — Phoneme vocabulary
    config.json              — Model configuration
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

SAMPLE_RATE = 24000
MAX_PHONEME_LEN = 510
STYLE_DIM = 256

# Standard buckets: (name, max_tokens, max_samples)
STANDARD_BUCKETS = [
    ("kokoro_21_5s", 124, 175_800),
    ("kokoro_21_10s", 249, 253_200),
    ("kokoro_21_15s", 249, 372_600),
]


# ─── Model Loading ──────────────────────────────────────────────────────────

def load_kokoro_model(cache_dir: Path):
    """Load Kokoro-82M model, handling both v0.19 and v1.0 checkpoint formats."""
    model_dir = cache_dir / "Kokoro-82M"
    if not model_dir.exists():
        from huggingface_hub import snapshot_download
        print("  Downloading Kokoro-82M from HuggingFace...")
        snapshot_download("hexgrad/Kokoro-82M", local_dir=str(model_dir),
                          ignore_patterns=["*.md", "*.txt"])

    # Try kokoro package first (handles all versions)
    try:
        from kokoro import KModel
        model = KModel()
        model.eval()
        print("  Loaded via kokoro package")
        return model, model_dir
    except (ImportError, Exception) as e:
        print(f"  kokoro package unavailable ({e}), loading checkpoint directly")

    # Direct checkpoint loading
    pth_files = sorted(model_dir.glob("*.pth"))
    if not pth_files:
        print("ERROR: No .pth checkpoint found. Install kokoro: pip install kokoro")
        sys.exit(1)

    ckpt_path = pth_files[-1]  # Latest version
    print(f"  Loading checkpoint: {ckpt_path.name}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict) and not hasattr(ckpt, "forward"):
        # v1.0 format: dict with component state dicts
        # Need kokoro package to reconstruct the model
        print("ERROR: v1.0 checkpoint requires the kokoro package.")
        print("  Install: pip install kokoro")
        print("  Or use an older checkpoint (kokoro-v0_19.pth)")
        sys.exit(1)

    if hasattr(ckpt, "eval"):
        ckpt.eval()
    print(f"  Loaded model from {ckpt_path.name}")
    return ckpt, model_dir


# ─── End-to-End Model Wrapper ───────────────────────────────────────────────

class KokoroE2EWrapper(nn.Module):
    """Wraps Kokoro for end-to-end tracing: tokens → audio."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, ref_s, random_phases):
        return self.model(input_ids, attention_mask, ref_s, random_phases)


# ─── CoreML Conversion ──────────────────────────────────────────────────────

def convert_bucket(model, name: str, max_tokens: int, max_samples: int,
                   output_dir: Path, palettize_bits: int = 0):
    """Convert one bucket to CoreML, optionally palettize."""
    import coremltools as ct

    print(f"  Converting {name} (max {max_tokens} tokens, {max_samples} samples)...")

    wrapper = KokoroE2EWrapper(model)
    wrapper.eval()

    example_ids = torch.zeros(1, max_tokens, dtype=torch.long)
    example_mask = torch.ones(1, max_tokens, dtype=torch.long)
    example_style = torch.randn(1, STYLE_DIM)
    example_phases = torch.randn(1, 9)

    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (example_ids, example_mask, example_style, example_phases))

    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="input_ids", shape=(1, max_tokens), dtype=np.int32),
            ct.TensorType(name="attention_mask", shape=(1, max_tokens), dtype=np.int32),
            ct.TensorType(name="ref_s", shape=(1, STYLE_DIM), dtype=np.float32),
            ct.TensorType(name="random_phases", shape=(1, 9), dtype=np.float32),
        ],
        outputs=[
            ct.TensorType(name="audio"),
            ct.TensorType(name="audio_length_samples"),
            ct.TensorType(name="pred_dur"),
        ],
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS16,
    )

    # Palettize if requested
    if palettize_bits > 0:
        from coremltools.optimize.coreml import (
            OpPalettizerConfig, OptimizationConfig, palettize_weights)
        print(f"    Palettizing to INT{palettize_bits}...")
        op_config = OpPalettizerConfig(mode="kmeans", nbits=palettize_bits)
        config = OptimizationConfig(global_config=op_config)
        mlmodel = palettize_weights(mlmodel, config)

    pkg_path = output_dir / f"{name}.mlpackage"
    if pkg_path.exists():
        shutil.rmtree(pkg_path)
    mlmodel.save(str(pkg_path))

    # Compile to .mlmodelc
    compiled = ct.utils.compile_model(str(pkg_path))
    compiled_path = output_dir / f"{name}.mlmodelc"
    if compiled_path.exists():
        shutil.rmtree(compiled_path)
    shutil.move(str(compiled), str(compiled_path))
    shutil.rmtree(pkg_path)  # Remove mlpackage, keep compiled

    size_mb = sum(f.stat().st_size for f in compiled_path.rglob("*") if f.is_file()) / 1024 / 1024
    print(f"    Done: {compiled_path.name} ({size_mb:.1f} MB)")


# ─── Voice & Vocab Extraction ───────────────────────────────────────────────

def extract_voices(model_dir: Path, output_dir: Path):
    """Extract per-voice style embeddings to individual JSON files."""
    voices_dir = model_dir / "voices"
    out_voices = output_dir / "voices"
    out_voices.mkdir(parents=True, exist_ok=True)

    count = 0
    for pt_file in sorted(voices_dir.glob("*.pt")):
        name = pt_file.stem
        emb = torch.load(pt_file, map_location="cpu", weights_only=True)
        if isinstance(emb, torch.Tensor):
            vec = emb.flatten().tolist()
        elif isinstance(emb, dict) and "style" in emb:
            vec = emb["style"].flatten().tolist()
        else:
            continue

        with open(out_voices / f"{name}.json", "w") as f:
            json.dump(vec, f)
        count += 1

    print(f"  Extracted {count} voice embeddings")
    return count


def extract_vocab(model_dir: Path, output_dir: Path):
    """Extract phoneme vocabulary."""
    config_path = model_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        if "vocab" in config:
            out = output_dir / "vocab_index.json"
            with open(out, "w") as f:
                json.dump(config["vocab"], f, ensure_ascii=False)
            print(f"  Saved vocab_index.json ({len(config['vocab'])} entries)")
            return len(config["vocab"])

    print("  WARNING: vocab not found in config.json")
    return 0


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Convert Kokoro-82M to CoreML")
    parser.add_argument("--output", type=str, default="kokoro-coreml",
                        help="Output directory")
    parser.add_argument("--cache-dir", type=str, default=None,
                        help="Cache directory for model download")
    parser.add_argument("--palettize", type=int, choices=[4, 8], default=0,
                        help="Post-training palettization (4=INT4, 8=INT8)")
    parser.add_argument("--all-buckets", action="store_true",
                        help="Convert all standard buckets (default: only 5s)")
    parser.add_argument("--max-tokens", type=int, default=124,
                        help="Max input tokens for single bucket (default: 124)")
    parser.add_argument("--max-samples", type=int, default=175_800,
                        help="Max output audio samples (default: 175800 = ~7.3s)")
    parser.add_argument("--name", type=str, default="kokoro_21_5s",
                        help="Model name for single bucket (default: kokoro_21_5s)")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir) if args.cache_dir else Path.home() / ".cache" / "kokoro-convert"

    print("Step 1: Load Kokoro-82M")
    model, model_dir = load_kokoro_model(cache_dir)

    print("\nStep 2: Extract voices")
    num_voices = extract_voices(model_dir, output_dir)

    print("\nStep 3: Extract vocabulary")
    vocab_size = extract_vocab(model_dir, output_dir)

    print("\nStep 4: Convert to CoreML")
    if args.all_buckets:
        for name, max_tokens, max_samples in STANDARD_BUCKETS:
            convert_bucket(model, name, max_tokens, max_samples,
                           output_dir, args.palettize)
    else:
        convert_bucket(model, args.name, args.max_tokens, args.max_samples,
                       output_dir, args.palettize)

    print("\nStep 5: Save config")
    config = {
        "sampleRate": SAMPLE_RATE,
        "vocabSize": vocab_size,
        "maxPhonemeLength": MAX_PHONEME_LEN,
        "styleDim": STYLE_DIM,
        "numVoices": num_voices,
        "languages": ["en", "fr", "es", "ja", "zh", "hi", "pt", "ko"],
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nDone! Output: {output_dir}/")
    for f in sorted(output_dir.iterdir()):
        if f.is_dir():
            size = sum(ff.stat().st_size for ff in f.rglob("*") if ff.is_file())
        else:
            size = f.stat().st_size
        print(f"  {f.name}: {size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
