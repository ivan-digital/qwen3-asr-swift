#!/usr/bin/env python3
"""Convert Kokoro-82M PyTorch model to end-to-end CoreML models.

Downloads the pretrained Kokoro-82M model and converts it to end-to-end CoreML
.mlpackage models matching the Swift KokoroNetwork interface.

Each model variant takes:
  - input_ids [1, N]: phoneme token IDs (padded)
  - attention_mask [1, N]: 1 for real tokens, 0 for padding
  - ref_s [1, 256]: voice style embedding
  - random_phases [1, 9]: random phases for iSTFTNet vocoder

And outputs:
  - audio [1, 1, S]: generated waveform at 24kHz
  - audio_length_samples [1]: actual valid sample count
  - pred_dur [1, N]: predicted phoneme durations

CPU compatibility fix: Replaces torch.nonzero()-based phase correction in
CustomSTFT with element-wise torch.where() to eliminate data-dependent shapes
that crash the BNNS (CPU) backend.

Requires:
    pip install torch transformers coremltools numpy huggingface_hub

Usage:
    python scripts/convert_kokoro_coreml.py [--output OUTPUT_DIR]

Output:
    kokoro_24_10s.mlpackage  — End-to-end TTS, 242 tokens, ~10s max
    kokoro_24_15s.mlpackage  — End-to-end TTS, 242 tokens, ~15s max
    kokoro_21_5s.mlpackage   — End-to-end TTS, 124 tokens, ~5s max
    kokoro_21_10s.mlpackage  — End-to-end TTS, 168 tokens, ~10s max
    kokoro_21_15s.mlpackage  — End-to-end TTS, 249 tokens, ~15s max
    voices/                  — Per-voice JSON embeddings
    vocab_index.json         — Phoneme vocabulary
    config.json              — Model configuration
"""

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── coremltools Compatibility Patch ─────────────────────────────────────────
# coremltools doesn't handle the 'cs' (list-of-compilation-units) attribute
# type on TorchScript nodes. Monkey-patch to skip unsupported attribute types.

def _patch_coremltools_torchscript():
    """Fix coremltools crash on unknown TorchScript node attribute types.

    coremltools doesn't handle the 'cs' attribute kind on TorchScript nodes
    (introduced by newer transformers). We patch the classmethod to skip
    unsupported attribute types instead of crashing.
    """
    try:
        from coremltools.converters.mil.frontend.torch import internal_graph
        from coremltools.converters.mil.frontend.torch.internal_graph import (
            InternalTorchIRBlock, sanitize_op_kind,
        )

        _orig = internal_graph.InternalTorchIRNode.from_torchscript_node.__func__

        @classmethod
        def _safe_from_node(cls, node, parent):
            # Build attr dict, skipping unsupported attribute kinds
            attr = {}
            for name in node.attributeNames():
                kind = node.kindOf(name)
                accessor = getattr(node, kind, None)
                if accessor is not None:
                    attr[name] = accessor(name)

            if "value" not in attr:
                attr["value"] = None
            if len(list(node.outputs())) == 1 and next(node.outputs()).type().str() == "bool":
                attr["value"] = bool(attr["value"])

            inputs = [_input.debugName() for _input in node.inputs()]
            outputs = [output.debugName() for output in node.outputs()]
            kind = sanitize_op_kind(node.kind())
            name = outputs[0] if len(outputs) > 0 else kind

            internal_node = cls(
                name=name,
                kind=kind,
                parent=parent,
                inputs=inputs,
                outputs=outputs,
                attr=attr,
                blocks=None,
                model_hierarchy=node.getModuleHierarchy(),
            )
            internal_node.blocks = [
                InternalTorchIRBlock.from_torchscript_block(block=b, parent=internal_node)
                for b in node.blocks()
            ]
            return internal_node

        internal_graph.InternalTorchIRNode.from_torchscript_node = _safe_from_node
        print("  Patched coremltools for TorchScript compatibility")
    except Exception as e:
        print(f"  WARNING: Could not patch coremltools: {e}")

_patch_coremltools_torchscript()


def _patch_coremltools_cast():
    """Fix numpy 2.x compatibility in coremltools _cast function.

    numpy 2.x rejects int(np.array([242])) for non-0d arrays.
    The fix: use .item() to extract the scalar before casting.
    """
    try:
        from coremltools.converters.mil.frontend.torch import ops
        from coremltools.converters.mil.frontend.torch.ops import _get_inputs
        from coremltools.converters.mil import Builder as mb

        def _cast_fixed(context, node, dtype, dtype_name):
            inputs = _get_inputs(context, node, expected=1)
            x = inputs[0]
            if not (len(x.shape) == 0 or np.all([d == 1 for d in x.shape])):
                raise ValueError("input to cast must be either a scalar or a length 1 tensor")
            if x.can_be_folded_to_const():
                val = x.val
                if hasattr(val, 'item'):
                    val = val.item()
                if not isinstance(val, dtype):
                    res = mb.const(val=dtype(val), name=node.name)
                else:
                    res = x
            elif len(x.shape) > 0:
                x = mb.squeeze(x=x, name=node.name + "_item")
                res = mb.cast(x=x, dtype=dtype_name, name=node.name)
            else:
                res = mb.cast(x=x, dtype=dtype_name, name=node.name)
            context.add(res, node.name)

        ops._cast = _cast_fixed
    except Exception as e:
        print(f"  WARNING: Could not patch _cast: {e}")

_patch_coremltools_cast()


def _register_missing_torch_ops():
    """Register missing PyTorch op converters in coremltools."""
    try:
        from coremltools.converters.mil.frontend.torch.ops import _get_inputs
        from coremltools.converters.mil import Builder as mb
        from coremltools.converters.mil.frontend.torch import register_torch_op

        @register_torch_op
        def multiply(context, node):
            inputs = _get_inputs(context, node, expected=2)
            res = mb.mul(x=inputs[0], y=inputs[1], name=node.name)
            context.add(res)

    except Exception as e:
        print(f"  WARNING: Could not register missing ops: {e}")

_register_missing_torch_ops()


# ─── Constants ────────────────────────────────────────────────────────────────

SAMPLE_RATE = 24000
STYLE_DIM = 256
NUM_PHASES = 9

# Model buckets matching Swift Configuration.swift ModelBucket enum.
# (name, max_tokens, max_samples)
MODEL_BUCKETS = [
    ("kokoro_21_5s",  124, 175_800),
    ("kokoro_21_10s", 168, 253_200),
    ("kokoro_21_15s", 249, 372_600),
    ("kokoro_24_10s", 242, 240_000),
    ("kokoro_24_15s", 242, 360_000),
]


# ─── CPU Compatibility Patch ─────────────────────────────────────────────────

def patch_custom_stft_for_cpu(custom_stft_module):
    """Monkey-patch CustomSTFT.transform to use torch.where instead of
    boolean indexing (which traces to non_zero + scatter_nd and breaks CPU).

    The original code:
        correction_mask = (imag_out == 0) & (real_out < 0)
        phase[correction_mask] = torch.pi

    Traces to non_zero (data-dependent shape [?, 3]) → scatter_nd, which
    the BNNS CPU backend cannot execute.

    The fix uses element-wise torch.where with identical semantics but
    fixed tensor shapes throughout:
        correction_mask = (imag_out == 0) & (real_out < 0)
        phase = torch.where(correction_mask, torch.tensor(math.pi), phase)
    """
    CustomSTFT = custom_stft_module.CustomSTFT

    def _patched_transform(self, waveform):
        if self.center:
            pad_len = self.n_fft // 2
            waveform = F.pad(waveform, (pad_len, pad_len), mode=self.pad_mode)

        x = waveform.unsqueeze(1)
        real_out = F.conv1d(x, self.weight_forward_real, bias=None,
                            stride=self.hop_length, padding=0)
        imag_out = F.conv1d(x, self.weight_forward_imag, bias=None,
                            stride=self.hop_length, padding=0)

        magnitude = torch.sqrt(real_out**2 + imag_out**2 + 1e-14)
        phase = torch.atan2(imag_out, real_out)

        # CPU-compatible phase correction: element-wise instead of boolean indexing.
        # Original: phase[correction_mask] = torch.pi  → nonzero + scatter_nd
        # Fixed:    torch.where (element-wise, fixed shapes)
        correction_mask = (imag_out == 0) & (real_out < 0)
        phase = torch.where(correction_mask,
                            torch.tensor(math.pi, dtype=phase.dtype, device=phase.device),
                            phase)
        return magnitude, phase

    CustomSTFT.transform = _patched_transform
    print("  Applied CustomSTFT CPU compatibility patch (torch.where)")


# ─── Model Download & Setup ─────────────────────────────────────────────────

def download_kokoro_source(cache_dir: Path):
    """Download kokoro source from GitHub if not installed as package."""
    kokoro_dir = cache_dir / "kokoro-src" / "kokoro"
    if kokoro_dir.exists() and (kokoro_dir / "model.py").exists():
        print(f"  Using cached source: {kokoro_dir}")
        return kokoro_dir.parent

    kokoro_dir.mkdir(parents=True, exist_ok=True)
    base_url = "https://raw.githubusercontent.com/hexgrad/kokoro/main/kokoro"
    files = ["model.py", "istftnet.py", "modules.py", "custom_stft.py"]

    for fname in files:
        url = f"{base_url}/{fname}"
        dest = kokoro_dir / fname
        print(f"    Downloading {fname}...")
        subprocess.run(["curl", "-sL", "-o", str(dest), url], check=True)

    # Write minimal __init__.py that only imports what we need (avoids
    # pulling in KPipeline and its heavy dependency chain).
    init_path = kokoro_dir / "__init__.py"
    init_path.write_text("from .model import KModel\n")

    print(f"  Downloaded kokoro source to {kokoro_dir}")
    return kokoro_dir.parent


def download_kokoro_weights(cache_dir: Path):
    """Download Kokoro-82M weights from HuggingFace."""
    model_dir = cache_dir / "Kokoro-82M"
    if model_dir.exists():
        pth_files = list(model_dir.glob("*.pth"))
        if pth_files:
            print(f"  Using cached weights: {model_dir}")
            return model_dir

    model_dir.mkdir(parents=True, exist_ok=True)
    print("  Downloading Kokoro-82M from HuggingFace...")

    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            "hexgrad/Kokoro-82M",
            local_dir=str(model_dir),
            ignore_patterns=["*.md", "*.txt", "samples/*", "eval/*"],
        )
        print(f"  Downloaded to: {model_dir}")
        return model_dir
    except ImportError:
        print("ERROR: huggingface_hub required. Install: pip install huggingface_hub")
        sys.exit(1)


# ─── Voice Embedding Extraction ──────────────────────────────────────────────

def extract_voices(model_dir: Path, output_dir: Path):
    """Extract voice style embeddings to per-voice JSON files."""
    voices_dir = model_dir / "voices"
    if not voices_dir.exists():
        print("  WARNING: No voices directory found")
        return 0

    out_voices_dir = output_dir / "voices"
    out_voices_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for pt_file in sorted(voices_dir.glob("*.pt")):
        name = pt_file.stem
        embedding = torch.load(pt_file, map_location="cpu", weights_only=True)
        if isinstance(embedding, torch.Tensor):
            vec = embedding.flatten()
        elif isinstance(embedding, dict) and "style" in embedding:
            vec = embedding["style"].flatten()
        else:
            continue

        # Voice embeddings are [256] (global) or [510, 256] (per-position).
        # The model's ref_s input is [1, 256], so extract only the first 256 dims.
        emb_list = vec[:STYLE_DIM].tolist()

        # Save as per-voice JSON matching Swift's expected format
        voice_json = {"embedding": emb_list}
        with open(out_voices_dir / f"{name}.json", "w") as f:
            json.dump(voice_json, f)
        count += 1

    print(f"  Saved {count} voice embeddings to {out_voices_dir}")
    return count


# ─── Vocabulary Extraction ────────────────────────────────────────────────────

def extract_vocab(model_dir: Path, output_dir: Path):
    """Extract phoneme vocabulary as vocab_index.json."""
    config_path = model_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        if "vocab" in config:
            vocab = config["vocab"]
            output_path = output_dir / "vocab_index.json"
            with open(output_path, "w") as f:
                json.dump(vocab, f, ensure_ascii=False, indent=2)
            print(f"  Saved vocabulary ({len(vocab)} entries) to {output_path}")
            return vocab

    print("  WARNING: No vocab found in config.json")
    return {}


# ─── End-to-End Wrapper ──────────────────────────────────────────────────────

class KokoroEndToEnd(nn.Module):
    """Wraps KModel for end-to-end CoreML tracing with fixed shapes.

    Matches the Swift KokoroNetwork interface:
      Inputs:  input_ids[1,N], attention_mask[1,N], ref_s[1,256], random_phases[1,9]
      Outputs: audio[1,1,S], audio_length_samples[1], pred_dur[1,N]

    The alignment step (duration-based expansion) uses fixed-size matrix
    multiplication to avoid data-dependent shapes. Output audio is padded
    to max_samples with audio_length_samples indicating valid length.
    """

    def __init__(self, kmodel, max_tokens: int, max_samples: int, speed: float = 1.0):
        super().__init__()
        self.kmodel = kmodel
        self.max_tokens = max_tokens
        self.max_samples = max_samples
        self.speed = speed

        # Compute max frames from max samples.
        # iSTFTNet hop_size=5, n_fft=20, so frames ≈ max_samples / (hop * upsample)
        # The vocoder upsamples by prod(upsample_rates) * gen_istft_hop_size
        # For Kokoro: upsample_rates=[10,6], hop_size=4 → total = 10*6*4 = 240
        # But actual output includes conv padding, so frame count = ceil(max_samples / 240)
        # We derive max_frames from the model's own computation during tracing.

    def forward(self, input_ids, attention_mask, ref_s, random_phases):
        # input_ids: [1, max_tokens] int64
        # attention_mask: [1, max_tokens] int32
        # ref_s: [1, 256] float32
        # random_phases: [1, 9] float32 (unused by model, reserved for vocoder variation)

        kmodel = self.kmodel

        input_lengths = attention_mask.sum(dim=-1).long()  # [1]

        text_mask = torch.arange(self.max_tokens, device=input_ids.device).unsqueeze(0)
        text_mask = torch.gt(text_mask + 1, input_lengths.unsqueeze(1))

        # BERT text encoding
        bert_dur = kmodel.bert(input_ids, attention_mask=attention_mask)
        d_en = kmodel.bert_encoder(bert_dur).transpose(-1, -2)  # [1, 512, N]

        # Style split: first 128 dims for decoder, last 128 for predictor
        s = ref_s[:, 128:]  # [1, 128]

        # Duration prediction
        d = kmodel.predictor.text_encoder(d_en, s, input_lengths, text_mask)
        x, _ = kmodel.predictor.lstm(d)
        duration = kmodel.predictor.duration_proj(x)
        duration = torch.sigmoid(duration).sum(axis=-1) / self.speed  # [1, N]
        pred_dur = torch.round(duration).clamp(min=0)  # [1, N]
        # Mask padding positions
        pred_dur = pred_dur * (~text_mask).float()

        # Fixed-shape alignment via comparison-based masking.
        # Instead of repeat_interleave (data-dependent size), we build
        # an alignment matrix [1, N, max_frames] using cumsum + comparisons.
        cumsum = torch.cumsum(pred_dur, dim=-1)  # [1, N]
        total_frames = cumsum[:, -1:]  # [1, 1] — total predicted frames

        # Determine max_frames from model architecture
        # Vocoder total upsample factor = prod(upsample_rates) * gen_istft_hop_size
        # For Kokoro default: 10 * 6 * 4 = 240 samples per frame
        upsample_factor = 240
        max_frames = self.max_samples // upsample_factor

        lower = torch.cat([
            torch.zeros(1, 1, device=pred_dur.device),
            cumsum[:, :-1]
        ], dim=-1)  # [1, N]

        frame_indices = torch.arange(
            max_frames, device=pred_dur.device, dtype=pred_dur.dtype
        ).unsqueeze(0)  # [1, max_frames]

        # [1, N, 1] vs [1, 1, max_frames] → [1, N, max_frames]
        aln = ((frame_indices.unsqueeze(1) >= lower.unsqueeze(-1)) &
               (frame_indices.unsqueeze(1) < cumsum.unsqueeze(-1))).float()

        # Aligned features
        en = d.transpose(-1, -2) @ aln  # [1, 512, max_frames]
        F0_pred, N_pred = kmodel.predictor.F0Ntrain(en, s)

        t_en = kmodel.text_encoder(input_ids, input_lengths, text_mask)
        asr = t_en @ aln  # [1, 512, max_frames]

        # Decoder → audio
        audio = kmodel.decoder(asr, F0_pred, N_pred, ref_s[:, :128])  # [1, 1, ~S]

        # Pad or trim to exact max_samples
        if audio.shape[-1] >= self.max_samples:
            audio = audio[:, :, :self.max_samples]
        else:
            pad_len = self.max_samples - audio.shape[-1]
            audio = F.pad(audio, (0, pad_len))

        # Valid audio length in samples
        audio_length = (total_frames * upsample_factor).clamp(max=self.max_samples).int()

        return audio, audio_length.squeeze(-1), pred_dur

    @staticmethod
    def trace_and_convert(kmodel, bucket_name, max_tokens, max_samples, output_dir, speed=1.0):
        """Trace and convert one bucket to CoreML."""
        import coremltools as ct

        wrapper = KokoroEndToEnd(kmodel, max_tokens, max_samples, speed)
        wrapper.eval()

        # Example inputs for tracing
        example_ids = torch.zeros(1, max_tokens, dtype=torch.long)
        example_mask = torch.ones(1, max_tokens, dtype=torch.int32)
        example_ref_s = torch.randn(1, STYLE_DIM)
        example_phases = torch.randn(1, NUM_PHASES)

        print(f"    Tracing {bucket_name} (tokens={max_tokens}, samples={max_samples})...")
        with torch.no_grad():
            traced = torch.jit.trace(wrapper, (example_ids, example_mask, example_ref_s, example_phases))

        print(f"    Converting to CoreML...")
        mlmodel = ct.convert(
            traced,
            inputs=[
                ct.TensorType(name="input_ids", shape=(1, max_tokens), dtype=np.int32),
                ct.TensorType(name="attention_mask", shape=(1, max_tokens), dtype=np.int32),
                ct.TensorType(name="ref_s", shape=(1, STYLE_DIM), dtype=np.float32),
                ct.TensorType(name="random_phases", shape=(1, NUM_PHASES), dtype=np.float32),
            ],
            outputs=[
                ct.TensorType(name="audio"),
                ct.TensorType(name="audio_length_samples"),
                ct.TensorType(name="pred_dur"),
            ],
            compute_precision=ct.precision.FLOAT32,
            minimum_deployment_target=ct.target.iOS17,
        )

        out_path = output_dir / f"{bucket_name}.mlpackage"
        mlmodel.save(str(out_path))
        print(f"    Saved: {out_path}")
        return out_path


# ─── Config Generation ────────────────────────────────────────────────────────

def save_config(output_dir: Path, vocab_size: int, num_voices: int):
    """Save model configuration as config.json."""
    config = {
        "sampleRate": SAMPLE_RATE,
        "maxPhonemeLength": 510,
        "styleDim": STYLE_DIM,
        "numPhases": NUM_PHASES,
        "numVoices": num_voices,
        "languages": ["en", "fr", "es", "ja", "zh", "hi", "pt", "ko"],
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Saved config.json")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Convert Kokoro-82M to end-to-end CoreML")
    parser.add_argument("--output", type=str, default="kokoro-coreml",
                        help="Output directory")
    parser.add_argument("--cache-dir", type=str, default=None,
                        help="Cache directory for downloads")
    parser.add_argument("--buckets", type=str, nargs="*", default=None,
                        help="Specific buckets to convert (e.g. kokoro_24_10s)")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Speech speed factor (default: 1.0)")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = Path(args.cache_dir) if args.cache_dir else Path.home() / ".cache" / "kokoro-convert"

    # Step 1: Download model weights
    print("Step 1: Download Kokoro-82M weights")
    model_dir = download_kokoro_weights(cache_dir)

    # Step 2: Ensure kokoro source is available
    print("\nStep 2: Setup kokoro source")
    kokoro_available = False
    try:
        from kokoro.custom_stft import CustomSTFT as _test
        kokoro_available = True
        print("  Using installed kokoro package")
    except (ImportError, Exception):
        src_dir = download_kokoro_source(cache_dir)
        sys.path.insert(0, str(src_dir))
        print(f"  Added {src_dir} to sys.path")

    # Step 3: Apply patches BEFORE loading model.
    # Must patch CustomSTFT before any model instantiation so that
    # the Generator's stft attribute uses the fixed transform.
    print("\nStep 3: Apply compatibility patches")
    import importlib
    custom_stft_mod = importlib.import_module("kokoro.custom_stft")
    patch_custom_stft_for_cpu(custom_stft_mod)

    # Patch ALBERT embeddings to avoid dynamic slice that coremltools can't convert.
    # The original: position_ids = self.position_ids[:, :seq_length]
    # Traces to a slice with inhomogeneous shape args. Fix: pre-register fixed-size
    # position_ids in forward so the slice is a no-op during tracing.
    try:
        from transformers.models.albert.modeling_albert import AlbertEmbeddings

        _orig_albert_forward = AlbertEmbeddings.forward

        def _patched_albert_forward(self, input_ids=None, token_type_ids=None,
                                     position_ids=None, inputs_embeds=None,
                                     past_key_values_length=0):
            if input_ids is not None:
                seq_length = input_ids.shape[1]
            else:
                seq_length = inputs_embeds.shape[1]

            device = input_ids.device if input_ids is not None else inputs_embeds.device

            if position_ids is None:
                position_ids = torch.arange(seq_length, device=device)
                position_ids = position_ids.unsqueeze(0).expand(1, -1)

            if token_type_ids is None:
                token_type_ids = torch.zeros(1, seq_length, dtype=torch.long, device=device)

            if inputs_embeds is None:
                inputs_embeds = self.word_embeddings(input_ids)

            position_embeddings = self.position_embeddings(position_ids)
            token_type_embeddings = self.token_type_embeddings(token_type_ids)

            embeddings = inputs_embeds + position_embeddings + token_type_embeddings
            embeddings = self.LayerNorm(embeddings)
            embeddings = self.dropout(embeddings)
            return embeddings

        AlbertEmbeddings.forward = _patched_albert_forward
        print("  Patched AlbertEmbeddings for CoreML tracing compatibility")
    except Exception as e:
        print(f"  WARNING: Could not patch AlbertEmbeddings: {e}")

    # Patch Generator to always use CustomSTFT (disable_complex=True).
    # TorchSTFT uses torch.angle() and complex ops that coremltools can't convert.
    try:
        kokoro_istft = importlib.import_module("kokoro.istftnet")
        _orig_generator_init = kokoro_istft.Generator.__init__

        def _patched_generator_init(self, *args, **kwargs):
            kwargs['disable_complex'] = True
            _orig_generator_init(self, *args, **kwargs)

        kokoro_istft.Generator.__init__ = _patched_generator_init
        print("  Patched Generator to use CustomSTFT (disable_complex=True)")
    except Exception as e:
        print(f"  WARNING: Could not patch Generator: {e}")

    # Patch istftnet AdainResBlk1d to use float tensor in rsqrt.
    # Original: torch.rsqrt(torch.tensor(2)) → int32 tensor, coremltools rejects.
    # Fix: torch.rsqrt(torch.tensor(2.0)) → float32 tensor.
    try:
        kokoro_istft = importlib.import_module("kokoro.istftnet")
        _orig_resblk_forward = kokoro_istft.AdainResBlk1d.forward

        def _patched_resblk_forward(self, x, s):
            out = self._residual(x, s)
            out = (out + self._shortcut(x)) * torch.rsqrt(torch.tensor(2.0))
            return out

        kokoro_istft.AdainResBlk1d.forward = _patched_resblk_forward
        print("  Patched AdainResBlk1d rsqrt for float32")
    except Exception as e:
        print(f"  WARNING: Could not patch AdainResBlk1d: {e}")

    # Step 4: Load PyTorch model
    print("\nStep 4: Load PyTorch model")
    from kokoro.model import KModel
    kmodel = KModel()
    kmodel.eval()
    print("  Loaded KModel")

    # Step 5: Extract voice embeddings
    print("\nStep 5: Extract voice embeddings")
    num_voices = extract_voices(model_dir, output_dir)

    # Step 6: Extract vocabulary
    print("\nStep 6: Extract vocabulary")
    vocab = extract_vocab(model_dir, output_dir)

    # Step 7: Convert end-to-end models
    print("\nStep 7: Convert end-to-end CoreML models")
    buckets = MODEL_BUCKETS
    if args.buckets:
        buckets = [(n, t, s) for n, t, s in MODEL_BUCKETS if n in args.buckets]

    mlpackages = []
    for bucket_name, max_tokens, max_samples in buckets:
        print(f"\n  Converting {bucket_name}...")
        path = KokoroEndToEnd.trace_and_convert(
            kmodel, bucket_name, max_tokens, max_samples, output_dir, args.speed)
        mlpackages.append(path)

    # Step 8: Save configuration
    print("\nStep 8: Save configuration")
    save_config(output_dir, vocab_size=len(vocab), num_voices=num_voices)

    # Step 9: Validate
    print("\nStep 9: Validate CoreML models")
    import coremltools as ct
    for mlpackage in mlpackages:
        try:
            model = ct.models.MLModel(str(mlpackage))
            spec = model.get_spec()
            inputs = [inp.name for inp in spec.description.input]
            outputs = [out.name for out in spec.description.output]
            print(f"  {mlpackage.name}: inputs={inputs}, outputs={outputs}")
        except Exception as e:
            print(f"  {mlpackage.name}: validation error: {e}")

    print(f"\nDone! Output: {output_dir}/")
    print("Compile with xcrun coremlcompiler, then upload to HuggingFace.")


if __name__ == "__main__":
    main()
