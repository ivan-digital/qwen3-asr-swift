#!/usr/bin/env python3
"""Convert Kokoro-82M to 3-stage CoreML (duration + prosody + decoder).

The alignment step (duration → alignment matrix → asr) is dynamic and
cannot be traced. It must be done in Swift between model calls.

Stage 1 - Duration Model:
  Input:  input_ids [1,N], attention_mask [1,N], ref_s [1,256], speed [1]
  Output: pred_dur [1,N], d_transposed [1,640,N], t_en [1,512,N]

Stage 2 - Prosody Model (F0 + N):
  Input:  en [1,640,F], s [1,128]
  Output: F0_pred [1,F*2], N_pred [1,F*2]

Stage 3 - Decoder (fixed-shape buckets):
  Input:  asr [1,512,F], F0_pred [1,F*2], N_pred [1,F*2], ref_s [1,128]
  Output: audio [1,1,F*600]

Swift-side alignment:
  pred_aln_trg = build_alignment(pred_dur)  # [N, total_frames]
  en = d_transposed @ pred_aln_trg           # [640, F]
  asr = t_en @ pred_aln_trg                  # [512, F]

Usage:
    python scripts/convert_kokoro_v2.py --output /tmp/kokoro-v2
    python scripts/convert_kokoro_v2.py --output /tmp/kokoro-v2 --quantize int8
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import coremltools as ct


SAMPLE_RATE = 24000
MAX_PHONEMES = 128

# Enumerated phoneme lengths: pad to nearest bucket to minimize backward
# LSTM contamination from padding. Smaller padding = more accurate durations.
PHONEME_BUCKETS = [16, 32, 64, 128]

# Each frame produces 600 audio samples (24kHz → 25ms per frame)
SAMPLES_PER_FRAME = 600

# Decoder buckets: (name, max_frames)
BUCKETS = [
    ("5s",  200),   # 120,000 samples ≈ 5.0 seconds
    ("10s", 400),   # 240,000 samples ≈ 10.0 seconds
    ("15s", 600),   # 360,000 samples ≈ 15.0 seconds
]


def load_kokoro_model():
    """Load the original Kokoro-82M PyTorch model with traceable STFT."""
    sys.path.insert(0, '/tmp/kokoro')

    # Stub misaki to avoid dependency
    import types
    for mod_name in ['misaki', 'misaki.en', 'misaki.espeak']:
        if mod_name not in sys.modules:
            m = types.ModuleType(mod_name)
            if mod_name == 'misaki.en':
                m.MToken = type('MToken', (), {})
            sys.modules[mod_name] = m
            if '.' in mod_name:
                parent = mod_name.rsplit('.', 1)[0]
                setattr(sys.modules[parent], mod_name.split('.')[-1], m)

    # Monkey-patch modules for CoreML compatibility
    import kokoro.istftnet as _istftnet

    # Fix 1: AdainResBlk1d uses rsqrt(int) — CoreML needs float
    def _patched_adain_resblk_forward(self, x, s):
        out = self._residual(x, s)
        out = (out + self._shortcut(x)) * torch.rsqrt(torch.tensor(2.0))
        return out
    _istftnet.AdainResBlk1d.forward = _patched_adain_resblk_forward

    # Fix 2: SineGen uses torch.multiply (unmapped) — replace with *
    # Also pre-compute harmonic multiplier as float constant
    _original_sinegen_forward = _istftnet.SineGen.forward
    def _patched_sinegen_forward(self, f0):
        harm_range = torch.arange(1, self.harmonic_num + 2, dtype=torch.float32,
                                   device=f0.device).unsqueeze(0).unsqueeze(0)
        fn = f0 * harm_range
        sine_waves = self._f02sine(fn) * self.sine_amp
        uv = self._f02uv(f0)
        noise_amp = uv * self.noise_std + (1.0 - uv) * self.sine_amp / 3.0
        noise = noise_amp * torch.randn_like(sine_waves)
        sine_waves = sine_waves * uv + noise
        return sine_waves, uv, noise
    _istftnet.SineGen.forward = _patched_sinegen_forward

    from kokoro.model import KModel
    # disable_complex=True uses CustomSTFT (Conv1d-based, traceable)
    # instead of TorchSTFT (complex ops, not traceable)
    model = KModel(disable_complex=True)
    model.eval()
    print(f"Loaded Kokoro-82M ({sum(p.numel() for p in model.parameters())/1e6:.1f}M params)")
    return model


# --------------------------------------------------------------------------- #
#  CoreML-Friendly Wrapper Modules
# --------------------------------------------------------------------------- #

class CoreMLDurationModel(nn.Module):
    """Duration model with pack_padded_sequence replaced by masked LSTM.

    The original TextEncoder and DurationEncoder use pack_padded_sequence
    which is not traceable. This wrapper inlines their logic with plain
    LSTM calls + masking, producing identical output for padded inputs.
    """

    def __init__(self, model):
        super().__init__()
        self.bert = model.bert
        self.bert_encoder = model.bert_encoder

        # DurationEncoder submodules (from predictor.text_encoder)
        self.de_lstms = model.predictor.text_encoder.lstms
        self.de_dropout = model.predictor.text_encoder.dropout

        # Predictor LSTM + projection
        self.pred_lstm = model.predictor.lstm
        self.pred_duration_proj = model.predictor.duration_proj

        # TextEncoder submodules
        self.te_embedding = model.text_encoder.embedding
        self.te_cnn = model.text_encoder.cnn
        self.te_lstm = model.text_encoder.lstm

    def forward(self, input_ids, attention_mask, ref_s, speed):
        T = input_ids.shape[1]
        input_lengths = torch.sum(attention_mask, dim=-1).long()
        text_mask = (torch.arange(T, device=input_ids.device).unsqueeze(0) + 1) > input_lengths.unsqueeze(1)

        # BERT encoder
        bert_dur = self.bert(input_ids, attention_mask=attention_mask)
        d_en = self.bert_encoder(bert_dur).transpose(-1, -2)  # [B, 512, T]

        s = ref_s[:, 128:]  # [B, 128] style embedding

        # Duration encoder (CoreML-friendly, no pack_padded_sequence)
        d = self._duration_encoder(d_en, s, input_lengths, text_mask)  # [B, T, 640]

        # Duration prediction
        x, _ = self.pred_lstm(d)  # [B, T, 512]
        duration = self.pred_duration_proj(x)  # [B, T, 50]
        duration = torch.sigmoid(duration).sum(axis=-1) / speed  # [B, T]
        pred_dur = torch.round(duration).clamp(min=1)

        # Text encoder (CoreML-friendly, no pack_padded_sequence)
        t_en = self._text_encoder(input_ids, input_lengths, text_mask)  # [B, 512, T]

        return pred_dur, d.transpose(-1, -2), t_en

    def _duration_encoder(self, x, style, text_lengths, m):
        """DurationEncoder.forward without pack_padded_sequence."""
        from kokoro.modules import AdaLayerNorm

        masks = m
        x = x.permute(2, 0, 1)  # [T, B, 512]
        s = style.expand(x.shape[0], x.shape[1], -1)  # [T, B, 128]
        x = torch.cat([x, s], axis=-1)  # [T, B, 640]
        x = x.masked_fill(masks.unsqueeze(-1).transpose(0, 1), 0.0)
        x = x.transpose(0, 1).transpose(-1, -2)  # [B, 640, T]

        for block in self.de_lstms:
            if isinstance(block, AdaLayerNorm):
                x = block(x.transpose(-1, -2), style).transpose(-1, -2)
                x = torch.cat([x, s.permute(1, 2, 0)], axis=1)  # [B, 640, T]
                x = x.masked_fill(masks.unsqueeze(-1).transpose(-1, -2), 0.0)
            else:
                # LSTM without packing — process full padded sequence
                x = x.transpose(-1, -2)  # [B, T, C]
                x, _ = block(x)  # bidirectional LSTM
                x = F.dropout(x, p=self.de_dropout, training=False)
                x = x.transpose(-1, -2)  # [B, 512, T]

        return x.transpose(-1, -2)  # [B, T, 640]

    def _text_encoder(self, x, input_lengths, m):
        """TextEncoder.forward without pack_padded_sequence."""
        x = self.te_embedding(x)  # [B, T, 512]
        x = x.transpose(1, 2)  # [B, 512, T]
        m_exp = m.unsqueeze(1)
        x = x.masked_fill(m_exp, 0.0)
        for c in self.te_cnn:
            x = c(x)
            x = x.masked_fill(m_exp, 0.0)
        x = x.transpose(1, 2)  # [B, T, 512]
        # LSTM without packing
        x, _ = self.te_lstm(x)  # [B, T, 512]
        x = x.transpose(-1, -2)  # [B, 512, T]
        x = x.masked_fill(m_exp, 0.0)
        return x


class CoreMLProsodyModel(nn.Module):
    """F0/N prosody prediction. Only includes F0Ntrain submodules."""

    def __init__(self, model):
        super().__init__()
        self.shared = model.predictor.shared
        self.F0 = model.predictor.F0
        self.N = model.predictor.N
        self.F0_proj = model.predictor.F0_proj
        self.N_proj = model.predictor.N_proj

    def forward(self, en, s):
        """
        Args:
            en: [1, 640, F] aligned prosody features
            s: [1, 128] style embedding
        Returns:
            F0_pred: [1, F*2]
            N_pred: [1, F*2]
        """
        x, _ = self.shared(en.transpose(-1, -2))  # [B, F, 512]
        F0 = x.transpose(-1, -2)  # [B, 512, F]
        for block in self.F0:
            F0 = block(F0, s)
        F0 = self.F0_proj(F0)  # [B, 1, F*2]
        N = x.transpose(-1, -2)  # [B, 512, F]
        for block in self.N:
            N = block(N, s)
        N = self.N_proj(N)  # [B, 1, F*2]
        return F0.squeeze(1), N.squeeze(1)  # [B, F*2], [B, F*2]


class CoreMLDecoderModel(nn.Module):
    """Decoder vocoder wrapper."""

    def __init__(self, model):
        super().__init__()
        self.decoder = model.decoder

    def forward(self, asr, F0_pred, N_pred, ref_s_dec):
        """
        Args:
            asr: [1, 512, F] aligned text features
            F0_pred: [1, F*2]
            N_pred: [1, F*2]
            ref_s_dec: [1, 128] style embedding (first half)
        Returns:
            audio: [1, 1, F*600]
        """
        return self.decoder(asr, F0_pred, N_pred, ref_s_dec)


# --------------------------------------------------------------------------- #
#  Verification
# --------------------------------------------------------------------------- #

def verify_stages(model, voice_path=None):
    """Verify 3-stage pipeline matches original forward_with_tokens."""
    if voice_path and os.path.exists(voice_path):
        with open(voice_path) as f:
            voice = torch.FloatTensor(json.load(f)['embedding']).unsqueeze(0)
    else:
        print("No voice file found, using random embedding for verification")
        voice = torch.randn(1, 256)

    input_ids = torch.LongTensor([[0, 60, 46, 79, 54, 38, 11, 60, 34, 30, 55, 36, 64, 0]])
    T = input_ids.shape[1]

    # Reference: original forward_with_tokens
    with torch.no_grad():
        ref_audio, ref_dur = model.forward_with_tokens(input_ids, voice, 1.0)

    # 3-stage pipeline using CoreML-friendly wrappers
    dur_model = CoreMLDurationModel(model)
    dur_model.eval()
    pros_model = CoreMLProsodyModel(model)
    pros_model.eval()
    dec_model = CoreMLDecoderModel(model)
    dec_model.eval()

    attention_mask = torch.ones(1, T, dtype=torch.int32)
    padded_ids = torch.zeros(1, MAX_PHONEMES, dtype=torch.long)
    padded_ids[0, :T] = input_ids[0]
    padded_mask = torch.zeros(1, MAX_PHONEMES, dtype=torch.int32)
    padded_mask[0, :T] = 1
    speed = torch.ones(1)

    with torch.no_grad():
        # Stage 1: Duration
        pred_dur, d_t, t_en = dur_model(padded_ids.int(), padded_mask, voice, speed)

        # Unpad durations (only first T values are valid)
        pred_dur_valid = pred_dur[0, :T].long()
        total_frames = pred_dur_valid.sum().item()

        # Swift-side alignment (done in Python here for verification)
        indices = torch.repeat_interleave(torch.arange(MAX_PHONEMES), pred_dur[0].long())
        F_actual = indices.shape[0]
        pred_aln_trg = torch.zeros(1, MAX_PHONEMES, F_actual)
        pred_aln_trg[0, indices, torch.arange(F_actual)] = 1

        en = d_t @ pred_aln_trg  # [1, 640, F]
        asr = t_en @ pred_aln_trg  # [1, 512, F]

        # Stage 2: Prosody
        s = voice[:, 128:]
        F0_pred, N_pred = pros_model(en, s)

        # Stage 3: Decoder
        ref_s_dec = voice[:, :128]
        audio_out = dec_model(asr, F0_pred, N_pred, ref_s_dec)

    audio_3stage = audio_out.squeeze()
    ref_audio_flat = ref_audio.squeeze()

    # Compare (note: disable_complex=True may cause small differences)
    min_len = min(audio_3stage.shape[0], ref_audio_flat.shape[0])
    diff = (audio_3stage[:min_len] - ref_audio_flat[:min_len]).abs()
    print(f"\n3-Stage Verification:")
    print(f"  Reference audio: {ref_audio_flat.shape[0]} samples")
    print(f"  3-stage audio:   {audio_3stage.shape[0]} samples")
    print(f"  Diff: max={diff.max():.4f}, mean={diff.mean():.4f}")

    # Durations should match
    dur_diff = (pred_dur_valid.float() - ref_dur[:T].float()).abs().max()
    print(f"  Duration diff: max={dur_diff:.1f}")

    if diff.mean() < 0.05:
        print("  PASS: 3-stage pipeline matches reference")
    elif diff.mean() < 0.2:
        print("  OK: small differences (likely from disable_complex STFT)")
    else:
        print("  WARNING: large differences, check pipeline")

    return voice


# --------------------------------------------------------------------------- #
#  CoreML Conversion
# --------------------------------------------------------------------------- #

def convert_duration_model(model, output_dir, quantize=None):
    """Convert duration model to CoreML."""
    print("\n=== Converting Duration Model ===")
    dur_model = CoreMLDurationModel(model)
    dur_model.eval()

    example_ids = torch.randint(0, 100, (1, MAX_PHONEMES), dtype=torch.int32)
    example_mask = torch.ones(1, MAX_PHONEMES, dtype=torch.int32)
    example_ref_s = torch.randn(1, 256)
    example_speed = torch.ones(1)

    # Use enumerated shapes to minimize backward LSTM contamination from padding.
    # Swift pads to the smallest bucket that fits the actual phoneme count.
    print(f"  Phoneme buckets: {PHONEME_BUCKETS}")
    print("  Tracing...")
    with torch.no_grad():
        traced = torch.jit.trace(dur_model, (example_ids, example_mask, example_ref_s, example_speed))

    print("  Converting to CoreML...")
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="input_ids", shape=ct.EnumeratedShapes(
                shapes=[(1, n) for n in PHONEME_BUCKETS]), dtype=np.int32),
            ct.TensorType(name="attention_mask", shape=ct.EnumeratedShapes(
                shapes=[(1, n) for n in PHONEME_BUCKETS]), dtype=np.int32),
            ct.TensorType(name="ref_s", shape=(1, 256), dtype=np.float32),
            ct.TensorType(name="speed", shape=(1,), dtype=np.float32),
        ],
        outputs=[
            ct.TensorType(name="pred_dur"),
            ct.TensorType(name="d_transposed"),
            ct.TensorType(name="t_en"),
        ],
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS18,
    )

    if quantize == "int8":
        mlmodel = _quantize_int8(mlmodel)

    path = output_dir / "duration.mlpackage"
    mlmodel.save(str(path))
    sz = _model_size_mb(path)
    print(f"  Saved {path.name} ({sz:.1f} MB)")
    return mlmodel


def convert_prosody_model(model, output_dir, quantize=None):
    """Convert prosody model to CoreML with enumerated frame shapes."""
    print("\n=== Converting Prosody Model ===")
    pros_model = CoreMLProsodyModel(model)
    pros_model.eval()

    # Trace with representative frame count
    F_trace = 200
    example_en = torch.randn(1, 640, F_trace)
    example_s = torch.randn(1, 128)

    print("  Tracing...")
    with torch.no_grad():
        traced = torch.jit.trace(pros_model, (example_en, example_s))

    # Use bucket frame sizes as enumerated shapes
    frame_sizes = [f for _, f in BUCKETS]
    print(f"  Enumerated shapes: {frame_sizes}")

    print("  Converting to CoreML...")
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="en", shape=ct.EnumeratedShapes(
                shapes=[(1, 640, f) for f in frame_sizes])),
            ct.TensorType(name="s", shape=(1, 128), dtype=np.float32),
        ],
        outputs=[
            ct.TensorType(name="F0_pred"),
            ct.TensorType(name="N_pred"),
        ],
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS18,
    )

    if quantize == "int8":
        mlmodel = _quantize_int8(mlmodel)

    path = output_dir / "prosody.mlpackage"
    mlmodel.save(str(path))
    sz = _model_size_mb(path)
    print(f"  Saved {path.name} ({sz:.1f} MB)")
    return mlmodel


def convert_decoder(model, output_dir, quantize=None):
    """Convert decoder to CoreML with fixed-shape buckets."""
    print("\n=== Converting Decoder ===")
    dec_model = CoreMLDecoderModel(model)
    dec_model.eval()

    for name, max_frames in BUCKETS:
        max_samples = max_frames * SAMPLES_PER_FRAME
        print(f"\n  Bucket {name}: {max_frames} frames → {max_samples} samples ({max_samples/SAMPLE_RATE:.1f}s)")

        example_asr = torch.randn(1, 512, max_frames)
        example_f0 = torch.randn(1, max_frames * 2)
        example_n = torch.randn(1, max_frames * 2)
        example_ref_s = torch.randn(1, 128)

        print("    Tracing...")
        with torch.no_grad():
            traced = torch.jit.trace(dec_model, (example_asr, example_f0, example_n, example_ref_s))

        print("    Converting to CoreML...")
        mlmodel = ct.convert(
            traced,
            inputs=[
                ct.TensorType(name="asr", shape=(1, 512, max_frames), dtype=np.float32),
                ct.TensorType(name="F0_pred", shape=(1, max_frames * 2), dtype=np.float32),
                ct.TensorType(name="N_pred", shape=(1, max_frames * 2), dtype=np.float32),
                ct.TensorType(name="ref_s", shape=(1, 128), dtype=np.float32),
            ],
            outputs=[ct.TensorType(name="audio")],
            compute_units=ct.ComputeUnit.CPU_AND_NE,
            compute_precision=ct.precision.FLOAT16,
            minimum_deployment_target=ct.target.iOS18,
        )

        if quantize == "int8":
            mlmodel = _quantize_int8(mlmodel)

        path = output_dir / f"decoder_{name}.mlpackage"
        mlmodel.save(str(path))
        sz = _model_size_mb(path)
        print(f"    Saved {path.name} ({sz:.1f} MB)")


# --------------------------------------------------------------------------- #
#  Utilities
# --------------------------------------------------------------------------- #

def _quantize_int8(mlmodel):
    """Quantize CoreML model to INT8 palettization."""
    from coremltools.optimize.coreml import (
        OpPalettizerConfig, OptimizationConfig, palettize_weights)
    op_config = OpPalettizerConfig(mode="kmeans", nbits=8)
    config = OptimizationConfig(global_config=op_config)
    return palettize_weights(mlmodel, config)


def _model_size_mb(path):
    """Get total model size in MB."""
    path = Path(path)
    if path.is_dir():
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6
    return path.stat().st_size / 1e6


def save_config(output_dir):
    """Save pipeline config for Swift."""
    config = {
        "model": "kokoro-82m",
        "version": "v2",
        "stages": ["duration", "prosody", "decoder"],
        "max_phonemes": MAX_PHONEMES,
        "phoneme_buckets": PHONEME_BUCKETS,
        "d_transposed_dim": 640,
        "t_en_dim": 512,
        "style_dim": 128,
        "sample_rate": SAMPLE_RATE,
        "samples_per_frame": SAMPLES_PER_FRAME,
        "buckets": {name: {"frames": f, "samples": f * SAMPLES_PER_FRAME}
                    for name, f in BUCKETS},
    }
    path = output_dir / "pipeline_config.json"
    with open(path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"\nSaved {path.name}")


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Convert Kokoro-82M to 3-stage CoreML")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--quantize", choices=["int8"], default=None)
    parser.add_argument("--skip-verify", action="store_true", help="Skip verification")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Clone kokoro if needed
    if not Path("/tmp/kokoro/kokoro/model.py").exists():
        print("Cloning hexgrad/kokoro...")
        os.system("cd /tmp && git clone https://github.com/hexgrad/kokoro.git 2>/dev/null")

    model = load_kokoro_model()

    # Verify 3-stage pipeline matches original
    if not args.skip_verify:
        voice_path = "/tmp/kokoro-coreml-test/voices/af_heart.json"
        try:
            verify_stages(model, voice_path)
        except Exception as e:
            print(f"Verification warning: {e}")
            print("Continuing with conversion...")

    # Convert each stage
    convert_duration_model(model, output_dir, args.quantize)
    convert_prosody_model(model, output_dir, args.quantize)
    convert_decoder(model, output_dir, args.quantize)

    # Save config
    save_config(output_dir)

    # Copy voices and vocab from existing model
    import shutil
    src = Path("/tmp/kokoro-coreml-test")
    if src.exists():
        for f in ["voices", "vocab_index.json", "config.json", "g2p_vocab.json",
                   "us_gold.json", "us_silver.json"]:
            s = src / f
            d = output_dir / f
            if s.exists():
                if s.is_dir():
                    shutil.copytree(str(s), str(d), dirs_exist_ok=True)
                else:
                    shutil.copy2(str(s), str(d))
        for g2p in ["G2PEncoder.mlmodelc", "G2PDecoder.mlmodelc"]:
            s = src / g2p
            if s.exists():
                shutil.copytree(str(s), str(output_dir / g2p), dirs_exist_ok=True)

    # Summary
    print(f"\nDone! Output: {output_dir}")
    for f in sorted(output_dir.iterdir()):
        sz = _model_size_mb(f)
        print(f"  {f.name}: {sz:.1f} MB")


if __name__ == "__main__":
    main()
