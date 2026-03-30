#!/usr/bin/env python3
"""Convert Kokoro-82M to 3-stage CoreML (duration + prosody + decoder).

The alignment step (duration → alignment matrix → asr) is dynamic and
cannot be traced. It must be done in Swift between model calls.

Stage 1 - Duration Model:
  Input:  input_ids [1,N], attention_mask [1,N], ref_s [1,256], speed [1]
  Output: pred_dur [1,N], d [1,N,640], t_en [1,512,N]

Stage 2 - Prosody Model (F0 + N):
  Input:  en [1,640,F], s [1,128]
  Output: F0_pred [1,F*2], N_pred [1,F*2]

Stage 3 - Decoder (fixed-shape buckets):
  Input:  asr [1,512,F], F0_pred [1,F*2], N_pred [1,F*2], ref_s [1,128]
  Output: audio [1,1,S]

Swift-side alignment:
  pred_aln_trg = build_alignment(pred_dur)  # [N, total_frames]
  en = d.T @ pred_aln_trg                   # [640, F]
  asr = t_en @ pred_aln_trg                 # [512, F]

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
import coremltools as ct


SAMPLE_RATE = 24000
MAX_PHONEMES = 128  # Max phoneme sequence length

# Decoder buckets: (name, max_frames, max_samples)
# frames = sum of pred_dur; samples = frames * hop * 2
# hop varies by bucket to fit CoreML fixed shapes
BUCKETS = [
    ("5s",  125, 120_000),   # ~5 seconds
    ("10s", 250, 240_000),   # ~10 seconds
    ("15s", 375, 360_000),   # ~15 seconds
]


def load_kokoro_model():
    """Load the original Kokoro-82M PyTorch model."""
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

    from kokoro.model import KModel
    model = KModel()
    model.eval()
    print(f"Loaded Kokoro-82M ({sum(p.numel() for p in model.parameters())/1e6:.1f}M params)")
    return model


class DurationModel(nn.Module):
    """Wraps BERT + bert_encoder + predictor duration components."""

    def __init__(self, model):
        super().__init__()
        self.bert = model.bert
        self.bert_encoder = model.bert_encoder
        self.predictor_text_encoder = model.predictor.text_encoder
        self.predictor_lstm = model.predictor.lstm
        self.predictor_duration_proj = model.predictor.duration_proj
        self.text_encoder = model.text_encoder

    def forward(self, input_ids, attention_mask, ref_s, speed):
        """
        Args:
            input_ids: [1, N] int32
            attention_mask: [1, N] int32 (1=real, 0=pad)
            ref_s: [1, 256] float32
            speed: [1] float32
        Returns:
            pred_dur: [1, N] float32 (rounded durations)
            d_transposed: [1, 640, N] float32 (prosody features)
            t_en: [1, 512, N] float32 (text encoding)
        """
        T = input_ids.shape[1]
        # Use float for input_lengths to avoid int32→rsqrt CoreML error
        input_lengths = torch.sum(attention_mask, dim=-1).long()
        text_mask = (torch.arange(T).unsqueeze(0).to(input_ids.device) + 1) > input_lengths.unsqueeze(1)

        bert_dur = self.bert(input_ids, attention_mask=attention_mask)
        d_en = self.bert_encoder(bert_dur).transpose(-1, -2)

        s = ref_s[:, 128:]
        d = self.predictor_text_encoder(d_en, s, input_lengths, text_mask)

        x, _ = self.predictor_lstm(d)
        duration = self.predictor_duration_proj(x)
        duration = torch.sigmoid(duration).sum(axis=-1) / speed
        pred_dur = torch.round(duration).clamp(min=1)

        t_en = self.text_encoder(input_ids, input_lengths, text_mask)

        return pred_dur, d.transpose(-1, -2), t_en


class ProsodyModel(nn.Module):
    """Wraps predictor.F0Ntrain for prosody prediction."""

    def __init__(self, model):
        super().__init__()
        self.predictor = model.predictor

    def forward(self, en, s):
        """
        Args:
            en: [1, 640, F] float32 (aligned prosody features)
            s: [1, 128] float32 (style embedding, second half)
        Returns:
            F0_pred: [1, F*2] float32
            N_pred: [1, F*2] float32
        """
        F0_pred, N_pred = self.predictor.F0Ntrain(en, s)
        return F0_pred, N_pred


class DecoderModel(nn.Module):
    """Wraps the decoder vocoder with fixed-shape inputs."""

    def __init__(self, model):
        super().__init__()
        self.decoder = model.decoder

    def forward(self, asr, F0_pred, N_pred, ref_s_dec):
        """
        Args:
            asr: [1, 512, F] float32 (aligned text features)
            F0_pred: [1, F*2] float32
            N_pred: [1, F*2] float32
            ref_s_dec: [1, 128] float32 (first half of ref_s)
        Returns:
            audio: [1, 1, S] float32
        """
        return self.decoder(asr, F0_pred, N_pred, ref_s_dec)


def verify_pipeline(model, voice_path):
    """Run end-to-end verification with a real voice."""
    import json as json_mod
    with open(voice_path) as f:
        voice = torch.FloatTensor(json_mod.load(f)['embedding']).unsqueeze(0)

    # Use model's forward_with_tokens as ground truth
    input_ids = torch.LongTensor([[0, 60, 46, 79, 54, 38, 11, 60, 34, 30, 55, 36, 64, 0]])
    ref_audio, _ = model.forward_with_tokens(input_ids, voice, 1.0)

    print(f"\nVerification: reference audio shape={ref_audio.shape}, "
          f"range=[{ref_audio.min():.3f}, {ref_audio.max():.3f}]")

    # Manual pipeline
    T = input_ids.shape[1]
    ref_s = voice
    input_lengths = torch.LongTensor([T])
    text_mask = torch.arange(T).unsqueeze(0) + 1 > input_lengths.unsqueeze(1)

    with torch.no_grad():
        s = ref_s[:, 128:]
        bert_dur = model.bert(input_ids, attention_mask=(~text_mask).int())
        d_en = model.bert_encoder(bert_dur).transpose(-1, -2)
        d = model.predictor.text_encoder(d_en, s, input_lengths, text_mask)
        x, _ = model.predictor.lstm(d)
        duration = model.predictor.duration_proj(x)
        pred_dur = torch.round(torch.sigmoid(duration).sum(axis=-1)).clamp(min=1).long().squeeze()

        total_frames = pred_dur.sum().item()
        indices = torch.repeat_interleave(torch.arange(T), pred_dur)
        pred_aln_trg = torch.zeros(T, total_frames).unsqueeze(0)
        pred_aln_trg[0, indices, torch.arange(len(indices))] = 1

        en = d.transpose(-1, -2) @ pred_aln_trg
        F0_pred, N_pred = model.predictor.F0Ntrain(en, s)
        t_en = model.text_encoder(input_ids, input_lengths, text_mask)
        asr = t_en @ pred_aln_trg
        audio = model.decoder(asr, F0_pred, N_pred, ref_s[:, :128]).squeeze()

    diff = (ref_audio[:audio.shape[0]] - audio[:ref_audio.shape[0]]).abs()
    print(f"Manual vs reference: max_diff={diff.max():.4f}, mean_diff={diff.mean():.4f}")
    assert diff.max() < 0.2, f"Pipeline mismatch: max_diff={diff.max()}"
    print("PASS: manual pipeline matches reference\n")
    return voice


def convert_duration_model(model, output_dir, quantize=None):
    """Convert duration model to CoreML."""
    print("Converting duration model...")
    dur_model = DurationModel(model)
    dur_model.eval()

    example_ids = torch.randint(0, 100, (1, MAX_PHONEMES), dtype=torch.int32)
    example_mask = torch.ones(1, MAX_PHONEMES, dtype=torch.int32)
    example_ref_s = torch.randn(1, 256)
    example_speed = torch.ones(1)

    traced = torch.jit.trace(dur_model, (example_ids, example_mask, example_ref_s, example_speed))

    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="input_ids", shape=(1, MAX_PHONEMES), dtype=np.int32),
            ct.TensorType(name="attention_mask", shape=(1, MAX_PHONEMES), dtype=np.int32),
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
        minimum_deployment_target=ct.target.iOS17,
    )

    path = output_dir / "duration.mlpackage"
    mlmodel.save(str(path))
    print(f"  Saved {path.name} ({os.path.getsize(path / 'Data/com.apple.CoreML/weights/weight.bin') / 1e6:.1f} MB)")
    return mlmodel


def convert_prosody_model(model, output_dir, quantize=None):
    """Convert F0/N prosody predictor to CoreML."""
    print("Converting prosody model...")
    pros_model = ProsodyModel(model)
    pros_model.eval()

    # Use a representative frame count (125 frames ~ 5s)
    F = 125
    example_en = torch.randn(1, 640, F)
    example_s = torch.randn(1, 128)

    traced = torch.jit.trace(pros_model, (example_en, example_s))

    # Use flexible shape for frame dimension
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="en", shape=ct.EnumeratedShapes(
                shapes=[(1, 640, f) for f in [50, 75, 100, 125, 150, 200, 250, 300, 375]])),
            ct.TensorType(name="s", shape=(1, 128), dtype=np.float32),
        ],
        outputs=[
            ct.TensorType(name="F0_pred"),
            ct.TensorType(name="N_pred"),
        ],
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS17,
    )

    path = output_dir / "prosody.mlpackage"
    mlmodel.save(str(path))
    print(f"  Saved {path.name}")
    return mlmodel


def convert_decoder(model, output_dir, quantize=None):
    """Convert decoder to CoreML with fixed-shape buckets."""
    print("Converting decoder...")
    dec_model = DecoderModel(model)
    dec_model.eval()

    for name, max_frames, max_samples in BUCKETS:
        print(f"  Bucket {name}: {max_frames} frames, {max_samples} samples...")

        example_asr = torch.randn(1, 512, max_frames)
        example_f0 = torch.randn(1, max_frames * 2)
        example_n = torch.randn(1, max_frames * 2)
        example_ref_s = torch.randn(1, 128)

        traced = torch.jit.trace(dec_model, (example_asr, example_f0, example_n, example_ref_s))

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
            minimum_deployment_target=ct.target.iOS17,
        )

        if quantize == "int8":
            from coremltools.optimize.coreml import (
                OpPalettizerConfig, OptimizationConfig, palettize_weights)
            op_config = OpPalettizerConfig(mode="kmeans", nbits=8)
            config = OptimizationConfig(global_config=op_config)
            mlmodel = palettize_weights(mlmodel, config)

        path = output_dir / f"decoder_{name}.mlpackage"
        mlmodel.save(str(path))
        sz = os.path.getsize(path / 'Data/com.apple.CoreML/weights/weight.bin') / 1e6
        print(f"    Saved {path.name} ({sz:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Convert Kokoro-82M to 3-stage CoreML")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--quantize", choices=["int8"], default=None)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Clone kokoro if needed
    if not Path("/tmp/kokoro/kokoro/model.py").exists():
        print("Cloning hexgrad/kokoro...")
        os.system("cd /tmp && git clone https://github.com/hexgrad/kokoro.git 2>/dev/null")

    model = load_kokoro_model()

    # Verify PyTorch pipeline first
    voice_path = "/tmp/kokoro-coreml-test/voices/af_heart.json"
    if os.path.exists(voice_path):
        verify_pipeline(model, voice_path)

    # Convert each stage
    convert_duration_model(model, output_dir, args.quantize)
    convert_prosody_model(model, output_dir, args.quantize)
    convert_decoder(model, output_dir, args.quantize)

    # Copy voices and vocab from existing model
    import shutil
    src = Path("/tmp/kokoro-coreml-test")
    for f in ["voices", "vocab_index.json", "config.json", "g2p_vocab.json",
              "us_gold.json", "us_silver.json"]:
        s = src / f
        d = output_dir / f
        if s.exists():
            if s.is_dir():
                shutil.copytree(str(s), str(d), dirs_exist_ok=True)
            else:
                shutil.copy2(str(s), str(d))

    # Copy G2P models
    for g2p in ["G2PEncoder.mlmodelc", "G2PDecoder.mlmodelc"]:
        s = src / g2p
        if s.exists():
            shutil.copytree(str(s), str(output_dir / g2p), dirs_exist_ok=True)

    print(f"\nDone! Output: {output_dir}")
    for f in sorted(output_dir.iterdir()):
        if f.is_dir():
            sz = sum(ff.stat().st_size for ff in f.rglob("*") if ff.is_file()) / 1e6
        else:
            sz = f.stat().st_size / 1e6
        print(f"  {f.name}: {sz:.1f} MB")


if __name__ == "__main__":
    main()
