#!/usr/bin/env python3
"""
Convert NVIDIA Parakeet-TDT 0.6B v3 from NeMo to CoreML with INT4 encoder.

Pipeline:
  NeMo .nemo → extract 4 sub-modules → torch.jit.trace → save .pt
  → load .pt → coremltools.convert() → INT4 palettize encoder
  → save .mlpackage/.mlmodelc + vocab.json + config.json

Requires coremltools 8.1 (9.0 crashes with SIGSEGV on encoder save):
  pip install nemo_toolkit[asr] 'coremltools==8.1'
  python scripts/convert_parakeet.py --output-dir ./parakeet-coreml

Four CoreML models are produced:
  preprocessor.mlpackage  - audio → mel (CPU only)
  encoder.mlpackage       - mel → encoded (CPU + Neural Engine, INT4)
  decoder.mlpackage       - token + LSTM state → output (CPU + Neural Engine)
  joint.mlpackage         - encoder + decoder → logits (CPU + Neural Engine)

Publish to HuggingFace:
  huggingface-cli upload aufklarer/Parakeet-TDT-v3-CoreML-INT4 ./parakeet-coreml
"""

import argparse
import gc
import json
import shutil
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
import torch.nn as nn


def load_nemo_model():
    """Load the Parakeet-TDT model from NeMo."""
    import nemo.collections.asr as nemo_asr

    model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(
        "nvidia/parakeet-tdt-0.6b-v2"
    )
    model.eval()
    return model


class PreprocessorWrapper(nn.Module):
    """Wraps the NeMo preprocessor (mel spectrogram extraction).

    Note: The CoreML preprocessor is NOT used at inference time.
    Mel features are computed in Swift using Accelerate/vDSP (MelPreprocessor.swift)
    because torch.stft tracing bakes the audio length as a constant, breaking
    per-feature normalization for variable-length inputs. This wrapper is only
    used to get mel features for encoder tracing.
    """

    def __init__(self, preprocessor):
        super().__init__()
        self.preprocessor = preprocessor

    def forward(self, audio):
        # audio: [1, T]
        length = torch.tensor([audio.shape[1]], dtype=torch.long)
        mel, mel_length = self.preprocessor(input_signal=audio, length=length)
        return mel, mel_length


class EncoderWrapper(nn.Module):
    """Wraps the FastConformer encoder."""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, mel, length):
        # mel: [1, 128, T'], length: [1]
        encoded, encoded_length = self.encoder(audio_signal=mel, length=length)
        # NeMo encoder outputs [B, C, T] (channels-first), transpose to [B, T, C]
        encoded = encoded.transpose(1, 2)
        return encoded, encoded_length


class DecoderWrapper(nn.Module):
    """Wraps the LSTM prediction network."""

    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    def forward(self, token, h, c):
        # token: [1, 1], h: [2, 1, 640], c: [2, 1, 640]
        state = (h, c)
        output, (h_out, c_out) = self.decoder.predict(
            token, state=state, add_sos=False, batch_size=None
        )
        return output, h_out, c_out


class JointWrapper(nn.Module):
    """Wraps the TDT joint network with dual heads."""

    def __init__(self, joint, vocab_size):
        super().__init__()
        self.joint = joint
        self.num_tokens = vocab_size + 1  # +1 for blank

    def forward(self, encoder_output, decoder_output):
        # encoder_output: [1, 1, 1024], decoder_output: [1, 1, 640]
        # Joint outputs [1, 1, 1, num_tokens + num_durations] combined
        combined = self.joint.joint(encoder_output, decoder_output)
        # Remove extra dim: [1, 1, 1, 1030] → [1, 1, 1030]
        combined = combined.squeeze(2)
        # Split: first num_tokens are token logits, rest are duration logits
        token_logits = combined[..., : self.num_tokens]  # [1, 1, 1025]
        duration_logits = combined[..., self.num_tokens :]  # [1, 1, 5]
        return token_logits, duration_logits


def convert_to_coreml(traced_path, name, input_specs, output_names, compute_units):
    """Convert a traced TorchScript model to CoreML."""
    print(f"  Loading traced model from {traced_path}...")
    traced = torch.jit.load(str(traced_path))

    print(f"  Converting {name} to CoreML...")
    mlmodel = ct.convert(
        traced,
        inputs=input_specs,
        outputs=[ct.TensorType(name=n) for n in output_names],
        compute_units=compute_units,
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS17,
        skip_model_load=True,
    )
    return mlmodel


def quantize_encoder(mlmodel):
    """Apply INT4 palettization to the encoder for Neural Engine efficiency."""
    from coremltools.optimize.coreml import (
        OpPalettizerConfig,
        OptimizationConfig,
        palettize_weights,
    )

    op_config = OpPalettizerConfig(mode="kmeans", nbits=4)
    config = OptimizationConfig(global_config=op_config)
    return palettize_weights(mlmodel, config)


def extract_vocab(model, output_dir):
    """Extract vocabulary from the NeMo model's tokenizer."""
    tokenizer = model.tokenizer
    vocab = {}
    for i in range(tokenizer.vocab_size):
        token = tokenizer.ids_to_tokens([i])[0]
        vocab[str(i)] = token

    vocab_path = output_dir / "vocab.json"
    with open(vocab_path, "w") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    print(f"  Saved vocabulary ({len(vocab)} tokens) to {vocab_path}")


def save_config(model, output_dir):
    """Save the model configuration derived from the NeMo model."""
    vocab_size = model.tokenizer.vocab_size
    durations = list(model.cfg.model_defaults.tdt_durations)
    config = {
        "numMelBins": 128,
        "sampleRate": 16000,
        "nFFT": 512,
        "hopLength": 160,
        "winLength": 400,
        "preEmphasis": 0.97,
        "encoderHidden": 1024,
        "encoderLayers": 24,
        "subsamplingFactor": 8,
        "decoderHidden": 640,
        "decoderLayers": 2,
        "vocabSize": vocab_size,
        "blankTokenId": vocab_size,
        "numDurationBins": len(durations),
        "durationBins": durations,
    }

    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Saved config (vocabSize={vocab_size}, blank={vocab_size}) to {config_path}")


def compile_mlpackage(output_dir, name):
    """Compile .mlpackage to .mlmodelc for distribution."""
    pkg_path = output_dir / f"{name}.mlpackage"
    compiled_path = output_dir / f"{name}.mlmodelc"

    if compiled_path.exists():
        shutil.rmtree(compiled_path)

    print(f"  Compiling {name}.mlpackage → {name}.mlmodelc ...")
    compiled_url = ct.utils.compile_model(str(pkg_path))
    shutil.move(str(compiled_url), str(compiled_path))

    # Remove the .mlpackage to save space
    shutil.rmtree(pkg_path)
    print(f"  Compiled {name}.mlmodelc")


def main():
    parser = argparse.ArgumentParser(description="Convert Parakeet-TDT to CoreML")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./parakeet-coreml",
        help="Output directory for CoreML models",
    )
    parser.add_argument(
        "--no-quantize",
        action="store_true",
        help="Skip INT4 quantization of encoder",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Compile .mlpackage to .mlmodelc",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    traced_dir = output_dir / "_traced"
    traced_dir.mkdir(exist_ok=True)

    # ─── Phase 1: Load NeMo, extract metadata, trace all modules ───
    print("=" * 60)
    print("Phase 1: Loading NeMo model and tracing sub-modules")
    print("=" * 60)

    print("Loading NeMo model...")
    model = load_nemo_model()
    vocab_size = model.tokenizer.vocab_size
    blank_id = vocab_size

    # Extract vocabulary and config
    print("Extracting vocabulary...")
    extract_vocab(model, output_dir)
    print("Saving configuration...")
    save_config(model, output_dir)

    # Get mel for encoder tracing (preprocessor is NOT converted to CoreML —
    # mel features are computed in Swift using Accelerate/vDSP)
    print("Computing mel for encoder tracing...")
    preprocessor = PreprocessorWrapper(model.preprocessor)
    example_audio = torch.randn(1, 16000)
    with torch.no_grad():
        mel, mel_len = preprocessor(example_audio)

    # Trace encoder
    print("Tracing encoder...")
    encoder = EncoderWrapper(model.encoder)
    with torch.no_grad():
        traced_enc = torch.jit.trace(encoder, (mel, mel_len))
    traced_enc.save(str(traced_dir / "encoder.pt"))

    # Get encoder output for joint tracing
    with torch.no_grad():
        encoded, _ = encoder(mel, mel_len)
    example_enc_slice = encoded[:, :1, :]  # [1, 1, 1024]

    # Trace decoder
    print("Tracing decoder...")
    decoder = DecoderWrapper(model.decoder)
    example_token = torch.tensor([[blank_id]], dtype=torch.long)
    example_h = torch.zeros(2, 1, 640)
    example_c = torch.zeros(2, 1, 640)
    with torch.no_grad():
        traced_dec = torch.jit.trace(decoder, (example_token, example_h, example_c))
    traced_dec.save(str(traced_dir / "decoder.pt"))

    # Get decoder output for joint tracing
    with torch.no_grad():
        dec_out, _, _ = decoder(example_token, example_h, example_c)

    # Trace joint
    print("Tracing joint...")
    joint = JointWrapper(model.joint, vocab_size=vocab_size)
    with torch.no_grad():
        traced_jnt = torch.jit.trace(joint, (example_enc_slice, dec_out))
    traced_jnt.save(str(traced_dir / "joint.pt"))

    # Free NeMo model and all wrappers
    print("Freeing NeMo model...")
    del model, preprocessor, encoder, decoder, joint
    del traced_enc, traced_dec, traced_jnt
    del mel, mel_len, encoded, example_enc_slice, dec_out
    gc.collect()

    # ─── Phase 2: Convert each traced model to CoreML (one at a time) ───
    # Note: No preprocessor conversion — mel features are computed in Swift
    # using Accelerate/vDSP (MelPreprocessor.swift) because torch.stft tracing
    # bakes the audio length as a constant, breaking per-feature normalization.
    print()
    print("=" * 60)
    print("Phase 2: Converting traced models to CoreML")
    print("=" * 60)

    # Encoder — use EnumeratedShapes to avoid BNNS crash with dynamic shapes.
    # Cover durations from 1s to 30s of audio (100 to 3000 mel frames).
    print("\nConverting encoder...")
    mel_shapes = [
        ct.Shape(shape=(1, 128, l))
        for l in [100, 200, 300, 400, 500, 750, 1000, 1500, 2000, 3000]
    ]
    enc_ml = convert_to_coreml(
        traced_dir / "encoder.pt",
        "encoder",
        [
            ct.TensorType(
                name="mel",
                shape=ct.EnumeratedShapes(shapes=mel_shapes),
                dtype=np.float32,
            ),
            ct.TensorType(name="length", shape=[1], dtype=np.int32),
        ],
        ["encoded", "encoded_length"],
        ct.ComputeUnit.CPU_AND_NE,
    )
    if not args.no_quantize:
        print("  Quantizing encoder to INT4...")
        enc_ml = quantize_encoder(enc_ml)
    enc_ml.save(str(output_dir / "encoder.mlpackage"))
    del enc_ml
    gc.collect()
    print("  Encoder saved.")

    # Decoder
    print("\nConverting decoder...")
    dec_ml = convert_to_coreml(
        traced_dir / "decoder.pt",
        "decoder",
        [
            ct.TensorType(name="token", shape=[1, 1], dtype=np.int32),
            ct.TensorType(name="h", shape=[2, 1, 640], dtype=np.float16),
            ct.TensorType(name="c", shape=[2, 1, 640], dtype=np.float16),
        ],
        ["decoder_output", "h_out", "c_out"],
        ct.ComputeUnit.CPU_AND_NE,
    )
    dec_ml.save(str(output_dir / "decoder.mlpackage"))
    del dec_ml
    gc.collect()
    print("  Decoder saved.")

    # Joint
    print("\nConverting joint network...")
    jnt_ml = convert_to_coreml(
        traced_dir / "joint.pt",
        "joint",
        [
            ct.TensorType(name="encoder_output", shape=[1, 1, 1024], dtype=np.float16),
            ct.TensorType(name="decoder_output", shape=[1, 1, 640], dtype=np.float16),
        ],
        ["token_logits", "duration_logits"],
        ct.ComputeUnit.CPU_AND_NE,
    )
    jnt_ml.save(str(output_dir / "joint.mlpackage"))
    del jnt_ml
    gc.collect()
    print("  Joint saved.")

    # Clean up traced models
    shutil.rmtree(traced_dir)

    # Optionally compile
    if args.compile:
        print("\nCompiling CoreML models...")
        for name in ["encoder", "decoder", "joint"]:
            compile_mlpackage(output_dir, name)

    print(f"\nDone! Models saved to {output_dir}/")
    print("Files:")
    for f in sorted(output_dir.iterdir()):
        size = (
            sum(ff.stat().st_size for ff in f.rglob("*") if ff.is_file())
            if f.is_dir()
            else f.stat().st_size
        )
        print(f"  {f.name}: {size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
