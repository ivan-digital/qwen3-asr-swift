#!/usr/bin/env python3
"""
Convert NVIDIA Parakeet-TDT 0.6B v3 from NeMo to CoreML with INT4 encoder.

Pipeline:
  NeMo .nemo → extract 4 sub-modules → torch.jit.trace → coremltools.convert()
  → INT4 palettize encoder → save .mlmodelc + vocab.json + config.json

Usage:
  pip install nemo_toolkit[asr] coremltools
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
import json
import os
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
    """Wraps the NeMo preprocessor (mel spectrogram extraction)."""

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

    def __init__(self, joint):
        super().__init__()
        self.joint = joint

    def forward(self, encoder_output, decoder_output):
        # encoder_output: [1, 1, 1024], decoder_output: [1, 1, 640]
        # Returns: token_logits [1, 1, 8193], duration_logits [1, 1, 5]
        outputs = self.joint.joint(encoder_output, decoder_output)
        if isinstance(outputs, (list, tuple)):
            token_logits = outputs[0]
            duration_logits = outputs[1]
        else:
            token_logits = outputs
            duration_logits = None
        return token_logits, duration_logits


def trace_and_convert(wrapper, example_inputs, name, compute_units, output_names):
    """Trace a PyTorch module and convert to CoreML."""
    print(f"  Tracing {name}...")
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, example_inputs)

    print(f"  Converting {name} to CoreML...")
    input_specs = []
    for i, inp in enumerate(example_inputs):
        shape = list(inp.shape)
        # Use flexible shapes for time dimension
        if name == "preprocessor" and i == 0:
            input_specs.append(
                ct.TensorType(
                    name="audio",
                    shape=ct.Shape(
                        shape=(1, ct.RangeDim(lower_bound=160, upper_bound=480000))
                    ),
                    dtype=np.float32,
                )
            )
        elif name == "encoder" and i == 0:
            input_specs.append(
                ct.TensorType(
                    name="mel",
                    shape=ct.Shape(
                        shape=(
                            1,
                            128,
                            ct.RangeDim(lower_bound=1, upper_bound=3000),
                        )
                    ),
                    dtype=np.float32,
                )
            )
        elif name == "encoder" and i == 1:
            input_specs.append(
                ct.TensorType(name="length", shape=list(inp.shape), dtype=np.int32)
            )
        elif name == "decoder":
            names = ["token", "h", "c"]
            dtypes = [np.int32, np.float16, np.float16]
            input_specs.append(
                ct.TensorType(name=names[i], shape=shape, dtype=dtypes[i])
            )
        elif name == "joint":
            names = ["encoder_output", "decoder_output"]
            input_specs.append(
                ct.TensorType(name=names[i], shape=shape, dtype=np.float16)
            )
        else:
            input_specs.append(ct.TensorType(shape=shape))

    compute_precision = ct.precision.FLOAT16
    mlmodel = ct.convert(
        traced,
        inputs=input_specs,
        outputs=[ct.TensorType(name=n) for n in output_names],
        compute_units=compute_units,
        compute_precision=compute_precision,
        minimum_deployment_target=ct.target.iOS17,
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


def save_config(output_dir):
    """Save the model configuration."""
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
        "vocabSize": 8192,
        "blankTokenId": 8192,
        "numDurationBins": 5,
        "durationBins": [0, 1, 2, 3, 4],
    }

    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Saved config to {config_path}")


def compile_mlpackage(output_dir, name):
    """Compile .mlpackage to .mlmodelc for distribution."""
    pkg_path = output_dir / f"{name}.mlpackage"
    compiled_path = output_dir / f"{name}.mlmodelc"

    if compiled_path.exists():
        shutil.rmtree(compiled_path)

    print(f"  Compiling {name}.mlpackage → {name}.mlmodelc ...")
    compiled = ct.models.MLModel(str(pkg_path), compute_units=ct.ComputeUnit.ALL)
    compiled_url = ct.utils.compile_model(compiled)
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

    # Load model
    print("Loading NeMo model...")
    model = load_nemo_model()

    # Extract vocabulary
    print("Extracting vocabulary...")
    extract_vocab(model, output_dir)

    # Save config
    print("Saving configuration...")
    save_config(output_dir)

    # Preprocessor
    print("Converting preprocessor...")
    preprocessor = PreprocessorWrapper(model.preprocessor)
    example_audio = torch.randn(1, 16000)  # 1 second
    preprocessor_ml = trace_and_convert(
        preprocessor,
        (example_audio,),
        "preprocessor",
        ct.ComputeUnit.CPU_ONLY,
        ["mel", "mel_length"],
    )
    preprocessor_ml.save(str(output_dir / "preprocessor.mlpackage"))

    # Get mel output shape for encoder input
    with torch.no_grad():
        mel, mel_len = preprocessor(example_audio)

    # Encoder
    print("Converting encoder...")
    encoder = EncoderWrapper(model.encoder)
    encoder_ml = trace_and_convert(
        encoder,
        (mel, mel_len),
        "encoder",
        ct.ComputeUnit.CPU_AND_NE,
        ["encoded", "encoded_length"],
    )
    if not args.no_quantize:
        print("  Quantizing encoder to INT4...")
        encoder_ml = quantize_encoder(encoder_ml)
    encoder_ml.save(str(output_dir / "encoder.mlpackage"))

    # Decoder
    print("Converting decoder...")
    decoder = DecoderWrapper(model.decoder)
    example_token = torch.tensor([[8192]], dtype=torch.long)  # blank
    example_h = torch.zeros(2, 1, 640)
    example_c = torch.zeros(2, 1, 640)
    decoder_ml = trace_and_convert(
        decoder,
        (example_token, example_h, example_c),
        "decoder",
        ct.ComputeUnit.CPU_AND_NE,
        ["decoder_output", "h_out", "c_out"],
    )
    decoder_ml.save(str(output_dir / "decoder.mlpackage"))

    # Joint
    print("Converting joint network...")
    joint = JointWrapper(model.joint)
    with torch.no_grad():
        encoded, _ = encoder(mel, mel_len)
    example_enc_slice = encoded[:, :1, :]  # [1, 1, 1024]
    with torch.no_grad():
        dec_out, _, _ = decoder(example_token, example_h, example_c)
    joint_ml = trace_and_convert(
        joint,
        (example_enc_slice, dec_out),
        "joint",
        ct.ComputeUnit.CPU_AND_NE,
        ["token_logits", "duration_logits"],
    )
    joint_ml.save(str(output_dir / "joint.mlpackage"))

    # Optionally compile
    if args.compile:
        print("Compiling CoreML models...")
        for name in ["preprocessor", "encoder", "decoder", "joint"]:
            compile_mlpackage(output_dir, name)

    print(f"\nDone! Models saved to {output_dir}/")
    print("Files:")
    for f in sorted(output_dir.iterdir()):
        size = sum(
            ff.stat().st_size for ff in f.rglob("*") if ff.is_file()
        ) if f.is_dir() else f.stat().st_size
        print(f"  {f.name}: {size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
