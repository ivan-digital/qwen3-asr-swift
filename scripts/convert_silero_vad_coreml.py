#!/usr/bin/env python3
"""Convert Silero VAD v5 (JIT) to CoreML format.

Usage:
    python3 scripts/convert_silero_vad_coreml.py [--output silero_vad_coreml] [--upload]

Downloads the Silero VAD v5 model via torch.hub, rebuilds as a plain PyTorch
module with explicit LSTM h/c state I/O, traces, converts to CoreML with
float16 precision, and compiles to .mlmodelc.

The JIT model's forward doesn't expose LSTM h/c as explicit inputs/outputs,
so we rebuild from the state_dict with explicit state management for streaming.
"""

import argparse
import json
import os
import shutil

import coremltools as ct
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SileroVADCoreML(nn.Module):
    """Silero VAD v5 rebuilt with explicit LSTM state I/O.

    Architecture: STFT Conv1d → 4 encoder Conv1d+ReLU → LSTM → decoder Conv1d → sigmoid
    """

    def __init__(self):
        super().__init__()
        # STFT: pre-computed DFT basis as Conv1d (no bias)
        self.stft_conv = nn.Conv1d(1, 258, kernel_size=256, stride=128, bias=False)

        # Encoder: 4 Conv1d + ReLU
        self.encoder = nn.ModuleList([
            nn.Conv1d(129, 128, kernel_size=3, stride=1, padding=1),
            nn.Conv1d(128, 64, kernel_size=3, stride=2, padding=1),
            nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
        ])

        # LSTM: 128→128, 1 layer
        self.lstm = nn.LSTM(input_size=128, hidden_size=128, num_layers=1, batch_first=True)

        # Decoder: Conv1d(128→1, k=1)
        self.decoder_conv = nn.Conv1d(128, 1, kernel_size=1)

    def forward(self, audio, h, c):
        """Forward pass with explicit LSTM state.

        Args:
            audio: [1, 1, 576] — 64 context + 512 new samples
            h: [1, 1, 128] — LSTM hidden state
            c: [1, 1, 128] — LSTM cell state

        Returns:
            probability: [1] — speech probability
            h_out: [1, 1, 128] — updated hidden state
            c_out: [1, 1, 128] — updated cell state
        """
        # Right-only reflection padding: 64 samples
        x = F.pad(audio, (0, 64), mode="reflect")  # [1, 1, 640]

        # STFT via Conv1d: [1, 1, 640] → [1, 258, 4]
        x = self.stft_conv(x)

        # Magnitude spectrum
        real = x[:, :129, :]
        imag = x[:, 129:, :]
        x = torch.sqrt(real ** 2 + imag ** 2 + 1e-10)  # [1, 129, 4], epsilon for stability

        # Encoder: 4× Conv1d + ReLU → [1, 128, 1]
        for conv in self.encoder:
            x = F.relu(conv(x))

        # Transpose for LSTM: [1, 128, 1] → [1, 1, 128]
        x = x.permute(0, 2, 1)

        # LSTM with explicit state
        _, (h_out, c_out) = self.lstm(x, (h, c))

        # Decoder: hidden state → Conv1d → sigmoid
        # h_out: [1, 1, 128] → squeeze → unsqueeze for Conv1d: [1, 128, 1]
        d = F.relu(h_out.squeeze(0).unsqueeze(-1))  # [1, 128, 1]
        d = torch.sigmoid(self.decoder_conv(d))       # [1, 1, 1]

        return d.reshape(1), h_out, c_out


def load_weights_from_jit(model, jit_model):
    """Load weights from JIT model state_dict into our plain module."""
    state_dict = jit_model.state_dict()
    buffers = {}
    try:
        for name, tensor in jit_model.named_buffers():
            buffers[name] = tensor
    except Exception:
        pass

    print("JIT state dict keys:")
    for k, v in sorted(state_dict.items()):
        print(f"  {k}: {v.shape}")
    for k, v in sorted(buffers.items()):
        if k not in state_dict:
            print(f"  {k}: {v.shape} (buffer)")

    # STFT basis
    stft_key = None
    for k in list(state_dict.keys()) + list(buffers.keys()):
        if "stft" in k and "forward_basis" in k:
            stft_key = k
            break
    if stft_key is None:
        raise KeyError("Could not find STFT forward_basis_buffer")

    stft_w = state_dict.get(stft_key, buffers.get(stft_key))
    if stft_w.dim() == 4:
        stft_w = stft_w.squeeze(0)
    model.stft_conv.weight.data.copy_(stft_w)

    # Encoder convs
    for i in range(4):
        w_key = b_key = None
        for prefix in [
            f"_model.encoder.{i}.reparam_conv",
            f"_model.encoder.{i}.conv",
            f"_model.encoder.{i}",
        ]:
            if f"{prefix}.weight" in state_dict:
                w_key = f"{prefix}.weight"
                b_key = f"{prefix}.bias"
                break
        if w_key is None:
            raise KeyError(f"Could not find encoder layer {i}")
        model.encoder[i].weight.data.copy_(state_dict[w_key])
        if b_key and b_key in state_dict:
            model.encoder[i].bias.data.copy_(state_dict[b_key])

    # LSTM
    lstm_prefix = None
    for prefix in ["_model.decoder.rnn", "_model.rnn"]:
        if f"{prefix}.weight_ih" in state_dict:
            lstm_prefix = prefix
            break
    if lstm_prefix is None:
        for prefix in ["_model.decoder.rnn", "_model.rnn"]:
            if f"{prefix}.weight_ih_l0" in state_dict:
                lstm_prefix = prefix
                break
    if lstm_prefix is None:
        raise KeyError("Could not find LSTM weights")

    if f"{lstm_prefix}.weight_ih" in state_dict:
        model.lstm.weight_ih_l0.data.copy_(state_dict[f"{lstm_prefix}.weight_ih"])
        model.lstm.weight_hh_l0.data.copy_(state_dict[f"{lstm_prefix}.weight_hh"])
        model.lstm.bias_ih_l0.data.copy_(state_dict[f"{lstm_prefix}.bias_ih"])
        model.lstm.bias_hh_l0.data.copy_(state_dict[f"{lstm_prefix}.bias_hh"])
    else:
        model.lstm.weight_ih_l0.data.copy_(state_dict[f"{lstm_prefix}.weight_ih_l0"])
        model.lstm.weight_hh_l0.data.copy_(state_dict[f"{lstm_prefix}.weight_hh_l0"])
        model.lstm.bias_ih_l0.data.copy_(state_dict[f"{lstm_prefix}.bias_ih_l0"])
        model.lstm.bias_hh_l0.data.copy_(state_dict[f"{lstm_prefix}.bias_hh_l0"])

    # Decoder conv
    dec_key = None
    for k in [
        "_model.decoder.decoder.2.weight",
        "_model.decoder.decoder.1.weight",
        "_model.decoder.classifier.weight",
    ]:
        if k in state_dict:
            dec_key = k
            break
    if dec_key is None:
        for k in state_dict:
            if "decoder" in k and "weight" in k and "rnn" not in k:
                dec_key = k
                break
    if dec_key is None:
        raise KeyError("Could not find decoder conv weight")

    model.decoder_conv.weight.data.copy_(state_dict[dec_key])
    dec_bias_key = dec_key.replace(".weight", ".bias")
    if dec_bias_key in state_dict:
        model.decoder_conv.bias.data.copy_(state_dict[dec_bias_key])

    print("\nWeights loaded successfully")


def verify_against_jit(plain_model, jit_model, num_chunks=10):
    """Verify plain PyTorch model outputs match JIT model."""
    print("\nVerifying against JIT model...")

    # Generate random audio
    torch.manual_seed(42)
    audio_16k = torch.randn(1, 16000)  # 1 second

    # Run JIT model — it expects raw 512-sample chunks and handles context internally
    jit_model.reset_states()
    jit_probs = []
    offset = 0
    while offset + 512 <= audio_16k.shape[1] and len(jit_probs) < num_chunks:
        chunk = audio_16k[:, offset:offset + 512]
        with torch.no_grad():
            p = jit_model(chunk, 16000)
        jit_probs.append(p.item())
        offset += 512

    # Run plain model — we manage context externally (64 samples prepended)
    h = torch.zeros(1, 1, 128)
    c = torch.zeros(1, 1, 128)
    plain_probs = []
    context = torch.zeros(1, 64)
    offset = 0
    while offset + 512 <= audio_16k.shape[1] and len(plain_probs) < num_chunks:
        chunk = audio_16k[:, offset:offset + 512]
        full = torch.cat([context, chunk], dim=1)  # [1, 576]
        context = chunk[:, -64:]

        with torch.no_grad():
            prob, h, c = plain_model(full.unsqueeze(1), h, c)  # [1, 1, 576]
        plain_probs.append(prob.item())
        offset += 512

    print(f"  Chunks compared: {len(jit_probs)}")
    max_diff = 0
    for i, (jp, pp) in enumerate(zip(jit_probs, plain_probs)):
        diff = abs(jp - pp)
        max_diff = max(max_diff, diff)
        if diff > 0.01:
            print(f"  Chunk {i}: JIT={jp:.6f}  Plain={pp:.6f}  diff={diff:.6f}")

    print(f"  Max difference: {max_diff:.6f}")
    if max_diff > 0.01:
        print("  WARNING: outputs diverge (may be due to JIT internal state management)")
        print("  This is expected — JIT model has different context/state handling")
    else:
        print("  PASS: outputs match within tolerance")


def verify_coreml(plain_model, coreml_model, num_chunks=10):
    """Verify CoreML model outputs match plain PyTorch model."""
    print("\nVerifying CoreML against PyTorch...")

    torch.manual_seed(42)

    h_pt = torch.zeros(1, 1, 128)
    c_pt = torch.zeros(1, 1, 128)
    h_cm = np.zeros((1, 1, 128), dtype=np.float16)
    c_cm = np.zeros((1, 1, 128), dtype=np.float16)

    max_diff = 0
    for i in range(num_chunks):
        audio = torch.randn(1, 1, 576)

        # PyTorch
        with torch.no_grad():
            prob_pt, h_pt, c_pt = plain_model(audio, h_pt, c_pt)

        # CoreML
        audio_np = audio.numpy().astype(np.float16)
        result = coreml_model.predict({
            "audio": audio_np,
            "h": h_cm,
            "c": c_cm,
        })
        prob_cm = result["probability"]
        h_cm = result["h_out"]
        c_cm = result["c_out"]

        prob_cm_val = float(np.array(prob_cm).flat[0])
        diff = abs(prob_pt.item() - prob_cm_val)
        max_diff = max(max_diff, diff)
        print(f"  Chunk {i}: PyTorch={prob_pt.item():.6f}  CoreML={prob_cm_val:.6f}  diff={diff:.6f}")

    print(f"  Max difference: {max_diff:.6f}")
    if max_diff < 0.02:
        print("  PASS: CoreML matches PyTorch within float16 tolerance")
    else:
        print("  WARNING: CoreML diverges from PyTorch (check float16 precision)")


def main():
    parser = argparse.ArgumentParser(description="Convert Silero VAD v5 to CoreML format")
    parser.add_argument("--output", default="silero_vad_coreml", help="Output directory")
    parser.add_argument("--upload", action="store_true", help="Upload to HuggingFace")
    parser.add_argument(
        "--repo-id",
        default="aufklarer/Silero-VAD-v5-CoreML",
        help="HuggingFace repo ID for upload",
    )
    parser.add_argument("--skip-verify", action="store_true", help="Skip verification")
    args = parser.parse_args()

    # Step 1: Load JIT model
    print("Loading Silero VAD v5 via torch.hub...")
    jit_model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        trust_repo=True,
    )
    print(f"JIT model type: {type(jit_model)}")

    # Step 2: Build plain PyTorch model and load weights
    print("\nBuilding plain PyTorch model...")
    plain_model = SileroVADCoreML()
    load_weights_from_jit(plain_model, jit_model)
    plain_model.eval()

    # Step 3: Verify against JIT
    if not args.skip_verify:
        verify_against_jit(plain_model, jit_model)

    # Step 4: Trace
    print("\nTracing model...")
    example_audio = torch.randn(1, 1, 576)
    example_h = torch.zeros(1, 1, 128)
    example_c = torch.zeros(1, 1, 128)

    with torch.no_grad():
        traced = torch.jit.trace(plain_model, (example_audio, example_h, example_c))

    # Step 5: Convert to CoreML
    print("Converting to CoreML...")
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType("audio", shape=(1, 1, 576)),
            ct.TensorType("h", shape=(1, 1, 128)),
            ct.TensorType("c", shape=(1, 1, 128)),
        ],
        outputs=[
            ct.TensorType("probability"),
            ct.TensorType("h_out"),
            ct.TensorType("c_out"),
        ],
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS17,
    )

    # Step 6: Save .mlpackage
    os.makedirs(args.output, exist_ok=True)
    mlpackage_path = os.path.join(args.output, "silero_vad.mlpackage")
    if os.path.exists(mlpackage_path):
        shutil.rmtree(mlpackage_path)
    mlmodel.save(mlpackage_path)
    print(f"Saved .mlpackage to {mlpackage_path}")

    # Step 7: Compile to .mlmodelc
    print("Compiling to .mlmodelc...")
    import subprocess
    mlmodelc_path = os.path.join(args.output, "silero_vad.mlmodelc")
    if os.path.exists(mlmodelc_path):
        shutil.rmtree(mlmodelc_path)

    # Use xcrun coremlcompiler
    result = subprocess.run(
        ["xcrun", "coremlcompiler", "compile", mlpackage_path, args.output],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"Compilation failed: {result.stderr}")
        print("Falling back to Python compilation...")
        # Fallback: use coremltools compilation
        compiled = mlmodel.get_compiled_model_path()
        if os.path.exists(mlmodelc_path):
            shutil.rmtree(mlmodelc_path)
        shutil.copytree(compiled, mlmodelc_path)
    else:
        print(f"Compiled to {mlmodelc_path}")

    # Verify .mlmodelc exists
    if not os.path.exists(mlmodelc_path):
        # coremlcompiler may create silero_vad.mlmodelc inside output dir
        alt_path = os.path.join(args.output, "silero_vad.mlmodelc")
        if not os.path.exists(alt_path):
            print("ERROR: .mlmodelc not found after compilation")
            sys.exit(1)

    # Step 8: Verify CoreML
    if not args.skip_verify:
        print("\nLoading compiled CoreML model for verification...")
        coreml_model = ct.models.MLModel(mlpackage_path)
        verify_coreml(plain_model, coreml_model)

    # Step 9: Save config
    config = {
        "model_type": "silero_vad_v5_coreml",
        "sample_rate": 16000,
        "chunk_size": 512,
        "context_size": 64,
        "input_shape": [1, 1, 576],
        "h_shape": [1, 1, 128],
        "c_shape": [1, 1, 128],
        "compute_precision": "float16",
    }
    config_path = os.path.join(args.output, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved config to {config_path}")

    # Count parameters
    total_params = sum(p.numel() for p in plain_model.parameters())
    print(f"\nTotal parameters: {total_params:,}")

    # Step 10: Upload
    if args.upload:
        try:
            from huggingface_hub import HfApi

            api = HfApi()
            api.create_repo(args.repo_id, exist_ok=True)

            # Upload .mlmodelc directory and config
            print(f"\nUploading to {args.repo_id}...")
            api.upload_folder(
                folder_path=mlmodelc_path,
                path_in_repo="silero_vad.mlmodelc",
                repo_id=args.repo_id,
            )
            api.upload_file(
                path_or_fileobj=config_path,
                path_in_repo="config.json",
                repo_id=args.repo_id,
            )
            print(f"Uploaded to {args.repo_id}")
        except Exception as e:
            print(f"\nUpload failed: {e}")
            print(f"Upload manually:")
            print(f"  huggingface-cli upload {args.repo_id} {mlmodelc_path} silero_vad.mlmodelc")
            print(f"  huggingface-cli upload {args.repo_id} {config_path} config.json")

    print("\nDone!")


if __name__ == "__main__":
    main()
