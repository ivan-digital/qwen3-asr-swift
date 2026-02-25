#!/usr/bin/env python3
"""Convert Silero VAD v5 (JIT) to MLX safetensors format.

Usage:
    python3 scripts/convert_silero_vad.py [--output silero_vad_mlx] [--upload]

Downloads the Silero VAD v5 model via torch.hub, extracts weights from the
JIT model, remaps keys for MLX Conv1d (channels-last), sums LSTM biases,
and saves as safetensors + config.json.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from safetensors.numpy import save_file


def extract_jit_state_dict(model):
    """Extract all parameters and buffers from a JIT ScriptModule."""
    state_dict = {}

    # state_dict() works on JIT models
    for name, tensor in model.state_dict().items():
        state_dict[name] = tensor.detach().cpu().numpy()

    # Also grab named buffers (forward_basis_buffer is a buffer, not a parameter)
    try:
        for name, tensor in model.named_buffers():
            if name not in state_dict:
                state_dict[name] = tensor.detach().cpu().numpy()
    except Exception:
        pass

    return state_dict


def convert_weights(state_dict, verbose=True):
    """Map Silero VAD JIT keys to MLX keys with appropriate transpositions."""
    mlx_weights = {}

    if verbose:
        print("Source keys:")
        for k, v in sorted(state_dict.items()):
            print(f"  {k}: {v.shape} {v.dtype}")
        print()

    # --- STFT ---
    # forward_basis_buffer: [n_fft+2, 1, filter_length] = [258, 1, 256]
    # MLX Conv1d: [out_channels, kernel_size, in_channels] = [258, 256, 1]
    stft_key = None
    for k in state_dict:
        if "stft" in k and "forward_basis" in k:
            stft_key = k
            break
    if stft_key is None:
        raise KeyError("Could not find STFT forward_basis_buffer in state dict")

    stft_w = state_dict[stft_key].astype(np.float32)
    if stft_w.ndim == 4:
        stft_w = stft_w.squeeze(0)  # Remove batch dim if present
    assert stft_w.ndim == 3, f"Expected 3D STFT weight, got shape {stft_w.shape}"
    # [O, I, K] → [O, K, I]
    mlx_weights["stft.weight"] = stft_w.transpose(0, 2, 1)

    # --- Encoder: 4 Conv1d layers ---
    for i in range(4):
        # Try reparam_conv first (fused weights), then plain conv
        w_key = None
        b_key = None
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
            raise KeyError(f"Could not find encoder layer {i} weights")

        w = state_dict[w_key].astype(np.float32)
        # PyTorch Conv1d [O, I, K] → MLX Conv1d [O, K, I]
        mlx_weights[f"encoder.{i}.weight"] = w.transpose(0, 2, 1)

        if b_key and b_key in state_dict:
            mlx_weights[f"encoder.{i}.bias"] = state_dict[b_key].astype(np.float32)

    # --- LSTM ---
    # Silero uses: _model.decoder.rnn (a single-layer LSTM)
    lstm_prefix = None
    for prefix in ["_model.decoder.rnn", "_model.rnn"]:
        if f"{prefix}.weight_ih" in state_dict:
            lstm_prefix = prefix
            break
    if lstm_prefix is None:
        # Try weight_ih_l0 (multi-layer LSTM format)
        for prefix in ["_model.decoder.rnn", "_model.rnn"]:
            if f"{prefix}.weight_ih_l0" in state_dict:
                lstm_prefix = prefix
                break

    if lstm_prefix is None:
        raise KeyError("Could not find LSTM weights in state dict")

    # Handle both single-layer and multi-layer LSTM key formats
    if f"{lstm_prefix}.weight_ih" in state_dict:
        wih = state_dict[f"{lstm_prefix}.weight_ih"].astype(np.float32)
        whh = state_dict[f"{lstm_prefix}.weight_hh"].astype(np.float32)
        bih = state_dict[f"{lstm_prefix}.bias_ih"].astype(np.float32)
        bhh = state_dict[f"{lstm_prefix}.bias_hh"].astype(np.float32)
    else:
        wih = state_dict[f"{lstm_prefix}.weight_ih_l0"].astype(np.float32)
        whh = state_dict[f"{lstm_prefix}.weight_hh_l0"].astype(np.float32)
        bih = state_dict[f"{lstm_prefix}.bias_ih_l0"].astype(np.float32)
        bhh = state_dict[f"{lstm_prefix}.bias_hh_l0"].astype(np.float32)

    mlx_weights["lstm.Wx"] = wih  # [4*H, input_size]
    mlx_weights["lstm.Wh"] = whh  # [4*H, hidden_size]
    mlx_weights["lstm.bias"] = bih + bhh  # Sum biases [4*H]

    # --- Decoder: Conv1d(128, 1, k=1) ---
    # decoder.decoder is Sequential: [0]=ReLU, [1]=Dropout, [2]=Conv1d
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
        # Find any decoder conv weight
        for k in state_dict:
            if "decoder" in k and "decoder" in k and "weight" in k and "rnn" not in k:
                dec_key = k
                break
    if dec_key is None:
        raise KeyError("Could not find decoder conv weight")

    dec_bias_key = dec_key.replace(".weight", ".bias")

    w = state_dict[dec_key].astype(np.float32)
    # [O, I, K] → [O, K, I]
    mlx_weights["decoder.weight"] = w.transpose(0, 2, 1)

    if dec_bias_key in state_dict:
        mlx_weights["decoder.bias"] = state_dict[dec_bias_key].astype(np.float32)

    return mlx_weights


def main():
    parser = argparse.ArgumentParser(description="Convert Silero VAD v5 to MLX format")
    parser.add_argument(
        "--output", default="silero_vad_mlx", help="Output directory"
    )
    parser.add_argument(
        "--upload", action="store_true", help="Upload to HuggingFace"
    )
    parser.add_argument(
        "--repo-id",
        default="aufklarer/Silero-VAD-v5-MLX",
        help="HuggingFace repo ID for upload",
    )
    args = parser.parse_args()

    print("Loading Silero VAD v5 via torch.hub...")
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        trust_repo=True,
    )
    print(f"Model type: {type(model)}")

    print("\nExtracting state dict...")
    state_dict = extract_jit_state_dict(model)
    print(f"Found {len(state_dict)} tensors")

    print("\nConverting weights...")
    mlx_weights = convert_weights(state_dict)

    # Save
    os.makedirs(args.output, exist_ok=True)

    weights_path = os.path.join(args.output, "model.safetensors")
    save_file(mlx_weights, weights_path)
    print(f"\nSaved weights to {weights_path}")

    # Print output shapes
    print("\nMLX weight shapes:")
    total_params = 0
    for k, v in sorted(mlx_weights.items()):
        print(f"  {k}: {v.shape}")
        total_params += v.size
    print(f"\nTotal parameters: {total_params:,}")

    # Config
    config = {
        "model_type": "silero_vad_v5",
        "sample_rate": 16000,
        "chunk_size": 512,
        "context_size": 64,
        "filter_length": 256,
        "hop_length": 128,
        "encoder_channels": [129, 128, 64, 64, 128],
        "encoder_kernel_sizes": [3, 3, 3, 3],
        "encoder_strides": [1, 2, 2, 1],
        "lstm_hidden_size": 128,
        "lstm_num_layers": 1,
    }
    config_path = os.path.join(args.output, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved config to {config_path}")

    # Verify shapes
    print("\nVerifying weight shapes...")
    assert mlx_weights["stft.weight"].shape == (258, 256, 1), (
        f"STFT weight shape mismatch: {mlx_weights['stft.weight'].shape}"
    )
    assert mlx_weights["lstm.Wx"].shape[0] == 512, (
        f"LSTM Wx shape mismatch: {mlx_weights['lstm.Wx'].shape}"
    )
    assert mlx_weights["lstm.Wh"].shape == (512, 128), (
        f"LSTM Wh shape mismatch: {mlx_weights['lstm.Wh'].shape}"
    )
    assert mlx_weights["lstm.bias"].shape == (512,), (
        f"LSTM bias shape mismatch: {mlx_weights['lstm.bias'].shape}"
    )
    print("All shapes verified!")

    # Upload
    if args.upload:
        try:
            from huggingface_hub import HfApi

            api = HfApi()
            api.create_repo(args.repo_id, exist_ok=True)
            api.upload_folder(
                folder_path=args.output,
                repo_id=args.repo_id,
            )
            print(f"\nUploaded to {args.repo_id}")
        except Exception as e:
            print(f"\nUpload failed: {e}")
            print(
                f"Upload manually: huggingface-cli upload {args.repo_id} {args.output}/"
            )


if __name__ == "__main__":
    main()
