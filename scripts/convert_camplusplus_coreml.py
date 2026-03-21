#!/usr/bin/env python3
"""Convert CAM++ speaker embedding model to CoreML with EnumeratedShapes.

Source: campplus.onnx from FunAudioLLM/Fun-CosyVoice3-0.5B-2512
Pipeline: ONNX → onnx2torch → torch.jit.trace → CoreML FP16

Uses EnumeratedShapes for variable mel frame dimension to avoid fixed-size
padding/truncation that hurts short utterances.

Requires: pip install onnx onnx2torch torch coremltools huggingface_hub

Usage:
    python3 scripts/convert_camplusplus_coreml.py
    python3 scripts/convert_camplusplus_coreml.py --upload
"""

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

import coremltools as ct
import numpy as np
import onnx
from huggingface_hub import hf_hub_download

# Enumerated T values for mel frame dimension.
# T=20 ~ 0.2s, T=100 ~ 1s, T=500 ~ 5s, T=3000 ~ 30s
T_VALUES = [20, 50, 100, 200, 300, 500, 750, 1000, 1500, 2000, 3000]

SOURCE_REPO = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"
ONNX_FILE = "campplus.onnx"


def verify(coreml_model, onnx_path, t_values=None):
    """Verify CoreML vs ONNX outputs match."""
    import onnxruntime as ort

    if t_values is None:
        t_values = [100, 500]

    session = ort.InferenceSession(onnx_path)

    print("\nVerification (ONNX vs CoreML):")
    for t in t_values:
        x = np.random.randn(1, t, 80).astype(np.float32)
        onnx_out = session.run(None, {"mel_features": x})[0].flatten()
        cml_out = np.array(
            coreml_model.predict({"mel_features": x})["embedding"]
        ).flatten()

        diff = np.abs(onnx_out - cml_out).max()
        cos = np.dot(onnx_out, cml_out) / (
            np.linalg.norm(onnx_out) * np.linalg.norm(cml_out))
        print(f"  T={t:>4d}: max_diff={diff:.4f}, cosine={cos:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert CAM++ to CoreML with EnumeratedShapes")
    parser.add_argument("--output-dir", default="./camplusplus-coreml")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--repo-id",
                        default="aufklarer/CamPlusPlus-Speaker-CoreML")
    parser.add_argument("--skip-verify", action="store_true")
    args = parser.parse_args()

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Step 1: Download ONNX
    print(f"Downloading {ONNX_FILE} from {SOURCE_REPO}...")
    onnx_path = hf_hub_download(SOURCE_REPO, ONNX_FILE)
    print(f"  Downloaded to {onnx_path}")

    # Step 2: Convert ONNX → CoreML with EnumeratedShapes
    print(f"Converting ONNX → CoreML with EnumeratedShapes...")
    print(f"  T values: {T_VALUES}")

    shapes = [(1, t, 80) for t in T_VALUES]
    mel_input = ct.TensorType(
        "mel_features",
        shape=ct.EnumeratedShapes(shapes=shapes),
    )

    mlmodel = ct.convert(
        onnx_path,
        source="milinternal",
        inputs=[mel_input],
        outputs=[ct.TensorType("embedding")],
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS17,
    )

    # Step 6: Save .mlpackage
    mlpackage_path = str(output_path / "CamPlusPlus.mlpackage")
    if os.path.exists(mlpackage_path):
        shutil.rmtree(mlpackage_path)
    mlmodel.save(mlpackage_path)
    print(f"Saved .mlpackage to {mlpackage_path}")

    # Step 7: Compile to .mlmodelc
    print("Compiling to .mlmodelc...")
    mlmodelc_path = str(output_path / "CamPlusPlus.mlmodelc")
    if os.path.exists(mlmodelc_path):
        shutil.rmtree(mlmodelc_path)

    result = subprocess.run(
        ["xcrun", "coremlcompiler", "compile", mlpackage_path,
         str(output_path)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"xcrun failed: {result.stderr}")
        print("Falling back to Python compilation...")
        compiled = mlmodel.get_compiled_model_path()
        if os.path.exists(mlmodelc_path):
            shutil.rmtree(mlmodelc_path)
        shutil.copytree(compiled, mlmodelc_path)
    else:
        print(f"Compiled to {mlmodelc_path}")

    # Step 8: Verify
    if not args.skip_verify:
        coreml_model = ct.models.MLModel(mlpackage_path)
        verify(coreml_model, onnx_path)

    # Step 9: Save config
    config = {
        "model_type": "camplusplus-speaker-coreml",
        "sample_rate": 16000,
        "n_mels": 80,
        "embedding_dim": 192,
        "enumerated_mel_lengths": T_VALUES,
        "compute_precision": "float16",
        "source": f"{SOURCE_REPO}/{ONNX_FILE}",
    }
    config_path = str(output_path / "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved config to {config_path}")

    # Cleanup mlpackage (only need mlmodelc)
    if os.path.exists(mlpackage_path):
        shutil.rmtree(mlpackage_path)
        print("Removed .mlpackage (keeping .mlmodelc only)")

    print(f"\nConversion complete! Output in: {output_path}")

    # Step 10: Upload
    if args.upload:
        try:
            from huggingface_hub import HfApi

            api = HfApi()
            api.create_repo(args.repo_id, exist_ok=True)

            print(f"\nUploading to {args.repo_id}...")
            api.upload_folder(
                folder_path=mlmodelc_path,
                path_in_repo="CamPlusPlus.mlmodelc",
                repo_id=args.repo_id,
            )
            api.upload_file(
                path_or_fileobj=config_path,
                path_in_repo="config.json",
                repo_id=args.repo_id,
            )
            print(f"Uploaded to https://huggingface.co/{args.repo_id}")
        except Exception as e:
            print(f"\nUpload failed: {e}")

    print("\nDone!")


if __name__ == "__main__":
    main()
