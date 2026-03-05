#!/usr/bin/env python3
"""
VoxConverse DER benchmark for qwen3-asr-swift diarization pipeline.

Downloads VoxConverse test set, runs diarization via CLI, computes DER.
Uses dscore for standard evaluation when available, falls back to built-in scoring.

Usage:
    python scripts/benchmark_diarization.py [--cli-path .build/release/audio] [--num-files 5]
    python scripts/benchmark_diarization.py --download-only
    python scripts/benchmark_diarization.py --score-only
"""

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

VOXCONVERSE_BASE = "https://www.robots.ox.ac.uk/~vgg/data/voxconverse"
VOXCONVERSE_TEST_AUDIO = f"{VOXCONVERSE_BASE}/data/voxconverse_test_wav.zip"  # ~4.2 GB
VOXCONVERSE_TEST_RTTM = "https://raw.githubusercontent.com/joonson/voxconverse/master/test"

BENCHMARK_DIR = Path("benchmarks/voxconverse")
AUDIO_DIR = BENCHMARK_DIR / "audio"
REF_DIR = BENCHMARK_DIR / "ref"
HYP_DIR = BENCHMARK_DIR / "hyp"
RESULTS_FILE = BENCHMARK_DIR / "results.json"

# VoxConverse test set files (subset for quick benchmarking)
VOXCONVERSE_TEST_FILES = [
    "abjxc", "afjiv", "aggyz", "ahkbf", "aigor",
    "ajfnv", "ajuog", "akxfp", "alyuf", "amwyy",
    "aobwi", "aofam", "aplmj", "aqgrp", "arnej",
    "asgft", "asgor", "ateyp", "aujgc", "auzru",
    "avfbp", "avpcl", "awgfk", "axgbr", "axxjq",
    "ayowh", "azmff", "babzf", "bakth", "baskq",
    "bbfbx", "bcfkm", "bcgir", "bdczo", "bdzgr",
    "bfbwk", "bfrxl", "bgnet", "bgoij", "bhcdr",
    "bhyij", "bjwrj", "bkfxq", "bkucv", "blakb",
    "bmion", "bnrth", "boxqv", "bpjqs", "bqfpj",
]


def download_voxconverse(num_files: int = 0):
    """Download VoxConverse test audio and reference RTTM files."""
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    REF_DIR.mkdir(parents=True, exist_ok=True)
    HYP_DIR.mkdir(parents=True, exist_ok=True)

    files = VOXCONVERSE_TEST_FILES[:num_files] if num_files > 0 else VOXCONVERSE_TEST_FILES

    # Download reference RTTM files from GitHub
    print(f"Downloading {len(files)} reference RTTM files...")
    for name in files:
        rttm_path = REF_DIR / f"{name}.rttm"
        if rttm_path.exists():
            continue
        url = f"{VOXCONVERSE_TEST_RTTM}/{name}.rttm"
        try:
            urllib.request.urlretrieve(url, rttm_path)
            print(f"  Downloaded {name}.rttm")
        except Exception as e:
            print(f"  Failed to download {name}.rttm: {e}")

    # Check if audio needs downloading
    missing_audio = [f for f in files if not (AUDIO_DIR / f"{f}.wav").exists()]
    if not missing_audio:
        print("All audio files already present.")
        return

    # Download full test set zip (~4.2 GB)
    zip_path = BENCHMARK_DIR / "voxconverse_test_wav.zip"
    if not zip_path.exists():
        print(f"Downloading VoxConverse test audio zip (~4.2 GB)...")
        print(f"  From: {VOXCONVERSE_TEST_AUDIO}")
        print(f"  To:   {zip_path}")
        print(f"  (This may take a while. You can also download manually with:")
        print(f"   curl -L -o {zip_path} '{VOXCONVERSE_TEST_AUDIO}')")
        try:
            subprocess.run(
                ["curl", "-L", "-o", str(zip_path), "--progress-bar", VOXCONVERSE_TEST_AUDIO],
                check=True
            )
            print(f"  Downloaded to {zip_path}")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"  Failed to download zip: {e}")
            print(f"  Download manually and place at: {zip_path}")
            return

    # Extract needed files
    import zipfile
    print(f"Extracting {len(missing_audio)} audio files from zip...")
    with zipfile.ZipFile(zip_path) as zf:
        for name in missing_audio:
            # Files may be in a subdirectory within the zip
            matches = [n for n in zf.namelist() if n.endswith(f"{name}.wav")]
            if matches:
                with zf.open(matches[0]) as src, open(AUDIO_DIR / f"{name}.wav", "wb") as dst:
                    dst.write(src.read())
                print(f"  Extracted {name}.wav")
            else:
                print(f"  {name}.wav not found in zip")


def run_diarization(cli_path: str, num_files: int = 0, engine: str = "mlx"):
    """Run diarization on VoxConverse test files."""
    files = sorted(AUDIO_DIR.glob("*.wav"))
    if num_files > 0:
        files = files[:num_files]

    if not files:
        print("No audio files found. Run with --download-only first.")
        sys.exit(1)

    print(f"\nRunning diarization on {len(files)} files (engine: {engine})...")
    results = []

    for wav_path in files:
        name = wav_path.stem
        hyp_rttm = HYP_DIR / f"{name}.rttm"
        ref_rttm = REF_DIR / f"{name}.rttm"

        if not ref_rttm.exists():
            print(f"  Skipping {name} (no reference RTTM)")
            continue

        print(f"  Processing {name}...", end=" ", flush=True)
        start = time.time()

        try:
            result = subprocess.run(
                [cli_path, "diarize", str(wav_path), "--rttm",
                 "--embedding-engine", engine],
                capture_output=True, text=True, timeout=300
            )
            elapsed = time.time() - start

            if result.returncode != 0:
                print(f"FAILED ({elapsed:.1f}s)")
                stderr = result.stderr.strip()
                if stderr:
                    print(f"    Error: {stderr[:200]}")
                continue

            # Save hypothesis RTTM
            rttm_output = result.stdout.strip()
            # Filter to only SPEAKER lines (skip progress output)
            rttm_lines = [l for l in rttm_output.split("\n") if l.startswith("SPEAKER")]
            hyp_rttm.write_text("\n".join(rttm_lines) + "\n")

            # Get audio duration from reference
            ref_content = ref_rttm.read_text()
            ref_segments = parse_rttm(ref_content)
            hyp_segments = parse_rttm("\n".join(rttm_lines))

            ref_speakers = len(set(s["speaker"] for s in ref_segments))
            hyp_speakers = len(set(s["speaker"] for s in hyp_segments))

            print(f"OK ({elapsed:.1f}s, ref={ref_speakers}spk, hyp={hyp_speakers}spk)")

            results.append({
                "file": name,
                "elapsed": elapsed,
                "ref_speakers": ref_speakers,
                "hyp_speakers": hyp_speakers,
            })

        except subprocess.TimeoutExpired:
            print("TIMEOUT (>300s)")
        except Exception as e:
            print(f"ERROR: {e}")

    return results


def parse_rttm(content: str) -> list:
    """Parse RTTM content into segment dicts."""
    segments = []
    for line in content.strip().split("\n"):
        parts = line.split()
        if len(parts) < 8 or parts[0] != "SPEAKER":
            continue
        segments.append({
            "file": parts[1],
            "start": float(parts[3]),
            "duration": float(parts[4]),
            "speaker": parts[7],
        })
    return segments


def compute_der_dscore():
    """Compute DER using dscore if available."""
    dscore_path = Path("dscore/score.py")
    if not dscore_path.exists():
        print("\ndscore not found. To use standard scoring:")
        print("  git clone https://github.com/nryant/dscore.git")
        print("  pip install intervaltree tabulate")
        return None

    # Build file list
    hyp_files = sorted(HYP_DIR.glob("*.rttm"))
    if not hyp_files:
        print("No hypothesis files found.")
        return None

    ref_args = []
    hyp_args = []
    for hyp in hyp_files:
        ref = REF_DIR / hyp.name
        if ref.exists():
            ref_args.extend(["-r", str(ref)])
            hyp_args.extend(["-s", str(hyp)])

    if not ref_args:
        print("No matching reference files found.")
        return None

    print(f"\nScoring {len(ref_args) // 2} files with dscore (collar=0.25s)...")
    cmd = [sys.executable, str(dscore_path), "--collar", "0.25"] + ref_args + hyp_args
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"dscore failed: {result.stderr}")
        return None

    print(result.stdout)
    return result.stdout


def compute_der_builtin(cli_path: str):
    """Compute DER using built-in scoring in the CLI."""
    hyp_files = sorted(HYP_DIR.glob("*.rttm"))
    if not hyp_files:
        print("No hypothesis files found.")
        return

    print(f"\nScoring {len(hyp_files)} files with built-in DER (collar=0.25s)...")
    total_speech = 0
    total_fa = 0
    total_miss = 0
    total_conf = 0
    file_results = []

    for hyp in hyp_files:
        ref = REF_DIR / hyp.name
        wav = AUDIO_DIR / f"{hyp.stem}.wav"
        if not ref.exists() or not wav.exists():
            continue

        result = subprocess.run(
            [cli_path, "diarize", str(wav), "--rttm",
             "--score-against", str(ref)],
            capture_output=True, text=True, timeout=300
        )

        if result.returncode != 0:
            continue

        # Parse DER output
        lines = result.stdout.strip().split("\n")
        der_data = {}
        for line in lines:
            if "Total speech:" in line:
                der_data["total_speech"] = float(line.split(":")[1].strip().rstrip("s"))
            elif "Missed speech:" in line:
                der_data["missed"] = float(line.split(":")[1].strip().rstrip("s"))
            elif "False alarm:" in line:
                der_data["fa"] = float(line.split(":")[1].strip().rstrip("s"))
            elif "Confusion:" in line:
                der_data["confusion"] = float(line.split(":")[1].strip().rstrip("s"))
            elif "DER:" in line:
                der_data["der"] = float(line.split(":")[1].strip().rstrip("%"))

        if "der" in der_data:
            total_speech += der_data.get("total_speech", 0)
            total_fa += der_data.get("fa", 0)
            total_miss += der_data.get("missed", 0)
            total_conf += der_data.get("confusion", 0)
            file_results.append({"file": hyp.stem, **der_data})
            print(f"  {hyp.stem}: DER={der_data['der']:.1f}%")

    if total_speech > 0:
        aggregate_der = (total_fa + total_miss + total_conf) / total_speech * 100
        print(f"\n{'='*50}")
        print(f"Aggregate DER: {aggregate_der:.1f}%")
        print(f"  Total speech: {total_speech:.1f}s")
        print(f"  False alarm:  {total_fa:.1f}s ({total_fa/total_speech*100:.1f}%)")
        print(f"  Missed:       {total_miss:.1f}s ({total_miss/total_speech*100:.1f}%)")
        print(f"  Confusion:    {total_conf:.1f}s ({total_conf/total_speech*100:.1f}%)")
        print(f"{'='*50}")

        return {
            "aggregate_der": aggregate_der,
            "total_speech": total_speech,
            "false_alarm": total_fa,
            "missed_speech": total_miss,
            "confusion": total_conf,
            "per_file": file_results,
        }

    return None


def main():
    parser = argparse.ArgumentParser(description="VoxConverse DER benchmark")
    parser.add_argument("--cli-path", default=".build/release/audio",
                       help="Path to audio CLI binary")
    parser.add_argument("--num-files", type=int, default=0,
                       help="Number of test files (0 = all)")
    parser.add_argument("--engine", default="mlx", choices=["mlx", "coreml"],
                       help="Speaker embedding engine")
    parser.add_argument("--download-only", action="store_true",
                       help="Only download test data")
    parser.add_argument("--score-only", action="store_true",
                       help="Only score existing hypotheses")
    parser.add_argument("--use-dscore", action="store_true",
                       help="Use dscore for evaluation (must be cloned)")
    args = parser.parse_args()

    # Download
    if not args.score_only:
        download_voxconverse(args.num_files)

    if args.download_only:
        print("\nDownload complete.")
        return

    # Run diarization
    if not args.score_only:
        if not Path(args.cli_path).exists():
            print(f"CLI binary not found at {args.cli_path}")
            print("Build with: swift build -c release")
            sys.exit(1)

        run_results = run_diarization(args.cli_path, args.num_files, args.engine)

    # Score
    if args.use_dscore:
        compute_der_dscore()
    else:
        der_results = compute_der_builtin(args.cli_path)
        if der_results:
            RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(RESULTS_FILE, "w") as f:
                json.dump(der_results, f, indent=2)
            print(f"\nResults saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
