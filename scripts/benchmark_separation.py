#!/usr/bin/env python3
"""
Music source separation benchmark on MUSDB18-HQ.

Evaluates SDR (Signal-to-Distortion Ratio) per stem using museval.

Usage:
    pip install musdb museval
    python scripts/benchmark_separation.py
    python scripts/benchmark_separation.py --num-tracks 5  # quick test
    python scripts/benchmark_separation.py --download-only
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def check_dependencies():
    try:
        import musdb
        import museval
    except ImportError:
        print("Install: pip install musdb museval soundfile")
        sys.exit(1)


def run_separation(cli_path: str, audio_path: str, output_dir: str) -> float:
    """Run separation and return wall time."""
    start = time.monotonic()
    result = subprocess.run(
        [cli_path, "separate", audio_path, "--output-dir", output_dir],
        capture_output=True, text=True, timeout=600)
    elapsed = time.monotonic() - start
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr[:200]}")
    return elapsed


def compute_sdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Compute SDR (Signal-to-Distortion Ratio) in dB."""
    ref = reference.astype(np.float64)
    est = estimate.astype(np.float64)

    # Flatten to mono if needed
    if ref.ndim > 1:
        ref = ref.mean(axis=-1)
        est = est.mean(axis=-1)

    # Truncate to same length
    min_len = min(len(ref), len(est))
    ref = ref[:min_len]
    est = est[:min_len]

    noise = ref - est
    ref_energy = np.sum(ref ** 2)
    noise_energy = np.sum(noise ** 2)

    if noise_energy < 1e-10:
        return 100.0  # Perfect separation
    if ref_energy < 1e-10:
        return -100.0

    return 10 * np.log10(ref_energy / noise_energy)


def main():
    parser = argparse.ArgumentParser(description="Source separation benchmark")
    parser.add_argument("--num-tracks", type=int, default=0,
                        help="Number of test tracks (0=all, 50 for MUSDB18)")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--data-dir", default="benchmarks/data/musdb18")
    parser.add_argument("--cli", default=".build/release/audio")
    args = parser.parse_args()

    check_dependencies()
    import musdb
    import soundfile as sf

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Download MUSDB18 (7.4 GB)
    print("Loading MUSDB18 dataset...")
    db = musdb.DB(root=str(data_dir), download=True, subsets="test")
    print(f"  {len(db)} test tracks")

    if args.download_only:
        return

    tracks = db.tracks
    if args.num_tracks > 0:
        tracks = tracks[:args.num_tracks]

    results = {"vocals": [], "drums": [], "bass": [], "other": []}
    timings = []
    targets = ["vocals", "drums", "bass", "other"]

    for i, track in enumerate(tracks):
        print(f"\n[{i+1}/{len(tracks)}] {track.name}")

        # Export mix as WAV
        mix_dir = data_dir / "benchmark_tmp"
        mix_dir.mkdir(exist_ok=True)
        mix_path = mix_dir / f"{track.name}_mix.wav"
        sf.write(str(mix_path), track.audio, track.rate)

        # Run separation
        out_dir = str(mix_dir / f"{track.name}_stems")
        elapsed = run_separation(args.cli, str(mix_path), out_dir)
        duration = track.audio.shape[0] / track.rate
        rtf = elapsed / duration
        timings.append({"track": track.name, "duration": duration, "elapsed": elapsed, "rtf": rtf})
        print(f"  Time: {elapsed:.1f}s, RTF: {rtf:.2f}")

        # Compute SDR per stem
        for target in targets:
            stem_path = Path(out_dir) / f"{target}.wav"
            if not stem_path.exists():
                print(f"  {target}: MISSING")
                continue

            estimate, _ = sf.read(str(stem_path))
            reference = track.targets[target].audio

            # Ensure same length (mono comparison)
            if estimate.ndim == 1:
                estimate = estimate[:, np.newaxis]
            if reference.ndim == 1:
                reference = reference[:, np.newaxis]

            sdr = compute_sdr(reference, estimate)
            results[target].append(sdr)
            print(f"  {target}: {sdr:.2f} dB SDR")

    # Summary
    print("\n" + "=" * 60)
    print("Source Separation Benchmark Results")
    print("=" * 60)
    print(f"Tracks: {len(tracks)}")
    print(f"Model: Open-Unmix HQ (MLX)")
    print()

    print("SDR (dB) — higher is better:")
    print(f"  {'Target':<10} {'Mean':>8} {'Median':>8} {'Min':>8} {'Max':>8}")
    print(f"  {'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    all_sdrs = []
    for target in targets:
        sdrs = results[target]
        if sdrs:
            mean = np.mean(sdrs)
            median = np.median(sdrs)
            all_sdrs.extend(sdrs)
            print(f"  {target:<10} {mean:>8.2f} {median:>8.2f} {min(sdrs):>8.2f} {max(sdrs):>8.2f}")
    if all_sdrs:
        print(f"  {'AVERAGE':<10} {np.mean(all_sdrs):>8.2f} {np.median(all_sdrs):>8.2f}")

    print()
    rtfs = [t["rtf"] for t in timings]
    print(f"Performance:")
    print(f"  Mean RTF: {np.mean(rtfs):.3f}")
    print(f"  Total time: {sum(t['elapsed'] for t in timings):.0f}s")
    print(f"  Total audio: {sum(t['duration'] for t in timings):.0f}s")

    # Save results
    output = {
        "model": "OpenUnmix-HQ-MLX",
        "tracks": len(tracks),
        "sdr": {t: {"mean": float(np.mean(results[t])), "median": float(np.median(results[t]))}
                for t in targets if results[t]},
        "rtf_mean": float(np.mean(rtfs)),
        "timings": timings,
    }
    out_path = Path("benchmarks") / "separation_results.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
