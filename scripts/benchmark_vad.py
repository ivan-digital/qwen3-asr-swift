#!/usr/bin/env python3
"""
VAD benchmark for speech-swift.

Evaluates Pyannote and Silero VAD against VoxConverse reference RTTM files.
Computes frame-level F1, False Alarm Rate, Miss Rate.

Also compares against published FireRedVAD numbers (FLEURS-VAD-102).

Usage:
    python scripts/benchmark_vad.py [--engine mlx] [--num-files 5]
    python scripts/benchmark_vad.py --compare
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

BENCHMARK_DIR = Path("benchmarks/voxconverse")
AUDIO_DIR = BENCHMARK_DIR / "audio"
REF_DIR = BENCHMARK_DIR / "ref"
RESULTS_DIR = Path("benchmarks/vad")

# FireRedVAD paper numbers (FLEURS-VAD-102, Table 3)
PAPER_NUMBERS = {
    "FireRedVAD": {"f1": 97.57, "far": 2.69, "mr": 3.62, "auc_roc": 99.60},
    "Silero-VAD":  {"f1": 95.95, "far": 9.41, "mr": 3.95, "auc_roc": 97.99},
    "TEN-VAD":     {"f1": 95.19, "far": 15.47, "mr": 2.95, "auc_roc": 97.81},
    "FunASR-VAD":  {"f1": 90.91, "far": 44.03, "mr": 0.42},
    "WebRTC-VAD":  {"f1": 52.30, "far": 2.83, "mr": 64.15},
}


# ---------------------------------------------------------------------------
# RTTM parsing & frame-level evaluation
# ---------------------------------------------------------------------------

def parse_rttm(content: str) -> list:
    """Parse RTTM to list of (start, end) segments."""
    segments = []
    for line in content.strip().split("\n"):
        parts = line.split()
        if len(parts) < 5 or parts[0] != "SPEAKER":
            continue
        start = float(parts[3])
        duration = float(parts[4])
        segments.append((start, start + duration))
    return segments


def segments_to_frames(segments: list, duration: float,
                       frame_shift: float = 0.01) -> list:
    """Convert segments to binary frame-level labels."""
    n_frames = int(duration / frame_shift) + 1
    frames = [0] * n_frames
    for start, end in segments:
        i_start = int(start / frame_shift)
        i_end = int(end / frame_shift)
        for i in range(max(0, i_start), min(n_frames, i_end)):
            frames[i] = 1
    return frames


def compute_vad_metrics(ref_segments: list, hyp_segments: list,
                        duration: float) -> dict:
    """Compute frame-level VAD metrics (F1, FAR, MR)."""
    frame_shift = 0.01  # 10ms frames
    ref_frames = segments_to_frames(ref_segments, duration, frame_shift)
    hyp_frames = segments_to_frames(hyp_segments, duration, frame_shift)

    n = min(len(ref_frames), len(hyp_frames))
    tp = sum(1 for i in range(n) if ref_frames[i] == 1 and hyp_frames[i] == 1)
    fp = sum(1 for i in range(n) if ref_frames[i] == 0 and hyp_frames[i] == 1)
    fn = sum(1 for i in range(n) if ref_frames[i] == 1 and hyp_frames[i] == 0)
    tn = sum(1 for i in range(n) if ref_frames[i] == 0 and hyp_frames[i] == 0)

    total_speech = tp + fn
    total_nonspeech = fp + tn

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    far = fp / max(total_nonspeech, 1) * 100  # false alarm rate
    mr = fn / max(total_speech, 1) * 100  # miss rate

    return {
        "f1": round(f1 * 100, 2),
        "far": round(far, 2),
        "mr": round(mr, 2),
        "precision": round(precision * 100, 2),
        "recall": round(recall * 100, 2),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "total_speech_frames": total_speech,
        "total_nonspeech_frames": total_nonspeech,
    }


# ---------------------------------------------------------------------------
# Run VAD via CLI
# ---------------------------------------------------------------------------

def run_vad(cli_path: str, audio_path: str, engine: str = "pyannote",
            timeout: int = 300) -> list:
    """Run VAD on an audio file, return list of (start, end) segments."""
    if engine == "silero":
        # Silero uses vad-stream command
        cmd = [cli_path, "vad-stream", str(audio_path)]
    else:
        # Pyannote uses vad command
        cmd = [cli_path, "vad", str(audio_path)]

    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout)

    if result.returncode != 0:
        return []

    segments = []

    if engine == "silero":
        # Parse: [2.50s] Speech started / [6.02s] Speech ended (3.52s)
        current_start = None
        for line in result.stdout.split("\n"):
            m = re.search(r"\[([\d.]+)s\]\s*Speech started", line)
            if m:
                current_start = float(m.group(1))
            m = re.search(r"\[([\d.]+)s\]\s*Speech ended", line)
            if m and current_start is not None:
                segments.append((current_start, float(m.group(1))))
                current_start = None
    else:
        # Parse: [0.10s - 0.73s] (0.63s)
        for line in result.stdout.split("\n"):
            m = re.search(r"\[([\d.]+)s\s*[-–]\s*([\d.]+)s\]", line)
            if m:
                segments.append((float(m.group(1)), float(m.group(2))))

    return segments


def get_audio_duration(audio_path: str) -> float:
    """Get audio duration using ffprobe or soxi."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries",
             "format=duration", "-of", "csv=p=0", audio_path],
            capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            return float(result.stdout.strip())
    except Exception:
        pass

    # Fallback: estimate from file size (16kHz, 16-bit mono WAV)
    try:
        size = os.path.getsize(audio_path)
        return (size - 44) / (16000 * 2)  # rough estimate
    except Exception:
        return 600.0  # default 10 min


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(cli_path: str, engine: str, num_files: int = 0,
                  timeout: int = 300) -> list:
    """Run VAD benchmark on VoxConverse test files."""
    ref_files = sorted(REF_DIR.glob("*.rttm"))
    if num_files > 0:
        ref_files = ref_files[:num_files]

    if not ref_files:
        print("No reference RTTM files found. Run benchmark_diarization.py "
              "--download-only first.")
        return []

    results = []
    total = len(ref_files)

    for idx, ref_path in enumerate(ref_files):
        name = ref_path.stem
        audio_path = AUDIO_DIR / f"{name}.wav"
        if not audio_path.exists():
            print(f"  Skipping {name} (no audio)")
            continue

        print(f"  [{idx+1}/{total}] {name}...", end=" ", flush=True)

        # Get reference segments (merge all speakers → speech/non-speech)
        ref_content = ref_path.read_text()
        ref_segments = parse_rttm(ref_content)
        duration = get_audio_duration(str(audio_path))

        # Run VAD
        start = time.monotonic()
        try:
            hyp_segments = run_vad(cli_path, str(audio_path), engine, timeout)
        except subprocess.TimeoutExpired:
            print("TIMEOUT")
            continue
        elapsed = time.monotonic() - start

        if not hyp_segments:
            print("FAILED")
            results.append({"file": name, "error": "no_segments"})
            continue

        # Score
        metrics = compute_vad_metrics(ref_segments, hyp_segments, duration)
        metrics["file"] = name
        metrics["elapsed"] = round(elapsed, 2)
        metrics["rtf"] = round(elapsed / max(duration, 0.001), 4)
        metrics["num_ref_segments"] = len(ref_segments)
        metrics["num_hyp_segments"] = len(hyp_segments)
        results.append(metrics)

        print(f"F1={metrics['f1']:.1f}% FAR={metrics['far']:.1f}% "
              f"MR={metrics['mr']:.1f}% ({elapsed:.1f}s)")

    return results


def aggregate_vad_results(per_file: list, engine: str) -> dict:
    """Aggregate frame-level metrics across files."""
    scored = [r for r in per_file if "error" not in r]
    if not scored:
        return {}

    total_tp = sum(r["tp"] for r in scored)
    total_fp = sum(r["fp"] for r in scored)
    total_fn = sum(r["fn"] for r in scored)
    total_tn = sum(r["tn"] for r in scored)

    total_speech = total_tp + total_fn
    total_nonspeech = total_fp + total_tn

    precision = total_tp / max(total_tp + total_fp, 1)
    recall = total_tp / max(total_tp + total_fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    far = total_fp / max(total_nonspeech, 1) * 100
    mr = total_fn / max(total_speech, 1) * 100

    rtfs = [r["rtf"] for r in scored if r.get("rtf", 0) > 0]

    return {
        "engine": engine,
        "dataset": "voxconverse-test",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "num_files": len(scored),
        "aggregate_f1": round(f1 * 100, 2),
        "aggregate_far": round(far, 2),
        "aggregate_mr": round(mr, 2),
        "precision": round(precision * 100, 2),
        "recall": round(recall * 100, 2),
        "mean_rtf": round(sum(rtfs) / len(rtfs), 4) if rtfs else None,
        "per_file": per_file,
    }


def print_summary(results: dict):
    """Print VAD benchmark summary."""
    print(f"\n{'='*60}")
    print(f"VAD Benchmark: {results.get('dataset', 'unknown')}")
    print(f"Engine: {results['engine']}")
    print(f"{'='*60}")
    print(f"  Files:          {results['num_files']}")
    print(f"  Aggregate F1:   {results['aggregate_f1']:.2f}%")
    print(f"  False Alarm:    {results['aggregate_far']:.2f}%")
    print(f"  Miss Rate:      {results['aggregate_mr']:.2f}%")
    print(f"  Precision:      {results['precision']:.2f}%")
    print(f"  Recall:         {results['recall']:.2f}%")
    if results.get("mean_rtf"):
        print(f"  Mean RTF:       {results['mean_rtf']:.4f}")
    print(f"{'='*60}")


def print_comparison(our_results: list):
    """Print comparison table: our results vs published numbers."""
    print(f"\n{'='*70}")
    print(f"Comparison: Our VAD vs Published Numbers (FireRedVAD paper)")
    print(f"{'='*70}")
    print(f"{'Model':<20} {'F1%':>8} {'FAR%':>8} {'MR%':>8} {'Dataset':<20}")
    print(f"{'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*20}")

    # Our results
    for r in our_results:
        label = f"Ours ({r['engine']})"
        print(f"{label:<20} {r['aggregate_f1']:>7.2f}% "
              f"{r['aggregate_far']:>7.2f}% {r['aggregate_mr']:>7.2f}% "
              f"{'VoxConverse':<20}")

    print(f"{'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*20}")

    # Paper numbers
    for name, nums in PAPER_NUMBERS.items():
        f1 = f"{nums['f1']:.2f}%" if "f1" in nums else "N/A"
        far = f"{nums['far']:.2f}%" if "far" in nums else "N/A"
        mr = f"{nums['mr']:.2f}%" if "mr" in nums else "N/A"
        print(f"{name:<20} {f1:>8} {far:>8} {mr:>8} {'FLEURS-VAD-102':<20}")

    print(f"{'='*70}")
    print(f"Note: Different datasets — direct comparison is indicative only.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="VAD benchmark (VoxConverse + FireRedVAD comparison)")
    parser.add_argument("--cli-path", default=".build/release/audio")
    parser.add_argument("--engine", default="pyannote",
                        choices=["pyannote", "silero"],
                        help="VAD engine: pyannote (MLX) or silero (streaming)")
    parser.add_argument("--num-files", type=int, default=0,
                        help="Limit number of files (0 = all)")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--compare", action="store_true",
                        help="Run both engines and compare with paper numbers")
    args = parser.parse_args()

    if not AUDIO_DIR.exists() or not any(AUDIO_DIR.glob("*.wav")):
        print("VoxConverse audio not found. Download first:")
        print("  python scripts/benchmark_diarization.py --download-only")
        sys.exit(1)

    if not Path(args.cli_path).exists():
        print(f"CLI not found: {args.cli_path}. Build with: make build")
        sys.exit(1)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.compare:
        all_results = []
        for engine in ["pyannote", "silero"]:
            print(f"\n--- VAD engine: {engine} ---")
            per_file = run_benchmark(
                args.cli_path, engine, args.num_files, args.timeout)
            if per_file:
                agg = aggregate_vad_results(per_file, engine)
                all_results.append(agg)
                print_summary(agg)

                out_file = RESULTS_DIR / f"results_vad_{engine}.json"
                with open(out_file, "w") as f:
                    json.dump(agg, f, indent=2)

        if all_results:
            print_comparison(all_results)
        return

    # Single engine
    print(f"\nRunning VAD benchmark ({args.engine})...")
    per_file = run_benchmark(
        args.cli_path, args.engine, args.num_files, args.timeout)

    if not per_file:
        print("No results.")
        return

    results = aggregate_vad_results(per_file, args.engine)
    if not results:
        print("No results to aggregate.")
        return
    print_summary(results)
    print_comparison([results])

    out_file = RESULTS_DIR / f"results_vad_{args.engine}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    main()
