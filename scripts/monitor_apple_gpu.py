#!/usr/bin/env python3
"""Sample Apple Silicon GPU/CPU power while optionally running a command.

This wraps macOS `powermetrics` in a non-interactive way. It uses
`sudo -n`, so it never prompts for a password. If the current shell has no
sudo credential, it exits with the exact command to run in a prepared shell.
"""

from __future__ import annotations

import argparse
import plistlib
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SampleSummary:
    samples: int
    gpu_power_mw_avg: float | None
    gpu_power_mw_max: float | None
    cpu_power_mw_avg: float | None
    cpu_power_mw_max: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Monitor Apple GPU/CPU power via powermetrics."
    )
    parser.add_argument(
        "--interval-ms",
        type=int,
        default=1000,
        help="powermetrics sample interval in milliseconds (default: 1000)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=30,
        help="Number of samples when no command is supplied (default: 30)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON summary output path.",
    )
    parser.add_argument(
        "--raw-output",
        type=Path,
        help="Optional raw powermetrics plist output path.",
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Optional command to run while sampling. Prefix with -- before the command.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.interval_ms <= 0:
        raise ValueError("--interval-ms must be positive")
    if args.samples <= 0:
        raise ValueError("--samples must be positive")

    powermetrics = shutil.which("powermetrics")
    if not powermetrics:
        raise FileNotFoundError("powermetrics not found")

    command = normalize_command(args.command)
    with tempfile.NamedTemporaryFile(prefix="powermetrics-", suffix=".plist") as raw:
        raw_path = Path(raw.name)
        sample_count = args.samples if command is None else -1
        pm_cmd = [
            "sudo",
            "-n",
            powermetrics,
            "--samplers",
            "gpu_power,cpu_power",
            "--sample-rate",
            str(args.interval_ms),
            "--sample-count",
            str(sample_count),
            "--format",
            "plist",
            "--output-file",
            str(raw_path),
        ]

        started = subprocess.Popen(pm_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        time.sleep(0.25)
        if started.poll() is not None and started.returncode != 0:
            _, stderr = started.communicate()
            print(stderr.strip(), file=sys.stderr)
            print("powermetrics needs a sudo credential. Run this once in a terminal, then retry:", file=sys.stderr)
            print("  sudo -v", file=sys.stderr)
            print("Non-interactive sampler command:", file=sys.stderr)
            print("  " + " ".join(pm_cmd), file=sys.stderr)
            return started.returncode

        command_rc = 0
        try:
            if command is not None:
                print("+ " + " ".join(command), flush=True)
                command_rc = subprocess.call(command)
                started.terminate()
            stdout, stderr = started.communicate(timeout=max(5, args.interval_ms / 1000 + 2))
            if stdout:
                print(stdout, end="")
            if stderr:
                print(stderr, end="", file=sys.stderr)
        except subprocess.TimeoutExpired:
            started.kill()
            started.communicate()
            raise

        raw_bytes = raw_path.read_bytes()
        if args.raw_output:
            args.raw_output.parent.mkdir(parents=True, exist_ok=True)
            args.raw_output.write_bytes(raw_bytes)

        summary = summarize_plist_stream(raw_bytes)
        print_summary(summary)
        if args.output:
            import json

            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(
                    {
                        "samples": summary.samples,
                        "gpu_power_mw_avg": summary.gpu_power_mw_avg,
                        "gpu_power_mw_max": summary.gpu_power_mw_max,
                        "cpu_power_mw_avg": summary.cpu_power_mw_avg,
                        "cpu_power_mw_max": summary.cpu_power_mw_max,
                        "command_returncode": command_rc,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        return command_rc


def normalize_command(parts: list[str]) -> list[str] | None:
    if not parts:
        return None
    if parts[0] == "--":
        parts = parts[1:]
    return parts or None


def summarize_plist_stream(raw: bytes) -> SampleSummary:
    # powermetrics plist format is NUL-separated when multiple samples are emitted.
    samples = []
    for chunk in raw.split(b"\x00"):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            samples.append(plistlib.loads(chunk))
        except Exception:
            continue

    gpu = []
    cpu = []
    for sample in samples:
        gpu_value = first_numeric(sample, ("gpu_power", "GPU Power", "gpu_power_mw", "GPU Power mW"))
        cpu_value = first_numeric(sample, ("cpu_power", "CPU Power", "cpu_power_mw", "CPU Power mW"))
        if gpu_value is not None:
            gpu.append(gpu_value)
        if cpu_value is not None:
            cpu.append(cpu_value)

    return SampleSummary(
        samples=len(samples),
        gpu_power_mw_avg=mean(gpu),
        gpu_power_mw_max=max(gpu) if gpu else None,
        cpu_power_mw_avg=mean(cpu),
        cpu_power_mw_max=max(cpu) if cpu else None,
    )


def first_numeric(value: object, keys: tuple[str, ...]) -> float | None:
    if isinstance(value, dict):
        for key in keys:
            if key in value and isinstance(value[key], (int, float)):
                return float(value[key])
        for child in value.values():
            found = first_numeric(child, keys)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = first_numeric(child, keys)
            if found is not None:
                return found
    return None


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def print_summary(summary: SampleSummary) -> None:
    print()
    print("Apple GPU/CPU power summary")
    print(f"samples: {summary.samples}")
    print(f"gpu_power_mw_avg: {fmt(summary.gpu_power_mw_avg)}")
    print(f"gpu_power_mw_max: {fmt(summary.gpu_power_mw_max)}")
    print(f"cpu_power_mw_avg: {fmt(summary.cpu_power_mw_avg)}")
    print(f"cpu_power_mw_max: {fmt(summary.cpu_power_mw_max)}")


def fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.1f}"


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
