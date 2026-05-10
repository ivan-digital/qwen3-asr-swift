#!/usr/bin/env python3
"""Benchmark Qwen3-ASR batched greedy decode via the repo CLI.

This is intentionally a small harness around:

    .build/release/audio transcribe-batch <chunks> --engine qwen3 --jsonl

It creates fixed-duration WAV chunks from a local source file, then sweeps
batch sizes so batch-size 1/2/4/8 runs see the same audio in the same order.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import hashlib


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUDIO = REPO_ROOT / "Tests/Qwen3ASRTests/Resources/test_audio.wav"
DEFAULT_WORK_DIR = REPO_ROOT / ".build/qwen3-batch-decode-benchmark"
DEFAULT_CLI = REPO_ROOT / ".build/release/audio"


@dataclass
class RunResult:
    batch_size: int
    files: int
    total_audio: float
    total_inference: float
    aggregate_rtf: float
    process_wall: float
    cli_wall: float | None
    model_load: float | None
    warmup: float | None
    output_chars: int
    unique_outputs: int
    output_digest: str

    @property
    def audio_per_inference_second(self) -> float:
        if self.total_inference <= 0:
            return 0.0
        return self.total_audio / self.total_inference

    @property
    def chars_per_second(self) -> float:
        if self.total_inference <= 0:
            return 0.0
        return self.output_chars / self.total_inference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep Qwen3-ASR transcribe-batch batch sizes on fixed WAV chunks."
    )
    parser.add_argument(
        "--audio",
        type=Path,
        default=DEFAULT_AUDIO,
        help=f"Source WAV used to make fixed chunks (default: {DEFAULT_AUDIO})",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=DEFAULT_WORK_DIR,
        help=f"Temporary benchmark directory (default: {DEFAULT_WORK_DIR})",
    )
    parser.add_argument(
        "--cli",
        type=Path,
        default=DEFAULT_CLI,
        help=f"audio CLI path (default: {DEFAULT_CLI})",
    )
    parser.add_argument(
        "--model",
        default="0.6B",
        help="Qwen3 model argument passed to audio transcribe-batch (default: 0.6B)",
    )
    parser.add_argument(
        "--language",
        help="Optional language hint passed to audio transcribe-batch.",
    )
    parser.add_argument(
        "--batch-sizes",
        default="1,2,4,8",
        help="Comma-separated batch sizes to run (default: 1,2,4,8)",
    )
    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=10.0,
        help="Fixed chunk duration in seconds (default: 10.0)",
    )
    parser.add_argument(
        "--stride-seconds",
        type=float,
        default=0.0,
        help=(
            "Offset between generated chunks. 0 repeats the first chunk; "
            "set to --chunk-seconds for sequential slicing (default: 0.0)."
        ),
    )
    parser.add_argument(
        "--num-chunks",
        type=int,
        default=8,
        help="Number of fixed chunks to generate (default: 8)",
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help="Run `make build` before benchmarking.",
    )
    parser.add_argument(
        "--keep-work-dir",
        action="store_true",
        help="Keep generated chunks and raw command outputs.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print raw CLI output for each run.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Optional path to write structured run results as JSON.",
    )
    parser.add_argument(
        "--experimental-batched-decode",
        action="store_true",
        help=(
            "Set QWEN3_ASR_EXPERIMENTAL_BATCH_DECODE=1 for the CLI subprocess. "
            "Use only for decoder experiments; default runs the correctness-safe path."
        ),
    )
    parser.add_argument(
        "--require-identical-output",
        action="store_true",
        help=(
            "Fail if a run produces more than one distinct transcript. Useful with "
            "--stride-seconds 0, where every generated chunk has identical audio."
        ),
    )
    return parser.parse_args()


def parse_batch_sizes(raw: str) -> list[int]:
    sizes = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        size = int(part)
        if size <= 0:
            raise ValueError("batch sizes must be positive")
        sizes.append(size)
    if not sizes:
        raise ValueError("at least one batch size is required")
    return sizes


def run_checked(
    cmd: list[str],
    *,
    cwd: Path,
    verbose: bool = False,
    extra_env: dict[str, str] | None = None,
) -> str:
    if verbose:
        print("+ " + " ".join(cmd), flush=True)
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if verbose or proc.returncode != 0:
        print(proc.stdout)
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd, proc.stdout)
    return proc.stdout


def prepare_chunks(
    audio: Path,
    chunk_dir: Path,
    chunk_seconds: float,
    stride_seconds: float,
    num_chunks: int,
) -> None:
    if chunk_seconds <= 0:
        raise ValueError("--chunk-seconds must be positive")
    if stride_seconds < 0:
        raise ValueError("--stride-seconds must be non-negative")
    if num_chunks <= 0:
        raise ValueError("--num-chunks must be positive")
    if not audio.exists():
        raise FileNotFoundError(audio)

    if chunk_dir.exists():
        shutil.rmtree(chunk_dir)
    chunk_dir.mkdir(parents=True, exist_ok=True)

    with wave.open(str(audio), "rb") as src:
        params = src.getparams()
        frame_rate = src.getframerate()
        total_frames = src.getnframes()
        frames = src.readframes(total_frames)

    if total_frames <= 0:
        raise ValueError(f"empty WAV: {audio}")

    frame_size = params.nchannels * params.sampwidth
    chunk_frames = max(1, int(round(frame_rate * chunk_seconds)))
    stride_frames = int(round(frame_rate * stride_seconds))

    for idx in range(num_chunks):
        start_frame = (idx * stride_frames) % total_frames
        chunk = take_wrapped_frames(frames, frame_size, start_frame, chunk_frames, total_frames)
        out_path = chunk_dir / f"chunk_{idx:03d}.wav"
        with wave.open(str(out_path), "wb") as dst:
            dst.setparams(params)
            dst.writeframes(chunk)


def take_wrapped_frames(
    frames: bytes,
    frame_size: int,
    start_frame: int,
    needed_frames: int,
    total_frames: int,
) -> bytes:
    remaining = needed_frames
    pos = start_frame
    out = bytearray()
    while remaining > 0:
        take = min(remaining, total_frames - pos)
        start_byte = pos * frame_size
        end_byte = start_byte + take * frame_size
        out.extend(frames[start_byte:end_byte])
        remaining -= take
        pos = (pos + take) % total_frames
    return bytes(out)


def benchmark_one(
    cli: Path,
    chunk_dir: Path,
    batch_size: int,
    model: str,
    language: str | None,
    experimental_batched_decode: bool,
    raw_output_path: Path,
    verbose: bool,
    require_identical_output: bool,
) -> RunResult:
    cmd = [
        str(cli),
        "transcribe-batch",
        str(chunk_dir),
        "--engine",
        "qwen3",
        "--model",
        model,
        "--batch-size",
        str(batch_size),
        "--jsonl",
    ]
    if language:
        cmd.extend(["--language", language])

    t0 = time.perf_counter()
    extra_env = {}
    if experimental_batched_decode:
        extra_env["QWEN3_ASR_EXPERIMENTAL_BATCH_DECODE"] = "1"
    output = run_checked(cmd, cwd=REPO_ROOT, verbose=verbose, extra_env=extra_env)
    process_wall = time.perf_counter() - t0
    raw_output_path.write_text(output, encoding="utf-8")

    json_rows = []
    for line in output.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            json_rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass

    files = len(json_rows)
    total_audio = sum(float(row.get("duration", 0.0)) for row in json_rows)
    outputs = [str(row.get("text", "")) for row in json_rows]
    output_chars = sum(len(text) for text in outputs)
    unique_outputs = len(set(outputs))
    output_digest = hashlib.sha256("\n".join(outputs).encode("utf-8")).hexdigest()[:12]

    if require_identical_output and unique_outputs > 1:
        examples = []
        seen = set()
        for text in outputs:
            if text in seen:
                continue
            seen.add(text)
            examples.append(text)
            if len(examples) == 3:
                break
        raise ValueError(
            f"batch size {batch_size} produced {unique_outputs} distinct transcripts "
            f"for identical chunks; examples={examples}"
        )

    total_inference = parse_float(output, r"Total inference:\s*([0-9.]+)s")
    aggregate_rtf = parse_float(output, r"Aggregate RTF:\s*([0-9.]+)")
    cli_wall = parse_float(output, r"Wall time:\s*([0-9.]+)s")
    model_load = parse_float(output, r"Model load:\s*([0-9.]+)s")
    warmup = parse_float(output, r"Warmup:\s*([0-9.]+)s")

    if total_inference is None:
        # Each JSON row's "time" is allocated per item, so the sum equals
        # total group inference time across all batches.
        total_inference = sum(float(row.get("time", 0.0)) for row in json_rows)
    if aggregate_rtf is None:
        aggregate_rtf = total_inference / max(total_audio, 0.001)

    return RunResult(
        batch_size=batch_size,
        files=files,
        total_audio=total_audio,
        total_inference=total_inference,
        aggregate_rtf=aggregate_rtf,
        process_wall=process_wall,
        cli_wall=cli_wall,
        model_load=model_load,
        warmup=warmup,
        output_chars=output_chars,
        unique_outputs=unique_outputs,
        output_digest=output_digest,
    )


def parse_float(text: str, pattern: str) -> float | None:
    match = re.search(pattern, text)
    if not match:
        return None
    return float(match.group(1))


def print_table(results: Iterable[RunResult], *, experimental_batched_decode: bool) -> None:
    rows = list(results)
    if not rows:
        return

    baseline = next((row for row in rows if row.batch_size == 1), rows[0])
    print()
    print("Qwen3-ASR batched decode benchmark")
    mode = "experimental batched decoder" if experimental_batched_decode else "correctness-safe public path"
    print(f"mode: {mode}")
    print(
        "batch  files  audio_s  infer_s  agg_rtf  x_realtime  speedup  "
        "chars/s  uniq_out  digest        cli_wall  load_s  warmup_s"
    )
    for row in rows:
        speedup = baseline.total_inference / row.total_inference if row.total_inference else 0.0
        print(
            f"{row.batch_size:>5}  "
            f"{row.files:>5}  "
            f"{row.total_audio:>7.1f}  "
            f"{row.total_inference:>7.3f}  "
            f"{row.aggregate_rtf:>7.4f}  "
            f"{row.audio_per_inference_second:>10.2f}  "
            f"{speedup:>7.3f}  "
            f"{row.chars_per_second:>7.1f}  "
            f"{row.unique_outputs:>8}  "
            f"{row.output_digest:<12}  "
            f"{fmt_optional(row.cli_wall):>8}  "
            f"{fmt_optional(row.model_load):>6}  "
            f"{fmt_optional(row.warmup):>8}"
        )
    print()
    print("Note: chars/s is reported because the current CLI does not expose generated token counts.")


def fmt_optional(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.2f}"


def write_json(path: Path, results: list[RunResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "batch_size": row.batch_size,
            "files": row.files,
            "total_audio": row.total_audio,
            "total_inference": row.total_inference,
            "aggregate_rtf": row.aggregate_rtf,
            "x_realtime": row.audio_per_inference_second,
            "process_wall": row.process_wall,
            "cli_wall": row.cli_wall,
            "model_load": row.model_load,
            "warmup": row.warmup,
            "output_chars": row.output_chars,
            "unique_outputs": row.unique_outputs,
            "output_digest": row.output_digest,
            "chars_per_second": row.chars_per_second,
        }
        for row in results
    ]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    batch_sizes = parse_batch_sizes(args.batch_sizes)

    if args.build:
        run_checked(["make", "build"], cwd=REPO_ROOT, verbose=True)
    if not args.cli.exists():
        raise FileNotFoundError(f"{args.cli} does not exist; run `make build` first")

    chunk_dir = args.work_dir / "chunks"
    raw_dir = args.work_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    prepare_chunks(args.audio, chunk_dir, args.chunk_seconds, args.stride_seconds, args.num_chunks)

    results: list[RunResult] = []
    try:
        for batch_size in batch_sizes:
            raw_output_path = raw_dir / f"batch_{batch_size}.log"
            print(f"Running batch size {batch_size}...", flush=True)
            result = benchmark_one(
                cli=args.cli,
                chunk_dir=chunk_dir,
                batch_size=batch_size,
                model=args.model,
                language=args.language,
                experimental_batched_decode=args.experimental_batched_decode,
                raw_output_path=raw_output_path,
                verbose=args.verbose,
                require_identical_output=args.require_identical_output,
            )
            results.append(result)
    finally:
        if not args.keep_work_dir:
            shutil.rmtree(args.work_dir, ignore_errors=True)

    print_table(results, experimental_batched_decode=args.experimental_batched_decode)
    if args.json_output:
        write_json(args.json_output, results)
        print(f"Wrote JSON results to {args.json_output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError, subprocess.CalledProcessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
