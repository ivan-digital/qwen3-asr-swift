#!/usr/bin/env python3
"""
ASR WER benchmark for speech-swift.

Datasets:
  - LibriSpeech test-clean (English, 2620 utterances, ~350 MB)
  - CommonVoice test splits (multilingual: en, zh, de, es, fr)

Usage:
    python scripts/benchmark_asr.py [--engine qwen3] [--model 0.6B]
    python scripts/benchmark_asr.py --dataset commonvoice --language zh
    python scripts/benchmark_asr.py --download-only
    python scripts/benchmark_asr.py --compare
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import tarfile
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Dataset config
# ---------------------------------------------------------------------------

LIBRISPEECH_URL = "https://www.openslr.org/resources/12/test-clean.tar.gz"

# CommonVoice 17.0 — direct download without auth via HuggingFace mirror
# Users can also download manually from https://commonvoice.mozilla.org
COMMONVOICE_LANGUAGES = {
    "en": "English",
    "zh-CN": "Chinese",
    "de": "German",
    "es": "Spanish",
    "fr": "French",
}

# FLEURS — freely downloadable, ~400 utterances per language, 102 languages
FLEURS_LANGUAGES = {
    "en_us": "English",
    "cmn_hans_cn": "Chinese",
    "de_de": "German",
    "es_419": "Spanish",
    "fr_fr": "French",
    "ja_jp": "Japanese",
    "ko_kr": "Korean",
    "ru_ru": "Russian",
    "ar_eg": "Arabic",
    "hi_in": "Hindi",
    "it_it": "Italian",
    "pt_br": "Portuguese",
}

BENCHMARK_BASE = Path("benchmarks")


# ---------------------------------------------------------------------------
# Text normalization & WER
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_cjk(text: str) -> bool:
    """Check if text is primarily CJK (Chinese/Japanese/Korean)."""
    cjk_count = sum(1 for c in text if '\u4e00' <= c <= '\u9fff'
                    or '\u3040' <= c <= '\u309f'  # hiragana
                    or '\u30a0' <= c <= '\u30ff'  # katakana
                    or '\uac00' <= c <= '\ud7af')  # hangul
    return cjk_count > len(text) * 0.3


def compute_wer(reference: str, hypothesis: str) -> dict:
    """Word Error Rate (or Character Error Rate for CJK) via edit distance."""
    ref_norm = normalize_text(reference)
    hyp_norm = normalize_text(hypothesis)

    # Use character-level for CJK languages (no word boundaries)
    if is_cjk(ref_norm):
        ref = list(ref_norm.replace(" ", ""))
        hyp = list(hyp_norm.replace(" ", ""))
    else:
        ref = ref_norm.split()
        hyp = hyp_norm.split()

    n = len(ref)
    m = len(hyp)

    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = 1 + min(d[i - 1][j], d[i][j - 1], d[i - 1][j - 1])

    subs, ins, dels = 0, 0, 0
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1]:
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + 1:
            subs += 1
            i -= 1
            j -= 1
        elif j > 0 and d[i][j] == d[i][j - 1] + 1:
            ins += 1
            j -= 1
        else:
            dels += 1
            i -= 1

    errors = subs + ins + dels
    wer = errors / max(n, 1) * 100

    return {
        "wer": round(wer, 2),
        "errors": errors,
        "substitutions": subs,
        "insertions": ins,
        "deletions": dels,
        "ref_words": n,
        "hyp_words": m,
    }


# ---------------------------------------------------------------------------
# LibriSpeech download & loading
# ---------------------------------------------------------------------------

def download_librispeech():
    """Download and extract LibriSpeech test-clean (~350 MB)."""
    bench_dir = BENCHMARK_BASE / "librispeech"
    data_dir = bench_dir / "test-clean"
    bench_dir.mkdir(parents=True, exist_ok=True)

    if data_dir.exists() and any(data_dir.rglob("*.flac")):
        print(f"LibriSpeech test-clean already at {data_dir}")
        return

    tar_path = bench_dir / "test-clean.tar.gz"
    if not tar_path.exists():
        print(f"Downloading LibriSpeech test-clean (~350 MB)...")
        subprocess.run(
            ["curl", "-L", "-o", str(tar_path), "--progress-bar",
             LIBRISPEECH_URL], check=True)

    print(f"Extracting...")
    with tarfile.open(tar_path, "r:gz") as tf:
        tf.extractall(bench_dir)

    extracted = bench_dir / "LibriSpeech" / "test-clean"
    if extracted.exists() and not data_dir.exists():
        extracted.rename(data_dir)
    ls_dir = bench_dir / "LibriSpeech"
    if ls_dir.exists() and not any(ls_dir.iterdir()):
        ls_dir.rmdir()

    print(f"  {len(list(data_dir.rglob('*.flac')))} FLAC files")


def load_librispeech(num_files: int = 0) -> list:
    """Load LibriSpeech transcripts. Returns [(id, path, text, lang)]."""
    data_dir = BENCHMARK_BASE / "librispeech" / "test-clean"
    utterances = []
    for trans_file in sorted(data_dir.rglob("*.trans.txt")):
        chapter_dir = trans_file.parent
        for line in trans_file.read_text().strip().split("\n"):
            parts = line.strip().split(" ", 1)
            if len(parts) != 2:
                continue
            utt_id, text = parts
            flac_path = chapter_dir / f"{utt_id}.flac"
            if flac_path.exists():
                utterances.append((utt_id, str(flac_path), text, "en"))
    utterances.sort(key=lambda x: x[0])
    return utterances[:num_files] if num_files > 0 else utterances


# ---------------------------------------------------------------------------
# CommonVoice download & loading
# ---------------------------------------------------------------------------

def download_commonvoice(language: str):
    """Download CommonVoice test split for a language.

    CommonVoice requires manual download from https://commonvoice.mozilla.org
    or HuggingFace (requires auth token). This function checks for existing
    data and provides download instructions if missing.
    """
    bench_dir = BENCHMARK_BASE / "commonvoice" / language
    bench_dir.mkdir(parents=True, exist_ok=True)

    clips_dir = bench_dir / "clips"
    tsv_path = bench_dir / "test.tsv"

    if clips_dir.exists() and tsv_path.exists():
        print(f"CommonVoice {language} already at {bench_dir}")
        return True

    print(f"\nCommonVoice {language} data not found at {bench_dir}")
    print(f"CommonVoice requires manual download:")
    print(f"  1. Go to https://commonvoice.mozilla.org/en/datasets")
    print(f"  2. Download Common Voice Corpus for '{COMMONVOICE_LANGUAGES.get(language, language)}'")
    print(f"  3. Extract test.tsv and clips/ to {bench_dir}/")
    print(f"     {bench_dir}/test.tsv")
    print(f"     {bench_dir}/clips/*.mp3")
    print(f"")
    print(f"  Or use HuggingFace datasets (requires auth):")
    print(f"    pip install datasets")
    print(f"    python -c \"from datasets import load_dataset; "
          f"ds = load_dataset('mozilla-foundation/common_voice_17_0', "
          f"'{language}', split='test'); ds.save_to_disk('{bench_dir}')\"")
    return False


def load_commonvoice(language: str, num_files: int = 0) -> list:
    """Load CommonVoice test transcripts. Returns [(id, path, text, lang)]."""
    bench_dir = BENCHMARK_BASE / "commonvoice" / language
    tsv_path = bench_dir / "test.tsv"
    clips_dir = bench_dir / "clips"

    if not tsv_path.exists():
        return []

    utterances = []
    with open(tsv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            audio_path = clips_dir / row["path"]
            if not audio_path.exists():
                # Try .mp3 extension
                mp3_path = clips_dir / (Path(row["path"]).stem + ".mp3")
                if mp3_path.exists():
                    audio_path = mp3_path
                else:
                    continue
            text = row.get("sentence", "")
            if text:
                utt_id = Path(row["path"]).stem
                utterances.append((utt_id, str(audio_path), text, language))

    utterances.sort(key=lambda x: x[0])
    return utterances[:num_files] if num_files > 0 else utterances


# ---------------------------------------------------------------------------
# FLEURS download & loading
# ---------------------------------------------------------------------------

def download_fleurs(language: str):
    """Download FLEURS test split from HuggingFace."""
    bench_dir = BENCHMARK_BASE / "fleurs" / language
    audio_dir = bench_dir / "audio"
    tsv_path = bench_dir / "test.tsv"

    if audio_dir.exists() and tsv_path.exists() and \
       any(audio_dir.glob("*.wav")):
        print(f"FLEURS {language} already at {bench_dir}")
        return True

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("Install: pip install huggingface_hub")
        return False

    lang_name = FLEURS_LANGUAGES.get(language, language)
    print(f"Downloading FLEURS {lang_name} ({language}) test split...")

    bench_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Download test.tsv
        tsv_file = hf_hub_download(
            "google/fleurs", f"data/{language}/test.tsv",
            repo_type="dataset")
        import shutil
        shutil.copy2(tsv_file, str(tsv_path))

        # Download audio tar.gz and extract
        tar_file = hf_hub_download(
            "google/fleurs", f"data/{language}/audio/test.tar.gz",
            repo_type="dataset")
        print(f"  Extracting audio...")
        with tarfile.open(tar_file, "r:gz") as tf:
            tf.extractall(audio_dir)
    except Exception as e:
        print(f"  Download failed: {e}")
        return False

    count = len(list(audio_dir.rglob("*.wav")))
    print(f"  {count} audio files")
    return count > 0


def load_fleurs(language: str, num_files: int = 0) -> list:
    """Load FLEURS test transcripts. Returns [(id, path, text, lang)]."""
    bench_dir = BENCHMARK_BASE / "fleurs" / language
    tsv_path = bench_dir / "test.tsv"
    audio_dir = bench_dir / "audio"

    if not tsv_path.exists():
        return []

    lang_hint_map = {
        "en_us": "english", "cmn_hans_cn": "chinese", "de_de": "german",
        "es_419": "spanish", "fr_fr": "french", "ja_jp": "japanese",
        "ko_kr": "korean", "ru_ru": "russian", "ar_eg": "arabic",
        "hi_in": "hindi", "it_it": "italian", "pt_br": "portuguese",
    }
    lang_hint = lang_hint_map.get(language)

    # FLEURS TSV: id \t filename.wav \t raw_text \t normalized_text \t letters \t num_samples \t gender
    utterances = []
    for line in tsv_path.read_text().strip().split("\n"):
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        filename = parts[1]
        text = parts[2]  # raw transcription (with punctuation)
        wav_path = audio_dir / filename
        if not wav_path.exists():
            # Try subdirectory (tar may extract with path prefix)
            matches = list(audio_dir.rglob(filename))
            if matches:
                wav_path = matches[0]
        if wav_path.exists() and text.strip():
            utt_id = Path(filename).stem
            utterances.append((utt_id, str(wav_path), text.strip(), lang_hint))

    utterances.sort(key=lambda x: x[0])
    return utterances[:num_files] if num_files > 0 else utterances


# ---------------------------------------------------------------------------
# Transcription with latency tracking
# ---------------------------------------------------------------------------

def transcribe_file(cli_path: str, audio_path: str, engine: str,
                    model: str, language: str = None,
                    timeout: int = 120) -> dict:
    """Run CLI transcription. Returns text + timing."""
    cmd = [cli_path, "transcribe", audio_path, "--engine", engine]
    if engine in ("qwen3", "qwen3-coreml", "qwen3-coreml-full"):
        cmd.extend(["--model", model])
    if language:
        cmd.extend(["--language", language])

    wall_start = time.monotonic()
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout)
    wall_time = time.monotonic() - wall_start

    if result.returncode != 0:
        return {"error": result.stderr.strip()[:200]}

    text = ""
    rtf = 0.0
    inference_time = 0.0
    warmup_time = 0.0

    for line in result.stdout.split("\n"):
        if line.startswith("Result: "):
            text = line[len("Result: "):]
        elif "Time:" in line and "RTF:" in line:
            m = re.search(r"Time:\s*([\d.]+)s.*RTF:\s*([\d.]+)", line)
            if m:
                inference_time = float(m.group(1))
                rtf = float(m.group(2))
            # Parse warmup if present
            w = re.search(r"warmup:\s*([\d.]+)s", line)
            if w:
                warmup_time = float(w.group(1))

    return {
        "text": text,
        "rtf": rtf,
        "inference_time": inference_time,
        "warmup_time": warmup_time,
        "wall_time": round(wall_time, 3),
        "model_load_time": round(wall_time - inference_time - warmup_time, 3),
    }


def run_warmup(cli_path: str, audio_path: str, engine: str,
               model: str) -> dict:
    """Run a single warmup inference and return timing."""
    print("  Warmup inference...", end=" ", flush=True)
    out = transcribe_file(cli_path, audio_path, engine, model, timeout=300)
    if "error" in out:
        print(f"FAILED: {out['error']}")
        return {}
    print(f"done (wall={out['wall_time']:.1f}s, "
          f"inference={out['inference_time']:.2f}s)")
    return {
        "warmup_wall_time": out["wall_time"],
        "warmup_inference_time": out["inference_time"],
    }


def run_transcriptions(cli_path: str, utterances: list, engine: str,
                       model: str, timeout: int = 120,
                       measure_warmup: bool = True) -> tuple:
    """Transcribe all utterances. Returns (per_file_results, latency_info)."""
    hyp_dir = BENCHMARK_BASE / "librispeech" / "hyp"
    hyp_dir.mkdir(parents=True, exist_ok=True)
    hyp_subdir = hyp_dir / f"{engine}_{model}"
    hyp_subdir.mkdir(parents=True, exist_ok=True)

    # Warmup: first inference includes model load + shader compilation
    latency = {}
    if measure_warmup and utterances:
        latency = run_warmup(cli_path, utterances[0][1], engine, model)

    results = []
    total = len(utterances)
    failures = 0

    for idx, (utt_id, audio_path, ref_text, lang) in enumerate(utterances):
        pct = (idx + 1) / total * 100
        print(f"\r  [{idx+1}/{total}] ({pct:.0f}%) {utt_id}...",
              end="", flush=True)

        try:
            out = transcribe_file(
                cli_path, audio_path, engine, model, lang, timeout)
        except subprocess.TimeoutExpired:
            out = {"error": "timeout"}
        except Exception as e:
            out = {"error": str(e)}

        if "error" in out:
            failures += 1
            results.append({"utterance_id": utt_id, "error": out["error"]})
            continue

        (hyp_subdir / f"{utt_id}.txt").write_text(out["text"])
        wer_result = compute_wer(ref_text, out["text"])

        results.append({
            "utterance_id": utt_id,
            "reference": normalize_text(ref_text),
            "hypothesis": normalize_text(out["text"]),
            "language": lang,
            "wer": wer_result["wer"],
            "substitutions": wer_result["substitutions"],
            "insertions": wer_result["insertions"],
            "deletions": wer_result["deletions"],
            "ref_words": wer_result["ref_words"],
            "rtf": out["rtf"],
            "inference_time": out["inference_time"],
            "wall_time": out["wall_time"],
        })

    print()
    if failures:
        print(f"  {failures} utterances failed")

    return results, latency


def run_batch_transcription(cli_path: str, utterances: list, engine: str,
                            model: str, timeout: int = 600) -> tuple:
    """Transcribe using transcribe-batch CLI (model loaded once).

    Falls back to per-file mode if transcribe-batch is not available.
    """
    # Build ref map
    ref_map = {u[0]: (u[2], u[3]) for u in utterances}

    # Collect unique directories containing audio files
    audio_dirs = set()
    file_to_utt = {}  # filename_stem -> utt_id
    for utt_id, audio_path, ref_text, lang in utterances:
        audio_dirs.add(str(Path(audio_path).parent))
        stem = Path(audio_path).stem
        file_to_utt[stem] = utt_id

    # Use transcribe-batch with JSONL output
    # For LibriSpeech, files are spread across many dirs — use the root
    # Find common parent
    all_paths = [Path(u[1]) for u in utterances]
    common = all_paths[0].parent
    for p in all_paths[1:]:
        while not str(p).startswith(str(common)):
            common = common.parent

    hyp_dir = BENCHMARK_BASE / "librispeech" / "hyp" / f"{engine}_{model}"
    hyp_dir.mkdir(parents=True, exist_ok=True)

    cmd = [cli_path, "transcribe-batch", str(common),
           "--engine", engine, "--jsonl",
           "--output-dir", str(hyp_dir)]
    if engine in ("qwen3", "qwen3-coreml", "qwen3-coreml-full"):
        cmd.extend(["--model", model])

    print(f"  Running batch transcription on {common}...")
    print(f"  Command: {' '.join(cmd[:6])}...")

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=timeout * len(utterances))
    except subprocess.TimeoutExpired:
        print("  Batch transcription timed out, falling back to per-file mode")
        return run_transcriptions(cli_path, utterances, engine, model,
                                  timeout, measure_warmup=True)

    if result.returncode != 0:
        print(f"  transcribe-batch failed: {result.stderr[:200]}")
        print("  Falling back to per-file mode")
        return run_transcriptions(cli_path, utterances, engine, model,
                                  timeout, measure_warmup=True)

    # Parse JSONL output
    results = []
    latency = {}
    batch_results = {}

    for line in result.stdout.strip().split("\n"):
        line = line.strip()
        if not line or not line.startswith("{"):
            # Parse summary lines for latency
            m = re.search(r"Model load:\s*([\d.]+)s.*Warmup:\s*([\d.]+)s", line)
            if m:
                latency["model_load_time"] = float(m.group(1))
                latency["warmup_time"] = float(m.group(2))
            m = re.search(r"Aggregate RTF:\s*([\d.]+)", line)
            if m:
                latency["aggregate_rtf"] = float(m.group(1))
            continue
        try:
            entry = json.loads(line)
            batch_results[entry["file"]] = entry
        except (json.JSONDecodeError, KeyError):
            continue

    # Match with reference transcripts
    for utt_id, audio_path, ref_text, lang in utterances:
        stem = Path(audio_path).stem
        if stem not in batch_results:
            results.append({"utterance_id": utt_id, "error": "not_in_batch"})
            continue

        entry = batch_results[stem]
        if "error" in entry:
            results.append({"utterance_id": utt_id, "error": entry["error"]})
            continue

        wer_result = compute_wer(ref_text, entry["text"])
        results.append({
            "utterance_id": utt_id,
            "reference": normalize_text(ref_text),
            "hypothesis": normalize_text(entry["text"]),
            "language": lang,
            "wer": wer_result["wer"],
            "substitutions": wer_result["substitutions"],
            "insertions": wer_result["insertions"],
            "deletions": wer_result["deletions"],
            "ref_words": wer_result["ref_words"],
            "rtf": entry.get("rtf", 0),
            "inference_time": entry.get("time", 0),
        })

    scored = len([r for r in results if "error" not in r])
    failed = len([r for r in results if "error" in r])
    print(f"  Batch: {scored} transcribed, {failed} failed")

    return results, latency


# ---------------------------------------------------------------------------
# Aggregation & reporting
# ---------------------------------------------------------------------------

def aggregate_results(per_file: list, engine: str, model: str,
                      dataset: str, latency: dict = None) -> dict:
    """Compute aggregate WER + latency from per-file results."""
    scored = [r for r in per_file if "error" not in r]
    failed = [r for r in per_file if "error" in r]

    total_ref = sum(r["ref_words"] for r in scored)
    total_sub = sum(r.get("substitutions", 0) for r in scored)
    total_ins = sum(r.get("insertions", 0) for r in scored)
    total_del = sum(r.get("deletions", 0) for r in scored)
    total_errors = total_sub + total_ins + total_del
    total_infer = sum(r.get("inference_time", 0) for r in scored)
    total_wall = sum(r.get("wall_time", 0) for r in scored)

    agg_wer = total_errors / max(total_ref, 1) * 100

    result = {
        "engine": engine,
        "model": model,
        "dataset": dataset,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "num_utterances": len(scored),
        "num_failures": len(failed),
        "aggregate_wer": round(agg_wer, 2),
        "total_ref_words": total_ref,
        "total_substitutions": total_sub,
        "total_insertions": total_ins,
        "total_deletions": total_del,
        "latency": {
            "total_inference_s": round(total_infer, 2),
            "total_wall_s": round(total_wall, 2),
            "mean_model_load_overhead_s": round(
                (total_wall - total_infer) / max(len(scored), 1), 3),
        },
        "per_file": per_file,
    }

    rtfs = [r["rtf"] for r in scored if r.get("rtf", 0) > 0]
    if rtfs:
        result["latency"]["mean_rtf"] = round(sum(rtfs) / len(rtfs), 4)

    if latency:
        result["latency"]["warmup"] = latency

    return result


def print_summary(results: dict):
    """Print summary table."""
    lat = results.get("latency", {})
    print(f"\n{'='*60}")
    print(f"ASR Benchmark: {results['dataset']}")
    print(f"Engine: {results['engine']}, Model: {results['model']}")
    print(f"{'='*60}")
    print(f"  Utterances:     {results['num_utterances']}"
          f" ({results['num_failures']} failed)")
    print(f"  Aggregate WER:  {results['aggregate_wer']:.2f}%")
    print(f"  Total words:    {results['total_ref_words']}")
    print(f"  Substitutions:  {results['total_substitutions']}")
    print(f"  Insertions:     {results['total_insertions']}")
    print(f"  Deletions:      {results['total_deletions']}")
    if lat.get("mean_rtf"):
        print(f"  Mean RTF:       {lat['mean_rtf']:.4f}")
    warmup = lat.get("warmup", {})
    if warmup:
        print(f"  Warmup wall:    {warmup.get('warmup_wall_time', 0):.1f}s")
        print(f"  Warmup infer:   {warmup.get('warmup_inference_time', 0):.2f}s")
    if lat.get("mean_model_load_overhead_s"):
        print(f"  Load overhead:  {lat['mean_model_load_overhead_s']:.3f}s/file")
    print(f"{'='*60}")

    scored = [r for r in results["per_file"]
              if "error" not in r and r["wer"] > 0]
    if scored:
        worst = sorted(scored, key=lambda x: x["wer"], reverse=True)[:10]
        print(f"\nWorst 10 utterances:")
        for r in worst:
            print(f"  {r['utterance_id']}: WER={r['wer']:.1f}%")
            print(f"    ref: {r['reference']}")
            print(f"    hyp: {r['hypothesis']}")


# ---------------------------------------------------------------------------
# Multi-engine comparison
# ---------------------------------------------------------------------------

ENGINE_CONFIGS = [
    ("qwen3", "0.6B"),
    ("qwen3", "0.6B-8bit"),
    ("qwen3", "1.7B"),
    ("qwen3", "1.7B-4bit"),
    ("parakeet", "default"),
]


def run_comparison(cli_path: str, utterances: list, dataset: str,
                   timeout: int = 120):
    """Run all engine/model combinations."""
    all_results = []

    for engine, model in ENGINE_CONFIGS:
        print(f"\n--- {engine} / {model} ---")
        per_file, latency = run_transcriptions(
            cli_path, utterances, engine, model, timeout)
        agg = aggregate_results(per_file, engine, model, dataset, latency)
        all_results.append(agg)
        print_summary(agg)

        out_file = BENCHMARK_BASE / "librispeech" / f"results_{engine}_{model}.json"
        with open(out_file, "w") as f:
            json.dump(agg, f, indent=2)

    lat_key = lambda r: r.get("latency", {}).get("mean_rtf", 0)
    print(f"\n{'='*70}")
    print(f"{'Engine':<16} {'Model':<12} {'WER%':>8} {'RTF':>8} {'Warmup':>8}")
    print(f"{'-'*16} {'-'*12} {'-'*8} {'-'*8} {'-'*8}")
    for r in all_results:
        lat = r.get("latency", {})
        rtf = f"{lat['mean_rtf']:.4f}" if lat.get("mean_rtf") else "N/A"
        wu = lat.get("warmup", {})
        warmup = f"{wu['warmup_wall_time']:.1f}s" if wu.get("warmup_wall_time") else "N/A"
        print(f"{r['engine']:<16} {r['model']:<12} "
              f"{r['aggregate_wer']:>7.2f}% {rtf:>8} {warmup:>8}")
    print(f"{'='*70}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="ASR WER benchmark (LibriSpeech + CommonVoice)")
    parser.add_argument("--cli-path", default=".build/release/audio",
                        help="Path to audio CLI binary")
    parser.add_argument("--dataset", default="librispeech",
                        choices=["librispeech", "commonvoice", "fleurs"],
                        help="Benchmark dataset")
    parser.add_argument("--language", default="en",
                        help="Language code (CommonVoice: en, zh-CN, de, es, fr; "
                             "FLEURS: en_us, zh_cn, de_de, es_419, fr_fr, ja_jp, ...)")
    parser.add_argument("--engine", default="qwen3",
                        help="ASR engine: qwen3, parakeet, qwen3-coreml")
    parser.add_argument("--model", default="0.6B",
                        help="Model variant: 0.6B, 0.6B-8bit, 1.7B, 1.7B-4bit")
    parser.add_argument("--num-files", type=int, default=0,
                        help="Limit number of utterances (0 = all)")
    parser.add_argument("--timeout", type=int, default=120,
                        help="Per-file timeout in seconds")
    parser.add_argument("--no-warmup", action="store_true",
                        help="Skip warmup measurement")
    parser.add_argument("--download-only", action="store_true",
                        help="Only download test data")
    parser.add_argument("--score-only", action="store_true",
                        help="Re-score existing hypothesis transcriptions")
    parser.add_argument("--batch", action="store_true",
                        help="Use transcribe-batch CLI (loads model once, much faster)")
    parser.add_argument("--compare", action="store_true",
                        help="Run all engine/model combinations")
    args = parser.parse_args()

    # Dataset selection
    if args.dataset == "fleurs":
        if not args.score_only:
            available = download_fleurs(args.language)
            if args.download_only:
                return
            if not available:
                sys.exit(1)
        utterances = load_fleurs(args.language, args.num_files)
        dataset_name = f"fleurs-{args.language}"
    elif args.dataset == "commonvoice":
        if not args.score_only:
            available = download_commonvoice(args.language)
            if args.download_only:
                return
            if not available:
                sys.exit(1)
        utterances = load_commonvoice(args.language, args.num_files)
        dataset_name = f"commonvoice-{args.language}"
    else:
        if not args.score_only:
            download_librispeech()
        if args.download_only:
            print("\nDownload complete.")
            return
        utterances = load_librispeech(args.num_files)
        dataset_name = "librispeech-test-clean"

    if not utterances:
        print("No utterances found. Check dataset or run --download-only.")
        sys.exit(1)
    print(f"Loaded {len(utterances)} utterances ({dataset_name})")

    if not Path(args.cli_path).exists():
        print(f"CLI not found: {args.cli_path}. Build with: make build")
        sys.exit(1)

    if args.compare:
        run_comparison(args.cli_path, utterances, dataset_name, args.timeout)
        return

    if args.score_only:
        # Re-score from saved hypotheses
        hyp_dir = BENCHMARK_BASE / "librispeech" / "hyp" / f"{args.engine}_{args.model}"
        if not hyp_dir.exists():
            print(f"No hypotheses at {hyp_dir}")
            sys.exit(1)
        ref_map = {u[0]: u[2] for u in utterances}
        per_file = []
        for hf in sorted(hyp_dir.glob("*.txt")):
            uid = hf.stem
            if uid not in ref_map:
                continue
            wer_result = compute_wer(ref_map[uid], hf.read_text().strip())
            per_file.append({
                "utterance_id": uid,
                "reference": normalize_text(ref_map[uid]),
                "hypothesis": normalize_text(hf.read_text().strip()),
                **wer_result,
            })
        latency = {}
    elif args.batch:
        print(f"\nBatch transcribing with {args.engine}/{args.model}...")
        per_file, latency = run_batch_transcription(
            args.cli_path, utterances, args.engine, args.model, args.timeout)
    else:
        print(f"\nTranscribing with {args.engine}/{args.model}...")
        per_file, latency = run_transcriptions(
            args.cli_path, utterances, args.engine, args.model,
            args.timeout, measure_warmup=not args.no_warmup)

    if not per_file:
        print("No results.")
        return

    results = aggregate_results(
        per_file, args.engine, args.model, dataset_name, latency)
    print_summary(results)

    out_dir = BENCHMARK_BASE / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.language}" if args.dataset == "commonvoice" else ""
    out_file = out_dir / f"results_{args.engine}_{args.model}{suffix}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    main()
