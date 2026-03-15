#!/usr/bin/env python3
"""
TTS round-trip WER benchmark for speech-swift.

Synthesizes text, transcribes the audio back, computes WER vs original.
Parses TTS stage breakdown: embed / generate / decode.

Usage:
    python scripts/benchmark_tts.py [--tts-engine qwen3] [--num-sentences 10]
    python scripts/benchmark_tts.py --compare
    python scripts/benchmark_tts.py --input-file sentences.txt
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

BENCHMARK_DIR = Path("benchmarks/tts")

# Built-in test corpus: diverse lengths and phonemes
TEST_SENTENCES = [
    # Short (2-5 words)
    "Hello world.",
    "Good morning everyone.",
    "Thank you very much.",
    "What time is it?",
    "Nice to meet you.",
    # Medium (8-15 words)
    "The quick brown fox jumps over the lazy dog.",
    "Can you guarantee that the replacement part will be shipped tomorrow?",
    "The weather is beautiful today, perfect for a walk in the park.",
    "Please make sure to send the report before the end of the day.",
    "I would like to schedule a meeting for next Wednesday afternoon.",
    # Long (20+ words)
    "Scientists have discovered a new species of deep sea fish that can survive "
    "at extreme pressures found at the bottom of the ocean.",
    "The development team has been working around the clock to deliver the new "
    "software update, which includes several critical bug fixes and performance "
    "improvements.",
    "After careful consideration of all the available evidence, the committee "
    "decided to postpone the decision until the next quarterly review.",
    # Numbers and special content
    "The temperature today is twenty three degrees celsius.",
    "Our company was founded in nineteen ninety nine.",
    "The flight departs at three forty five in the afternoon.",
    # Questions and commands
    "Could you please explain how this algorithm works?",
    "Turn left at the next intersection and continue for two miles.",
    "Have you ever visited the national museum of natural history?",
    "Remember to water the plants every other day during the summer.",
    # Technical
    "Machine learning models require large amounts of training data.",
    "The server response time should be under two hundred milliseconds.",
    "Cloud computing enables on demand access to shared resources.",
    "Natural language processing is a subfield of artificial intelligence.",
    # Conversational
    "I think we should take a different approach to this problem.",
    "That sounds like a great idea, let me think about it.",
    "Sorry, I did not catch what you said, could you repeat that?",
    "The restaurant around the corner serves excellent Italian food.",
    "We need to finish this project before the deadline next Friday.",
    "Let me know if you have any questions about the presentation.",
]


# ---------------------------------------------------------------------------
# Text normalization & WER
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_wer(reference: str, hypothesis: str) -> dict:
    ref = normalize_text(reference).split()
    hyp = normalize_text(hypothesis).split()
    n, m = len(ref), len(hyp)

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
            i -= 1; j -= 1
        elif i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + 1:
            subs += 1; i -= 1; j -= 1
        elif j > 0 and d[i][j] == d[i][j - 1] + 1:
            ins += 1; j -= 1
        else:
            dels += 1; i -= 1

    errors = subs + ins + dels
    return {
        "wer": round(errors / max(n, 1) * 100, 2),
        "errors": errors, "substitutions": subs,
        "insertions": ins, "deletions": dels,
        "ref_words": n, "hyp_words": m,
    }


# ---------------------------------------------------------------------------
# TTS + ASR pipeline with latency breakdown
# ---------------------------------------------------------------------------

def parse_tts_timing(stdout: str) -> dict:
    """Parse detailed TTS timing from model output.

    Qwen3-TTS prints:
      Timing: embed=0.033s | generate=3.540s (29 steps, 122ms/step) | decode=0.123s | total=3.702s | audio=2.20s | RTF=1.68
    CosyVoice prints:
      Duration: 2.68s, Time: 4.53s, RTF: 1.69
    """
    timing = {}

    for line in stdout.split("\n"):
        # Qwen3-TTS detailed breakdown
        if "embed=" in line and "generate=" in line and "decode=" in line:
            m = re.search(r"embed=([\d.]+)s", line)
            if m:
                timing["embed_s"] = float(m.group(1))
            m = re.search(r"generate=([\d.]+)s\s*\((\d+)\s*steps?,\s*([\d.]+)ms/step\)", line)
            if m:
                timing["generate_s"] = float(m.group(1))
                timing["steps"] = int(m.group(2))
                timing["ms_per_step"] = float(m.group(3))
            m = re.search(r"decode=([\d.]+)s", line)
            if m:
                timing["decode_s"] = float(m.group(1))
            m = re.search(r"total=([\d.]+)s", line)
            if m:
                timing["total_s"] = float(m.group(1))
            m = re.search(r"audio=([\d.]+)s", line)
            if m:
                timing["audio_s"] = float(m.group(1))
            m = re.search(r"RTF=([\d.]+)", line)
            if m:
                timing["rtf"] = float(m.group(1))

        # CosyVoice / generic format
        m = re.search(
            r"Duration:\s*([\d.]+)s.*Time:\s*([\d.]+)s.*RTF:\s*([\d.]+)",
            line)
        if m and "total_s" not in timing:
            timing["audio_s"] = float(m.group(1))
            timing["total_s"] = float(m.group(2))
            timing["rtf"] = float(m.group(3))

    return timing


def synthesize(cli_path: str, text: str, output_path: str,
               engine: str, model: str, timeout: int = 180) -> dict:
    """Synthesize text to audio file. Returns timing breakdown."""
    if engine == "kokoro":
        cmd = [cli_path, "kokoro", text, "--output", output_path]
    else:
        cmd = [cli_path, "speak", text, "--output", output_path,
               "--engine", engine]
        if engine == "qwen3" and model != "default":
            cmd.extend(["--model", model])

    wall_start = time.monotonic()
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout)
    wall_time = time.monotonic() - wall_start

    if result.returncode != 0:
        return {"error": result.stderr.strip()[:200]}

    timing = parse_tts_timing(result.stdout)
    timing["wall_time"] = round(wall_time, 3)
    return timing


def transcribe(cli_path: str, audio_path: str, asr_engine: str,
               asr_model: str, timeout: int = 120) -> dict:
    cmd = [cli_path, "transcribe", audio_path, "--engine", asr_engine]
    if asr_engine in ("qwen3", "qwen3-coreml", "qwen3-coreml-full"):
        cmd.extend(["--model", asr_model])

    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        return {"error": result.stderr.strip()[:200]}

    text = ""
    asr_rtf = 0.0
    for line in result.stdout.split("\n"):
        if line.startswith("Result: "):
            text = line[len("Result: "):]
        elif "RTF:" in line:
            m = re.search(r"RTF:\s*([\d.]+)", line)
            if m:
                asr_rtf = float(m.group(1))
    return {"text": text, "asr_rtf": asr_rtf}


def run_benchmark(cli_path: str, sentences: list,
                  tts_engine: str, tts_model: str,
                  asr_engine: str, asr_model: str,
                  timeout: int = 180) -> list:
    """Run TTS round-trip benchmark on all sentences."""
    results = []
    total = len(sentences)

    with tempfile.TemporaryDirectory(prefix="tts_bench_") as tmpdir:
        for idx, text in enumerate(sentences):
            pct = (idx + 1) / total * 100
            short = text[:50] + "..." if len(text) > 50 else text
            print(f"\r  [{idx+1}/{total}] ({pct:.0f}%) {short}",
                  end="", flush=True)

            wav_path = os.path.join(tmpdir, f"tts_{idx:04d}.wav")

            # Synthesize
            try:
                tts_out = synthesize(
                    cli_path, text, wav_path, tts_engine, tts_model, timeout)
            except subprocess.TimeoutExpired:
                tts_out = {"error": "tts_timeout"}
            except Exception as e:
                tts_out = {"error": str(e)}

            if "error" in tts_out:
                results.append({
                    "index": idx, "input_text": text,
                    "error": f"TTS: {tts_out['error']}",
                })
                continue

            if not os.path.exists(wav_path):
                results.append({
                    "index": idx, "input_text": text,
                    "error": "TTS produced no output file",
                })
                continue

            # Transcribe
            try:
                asr_out = transcribe(
                    cli_path, wav_path, asr_engine, asr_model, timeout)
            except subprocess.TimeoutExpired:
                asr_out = {"error": "asr_timeout"}
            except Exception as e:
                asr_out = {"error": str(e)}

            if "error" in asr_out:
                results.append({
                    "index": idx, "input_text": text,
                    "error": f"ASR: {asr_out['error']}",
                })
                continue

            wer_result = compute_wer(text, asr_out["text"])

            entry = {
                "index": idx,
                "input_text": normalize_text(text),
                "transcription": normalize_text(asr_out["text"]),
                "wer": wer_result["wer"],
                "substitutions": wer_result["substitutions"],
                "insertions": wer_result["insertions"],
                "deletions": wer_result["deletions"],
                "ref_words": wer_result["ref_words"],
                "asr_rtf": asr_out["asr_rtf"],
                "tts_timing": {k: v for k, v in tts_out.items()
                               if k != "error"},
            }
            results.append(entry)

    print()
    return results


def aggregate_results(per_sentence: list, tts_engine: str, tts_model: str,
                      asr_engine: str, asr_model: str) -> dict:
    scored = [r for r in per_sentence if "error" not in r]
    failed = [r for r in per_sentence if "error" in r]

    total_ref = sum(r["ref_words"] for r in scored)
    total_sub = sum(r.get("substitutions", 0) for r in scored)
    total_ins = sum(r.get("insertions", 0) for r in scored)
    total_del = sum(r.get("deletions", 0) for r in scored)
    total_errors = total_sub + total_ins + total_del
    agg_wer = total_errors / max(total_ref, 1) * 100

    # Latency aggregation
    timings = [r["tts_timing"] for r in scored if "tts_timing" in r]
    latency = {}
    if timings:
        for key in ["embed_s", "generate_s", "decode_s", "total_s",
                     "audio_s", "rtf", "ms_per_step"]:
            vals = [t[key] for t in timings if key in t]
            if vals:
                latency[f"mean_{key}"] = round(sum(vals) / len(vals), 4)

    result = {
        "tts_engine": tts_engine,
        "tts_model": tts_model,
        "asr_engine": asr_engine,
        "asr_model": asr_model,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "num_sentences": len(scored),
        "num_failures": len(failed),
        "aggregate_wer": round(agg_wer, 2),
        "total_ref_words": total_ref,
        "total_substitutions": total_sub,
        "total_insertions": total_ins,
        "total_deletions": total_del,
        "latency": latency,
        "per_sentence": per_sentence,
    }
    return result


def print_summary(results: dict):
    lat = results.get("latency", {})
    print(f"\n{'='*60}")
    print(f"TTS Round-Trip Benchmark")
    print(f"TTS: {results['tts_engine']}/{results['tts_model']}  "
          f"ASR: {results['asr_engine']}/{results['asr_model']}")
    print(f"{'='*60}")
    print(f"  Sentences:      {results['num_sentences']}"
          f" ({results['num_failures']} failed)")
    print(f"  Round-trip WER: {results['aggregate_wer']:.2f}%")
    print(f"  Total words:    {results['total_ref_words']}")
    if lat.get("mean_rtf"):
        print(f"  Mean TTS RTF:   {lat['mean_rtf']:.2f}")
    if lat.get("mean_embed_s"):
        print(f"  Mean embed:     {lat['mean_embed_s']*1000:.0f}ms")
    if lat.get("mean_generate_s"):
        print(f"  Mean generate:  {lat['mean_generate_s']*1000:.0f}ms")
    if lat.get("mean_ms_per_step"):
        print(f"  Mean ms/step:   {lat['mean_ms_per_step']:.0f}ms")
    if lat.get("mean_decode_s"):
        print(f"  Mean decode:    {lat['mean_decode_s']*1000:.0f}ms")
    print(f"{'='*60}")

    scored = [r for r in results["per_sentence"] if "error" not in r]
    errors = [r for r in scored if r["wer"] > 0]
    if errors:
        print(f"\nMismatched sentences ({len(errors)}):")
        for r in sorted(errors, key=lambda x: x["wer"], reverse=True):
            print(f"  [{r['index']}] WER={r['wer']:.1f}%")
            print(f"    in:  {r['input_text']}")
            print(f"    out: {r['transcription']}")


# ---------------------------------------------------------------------------
# Multi-engine comparison
# ---------------------------------------------------------------------------

TTS_CONFIGS = [
    ("qwen3", "base"),
    ("qwen3", "base-8bit"),
    ("cosyvoice", "default"),
    ("kokoro", "default"),
]


def run_comparison(cli_path: str, sentences: list,
                   asr_engine: str, asr_model: str, timeout: int = 180):
    all_results = []

    for tts_engine, tts_model in TTS_CONFIGS:
        print(f"\n--- TTS: {tts_engine}/{tts_model} ---")
        per_sentence = run_benchmark(
            cli_path, sentences, tts_engine, tts_model,
            asr_engine, asr_model, timeout)
        agg = aggregate_results(
            per_sentence, tts_engine, tts_model, asr_engine, asr_model)
        all_results.append(agg)
        print_summary(agg)

        out_file = BENCHMARK_DIR / f"results_{tts_engine}_{tts_model}.json"
        with open(out_file, "w") as f:
            json.dump(agg, f, indent=2)

    print(f"\n{'='*70}")
    print(f"{'TTS Engine':<16} {'Model':<10} {'WER%':>7} {'RTF':>6} "
          f"{'ms/step':>8} {'embed':>7} {'decode':>7}")
    print(f"{'-'*16} {'-'*10} {'-'*7} {'-'*6} {'-'*8} {'-'*7} {'-'*7}")
    for r in all_results:
        lat = r.get("latency", {})
        rtf = f"{lat['mean_rtf']:.2f}" if lat.get("mean_rtf") else "N/A"
        ms = f"{lat['mean_ms_per_step']:.0f}" if lat.get("mean_ms_per_step") else "N/A"
        emb = f"{lat['mean_embed_s']*1000:.0f}ms" if lat.get("mean_embed_s") else "N/A"
        dec = f"{lat['mean_decode_s']*1000:.0f}ms" if lat.get("mean_decode_s") else "N/A"
        print(f"{r['tts_engine']:<16} {r['tts_model']:<10} "
              f"{r['aggregate_wer']:>6.2f}% {rtf:>6} {ms:>8} {emb:>7} {dec:>7}")
    print(f"{'='*70}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="TTS round-trip WER benchmark")
    parser.add_argument("--cli-path", default=".build/release/audio")
    parser.add_argument("--tts-engine", default="qwen3",
                        help="TTS engine: qwen3, cosyvoice, kokoro")
    parser.add_argument("--tts-model", default="base")
    parser.add_argument("--asr-engine", default="qwen3")
    parser.add_argument("--asr-model", default="0.6B")
    parser.add_argument("--num-sentences", type=int, default=0,
                        help="Limit sentences (0 = all)")
    parser.add_argument("--input-file", type=str, default=None)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--compare", action="store_true",
                        help="Run all TTS engines")
    args = parser.parse_args()

    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)

    if not Path(args.cli_path).exists():
        print(f"CLI not found: {args.cli_path}. Build with: make build")
        sys.exit(1)

    sentences = (
        [l.strip() for l in Path(args.input_file).read_text().split("\n")
         if l.strip()]
        if args.input_file else TEST_SENTENCES
    )
    if args.num_sentences > 0:
        sentences = sentences[:args.num_sentences]
    print(f"Loaded {len(sentences)} test sentences")

    if args.compare:
        run_comparison(args.cli_path, sentences,
                       args.asr_engine, args.asr_model, args.timeout)
        return

    print(f"\nTTS: {args.tts_engine}/{args.tts_model}, "
          f"ASR: {args.asr_engine}/{args.asr_model}")
    per_sentence = run_benchmark(
        args.cli_path, sentences,
        args.tts_engine, args.tts_model,
        args.asr_engine, args.asr_model, args.timeout)

    results = aggregate_results(
        per_sentence, args.tts_engine, args.tts_model,
        args.asr_engine, args.asr_model)
    print_summary(results)

    out_file = BENCHMARK_DIR / f"results_{args.tts_engine}_{args.tts_model}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    main()
