# Qwen3-ASR Batched Decode Benchmark

This benchmark measures Qwen3-ASR fixed-duration chunk throughput and can
optionally exercise the experimental greedy batched decoder path.

It is intentionally narrower than the WER benchmarks:

- fixed local WAV chunks, no dataset download
- one model load and warmup per CLI run
- batch-size sweep over the same chunks in the same order
- aggregate inference RTF and relative speedup against batch size 1

## Why This Shape

The asyncEval greedy decode PR was evaluated as a hot-path A/B benchmark:
same model, same audio, repeated hot iterations, and byte-identical output
between baseline and optimized branches. That is the right pattern for this
work too because the goal is decoder throughput and host/GPU scheduling, not
WER.

For batched decode, the first local benchmark should answer a smaller question:
when fixed-length ASR chunks are decoded with `--batch-size 1/2/4/8`, does
aggregate RTF improve and does the wall-clock speedup scale before CPU/GPU
synchronization becomes the bottleneck?

## Run

Build the release CLI and compiled MLX metallib first:

```bash
make build
```

Then run the benchmark sweep:

```bash
python3 scripts/benchmark_qwen3_batch_decode.py \
  --model 0.6B \
  --batch-sizes 1,2,4,8 \
  --chunk-seconds 10 \
  --stride-seconds 0 \
  --num-chunks 8
```

By default this uses the correctness-safe public path: per-row decoder forwards
with one batched token CPU copy per decode step. To reproduce the true `[B,1,H]`
decoder experiment, opt in explicitly:

```bash
python3 scripts/benchmark_qwen3_batch_decode.py \
  --experimental-batched-decode \
  --model 0.6B \
  --batch-sizes 1,2,4,8 \
  --chunk-seconds 10 \
  --stride-seconds 0 \
  --num-chunks 8
```

Or let the script build first:

```bash
python3 scripts/benchmark_qwen3_batch_decode.py --build
```

The default source audio is:

```text
Tests/Qwen3ASRTests/Resources/test_audio.wav
```

With the bundled test audio, `--stride-seconds 0` repeats the first 10s speech
chunk so each batch item performs similar decode work. For a long external
recording, set `--stride-seconds 10` to cut adjacent 10s chunks.

The script creates fixed-duration chunks under
`.build/qwen3-batch-decode-benchmark/`, runs:

```bash
.build/release/audio transcribe-batch <chunks> \
  --engine qwen3 \
  --model <model> \
  --batch-size <N> \
  --jsonl
```

With `--experimental-batched-decode`, the subprocess sets
`QWEN3_ASR_EXPERIMENTAL_BATCH_DECODE=1`.

and reports:

- total audio seconds
- total inference seconds
- aggregate RTF
- x-realtime throughput
- speedup relative to batch size 1
- output chars/s
- distinct transcript count and a short output digest
- CLI wall time, model load time, and warmup time when available

When all generated chunks are expected to be identical, add a correctness gate:

```bash
python3 scripts/benchmark_qwen3_batch_decode.py \
  --model 0.6B \
  --batch-sizes 1,2,4,8 \
  --chunk-seconds 10 \
  --stride-seconds 0 \
  --num-chunks 8 \
  --require-identical-output
```

This is the default local sanity check for batched decoder work: speedups are
not meaningful unless repeated identical chunks also produce identical text in
every row.

Current local 0.6B result on repeated 10s chunks:

| Chunks | Best batch size | Inference time | Aggregate RTF | Speedup |
|---:|---:|---:|---:|---:|
| 18 | 6 | 2.96s -> 2.66s | 0.0165 -> 0.0148 | 1.113x |
| 24 | 6 | 3.95s -> 3.55s | 0.0165 -> 0.0148 | 1.113x |

Batch size 8 was slower than 6 in both sweeps, so choose the cap from the
benchmark rather than assuming larger batches are always better.

## Current Limitations

The CLI currently returns text and per-file/batch wall-clock timing, but it
does not expose generated decoder token counts or split prefill/decode timing.
For that reason the benchmark reports `chars/s`, not true generated
`tokens/s`.

The experimental batched decoder path must also pass a row-level correctness
check before it becomes the default: repeated identical chunks should produce
identical output for every row at batch sizes 1/2/4/8. Any row-specific
truncation means the benchmark is finding a decoder/kernel correctness bug,
not a valid speedup.

Known current status: the true `[B,1,H]` experimental path fails this check at
batch size 2 on repeated test chunks, where row 0 completes the sentence and
row 1 truncates after "Can you guarantee". It remains behind
`QWEN3_ASR_EXPERIMENTAL_BATCH_DECODE=1`.

The next instrumentation step is to add an internal Qwen3-ASR metrics path
that returns:

- encoder time
- decoder prefill time
- decoder decode-loop time
- generated token count per item
- number of CPU synchronization points per decode step

Once those are exposed, this benchmark should report prefill/decode RTF and
generated tokens/s directly.

## Metal Trace

For a lightweight power/utilization run, use the local wrapper around
`powermetrics`:

```bash
sudo -v
python3 scripts/monitor_apple_gpu.py \
  --interval-ms 1000 \
  --raw-output .build/qwen3-batch-decode-gpu.plist \
  --output .build/qwen3-batch-decode-gpu.json \
  -- python3 scripts/benchmark_qwen3_batch_decode.py \
    --model 0.6B \
    --batch-sizes 1,2,4,8 \
    --chunk-seconds 10 \
    --stride-seconds 0 \
    --num-chunks 8 \
    --require-identical-output
```

The script calls `sudo -n`, so it does not open an interactive password prompt.
If the shell has no sudo credential, it exits and prints the exact
`powermetrics` command to run after `sudo -v`.

After the batch-size sweep shows a useful range, use Instruments Metal System
Trace on the same command. The questions to answer are:

- does GPU occupancy improve from batch size 1 to 2/4/8?
- does the process become CPU-bound on scalar token synchronization?
- does batch size 8 increase memory pressure enough to erase the decoder win?
