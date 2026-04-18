# KWS Zipformer (gigaspeech, 3.49M params)

On-device wake-word / keyword-spotting model based on icefall's streaming
Zipformer2 encoder + stateless RNN-T transducer. CoreML export with INT8
palettization fits in under 4 MB of compiled weights and hits ~26× real-time
on CPU + Neural Engine.

- **Source**: `pkufool/keyword-spotting-models@v0.11/icefall-kws-zipformer-gigaspeech-20240219`
- **License**: Apache-2.0
- **Language**: English only (gigaspeech fine-tune)
- **CoreML bundle**: `aufklarer/KWS-Zipformer-3M-CoreML-INT8` on HuggingFace

## Architecture

| Stage | Type | Purpose |
|---|---|---|
| Fbank | kaldi-native-fbank | 80 mel bins, 25 ms / 10 ms, Povey window, no CMVN, `high_freq = −400 Hz` |
| Encoder | 6-stage causal Zipformer2 | 128 → 128 hidden dims across stages; 1 layer per stage; chunk 16 × 2 (plus 13 padding) = 45 mel frames in → 8 output frames out (40 ms / frame) |
| Decoder | Stateless transducer | BPE-500 vocab; context size 2; 320 dim |
| Joiner | Linear + tanh | Folds `encoder_proj` and `decoder_proj` into the encoder/decoder wrappers, leaving only the output projection |

The encoder maintains 38 cache tensors per stream: 36 per-layer attention and
convolution caches plus `cached_embed_left_pad` (Conv2dSubsampling state) and
`processed_lens` (int32 counter, used to build the attention mask).

## Decoding

Modified beam search over the transducer (beam=4) driven through an
Aho-Corasick `ContextGraph` that holds user-supplied keyword phrases. For
each encoder frame the decoder:

1. Runs the decoder + joiner per beam hypothesis, softmaxes the logits.
2. Picks top-`beam` `(hypothesis, token)` pairs.
3. On non-blank / non-unk tokens advances the context-graph state and applies
   the phrase boost score. On blank/unk increments a tail-blank counter.
4. Checks the most-probable hypothesis: if it lands on an `is_end` node, has
   at least `num_tailing_blanks` trailing blanks, and the mean acoustic
   probability over the matched span clears the per-keyword
   `ac_threshold`, emits a detection and resets the beam.
5. Auto-resets the beam to root if no token has been emitted for
   `autoResetSeconds` of audio (default 1.5 s).

## Tuned thresholds

The export defaults (`ac_threshold = 0.15`, `context_score = 0.5`,
`num_tailing_blanks = 1`) were tuned on LibriSpeech test-clean:

| Setting | Recall | FP/utt |
|---|---|---|
| icefall defaults (`0.25`, `2.0`, `1`) | 62% | 0.43 |
| **tuned (shipped)** | **88%** | **0.27** |

Per-keyword overrides are expressed in the `keywords.txt` file shipped
with the export and via `KeywordSpec(phrase:, acThreshold:, boost:)` in Swift.

## Weight layout

The CoreML bundle ships three compiled models:

| Model | Compiled size | I/O |
|---|---|---|
| `encoder.mlmodelc` | 3.3 MB | mel + 38 caches → encoder_out + 38 new caches |
| `decoder.mlmodelc` | 525 KB | `(1, context_size)` int32 → `(1, joiner_dim)` fp16 |
| `joiner.mlmodelc` | 160 KB | `(enc, dec)` fp16 → `(1, vocab_size)` logits |

All three target `CPU_AND_NE` and `iOS17` minimum. The encoder and joiner are
INT8 palettized; the decoder stays FP16 (it's tiny).

## CPU / memory

- Encoder chunk latency: ~12 ms on M-series CPU+NE for a 320 ms audio chunk.
- Real-time factor: ~0.04 (26× real-time) on the export's LibriSpeech benchmark.
- Total memory footprint at runtime: ~6 MB (weights + encoder state).

## See also

- [Inference pipeline](../inference/wake-word.md) — fbank, streaming session,
  keyword file format.
- [`models/kws-zipformer/export/`](https://github.com/soniqo/speech-models/tree/main/models/kws-zipformer/export)
  in `soniqo/speech-models` for the full conversion + benchmark scripts.
