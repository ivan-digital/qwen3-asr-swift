"""Regenerate the fbank parity fixtures for SpeechWakeWordTests.

Run from speech-swift with the kws-zipformer poetry env on PATH:

    cd ~/speech-models/models/kws-zipformer/export
    poetry run python ~/speech-swift/scripts/kws/generate_fbank_reference.py

Produces two files in ``Tests/SpeechWakeWordTests/Resources/``:

- ``fbank_input.wav``     — deterministic 1 s PCM @ 16 kHz (mix of sine + noise)
- ``fbank_reference.bin`` — 100 frames × 80 mel bins, ``<f`` float32 bytes,
                             computed via ``streaming_fbank.waveform_to_fbank``.
"""
from __future__ import annotations

import struct
import sys
import wave
from pathlib import Path

import numpy as np

# Import from the sibling export module.
sys.path.insert(0, str(Path.home() / "speech-models/models/kws-zipformer/export"))
from streaming_fbank import waveform_to_fbank, SAMPLE_RATE

# Write next to the SpeechWakeWordTests resources in this repo.
OUT = Path(__file__).resolve().parents[2] / "Tests" / "SpeechWakeWordTests" / "Resources"
OUT.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(seed=20260418)
num_samples = SAMPLE_RATE  # exactly 1 s
t = np.arange(num_samples) / SAMPLE_RATE
signal = (
    0.4 * np.sin(2 * np.pi * 440.0 * t)          # pure tone
    + 0.2 * np.sin(2 * np.pi * 1200.0 * t)       # upper partial
    + 0.05 * rng.standard_normal(num_samples)    # white noise
).astype(np.float32)

# Write PCM16 WAV (what AudioFileLoader expects).
pcm16 = np.clip(signal * 32767.0, -32768, 32767).astype("<i2")
wav_path = OUT / "fbank_input.wav"
with wave.open(str(wav_path), "wb") as w:
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(SAMPLE_RATE)
    w.writeframes(pcm16.tobytes())

# Reconvert the quantised PCM back to float32 to match what AudioFileLoader
# hands to KaldiFbank in Swift (read-through the same int16 precision).
float_from_pcm = pcm16.astype(np.float32) / 32768.0
fbank = waveform_to_fbank(float_from_pcm).astype("<f4")  # (num_frames, 80)

ref_path = OUT / "fbank_reference.bin"
with ref_path.open("wb") as f:
    f.write(struct.pack("<ii", fbank.shape[0], fbank.shape[1]))
    f.write(fbank.tobytes(order="C"))

print(f"Wrote {wav_path.name}  ({pcm16.nbytes} bytes)")
print(f"Wrote {ref_path.name} ({fbank.shape[0]}x{fbank.shape[1]})")
