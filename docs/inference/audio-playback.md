# Streaming Audio Playback

## Architecture

`StreamingAudioPlayer` uses an event-driven architecture based on `AVAudioSourceNode`. Instead of scheduling buffers to a player node (push model), the audio hardware **pulls** data from a ring buffer via a render callback when it needs it.

```
TTS (producer thread)          Hardware (render thread)
        │                              │
        │  scheduleChunk([Float])      │
        ├─────────► Ring Buffer ◄──────┤  render callback
        │           (SPSC)             │  reads samples
        │                              │
        │  markGenerationComplete()    │  detects empty + done
        │                              │  → onPlaybackFinished
```

This is the standard approach in professional audio: the producer writes samples into a buffer, and the consumer (hardware) reads when ready. The buffer decouples the two — the producer doesn't need to match the hardware's timing.

## The Buffer Underflow Problem

TTS generates audio in chunks. Each chunk takes variable time to compute. If we feed chunks directly to the audio output, any gap between chunks causes silence and pops — **buffer underflow**.

```
Without pre-buffer:
  TTS:    [chunk1]---wait 870ms---[chunk2]---wait---[chunk3]
  Player: [play][silence/pop][play][silence/pop][play]
```

In audio, you cannot assume the processing pipeline will deliver data fast enough to keep the output fed. The output cannot wait — it runs at a fixed sample rate driven by hardware clocks.

## Pre-Buffer Solution

The solution is universal across all audio systems: introduce a buffer between the producer and consumer, and start playback only after the buffer has accumulated enough data.

```
With pre-buffer (2s):
  TTS:    [chunk1][chunk2][chunk3][chunk4]...
               ↓
  Ring Buffer: [accumulate 2s of audio]
               ↓ start playback when buffer is full
  Hardware:    [continuous audio pull, no gaps]
```

Once playback starts:
- The buffer drains at real-time rate (1s of audio per second)
- TTS fills it faster than real-time (RTF < 1.0 means it generates faster than playback)
- The buffer level only grows — underflow is impossible

If TTS temporarily stalls (CPU spike, GC pause), the buffer absorbs the jitter. This is why buffer size matters — it's the maximum stall duration you can tolerate without audible artifacts.

## Buffer Size: The Key Tradeoff

Buffer size is a latency vs quality tradeoff:

- **Larger buffer** = more resilient to jitter, but higher first-audio latency
- **Smaller buffer** = lower latency, but risk of underflow if generation hiccups

| `preBufferDuration` | First-audio latency | Risk | Use case |
|---------------------|--------------------:|------|----------|
| 0 | ~130ms | High (any gap = underflow) | Single-pass TTS (Kokoro) where all audio arrives at once |
| 0.5 | ~0.6s | Medium | Low-latency voice assistant, fast hardware |
| 1.0 | ~1.1s | Low | Streaming TTS (default, recommended for Qwen3-TTS) |
| 2.0 | ~2.1s | Very low | Slow hardware, high system load |
| 3.0 | ~3.1s | Minimal | High jitter, Bluetooth audio, slow hardware |

### Factors that affect optimal buffer size

1. **Generation speed (RTF)** — if RTF is 0.5 (2x real-time), a 2s buffer gives 2s of headroom before underflow. If RTF is 0.9, you need a larger buffer.

2. **Hardware output buffer** — the audio device has its own buffer (typically 256-1024 frames at 48kHz = 5-21ms). Your pre-buffer should be significantly larger than this.

3. **System load** — CPU/GPU contention can cause generation spikes. Under load, RTF can temporarily exceed 1.0. The pre-buffer absorbs these spikes.

4. **Audio route** — Bluetooth adds 40-200ms of output latency. The pre-buffer should account for this to avoid the output running dry during route switching.

In professional audio applications, buffer size is often exposed as a user setting — the user adjusts until they find the sweet spot between latency and stability for their specific hardware.

## How It Works: Event-Driven Model

The audio hardware drives playback via an event-driven callback (Apple's `AVAudioSourceNode`):

1. Audio hardware needs N frames of audio
2. It calls our render callback
3. We read N samples from the ring buffer
4. If the buffer is empty and generation isn't done → output silence (underflow)
5. If the buffer is empty and generation is done → fire `onPlaybackFinished`

This is Apple's implementation of the standard audio callback model. The render callback runs on a real-time priority thread managed by CoreAudio — it must return immediately and never block. The ring buffer enables this: reads are O(1) with no allocation or locking.

## Ring Buffer

`AudioSampleRingBuffer` is a single-producer single-consumer (SPSC) circular buffer:

- **Producer** (TTS thread): calls `write([Float])` — advances write pointer
- **Consumer** (render thread): calls `read(into:count:)` — advances read pointer
- **Capacity**: configurable, default 30s of audio at the engine's sample rate
- **Wrap-around**: both pointers wrap when they reach the end
- **Lock-free for SPSC**: safe for one writer + one reader without locks

## Usage

### Streaming TTS (Qwen3-TTS, CosyVoice)

```swift
let player = StreamingAudioPlayer()
player.preBufferDuration = 2.0  // 2s pre-buffer
try player.start(sampleRate: 24000)

// TTS generates chunks asynchronously
for try await chunk in ttsStream {
    try player.play(samples: chunk.samples, sampleRate: 24000)
}

// Signal end of stream — render callback will drain buffer then fire callback
player.markGenerationComplete()

player.onPlaybackFinished = {
    print("All audio played through speaker")
}
```

### Single-Pass TTS (Kokoro)

```swift
let player = StreamingAudioPlayer()
player.preBufferDuration = 0  // All audio arrives at once
try player.start(sampleRate: 24000)
try player.play(samples: allSamples, sampleRate: 24000)
player.markGenerationComplete()
```

### Voice Pipeline (shared engine with mic)

```swift
let engine = AVAudioEngine()
let player = StreamingAudioPlayer()
player.preBufferDuration = 2.0
player.attach(to: engine, format: playerFormat)

// engine also has mic input tap for VAD/ASR
try engine.start()
```

### Interrupting Playback

```swift
// User speaks over TTS — stop immediately
player.fadeOutAndStop()

// Start new generation
player.resetGeneration()
player.scheduleChunk(newAudioSamples)
player.markGenerationComplete()
```

## End-of-Stream Detection

When `markGenerationComplete()` is called, the render callback knows no more data is coming. Once it reads the last samples from the ring buffer and the buffer is empty, it fires `onPlaybackFinished` on the main thread. No sentinel buffers or grace periods needed — the render callback runs at hardware timing, so completion is detected at the exact moment the last sample is consumed.

For short utterances that don't fill the pre-buffer, `markGenerationComplete()` forces playback to start from whatever has accumulated.

## Generation Timing Reference (M2 Max)

| TTS Engine | RTF | Chunk size | Chunk interval | Recommended pre-buffer |
|------------|-----|-----------|----------------|----------------------:|
| Qwen3-TTS 0.6B (4-bit) | 0.53 | 2.0s | ~1.07s | 2.0s |
| Qwen3-TTS 1.7B (8-bit) | 0.79 | 2.0s | ~1.58s | 3.0s |
| CosyVoice3 (4-bit) | 0.59 | ~150ms | ~100ms | 1.0s |
| Kokoro-82M (CoreML) | N/A | all at once | ~45ms | 0s |

RTF = Real-Time Factor (time to generate / audio duration). RTF < 1.0 means generation is faster than playback.

## Source Files

```
Sources/AudioCommon/
  StreamingAudioPlayer.swift   Event-driven player with ring buffer + pre-buffer
```
